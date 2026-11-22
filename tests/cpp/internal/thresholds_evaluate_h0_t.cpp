#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detection/thresholds_engine.hpp"

using loki::Backend;
using loki::SizeType;
using loki::detection::State;
using loki::detection::detail::ThresholdsEngine;

namespace {

uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }

std::unique_ptr<ThresholdsEngine> make_engine(Backend backend,
                                              const std::vector<float>& bp,
                                              SizeType nbins = 32,
                                              int nthreads   = 2) {
    if (backend == Backend::kCPU) {
        return loki::detection::detail::make_thresholds_cpu(
            bp, 0.1F, nbins, 64, 6, 0.05F, 8.0F, 24, 0.3F, 1.5F, 1.2F, 1,
            "legacy", nthreads, /*seed=*/17, /*simulate=*/false);
    }
#ifdef LOKI_ENABLE_GPU
    return loki::detection::detail::make_thresholds_gpu(
        bp, 0.1F, nbins, 64, 6, 0.05F, 8.0F, 24, 0.3F, 1.5F, 1.2F, 1, "legacy",
        256, 0, 17);
#else
    return nullptr;
#endif
}

} // namespace

// The noise branch uses its own streams, so evaluating it alone reproduces
// the noise states of the full evaluation bit for bit; the signal side is 1.
TEST_CASE("evaluate(h0_only) reproduces the noise states", "[thresholds]") {
    const std::vector<float> bp(10, 2.0F);
    std::vector<float> thresholds(bp.size());
    for (SizeType s = 0; s < bp.size(); ++s) {
        thresholds[s] = 1.0F + (0.2F * static_cast<float>(s));
    }
    for (const auto backend : {Backend::kCPU, Backend::kCUDA}) {
        if (!loki::is_available(backend)) {
            continue;
        }
        DYNAMIC_SECTION("backend " << loki::to_string(backend)) {
            const auto engine = make_engine(backend, bp);
            for (const SizeType ntrials : {SizeType{200}, SizeType{1024}}) {
                const auto full =
                    engine->evaluate(thresholds, ntrials, 5, false);
                const auto h0 = engine->evaluate(thresholds, ntrials, 5, true);
                REQUIRE(h0.size() == bp.size());
                REQUIRE(full.size() == bp.size());
                for (SizeType s = 0; s < bp.size(); ++s) {
                    REQUIRE_FALSE(h0[s].is_empty);
                    REQUIRE(bits(h0[s].success_h0) == bits(full[s].success_h0));
                    REQUIRE(bits(h0[s].complexity) == bits(full[s].complexity));
                    REQUIRE(bits(h0[s].complexity_cumul) ==
                            bits(full[s].complexity_cumul));
                    REQUIRE(h0[s].success_h1 == 1.0F);
                    REQUIRE(h0[s].success_h1_cumul == 1.0F);
                }
            }
        }
    }
}

TEST_CASE("pruned H1 margins and survivor counts are bit-exact across nbins",
          "[thresholds]") {
    const std::vector<float> bp(18, 2.0F);
    std::vector<float> thresholds(bp.size());
    for (SizeType s = 0; s < bp.size(); ++s) {
        thresholds[s] = 1.5F + (0.22F * static_cast<float>(s));
    }
    for (const auto backend : {Backend::kCPU, Backend::kCUDA}) {
        if (!loki::is_available(backend)) {
            continue;
        }
        for (const SizeType nbins :
             {SizeType{32}, SizeType{256}, SizeType{1024}}) {
            DYNAMIC_SECTION("backend " << loki::to_string(backend) << " nbins "
                                       << nbins) {
                const auto engine     = make_engine(backend, bp, nbins, 4);
                const SizeType n      = 512;
                constexpr uint64_t sd = 0xBEEF1234ULL;
                const auto scores     = engine->simulate_h1_scores(n, sd);
                const loki::detection::detail::HostSignalBank ref(scores, n);
                const auto full_margins = ref.margins(thresholds);

                constexpr float kCut = -0.5F;
                const auto pruned =
                    engine->h1_margins_above(thresholds, kCut, n, sd);
                REQUIRE(pruned.size() == n);
                SizeType ref_surv = 0;
                for (SizeType t = 0; t < n; ++t) {
                    if (full_margins[t] > kCut) {
                        REQUIRE(bits(pruned[t]) == bits(full_margins[t]));
                    } else {
                        REQUIRE(pruned[t] <= kCut);
                    }
                    if (full_margins[t] > 0.0F) {
                        ++ref_surv;
                    }
                }
                REQUIRE(engine->count_h1_survivors(thresholds, n, sd) ==
                        ref_surv);

                if (backend == Backend::kCPU) {
                    // Single-thread vs multi-thread CPU H1 scores are
                    // bit-identical.
                    const auto single = make_engine(backend, bp, nbins, 1);
                    const auto sc1    = single->simulate_h1_scores(n, sd);
                    REQUIRE(sc1.size() == scores.size());
                    for (SizeType i = 0; i < scores.size(); ++i) {
                        REQUIRE(bits(sc1[i]) == bits(scores[i]));
                    }
                }
            }
        }
    }
}

// Production depth: 127 stages. On CUDA the pruned kernels run only above
// 128 bins, and the H1 fold buffers hold 64 MB per chunk, so 2^17 trials at
// 256 bins and 2^15 at 1024 bins each take four chunks (trial_offset > 0).
TEST_CASE("pruned H1 is bit-exact at production depth (127 stages)",
          "[thresholds]") {
    const std::vector<float> bp(127, 1.1F);
    std::vector<float> thresholds(bp.size());
    for (SizeType s = 0; s < bp.size(); ++s) {
        thresholds[s] = 1.0F + (0.045F * static_cast<float>(s));
    }
    constexpr uint64_t kSd = 0xCAFE5678ULL;
    constexpr float kCut   = -0.5F;
    struct Case {
        Backend backend;
        SizeType nbins;
        SizeType ntrials;
    };
    const std::vector<Case> cases{
        {Backend::kCPU, 64, SizeType{1} << 15U},
        {Backend::kCUDA, 256, SizeType{1} << 17U},
        {Backend::kCUDA, 1024, SizeType{1} << 15U},
    };
    for (const auto& c : cases) {
        if (!loki::is_available(c.backend)) {
            continue;
        }
        DYNAMIC_SECTION("backend " << loki::to_string(c.backend) << " nbins "
                                   << c.nbins << " ntrials " << c.ntrials) {
            const auto engine = make_engine(c.backend, bp, c.nbins, 4);
            const auto scores = engine->simulate_h1_scores(c.ntrials, kSd);
            const loki::detection::detail::HostSignalBank ref(scores,
                                                              c.ntrials);
            const auto full_margins = ref.margins(thresholds);
            const auto pruned =
                engine->h1_margins_above(thresholds, kCut, c.ntrials, kSd);
            REQUIRE(pruned.size() == c.ntrials);
            SizeType ref_surv = 0;
            SizeType n_exact  = 0;
            SizeType n_bad    = 0;
            for (SizeType t = 0; t < c.ntrials; ++t) {
                if (full_margins[t] > kCut) {
                    ++n_exact;
                    n_bad += bits(pruned[t]) == bits(full_margins[t]) ? 0 : 1;
                } else {
                    n_bad += pruned[t] <= kCut ? 0 : 1;
                }
                if (full_margins[t] > 0.0F) {
                    ++ref_surv;
                }
            }
            REQUIRE(n_bad == 0);
            // The schedule keeps a real share of trials to the last stage.
            REQUIRE(ref_surv > c.ntrials / 20);
            REQUIRE(n_exact > ref_surv);
            REQUIRE(engine->count_h1_survivors(thresholds, c.ntrials, kSd) ==
                    ref_surv);
        }
    }
}

// simulate_h1_scores follows every H1 trial on its own; evaluate() refills
// the H1 branch from survivors after a cut, so the two agree only while no
// trial has been cut. With every threshold before stage k far below the
// noise, evaluate()'s H1 survival at stage k is the share of the stage-k
// H1 scores above its threshold, bit for bit.
TEST_CASE("simulate_h1_scores follows evaluate()'s H1 trials", "[thresholds]") {
    const std::vector<float> bp(20, 2.0F);
    constexpr SizeType kN  = 1024;
    constexpr uint64_t kSd = 0x5EEDF00DULL;
    constexpr float kOpen  = -1.0e30F;
    for (const auto backend : {Backend::kCPU, Backend::kCUDA}) {
        if (!loki::is_available(backend)) {
            continue;
        }
        for (const SizeType nbins : {SizeType{32}, SizeType{256}}) {
            DYNAMIC_SECTION("backend " << loki::to_string(backend) << " nbins "
                                       << nbins) {
                const auto engine = make_engine(backend, bp, nbins, 4);
                const auto scores = engine->simulate_h1_scores(kN, kSd);
                REQUIRE(scores.size() == bp.size() * kN);
                for (const SizeType k :
                     {SizeType{0}, bp.size() / 2, bp.size() - 1}) {
                    // Cut at the median H1 score of stage k.
                    std::vector<float> stage_k(
                        scores.begin() + static_cast<std::ptrdiff_t>(k * kN),
                        scores.begin() +
                            static_cast<std::ptrdiff_t>((k + 1) * kN));
                    std::ranges::nth_element(stage_k,
                                             stage_k.begin() + (kN / 2));
                    const float cut = stage_k[kN / 2];
                    std::vector<float> thresholds(bp.size(), kOpen);
                    thresholds[k] = cut;
                    const auto states =
                        engine->evaluate(thresholds, kN, kSd, false);
                    SizeType above = 0;
                    for (SizeType t = 0; t < kN; ++t) {
                        above += scores[(k * kN) + t] > cut ? 1 : 0;
                    }
                    INFO("stage " << k << ": " << above << " of " << kN);
                    REQUIRE(above > 0);
                    REQUIRE(above < kN);
                    for (SizeType s = 0; s < k; ++s) {
                        REQUIRE(states[s].success_h1 == 1.0F);
                    }
                    REQUIRE(bits(states[k].success_h1) ==
                            bits(static_cast<float>(above) /
                                 static_cast<float>(kN)));
                }
            }
        }
    }
}
