#include <bit>
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
                    // bit-identical, and match evaluate()'s H1 survival.
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
