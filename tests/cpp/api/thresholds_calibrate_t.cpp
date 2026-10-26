#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/detection/thresholds.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace loki {
namespace {

using detection::CalibrationOptions;
using detection::DynamicThresholdScheme;
using detection::State;

// The empty-H0 floor and the calibration on both backends. A CUDA case is
// skipped when the build has no CUDA backend.
Exec exec_of(std::string_view backend, int nthreads = 4) {
    return backend == "cuda" ? Exec::cuda() : Exec::cpu(nthreads);
}

bool skip_backend(std::string_view backend) {
    return backend == "cuda" && !is_available(Backend::kCUDA);
}

// Small search: 8 stages, so a run takes milliseconds.
DynamicThresholdScheme
make_small(std::string_view backend, uint64_t seed, int nthreads = 4) {
    const std::vector<float> branching(8, 2.0F);
    return {branching,
            /*ref_ducy=*/0.1F,
            /*nbins=*/32,
            /*ntrials=*/512,
            /*nprobs=*/10,
            /*prob_min=*/0.05F,
            /*snr_final=*/8.0F,
            /*nthresholds=*/40,
            /*ducy_max=*/0.3F,
            /*wtsp=*/1.0F,
            /*beam_width=*/1.5F,
            /*trials_start=*/1,
            "improved",
            seed,
            /*batch_size=*/256,
            exec_of(backend, nthreads)};
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](float x, float y) {
               return std::bit_cast<uint32_t>(x) == std::bit_cast<uint32_t>(y);
           });
}

} // namespace

TEST_CASE("snr_boxcar_max_with_cache matches the boxcar S/N", "[thresholds]") {
    constexpr SizeType kNbins     = 32;
    constexpr SizeType kNprofiles = 64;
    const auto widths             = detection::generate_box_width_trials(
        kNbins, /*ducy_max=*/0.3, /*wtsp=*/1.0);
    detection::BoxcarWidthsCache cache(widths, kNbins);
    // Same arithmetic as snr_boxcar_1d, up to rounding in the prefix sums
    const float tol =
        static_cast<float>(kNbins) * std::numeric_limits<float>::epsilon();
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    std::vector<float> profile(kNbins);
    std::vector<float> snr(widths.size());
    for (SizeType i = 0; i < kNprofiles; ++i) {
        std::ranges::generate(profile, [&] { return normal(rng); });
        detection::snr_boxcar_1d(profile, widths, snr);
        const float snr_max =
            detection::snr_boxcar_max_with_cache(profile, kNbins, cache);
        REQUIRE_THAT(snr_max, WithinRel(std::ranges::max(snr), tol));
        // The threshold test passes just below the maximum, not at it
        REQUIRE_FALSE(detection::snr_boxcar_threshold_with_cache(
            profile, kNbins, cache, snr_max));
        REQUIRE(detection::snr_boxcar_threshold_with_cache(
            profile, kNbins, cache,
            std::nextafter(snr_max, std::numeric_limits<float>::lowest())));
    }
}

TEST_CASE("DynamicThresholdScheme keeps a noise trial when none passes",
          "[thresholds]") {
    // Stage 0's beam is far above the noise and well below the signal
    const std::vector<float> branching_pattern(7, 3.0F);
    constexpr SizeType kNtrials = 1024;
    constexpr float kSnrFinal   = 16.0F;
    const auto* backend         = GENERATE("cpu", "cuda");
    const auto* mode            = GENERATE("legacy", "improved");
    CAPTURE(backend, mode);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    DynamicThresholdScheme dyn_scheme(
        branching_pattern, /*ref_ducy=*/0.1F, /*nbins=*/32, kNtrials,
        /*nprobs=*/10, /*prob_min=*/0.05F, kSnrFinal, /*nthresholds=*/100,
        /*ducy_max=*/0.3F, /*wtsp=*/1.0F, /*beam_width=*/0.7F,
        /*trials_start=*/1, mode, /*seed=*/23, /*batch_size=*/256,
        exec_of(backend));
    dyn_scheme.run();
    REQUIRE(dyn_scheme.get_best_path_thresholds().size() ==
            branching_pattern.size());
    // Noise survival is floored at one trial, and stage 0 needs the floor
    float success_h0_min = 1.0F;
    for (const auto& state : dyn_scheme.get_states()) {
        if (!state.is_empty) {
            success_h0_min = std::min(success_h0_min, state.success_h0);
        }
    }
    REQUIRE(success_h0_min == 1.0F / static_cast<float>(kNtrials));
}

TEST_CASE("evaluate keeps a noise trial when none passes", "[thresholds]") {
    // Stage 1 prunes every noise trial and keeps every signal trial: its
    // threshold is half the signal's S/N there, from 3 of nstages + 1 segments
    const std::vector<float> branching_pattern(7, 3.0F);
    const auto nstages          = branching_pattern.size();
    constexpr SizeType kNtrials = 1024;
    constexpr float kSnrFinal   = 40.0F;
    const float snr_stage1 =
        kSnrFinal * std::sqrt(3.0F / static_cast<float>(nstages + 1));
    std::vector<float> thresholds(nstages,
                                  std::numeric_limits<float>::lowest());
    thresholds[1] = snr_stage1 / 2.0F;

    const auto check = [&](const std::vector<State>& states) {
        REQUIRE(states[1].success_h0 == 1.0F / static_cast<float>(kNtrials));
        REQUIRE(states[1].success_h1 == 1.0F);
        // Later stages keep every trial, so the floored noise branches on
        for (SizeType i = 2; i < nstages; ++i) {
            CAPTURE(i);
            REQUIRE_FALSE(states[i].is_empty);
            REQUIRE(states[i].complexity ==
                    states[i - 1].complexity * branching_pattern[i]);
            REQUIRE(states[i].success_h1_cumul == 1.0F);
        }
    };
    SECTION("evaluate_scheme") {
        check(detection::evaluate_scheme(thresholds, branching_pattern,
                                         /*ref_ducy=*/0.1F, /*nbins=*/32,
                                         kNtrials, kSnrFinal));
    }
    SECTION("DynamicThresholdScheme::evaluate") {
        const auto* backend = GENERATE("cpu", "cuda");
        CAPTURE(backend);
        if (skip_backend(backend)) {
            SKIP("needs a CUDA build");
        }
        const DynamicThresholdScheme dyn_scheme(
            branching_pattern, /*ref_ducy=*/0.1F, /*nbins=*/32, kNtrials,
            /*nprobs=*/10, /*prob_min=*/0.05F, kSnrFinal, /*nthresholds=*/100,
            /*ducy_max=*/0.3F, /*wtsp=*/1.0F, /*beam_width=*/0.7F,
            /*trials_start=*/1, "improved", /*seed=*/17, /*batch_size=*/256,
            exec_of(backend));
        check(dyn_scheme.evaluate(thresholds, kNtrials, 3));
    }
}

TEST_CASE("simulate_h1_scores is evaluate's unpruned signal", "[thresholds]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    const auto dyn            = make_small(backend, 5);
    constexpr SizeType kTrial = 2048;
    const auto nstages        = dyn.get_nstages();
    const auto scores         = dyn.simulate_h1_scores(kTrial, 9);
    REQUIRE(scores.size() == nstages * kTrial);
    REQUIRE(same_bits(scores, dyn.simulate_h1_scores(kTrial, 9)));
    // The signal accumulates: the mean score rises stage by stage
    float mean_prev = std::numeric_limits<float>::lowest();
    for (SizeType s = 0; s < nstages; ++s) {
        double sum = 0.0;
        for (SizeType i = 0; i < kTrial; ++i) {
            sum += scores[(s * kTrial) + i];
        }
        const auto mean = static_cast<float>(sum / kTrial);
        REQUIRE(mean > mean_prev);
        mean_prev = mean;
    }
    // With nothing pruned, evaluate() keeps every signal trial
    const std::vector<float> open(nstages,
                                  std::numeric_limits<float>::lowest());
    const auto states = dyn.evaluate(open, kTrial, 9);
    REQUIRE(states.back().success_h1_cumul == 1.0F);
}

TEST_CASE("simulate_h1_scores: a partial last chunk leaves the others intact",
          "[thresholds]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    // At nbins 1024 CUDA simulates 4096 trials per chunk, so 5000 trials end
    // with a partial chunk of 904. Each trial keeps the random streams of
    // its index: the first 4096 trials of both calls must be identical at
    // every stage (the partial chunk once wrote into the next stage's row).
    const std::vector<float> branching(8, 2.0F);
    const DynamicThresholdScheme dyn(
        branching, /*ref_ducy=*/0.1F, /*nbins=*/1024, /*ntrials=*/64,
        /*nprobs=*/4, /*prob_min=*/0.05F, /*snr_final=*/8.0F,
        /*nthresholds=*/8, /*ducy_max=*/0.3F, /*wtsp=*/1.0F,
        /*beam_width=*/1.5F, /*trials_start=*/1, "improved", 5,
        /*batch_size=*/256, exec_of(backend));
    constexpr SizeType kFull    = 4096;
    constexpr SizeType kPartial = 5000;
    const auto nstages          = dyn.get_nstages();
    const auto full             = dyn.simulate_h1_scores(kFull, 9);
    const auto partial          = dyn.simulate_h1_scores(kPartial, 9);
    REQUIRE(full.size() == nstages * kFull);
    REQUIRE(partial.size() == nstages * kPartial);
    for (SizeType s = 0; s < nstages; ++s) {
        CAPTURE(s);
        const std::vector<float> head(
            partial.begin() + static_cast<std::ptrdiff_t>(s * kPartial),
            partial.begin() +
                static_cast<std::ptrdiff_t>((s * kPartial) + kFull));
        const std::vector<float> ref(
            full.begin() + static_cast<std::ptrdiff_t>(s * kFull),
            full.begin() + static_cast<std::ptrdiff_t>((s + 1) * kFull));
        REQUIRE(same_bits(head, ref));
    }
}

TEST_CASE("calibrate delivers min_pd on independent trials", "[thresholds]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    constexpr float kMinPd       = 0.3F;
    constexpr SizeType kNtrialsC = 16384;
    constexpr SizeType kNcheck   = 65536;
    auto dyn                     = make_small(backend, 7);
    dyn.run(6);
    const auto states_before = dyn.get_states();
    const auto path_before   = dyn.get_best_path_thresholds(kMinPd);

    CalibrationOptions opts;
    opts.ntrials_cal = kNtrialsC;
    opts.seed        = 101;
    const auto cal   = dyn.calibrate(kMinPd, opts);
    REQUIRE(cal.thresholds.size() == dyn.get_nstages());
    REQUIRE(cal.ncandidates >= 1);
    // At least ceil(min_pd * N) trials of the calibration sample survive,
    // exactly that many when the path had to move down
    const auto target = std::ceil(kMinPd * static_cast<float>(kNtrialsC));
    const auto kept   = cal.pd_certified * static_cast<float>(kNtrialsC);
    REQUIRE(kept >= target);
    if (cal.delta < 0.0F) {
        REQUIRE(kept == target);
    }
    REQUIRE(std::abs(cal.delta) <= opts.delta_max);
    REQUIRE(cal.pd_sigma > 0.0F);
    // An independent rescore agrees within its and the sample's error
    const float sigma = std::sqrt(kMinPd * (1.0F - kMinPd) *
                                  (1.0F / static_cast<float>(kNtrialsC) +
                                   1.0F / static_cast<float>(kNcheck)));
    const auto check  = dyn.evaluate(cal.thresholds, kNcheck, 202);
    REQUIRE_FALSE(check.back().is_empty);
    REQUIRE_THAT(check.back().success_h1_cumul,
                 WithinAbs(cal.pd_certified, 5.0F * sigma));
    REQUIRE(check.back().success_h1_cumul >= kMinPd - (5.0F * sigma));
    // Deterministic for a fixed seed, and the search is untouched
    const auto again = dyn.calibrate(kMinPd, opts);
    REQUIRE(same_bits(cal.thresholds, again.thresholds));
    REQUIRE(std::bit_cast<uint32_t>(cal.cost) ==
            std::bit_cast<uint32_t>(again.cost));
    REQUIRE(dyn.get_best_path_thresholds(kMinPd) == path_before);
    const auto states_after = dyn.get_states();
    REQUIRE(states_after.size() == states_before.size());
    for (SizeType i = 0; i < states_after.size(); ++i) {
        REQUIRE(std::bit_cast<uint32_t>(states_after[i].cost) ==
                std::bit_cast<uint32_t>(states_before[i].cost));
    }
}

TEST_CASE("calibrate on the CPU does not depend on thread count",
          "[thresholds]") {
    auto one   = make_small("cpu", 7, 1);
    auto eight = make_small("cpu", 7, 8);
    one.run(6);
    eight.run(6);
    CalibrationOptions opts;
    opts.ntrials_cal = 4096;
    opts.seed        = 3;
    const auto a     = one.calibrate(0.2F, opts);
    const auto b     = eight.calibrate(0.2F, opts);
    REQUIRE_FALSE(a.thresholds.empty());
    REQUIRE(same_bits(a.thresholds, b.thresholds));
    REQUIRE(std::bit_cast<uint32_t>(a.cost) == std::bit_cast<uint32_t>(b.cost));
}

TEST_CASE("DynamicThresholdScheme pilot beam guess", "[thresholds]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    const std::vector<float> branching(8, 2.0F);
    const auto make = [&](std::string_view guess) {
        return DynamicThresholdScheme(branching, 0.1F, 32, 512, 10, 0.05F, 8.0F,
                                      40, 0.3F, 1.0F, 1.5F, 1, "improved",
                                      /*seed=*/7, /*batch_size=*/256, guess,
                                      exec_of(backend));
    };
    auto pilot = make("pilot");
    pilot.run(6);
    REQUIRE(pilot.get_best_path_thresholds(0.05F).size() == branching.size());
    // The default constructor is the analytic guess
    auto analytic = make("analytic");
    auto dflt     = make_small(backend, 7);
    analytic.run(6);
    dflt.run(6);
    REQUIRE(same_bits(analytic.get_best_path_thresholds(0.05F),
                      dflt.get_best_path_thresholds(0.05F)));
    REQUIRE_THROWS_AS(make("median"), std::invalid_argument);
}

TEST_CASE("DynamicThresholdScheme window beam", "[thresholds]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    const std::vector<float> branching(8, 2.0F);
    constexpr float kProbMin = 0.05F;
    // A beam_width of 0.1 would starve a centred beam; the window ignores it
    auto window = DynamicThresholdScheme(
        branching, 0.1F, 32, 512, 10, kProbMin, 8.0F, 40, 0.3F, 1.0F, 0.1F, 1,
        "improved", /*seed=*/7, /*batch_size=*/256, "window", exec_of(backend));
    const auto thresholds = window.get_thresholds();
    const auto h1         = window.simulate_h1_scores(4096, /*seed=*/11);
    for (SizeType s = 0; s < branching.size(); ++s) {
        const auto beam = window.get_current_thresholds_idx(s);
        REQUIRE(beam.size() > 2);
        // Every beam threshold keeps at least ~prob_min / 2 of the signal
        const float top = thresholds[beam.back()];
        const auto kept =
            std::ranges::count_if(std::span(h1).subspan(s * 4096, 4096),
                                  [top](float x) { return x > top; });
        CHECK(static_cast<float>(kept) / 4096.0F > 0.3F * kProbMin);
    }
    window.run(1);
    const auto path = window.get_best_path_thresholds(0.1F);
    REQUIRE(path.size() == branching.size());
    const auto ev = window.evaluate(path, 1U << 14U, /*seed=*/99);
    CHECK(ev.back().success_h1_cumul > 0.08F);
}

TEST_CASE("Shared mode: reproducible, and its in-run Pd is honest",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const std::vector<float> branching(8, 2.0F);
    constexpr SizeType kNtrials = 512;
    const auto make             = [&](uint64_t seed) {
        return DynamicThresholdScheme(branching, 0.1F, 32, kNtrials, 10, 0.05F,
                                      8.0F, 40, 0.3F, 1.0F, 1.5F, 1, "shared",
                                      seed, /*batch_size=*/256, Exec::cuda());
    };
    auto a = make(7);
    a.run(6);
    auto b = make(7);
    b.run(6);
    const auto sa = a.get_states();
    const auto sb = b.get_states();
    REQUIRE(sa.size() == sb.size());
    bool identical = true;
    for (SizeType i = 0; i < sa.size(); ++i) {
        identical = identical &&
                    std::bit_cast<uint32_t>(sa[i].cost) ==
                        std::bit_cast<uint32_t>(sb[i].cost) &&
                    sa[i].is_empty == sb[i].is_empty;
    }
    REQUIRE(identical);
    // The selected path's in-run Pd is counted on the shared trials (16 x
    // ntrials of them); an independent rescore agrees within the errors.
    constexpr float kMinPd = 0.3F;
    const auto path        = a.get_best_path_thresholds(kMinPd);
    REQUIRE(path.size() == branching.size());
    float pd_inrun   = 0.0F;
    float best       = std::numeric_limits<float>::max();
    const auto nlast = a.get_nthresholds() * a.get_nprobs();
    for (SizeType i = sa.size() - nlast; i < sa.size(); ++i) {
        if (!sa[i].is_empty && sa[i].success_h1_cumul >= kMinPd &&
            sa[i].cost < best) {
            best     = sa[i].cost;
            pd_inrun = sa[i].success_h1_cumul;
        }
    }
    constexpr SizeType kNcheck = 65536;
    const float sigma          = std::sqrt(
        pd_inrun * (1.0F - pd_inrun) *
        ((1.0F / (16.0F * kNtrials)) + (1.0F / static_cast<float>(kNcheck))));
    const auto check = a.evaluate(path, kNcheck, 99);
    REQUIRE_THAT(check.back().success_h1_cumul,
                 WithinAbs(pd_inrun, 5.0F * sigma));
}

TEST_CASE("CUDA simulate_h1_scores does not depend on its chunking",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    // At nbins 1024 the simulation runs in chunks of 4096 trials: 8192
    // trials are two chunks, and their first 4096 match one chunk of 4096.
    const std::vector<float> branching(4, 2.0F);
    const DynamicThresholdScheme dyn(branching, 0.1F, 1024, 64, 4, 0.05F, 8.0F,
                                     40, 0.3F, 1.0F, 1.5F, 1, "shared", 7, 256,
                                     Exec::cuda());
    const auto two = dyn.simulate_h1_scores(8192, 5);
    const auto one = dyn.simulate_h1_scores(4096, 5);
    for (SizeType s = 0; s < branching.size(); ++s) {
        CAPTURE(s);
        REQUIRE(same_bits(
            std::vector<float>(
                one.begin() + static_cast<std::ptrdiff_t>(s * 4096),
                one.begin() + static_cast<std::ptrdiff_t>((s + 1) * 4096)),
            std::vector<float>(
                two.begin() + static_cast<std::ptrdiff_t>(s * 8192),
                two.begin() + static_cast<std::ptrdiff_t>((s * 8192) + 4096))));
    }
}

TEST_CASE("calibrate rejects invalid input", "[thresholds]") {
    auto dyn = make_small("cpu", 7);
    dyn.run(6);
    REQUIRE_THROWS_AS(dyn.calibrate(0.0F), std::invalid_argument);
    REQUIRE_THROWS_AS(dyn.calibrate(1.0F), std::invalid_argument);
    CalibrationOptions opts;
    opts.ntrials_cal = 0;
    REQUIRE_THROWS_AS(dyn.calibrate(0.1F, opts), std::invalid_argument);
    opts             = {};
    opts.ncandidates = 0;
    REQUIRE_THROWS_AS(dyn.calibrate(0.1F, opts), std::invalid_argument);
}

} // namespace loki
