#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/algorithms/regions.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using Catch::Matchers::ContainsSubstring;
using loki::ComplexType;
using loki::ParamLimit;
using loki::SizeType;
using loki::regions::FFARegionPlanner;
using loki::regions::generate_ffa_regions;
using loki::search::PulsarSearchConfig;

// --- generate_ffa_regions: correctness / foolproof-ness guards ---

TEST_CASE("generate_ffa_regions rejects octave_scale <= 1.0", "[regions]") {
    // octave_scale == 1.0 used to make p_cur_high == p_cur_low on every
    // iteration of the covering loop, so it never advanced and hung
    // forever. It must now be rejected immediately instead of hanging.
    REQUIRE_THROWS_AS(generate_ffa_regions(/*p_min=*/0.002, /*p_max=*/1.0,
                                           /*tsamp=*/1e-5,
                                           /*nbins_min=*/32, /*eta_min=*/1.0,
                                           /*octave_scale=*/1.0),
                      std::runtime_error);
    REQUIRE_THROWS_AS(
        generate_ffa_regions(0.002, 1.0, 1e-5, 32, 1.0, /*octave_scale=*/0.9),
        std::runtime_error);
}

TEST_CASE("generate_ffa_regions accepts octave_scale > 1.0 and terminates",
          "[regions]") {
    const auto regions =
        generate_ffa_regions(0.002, 1.0, 1e-5, 32, 1.0, /*octave_scale=*/1.5);
    REQUIRE_FALSE(regions.empty());
    // Coverage should span the full requested frequency range: the first
    // region's upper edge is 1/p_min, the last region's lower edge is
    // 1/p_max.
    constexpr double kTol = 1e-6;
    CHECK(std::abs(regions.front().f_end - (1.0 / 0.002)) < kTol);
    CHECK(std::abs(regions.back().f_start - (1.0 / 1.0)) < kTol);
}

TEST_CASE("generate_ffa_regions adjacent octave bands share frequency edges",
          "[regions]") {
    const auto regions =
        generate_ffa_regions(/*p_min=*/0.04, /*p_max=*/1.0, /*tsamp=*/1e-3,
                             /*nbins_min=*/32, /*eta_min=*/1.0,
                             /*octave_scale=*/2.0);
    REQUIRE(regions.size() >= 2);
    constexpr double kTol = 1e-9;
    for (SizeType i = 0; i + 1 < regions.size(); ++i) {
        CHECK(std::abs(regions[i].f_start - regions[i + 1].f_end) < kTol);
    }
}

TEST_CASE("generate_ffa_regions rejects f_max at/above the Nyquist frequency",
          "[regions]") {
    // p_min == 2*tsamp means f_max == Nyquist frequency exactly. Previously
    // this silently truncated nbins_cur to 0, propagating division-by-zero
    // (inf/nan) through every downstream region instead of failing clearly.
    constexpr double kTsamp = 1e-3;
    REQUIRE_THROWS_AS(
        generate_ffa_regions(/*p_min=*/2.0 * kTsamp, /*p_max=*/1.0, kTsamp, 32,
                             1.0),
        std::runtime_error);
    // Comfortably above the Nyquist limit must still throw if it can't
    // resolve at least 2 samples per bin for the requested nbins_min.
    REQUIRE_THROWS_AS(
        generate_ffa_regions(/*p_min=*/1.5 * kTsamp, /*p_max=*/1.0, kTsamp, 32,
                             1.0),
        std::runtime_error);
    // Comfortably above both Nyquist and nbins_min*tsamp must succeed.
    REQUIRE_NOTHROW(generate_ffa_regions(/*p_min=*/40.0 * kTsamp,
                                         /*p_max=*/1.0, kTsamp, 32, 1.0));
}

TEST_CASE("generate_ffa_regions throws when nbins_min does not fit in "
          "p_min/tsamp",
          "[regions]") {
    // fold_bins=32 at f_max=500 Hz with tsamp=6.4e-5 only has ~31.25 samples
    // per period. Previously this silently planned nbins=31.
    constexpr double kTsamp = 6.4e-5;
    constexpr double kPmin  = 1.0 / 500.0; // 0.002 s
    REQUIRE(kPmin / kTsamp < 32.0);
    REQUIRE_THROWS_AS(generate_ffa_regions(kPmin, /*p_max=*/1.0, kTsamp,
                                           /*nbins_min=*/32,
                                           /*eta_min=*/0.5),
                      std::runtime_error);
    try {
        generate_ffa_regions(kPmin, 1.0, kTsamp, 32, 0.5);
        FAIL("expected throw");
    } catch (const std::runtime_error& err) {
        const std::string msg = err.what();
        CHECK_THAT(msg, ContainsSubstring("nbins_min"));
        CHECK_THAT(msg, ContainsSubstring("f_max"));
    }
}

// --- FFARegionPlanner: memory-fitting foolproof-ness ---

TEST_CASE("FFARegionPlanner fails fast with an actionable diagnostic when "
          "drift expansion (driven by a fixed acceleration range) makes the "
          "minimum-viable chunk infeasible",
          "[regions]") {
    // Mirrors a reported real-world failure: an extreme acceleration range
    // combined with a very tight eta makes the top of the frequency range
    // infeasible. The acceleration range is held fixed across every
    // frequency chunk (only frequency is narrowed), and it drives a drift
    // fraction large enough that its drift-expansion floor alone dominates
    // the actual (drift-expanded) search window -- so no amount of
    // frequency-chunk subdivision can fix this. The planner must say so
    // explicitly rather than implying frequency granularity is the fixable
    // variable.
    constexpr SizeType kNsamps = 1U << 24U; // ~16.8M samples
    constexpr double kTsamp    = 6.4e-5;    // tobs ~= 1073.7 s
    // f_max must satisfy p_min >= nbins*tsamp (f_max <= ~488 Hz here) so
    // generate_ffa_regions accepts the request and the planner itself fails.
    const std::vector<ParamLimit> param_limits = {
        {.min = -5000.0, .max = 5000.0}, // acceleration (m/s^2)
        {.min = 1.0, .max = 400.0},      // frequency (Hz)
    };
    const PulsarSearchConfig cfg(kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.5,
                                 param_limits, /*ducy_max=*/0.2,
                                 /*wtsp=*/1.5, /*use_fourier=*/true,
                                 /*nthreads=*/1,
                                 /*max_process_memory_gb=*/8.0);

    bool threw = false;
    try {
        FFARegionPlanner<ComplexType> planner(cfg);
    } catch (const std::runtime_error& err) {
        threw                  = true;
        const std::string msg = err.what();
        CHECK_THAT(msg, ContainsSubstring("Cannot fit minimum viable chunk"));
        CHECK_THAT(msg, ContainsSubstring("drift"));
    }
    CHECK(threw);
}

TEST_CASE("FFARegionPlanner succeeds for a modest, physically reasonable "
          "search",
          "[regions]") {
    // Regression guard: a small, realistic search should still plan
    // successfully after the hardening changes above.
    constexpr SizeType kNsamps = 1U << 14U; // 16384 samples
    constexpr double kTsamp    = 1e-3;      // tobs ~= 16.384 s
    const std::vector<ParamLimit> param_limits = {
        {.min = -10.0, .max = 10.0}, // acceleration (m/s^2), realistic
        {.min = 1.0, .max = 20.0},   // frequency (Hz); p_min >= nbins*tsamp
    };
    const PulsarSearchConfig cfg(kNsamps, kTsamp, /*nbins=*/32, /*eta=*/1.0,
                                 param_limits, 0.2, 1.5, true, 1, 8.0);

    REQUIRE_NOTHROW(FFARegionPlanner<ComplexType>(cfg));
}

TEST_CASE("FFARegionPlanner does not fail for drift above the suspicious "
          "warning threshold but below the impossible (100% of c) limit",
          "[regions]") {
    // max_drift = accel * (tobs/2) / c. Chosen so max_drift is comfortably
    // above the 1% "suspicious" warning threshold but nowhere near the
    // 100%-of-c hard error: this must only warn, never throw.
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 100.0 / static_cast<double>(kNsamps); // tobs=100s
    constexpr double kAccel = 299800.0; // m/s^2 (unphysically large, by design)
    const std::vector<ParamLimit> param_limits = {
        {.min = -kAccel, .max = kAccel},
        {.min = 1.0, .max = 15.0}, // p_min >= nbins*tsamp
    };
    // Loose eta and few bins keep the parameter grid small so the only
    // thing under test is the drift-fraction warning/error boundary, not
    // the memory-fitting logic exercised by the other test cases above.
    const PulsarSearchConfig cfg(kNsamps, kTsamp, /*nbins=*/8, /*eta=*/4.0,
                                 param_limits, 0.2, 1.5, true, 1, 8.0);

    REQUIRE_NOTHROW(FFARegionPlanner<ComplexType>(cfg));
}

TEST_CASE("FFARegionPlanner frequency-only search has no drift expansion",
          "[regions]") {
    constexpr SizeType kNsamps                 = 1U << 14U;
    constexpr double kTsamp                    = 1e-3;
    const std::vector<ParamLimit> param_limits = {
        {.min = 1.0, .max = 20.0},
    };
    const PulsarSearchConfig cfg(kNsamps, kTsamp, /*nbins=*/16, /*eta=*/1.0,
                                 param_limits, 0.2, 1.5, true, 1, 8.0);

    FFARegionPlanner<ComplexType> planner(cfg);
    REQUIRE(planner.get_nregions() >= 1);
    for (const auto& chunk_cfg : planner.get_cfgs()) {
        const auto freq = chunk_cfg.get_param_limits().back();
        CHECK(freq.min >= 1.0);
        CHECK(freq.max <= 20.0);
    }
}

TEST_CASE("FFARegionPlanner splits a band when the full range exceeds memory",
          "[regions]") {
    // One octave band [40, 80] Hz; dense enough that the whole band does
    // not fit in the 0.5 GB effective budget, so the planner must bisect.
    constexpr SizeType kNsamps                 = 1U << 20U;
    constexpr double kTsamp                    = 6.4e-5;
    const std::vector<ParamLimit> param_limits = {
        {.min = -200.0, .max = 200.0},
        {.min = 40.0, .max = 80.0},
    };
    // 0.7 GB cap -> 0.2 GB effective after the 0.5 GB safety margin. The
    // full [40, 80] Hz band is ~0.39 GB, so the planner must bisect.
    const PulsarSearchConfig cfg(kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.25,
                                 param_limits, 0.2, 1.5, true, 1, 0.7);

    const auto bands =
        generate_ffa_regions(1.0 / 80.0, 1.0 / 40.0, kTsamp, 32, 0.25, 2.0);
    REQUIRE(bands.size() == 1);
    FFARegionPlanner<ComplexType> planner(cfg);
    REQUIRE(planner.get_nregions() > 1);
    constexpr double kEffectiveLimitGB = 0.7 - 0.5;
    CHECK(planner.get_stats().get_freq_sweep_memory_usage() <=
          kEffectiveLimitGB + 1.0e-6);
}
