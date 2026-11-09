#include <cmath>
#include <optional>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

// The EP planner forecasts a chunk's branching once and uses it for every run
// (lib/algorithms/ep_chunking.cpp). That relies on the forecast being the
// same for every reference segment.

namespace {

using loki::ParamLimit;
using loki::SizeType;

loki::search::PulsarSearchConfig
make_cfg(const std::vector<ParamLimit>& limits) {
    constexpr SizeType kNsamps = 1U << 22U;
    return {kNsamps,
            64e-6,
            /*nbins=*/64,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.2,
            /*use_fourier=*/true,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/8.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/std::nullopt,
            /*nsegments=*/32,
            /*snr_min=*/6.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/limits.size()};
}

void check_ref_seg_independent(const std::vector<ParamLimit>& limits,
                               std::string_view poly_basis) {
    const loki::plans::FFAPlan<loki::ComplexType> plan(make_cfg(limits));
    const SizeType nsegments = plan.get_nsegments().back();
    REQUIRE(nsegments > 2);
    const auto ref = plan.forecast_branching(poly_basis, 0);
    for (SizeType ref_seg = 1; ref_seg < nsegments; ++ref_seg) {
        INFO("basis " << poly_basis << " nparams " << limits.size()
                      << " ref_seg " << ref_seg);
        const auto forecast = plan.forecast_branching(poly_basis, ref_seg);
        REQUIRE(forecast.max_children == ref.max_children);
        REQUIRE(forecast.mean == ref.mean);
    }
}

} // namespace

TEST_CASE("Branching forecast does not depend on the reference segment",
          "[ep_branching]") {
    const ParamLimit freq{.min = 120.0, .max = 240.0};
    const ParamLimit accel{.min = -500.0, .max = 500.0};
    const ParamLimit jerk{.min = -2.0, .max = 2.0};
    const ParamLimit snap{.min = -0.01, .max = 0.01};
    const ParamLimit crackle{.min = -1e-5, .max = 1e-5};
    check_ref_seg_independent({accel, freq}, "taylor");
    check_ref_seg_independent({jerk, accel, freq}, "taylor");
    check_ref_seg_independent({snap, jerk, accel, freq}, "taylor");
    check_ref_seg_independent({crackle, snap, jerk, accel, freq}, "taylor");
    check_ref_seg_independent({jerk, accel, freq}, "chebyshev");
}

TEST_CASE("Non-power-of-two nsamps splits into equal EP segments",
          "[ep_branching]") {
    const std::vector<ParamLimit> limits{
        {.min = -10.0, .max = 10.0},
        {.min = 140.0, .max = 142.0},
    };
    // bseg_brute 1024 is a minimum: 36000 samples are 32 brute segments of
    // 1125, so the 8 EP segments of 4500 samples all hold data.
    const loki::search::PulsarSearchConfig exact(
        /*nsamps=*/36000, /*tsamp=*/64e-6, /*nbins=*/32, /*eta=*/1.0, limits,
        /*ducy_max=*/0.3, /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/1,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0, /*nbins_max=*/1024,
        /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/1024, /*nsegments=*/8);
    REQUIRE(exact.get_bseg_brute() == 1125U);
    REQUIRE(exact.get_bseg_ffa() == 4500U);
    const loki::plans::FFAPlan<float> exact_plan(exact);
    REQUIRE(exact_plan.get_nsegments().back() == 8U);
    REQUIRE(exact_plan.get_nsegments_active() == 8U);

    // EP keeps whole segments only: 35800 samples with short brute segments
    // are 2048 brute segments of floor(35800 / 2048) = 17, so EP analyses
    // the first 34816 samples (dropping 984 < 2048) as 128 full segments of
    // 272. Zero-filling instead would leave 3 empty EP segments and a
    // partial one.
    const loki::search::PulsarSearchConfig cfg(
        /*nsamps=*/35800, /*tsamp=*/64e-6, /*nbins=*/32, /*eta=*/1.0, limits,
        /*ducy_max=*/0.3, /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/1,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0, /*nbins_max=*/1024,
        /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/16, /*nsegments=*/128);
    REQUIRE(cfg.get_bseg_brute() == 17U);
    REQUIRE(cfg.get_bseg_ffa() == 272U);
    REQUIRE(cfg.get_nsamps() == 34816U);
    REQUIRE(cfg.get_nsamps_input() == 35800U);
    const loki::plans::FFAPlan<float> plan(cfg);
    REQUIRE(plan.get_nsegments().back() == 128U);
    REQUIRE(plan.get_nsegments_active() == 128U);
    REQUIRE(plan.get_nsegments_ref_pool(124U, false) == 128U);
    REQUIRE(plan.get_nsegments_ref_pool(std::nullopt, true) == 128U);

    const auto pattern = plan.get_branching_pattern_approx("taylor", 0U);
    REQUIRE(pattern.size() == plan.get_nsegments_active() - 1U);
    for (const double val : pattern) {
        REQUIRE(std::isfinite(val));
        REQUIRE(val >= 1.0);
    }
    const auto forecast = plan.forecast_branching("taylor", 0U);
    REQUIRE(forecast.mean.size() == plan.get_nsegments_active() - 1U);
    REQUIRE(forecast.max_children >= 1U);
}
