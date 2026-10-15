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
            /*bseg_ffa=*/kNsamps / 32,
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
