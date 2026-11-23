#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/core/dynamic.hpp"
#include "lib/search/ep_support.hpp"

using Catch::Matchers::ContainsSubstring;
using loki::ParamLimit;
using loki::SizeType;
using loki::search::PulsarSearchConfig;
using loki::search::detail::check_ep_seed_grid;

// EP seeds pruning on the FFA's acceleration x frequency grid and branches
// every higher derivative in pruning, so a plan that grids jerk, snap or
// crackle is refused before anything is seeded.

namespace {

PulsarSearchConfig make_cfg(SizeType nparams) {
    std::vector<ParamLimit> limits(nparams,
                                   ParamLimit{.min = -1.0, .max = 1.0});
    limits.back() = ParamLimit{.min = 140.0, .max = 145.0};
    return {1U << 16U,
            64e-6,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*region_growth_factor=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*nsegments=*/8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 20U,
            /*prune_poly_order=*/nparams,
            /*p_orb_min=*/1e-5,
            /*m_c_max=*/10.0,
            /*m_p_min=*/1.4,
            /*propagator_significance=*/2.0,
            /*validation_significance=*/5.0,
            /*use_conservative_tile=*/false,
            /*use_boxcar_kadane=*/false};
}

} // namespace

TEST_CASE("EP seed grid holds one point of every derivative above "
          "acceleration",
          "[ep][prune]") {
    CHECK_NOTHROW(check_ep_seed_grid(std::vector<SizeType>{4, 5}, "test"));
    CHECK_NOTHROW(check_ep_seed_grid(std::vector<SizeType>{1, 4, 5}, "test"));
    CHECK_NOTHROW(
        check_ep_seed_grid(std::vector<SizeType>{1, 1, 1, 4, 5}, "test"));
    CHECK_THROWS_WITH(
        check_ep_seed_grid(std::vector<SizeType>{3, 4, 5}, "test"),
        ContainsSubstring("grids jerk over 3 points") &&
            ContainsSubstring("use more segments"));
    CHECK_THROWS_WITH(
        check_ep_seed_grid(std::vector<SizeType>{2, 1, 4, 5}, "test"),
        ContainsSubstring("grids snap over 2 points"));
    CHECK_THROWS_WITH(
        check_ep_seed_grid(std::vector<SizeType>{2, 1, 1, 4, 5}, "test"),
        ContainsSubstring("grids crackle over 2 points"));
}

TEST_CASE("CPU pruning refuses an FFA seed grid over jerk", "[ep][prune]") {
    // Order 3 (Taylor basis) and five parameters (circular orbits), each
    // with three jerk points.
    const std::vector<std::vector<SizeType>> grids = {{3, 4, 5},
                                                      {1, 1, 3, 4, 5}};
    for (const auto& counts : grids) {
        const SizeType nparams = counts.size();
        CAPTURE(nparams);
        const std::vector<double> dparams(nparams, 0.1);
        CHECK_THROWS_WITH(loki::core::create_prune_dp_functs<float>(
                              "taylor", counts, dparams, /*nseg_ffa=*/8,
                              /*tseg_ffa=*/1.0, make_cfg(nparams),
                              /*batch_size=*/64, /*branch_max=*/16),
                          ContainsSubstring("use more segments"));
    }
}
