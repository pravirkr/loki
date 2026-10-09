#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/plans.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/prune_engine.hpp"
#include "lib/core/taylor.hpp"
#include "lib/utils/workspace_impl.hpp"

using loki::SizeType;

namespace {

constexpr SizeType kParams   = 3; // jerk, accel, freq
constexpr SizeType kStride   = (kParams + 2) * 2;
constexpr SizeType kNbins    = 64;
constexpr double kEta        = 1.0;
constexpr double kDt         = 100.0;
constexpr double kF0         = 100.0;
constexpr double kC          = 299792458.0;
constexpr SizeType kCanary   = 64;
constexpr double kCanaryVal  = -12345.0;
constexpr SizeType kCanaryIx = 987654321;

// Leaves whose current steps are `ratio` times the steps the next stage needs,
// so each of the three parameters branches about `ratio` ways.
std::vector<double> make_leaves(SizeType n_leaves, double ratio) {
    const double dphi    = kEta / static_cast<double>(kNbins);
    const double dfactor = kC / kF0;
    const double d3_new  = dphi * dfactor * 24.0 / (kDt * kDt * kDt);
    const double d2_new  = dphi * dfactor * 4.0 / (kDt * kDt);
    const double d1_new  = dphi * dfactor * 1.0 / kDt;
    std::vector<double> leaves(n_leaves * kStride, 0.0);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kStride;
        leaves[lo + 1]    = ratio * d3_new;
        leaves[lo + 3]    = ratio * d2_new;
        leaves[lo + 5]    = ratio * d1_new;
        leaves[lo + 8]    = kF0;
    }
    return leaves;
}

// The rectangular box from the overflow reproduction: 2^22 samples at 100 us,
// 64 segments, 64 bins, order 3.
loki::search::PulsarSearchConfig overflow_config() {
    constexpr SizeType kNsamps = 1U << 22U;
    constexpr SizeType kNseg   = 64;
    const std::vector<loki::ParamLimit> limits{
        {-146.0, 146.0},
        {-9743.0, 9743.0},
        {98.09, 114.94},
    };
    return {
        kNsamps,
        1.0e-4,
        kNbins,
        kEta,
        limits,
        /*ducy_max=*/0.5,
        /*wtsp=*/1.2,
        /*use_fourier=*/true,
        /*nthreads=*/1,
        /*max_process_memory_gb=*/8.0,
        /*octave_scale=*/2.0,
        /*nbins_max=*/1024,
        /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/kNsamps / kNseg / 16,
        /*bseg_ffa=*/kNsamps / kNseg,
        /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U,
        /*prune_poly_order=*/3,
    };
}

struct BranchRun {
    SizeType returned;
    bool canaries_intact;
};

// Branch into an output span of exactly `slots_per_leaf` per parent, followed
// by canaries that a write past the span would overwrite.
BranchRun branch_with_canaries(std::span<const double> leaves,
                               SizeType n_leaves,
                               SizeType slots_per_leaf) {
    const SizeType capacity = n_leaves * slots_per_leaf;
    std::vector<double> branch_buf((capacity + kCanary) * kStride, kCanaryVal);
    std::vector<SizeType> origins_buf(capacity + kCanary, kCanaryIx);
    loki::memory::BranchingWorkspace ws(n_leaves, slots_per_leaf, kParams);
    const SizeType returned = loki::core::poly_taylor_branch_batch(
        leaves, std::span(branch_buf).first(capacity * kStride),
        std::span(origins_buf).first(capacity), {0.0, kDt}, kNbins, kEta,
        slots_per_leaf, n_leaves, kParams, ws);
    const bool leaves_ok =
        std::all_of(branch_buf.begin() + static_cast<long>(capacity * kStride),
                    branch_buf.end(), [](double v) { return v == kCanaryVal; });
    const bool origins_ok = std::all_of(
        origins_buf.begin() + static_cast<long>(capacity), origins_buf.end(),
        [](SizeType v) { return v == kCanaryIx; });
    return {returned, leaves_ok && origins_ok};
}

} // namespace

TEST_CASE("Branch workspace uses the worst simulated leaf",
          "[branch_capacity]") {
    const loki::plans::FFAPlan<float> plan(overflow_config());
    const auto forecast = plan.forecast_branching("taylor");
    REQUIRE_FALSE(forecast.mean.empty());
    const auto peak_mean      = *std::ranges::max_element(forecast.mean);
    const auto old_branch_max = std::max(
        static_cast<SizeType>(std::ceil(peak_mean * 2.0)), SizeType{32});
    // The worst frequency's product sits above twice the mean, which is what
    // used to size the workspace (and above the floor of 32 on this box).
    REQUIRE(forecast.max_children > old_branch_max);
    REQUIRE(
        loki::algorithms::detail::compute_branch_max(forecast.max_children) ==
        std::max(forecast.max_children, SizeType{32}));
    // A quiet plan still keeps the floor.
    REQUIRE(loki::algorithms::detail::compute_branch_max(1) == 32);
}

TEST_CASE("Taylor branching that fits is written in full",
          "[branch_capacity]") {
    constexpr SizeType kLeaves = 4;
    const auto leaves          = make_leaves(kLeaves, 6.0);
    const auto run             = branch_with_canaries(leaves, kLeaves, 512);
    REQUIRE(run.returned > kLeaves * 8);
    REQUIRE(run.returned <= kLeaves * 512);
    REQUIRE(run.canaries_intact);
}

TEST_CASE("Taylor branching past the workspace throws before writing past it",
          "[branch_capacity]") {
    // About 6 ways per parameter, ~216 children per leaf, against 8 slots per
    // leaf. The check runs before that parent's Cartesian write.
    constexpr SizeType kLeaves       = 4;
    constexpr SizeType kSlotsPerLeaf = 8;
    const auto leaves                = make_leaves(kLeaves, 6.0);
    const SizeType capacity          = kLeaves * kSlotsPerLeaf;
    std::vector<double> branch_buf((capacity + kCanary) * kStride, kCanaryVal);
    std::vector<SizeType> origins_buf(capacity + kCanary, kCanaryIx);
    loki::memory::BranchingWorkspace ws(kLeaves, kSlotsPerLeaf, kParams);
    REQUIRE_THROWS_AS(loki::core::poly_taylor_branch_batch(
                          leaves,
                          std::span(branch_buf).first(capacity * kStride),
                          std::span(origins_buf).first(capacity), {0.0, kDt},
                          kNbins, kEta, kSlotsPerLeaf, kLeaves, kParams, ws),
                      std::runtime_error);
    REQUIRE(std::all_of(
        branch_buf.begin() + static_cast<long>(capacity * kStride),
        branch_buf.end(), [](double v) { return v == kCanaryVal; }));
    REQUIRE(std::all_of(origins_buf.begin() + static_cast<long>(capacity),
                        origins_buf.end(),
                        [](SizeType v) { return v == kCanaryIx; }));
}
