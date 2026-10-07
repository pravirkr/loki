#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <span>
#include <vector>

#include "lib/core/taylor.hpp"
#include "lib/utils/workspace_impl.hpp"

using loki::SizeType;

namespace {

constexpr SizeType kParams   = 3; // jerk, accel, freq
constexpr SizeType kStride   = (kParams + 2) * 2;
constexpr SizeType kNbins    = 64;
constexpr double kEta        = 1.0;
constexpr double kDt         = 100.0; // t_obs - t_ref of the current coord
constexpr double kF0         = 100.0;
constexpr double kC          = 299792458.0;
constexpr SizeType kCanary   = 64;
constexpr double kCanaryVal  = -12345.0;
constexpr SizeType kCanaryIx = 987654321;

// Leaves whose current steps are `ratio` times the steps the next stage needs,
// so each of the three parameters branches about `ratio` ways and a leaf has
// about ratio^3 children.
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

struct BranchRun {
    SizeType returned;
    bool canaries_intact;
};

// Branch into an output span of exactly `capacity` leaves, followed by
// canaries that a write past the span would overwrite.
BranchRun branch_with_canaries(std::span<const double> leaves,
                               SizeType n_leaves,
                               SizeType branch_max) {
    const SizeType capacity = n_leaves * branch_max;
    std::vector<double> branch_buf((capacity + kCanary) * kStride, kCanaryVal);
    std::vector<SizeType> origins_buf(capacity + kCanary, kCanaryIx);
    loki::memory::BranchingWorkspace ws(n_leaves, branch_max, kParams);
    const SizeType returned = loki::core::poly_taylor_branch_batch(
        leaves, std::span(branch_buf).first(capacity * kStride),
        std::span(origins_buf).first(capacity), {0.0, kDt}, kNbins, kEta,
        branch_max, n_leaves, kParams, ws);
    const bool leaves_ok =
        std::all_of(branch_buf.begin() + static_cast<long>(capacity * kStride),
                    branch_buf.end(), [](double v) { return v == kCanaryVal; });
    const bool origins_ok =
        std::all_of(origins_buf.begin() + static_cast<long>(capacity),
                    origins_buf.end(), [](SizeType v) { return v == kCanaryIx; });
    return {returned, leaves_ok && origins_ok};
}

} // namespace

TEST_CASE("Taylor branching that fits is written in full",
          "[branch_capacity]") {
    // Accepting case: the same leaves branch (about 6 ways per parameter), and
    // a workspace of 512 slots per leaf holds all of them.
    constexpr SizeType kLeaves = 4;
    const auto leaves          = make_leaves(kLeaves, 6.0);
    const auto run             = branch_with_canaries(leaves, kLeaves, 512);
    REQUIRE(run.returned > kLeaves * 8);
    REQUIRE(run.returned <= kLeaves * 512);
    REQUIRE(run.canaries_intact);
}

TEST_CASE("Taylor branching past the workspace writes nothing beyond it",
          "[branch_capacity]") {
    // About 6 ways per parameter, ~216 children per leaf, against 8 slots per
    // leaf. Before the fix this wrote past the span (the caller's size check
    // ran only afterwards); now it stops writing and reports the total.
    constexpr SizeType kLeaves = 4;
    const auto leaves          = make_leaves(kLeaves, 6.0);
    const auto run             = branch_with_canaries(leaves, kLeaves, 8);
    REQUIRE(run.returned > kLeaves * 8);
    REQUIRE(run.canaries_intact);
}
