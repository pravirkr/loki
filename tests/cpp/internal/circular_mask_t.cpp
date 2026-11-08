#include <array>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"

#include "lib/core/circular.hpp"
#include "lib/detail/psr_utils.hpp"

namespace {

constexpr loki::SizeType kParams       = 5;
constexpr loki::SizeType kLeavesStride = (kParams + 2) * 2;
constexpr double kSignificance         = 2.0;

// One circular-basis leaf: (value, step) for crackle, snap, jerk, accel and
// velocity, then (d0, 0) and (f0, basis flag).
struct Leaf {
    double crackle, dcrackle, snap, dsnap, jerk, djerk, accel, daccel;
};

std::vector<double> make_leaves(std::span<const Leaf> leaves) {
    std::vector<double> out;
    out.reserve(leaves.size() * kLeavesStride);
    for (const auto& l : leaves) {
        const std::array<double, kLeavesStride> row = {
            l.crackle, l.dcrackle, l.snap, l.dsnap, l.jerk, l.djerk, l.accel,
            l.daccel,  0.0,        1.0e3,  0.0,     0.0,    100.0,   0.0};
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

// The class of each leaf: 0 Taylor, 1 circular snap, 2 circular crackle.
std::vector<int> classes(std::span<const Leaf> leaves) {
    const auto batch                   = make_leaves(leaves);
    const auto [snap, crackle, taylor] = loki::core::get_circ_taylor_mask(
        batch, leaves.size(), kParams, kSignificance);
    std::vector<int> out(leaves.size(), -1);
    for (const auto i : snap) {
        out[i] = 1;
    }
    for (const auto i : crackle) {
        out[i] = 2;
    }
    for (const auto i : taylor) {
        out[i] = 0;
    }
    return out;
}

// A leaf in the crackle hole: snap insignificant, jerk at 37 steps.
constexpr Leaf kHole = {.crackle  = 0.0,
                        .dcrackle = 6.99e-8,
                        .snap     = 0.0,
                        .dsnap    = 8.34e-6,
                        .jerk     = 4.5032e-2,
                        .djerk    = 1.23e-3,
                        .accel    = 11.255,
                        .daccel   = 0.147};

} // namespace

TEST_CASE("A crackle within round-off of zero is classed Taylor",
          "[circular]") {
    auto residue    = kHole;
    residue.crackle = -1.4036e-24; // -crackle * jerk > 0, formally physical
    residue.snap    = -3.5081e-22;
    auto zero       = kHole;
    auto real       = kHole;
    real.crackle    = -kHole.dcrackle / 2.0; // a child half a step from zero
    auto snap       = kHole;
    snap.snap       = -10.0 * kHole.dsnap; // -snap * accel > 0, significant
    const std::array leaves = {residue, zero, real, snap};
    CHECK(classes(leaves) == std::vector<int>{0, 0, 2, 1});
}

TEST_CASE("The middle child of a crackle split around zero is classed Taylor",
          "[circular]") {
    constexpr loki::SizeType kChildren = 3;
    const double dcrackle_parent       = kChildren * kHole.dcrackle;
    std::array<double, kChildren> children{};
    loki::psr_utils::branch_crackle_padded(children, 0.0, dcrackle_parent,
                                           kChildren);
    std::vector<Leaf> leaves;
    for (const auto c : children) {
        auto leaf     = kHole;
        leaf.crackle  = c;
        leaf.dcrackle = dcrackle_parent / kChildren;
        leaves.push_back(leaf);
        leaf.jerk = -leaf.jerk; // the other sign of jerk
        leaves.push_back(leaf);
    }
    // Child 0 (negative crackle) is physical with positive jerk, child 2 with
    // negative jerk; child 1, at zero or its round-off, never is.
    CHECK(classes(leaves) == std::vector<int>{2, 0, 0, 0, 0, 2});
}
