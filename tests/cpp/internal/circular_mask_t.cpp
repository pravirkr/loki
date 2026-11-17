#include <array>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"
#include "loki/search/leaf_basis.hpp"

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

TEST_CASE("The circular transform records each leaf's basis in its slot",
          "[circular]") {
    using loki::search::leaf_basis_value;
    using loki::search::LeafBasis;
    auto real                      = kHole;
    real.crackle                   = -kHole.dcrackle / 2.0;
    auto snap                      = kHole;
    snap.snap                      = -10.0 * kHole.dsnap;
    const std::array leaves        = {kHole, real, snap};
    auto batch                     = make_leaves(leaves);
    constexpr loki::SizeType kSlot = kLeavesStride - 1;
    // A leaf a previous stage propagated as an orbit can fall back to Taylor.
    batch[kSlot] = leaf_basis_value(LeafBasis::kCircularSnap);
    std::vector<loki::SizeType> indices(leaves.size());
    std::iota(indices.begin(), indices.end(), loki::SizeType{0});
    std::array<loki::SizeType, 2> counts{1, 0}; // the transform adds to them
    loki::core::circ_taylor_transform_batch(batch, indices, {412.3, 100.0},
                                            {0.0, 400.0}, leaves.size(), false,
                                            kSignificance, &counts);
    CHECK(counts == std::array<loki::SizeType, 2>{2, 1});
    CHECK(batch[kSlot] == leaf_basis_value(LeafBasis::kPolynomial));
    CHECK(batch[kLeavesStride + kSlot] ==
          leaf_basis_value(LeafBasis::kCircularCrackle));
    CHECK(batch[(2 * kLeavesStride) + kSlot] ==
          leaf_basis_value(LeafBasis::kCircularSnap));
}

namespace {

// Leaves around p_orb_max with omega_min^2 = 1e-4 and k = 5: a significant
// acceleration of 1 +/- 0.01 bounds |snap| below by 1e-4 (1 - 0.05).
constexpr double kOmegaMinSq  = 1e-4;
constexpr double kValidSig    = 5.0;
constexpr double kPOrbMinTiny = 1e-5; // omega_max far above every leaf
constexpr double kXMassHuge   = 1e6;  // no leaf near the mass limit

constexpr Leaf
orbit_leaf(double snap, double dsnap, double accel, double daccel) {
    return {.crackle  = 0.0,
            .dcrackle = 0.0,
            .snap     = snap,
            .dsnap    = dsnap,
            .jerk     = 0.0,
            .djerk    = 0.0,
            .accel    = accel,
            .daccel   = daccel};
}

const std::array<Leaf, 6> kPOrbMaxLeaves = {
    // 0: snap consistent with zero at a significant acceleration: a wide orbit
    orbit_leaf(0.0, 1e-7, 1.0, 0.01),
    // 1: a resolved orbit faster than omega_min
    orbit_leaf(-2e-4, 1e-6, 1.0, 0.01),
    // 2: a slower orbit whose cell still reaches omega_min: kept by the margin
    orbit_leaf(-0.9e-4, 2e-6, 1.0, 0.01),
    // 3: the same orbit measured finely enough: killed
    orbit_leaf(-0.9e-4, 1e-7, 1.0, 0.01),
    // 4: near a node, acceleration not significant: always kept
    orbit_leaf(0.0, 1e-7, 0.03, 0.01),
    // 5: snap still unmeasured (wide cell): kept
    orbit_leaf(0.0, 1e-4, 1.0, 0.01),
};

std::vector<loki::SizeType> validate_survivors(double omega_min_sq) {
    auto batch = make_leaves(kPOrbMaxLeaves);
    std::vector<loki::SizeType> origins(kPOrbMaxLeaves.size());
    std::iota(origins.begin(), origins.end(), loki::SizeType{0});
    const auto n = loki::core::circ_taylor_validate_batch(
        batch, origins, kPOrbMaxLeaves.size(), kPOrbMinTiny, omega_min_sq,
        kXMassHuge, kValidSig);
    origins.resize(n);
    return origins;
}

// omega_max^2 = 1e-4 (p_orb_min = 2 pi / 0.01 s) and, separately, a mass
// limit |accel| <= 1000 omega^(4/3); k = 5. The leaves marked "margin" are
// killed by the point estimate omega^2 = -snap / accel but their cells,
// widened by k widths, still reach the limit.
constexpr double kPOrbMinFast = 2.0 * std::numbers::pi / 0.01;
constexpr double kXMassHeavy  = 1000.0;

const std::array<Leaf, 4> kFastLeaves = {
    orbit_leaf(-2e-4, 1e-6, 1.0, 0.01),    // 0: omega^2 twice the limit
    orbit_leaf(-1.2e-4, 1e-6, 1.0, 0.01),  // 1: wholly faster: killed
    orbit_leaf(-1.08e-4, 1e-6, 1.0, 0.01), // 2: margin: kept
    orbit_leaf(-0.9e-4, 1e-6, 1.0, 0.01),  // 3: slower than the limit
};

const std::array<Leaf, 3> kHeavyLeaves = {
    orbit_leaf(-4e-4, 1e-7, 4.0, 0.01),   // 0: wholly too heavy: killed
    orbit_leaf(-2.3e-4, 1e-6, 2.3, 0.05), // 1: margin: kept
    orbit_leaf(-2e-4, 1e-6, 2.0, 0.05),   // 2: inside the mass limit
};

std::vector<loki::SizeType>
orbit_survivors(std::span<const Leaf> leaves, double p_orb_min, double x_mass) {
    auto batch = make_leaves(leaves);
    std::vector<loki::SizeType> origins(leaves.size());
    std::iota(origins.begin(), origins.end(), loki::SizeType{0});
    const auto n = loki::core::circ_taylor_validate_batch(
        batch, origins, leaves.size(), p_orb_min, 0.0, x_mass, kValidSig);
    origins.resize(n);
    return origins;
}

} // namespace

TEST_CASE("p_orb_max kills only cells that are wholly slower orbits",
          "[circular]") {
    CHECK(validate_survivors(kOmegaMinSq) ==
          std::vector<loki::SizeType>{1, 2, 4, 5});
    // Without p_orb_max (0) no leaf here is killed.
    CHECK(validate_survivors(0.0) ==
          std::vector<loki::SizeType>{0, 1, 2, 3, 4, 5});
    CHECK(loki::core::circ_omega_min_sq(std::nullopt) == 0.0);
}

TEST_CASE("The p_orb_min and mass checks kill only cells wholly beyond them",
          "[circular]") {
    CHECK(orbit_survivors(kFastLeaves, kPOrbMinFast, kXMassHuge) ==
          std::vector<loki::SizeType>{2, 3});
    CHECK(orbit_survivors(kHeavyLeaves, kPOrbMinTiny, kXMassHeavy) ==
          std::vector<loki::SizeType>{1, 2});
}
