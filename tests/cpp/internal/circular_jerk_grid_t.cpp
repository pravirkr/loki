#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/core/circular.hpp"
#include "lib/core/dynamic.hpp"
#include "lib/core/taylor.hpp"
#include "lib/detail/psr_utils.hpp"

using Catch::Matchers::ContainsSubstring;
using loki::ParamLimit;
using loki::SizeType;
using loki::psr_utils::get_param_val_at_idx;

namespace {

// A circular-orbit search whose FFA grid holds three jerk points: the jerk
// range spans more than one segment-level step
constexpr SizeType kParams  = 5;
constexpr SizeType kStride  = (kParams + 2) * 2;
constexpr SizeType kNjerk   = 3;
constexpr SizeType kNaccel  = 4;
constexpr SizeType kNfreq   = 5;
constexpr SizeType kNbins   = 32;
constexpr double kF0        = 300.5;
constexpr double kPropSig   = 2.0;
constexpr double kOmegaOrb  = 2.0 * std::numbers::pi / 6000.0;
constexpr double kOmegaOrb2 = kOmegaOrb * kOmegaOrb;

std::vector<ParamLimit> make_limits() {
    return {
        ParamLimit{.min = -1.0e-7, .max = 1.0e-7}, // crackle
        ParamLimit{.min = -1.0e-4, .max = 1.0e-4}, // snap
        ParamLimit{.min = -0.06, .max = 0.06},     // jerk
        ParamLimit{.min = -50.0, .max = 50.0},     // acceleration
        ParamLimit{.min = 300.0, .max = 301.0},    // frequency
    };
}

// A leaf at the current coordinate, its steps small enough that every
// non-zero parameter is significant
std::vector<double>
make_leaf(double crackle, double snap, double jerk, double accel) {
    return {crackle, 1.0e-12, snap, 1.0e-9, jerk, 1.0e-6, accel,
            1.0e-3,  0.0,     1.0,  0.0,    0.0,  kF0,    0.0};
}

// The jerk plane of the leaf's FFA lookup at segments dts away (current and
// initial coordinates at 0), from the level-by-level and the ascend resolves
std::pair<std::vector<SizeType>, std::vector<SizeType>>
jerk_planes(const std::vector<double>& leaf, const std::vector<double>& dts) {
    const auto limits = make_limits();
    std::vector<SizeType> resolved;
    for (const auto dt : dts) {
        std::vector<SizeType> idx(1);
        std::vector<float> shift(1);
        loki::core::circ_taylor_resolve_batch(
            leaf, idx, shift, limits, {dt, 1.0}, {0.0, 1.0}, {0.0, 1.0}, kNjerk,
            kNaccel, kNfreq, kNbins, 1, kPropSig);
        resolved.push_back(idx[0] / (kNaccel * kNfreq));
    }
    std::vector<std::pair<double, double>> coord_segments;
    for (const auto dt : dts) {
        coord_segments.emplace_back(dt, 1.0);
    }
    std::vector<SizeType> idx(dts.size());
    std::vector<float> shift(dts.size());
    loki::core::circ_taylor_ascend_resolve_batch(
        leaf, idx, shift, limits, coord_segments, {0.0, 1.0}, kNjerk, kNaccel,
        kNfreq, kNbins, 1, dts.size(), kPropSig);
    std::vector<SizeType> ascended;
    for (const auto i : idx) {
        ascended.push_back(i / (kNaccel * kNfreq));
    }
    return {resolved, ascended};
}

} // namespace

TEST_CASE("poly_taylor_seed seeds every jerk point of the FFA grid",
          "[circular]") {
    const auto limits = make_limits();
    const std::vector<SizeType> counts{1, 1, kNjerk, kNaccel, kNfreq};
    const std::vector<double> dparams{1.0e-9, 1.0e-6, 0.04, 25.0, 0.2};
    const SizeType n_coords = kNjerk * kNaccel * kNfreq;
    std::vector<double> leaves(n_coords * kStride);

    const auto n_leaves = loki::core::poly_taylor_seed(
        counts, dparams, limits, leaves, {0.0, 1.0}, kParams);
    REQUIRE(n_leaves == n_coords);
    // In the FFA's coordinate order: jerk, acceleration, then frequency
    for (SizeType i = 0; i < n_coords; ++i) {
        CAPTURE(i);
        const auto* leaf = leaves.data() + (i * kStride);
        CHECK(leaf[0] == 0.0);
        CHECK(leaf[2] == 0.0);
        CHECK(leaf[4] ==
              get_param_val_at_idx(limits[2], kNjerk, i / (kNaccel * kNfreq)));
        CHECK(leaf[5] == dparams[2]);
        CHECK(leaf[6] ==
              get_param_val_at_idx(limits[3], kNaccel, (i / kNfreq) % kNaccel));
        CHECK(leaf[12] == get_param_val_at_idx(limits[4], kNfreq, i % kNfreq));
    }
}

TEST_CASE("poly_taylor_seed keeps zero jerk for a one-point jerk grid",
          "[circular]") {
    auto limits = make_limits();
    limits[2]   = ParamLimit{.min = -0.02, .max = 0.06};
    const std::vector<SizeType> counts{1, 1, 1, kNaccel, kNfreq};
    const std::vector<double> dparams{1.0e-9, 1.0e-6, 0.04, 25.0, 0.2};
    std::vector<double> leaves(kNaccel * kNfreq * kStride);

    REQUIRE(loki::core::poly_taylor_seed(counts, dparams, limits, leaves,
                                         {0.0, 1.0},
                                         kParams) == kNaccel * kNfreq);
    for (SizeType i = 0; i < kNaccel * kNfreq; ++i) {
        CHECK(leaves[(i * kStride) + 4] == 0.0);
    }
}

TEST_CASE("circular resolves look up the jerk plane of the leaf's jerk at the "
          "added segment",
          "[circular]") {
    // Jerk cells [-0.06, -0.02), [-0.02, 0.02) and [0.02, 0.06]
    SECTION("leaves just inside the bottom and top cells") {
        const std::vector<double> at_ref{0.0};
        CHECK(jerk_planes(make_leaf(0.0, 0.0, -0.059, 0.0), at_ref) ==
              std::pair{std::vector<SizeType>{0}, std::vector<SizeType>{0}});
        CHECK(jerk_planes(make_leaf(0.0, 0.0, 0.0, 0.0), at_ref) ==
              std::pair{std::vector<SizeType>{1}, std::vector<SizeType>{1}});
        CHECK(jerk_planes(make_leaf(0.0, 0.0, 0.059, 0.0), at_ref) ==
              std::pair{std::vector<SizeType>{2}, std::vector<SizeType>{2}});
    }
    SECTION("a Taylor leaf whose jerk crosses both cell edges") {
        // j(dt) = -0.03 + 1e-5 dt: -0.025, -0.01 and 0.03
        const std::vector<SizeType> planes{0, 1, 2};
        CHECK(jerk_planes(make_leaf(0.0, 1.0e-5, -0.03, 0.0),
                          {500.0, 2000.0, 6000.0}) ==
              std::pair{planes, planes});
    }
    SECTION("a snap-circular leaf whose jerk crosses both cell edges") {
        // j(dt) = -a omega sin(omega dt), a omega = 0.031
        const std::vector<SizeType> planes{0, 1, 2};
        CHECK(jerk_planes(make_leaf(0.0, -kOmegaOrb2 * 30.0, 0.0, 30.0),
                          {1500.0, 0.0, 4500.0}) == std::pair{planes, planes});
    }
    SECTION("a crackle-circular leaf whose jerk crosses both cell edges") {
        // j(dt) = 0.03 cos(omega dt)
        const std::vector<SizeType> planes{2, 1, 0};
        CHECK(jerk_planes(make_leaf(-kOmegaOrb2 * 0.03, 0.0, 0.03, 0.0),
                          {0.0, 1500.0, 3000.0}) == std::pair{planes, planes});
    }
}

TEST_CASE("Taylor and Chebyshev pruning refuse an FFA jerk grid",
          "[circular]") {
    constexpr SizeType kNsamps = 1U << 16U;
    constexpr SizeType kNseg   = 8;
    constexpr double kTsamp    = 64e-6;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -0.06, .max = 0.06},
        ParamLimit{.min = -50.0, .max = 50.0},
        ParamLimit{.min = 300.0, .max = 301.0},
    };
    const loki::search::PulsarSearchConfig cfg(
        kNsamps, kTsamp, kNbins, /*eta=*/1.0, limits, /*ducy_max=*/0.3,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/1,
        /*max_process_memory_gb=*/4.0, /*region_growth_factor=*/1.5,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/1024,
        /*nsegments=*/kNseg, /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U, /*prune_poly_order=*/3);
    const std::vector<SizeType> counts{kNjerk, kNaccel, kNfreq};
    const std::vector<double> dparams{0.04, 25.0, 0.2};
    const double tseg = static_cast<double>(kNsamps / kNseg) * kTsamp;
    CHECK_THROWS_WITH(loki::core::PrunePolyTaylorDPFuncts<float>(
                          counts, dparams, kNseg, tseg, cfg, 1024, 16),
                      ContainsSubstring("needs the circular basis"));
    CHECK_THROWS_WITH(loki::core::PrunePolyChebyshevDPFuncts<float>(
                          counts, dparams, kNseg, tseg, cfg, 1024, 16),
                      ContainsSubstring("needs the circular basis"));
}
