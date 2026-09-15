#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/prune_mask.hpp"
#include "loki/psr_utils.hpp"
#include "loki/utils.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::GridMask;
using loki::algorithms::is_impulsive_segment;
using loki::algorithms::kBirdieAccelPad;
using loki::algorithms::make_birdie_window;
using loki::algorithms::make_harvest_window;
using loki::algorithms::make_pulsar_window;
using loki::algorithms::ParamWindow;
using loki::algorithms::PruneRFIConfig;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

const ParamLimit kLimAccel{.min = -100.0, .max = 100.0};
const ParamLimit kLimFreq{.min = 100.0, .max = 200.0};
constexpr SizeType kNAccel = 11;
constexpr SizeType kNFreq  = 1000;

// Brute-force reference: is the grid cell (ia, if) inside any window?
bool cell_in_windows(SizeType ia,
                     SizeType ifreq,
                     const std::vector<ParamWindow>& windows) {
    // resolve() snaps a value v to get_nearest_idx_analytical(v); a cell is
    // covered iff some v in the window snaps to it. Check the cell's value
    // range against the window using the same snapping on the bounds.
    for (const auto& w : windows) {
        if (w.f_hi < kLimFreq.min || w.f_lo > kLimFreq.max ||
            w.a_hi < kLimAccel.min || w.a_lo > kLimAccel.max) {
            continue;
        }
        const auto f_lo = std::clamp(w.f_lo, kLimFreq.min, kLimFreq.max);
        const auto f_hi = std::clamp(w.f_hi, kLimFreq.min, kLimFreq.max);
        const auto a_lo = std::clamp(w.a_lo, kLimAccel.min, kLimAccel.max);
        const auto a_hi = std::clamp(w.a_hi, kLimAccel.min, kLimAccel.max);
        const auto if_lo =
            loki::psr_utils::get_nearest_idx_analytical(f_lo, kLimFreq, kNFreq);
        const auto if_hi =
            loki::psr_utils::get_nearest_idx_analytical(f_hi, kLimFreq, kNFreq);
        const auto ia_lo = loki::psr_utils::get_nearest_idx_analytical(
            a_lo, kLimAccel, kNAccel);
        const auto ia_hi = loki::psr_utils::get_nearest_idx_analytical(
            a_hi, kLimAccel, kNAccel);
        if (ifreq >= if_lo && ifreq <= if_hi && ia >= ia_lo && ia <= ia_hi) {
            return true;
        }
    }
    return false;
}

SizeType brute_count(const std::vector<ParamWindow>& windows) {
    SizeType n = 0;
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        for (SizeType ifreq = 0; ifreq < kNFreq; ++ifreq) {
            n += cell_in_windows(ia, ifreq, windows) ? 1U : 0U;
        }
    }
    return n;
}

GridMask make_mask() { return {kLimAccel, kNAccel, kLimFreq, kNFreq}; }

} // namespace

TEST_CASE("GridMask: construction and empty state", "[prune_mask]") {
    const auto mask = make_mask();
    REQUIRE(mask.empty());
    REQUIRE(mask.count() == 0);
    REQUIRE(mask.get_n_cells() == kNAccel * kNFreq);
    REQUIRE(mask.get_n_windows() == 0);
    for (SizeType i = 0; i < mask.get_n_cells(); i += 97) {
        REQUIRE_FALSE(mask.is_masked(i));
    }
    REQUIRE_THROWS(GridMask(kLimAccel, 0, kLimFreq, kNFreq));
    REQUIRE_THROWS(GridMask(ParamLimit{.min = 1.0, .max = 0.0}, kNAccel,
                            kLimFreq, kNFreq));
}

TEST_CASE("GridMask: frequency-only window masks all accelerations",
          "[prune_mask]") {
    auto mask = make_mask();
    const ParamWindow w{.f_lo = 150.0, .f_hi = 150.5};
    mask.add_window(w);
    REQUIRE_FALSE(mask.empty());
    REQUIRE(mask.get_n_windows() == 1);
    const std::vector<ParamWindow> windows{w};
    REQUIRE(mask.count() == brute_count(windows));
    // Every accel row has the same masked frequency cells
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        for (SizeType ifreq = 0; ifreq < kNFreq; ++ifreq) {
            const auto idx = (ia * kNFreq) + ifreq;
            REQUIRE(mask.is_masked(idx) == cell_in_windows(ia, ifreq, windows));
        }
    }
}

TEST_CASE("GridMask: 2D window and clipping", "[prune_mask]") {
    auto mask = make_mask();
    // Partially outside the grid in both dimensions
    const ParamWindow w{
        .f_lo = 195.0, .f_hi = 250.0, .a_lo = 80.0, .a_hi = 500.0};
    mask.add_window(w);
    const std::vector<ParamWindow> windows{w};
    REQUIRE(mask.count() == brute_count(windows));
    REQUIRE(mask.count() > 0);
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        for (SizeType ifreq = 0; ifreq < kNFreq; ++ifreq) {
            const auto idx = (ia * kNFreq) + ifreq;
            REQUIRE(mask.is_masked(idx) == cell_in_windows(ia, ifreq, windows));
        }
    }
}

TEST_CASE("GridMask: disjoint windows are ignored", "[prune_mask]") {
    auto mask = make_mask();
    mask.add_window(ParamWindow{.f_lo = 10.0, .f_hi = 50.0});
    mask.add_window(ParamWindow{.f_lo = 300.0, .f_hi = 400.0});
    mask.add_window(ParamWindow{
        .f_lo = 150.0, .f_hi = 151.0, .a_lo = 200.0, .a_hi = 300.0});
    mask.add_window(ParamWindow{
        .f_lo = 150.0, .f_hi = 151.0, .a_lo = -kInf, .a_hi = -150.0});
    REQUIRE(mask.empty());
    REQUIRE(mask.get_n_windows() == 0);
}

TEST_CASE("GridMask: grid edges and single-cell windows", "[prune_mask]") {
    auto mask = make_mask();
    // Exactly the grid maximum should map to the last cell, not overflow
    mask.add_window(ParamWindow{.f_lo = kLimFreq.max, .f_hi = kLimFreq.max});
    REQUIRE(mask.count() == kNAccel);
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        REQUIRE(mask.is_masked((ia * kNFreq) + kNFreq - 1));
    }
    mask.clear();
    REQUIRE(mask.empty());
    mask.add_window(ParamWindow{.f_lo = kLimFreq.min, .f_hi = kLimFreq.min});
    REQUIRE(mask.count() == kNAccel);
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        REQUIRE(mask.is_masked(ia * kNFreq));
    }
}

TEST_CASE("GridMask: count is idempotent for overlapping windows",
          "[prune_mask]") {
    auto mask = make_mask();
    const ParamWindow w1{.f_lo = 120.0, .f_hi = 130.0};
    const ParamWindow w2{.f_lo = 125.0, .f_hi = 135.0};
    mask.add_window(w1);
    const auto c1 = mask.count();
    mask.add_window(w1);
    REQUIRE(mask.count() == c1);
    mask.add_window(w2);
    REQUIRE(mask.count() == brute_count({w1, w2}));
}

TEST_CASE("GridMask: harmonics expand the window", "[prune_mask]") {
    auto mask = make_mask();
    const ParamWindow w{
        .f_lo = 60.0, .f_hi = 61.0, .a_lo = -10.0, .a_hi = 10.0};
    // Fundamental is below the grid; 2nd (120-122) and 3rd (180-183) harmonics
    // are inside, subharmonics (30, 20) are outside.
    mask.add_window(w, 3);
    const std::vector<ParamWindow> expected{
        ParamWindow{.f_lo = 120.0, .f_hi = 122.0, .a_lo = -10.0, .a_hi = 10.0},
        ParamWindow{.f_lo = 180.0, .f_hi = 183.0, .a_lo = -10.0, .a_hi = 10.0},
    };
    REQUIRE(mask.get_n_windows() == 2);
    REQUIRE(mask.count() == brute_count(expected));
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        for (SizeType ifreq = 0; ifreq < kNFreq; ++ifreq) {
            const auto idx = (ia * kNFreq) + ifreq;
            REQUIRE(mask.is_masked(idx) ==
                    cell_in_windows(ia, ifreq, expected));
        }
    }

    // Subharmonic: fundamental at 300 Hz (outside), f/2 = 150 inside
    auto mask2 = make_mask();
    mask2.add_window(ParamWindow{.f_lo = 300.0, .f_hi = 302.0}, 2);
    REQUIRE(mask2.count() ==
            brute_count({ParamWindow{.f_lo = 150.0, .f_hi = 151.0}}));
}

TEST_CASE("GridMask: is_masked matches brute force on random windows",
          "[prune_mask]") {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> uf(90.0, 210.0);
    std::uniform_real_distribution<double> ua(-120.0, 120.0);
    std::uniform_real_distribution<double> width(0.0, 5.0);
    auto mask = make_mask();
    std::vector<ParamWindow> windows;
    for (int i = 0; i < 40; ++i) {
        const auto f  = uf(rng);
        const auto a  = ua(rng);
        const auto wf = width(rng);
        const auto wa = width(rng) * 10.0;
        ParamWindow w{.f_lo = f, .f_hi = f + wf, .a_lo = a, .a_hi = a + wa};
        if (i % 5 == 0) {
            w.a_lo = -kInf;
            w.a_hi = kInf;
        }
        windows.push_back(w);
        mask.add_window(w);
    }
    REQUIRE(mask.count() == brute_count(windows));
    for (SizeType ia = 0; ia < kNAccel; ++ia) {
        for (SizeType ifreq = 0; ifreq < kNFreq; ++ifreq) {
            const auto idx = (ia * kNFreq) + ifreq;
            REQUIRE(mask.is_masked(idx) == cell_in_windows(ia, ifreq, windows));
        }
    }
}

TEST_CASE("GridMask: assign copies the base mask", "[prune_mask]") {
    auto base = make_mask();
    base.add_window(ParamWindow{.f_lo = 140.0, .f_hi = 141.0});
    GridMask overlay;
    overlay.assign(base);
    REQUIRE(overlay.count() == base.count());
    REQUIRE(overlay.get_n_cells() == base.get_n_cells());
    // Overlay additions must not touch the base
    overlay.add_window(ParamWindow{.f_lo = 160.0, .f_hi = 161.0});
    REQUIRE(overlay.count() > base.count());
    REQUIRE(base.count() ==
            brute_count({ParamWindow{.f_lo = 140.0, .f_hi = 141.0}}));
    // Re-assign resets the overlay
    overlay.assign(base);
    REQUIRE(overlay.count() == base.count());
}

TEST_CASE("GridMask: filter_resolved compacts all arrays consistently",
          "[prune_mask]") {
    auto mask = make_mask();
    mask.add_window(ParamWindow{.f_lo = 150.0, .f_hi = 160.0});

    constexpr SizeType kStride = 8;
    constexpr SizeType kN      = 500;
    std::mt19937 rng(42);
    std::uniform_int_distribution<SizeType> cell(0, (kNAccel * kNFreq) - 1);

    std::vector<double> leaves(kN * kStride);
    std::vector<SizeType> origins(kN);
    std::vector<SizeType> param_idx(kN);
    std::vector<float> phase(kN);
    for (SizeType i = 0; i < kN; ++i) {
        for (SizeType j = 0; j < kStride; ++j) {
            leaves[(i * kStride) + j] = static_cast<double>((i * 100) + j);
        }
        origins[i]   = i * 3;
        param_idx[i] = cell(rng);
        phase[i]     = static_cast<float>(i) * 0.5F;
    }
    // Expected survivors (in order)
    std::vector<SizeType> expected;
    for (SizeType i = 0; i < kN; ++i) {
        if (!mask.is_masked(param_idx[i])) {
            expected.push_back(i);
        }
    }
    REQUIRE(!expected.empty());
    REQUIRE(expected.size() < kN);

    const auto n_keep =
        mask.filter_resolved(leaves, origins, param_idx, phase, kStride, kN);
    REQUIRE(n_keep == expected.size());
    for (SizeType k = 0; k < n_keep; ++k) {
        const auto i = expected[k];
        REQUIRE(origins[k] == i * 3);
        REQUIRE(phase[k] == static_cast<float>(i) * 0.5F);
        REQUIRE_FALSE(mask.is_masked(param_idx[k]));
        for (SizeType j = 0; j < kStride; ++j) {
            REQUIRE(leaves[(k * kStride) + j] ==
                    static_cast<double>((i * 100) + j));
        }
    }

    // Empty mask is a no-op
    const auto empty = make_mask();
    std::vector<SizeType> origins2(origins);
    const auto n_all = empty.filter_resolved(leaves, origins2, param_idx, phase,
                                             kStride, n_keep);
    REQUIRE(n_all == n_keep);
    REQUIRE(origins2 == origins);
}

TEST_CASE("GridMask: select_seeds returns unmasked identity indices",
          "[prune_mask]") {
    auto mask = make_mask();
    mask.add_window(
        ParamWindow{.f_lo = 100.0, .f_hi = 110.0, .a_lo = 0.0, .a_hi = 0.0});
    const auto n_seeds = mask.get_n_cells();
    std::vector<SizeType> keep(n_seeds);
    const auto n_keep = mask.select_seeds(keep, n_seeds);
    REQUIRE(n_keep == n_seeds - mask.count());
    SizeType k = 0;
    for (SizeType i = 0; i < n_seeds; ++i) {
        if (!mask.is_masked(i)) {
            REQUIRE(keep[k] == i);
            ++k;
        }
    }
    REQUIRE(k == n_keep);
    REQUIRE_THROWS(mask.select_seeds(keep, n_seeds + 1));
}

TEST_CASE("make_pulsar_window covers the Doppler sweep", "[prune_mask]") {
    const double f    = 150.0;
    const double a    = 50.0;
    const double tobs = 1000.0;
    const auto w      = make_pulsar_window(f, a, tobs, 0.01, 5.0);
    const auto sweep  = f * a * tobs * loki::utils::kInvCval;
    using Catch::Matchers::WithinRel;
    REQUIRE_THAT(w.f_lo, WithinRel(f - sweep - 0.01, 1e-12));
    REQUIRE_THAT(w.f_hi, WithinRel(f + sweep + 0.01, 1e-12));
    REQUIRE(w.a_lo == 45.0);
    REQUIRE(w.a_hi == 55.0);
    // Symmetric in the sign of a; default a_pad masks all accelerations
    const auto w_neg = make_pulsar_window(f, -a, tobs, 0.01, 5.0);
    REQUIRE(w_neg.f_lo == w.f_lo);
    REQUIRE(w_neg.f_hi == w.f_hi);
    const auto w_all = make_pulsar_window(f, a, tobs);
    REQUIRE(w_all.a_lo == std::numeric_limits<double>::lowest());
    REQUIRE(w_all.a_hi == std::numeric_limits<double>::max());
    REQUIRE_THROWS(make_pulsar_window(-1.0, a, tobs));
}

TEST_CASE("PruneRFIConfig::validate", "[prune_mask]") {
    PruneRFIConfig cfg;
    REQUIRE_NOTHROW(cfg.validate(16));
    REQUIRE_FALSE(cfg.is_active());

    cfg.pulsar_mask.push_back(ParamWindow{.f_lo = 2.0, .f_hi = 1.0});
    REQUIRE_THROWS(cfg.validate(16));
    cfg.pulsar_mask.back() = ParamWindow{.f_lo = 1.0, .f_hi = 2.0};
    REQUIRE_NOTHROW(cfg.validate(16));
    REQUIRE(cfg.is_active());

    cfg.harvest_scheme.assign(10, 12.0F);
    REQUIRE_THROWS(cfg.validate(16)); // wrong size
    cfg.harvest_scheme.assign(15, 12.0F);
    REQUIRE_NOTHROW(cfg.validate(16));
    REQUIRE(cfg.has_harvest());
    cfg.harvest_mask_ntiles = 0.0;
    REQUIRE_THROWS(cfg.validate(16));
    cfg.harvest_mask_ntiles = 4.0;

    cfg.impulsive_veto  = true;
    cfg.impulsive_kappa = -1.0;
    REQUIRE_THROWS(cfg.validate(16));
    cfg.impulsive_kappa     = 6.0;
    cfg.impulsive_min_level = 0;
    REQUIRE_THROWS(cfg.validate(16));
    cfg.impulsive_min_level = 4;
    REQUIRE_NOTHROW(cfg.validate(16));
}

TEST_CASE("make_birdie_window is a narrow zero-acceleration band",
          "[prune_mask]") {
    const auto w = make_birdie_window(150.0, 0.05);
    REQUIRE(w.f_lo == 150.0 - 0.05);
    REQUIRE(w.f_hi == 150.0 + 0.05);
    REQUIRE(w.a_lo == -kBirdieAccelPad);
    REQUIRE(w.a_hi == kBirdieAccelPad);
    const auto w2 = make_birdie_window(150.0, 0.1, 0.5);
    REQUIRE(w2.a_lo == -0.5);
    REQUIRE(w2.a_hi == 0.5);
    REQUIRE_THROWS(make_birdie_window(-1.0, 0.1));
    REQUIRE_THROWS(make_birdie_window(150.0, -0.1));
}

TEST_CASE("make_harvest_window pads by tiles plus Doppler reach",
          "[prune_mask]") {
    const double f      = 150.0;
    const double a      = 40.0;
    const double df     = 0.02;
    const double da     = 1.5;
    const double t_ref  = 200.0;
    const double tobs   = 1000.0;
    const double ntiles = 4.0;
    const auto w     = make_harvest_window(f, a, df, da, t_ref, tobs, ntiles);
    const auto reach = std::max(t_ref, tobs - t_ref);
    const auto sweep = f * a * reach * loki::utils::kInvCval;
    const auto f_pad = (ntiles * df) + sweep;
    using Catch::Matchers::WithinRel;
    REQUIRE_THAT(w.f_lo, WithinRel(f - f_pad, 1e-12));
    REQUIRE_THAT(w.f_hi, WithinRel(f + f_pad, 1e-12));
    REQUIRE_THAT(w.a_lo, WithinRel(a - (ntiles * da), 1e-12));
    REQUIRE_THAT(w.a_hi, WithinRel(a + (ntiles * da), 1e-12));
    REQUIRE_THROWS(make_harvest_window(f, a, df, da, t_ref, tobs, 0.0));
}

TEST_CASE("is_impulsive_segment matches the share-of-power contract",
          "[prune_mask]") {
    constexpr double kKappa = 3.0;
    constexpr float kMinSnr = 4.0F;
    // Coherent continuation: gain = Sc^2 / (n_seg+1) is never > kappa times
    // that share when kappa > 1.
    const float sc       = 10.0F;
    const SizeType n_seg = 4; // n_seg+1 = 5 > kappa, so a new spike would fire
    const float sp_coherent = sc * std::sqrt(static_cast<float>(n_seg) /
                                             static_cast<float>(n_seg + 1));
    REQUIRE_FALSE(
        is_impulsive_segment(sc, sp_coherent, n_seg, kKappa, kMinSnr));
    // Brand-new spike: Sp = 0 fires iff n_seg + 1 > kappa (1 > kappa/(n_seg+1))
    REQUIRE(is_impulsive_segment(sc, 0.0F, n_seg, kKappa, kMinSnr));
    REQUIRE_FALSE(is_impulsive_segment(sc, 0.0F, 1, kKappa, kMinSnr));
    REQUIRE_FALSE(is_impulsive_segment(sc, 0.0F, 2, kKappa, kMinSnr));
    REQUIRE(is_impulsive_segment(sc, 0.0F, 3, kKappa, kMinSnr));
    // min_snr gate
    REQUIRE_FALSE(is_impulsive_segment(3.0F, 0.0F, n_seg, kKappa, kMinSnr));
}
