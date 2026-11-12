#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"
#include "loki/utils/psr_utils.hpp"

#include "lib/core/circular.hpp"
#include "lib/core/taylor.hpp"

// leaf_phase must give the phase the search itself gives a leaf: the search's
// ascend resolve on the leaf, against leaf_phase on the same leaf after the
// search's report (gauge) step. The resolve's phase includes the leaf's own
// delay d0, so it is f0 * d0 / c behind leaf_phase.

namespace {

using loki::ParamLimit;
using loki::SizeType;
using loki::psr_utils::LeafModel;

constexpr double kC           = 299792458.0;
constexpr SizeType kNbins     = 64;
constexpr SizeType kGrid      = 16;
constexpr double kSignificant = 2.0;
// float phase indices: a few ulps of 64 bins
constexpr double kTolBins = 1e-3;

double bin_distance(double a, double b) {
    const double d = std::fmod(std::abs(a - b), static_cast<double>(kNbins));
    return std::min(d, static_cast<double>(kNbins) - d);
}

// Compare every leaf at every segment time; return the models leaf_phase used.
std::vector<LeafModel>
check_against_resolve(std::vector<double> leaves,
                      SizeType n_params,
                      std::span<const float> phase_shift,
                      std::span<const std::pair<double, double>> segments,
                      std::pair<double, double> coord_cur) {
    const SizeType stride   = (n_params + 2) * 2;
    const SizeType n_leaves = leaves.size() / stride;
    std::vector<double> f0(n_leaves);
    std::vector<double> d0(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        f0[i] = leaves[(i * stride) + ((n_params + 1) * 2)];
        d0[i] = leaves[(i * stride) + (n_params * 2)];
    }
    loki::core::poly_taylor_report_batch(leaves, coord_cur, n_leaves, n_params);
    std::vector<LeafModel> models;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const std::span<const double> leaf(leaves.data() + (i * stride),
                                           stride);
        for (SizeType s = 0; s < segments.size(); ++s) {
            const std::vector<double> times{segments[s].first};
            std::vector<double> phase(1);
            const auto model = loki::psr_utils::leaf_phase(
                leaf, n_params, times, coord_cur.first, phase, kSignificant);
            if (s == 0) {
                models.push_back(model);
            }
            const double cycles = phase[0] - (f0[i] * d0[i] / kC);
            const double bins =
                (cycles - std::floor(cycles)) * static_cast<double>(kNbins);
            INFO("leaf " << i << " segment " << s << " n_params " << n_params);
            REQUIRE(bin_distance(bins, phase_shift[(s * n_leaves) + i]) <
                    kTolBins);
        }
    }
    return models;
}

std::vector<std::pair<double, double>> segment_coords() {
    // Segment centres from 30 minutes before to 30 minutes after the leaf's
    // reference time.
    std::vector<std::pair<double, double>> segments;
    for (int k = -6; k <= 6; ++k) {
        segments.emplace_back(1800.0 + (300.0 * k), 150.0);
    }
    return segments;
}

} // namespace

TEST_CASE("leaf_phase matches the Taylor ascend resolve", "[leaf_phase]") {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    const auto segments = segment_coords();
    const std::pair<double, double> coord_cur{1800.0, 1800.0};
    for (SizeType n_params = 2; n_params <= 4; ++n_params) {
        const SizeType stride   = (n_params + 2) * 2;
        const SizeType n_leaves = 20;
        // Derivative scales: accel (m/s^2), jerk, snap.
        const std::vector<double> scale{50.0, 5e-3, 1e-6};
        std::vector<double> leaves(n_leaves * stride, 0.0);
        for (SizeType i = 0; i < n_leaves; ++i) {
            double* leaf = leaves.data() + (i * stride);
            for (SizeType k = 0; k + 1 < n_params; ++k) {
                // Highest derivative first.
                leaf[(n_params - 2 - k) * 2]       = scale[k] * unit(rng);
                leaf[((n_params - 2 - k) * 2) + 1] = scale[k];
            }
            leaf[(n_params - 1) * 2]       = 2e4 * unit(rng); // v, m/s
            leaf[((n_params - 1) * 2) + 1] = 1.0;
            leaf[n_params * 2]             = 3e6 * unit(rng); // d0, m
            leaf[(n_params + 1) * 2]       = 300.0 + (250.0 * unit(rng));
        }
        std::vector<ParamLimit> limits(n_params, ParamLimit{-1e9, 1e9});
        std::vector<SizeType> indices(n_leaves * segments.size());
        std::vector<float> phase_shift(n_leaves * segments.size());
        loki::core::poly_taylor_ascend_resolve_batch(
            leaves, indices, phase_shift, limits, segments, coord_cur, kGrid,
            kGrid, kNbins, n_leaves, n_params, segments.size());
        const auto models = check_against_resolve(leaves, n_params, phase_shift,
                                                  segments, coord_cur);
        for (const auto model : models) {
            REQUIRE(model == LeafModel::kTaylor);
        }
    }
}

TEST_CASE("leaf_phase matches the circular ascend resolve", "[leaf_phase]") {
    constexpr SizeType kParams = 5;
    constexpr SizeType kStride = (kParams + 2) * 2;
    const auto segments        = segment_coords();
    const std::pair<double, double> coord_cur{1800.0, 1800.0};
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<double> leaves;
    // Circular orbits of 1-4 hours and 0.1-1 light-second, seen at assorted
    // orbital phases: snap-branch leaves; leaves whose accel and snap are
    // measured as zero: crackle-branch leaves; and leaves whose terms are all
    // insignificant: the Taylor fallback.
    for (int kind = 0; kind < 3; ++kind) {
        for (int n = 0; n < 8; ++n) {
            const double omega =
                2.0 * std::numbers::pi / (3600.0 * (1.0 + (3.0 * unit(rng))));
            const double radius = kC * (0.1 + (0.9 * unit(rng)));
            const double theta  = 2.0 * std::numbers::pi * unit(rng);
            // d^k/dt^k of R sin(theta) at the reference time
            auto deriv = [&](int k) {
                return radius * std::pow(omega, k) *
                       std::sin(theta + (k * std::numbers::pi / 2.0));
            };
            double crackle = deriv(5);
            double snap    = deriv(4);
            double jerk    = deriv(3);
            double accel   = deriv(2);
            double rel_err = 1e-3;
            if (kind == 1) {
                snap  = 0.0;
                accel = 0.0;
                // Keep crackle and jerk of opposite signs: a real omega.
                crackle = std::abs(crackle) * (jerk > 0 ? -1.0 : 1.0);
            }
            if (kind == 2) {
                rel_err = 10.0;
            }
            const double v  = 2e4 * ((2.0 * unit(rng)) - 1.0);
            const double d0 = 3e6 * ((2.0 * unit(rng)) - 1.0);
            const double f0 = 50.0 + (450.0 * unit(rng));
            const std::vector<double> leaf{
                crackle, std::abs(crackle) * rel_err,
                snap,    kind == 1 ? 1.0 : std::abs(snap) * rel_err,
                jerk,    std::abs(jerk) * rel_err,
                accel,   kind == 1 ? 1.0 : std::abs(accel) * rel_err,
                v,       1.0,
                d0,      0.0,
                f0,      0.0};
            leaves.insert(leaves.end(), leaf.begin(), leaf.end());
        }
    }
    const SizeType n_leaves = leaves.size() / kStride;
    std::vector<ParamLimit> limits(kParams, ParamLimit{-1e12, 1e12});
    std::vector<SizeType> indices(n_leaves * segments.size());
    std::vector<float> phase_shift(n_leaves * segments.size());
    loki::core::circ_taylor_ascend_resolve_batch(
        leaves, indices, phase_shift, limits, segments, coord_cur, kGrid, kGrid,
        kNbins, n_leaves, segments.size(), kSignificant);
    const auto models = check_against_resolve(leaves, kParams, phase_shift,
                                              segments, coord_cur);
    for (SizeType i = 0; i < models.size(); ++i) {
        const auto expected = i < 8    ? LeafModel::kCircularSnap
                              : i < 16 ? LeafModel::kCircularCrackle
                                       : LeafModel::kTaylor;
        INFO("leaf " << i);
        REQUIRE(models[i] == expected);
    }
}
