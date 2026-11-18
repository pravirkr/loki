#pragma once

// Leaves and checks shared by moving_resolve_t.cpp and
// moving_resolve_cuda_t.cpp: on the moving grid a leaf's f0 stays the seed's,
// so its frequency offset is carried by its velocity. The resolver must pick
// the frequency cell of the added segment from the leaf's absolute velocity
// there, i.e. the cell that holds the time derivative of the phase it returns.

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"

#include "lib/detail/psr_utils.hpp"

namespace moving_resolve {

using loki::ParamLimit;
using loki::SizeType;

constexpr double kC         = 299792458.0;
constexpr double kF0        = 100.0;
constexpr SizeType kNbins   = 64;
constexpr SizeType kNFreq   = 64;
constexpr SizeType kNAccel  = 16;
constexpr double kTInit     = 10.0;
constexpr double kTAdd      = 30.0037;
constexpr double kScale     = 10.0;
constexpr double kEps       = 1e-3; // s, for the phase derivative
constexpr double kSignif    = 2.0;
constexpr ParamLimit kFreq  = {.min = 99.0, .max = 101.0};
constexpr ParamLimit kAccel = {.min = -20.0, .max = 20.0};
constexpr ParamLimit kWide  = {.min = -1.0, .max = 1.0};
// Offsets of the leaf's frequency at t_init, in frequency cells.
constexpr std::array<double, 6> kOffsets = {-0.45, -0.3, -0.15,
                                            0.15,  0.3,  0.45};

inline double cell_velocity() {
    return kC * ((kFreq.max - kFreq.min) / static_cast<double>(kNFreq)) / kF0;
}

inline std::vector<ParamLimit> limits(SizeType n_params) {
    std::vector<ParamLimit> out(n_params, kWide);
    out[n_params - 2] = kAccel;
    out[n_params - 1] = kFreq;
    return out;
}

/// One leaf case: its rows (value, width) and a name.
struct Leaf {
    std::string name;
    std::vector<double> rows;
};

/// Taylor leaves: (highest derivative ... accel, velocity, d0, f0).
inline std::vector<Leaf> taylor_leaves(SizeType n_params) {
    std::vector<Leaf> out;
    for (const double off : kOffsets) {
        std::vector<double> r;
        // snap, jerk (only those the basis has), accel
        const std::array<double, 3> higher = {1e-4, 0.01, 3.0};
        for (SizeType k = 0; k + 1 < n_params; ++k) {
            r.push_back(higher[3 - (n_params - 1) + k]);
            r.push_back(1.0);
        }
        r.push_back(-off * cell_velocity()); // velocity
        r.push_back(1.0);
        r.push_back(0.0); // d0
        r.push_back(0.0);
        r.push_back(kF0);
        r.push_back(0.0);
        out.push_back({"taylor n=" + std::to_string(n_params) + " offset " +
                           std::to_string(off),
                       r});
    }
    return out;
}

/// Chebyshev leaves: (alpha_n ... alpha_1, alpha_0, f0); velocity at the
/// centre is alpha_1 / ts.
inline std::vector<Leaf> chebyshev_leaves(SizeType n_params) {
    std::vector<Leaf> out;
    for (const double off : kOffsets) {
        std::vector<double> r;
        const std::array<double, 3> higher = {1e-2, 0.5,
                                              3.0 * kScale * kScale / 4.0};
        for (SizeType k = 0; k + 1 < n_params; ++k) {
            r.push_back(higher[3 - (n_params - 1) + k]);
            r.push_back(1.0);
        }
        r.push_back(-off * cell_velocity() * kScale); // alpha_1
        r.push_back(1.0);
        r.push_back(0.0); // alpha_0
        r.push_back(0.0);
        r.push_back(kF0);
        r.push_back(0.0);
        out.push_back({"chebyshev n=" + std::to_string(n_params) + " offset " +
                           std::to_string(off),
                       r});
    }
    return out;
}

/// Circular leaves (crackle, snap, jerk, accel, velocity, d0, f0): a Taylor
/// leaf, an orbit resolved by snap and accel and one resolved by crackle and
/// jerk (the hole), each at every offset.
inline std::vector<Leaf> circular_leaves() {
    constexpr double kOmega2 = 1e-4;
    std::vector<Leaf> out;
    for (const double off : kOffsets) {
        const double v = -off * cell_velocity();
        // name, crackle, snap, dsnap, jerk, djerk, accel, daccel
        const std::array<std::array<double, 7>, 3> cls = {{
            {0.0, 0.0, 1.0, 0.01, 1.0, 3.0, 1.0},              // Taylor
            {0.0, -kOmega2 * 3.0, 1e-6, 0.01, 1.0, 3.0, 1e-3}, // snap orbit
            {-kOmega2 * 0.05, 0.0, 1.0, 0.05, 1e-4, 0.0, 1.0}, // crackle orbit
        }};
        const std::array<std::string, 3> names = {"taylor", "snap orbit",
                                                  "crackle orbit"};
        for (SizeType c = 0; c < 3; ++c) {
            const auto& p = cls[c];
            out.push_back(
                {"circular " + names[c] + " offset " + std::to_string(off),
                 {p[0], 1e-6, p[1], p[2], p[3], p[4], p[5], p[6], v, 1.0, 0.0,
                  0.0, kF0, 0.0}});
        }
    }
    return out;
}

/// A resolver run on a batch of leaves: fills (index, phase) for an added
/// segment at t_add, with the leaves referenced at t_cur.
using Resolver = std::function<void(const std::vector<double>& leaves,
                                    SizeType n_leaves,
                                    double t_add,
                                    double t_cur,
                                    std::vector<SizeType>& idx,
                                    std::vector<float>& phase)>;

/// For every leaf: the resolved frequency cell must hold the derivative of
/// the returned phase with respect to t_add. Returns the failing case names.
inline std::vector<std::string> failures(const std::vector<Leaf>& cases,
                                         const Resolver& resolve,
                                         double t_cur) {
    std::vector<double> leaves;
    for (const auto& c : cases) {
        leaves.insert(leaves.end(), c.rows.begin(), c.rows.end());
    }
    const SizeType n = cases.size();
    std::vector<SizeType> idx(n);
    std::vector<SizeType> idx_lo(n);
    std::vector<SizeType> idx_hi(n);
    std::vector<float> ph(n);
    std::vector<float> ph_lo(n);
    std::vector<float> ph_hi(n);
    resolve(leaves, n, kTAdd, t_cur, idx, ph);
    resolve(leaves, n, kTAdd - kEps, t_cur, idx_lo, ph_lo);
    resolve(leaves, n, kTAdd + kEps, t_cur, idx_hi, ph_hi);
    std::vector<std::string> bad;
    const auto nb = static_cast<double>(kNbins);
    for (SizeType i = 0; i < n; ++i) {
        double dph = static_cast<double>(ph_hi[i]) - ph_lo[i];
        dph -= nb * std::floor(dph / nb); // phase advances: unwrap to [0, nb)
        // whole turns over 2 eps
        const double turns = std::round((kF0 * 2.0 * kEps) - (dph / nb));
        const double freq  = ((dph / nb) + turns) / (2.0 * kEps);
        const SizeType want_f =
            loki::psr_utils::get_nearest_idx_analytical(freq, kFreq, kNFreq);
        if (idx[i] % kNFreq != want_f) {
            bad.push_back(cases[i].name + " (cell " +
                          std::to_string(idx[i] % kNFreq) + ", phase says " +
                          std::to_string(want_f) + ")");
        }
    }
    return bad;
}

} // namespace moving_resolve
