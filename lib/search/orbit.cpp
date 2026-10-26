#include "loki/search/orbit.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "loki/common/types.hpp"

#include "lib/detail/utils.hpp"

namespace loki::search {

namespace {

void require_positive(double value, const char* name) {
    if (!utils::is_finite(value) || value <= 0.0) {
        throw std::invalid_argument(std::format(
            "circular orbit limits: {} must be positive and finite, got {}",
            name, value));
    }
}

/// (G M_sun)^(1/3) m_c (m_p + m_c)^(-2/3): the line-of-sight derivatives of
/// a circular orbit are this times omega^(k - 2/3).
double orbit_scale(double m_c, double m_p) {
    return utils::kGMsunOneThird * m_c / std::cbrt((m_p + m_c) * (m_p + m_c));
}

} // namespace

std::vector<ParamLimit>
circular_param_limits(double f_min,
                      double f_max,
                      SizeType poly_order,
                      const CircularOrbitLimits& orbit) {
    require_positive(f_min, "f_min");
    require_positive(f_max, "f_max");
    require_positive(orbit.p_orb_min, "p_orb_min");
    require_positive(orbit.m_c_max, "m_c_max");
    require_positive(orbit.m_p_min, "m_p_min");
    if (f_max <= f_min) {
        throw std::invalid_argument(std::format(
            "circular orbit limits: f_max ({}) must exceed f_min ({})", f_max,
            f_min));
    }
    if (poly_order < 2 || poly_order > 4) {
        throw std::invalid_argument(std::format(
            "circular orbit limits: poly_order must be 2 (accel), 3 (jerk) "
            "or 4 (snap), got {}",
            poly_order));
    }
    const double omega = 2.0 * std::numbers::pi / orbit.p_orb_min;
    const double k     = orbit_scale(orbit.m_c_max, orbit.m_p_min);
    // |d^n x / dt^n| <= K omega^(n - 2/3): velocity n = 1, acceleration 2.
    const auto deriv = [&](int n) {
        return k * std::pow(omega, static_cast<double>(n) - (2.0 / 3.0));
    };
    const double beta = deriv(1) / utils::kCval;
    if (!(beta < 1.0)) {
        throw std::invalid_argument(std::format(
            "circular orbit limits: p_orb_min = {} s with m_c_max = {} and "
            "m_p_min = {} needs a velocity of {:.3g} c",
            orbit.p_orb_min, orbit.m_c_max, orbit.m_p_min, beta));
    }
    std::vector<ParamLimit> limits;
    limits.reserve(poly_order);
    for (auto n = static_cast<int>(poly_order); n >= 2; --n) {
        const double d = deriv(n);
        limits.push_back({.min = -d, .max = d});
    }
    limits.push_back(
        {.min = f_min * (1.0 - beta), .max = f_max * (1.0 + beta)});
    return limits;
}

std::optional<ImpliedCircularOrbit>
implied_circular_orbit(std::span<const ParamLimit> param_limits, double m_p) {
    require_positive(m_p, "m_p");
    // param_limits is [..., jerk, accel, freq]
    if (param_limits.size() < 3) {
        return std::nullopt;
    }
    const auto max_abs = [](const ParamLimit& lim) {
        return std::max(std::abs(lim.min), std::abs(lim.max));
    };
    const auto n   = param_limits.size();
    const double a = max_abs(param_limits[n - 2]);
    const double j = max_abs(param_limits[n - 3]);
    if (!utils::is_finite(a) || !utils::is_finite(j) || a <= 0.0 || j <= 0.0) {
        return std::nullopt;
    }
    const double omega = j / a;
    // a = K(m_c) omega^(4/3), with K increasing in m_c without bound.
    const double target = a / std::pow(omega, 4.0 / 3.0);
    double lo           = 0.0;
    double hi           = 1.0;
    while (orbit_scale(hi, m_p) < target) {
        hi *= 2.0;
        if (hi > 1.0e12) {
            return std::nullopt;
        }
    }
    for (int i = 0; i < 100; ++i) {
        const double mid                           = std::midpoint(lo, hi);
        (orbit_scale(mid, m_p) < target ? lo : hi) = mid;
    }
    return ImpliedCircularOrbit{.p_orb   = 2.0 * std::numbers::pi / omega,
                                .m_c_min = hi};
}

} // namespace loki::search
