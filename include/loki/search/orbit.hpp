#pragma once

/**
 * @file orbit.hpp
 * @brief Search parameter limits from physical limits of circular binary
 * orbits, and the orbit a set of derivative limits implies.
 */

#include <optional>
#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::search {

/**
 * @brief Physical limits of the circular binary orbits a search covers.
 *
 * Orbital periods of at least @p p_orb_min seconds, a companion of at most
 * @p m_c_max and a pulsar of at least @p m_p_min solar masses: together they
 * bound the line-of-sight velocity and every derivative of it.
 */
struct CircularOrbitLimits {
    double p_orb_min{0.0};
    double m_c_max{10.0};
    double m_p_min{1.2};
};

/**
 * @brief Search parameter limits that cover every circular orbit within
 * @p orbit, for intrinsic spin frequencies in [@p f_min, @p f_max].
 *
 * With omega = 2 pi / p_orb_min and K = (G M_sun)^(1/3) m_c_max
 * (m_p_min + m_c_max)^(-2/3), the k-th derivative of the line-of-sight
 * position is at most K omega^(k - 2/3) in magnitude (k = 1 is the velocity
 * v). The limits are, in param_limits order, [+/- snap] (order 4),
 * [+/- jerk] (order >= 3), +/- acceleration, and the frequency range widened
 * by the Doppler factor: [f_min (1 - v/c), f_max (1 + v/c)]. This is pyloki's
 * ParamLimits.from_circular.
 *
 * @param poly_order Highest searched derivative of the frequency's phase
 * model: 2 (acceleration), 3 (jerk) or 4 (snap).
 * @throws std::invalid_argument for a non-positive or non-finite input, an
 * empty frequency range, a poly_order outside [2, 4], or limits whose
 * velocity reaches the speed of light.
 */
[[nodiscard]] std::vector<ParamLimit>
circular_param_limits(double f_min,
                      double f_max,
                      SizeType poly_order,
                      const CircularOrbitLimits& orbit);

/// The tightest circular orbit that reaches both derivative limits.
struct ImpliedCircularOrbit {
    /// Orbital period (s) at which |accel| = a_max and |jerk| = j_max.
    double p_orb{0.0};
    /// Smallest companion mass (solar masses) that reaches a_max there.
    double m_c_min{0.0};
};

/**
 * @brief The circular orbit implied by a parameter box with acceleration and
 * jerk, for a pulsar of @p m_p solar masses.
 *
 * A circular orbit reaches |accel| = a and |jerk| = j at omega = j / a, so a
 * box of +/- a_max and +/- j_max reaches orbits as fast as
 * p_orb = 2 pi a_max / j_max, with a companion of at least m_c_min. Empty for
 * a box without both (fewer than three parameters, or a zero limit).
 */
[[nodiscard]] std::optional<ImpliedCircularOrbit>
implied_circular_orbit(std::span<const ParamLimit> param_limits,
                       double m_p = 1.4);

} // namespace loki::search
