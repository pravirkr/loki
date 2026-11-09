#pragma once

/**
 * @file psr_utils.hpp
 * @brief Small pulsar-timing helpers (phase binning, Taylor-parameter shift,
 * the phase model of an EP leaf).
 */

#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::psr_utils {

/// The delay model leaf_phase used for a leaf.
enum class LeafModel : std::uint8_t {
    kTaylor,          ///< the Taylor polynomial of its derivatives
    kCircularSnap,    ///< a circular orbit, omega^2 = -snap / accel
    kCircularCrackle, ///< a circular orbit, omega^2 = -crackle / jerk
};

/**
 * @brief Pulse phase, in cycles, of a reported EP leaf at each of @p times.
 *
 * @p leaf is one leaf as an EP result stores it (a `param_sets` row):
 * n_params + 2 rows of (value, error), the derivatives highest first down to
 * the acceleration (m/s^k, gauge-normalised), then (frequency in Hz, error),
 * (delay in m, error) and (f0, basis). Its parameters refer to @p t_ref, the
 * middle of the searched span for a reported leaf. The phase is
 *
 *     f * ((t - t_ref) - D(t - t_ref) / c),
 *
 * with D the delay of the model the search propagated the leaf with: the
 * Taylor polynomial sum_k p_k dt^k / k!, or, in a five-parameter search, a
 * circular orbit when snap and accel (else crackle and jerk) are significant
 * at @p propagator_significance and imply a real angular frequency, as the
 * search decides. The reported errors carry a term of order dv / c from the
 * gauge step, so a leaf within about that of a significance threshold may be
 * classed differently from the search's last level.
 *
 * @param leaf      (n_params + 2) * 2 values.
 * @param n_params  1 (frequency only) to 5.
 * @param times     Times in seconds, on the same clock as @p t_ref.
 * @param t_ref     The time the leaf's parameters refer to.
 * @param phase     Output, one phase per time.
 * @param propagator_significance  As the search's (EP config, default 2).
 * @return The model used.
 */
LeafModel leaf_phase(std::span<const double> leaf,
                     SizeType n_params,
                     std::span<const double> times,
                     double t_ref,
                     std::span<double> phase,
                     double propagator_significance = 2.0);

/**
 * @brief Phase-bin index of an event: frac((proper_time - delay) * freq) *
 * nbins, in [0, nbins).
 */
[[nodiscard]] float phase_index(double proper_time,
                                double freq,
                                SizeType nbins,
                                double delay = 0.0);

/**
 * @brief Shift kinematic Taylor parameters (ordering [..., j, a, f]) to a
 * reference time @p delta_t later.
 * @return The shifted parameters and the accumulated delay.
 */
[[nodiscard]] std::tuple<std::vector<double>, double>
shift_taylor_params_d_f(std::span<const double> param_vec, double delta_t);

} // namespace loki::psr_utils
