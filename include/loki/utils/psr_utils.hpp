#pragma once

/**
 * @file psr_utils.hpp
 * @brief Small pulsar-timing helpers (phase binning, Taylor-parameter shift).
 */

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::psr_utils {

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
