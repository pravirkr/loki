#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::simulation::detail {

enum class PulseShapeKind : std::uint8_t { kBoxcar, kGaussian, kVonMises };

[[nodiscard]] PulseShapeKind parse_pulse_shape(std::string_view shape);

/**
 * @brief Wrapped pulse CDF on `j / ngrid` for `j = 0 .. ngrid`.
 *
 * The table is non-decreasing, starts at 0, and ends at 1.
 */
[[nodiscard]] std::vector<float>
build_cdf_lut(PulseShapeKind shape, double width, double pos, SizeType ngrid);

/// Probability mass of the pulse in each `[t, t + dt)` bin.
[[nodiscard]] std::vector<float>
generate_pulse_template(std::span<const double> proper_time,
                        double dt,
                        double period,
                        std::span<const float> cdf_lut);

} // namespace loki::simulation::detail
