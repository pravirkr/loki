#pragma once

/**
 * @file progress.hpp
 * @brief How Loki draws its progress bars.
 *
 * Every long-running call takes a `show_progress` flag; this process-wide
 * mode decides how a shown bar is drawn. `show_progress = false` never
 * writes anything, whatever the mode.
 */

#include <cstdint>
#include <string_view>

namespace loki {

/// @brief Drawing mode of the progress bars.
enum class ProgressMode : std::uint8_t {
    /// Rich when stdout is an interactive terminal, plain otherwise (pipes,
    /// log files, Jupyter kernels). The environment variable LOKI_PROGRESS
    /// ("rich", "plain" or "off") overrides the detection.
    kAuto,
    /// Live bars redrawn in place: colours, spinner, cursor control.
    kRich,
    /// A few self-contained lines per bar (25 %, 50 %, 75 %, done): no
    /// carriage returns, colours or escape sequences.
    kPlain,
    /// No bars.
    kOff,
};

/// @brief Lower-case name: "auto", "rich", "plain" or "off".
[[nodiscard]] std::string_view to_string(ProgressMode mode) noexcept;

/// @brief Inverse of to_string().
/// @throws std::invalid_argument for an unknown name.
[[nodiscard]] ProgressMode parse_progress_mode(std::string_view name);

/// @brief Set the drawing mode for the whole process. Thread-safe; bars
/// already on screen keep the mode they started with.
void set_progress_mode(ProgressMode mode) noexcept;

/// @brief The mode last passed to set_progress_mode() (kAuto by default).
[[nodiscard]] ProgressMode get_progress_mode() noexcept;

} // namespace loki
