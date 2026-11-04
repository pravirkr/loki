#pragma once

/**
 * @file terminal.hpp
 * @brief Loki's terminal look: the palette and the capability checks shared
 * by the progress bars and the loki CLI.
 */

#include <cstdint>

namespace loki {

/// @brief Loki's terminal palette, as 24-bit RGB values (0xRRGGBB).
enum class TermColor : std::uint32_t {
    kOrange      = 0xFF8C42, ///< titles, spinner
    kAmber       = 0xFFB366, ///< headers, percentages, counts
    kDarkOrange  = 0xE6722A, ///< accents, timings
    kRed         = 0xF92672, ///< bar fill, errors
    kBrightGreen = 0x66BB6A, ///< scores
    kBackground  = 0x3A3A3A, ///< bar remainder
    kMuted       = 0x8A8A8A, ///< secondary text
};

/// @brief The 0xRRGGBB value of @p color.
[[nodiscard]] constexpr std::uint32_t to_rgb(TermColor color) noexcept {
    return static_cast<std::uint32_t>(color);
}

/// @brief Whether stdout is an interactive terminal that takes cursor
/// control and styling: a TTY whose TERM is not "dumb".
[[nodiscard]] bool stdout_is_terminal() noexcept;

/// @brief Whether colour is allowed: NO_COLOR (https://no-color.org) is
/// unset or empty.
[[nodiscard]] bool color_allowed() noexcept;

/// @brief Whether colour is forced on where it would be off (a pipe, a log
/// file): CLICOLOR_FORCE set and not "0". NO_COLOR still wins.
[[nodiscard]] bool color_forced() noexcept;

} // namespace loki
