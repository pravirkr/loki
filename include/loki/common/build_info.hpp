#pragma once

/**
 * @file build_info.hpp
 * @brief The loki source a build was made from.
 */

#include <string_view>

namespace loki {

/**
 * @brief The git commit of the source this library was built from.
 *
 * `git describe --always --dirty --abbrev=12` at build time, so "-dirty"
 * marks uncommitted changes, or "unknown" when the source was not a git
 * checkout. Command-line results record it as `/run_stats` `loki_commit`.
 */
[[nodiscard]] std::string_view git_commit() noexcept;

} // namespace loki
