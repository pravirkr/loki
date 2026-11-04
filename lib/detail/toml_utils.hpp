#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <toml++/toml.hpp> // NOLINT(misc-include-cleaner) -- toml::table, node, source_position

#include "loki/common/types.hpp"

namespace loki::toml_utils {

/// Shared readers for Loki's TOML configuration files. Every reader rejects
/// unknown keys, non-finite floats and values of the wrong type, and reads
/// whole numbers where a real is expected.

/// Parses @p text, turning a toml++ parse error into a std::runtime_error
/// that names the line and column.
[[nodiscard]] toml::table parse(std::string_view text);

/// Reads the whole file at @p path, or throws if it is missing.
[[nodiscard]] std::string read_file(const std::filesystem::path& path);

/// Reads the config file at @p path and parses it with @p from_string. An
/// error in the file names it: "<path>: <message>", same exception type.
template <typename Fn>
[[nodiscard]] auto load_file(const std::filesystem::path& path,
                             Fn&& from_string) {
    const auto text = read_file(path);
    try {
        return std::forward<Fn>(from_string)(text);
    } catch (const std::invalid_argument& err) {
        throw std::invalid_argument(path.string() + ": " + err.what());
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(path.string() + ": " + err.what());
    }
}

[[nodiscard]] bool is_allowed_key(std::string_view key,
                                  std::span<const std::string_view> allowed);

/// " (did you mean 'x'?)" for the allowed key closest to @p key, when one
/// is within a small edit distance; empty otherwise.
[[nodiscard]] std::string
did_you_mean(std::string_view key, std::span<const std::string_view> allowed);

/// Hint for a key that is no longer in a schema (empty when there is none).
using RemovedKeyHint = std::string_view (*)(std::string_view path);

/// Appends "<table_path>.<key>: unknown key" (or the hint) to @p errors for
/// every key of @p table that is not in @p allowed.
void collect_unknown_keys(const toml::table& table,
                          std::span<const std::string_view> allowed,
                          std::string_view table_path,
                          std::vector<std::string>& errors,
                          RemovedKeyHint hint = nullptr);

/// Throws std::invalid_argument("<label>: e1; e2; ...") when @p errors is
/// not empty.
void throw_if_errors(const std::vector<std::string>& errors,
                     std::string_view label);

/// No config value may be non-finite. TOML allows the float literals `nan`
/// and `inf`, and toml++ builds them from quiet_NaN() and infinity(); under
/// -ffast-math (Release) those are poison, so the parsed value is garbage and
/// no check on it can catch them. They are rejected by their source text.
void reject_nonfinite_floats(const toml::node& node,
                             std::string_view text,
                             const std::string& path);

[[nodiscard]] int64_t require_non_negative_int64(int64_t value,
                                                 std::string_view path);

/// A real; an integer is read as a real.
[[nodiscard]] std::optional<double> read_number(const toml::table& table,
                                                std::string_view key,
                                                std::string_view path);

[[nodiscard]] std::optional<bool> read_bool(const toml::table& table,
                                            std::string_view key,
                                            std::string_view path);

/// A non-negative integer.
[[nodiscard]] std::optional<SizeType> read_count(const toml::table& table,
                                                 std::string_view key,
                                                 std::string_view path);

[[nodiscard]] std::optional<std::string> read_string(const toml::table& table,
                                                     std::string_view key,
                                                     std::string_view path);

/// An array element that must be a number.
[[nodiscard]] double array_number(const toml::node& node,
                                  std::string_view path);

/// An array of numbers.
[[nodiscard]] std::optional<std::vector<double>> read_number_array(
    const toml::table& table, std::string_view key, std::string_view path);

/// An array of non-negative integers.
[[nodiscard]] std::optional<std::vector<SizeType>> read_count_array(
    const toml::table& table, std::string_view key, std::string_view path);

/// @p root as TOML text: the tables named in @p table_order, in that order,
/// each followed by its sub-tables. Floats take their shortest exact form.
[[nodiscard]] std::string
to_toml_text(const toml::table& root,
             std::span<const std::string_view> table_order);

} // namespace loki::toml_utils
