#include "lib/detail/toml_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <toml++/toml.hpp> // NOLINT(misc-include-cleaner) -- toml::table, parse, parse_error

#include "loki/common/types.hpp"

namespace loki::toml_utils {

namespace {

/// True when the float at @p pos of @p text is written `nan` or `inf` (with
/// an optional sign). toml++ positions are 1-based and count code points.
[[nodiscard]] bool is_nonfinite_literal(std::string_view text,
                                        const toml::source_position& pos) {
    std::size_t i = text.starts_with("\xEF\xBB\xBF") ? 3 : 0; // BOM
    for (toml::source_index line = 1; line < pos.line; ++line) {
        i = text.find('\n', i);
        if (i == std::string_view::npos) {
            return false;
        }
        ++i;
    }
    for (toml::source_index col = 1;
         col < pos.column && i < text.size() && text[i] != '\n'; ++col) {
        ++i; // one code point: skip its UTF-8 continuation bytes
        while (i < text.size() &&
               (static_cast<unsigned char>(text[i]) & 0xC0U) == 0x80U) {
            ++i;
        }
    }
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
        ++i;
    }
    const auto token = text.substr(std::min(i, text.size()));
    return token.starts_with("nan") || token.starts_with("inf");
}

/// The value of a node known to be a number. A whole number too large for a
/// double's exact range is still read, rounded.
[[nodiscard]] double number_value(const toml::node& node) {
    if (const auto val = node.value_exact<double>()) {
        return *val;
    }
    return static_cast<double>(*node.value_exact<int64_t>());
}

/// Levenshtein distance between @p a and @p b.
[[nodiscard]] std::size_t edit_distance(std::string_view a,
                                        std::string_view b) {
    std::vector<std::size_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) {
        row[j] = j;
    }
    for (std::size_t i = 1; i <= a.size(); ++i) {
        std::size_t diag = row[0];
        row[0]           = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t up = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1,
                               diag + (a[i - 1] == b[j - 1] ? 0U : 1U)});
            diag   = up;
        }
    }
    return row[b.size()];
}

} // namespace

std::string did_you_mean(std::string_view key,
                         std::span<const std::string_view> allowed) {
    // A typo, not a different word: at most 1 edit for short keys, 2 for
    // longer ones.
    const std::size_t max_distance = key.size() <= 4 ? 1 : 2;
    std::string_view best;
    std::size_t best_distance = max_distance + 1;
    for (const auto candidate : allowed) {
        const auto distance = edit_distance(key, candidate);
        if (distance < best_distance) {
            best          = candidate;
            best_distance = distance;
        }
    }
    return best.empty() ? std::string{}
                        : std::format(" (did you mean '{}'?)", best);
}

toml::table parse(std::string_view text) {
    try {
        return toml::parse(text);
    } catch (const toml::parse_error& err) {
        throw std::runtime_error(std::format(
            "TOML parse error at line {}, col {}: {}", err.source().begin.line,
            err.source().begin.column, err.description()));
    }
}

std::string read_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            std::format("Configuration file not found: '{}'", path.string()));
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(std::format(
            "Could not open configuration file: '{}'", path.string()));
    }
    return {std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()};
}

bool is_allowed_key(std::string_view key,
                    std::span<const std::string_view> allowed) {
    return std::ranges::any_of(
        allowed, [&](std::string_view candidate) { return key == candidate; });
}

void collect_unknown_keys(const toml::table& table,
                          std::span<const std::string_view> allowed,
                          std::string_view table_path,
                          std::vector<std::string>& errors,
                          RemovedKeyHint hint) {
    for (const auto& [key, _] : table) {
        if (!is_allowed_key(key.str(), allowed)) {
            const auto path = std::format("{}.{}", table_path, key.str());
            const auto text = hint == nullptr ? std::string_view{} : hint(path);
            errors.push_back(text.empty()
                                 ? std::format("{}: unknown key{}", path,
                                               did_you_mean(key.str(), allowed))
                                 : std::format("{}: {}", path, text));
        }
    }
}

void throw_if_errors(const std::vector<std::string>& errors,
                     std::string_view label) {
    if (errors.empty()) {
        return;
    }
    std::string message = errors.front();
    for (std::size_t i = 1; i < errors.size(); ++i) {
        message += "; ";
        message += errors[i];
    }
    throw std::invalid_argument(std::format("{}: {}", label, message));
}

void reject_nonfinite_floats(const toml::node& node,
                             std::string_view text,
                             const std::string& path) {
    if (const auto* tbl = node.as_table()) {
        for (const auto& [key, child] : *tbl) {
            reject_nonfinite_floats(
                child, text,
                path.empty() ? std::string(key.str())
                             : std::format("{}.{}", path, key.str()));
        }
    } else if (const auto* arr = node.as_array()) {
        for (std::size_t j = 0; j < arr->size(); ++j) {
            reject_nonfinite_floats((*arr)[j], text,
                                    std::format("{}[{}]", path, j));
        }
    } else if (node.is_floating_point() &&
               is_nonfinite_literal(text, node.source().begin)) {
        throw std::invalid_argument(
            std::format("{} must be finite (got nan or inf)", path));
    }
}

int64_t require_non_negative_int64(int64_t value, std::string_view path) {
    if (value < 0) {
        throw std::invalid_argument(
            std::format("{} must be non-negative (got {})", path, value));
    }
    return value;
}

std::optional<double> read_number(const toml::table& table,
                                  std::string_view key,
                                  std::string_view path) {
    const auto* node = table.get(key);
    if (node == nullptr) {
        return std::nullopt;
    }
    if (!node->is_number()) {
        throw std::invalid_argument(std::format("{} must be a number", path));
    }
    return number_value(*node);
}

std::optional<bool> read_bool(const toml::table& table,
                              std::string_view key,
                              std::string_view path) {
    const auto* node = table.get(key);
    if (node == nullptr) {
        return std::nullopt;
    }
    if (!node->is_boolean()) {
        throw std::invalid_argument(std::format("{} must be a boolean", path));
    }
    return node->value_exact<bool>();
}

std::optional<SizeType> read_count(const toml::table& table,
                                   std::string_view key,
                                   std::string_view path) {
    const auto* node = table.get(key);
    if (node == nullptr) {
        return std::nullopt;
    }
    if (!node->is_integer()) {
        throw std::invalid_argument(std::format("{} must be an integer", path));
    }
    return static_cast<SizeType>(
        require_non_negative_int64(*node->value_exact<int64_t>(), path));
}

std::optional<std::string> read_string(const toml::table& table,
                                       std::string_view key,
                                       std::string_view path) {
    const auto* node = table.get(key);
    if (node == nullptr) {
        return std::nullopt;
    }
    if (!node->is_string()) {
        throw std::invalid_argument(std::format("{} must be a string", path));
    }
    return node->value_exact<std::string>();
}

double array_number(const toml::node& node, std::string_view path) {
    if (!node.is_number()) {
        throw std::invalid_argument(std::format("{} must hold numbers", path));
    }
    return number_value(node);
}

std::optional<std::vector<double>> read_number_array(const toml::table& table,
                                                     std::string_view key,
                                                     std::string_view path) {
    const auto* node = table.get(key);
    if (node == nullptr) {
        return std::nullopt;
    }
    const auto* arr = node->as_array();
    if (arr == nullptr) {
        throw std::invalid_argument(
            std::format("{} must be an array of numbers", path));
    }
    std::vector<double> values;
    values.reserve(arr->size());
    for (const auto& el : *arr) {
        values.push_back(array_number(el, path));
    }
    return values;
}

std::optional<std::vector<SizeType>> read_count_array(const toml::table& table,
                                                      std::string_view key,
                                                      std::string_view path) {
    const auto* node = table.get(key);
    if (node == nullptr) {
        return std::nullopt;
    }
    const auto* arr = node->as_array();
    if (arr == nullptr) {
        throw std::invalid_argument(
            std::format("{} must be an array of integers", path));
    }
    std::vector<SizeType> values;
    values.reserve(arr->size());
    for (const auto& el : *arr) {
        const auto val = el.value_exact<int64_t>();
        if (!val) {
            throw std::invalid_argument(
                std::format("{} must be an array of integers", path));
        }
        values.push_back(
            static_cast<SizeType>(require_non_negative_int64(*val, path)));
    }
    return values;
}

namespace {

/// A TOML basic string: quotes, backslashes and control characters escaped.
[[nodiscard]] std::string toml_quote(std::string_view text) {
    std::string out = "\"";
    for (const char ch : text) {
        const auto byte = static_cast<unsigned char>(ch);
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (byte < 0x20U || byte == 0x7FU) {
            out += std::format("\\u{:04X}", static_cast<unsigned>(byte));
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

/// A value as TOML text. Floats take their shortest exact form, with ".0"
/// on whole numbers so that they read back as floats.
[[nodiscard]] std::string toml_value(const toml::node& node) {
    if (const auto* arr = node.as_array()) {
        std::string out = "[";
        for (std::size_t i = 0; i < arr->size(); ++i) {
            out += i == 0 ? "" : ", ";
            out += toml_value((*arr)[i]);
        }
        return out + "]";
    }
    if (const auto val = node.value_exact<bool>()) {
        return *val ? "true" : "false";
    }
    if (const auto val = node.value_exact<int64_t>()) {
        return std::format("{}", *val);
    }
    if (const auto val = node.value_exact<double>()) {
        auto text = std::format("{}", *val);
        if (text.find_first_of(".en") == std::string::npos) {
            text += ".0";
        }
        return text;
    }
    return toml_quote(*node.value_exact<std::string>());
}

void append_table(std::string& out,
                  std::string_view name,
                  const toml::table& table) {
    out += std::format("{}[{}]\n", out.empty() ? "" : "\n", name);
    for (const auto& [key, value] : table) {
        if (!value.is_table()) {
            out += std::format("{} = {}\n", key.str(), toml_value(value));
        }
    }
    for (const auto& [key, value] : table) {
        if (const auto* sub = value.as_table()) {
            append_table(out, std::format("{}.{}", name, key.str()), *sub);
        }
    }
}

} // namespace

std::string to_toml_text(const toml::table& root,
                         std::span<const std::string_view> table_order) {
    std::string out;
    for (const auto name : table_order) {
        if (const auto* table = root[name].as_table()) {
            append_table(out, name, *table);
        }
    }
    return out;
}

} // namespace loki::toml_utils
