#pragma once

/// One table of fields per config type: each row names a key and the member
/// it fills, and the allowed keys, the reader and the writer all come from
/// it. A key is added in one place (plus its line in the documented
/// template, which a test keeps in step).
///
///     field<Cfg>("search", "f_min", [](auto& c) -> auto& { return c.f_min; })
///
/// The member's type picks the TOML type: double and float are numbers (a
/// whole number reads as a real), bool a boolean, integers non-negative
/// integers within the member's range, std::string and paths strings,
/// std::vector<double> / std::vector<SizeType> arrays. A std::optional
/// member is left empty when the key is missing and is not written when
/// empty. Checks that span several keys stay with the config type.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <toml++/toml.hpp> // NOLINT(misc-include-cleaner) -- toml::table, node

#include "loki/common/types.hpp"

#include "lib/detail/toml_utils.hpp"

namespace loki::toml_utils {

template <typename Cfg> struct Field {
    std::string_view table; ///< Top-level table; "" in a sub-table schema
    std::string_view key;
    /// Reads the key (if present) of @p table into the config.
    std::function<void(
        const toml::table& table, const std::string& path, Cfg& cfg)>
        read;
    /// Writes the key into @p table (or nothing, for an unset optional).
    std::function<void(const Cfg& cfg, toml::table& table)> write;
    /// A sub-table field (search.orbit): checks the keys of the sub-table.
    std::function<void(const toml::table& table,
                       const std::string& path,
                       std::vector<std::string>& errors)>
        check_sub_table;
    /// When set, a missing key is an error, with this hint.
    std::optional<std::string_view> required;
};

template <typename Cfg> using Schema = std::vector<Field<Cfg>>;

namespace detail {

template <typename T> struct IsOptional : std::false_type {};
template <typename T> struct IsOptional<std::optional<T>> : std::true_type {};

template <typename T>
[[nodiscard]] std::optional<T> read_as(const toml::table& table,
                                       std::string_view key,
                                       const std::string& path) {
    if constexpr (std::is_same_v<T, bool>) {
        return read_bool(table, key, path);
    } else if constexpr (std::is_same_v<T, double>) {
        return read_number(table, key, path);
    } else if constexpr (std::is_same_v<T, float>) {
        const auto val = read_number(table, key, path);
        return val ? std::optional<float>(static_cast<float>(*val))
                   : std::nullopt;
    } else if constexpr (std::is_same_v<T, std::string>) {
        return read_string(table, key, path);
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        auto val = read_string(table, key, path);
        return val ? std::optional<std::filesystem::path>(std::move(*val))
                   : std::nullopt;
    } else if constexpr (std::is_integral_v<T>) {
        const auto val = read_count(table, key, path);
        if (!val) {
            return std::nullopt;
        }
        constexpr auto kMax =
            static_cast<std::uint64_t>(std::numeric_limits<T>::max());
        if (static_cast<std::uint64_t>(*val) > kMax) {
            throw std::invalid_argument(
                fmt::format("{} must be at most {}", path, kMax));
        }
        return static_cast<T>(*val);
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
        return read_number_array(table, key, path);
    } else if constexpr (std::is_same_v<T, std::vector<SizeType>>) {
        return read_count_array(table, key, path);
    } else {
        static_assert(sizeof(T) == 0, "no TOML reader for this member type");
    }
}

template <typename T>
void write_as(toml::table& table, std::string_view key, const T& value) {
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, double> ||
                  std::is_same_v<T, std::string>) {
        table.insert(key, value);
    } else if constexpr (std::is_same_v<T, float>) {
        // The shortest text of the float (0.1, not 0.10000000149011612):
        // it reads back as the same float.
        table.insert(key, std::stod(fmt::format("{}", value)));
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        table.insert(key, value.string());
    } else if constexpr (std::is_integral_v<T>) {
        table.insert(key, static_cast<std::int64_t>(value));
    } else if constexpr (std::is_same_v<T, std::vector<double>> ||
                         std::is_same_v<T, std::vector<SizeType>>) {
        toml::array arr;
        for (const auto v : value) {
            if constexpr (std::is_same_v<T, std::vector<double>>) {
                arr.push_back(v);
            } else {
                arr.push_back(static_cast<std::int64_t>(v));
            }
        }
        table.insert(key, std::move(arr));
    } else {
        static_assert(sizeof(T) == 0, "no TOML writer for this member type");
    }
}

[[nodiscard]] inline const toml::table* table_at(const toml::table& root,
                                                 std::string_view name) {
    return name.empty() ? &root : root[name].as_table();
}

[[nodiscard]] inline toml::table& table_at(toml::table& root,
                                           std::string_view name) {
    if (name.empty()) {
        return root;
    }
    if (root[name].as_table() == nullptr) {
        root.insert_or_assign(name, toml::table{});
    }
    return *root[name].as_table();
}

[[nodiscard]] inline std::string join_path(std::string_view prefix,
                                           std::string_view table,
                                           std::string_view key) {
    std::string path(prefix);
    for (const auto part : {table, key}) {
        if (!part.empty()) {
            path += path.empty() ? "" : ".";
            path += part;
        }
    }
    return path;
}

} // namespace detail

/// A key that fills the member @p get returns (see the file comment).
template <typename Cfg, typename Get>
[[nodiscard]] Field<Cfg>
field(std::string_view table, std::string_view key, Get get) {
    Field<Cfg> f;
    f.table = table;
    f.key   = key;
    f.read  = [get, key](const toml::table& t, const std::string& path,
                         Cfg& cfg) {
        auto& member = get(cfg);
        using T      = std::remove_cvref_t<decltype(member)>;
        if constexpr (detail::IsOptional<T>::value) {
            if (auto val =
                    detail::read_as<typename T::value_type>(t, key, path)) {
                member = std::move(*val);
            }
        } else {
            if (auto val = detail::read_as<T>(t, key, path)) {
                member = std::move(*val);
            }
        }
    };
    f.write = [get, key](const Cfg& cfg, toml::table& t) {
        const auto& member = get(cfg);
        using T            = std::remove_cvref_t<decltype(member)>;
        if constexpr (detail::IsOptional<T>::value) {
            if (member) {
                detail::write_as(t, key, *member);
            }
        } else {
            detail::write_as(t, key, member);
        }
    };
    return f;
}

/// A string key naming one of @p names, stored as its enum value.
template <typename Cfg, typename Get, typename E, std::size_t N>
[[nodiscard]] Field<Cfg>
choice(std::string_view table,
       std::string_view key,
       Get get,
       std::array<std::pair<std::string_view, E>, N> names) {
    Field<Cfg> f;
    f.table = table;
    f.key   = key;
    f.read  = [get, key, names](const toml::table& t, const std::string& path,
                                Cfg& cfg) {
        const auto val = read_string(t, key, path);
        if (!val) {
            return;
        }
        std::array<std::string_view, N> allowed{};
        std::string listed;
        for (std::size_t i = 0; i < N; ++i) {
            if (names[i].first == *val) {
                get(cfg) = names[i].second;
                return;
            }
            allowed[i] = names[i].first;
            listed +=
                fmt::format("{}\"{}\"", i == 0 ? "" : " or ", names[i].first);
        }
        throw std::invalid_argument(fmt::format("{} must be {} (got \"{}\"){}",
                                                path, listed, *val,
                                                did_you_mean(*val, allowed)));
    };
    f.write = [get, key, names](const Cfg& cfg, toml::table& t) {
        const auto value = get(cfg);
        for (const auto& [name, e] : names) {
            if (e == value) {
                t.insert(key, std::string(name));
                return;
            }
        }
    };
    return f;
}

/// A key with its own reader and writer (arrays of pairs, aliases).
template <typename Cfg>
[[nodiscard]] Field<Cfg>
custom(std::string_view table,
       std::string_view key,
       std::function<void(const toml::table&, const std::string&, Cfg&)> read,
       std::function<void(const Cfg&, toml::table&)> write) {
    Field<Cfg> f;
    f.table = table;
    f.key   = key;
    f.read  = std::move(read);
    f.write = std::move(write);
    return f;
}

/// Marks @p f as required: a missing key is "<path> must be set: <hint>".
template <typename Cfg>
[[nodiscard]] Field<Cfg> required(Field<Cfg> f, std::string_view hint) {
    f.required = hint;
    return f;
}

/// Reads every field of @p schema present in @p root into @p cfg.
template <typename Cfg>
void read_fields(const Schema<Cfg>& schema,
                 const toml::table& root,
                 Cfg& cfg,
                 std::string_view prefix = {}) {
    for (const auto& f : schema) {
        const auto* table = detail::table_at(root, f.table);
        const auto path   = detail::join_path(prefix, f.table, f.key);
        if (table == nullptr || !table->contains(f.key)) {
            if (f.required) {
                throw std::invalid_argument(
                    fmt::format("{} must be set{}{}", path,
                                f.required->empty() ? "" : ": ", *f.required));
            }
            continue;
        }
        f.read(*table, path, cfg);
    }
}

/// The tables of @p schema with every field of @p cfg written.
template <typename Cfg>
[[nodiscard]] toml::table write_fields(const Schema<Cfg>& schema,
                                       const Cfg& cfg) {
    toml::table root;
    for (const auto& f : schema) {
        f.write(cfg, detail::table_at(root, f.table));
    }
    return root;
}

/// A [search.orbit]-style sub-table: @p get returns the std::optional
/// member it fills, read and written with @p sub (fields with table "").
template <typename Cfg, typename Sub, typename Get>
[[nodiscard]] Field<Cfg> sub_table(std::string_view table,
                                   std::string_view key,
                                   Get get,
                                   const Schema<Sub>& sub) {
    Field<Cfg> f;
    f.table = table;
    f.key   = key;
    f.read  = [get, key, &sub](const toml::table& t, const std::string& path,
                               Cfg& cfg) {
        Sub value;
        read_fields(sub, *t[key].as_table(), value, path);
        get(cfg) = std::move(value);
    };
    f.write = [get, key, &sub](const Cfg& cfg, toml::table& t) {
        if (const auto& value = get(cfg)) {
            t.insert(key, write_fields(sub, *value));
        }
    };
    f.check_sub_table = [&sub](const toml::table& t, const std::string& path,
                               std::vector<std::string>& errors) {
        std::vector<std::string_view> allowed;
        for (const auto& s : sub) {
            allowed.push_back(s.key);
        }
        collect_unknown_keys(t, allowed, path, errors);
    };
    return f;
}

/// Checks the shape of @p root against @p schema: unknown tables and keys
/// (with a hint for a removed one), and a table written as a value. Throws
/// "<label>: e1; e2" listing every problem.
template <typename Cfg>
void check_document(const Schema<Cfg>& schema,
                    const toml::table& root,
                    std::string_view label,
                    RemovedKeyHint hint = nullptr) {
    std::vector<std::string_view> tables;
    for (const auto& f : schema) {
        if (std::ranges::find(tables, f.table) == tables.end()) {
            tables.push_back(f.table);
        }
    }
    std::vector<std::string> errors;
    for (const auto& [key, node] : root) {
        const auto name = key.str();
        if (std::ranges::find(tables, name) == tables.end()) {
            const auto text = hint == nullptr ? std::string_view{} : hint(name);
            errors.push_back(text.empty()
                                 ? fmt::format("unknown top-level key '{}'{}",
                                               name, did_you_mean(name, tables))
                                 : fmt::format("{}: {}", name, text));
        } else if (!node.is_table()) {
            errors.push_back(fmt::format("{} must be a table", name));
        }
    }
    for (const auto name : tables) {
        const auto* table = root[name].as_table();
        if (table == nullptr) {
            continue;
        }
        std::vector<std::string_view> allowed;
        for (const auto& f : schema) {
            if (f.table == name) {
                allowed.push_back(f.key);
            }
        }
        collect_unknown_keys(*table, allowed, name, errors, hint);
        for (const auto& f : schema) {
            if (f.table != name || !f.check_sub_table) {
                continue;
            }
            const auto* node = table->get(f.key);
            if (node == nullptr) {
                continue;
            }
            const auto path = detail::join_path({}, name, f.key);
            if (const auto* sub = node->as_table()) {
                f.check_sub_table(*sub, path, errors);
            } else {
                errors.push_back(fmt::format("{} must be a table", path));
            }
        }
    }
    throw_if_errors(errors, label);
}

/// The keys of @p schema as "table.key", without its sub-tables, for the
/// test that keeps the documented templates in step.
template <typename Cfg>
[[nodiscard]] std::vector<std::string> schema_keys(const Schema<Cfg>& schema) {
    std::vector<std::string> keys;
    for (const auto& f : schema) {
        if (!f.check_sub_table) {
            keys.push_back(detail::join_path({}, f.table, f.key));
        }
    }
    return keys;
}

} // namespace loki::toml_utils
