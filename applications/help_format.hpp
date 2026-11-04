#pragma once

// The loki CLI's --help: CLI11's layout, with option groups as titled lists
// at the level of the other options, without CLI11's "Excludes:" notes, and
// in colour on a colour terminal (loki::cli::Console). Styling is applied to
// the finished text, so the columns line up as they do without colour.

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include <CLI/CLI.hpp>

#include "console.hpp"

namespace loki::cli {

class HelpFormatter : public CLI::Formatter {
public:
    std::string make_help(const CLI::App* app,
                          std::string name,
                          CLI::AppFormatMode mode) const override {
        auto text = CLI::Formatter::make_help(app, std::move(name), mode);
        // At most one blank line between blocks.
        for (auto pos = text.find("\n\n\n"); pos != std::string::npos;
             pos      = text.find("\n\n\n", pos)) {
            text.erase(pos, 1);
        }
        return Console::get().color() ? colorize(text) : text;
    }

    std::string make_expanded(const CLI::App* sub,
                              CLI::AppFormatMode mode) const override {
        if (!sub->get_name().empty()) {
            return CLI::Formatter::make_expanded(sub, mode);
        }
        // An option group: "<group>:" and its options, as "OPTIONS:".
        std::string out = "\n" + sub->get_group() + ":\n";
        for (const auto* opt : sub->get_options(
                 [](const CLI::Option* o) { return o->nonpositional(); })) {
            out += make_option(opt, /*is_positional=*/false);
        }
        return out;
    }

    std::string make_option_opts(const CLI::Option* opt) const override {
        auto text = CLI::Formatter::make_option_opts(opt);
        // Conflicts are reported when they happen; the help stays short.
        for (const std::string_view note : {" Needs:", " Excludes:"}) {
            if (const auto pos = text.find(note); pos != std::string::npos) {
                text.erase(pos);
            }
        }
        // A counting flag (-v, -q) shows its default count.
        if (text == " [0] ") {
            text.clear();
        }
        return text;
    }

private:
    [[nodiscard]] std::string colorize(const std::string& text) const {
        const auto& con = Console::get();
        const auto col  = get_column_width();
        std::string out;
        bool first             = true;
        bool in_subcommands    = false;
        std::size_t line_start = 0;
        while (line_start < text.size()) {
            auto end = text.find('\n', line_start);
            end      = end == std::string::npos ? text.size() : end;
            const std::string_view line(text.data() + line_start,
                                        end - line_start);
            line_start = end + 1;

            const auto indent = line.find_first_not_of(' ');
            // NOLINTNEXTLINE(bugprone-branch-clone): blank and plain lines
            if (line.empty() || indent == std::string_view::npos) {
                out += line;
            } else if (first) {
                out += con.title(line); // the description
                first = false;
            } else if (indent == 0 && line.ends_with(':')) {
                in_subcommands = line == "SUBCOMMANDS:";
                out += con.header(line);
            } else if (indent == 0) {
                out += con.muted(line); // usage line and footer
            } else if (line[indent] == '-') {
                // Names and value types in the left column; the description
                // after it as it is.
                const bool split = line.size() > col && line[col - 1] == ' ' &&
                                   line[col] != ' ';
                const auto left  = split ? line.substr(0, col) : line;
                out += style_option_names(left);
                if (split) {
                    out += line.substr(col);
                }
            } else if (in_subcommands && indent == 2) {
                const auto name_end = line.find(' ', indent);
                out += line.substr(0, indent);
                out += con.accent(line.substr(indent, name_end - indent));
                if (name_end != std::string_view::npos) {
                    out += line.substr(name_end);
                }
            } else {
                out += line;
            }
            if (end < text.size()) {
                out += '\n';
            }
        }
        return out;
    }

    // "-o, --outdir TEXT": option names accented, value types muted.
    [[nodiscard]] static std::string style_option_names(std::string_view left) {
        const auto& con = Console::get();
        std::string out;
        std::size_t i = 0;
        while (i < left.size()) {
            if (left[i] == ' ') {
                out += ' ';
                ++i;
                continue;
            }
            auto end   = left.find(' ', i);
            end        = end == std::string_view::npos ? left.size() : end;
            auto token = left.substr(i, end - i);
            i          = end;
            if (!token.starts_with('-')) {
                out += con.muted(token);
                continue;
            }
            const bool comma = token.ends_with(',');
            if (comma) {
                token.remove_suffix(1);
            }
            // "--no-preprocess{false}": the name, then its muted default.
            const auto brace = token.find('{');
            out += con.accent(token.substr(0, brace));
            if (brace != std::string_view::npos) {
                out += con.muted(token.substr(brace));
            }
            if (comma) {
                out += ',';
            }
        }
        return out;
    }
};

} // namespace loki::cli
