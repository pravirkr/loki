#pragma once

// Terminal presentation of the loki CLI: styled plan tables, the search
// summary and error messages. Colour only on an interactive terminal (or
// with --color always); redirected output, as in batch jobs, stays plain
// text and keeps the log lines it always had. Nothing here runs in a
// compute loop: it prints before and after the search.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/color.h>
#include <fmt/format.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/spdlog.h>

#include "loki/common/terminal.hpp"

namespace loki::cli {

enum class ColorChoice : uint8_t { kAuto, kAlways, kNever };

class Console {
public:
    static Console& get() {
        static Console console;
        return console;
    }

    /// Decides once whether stdout gets colour.
    void configure(ColorChoice choice, bool verbose) {
        m_verbose = verbose;
        switch (choice) {
        case ColorChoice::kAlways:
            m_color = true;
            break;
        case ColorChoice::kNever:
            m_color = false;
            break;
        case ColorChoice::kAuto:
            m_color = stdout_is_terminal() && color_allowed();
            break;
        }
    }

    /// The --log-file sink, which keeps the plain lines a styled terminal
    /// shows differently.
    void set_log_file(spdlog::sink_ptr sink) { m_log_file = std::move(sink); }

    [[nodiscard]] bool color() const { return m_color; }

    // Styles (no-ops without colour, so column alignment never changes).
    [[nodiscard]] std::string title(std::string_view text) const {
        return style(text, fg(TermColor::kOrange) | fmt::emphasis::bold);
    }
    [[nodiscard]] std::string header(std::string_view text) const {
        return style(text, fg(TermColor::kAmber) | fmt::emphasis::bold);
    }
    [[nodiscard]] std::string accent(std::string_view text) const {
        return style(text, fg(TermColor::kDarkOrange));
    }
    [[nodiscard]] std::string muted(std::string_view text) const {
        return style(text, fg(TermColor::kMuted));
    }
    [[nodiscard]] std::string strong(std::string_view text) const {
        return style(text, fmt::emphasis::bold);
    }

    /// Writes @p text to stdout as is.
    static void print(std::string_view text) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fflush(stdout);
    }

    /// A titled block of "key: value" rows. Styled panel on a colour
    /// terminal (and the plain rows in the log file); otherwise the info log
    /// lines of earlier versions.
    void summary(
        std::string_view heading,
        const std::vector<std::pair<std::string, std::string>>& rows) const {
        std::size_t width = 0;
        for (const auto& [key, value] : rows) {
            width = std::max(width, key.size());
        }
        if (!m_color) {
            for (const auto& line : plain_rows(heading, rows, width)) {
                spdlog::info("{}", line);
            }
            return;
        }
        std::string rule;
        for (std::size_t i = 0; i < kRuleWidth; ++i) {
            rule += "─";
        }
        std::string out =
            "\n" + title(fmt::format("── {} ", heading)) + muted(rule) + "\n";
        for (const auto& [key, value] : rows) {
            out += fmt::format("  {} {}\n",
                               muted(fmt::format("{:<{}}", key, width)),
                               strong(value));
        }
        print(out);
        for (const auto& line : plain_rows(heading, rows, width)) {
            to_log_file(line);
        }
    }

    /// The final error report. A colour terminal shows "error: <message>"
    /// (where it was raised only with -v); otherwise, and in the log file,
    /// the full message as an error log line.
    void error(std::string_view what) const {
        if (!m_color) {
            spdlog::error("{}", what);
            return;
        }
        const auto [message, location] = split_error(what);
        std::string out =
            style("error:", fg(TermColor::kRed) | fmt::emphasis::bold) + " " +
            std::string(message) + "\n";
        if (m_verbose && !location.empty()) {
            out += muted(location) + "\n";
        }
        std::fflush(stdout);
        std::fwrite(out.data(), 1, out.size(), stderr);
        std::fflush(stderr);
        to_log_file(std::string(what), spdlog::level::err);
    }

private:
    static constexpr std::size_t kRuleWidth = 40;

    bool m_color{false};
    bool m_verbose{false};
    spdlog::sink_ptr m_log_file;

    static fmt::text_style fg(TermColor color) {
        return fmt::fg(fmt::rgb(to_rgb(color)));
    }

    // The summary as log lines: the terminal without colour and the log file.
    static std::vector<std::string>
    plain_rows(std::string_view heading,
               const std::vector<std::pair<std::string, std::string>>& rows,
               std::size_t width) {
        std::vector<std::string> lines;
        lines.reserve(rows.size() + 1);
        lines.push_back(fmt::format("=== {} ===", heading));
        for (const auto& [key, value] : rows) {
            lines.push_back(fmt::format("{:<{}} : {}", key, width, value));
        }
        return lines;
    }

    [[nodiscard]] std::string style(std::string_view text,
                                    fmt::text_style ts) const {
        if (!m_color) {
            return std::string(text);
        }
        return fmt::format("{}", fmt::styled(text, ts));
    }

    void
    to_log_file(const std::string& line,
                spdlog::level::level_enum level = spdlog::level::info) const {
        if (!m_log_file) {
            return;
        }
        spdlog::logger file_only("loki_cli", m_log_file);
        file_only.set_level(spdlog::level::trace);
        file_only.log(level, "{}", line);
        file_only.flush();
    }

    // Library errors read "file:line:col: error: message\n  in function
    // '...'": the message, and where it was raised ("file:line:col" and the
    // function line) for -v.
    static std::pair<std::string_view, std::string>
    split_error(std::string_view what) {
        constexpr std::string_view kMarker = ": error: ";
        const auto marker                  = what.find(kMarker);
        if (marker == std::string_view::npos) {
            return {what, {}};
        }
        auto message = what.substr(marker + kMarker.size());
        std::string location(what.substr(0, marker));
        if (const auto nl = message.find('\n'); nl != std::string_view::npos) {
            location += "\n" + std::string(message.substr(nl + 1));
            message = message.substr(0, nl);
        }
        return {message, location};
    }
};

} // namespace loki::cli
