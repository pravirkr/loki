#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/color.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <sys/ioctl.h>
#include <unistd.h>

#include "loki/common/types.hpp"

namespace loki::progress {
namespace details {

inline int get_terminal_width() noexcept {
    struct winsize w{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) { // NOLINT
        return w.ws_col;
    }
    return 80;
}

template <typename Rep, typename Period>
inline std::ostream& write_duration(std::ostream& os,
                                    std::chrono::duration<Rep, Period> dur) {
    auto ns        = std::chrono::duration_cast<std::chrono::nanoseconds>(dur);
    auto secs      = std::chrono::floor<std::chrono::seconds>(ns);
    const auto hms = std::chrono::hh_mm_ss{secs};
    const auto old_fill = os.fill('0');

    auto days = std::chrono::duration_cast<
        std::chrono::duration<int, std::ratio<86400>>>(secs);
    if (days.count() > 0) {
        os << std::setw(2) << days.count() << "d:";
    }
    if (hms.hours().count() > 0) {
        os << std::setw(2) << hms.hours().count() << ':';
    }
    os << std::setw(2) << hms.minutes().count() << ':' << std::setw(2)
       << hms.seconds().count() << 's';
    os.fill(old_fill);
    return os;
}

inline void show_console_cursor(bool const show) {
    std::fputs(show ? "\033[?25h" : "\033[?25l", stdout);
}

inline void erase_line() { std::fputs("\r\033[K", stdout); }

inline std::string repeat_unicode(const std::string_view s, SizeType n) {
    std::string result;
    result.reserve(s.size() * n);
    for (SizeType i = 0; i < n; ++i) {
        result += s;
    }
    return result;
}
} // namespace details

struct Style {
    fmt::text_style value;
};

// Base class for all progress bar columns
class ProgressBar;

class Column {
public:
    Column()                                           = default;
    virtual ~Column()                                  = default;
    Column(const Column&)                              = delete;
    Column& operator=(const Column&)                   = delete;
    Column(Column&&)                                   = delete;
    Column& operator=(Column&&)                        = delete;
    virtual std::string render(const ProgressBar& bar) = 0;
};

class TextColumn : public Column {
public:
    explicit TextColumn(std::string_view text, Style style = {})
        : m_text(text),
          m_style(style) {}

    std::string render(const ProgressBar& /*bar*/) override {
        return fmt::format("{}", fmt::styled(m_text, m_style.value));
    }

private:
    std::string_view m_text;
    Style m_style;
};

class SpinnerColumn : public Column {
public:
    explicit SpinnerColumn(
        Style style = {.value = fmt::fg(fmt::terminal_color::bright_green)})
        : m_style(style) {}
    std::string render(const ProgressBar& bar) override;

private:
    Style m_style;
    static constexpr std::array<std::string_view, 10> kFrames = {
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
};

class LineBarColumn : public Column {
public:
    explicit LineBarColumn(
        int width              = 40,
        Style bar_style        = {.value = fmt::fg(fmt::color{0xF92672})},
        Style background_style = {.value = fmt::fg(fmt::color{0x3A3A3A})})
        : m_width(width),
          m_bar_style(bar_style),
          m_background_style(background_style) {}

    std::string render(const ProgressBar& bar) override;

private:
    std::string render_pulse(const ProgressBar& bar) const;
    int m_width;
    Style m_bar_style;
    Style m_background_style;
    static constexpr std::string_view kBarChar = "\u2501";
};

class PercentageColumn : public Column {
public:
    std::string render(const ProgressBar& bar) override;
};

class TimeStatsColumn : public Column {
public:
    explicit TimeStatsColumn(
        Style style = {.value = fmt::fg(fmt::terminal_color::bright_yellow)})
        : m_style(style) {}
    std::string render(const ProgressBar& bar) override;

private:
    Style m_style;
};

class ScoreColumn : public Column {
public:
    explicit ScoreColumn(
        Style style = {.value = fmt::fg(fmt::terminal_color::bright_red)})
        : m_style(style) {}
    std::string render(const ProgressBar& bar) override;

private:
    Style m_style;
};

class LeavesColumn : public Column {
public:
    explicit LeavesColumn(
        Style style = {.value = fmt::fg(fmt::terminal_color::bright_green)})
        : m_style(style) {}
    std::string render(const ProgressBar& bar) override;

private:
    Style m_style;
};

class ProgressBar {
public:
    explicit ProgressBar(std::string_view prefix,
                         SizeType max_progress = 100,
                         int bar_width         = 40,
                         bool transient        = true,
                         bool is_managed       = false);

    ~ProgressBar() {
        if (!m_is_managed) {
            mark_as_completed();
        }
    }
    ProgressBar(const ProgressBar&)            = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&&)                 = delete;
    ProgressBar& operator=(ProgressBar&&)      = delete;

    ProgressBar& add_column(std::unique_ptr<Column> col) {
        m_columns.push_back(std::move(col));
        return *this;
    }

    ProgressBar& add_score_column() {
        add_column(std::make_unique<TextColumn>(kSeparator));
        add_column(std::make_unique<ScoreColumn>());
        return *this;
    }

    ProgressBar& add_leaves_column() {
        add_column(std::make_unique<TextColumn>(kSeparator));
        add_column(std::make_unique<LeavesColumn>());
        return *this;
    }

    void set_progress(SizeType new_progress) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_start_time_saved) {
                m_start_time       = std::chrono::steady_clock::now();
                m_start_time_saved = true;
            }
            m_progress = new_progress;
            if (m_progress >= m_max_progress && m_max_progress > 0) {
                m_completed = true;
            }
        }
        if (!m_is_managed) {
            print_progress();
        }
    }

    void set_score(double s) { m_score = s; }
    void set_leaves(double l) { m_leaves = l; }

    void tick() { set_progress(m_progress + 1); }
    bool is_completed() const { return m_completed.load(); }

    void mark_as_completed() {
        if (!is_completed()) {
            m_completed = true;
            if (!m_is_managed) {
                print_progress();
            }
        }
    }

    SizeType get_progress() const { return m_progress.load(); }
    SizeType get_max_progress() const { return m_max_progress; }
    double get_score() const { return m_score.load(); }
    double get_leaves() const { return m_leaves.load(); }

    std::chrono::nanoseconds get_elapsed() const {
        if (is_completed()) {
            return m_elapsed;
        }
        return std::chrono::steady_clock::now() - m_start_time;
    }

    // Public method to render the bar state to a string
    std::string to_string() const {
        std::stringstream line_ss;
        for (const auto& column : m_columns) {
            line_ss << column->render(*this);
        }
        return line_ss.str();
    }

    void print_progress() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!is_completed()) {
            m_elapsed = std::chrono::steady_clock::now() - m_start_time;
        }

        std::ostream& os = std::cerr;
        details::erase_line();

        // Transient behavior: only print if not completed, or if not transient
        if (!is_completed()) {
            os << to_string();
        } else if (!m_transient) {
            os << to_string() << '\n';
        }
        os.flush();
    }

private:
    std::vector<std::unique_ptr<Column>> m_columns;
    SizeType m_max_progress;
    std::atomic<SizeType> m_progress{0};
    std::atomic<bool> m_completed{false};
    std::atomic<double> m_score{0.0};
    std::atomic<double> m_leaves{0.0};
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::nanoseconds m_elapsed{};
    bool m_start_time_saved{false};
    bool m_transient;
    bool m_is_managed;
    std::mutex m_mutex;

    static constexpr std::string_view kSeparator = " \u2022 ";
};

// --- Column Implementations ---

inline std::string PercentageColumn::render(const ProgressBar& bar) {
    if (bar.get_max_progress() == 0) {
        return "";
    }
    const auto percentage = (static_cast<double>(bar.get_progress()) /
                             static_cast<double>(bar.get_max_progress())) *
                            100.0;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << percentage << "%";
    return ss.str();
}

inline std::string TimeStatsColumn::render(const ProgressBar& bar) {
    std::stringstream ss;
    ss << "[";
    details::write_duration(ss, bar.get_elapsed());
    ss << "<";

    if (bar.get_max_progress() == 0) {
        ss << "??:??s";
    } else if (bar.get_progress() == 0) {
        ss << "--:--s";
    } else {
        auto eta = std::chrono::duration_cast<std::chrono::nanoseconds>(
            bar.get_elapsed() * (static_cast<double>(bar.get_max_progress()) /
                                 static_cast<double>(bar.get_progress())));
        auto remaining = (eta > bar.get_elapsed())
                             ? (eta - bar.get_elapsed())
                             : std::chrono::nanoseconds(0);
        details::write_duration(ss, remaining);
    }
    ss << "]";
    return fmt::format("{}", fmt::styled(ss.str(), m_style.value));
}

inline std::string SpinnerColumn::render(const ProgressBar& bar) {
    if (bar.is_completed()) {
        return fmt::format(
            "{}", fmt::styled("✔", fmt::fg(fmt::terminal_color::bright_green)));
    }
    constexpr double kIntervalMs = 80.0;
    auto frame_no =
        (static_cast<double>(bar.get_elapsed().count()) / 1e6) / kIntervalMs;
    return fmt::format(
        "{}",
        fmt::styled(kFrames[static_cast<SizeType>(frame_no) % kFrames.size()],
                    m_style.value));
}

inline std::string LineBarColumn::render_pulse(const ProgressBar& bar) const {
    const double current_time =
        std::chrono::duration<double>(bar.get_elapsed()).count();
    const int pulse_position =
        static_cast<int>(std::fmod(current_time * 1.5, 1.0) * (m_width * 1.5));
    const int pulse_width = m_width / 4;

    std::string pulse_bar(m_width, ' ');
    for (int i = 0; i < m_width; ++i) {
        pulse_bar[i] = kBarChar[0];
    }
    std::string p1 = details::repeat_unicode(kBarChar, m_width);

    std::string result;
    result += fmt::format("{}", fmt::styled(p1.substr(0, pulse_position),
                                            m_background_style.value));
    result +=
        fmt::format("{}", fmt::styled(p1.substr(pulse_position, pulse_width),
                                      m_bar_style.value));
    result +=
        fmt::format("{}", fmt::styled(p1.substr(pulse_position + pulse_width),
                                      m_background_style.value));
    return result;
}

inline std::string LineBarColumn::render(const ProgressBar& bar) {
    if (bar.get_max_progress() == 0) { // Indeterminate
        return render_pulse(bar);
    }

    double progress_fraction = static_cast<double>(bar.get_progress()) /
                               static_cast<double>(bar.get_max_progress());
    progress_fraction = std::clamp(progress_fraction, 0.0, 1.0);

    const int completed_width = static_cast<int>(m_width * progress_fraction);
    std::string part1 = details::repeat_unicode(kBarChar, completed_width);
    std::string part2 =
        details::repeat_unicode(kBarChar, m_width - completed_width);

    return fmt::format("{}{}", fmt::styled(part1, m_bar_style.value),
                       fmt::styled(part2, m_background_style.value));
}

inline std::string ScoreColumn::render(const ProgressBar& bar) {
    return fmt::format(
        "{}", fmt::styled(fmt::format("Score: {:.2f}", bar.get_score()),
                          m_style.value));
}

inline std::string LeavesColumn::render(const ProgressBar& bar) {
    return fmt::format(
        "{}", fmt::styled(fmt::format("Leaves: {:.2f}", bar.get_leaves()),
                          m_style.value));
}

inline ProgressBar::ProgressBar(std::string_view prefix,
                                SizeType max_progress,
                                int bar_width,
                                bool transient,
                                bool managed)
    : m_max_progress(max_progress),
      m_transient(transient),
      m_is_managed(managed) {
    add_column(std::make_unique<TextColumn>(
        prefix, Style{.value = fmt::fg(fmt::terminal_color::bright_magenta) |
                               fmt::emphasis::bold}));
    add_column(std::make_unique<TextColumn>(" "));
    add_column(std::make_unique<SpinnerColumn>());
    add_column(std::make_unique<TextColumn>(" "));
    add_column(std::make_unique<LineBarColumn>(bar_width));
    add_column(std::make_unique<TextColumn>(" "));
    add_column(std::make_unique<PercentageColumn>());
    add_column(std::make_unique<TextColumn>(kSeparator));
    add_column(std::make_unique<TimeStatsColumn>());
}

inline ProgressBar make_standard_bar(std::string_view prefix,
                                     SizeType max_progress,
                                     bool transient = true) {
    return ProgressBar(prefix, max_progress, 40, transient, false);
}

inline std::unique_ptr<ProgressBar> make_ffa_bar(std::string_view prefix,
                                                 SizeType max_progress,
                                                 bool transient = true) {
    auto bar = std::make_unique<ProgressBar>(prefix, max_progress, 40,
                                             transient, false);
    bar->add_leaves_column();
    return bar;
}

inline std::unique_ptr<ProgressBar> make_pruning_bar(std::string_view prefix,
                                                     SizeType max_progress,
                                                     bool transient = true,
                                                     bool managed   = false) {
    auto bar = std::make_unique<ProgressBar>(prefix, max_progress, 40,
                                             transient, managed);
    bar->add_score_column();
    bar->add_leaves_column();
    return bar;
}

struct ProgressState {
    std::unique_ptr<ProgressBar> bar;
    std::atomic<bool> visible;
    std::atomic<bool> active;

    ProgressState() : visible(true), active(false) {}
    ProgressState(ProgressState&& other) noexcept
        : bar(std::move(other.bar)),
          visible(other.visible.load()),
          active(other.active.load()) {}
    ProgressState& operator=(ProgressState&& other) noexcept {
        if (this != &other) {
            bar = std::move(other.bar);
            visible.store(other.visible.load());
            active.store(other.active.load());
        }
        return *this;
    }
    ProgressState(const ProgressState&)            = delete;
    ProgressState& operator=(const ProgressState&) = delete;
    ~ProgressState()                               = default;
};

class MultiprocessProgressTracker {
public:
    explicit MultiprocessProgressTracker(
        std::string overall_description = "Working...")
        : m_overall_description(std::move(overall_description)),
          m_running(false),
          m_permanently_stopped(false) {
        auto stderr_sink =
            std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("logger", stderr_sink);
        spdlog::set_default_logger(logger);
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    }

    MultiprocessProgressTracker(const MultiprocessProgressTracker&) = delete;
    MultiprocessProgressTracker&
    operator=(const MultiprocessProgressTracker&)              = delete;
    MultiprocessProgressTracker(MultiprocessProgressTracker&&) = delete;
    MultiprocessProgressTracker&
    operator=(MultiprocessProgressTracker&&) = delete;

    ~MultiprocessProgressTracker() { stop(); }

    void start() {
        std::lock_guard<std::mutex> lock(m_control_mutex);
        if (!m_running.load() && !m_permanently_stopped.load()) {
            m_running.store(true);
            m_render_thread =
                std::thread(&MultiprocessProgressTracker::render_loop, this);
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(m_control_mutex);
            if (!m_running.load()) {
                return;
            }
            m_running.store(false);
        }
        if (m_render_thread.joinable()) {
            m_render_thread.join();
        }
        // Final cleanup with robust clear-screen command
        if (m_last_rendered_lines > 0) {
            std::cerr << "\r\033[" << m_last_rendered_lines - 1 << "A"
                      << "\033[J";
        }
        m_permanently_stopped.store(true);
    }

    int add_task(std::string_view description, SizeType total) {
        std::lock_guard<std::mutex> lock(m_tasks_mutex);
        int task_id = m_next_task_id++;
        auto bar    = make_pruning_bar(description, total, /*transient=*/true,
                                       /*managed=*/true);
        m_tasks[task_id].bar = std::move(bar);
        return task_id;
    }

    void update_task(int task_id,
                     SizeType completed,
                     double score        = -1.0,
                     double leaves       = -1.0,
                     bool force_complete = false) {
        std::lock_guard<std::mutex> lock(m_tasks_mutex);
        auto it = m_tasks.find(task_id);
        if (it != m_tasks.end()) {
            if (!it->second.active.load()) {
                it->second.active.store(true);
            }
            if (completed != static_cast<SizeType>(-1)) {
                it->second.bar->set_progress(completed);
            }
            if (score >= 0) {
                it->second.bar->set_score(score);
            }
            if (leaves >= 0) {
                it->second.bar->set_leaves(leaves);
            }
            if (force_complete) {
                it->second.bar->mark_as_completed();
            }
            if (it->second.bar->is_completed()) {
                it->second.visible.store(false);
            }
        }
    }

    template <typename... Args>
    void log(spdlog::level::level_enum level,
             fmt::format_string<Args...> fmt,
             Args&&... args) {
        // Pause rendering to safely write log message
        std::lock_guard<std::mutex> lock(m_control_mutex);
        bool was_running = m_running.exchange(false);
        if (was_running && m_render_thread.joinable()) {
            m_render_thread.join();
        }

        // Erase all progress bars temporarily using robust clear-screen
        if (m_last_rendered_lines > 0) {
            std::cerr << "\r\033[" << m_last_rendered_lines - 1 << "A"
                      << "\033[J";
            m_last_rendered_lines = 0;
        }

        // Print the log message
        spdlog::default_logger()->log(level, fmt, std::forward<Args>(args)...);
        std::cerr << std::flush;

        // Resume rendering
        if (was_running && !m_permanently_stopped.load()) {
            m_running.store(true);
            m_render_thread =
                std::thread(&MultiprocessProgressTracker::render_loop, this);
        }
    }

private:
    std::string m_overall_description;
    std::map<int, ProgressState> m_tasks;
    std::atomic<bool> m_running;
    std::thread m_render_thread;
    std::mutex m_tasks_mutex;
    std::mutex m_control_mutex;
    std::atomic<bool> m_permanently_stopped;
    int m_next_task_id{};
    int m_last_rendered_lines{};

    void render_loop() {
        while (m_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            std::vector<std::string> lines_to_render;
            {
                std::lock_guard<std::mutex> lock(m_tasks_mutex);
                std::vector<int> active_and_visible_task_ids;
                for (auto const& [id, task] : m_tasks) {
                    if (task.visible.load() && task.active.load()) {
                        active_and_visible_task_ids.push_back(id);
                    }
                }
                std::ranges::sort(active_and_visible_task_ids);

                for (int id : active_and_visible_task_ids) {
                    lines_to_render.push_back(m_tasks.at(id).bar->to_string());
                }
            }

            std::ostream& os = std::cerr;

            // Move cursor to start of the block, if it exists
            if (m_last_rendered_lines > 0) {
                os << "\r\033[" << m_last_rendered_lines - 1 << "A";
            }

            // Render the new block of lines
            for (size_t i = 0; i < lines_to_render.size(); ++i) {
                os << "\033[K" << lines_to_render[i]
                   << (i == lines_to_render.size() - 1 ? "" : "\n");
            }

            // If the new block is smaller, clear the old lines below it
            if (lines_to_render.size() < m_last_rendered_lines) {
                os << "\033[J";
            }

            m_last_rendered_lines = static_cast<int>(
                lines_to_render.size() > 0 ? lines_to_render.size() : 0);
            os.flush();
        }
    }
};

class ProgressGuard {
    bool m_show;
    // This class is used to hide the console cursor during progress bar updates
    // and restore it after the progress bar is done.
public:
    explicit ProgressGuard(bool show) : m_show(show) {
        if (m_show) {
            details::show_console_cursor(false);
        }
    }
    ~ProgressGuard() {
        if (m_show) {
            details::show_console_cursor(true);
        }
    }
    ProgressGuard(const ProgressGuard&)            = delete;
    ProgressGuard& operator=(const ProgressGuard&) = delete;
    ProgressGuard(ProgressGuard&&)                 = delete;
    ProgressGuard& operator=(ProgressGuard&&)      = delete;
};

class ProgressTracker {
public:
    ProgressTracker(std::string_view description,
                    SizeType max_progress,
                    MultiprocessProgressTracker* tracker,
                    int task_id = -1);

    ~ProgressTracker();

    ProgressTracker(const ProgressTracker&)            = delete;
    ProgressTracker& operator=(const ProgressTracker&) = delete;
    ProgressTracker(ProgressTracker&&)                 = delete;
    ProgressTracker& operator=(ProgressTracker&&)      = delete;

    void set_progress(SizeType value);
    void set_score(double value);
    void set_leaves(double value);
    void mark_as_completed();

    template <typename... Args>
    void log(spdlog::level::level_enum level,
             fmt::format_string<Args...> fmt,
             Args&&... args);

private:
    std::unique_ptr<ProgressBar> m_owned_bar;
    MultiprocessProgressTracker* m_tracker = nullptr;
    int m_task_id                          = -1;
    // To batch updates for the tracker
    double m_current_score  = -1.0;
    double m_current_leaves = -1.0;
};

// Implementations for ProgressTracker
inline ProgressTracker::ProgressTracker(std::string_view description,
                                        SizeType max_progress,
                                        MultiprocessProgressTracker* tracker,
                                        int task_id)
    : m_tracker(tracker),
      m_task_id(task_id) {
    if (m_tracker == nullptr) {
        // Standalone mode: create and own a non-transient bar
        m_owned_bar =
            make_pruning_bar(description, max_progress, /*transient=*/false);
    }
}

inline ProgressTracker::~ProgressTracker() {
    // RAII cleanup: ensure the bar is marked as finished when the scope ends
    mark_as_completed();
}

inline void ProgressTracker::set_score(double value) {
    m_current_score = value;
    if (m_owned_bar) {
        m_owned_bar->set_score(value);
    }
}

inline void ProgressTracker::set_leaves(double value) {
    m_current_leaves = value;
    if (m_owned_bar) {
        m_owned_bar->set_leaves(value);
    }
}

inline void ProgressTracker::set_progress(SizeType value) {
    if (m_tracker != nullptr) {
        m_tracker->update_task(m_task_id, value, m_current_score,
                               m_current_leaves);
        // Reset batched values
        m_current_score  = -1.0;
        m_current_leaves = -1.0;
    } else {
        m_owned_bar->set_progress(value);
    }
}

inline void ProgressTracker::mark_as_completed() {
    if (m_tracker != nullptr) {
        if (m_task_id != -1) { // Prevent double-completion
            // Final update to mark as complete
            m_tracker->update_task(m_task_id, -1, -1.0, -1.0, true);
            m_task_id = -1; // Mark as done
        }
    } else if (m_owned_bar) {
        m_owned_bar->mark_as_completed();
    }
}

template <typename... Args>
inline void ProgressTracker::log(spdlog::level::level_enum level,
                                 fmt::format_string<Args...> fmt,
                                 Args&&... args) {
    if (m_tracker != nullptr) {
        m_tracker->log(level, fmt, std::forward<Args>(args)...);
    } else {
        spdlog::default_logger()->log(level, fmt, std::forward<Args>(args)...);
    }
}

} // namespace loki::progress