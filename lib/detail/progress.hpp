#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <queue>

#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/color.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"

namespace loki::progress {

namespace tva_colors {
constexpr auto kOrange      = fmt::color{0xFF8C42};
constexpr auto kAmber       = fmt::color{0xFFB366};
constexpr auto kDarkOrange  = fmt::color{0xE6722A};
constexpr auto kBrightGreen = fmt::color{0x66BB6A};
constexpr auto kBackground  = fmt::color{0x3A3A3A};
constexpr auto kRed         = fmt::color{0xF92672};
} // namespace tva_colors

struct Style {
    fmt::text_style value;
};

// Base class for all progress bar columns
class ProgressBar;

class Column {
public:
    Column() = default;
    explicit Column(Style style) : m_style(style) {}
    virtual ~Column()                                  = default;
    Column(const Column&)                              = delete;
    Column& operator=(const Column&)                   = delete;
    Column(Column&&)                                   = delete;
    Column& operator=(Column&&)                        = delete;
    virtual std::string render(const ProgressBar& bar) = 0;

protected:
    Style m_style{};
};

class TextColumn : public Column {
public:
    explicit TextColumn(std::string_view text, Style style = {})
        : Column(style),
          m_text(text) {}
    std::string render(const ProgressBar& /*bar*/) override;

private:
    std::string m_text;
};

class SpinnerColumn : public Column {
public:
    explicit SpinnerColumn(
        Style style         = {.value = fmt::fg(tva_colors::kOrange)},
        bool use_tva_frames = false)
        : Column(style),
          m_use_tva_frames(use_tva_frames) {}
    std::string render(const ProgressBar& bar) override;

private:
    bool m_use_tva_frames;
    static constexpr std::array<std::string_view, 10> kFrames = {
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    // TVA elevator symbols (ASCII approximations)
    static constexpr std::array<std::string_view, 6> kTVAFrames = {
        "o", // Null/empty set
        "+", // Cross/intersection
        "=", // Identity/equivalence
        ":", // Observer/monitoring
        "*", // Variant/anomaly
        "-"  // Timeline/linear
    };
};

class BarColumn : public Column {
public:
    explicit BarColumn(
        int width              = 40,
        Style style            = {.value = fmt::fg(tva_colors::kRed)},
        Style background_style = {.value = fmt::fg(tva_colors::kBackground)})
        : Column(style),
          m_width(width),
          m_background_style(background_style) {}

    std::string render(const ProgressBar& bar) override;

private:
    std::string render_pulse(const ProgressBar& bar) const;
    int m_width;
    Style m_background_style;
    static constexpr std::string_view kBarChar = "\u2501";
};

class PercentageColumn : public Column {
public:
    explicit PercentageColumn(
        Style style = {.value = fmt::fg(tva_colors::kAmber)})
        : Column(style) {}
    std::string render(const ProgressBar& bar) override;
};

class TimeStatsColumn : public Column {
public:
    explicit TimeStatsColumn(
        Style style = {.value = fmt::fg(tva_colors::kDarkOrange)})
        : Column(style) {}
    std::string render(const ProgressBar& bar) override;
};

class ScoreColumn : public Column {
public:
    explicit ScoreColumn(
        Style style = {.value = fmt::fg(fmt::terminal_color::bright_red)})
        : Column(style) {}
    std::string render(const ProgressBar& bar) override;
};

class LeavesColumn : public Column {
public:
    explicit LeavesColumn(
        Style style = {.value = fmt::fg(tva_colors::kBrightGreen)})
        : Column(style) {}
    std::string render(const ProgressBar& bar) override;
};

class ProgressBar {
public:
    explicit ProgressBar(std::string_view prefix,
                         SizeType max_progress = 100,
                         int bar_width         = 40,
                         bool transient        = true,
                         bool is_managed       = false);

    ~ProgressBar();

    ProgressBar(const ProgressBar&)            = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&&)                 = delete;
    ProgressBar& operator=(ProgressBar&&)      = delete;

    ProgressBar& add_column(std::unique_ptr<Column> col);
    ProgressBar& add_score_column();
    ProgressBar& add_leaves_column();

    void set_progress(SizeType new_progress);
    void set_score(double s);
    void set_leaves(double l);
    void tick();
    bool is_completed() const;
    bool is_transient() const;
    void mark_as_completed();

    SizeType get_progress() const;
    SizeType get_max_progress() const;
    double get_score() const;
    double get_leaves() const;

    std::chrono::nanoseconds get_elapsed() const;
    // Public method to render the bar state to a string
    std::string to_string() const;
    void print_progress();

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

struct ProgressState {
    std::unique_ptr<ProgressBar> bar;
    std::atomic<bool> visible{true};
    std::atomic<bool> active{false};
};

class MultiprocessProgressTracker; // Forward declaration

template <typename Mutex>
class ProgressTrackerSink : public spdlog::sinks::base_sink<Mutex> {
public:
    explicit ProgressTrackerSink(
        MultiprocessProgressTracker* tracker,
        spdlog::color_mode mode = spdlog::color_mode::automatic);
    void set_tracker(MultiprocessProgressTracker* tracker);

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;
    void flush_() override;

private:
    MultiprocessProgressTracker* m_tracker;
    bool m_should_color{false};
    std::array<spdlog::string_view_t, spdlog::level::n_levels> m_colors{};

    void set_color_mode(spdlog::color_mode mode);
};

class MultiprocessProgressTracker {
public:
    explicit MultiprocessProgressTracker(
        std::string overall_description = "Working...");

    ~MultiprocessProgressTracker();
    MultiprocessProgressTracker(const MultiprocessProgressTracker&) = delete;
    MultiprocessProgressTracker&
    operator=(const MultiprocessProgressTracker&)              = delete;
    MultiprocessProgressTracker(MultiprocessProgressTracker&&) = delete;
    MultiprocessProgressTracker&
    operator=(MultiprocessProgressTracker&&) = delete;

    void start();
    void stop();
    int add_task(std::string_view description, SizeType total, bool transient);
    void update_task(int task_id,
                     SizeType completed,
                     double score        = -1.0,
                     double leaves       = -1.0,
                     bool force_complete = false);
    void queue_log(std::string msg);

private:
    std::string m_overall_description;
    std::map<int, ProgressState> m_tasks;
    std::atomic<bool> m_running;
    std::thread m_render_thread;
    std::mutex m_tasks_mutex;
    std::mutex m_control_mutex;
    std::mutex m_log_mutex;
    std::mutex m_output_mutex;
    std::condition_variable m_cv;
    std::queue<std::string> m_log_queue;
    std::atomic<bool> m_permanently_stopped;
    int m_next_task_id{};
    int m_last_rendered_lines{};
    std::shared_ptr<ProgressTrackerSink<std::mutex>> m_sink;
    std::shared_ptr<spdlog::logger> m_previous_logger;

    void final_cleanup();
    void clear_progress_lines() const noexcept;
    void render_loop();
    void render_frame(bool is_final_render);
};

// This class is used to hide the console cursor during progress bar updates
// and restore it after the progress bar is done.
class ProgressGuard {
public:
    explicit ProgressGuard(bool show);
    ~ProgressGuard();
    ProgressGuard(const ProgressGuard&)            = delete;
    ProgressGuard& operator=(const ProgressGuard&) = delete;
    ProgressGuard(ProgressGuard&&)                 = delete;
    ProgressGuard& operator=(ProgressGuard&&)      = delete;

private:
    bool m_show;
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

private:
    std::unique_ptr<ProgressBar> m_owned_bar;
    MultiprocessProgressTracker* m_tracker = nullptr;
    // To batch updates for the tracker
    int m_task_id           = -1;
    double m_current_score  = -1.0;
    double m_current_leaves = -1.0;
};

// Progress bar factory functions
ProgressBar make_standard_bar(std::string_view prefix,
                              SizeType max_progress,
                              bool transient = true);
std::unique_ptr<ProgressBar> make_ffa_bar(std::string_view prefix,
                                          SizeType max_progress,
                                          bool transient = true);
std::unique_ptr<ProgressBar> make_pruning_bar(std::string_view prefix,
                                              SizeType max_progress,
                                              bool transient = true,
                                              bool managed   = false);

} // namespace loki::progress