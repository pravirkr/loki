#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
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

#include "loki/common/terminal.hpp"
#include "loki/common/types.hpp"

namespace loki::progress {

namespace detail {
// A foreground style in the shared palette (loki/common/terminal.hpp).
constexpr fmt::text_style fg(TermColor color) {
    return fmt::fg(fmt::rgb(to_rgb(color)));
}
} // namespace detail

// How a shown bar is drawn: loki::get_progress_mode() resolved against the
// environment (LOKI_PROGRESS) and stdout (see loki/common/progress.hpp).
enum class Render : uint8_t { kOff, kRich, kPlain };

// kOff when !show; otherwise the resolved process-wide mode.
Render resolve_render(bool show);

struct Style {
    fmt::text_style value;
};

// Options of one rendered line.
struct RenderContext {
    bool live{true};  // rich frame: spinner and bar are drawn
    bool color{true}; // ANSI styles
    int bar_width{40};
};

class ProgressBar;

// One field of a bar line. A column that renders "" is skipped together
// with its separator.
class Column {
public:
    explicit Column(Style style = {}, std::string_view separator = " ")
        : m_style(style),
          m_separator(separator) {}
    virtual ~Column()                                          = default;
    Column(const Column&)                                      = delete;
    Column& operator=(const Column&)                           = delete;
    Column(Column&&)                                           = delete;
    Column& operator=(Column&&)                                = delete;
    virtual std::string render(const ProgressBar& bar,
                               const RenderContext& ctx) const = 0;
    std::string_view separator() const { return m_separator; }

protected:
    std::string styled(std::string_view text, const RenderContext& ctx) const;
    static std::string
    styled(std::string_view text, const Style& style, const RenderContext& ctx);

    Style m_style{};

private:
    std::string m_separator;
};

class TextColumn : public Column {
public:
    explicit TextColumn(std::string_view text,
                        Style style                = {},
                        std::string_view separator = " ")
        : Column(style, separator),
          m_text(text) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;

private:
    std::string m_text;
};

class SpinnerColumn : public Column {
public:
    explicit SpinnerColumn(
        Style style         = {.value = detail::fg(TermColor::kOrange)},
        bool use_tva_frames = false)
        : Column(style),
          m_use_tva_frames(use_tva_frames) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;

private:
    bool m_use_tva_frames;
    static constexpr std::array<std::string_view, 10> kFrames = {
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
    };
    // TVA elevator symbols (ASCII approximations)
    static constexpr std::array<std::string_view, 6> kTVAFrames = {
        "o", // Null/empty set
        "+", // Cross/intersection
        "=", // Identity/equivalence
        ":", // Observer/monitoring
        "*", // Variant/anomaly
        "-", // Timeline/linear
    };
};

class BarColumn : public Column {
public:
    explicit BarColumn(
        Style style            = {.value = detail::fg(TermColor::kRed)},
        Style background_style = {.value = detail::fg(TermColor::kBackground)})
        : Column(style),
          m_background_style(background_style) {}

    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;

private:
    std::string render_pulse(const ProgressBar& bar,
                             const RenderContext& ctx) const;
    Style m_background_style;
    static constexpr std::string_view kBarChar     = "━";
    static constexpr std::string_view kBarRestChar = "─";
};

class PercentageColumn : public Column {
public:
    explicit PercentageColumn(
        Style style = {.value = detail::fg(TermColor::kAmber)})
        : Column(style) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;
};

// "done/total", for the overall line of a multi-task display.
class CountColumn : public Column {
public:
    explicit CountColumn(Style style = {.value = detail::fg(TermColor::kAmber)})
        : Column(style) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;
};

class TimeStatsColumn : public Column {
public:
    explicit TimeStatsColumn(
        Style style = {.value = detail::fg(TermColor::kDarkOrange)})
        : Column(style, kSeparator) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;
    static constexpr std::string_view kSeparator = " • ";
};

class ScoreColumn : public Column {
public:
    explicit ScoreColumn(
        Style style = {.value = fmt::fg(fmt::terminal_color::bright_red)})
        : Column(style, TimeStatsColumn::kSeparator) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;
};

class LeavesColumn : public Column {
public:
    explicit LeavesColumn(
        Style style = {.value = detail::fg(TermColor::kBrightGreen)})
        : Column(style, TimeStatsColumn::kSeparator) {}
    std::string render(const ProgressBar& bar,
                       const RenderContext& ctx) const override;
};

// State of one bar and its rendering to a line. Thread-safe; writes nothing:
// MultiprocessProgressTracker owns all output.
class ProgressBar {
public:
    enum class Kind : uint8_t {
        kStandard, // prefix, spinner, bar, percentage, time
        kFFA,      // + leaves
        kPruning,  // + score and leaves
        kOverall,  // prefix, spinner, bar, done/total, time
    };

    explicit ProgressBar(std::string_view prefix,
                         SizeType max_progress = 100,
                         Kind kind             = Kind::kStandard,
                         bool transient        = true);

    ~ProgressBar()                             = default;
    ProgressBar(const ProgressBar&)            = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&&)                 = delete;
    ProgressBar& operator=(ProgressBar&&)      = delete;

    // Restart the clock (a task that starts after its bar was made).
    void restart_clock();
    // Completes the bar when it reaches max_progress (> 0), except a
    // kOverall bar: tasks may still be added; it completes explicitly.
    void set_progress(SizeType new_progress);
    void set_max_progress(SizeType max_progress);
    void set_score(double s);
    void set_leaves(double l);
    void mark_as_completed();

    bool is_completed() const;
    bool is_transient() const;
    SizeType get_progress() const;
    SizeType get_max_progress() const;
    double get_score() const;
    double get_leaves() const;
    // Frozen at completion.
    std::chrono::nanoseconds get_elapsed() const;

    // One line. With max_cols > 0 the bar shrinks (then disappears) so the
    // visible width fits in max_cols.
    std::string to_string(const RenderContext& ctx, int max_cols = 0) const;

private:
    std::vector<std::unique_ptr<Column>> m_columns;
    std::atomic<SizeType> m_max_progress;
    std::atomic<SizeType> m_progress{0};
    std::atomic<bool> m_completed{false};
    std::atomic<double> m_score{0.0};
    std::atomic<double> m_leaves{0.0};
    std::atomic<int64_t> m_start_ns{0};
    std::atomic<int64_t> m_end_ns{0};
    Kind m_kind;
    bool m_transient;

    std::string render_columns(const RenderContext& ctx) const;
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

// The display of one or more bars ("tasks"); the only writer of progress
// output. The mode is fixed at construction (resolve_render):
//   kRich:  a live region at the bottom of the terminal, redrawn by a render
//           thread. Log lines are routed above it; completed transient bars
//           vanish and the others stay as permanent lines. The cursor is
//           hidden while it runs.
//   kPlain: self-contained lines at 25/50/75 % and at completion: of the
//           task for a standalone bar (ProgressTracker), of the count of
//           tasks done otherwise.
//   kOff:   nothing at all.
// Only one rich display runs at a time; a second one started meanwhile
// draws nothing.
class MultiprocessProgressTracker {
public:
    explicit MultiprocessProgressTracker(
        std::string overall_description = "Working...", bool show = true);

    ~MultiprocessProgressTracker();
    MultiprocessProgressTracker(const MultiprocessProgressTracker&) = delete;
    MultiprocessProgressTracker&
    operator=(const MultiprocessProgressTracker&)              = delete;
    MultiprocessProgressTracker(MultiprocessProgressTracker&&) = delete;
    MultiprocessProgressTracker&
    operator=(MultiprocessProgressTracker&&) = delete;

    void start();
    // Idempotent. Leaves the cursor below the permanent lines.
    void stop();
    int add_task(std::string_view description,
                 SizeType total,
                 bool transient         = true,
                 ProgressBar::Kind kind = ProgressBar::Kind::kPruning);
    // completed == SizeType(-1) leaves the progress unchanged; negative
    // score / leaves leave them unchanged.
    void update_task(int task_id,
                     SizeType completed,
                     double score        = -1.0,
                     double leaves       = -1.0,
                     bool force_complete = false);
    void queue_log(std::string msg);

private:
    friend class ProgressTracker;

    struct Task {
        std::unique_ptr<ProgressBar> bar;
        bool active{false};
        bool committed{false}; // final line printed (rich) or plain done
        int next_quarter{1};   // next plain milestone, in quarters
    };

    std::atomic<Render> m_render;
    std::string m_overall_description;
    ProgressBar m_overall;
    int m_overall_next_quarter{1};
    std::map<int, Task> m_tasks;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_stopped{false};
    bool m_owns_live{false};
    bool m_standalone{false}; // display of one ProgressTracker
    bool m_color{true};       // rich mode: false when NO_COLOR is set
    std::thread m_render_thread;
    std::mutex m_tasks_mutex;
    std::mutex m_control_mutex;
    std::mutex m_wait_mutex;
    std::mutex m_log_mutex;
    std::condition_variable m_cv;
    std::queue<std::string> m_log_queue;
    int m_next_task_id{};
    int m_last_rendered_lines{};
    std::shared_ptr<ProgressTrackerSink<std::mutex>> m_sink;
    std::shared_ptr<spdlog::logger> m_previous_logger;
    std::shared_ptr<spdlog::logger> m_logger;

    void render_loop();
    void render_frame(bool is_final_render);
    void install_logger();
    void restore_logger();
    // Plain milestones of the task (standalone) or of the overall count;
    // m_tasks_mutex held.
    std::vector<std::string> plain_lines(Task& task);
    void sync_overall_locked();
};

// Handle of one bar: either a task of a shared display or a standalone bar
// with a display of its own. Completes the bar when it goes out of scope,
// unless an exception (e.g. loki::Interrupted) unwinds it: then the bar is
// dropped without a completion line.
class ProgressTracker {
public:
    // Standalone bar. Draws nothing when !show.
    ProgressTracker(std::string_view description,
                    SizeType max_progress,
                    bool show,
                    ProgressBar::Kind kind = ProgressBar::Kind::kStandard,
                    bool transient         = true);
    // Task task_id of tracker. A null tracker draws nothing.
    ProgressTracker(MultiprocessProgressTracker* tracker, int task_id);

    ~ProgressTracker();
    ProgressTracker(const ProgressTracker&)            = delete;
    ProgressTracker& operator=(const ProgressTracker&) = delete;
    ProgressTracker(ProgressTracker&&)                 = delete;
    ProgressTracker& operator=(ProgressTracker&&)      = delete;

    void set_progress(SizeType value);
    void set_score(double value);
    void set_leaves(double value);
    // Completes the bar and, for a standalone bar, finishes its display, so
    // that what follows is printed below it.
    void mark_as_completed();

private:
    std::unique_ptr<MultiprocessProgressTracker> m_owned;
    MultiprocessProgressTracker* m_tracker = nullptr;
    int m_uncaught_exceptions{std::uncaught_exceptions()};
    // To batch updates for the tracker
    int m_task_id           = -1;
    double m_current_score  = -1.0;
    double m_current_leaves = -1.0;
};

} // namespace loki::progress
