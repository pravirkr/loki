#include "lib/detail/progress.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <memory>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/color.h>
#include <fmt/format.h>
#include <spdlog/common.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/details/os.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#ifndef _WIN32
#include <csignal>

#include <signal.h> // NOLINT(modernize-deprecated-headers): POSIX sigaction
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "loki/common/progress.hpp"
#include "loki/common/terminal.hpp"
#include "loki/common/types.hpp"

namespace loki::progress {

namespace detail {
namespace {

// Terminal control sequences of the rich mode.
constexpr std::string_view kHideCursor   = "\033[?25l";
constexpr std::string_view kShowCursor   = "\033[?25h";
constexpr std::string_view kWrapOff      = "\033[?7l";
constexpr std::string_view kWrapOn       = "\033[?7h";
constexpr std::string_view kEraseBelow   = "\033[J";
constexpr int kDefaultBarWidth           = 40;
constexpr int kMinBarWidth               = 10;
constexpr auto kFrameInterval            = std::chrono::milliseconds(100);
constexpr std::string_view kLoggerName   = "loki_progress";
constexpr std::string_view kLoggerFormat = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v";

struct TermSize {
    int cols{80};
    int rows{24};
};

TermSize terminal_size() {
    TermSize size;
#ifndef _WIN32
    // winsize and TIOCGWINSZ come from <sys/ioctl.h> through a platform
    // header (sys/ttycom.h on macOS, asm/termios.h on Linux).
    winsize ws{}; // NOLINT(misc-include-cleaner): see above
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg,misc-include-cleaner): ioctl
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        if (ws.ws_col > 0) {
            size.cols = ws.ws_col;
        }
        if (ws.ws_row > 0) {
            size.rows = ws.ws_row;
        }
    }
#endif
    return size;
}

// Columns taken by s on a terminal: ANSI CSI sequences take none, every
// UTF-8 code point one (all glyphs of the bars are narrow).
int visible_width(std::string_view s) {
    int width = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c == 0x1B && i + 1 < s.size() && s[i + 1] == '[') {
            i += 2;
            while (i < s.size() && (static_cast<unsigned char>(s[i]) < 0x40 ||
                                    static_cast<unsigned char>(s[i]) > 0x7E)) {
                ++i;
            }
            continue;
        }
        if ((c & 0xC0U) != 0x80U) {
            ++width;
        }
    }
    return width;
}

// All progress output goes through here, to C stdout (the stream spdlog
// writes to), so it orders with the logs and never calls back into Python.
std::mutex& output_mutex() {
    static std::mutex mutex;
    return mutex;
}

// Set by a fatal signal once the terminal is restored: nothing may draw after
// it. Constant-initialised and lock-free, so the signal handler may set it.
std::atomic<bool>& output_released() {
    static std::atomic<bool> released{false};
    static_assert(std::atomic<bool>::is_always_lock_free);
    return released;
}

void write_out(std::string_view s) {
    if (s.empty()) {
        return;
    }
    const std::scoped_lock lock(output_mutex());
    if (output_released().load()) {
        return;
    }
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

// Set while a rich display owns the bottom of the terminal.
std::atomic<bool>& live_display_active() {
    static std::atomic<bool> active{false};
    return active;
}

#ifndef _WIN32
// While a rich display is live, a fatal signal must not leave the terminal
// with a hidden cursor and wrapping off. Only one rich display runs at a
// time, so one set of saved handlers suffices.
struct SignalRestore {
    static constexpr std::array<int, 3> kSignals = {SIGINT, SIGTERM, SIGHUP};
    std::array<struct sigaction, 3> previous{};
    std::array<bool, 3> installed{};
};

SignalRestore& signal_restore() {
    static SignalRestore s;
    return s;
}

void on_terminal_signal(int sig, siginfo_t* info, void* context) {
    static constexpr std::string_view kRestore = "\033[?7h\033[?25h";
    auto& s                                    = signal_restore();
    const struct sigaction* prev               = nullptr;
    for (std::size_t i = 0; i < SignalRestore::kSignals.size(); ++i) {
        if (SignalRestore::kSignals[i] == sig && s.installed[i]) {
            prev = &s.previous[i];
            break;
        }
    }
    const bool siginfo =
        prev != nullptr && (static_cast<unsigned>(prev->sa_flags) &
                            static_cast<unsigned>(SA_SIGINFO)) != 0U;
    const bool fatal =
        prev != nullptr && !siginfo && prev->sa_handler == SIG_DFL;
    if (fatal) {
        // The default action ends the process: the render thread must not
        // draw a frame after the restore below.
        output_released().store(true);
    }
    // write() is async-signal-safe; stdio is not.
    [[maybe_unused]] const auto restored =
        write(STDOUT_FILENO, kRestore.data(), kRestore.size());
    if (prev == nullptr) {
        return;
    }
    // Hand the signal on to the handler that was there before.
    if (siginfo) {
        prev->sa_sigaction(sig, info, context);
    } else if (fatal) {
        [[maybe_unused]] const auto newline = write(STDOUT_FILENO, "\n", 1);
        sigaction(sig, prev, nullptr);
        std::raise(sig);
    } else if (prev->sa_handler != SIG_IGN) {
        prev->sa_handler(sig);
    }
}

void install_signal_restore() {
    auto& s = signal_restore();
    struct sigaction action{};
    action.sa_sigaction = &on_terminal_signal;
    action.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    for (std::size_t i = 0; i < SignalRestore::kSignals.size(); ++i) {
        s.installed[i] =
            sigaction(SignalRestore::kSignals[i], &action, &s.previous[i]) == 0;
    }
}

void uninstall_signal_restore() {
    auto& s = signal_restore();
    for (std::size_t i = 0; i < SignalRestore::kSignals.size(); ++i) {
        if (s.installed[i]) {
            sigaction(SignalRestore::kSignals[i], &s.previous[i], nullptr);
            s.installed[i] = false;
        }
    }
}
#else
void install_signal_restore() {}
void uninstall_signal_restore() {}
#endif

// The stdout/stderr sinks of spdlog, which the live display replaces.
bool is_console_sink(const spdlog::sink_ptr& sink) {
    using spdlog::details::console_mutex;
    using spdlog::details::console_nullmutex;
    using spdlog::sinks::ansicolor_sink;
    using spdlog::sinks::stdout_sink_base;
    return std::dynamic_pointer_cast<ansicolor_sink<console_mutex>>(sink) ||
           std::dynamic_pointer_cast<ansicolor_sink<console_nullmutex>>(sink) ||
           std::dynamic_pointer_cast<stdout_sink_base<console_mutex>>(sink) ||
           std::dynamic_pointer_cast<stdout_sink_base<console_nullmutex>>(sink);
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string repeat_unicode(const std::string_view s, SizeType n) {
    std::string result;
    result.reserve(s.size() * n);
    for (SizeType i = 0; i < n; ++i) {
        result += s;
    }
    return result;
}

template <typename Rep, typename Period>
std::string format_duration(std::chrono::duration<Rep, Period> dur) {
    const auto ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(dur);
    const auto secs = std::chrono::floor<std::chrono::seconds>(ns);
    const auto hms  = std::chrono::hh_mm_ss{secs};
    const auto days = std::chrono::duration_cast<
        std::chrono::duration<int, std::ratio<86400>>>(secs);

    std::string result;

    if (days.count() > 0) {
        result += fmt::format("{:02d}d:", days.count());
    }
    if (hms.hours().count() > 0) {
        result += fmt::format("{:02d}:", hms.hours().count());
    }
    result += fmt::format("{:02d}:{:02d}s", hms.minutes().count(),
                          hms.seconds().count());

    return result;
}

} // namespace
} // namespace detail

Render resolve_render(bool show) {
    if (!show) {
        return Render::kOff;
    }
    auto mode = loki::get_progress_mode();
    if (mode == ProgressMode::kAuto) {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): read only, loki never setenv
        const char* env = std::getenv("LOKI_PROGRESS");
        if (env != nullptr && env[0] != '\0') {
            try {
                mode = loki::parse_progress_mode(env);
            } catch (const std::invalid_argument&) {
                static std::once_flag warned;
                std::call_once(warned, [env] {
                    spdlog::warn("Ignoring LOKI_PROGRESS='{}': expected "
                                 "'auto', 'rich', 'plain' or 'off'",
                                 env);
                });
            }
        }
    }
    switch (mode) {
    case ProgressMode::kOff:
        return Render::kOff;
    case ProgressMode::kRich:
        return Render::kRich;
    case ProgressMode::kPlain:
        return Render::kPlain;
    case ProgressMode::kAuto:
        break;
    }
    return loki::stdout_is_terminal() ? Render::kRich : Render::kPlain;
}

// --- Column Implementations ---
std::string Column::styled(std::string_view text,
                           const RenderContext& ctx) const {
    return styled(text, m_style, ctx);
}

std::string Column::styled(std::string_view text,
                           const Style& style,
                           const RenderContext& ctx) {
    if (!ctx.color) {
        return std::string(text);
    }
    return fmt::format("{}", fmt::styled(text, style.value));
}

std::string TextColumn::render(const ProgressBar& /*bar*/,
                               const RenderContext& ctx) const {
    return styled(m_text, ctx);
}

std::string SpinnerColumn::render(const ProgressBar& bar,
                                  const RenderContext& ctx) const {
    if (bar.is_completed()) {
        const std::string_view completion_symbol = m_use_tva_frames ? "*" : "✔";
        return styled(completion_symbol,
                      Style{
                          .value = fmt::fg(fmt::terminal_color::bright_green) |
                                   fmt::emphasis::bold,
                      },
                      ctx);
    }
    if (!ctx.live) {
        return "";
    }

    constexpr double kIntervalMs = 150.0;
    const auto frame_no          = static_cast<SizeType>(
        (static_cast<double>(bar.get_elapsed().count()) / 1e6) / kIntervalMs);
    if (m_use_tva_frames) {
        return styled(kTVAFrames[frame_no % kTVAFrames.size()], ctx);
    }
    return styled(kFrames[frame_no % kFrames.size()], ctx);
}

std::string BarColumn::render_pulse(const ProgressBar& bar,
                                    const RenderContext& ctx) const {
    const int width = ctx.bar_width;
    const double current_time =
        std::chrono::duration<double>(bar.get_elapsed()).count();
    const int pulse_width = std::max(width / 4, 1);
    // The pulse enters from the left and leaves on the right.
    const int travel = width + pulse_width;
    const int head =
        static_cast<int>(std::fmod(current_time * 1.5, 1.0) * travel);
    const int start = std::max(head - pulse_width, 0);
    const int end   = std::min(head, width);

    const auto rest = ctx.color ? kBarChar : kBarRestChar;
    std::string result;
    result +=
        styled(detail::repeat_unicode(rest, start), m_background_style, ctx);
    result +=
        styled(detail::repeat_unicode(kBarChar, std::max(end - start, 0)), ctx);
    result += styled(detail::repeat_unicode(rest, width - std::max(end, start)),
                     m_background_style, ctx);
    return result;
}

std::string BarColumn::render(const ProgressBar& bar,
                              const RenderContext& ctx) const {
    if (!ctx.live || ctx.bar_width <= 0) {
        return "";
    }
    if (bar.get_max_progress() == 0) { // Indeterminate
        return render_pulse(bar, ctx);
    }

    double progress_fraction = static_cast<double>(bar.get_progress()) /
                               static_cast<double>(bar.get_max_progress());
    progress_fraction        = std::clamp(progress_fraction, 0.0, 1.0);
    if (bar.is_completed()) {
        progress_fraction = 1.0;
    }

    const int completed_width =
        static_cast<int>(ctx.bar_width * progress_fraction);
    // Without colour the remaining part is a lighter line.
    const auto rest = ctx.color ? kBarChar : kBarRestChar;
    return styled(detail::repeat_unicode(kBarChar, completed_width), ctx) +
           styled(detail::repeat_unicode(rest, ctx.bar_width - completed_width),
                  m_background_style, ctx);
}

std::string PercentageColumn::render(const ProgressBar& bar,
                                     const RenderContext& ctx) const {
    const auto progress     = bar.get_progress();
    const auto max_progress = bar.get_max_progress();
    if (max_progress == 0) {
        return "";
    }
    const auto percentage =
        std::min(static_cast<int>(static_cast<double>(progress) /
                                  static_cast<double>(max_progress) * 100.0),
                 100);
    return styled(fmt::format("{:>3d}%", percentage), ctx);
}

std::string CountColumn::render(const ProgressBar& bar,
                                const RenderContext& ctx) const {
    const auto max_progress = bar.get_max_progress();
    if (max_progress == 0) {
        return "";
    }
    return styled(fmt::format("{}/{}", bar.get_progress(), max_progress), ctx);
}

std::string TimeStatsColumn::render(const ProgressBar& bar,
                                    const RenderContext& ctx) const {
    const auto elapsed      = bar.get_elapsed();
    const auto progress     = bar.get_progress();
    const auto max_progress = bar.get_max_progress();
    if (bar.is_completed() || max_progress == 0) {
        return styled(fmt::format("[{}]", detail::format_duration(elapsed)),
                      ctx);
    }
    const auto eta_str = [&]() -> std::string {
        if (progress == 0) {
            return "--:--s";
        }
        const auto estimated_total =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed * (static_cast<double>(max_progress) /
                           static_cast<double>(progress)));
        const auto remaining =
            std::max(estimated_total - elapsed, std::chrono::nanoseconds{0});
        return detail::format_duration(remaining);
    };

    return styled(
        fmt::format("[{}<{}]", detail::format_duration(elapsed), eta_str()),
        ctx);
}

std::string ScoreColumn::render(const ProgressBar& bar,
                                const RenderContext& ctx) const {
    return styled(fmt::format("Score: {:.2f}", bar.get_score()), ctx);
}

std::string LeavesColumn::render(const ProgressBar& bar,
                                 const RenderContext& ctx) const {
    return styled(fmt::format("Leaves: {:.2f}", bar.get_leaves()), ctx);
}

// --- ProgressBar Implementations ---
ProgressBar::ProgressBar(std::string_view prefix,
                         SizeType max_progress,
                         Kind kind,
                         bool transient)
    : m_max_progress(max_progress),
      m_start_ns(detail::now_ns()),
      m_kind(kind),
      m_transient(transient) {
    m_columns.push_back(std::make_unique<TextColumn>(
        prefix,
        Style{.value = detail::fg(TermColor::kOrange) | fmt::emphasis::bold}));
    m_columns.push_back(std::make_unique<SpinnerColumn>(
        Style{.value = detail::fg(TermColor::kOrange)},
        /*use_tva_frames=*/true));
    m_columns.push_back(std::make_unique<BarColumn>());
    if (kind == Kind::kOverall) {
        m_columns.push_back(std::make_unique<CountColumn>());
    } else {
        m_columns.push_back(std::make_unique<PercentageColumn>());
    }
    m_columns.push_back(std::make_unique<TimeStatsColumn>());
    if (kind == Kind::kPruning) {
        m_columns.push_back(std::make_unique<ScoreColumn>());
    }
    if (kind == Kind::kFFA || kind == Kind::kPruning) {
        m_columns.push_back(std::make_unique<LeavesColumn>());
    }
}

void ProgressBar::restart_clock() { m_start_ns = detail::now_ns(); }
void ProgressBar::set_progress(SizeType new_progress) {
    m_progress              = new_progress;
    const auto max_progress = m_max_progress.load();
    if (m_kind != Kind::kOverall && max_progress > 0 &&
        new_progress >= max_progress) {
        mark_as_completed();
    }
}
void ProgressBar::set_max_progress(SizeType max_progress) {
    m_max_progress = max_progress;
}
void ProgressBar::set_score(double s) { m_score = s; }
void ProgressBar::set_leaves(double l) { m_leaves = l; }
void ProgressBar::mark_as_completed() {
    if (m_completed.load()) {
        return;
    }
    // The end time is stored before the flag that makes it visible.
    m_end_ns    = detail::now_ns();
    m_completed = true;
}
bool ProgressBar::is_completed() const { return m_completed.load(); }
bool ProgressBar::is_transient() const { return m_transient; }
SizeType ProgressBar::get_progress() const { return m_progress.load(); }
SizeType ProgressBar::get_max_progress() const { return m_max_progress.load(); }
double ProgressBar::get_score() const { return m_score.load(); }
double ProgressBar::get_leaves() const { return m_leaves.load(); }

std::chrono::nanoseconds ProgressBar::get_elapsed() const {
    const int64_t end = is_completed() ? m_end_ns.load() : detail::now_ns();
    return std::chrono::nanoseconds{std::max<int64_t>(end - m_start_ns, 0)};
}

std::string ProgressBar::render_columns(const RenderContext& ctx) const {
    std::string line;
    bool first = true;
    for (const auto& column : m_columns) {
        const auto field = column->render(*this, ctx);
        if (field.empty()) {
            continue;
        }
        if (!first) {
            line += column->separator();
        }
        line += field;
        first = false;
    }
    return line;
}

std::string ProgressBar::to_string(const RenderContext& ctx,
                                   int max_cols) const {
    auto line = render_columns(ctx);
    if (max_cols <= 0 || !ctx.live || ctx.bar_width <= 0) {
        return line;
    }
    const int overflow = detail::visible_width(line) - max_cols;
    if (overflow <= 0) {
        return line;
    }
    RenderContext narrow = ctx;
    narrow.bar_width     = ctx.bar_width - overflow;
    if (narrow.bar_width < detail::kMinBarWidth) {
        narrow.bar_width = 0;
    }
    return render_columns(narrow);
}

// --- ProgressTrackerSink Implementations ---
template <class Mutex>
ProgressTrackerSink<Mutex>::ProgressTrackerSink(
    MultiprocessProgressTracker* tracker, spdlog::color_mode mode)
    : m_tracker(tracker) {
    set_color_mode(mode);
    // Use colors similar to spdlog's default
    m_colors[spdlog::level::trace]    = "\033[38;5;244m"; // gray
    m_colors[spdlog::level::debug]    = "\033[36m";       // cyan
    m_colors[spdlog::level::info]     = "\033[32m";       // green
    m_colors[spdlog::level::warn]     = "\033[33m";       // yellow
    m_colors[spdlog::level::err]      = "\033[31m";       // red
    m_colors[spdlog::level::critical] = "\033[1;41m";     // bold red bg
    m_colors[spdlog::level::off]      = "\033[m";         // reset
}

template <class Mutex>
void ProgressTrackerSink<Mutex>::set_color_mode(spdlog::color_mode mode) {
    if (mode == spdlog::color_mode::automatic) {
        m_should_color = spdlog::details::os::is_color_terminal();
    } else {
        m_should_color = mode == spdlog::color_mode::always;
    }
}

template <class Mutex>
void ProgressTrackerSink<Mutex>::set_tracker(
    MultiprocessProgressTracker* tracker) {
    const std::scoped_lock lock(this->mutex_);
    m_tracker = tracker;
}
template <class Mutex>
void ProgressTrackerSink<Mutex>::sink_it_(const spdlog::details::log_msg& msg) {
    spdlog::memory_buf_t formatted;
    this->formatter_->format(msg, formatted);

    std::string result;
    if (m_should_color && msg.color_range_end > msg.color_range_start) {
        result.reserve(formatted.size() + 16);
        // Part before color
        result.append(formatted.data(), msg.color_range_start);
        // Apply color
        result.append(m_colors[msg.level].data(), m_colors[msg.level].size());
        result.append(formatted.data() + msg.color_range_start,
                      msg.color_range_end - msg.color_range_start);
        result.append("\033[m"); // reset color
        // Part after color
        result.append(formatted.data() + msg.color_range_end,
                      formatted.size() - msg.color_range_end);
    } else {
        result = fmt::to_string(formatted);
    }
    if (m_tracker == nullptr) {
        // A logger copy that outlived the display: print directly.
        detail::write_out(result);
        return;
    }
    m_tracker->queue_log(std::move(result));
}
template <class Mutex> void ProgressTrackerSink<Mutex>::flush_() {}

// Explicitly instantiate for std::mutex:
template class ProgressTrackerSink<std::mutex>;

// --- MultiprocessProgressTracker Implementations ---
MultiprocessProgressTracker::MultiprocessProgressTracker(
    std::string overall_description, bool show)
    : m_render(resolve_render(show)),
      m_overall_description(std::move(overall_description)),
      m_overall(m_overall_description,
                0,
                ProgressBar::Kind::kOverall,
                /*transient=*/true) {}

// NOLINTNEXTLINE(bugprone-exception-escape) -- destructor may format strings.
MultiprocessProgressTracker::~MultiprocessProgressTracker() {
    stop();
    restore_logger();
}

void MultiprocessProgressTracker::start() {
    const std::scoped_lock lock(m_control_mutex);
    if (m_running.load() || m_stopped.load()) {
        return;
    }
    m_overall.restart_clock();
    if (m_render != Render::kRich) {
        return;
    }
    if (detail::live_display_active().exchange(true)) {
        // Another live display owns the terminal: stay silent rather than
        // draw over it.
        m_render = Render::kOff;
        return;
    }
    m_owns_live = true;
    m_color     = loki::color_allowed();
    detail::install_signal_restore();
    install_logger();
    detail::write_out(std::string(detail::kHideCursor) +
                      std::string(detail::kWrapOff));
    m_running.store(true);
    m_render_thread =
        std::thread(&MultiprocessProgressTracker::render_loop, this);
}

void MultiprocessProgressTracker::stop() {
    {
        const std::scoped_lock lock(m_control_mutex);
        if (m_stopped.exchange(true)) {
            return;
        }
    }
    if (m_owns_live) {
        {
            const std::scoped_lock lock(m_wait_mutex);
            m_running.store(false);
        }
        m_cv.notify_all();
        if (m_render_thread.joinable()) {
            m_render_thread.join();
        }
        render_frame(/*is_final_render=*/true);
        restore_logger();
        detail::uninstall_signal_restore();
        detail::write_out(std::string(detail::kWrapOn) +
                          std::string(detail::kShowCursor));
        detail::live_display_active().store(false);
        m_owns_live = false;
        return;
    }
    if (m_render == Render::kPlain) {
        // Several tasks: the overall line, unless it is already printed.
        std::string line;
        {
            const std::scoped_lock lock(m_tasks_mutex);
            const bool any_active = std::ranges::any_of(
                m_tasks, [](const auto& kv) { return kv.second.active; });
            if (!m_standalone && any_active && m_overall_next_quarter <= 4) {
                sync_overall_locked();
                m_overall.mark_as_completed();
                line                   = m_overall.to_string(
                                             RenderContext{.live = false, .color = false}) +
                                         "\n";
                m_overall_next_quarter = 5;
            }
        }
        detail::write_out(line);
    }
}

int MultiprocessProgressTracker::add_task(std::string_view description,
                                          SizeType total,
                                          bool transient,
                                          ProgressBar::Kind kind) {
    const std::scoped_lock lock(m_tasks_mutex);
    const int task_id = m_next_task_id++;
    if (m_render == Render::kOff) {
        return task_id;
    }
    m_tasks[task_id].bar =
        std::make_unique<ProgressBar>(description, total, kind, transient);
    return task_id;
}

void MultiprocessProgressTracker::update_task(int task_id,
                                              SizeType completed,
                                              double score,
                                              double leaves,
                                              bool force_complete) {
    if (m_render == Render::kOff) {
        return;
    }
    std::vector<std::string> lines;
    {
        const std::scoped_lock lock(m_tasks_mutex);
        const auto it = m_tasks.find(task_id);
        if (it == m_tasks.end()) {
            return;
        }
        auto& task = it->second;
        if (!task.active) {
            task.active = true;
            task.bar->restart_clock();
        }
        if (std::cmp_not_equal(completed, static_cast<SizeType>(-1))) {
            task.bar->set_progress(completed);
        }
        if (score >= 0) {
            task.bar->set_score(score);
        }
        if (leaves >= 0) {
            task.bar->set_leaves(leaves);
        }
        if (force_complete) {
            task.bar->mark_as_completed();
        }
        if (m_render == Render::kPlain) {
            lines = plain_lines(task);
        }
    }
    std::string out;
    for (const auto& line : lines) {
        out += line;
        out += '\n';
    }
    detail::write_out(out);
}

void MultiprocessProgressTracker::queue_log(std::string msg) {
    {
        const std::scoped_lock lock(m_log_mutex);
        m_log_queue.push(std::move(msg));
    }
    m_cv.notify_one();
}

void MultiprocessProgressTracker::install_logger() {
    m_previous_logger = spdlog::default_logger();
    m_sink            = std::make_shared<ProgressTrackerSink<std::mutex>>(
        this, m_color ? spdlog::color_mode::always : spdlog::color_mode::never);
    // Pattern on this sink only: the caller's sinks keep theirs.
    m_sink->set_pattern(std::string(detail::kLoggerFormat));
    // The display takes over the console sinks; any other sink (a log file)
    // keeps receiving every message.
    std::vector<spdlog::sink_ptr> sinks{m_sink};
    bool has_console   = false;
    auto console_level = spdlog::level::trace;
    if (m_previous_logger) {
        for (const auto& sink : m_previous_logger->sinks()) {
            if (detail::is_console_sink(sink)) {
                console_level = has_console
                                    ? std::min(console_level, sink->level())
                                    : sink->level();
                has_console   = true;
            } else {
                sinks.push_back(sink);
            }
        }
    }
    m_sink->set_level(console_level);
    m_logger = std::make_shared<spdlog::logger>(
        std::string(detail::kLoggerName), sinks.begin(), sinks.end());
    if (m_previous_logger) {
        m_logger->set_level(m_previous_logger->level());
        m_logger->flush_on(m_previous_logger->flush_level());
    }
    spdlog::set_default_logger(m_logger);
}

void MultiprocessProgressTracker::restore_logger() {
    if (!m_logger) {
        return;
    }
    if (m_previous_logger) {
        // Keep a level set while the display was running.
        m_previous_logger->set_level(m_logger->level());
        spdlog::set_default_logger(m_previous_logger);
    }
    m_sink->set_tracker(nullptr);
    m_logger.reset();
    m_previous_logger.reset();
}

void MultiprocessProgressTracker::render_loop() {
    while (m_running.load()) {
        render_frame(/*is_final_render=*/false);
        std::unique_lock<std::mutex> lock(m_wait_mutex);
        m_cv.wait_for(lock, detail::kFrameInterval, [this] {
            if (!m_running.load()) {
                return true;
            }
            const std::scoped_lock log_lock(m_log_mutex);
            return !m_log_queue.empty();
        });
    }
}

void MultiprocessProgressTracker::sync_overall_locked() {
    const auto ndone = std::ranges::count_if(
        m_tasks, [](const auto& kv) { return kv.second.bar->is_completed(); });
    m_overall.set_max_progress(m_tasks.size());
    m_overall.set_progress(static_cast<SizeType>(ndone));
}

std::vector<std::string> MultiprocessProgressTracker::plain_lines(Task& task) {
    const RenderContext plain{.live = false, .color = false};
    std::vector<std::string> lines;
    if (task.committed) {
        return lines;
    }
    if (m_standalone) {
        const auto& bar = *task.bar;
        if (bar.is_completed()) {
            lines.push_back(bar.to_string(plain));
            task.committed = true;
            return lines;
        }
        const auto max_progress = bar.get_max_progress();
        if (max_progress == 0) {
            return lines;
        }
        const auto quarter =
            static_cast<int>(4 * bar.get_progress() / max_progress);
        if (quarter >= task.next_quarter) {
            lines.push_back(bar.to_string(plain));
            task.next_quarter = quarter + 1;
        }
        return lines;
    }
    // Several tasks: one line per quarter of the tasks done.
    if (!task.bar->is_completed()) {
        return lines;
    }
    task.committed = true;
    if (m_overall_next_quarter > 4) {
        return lines;
    }
    sync_overall_locked();
    // The line of all tasks done is printed by stop(): more tasks may come.
    const auto quarter = static_cast<int>(4 * m_overall.get_progress() /
                                          m_overall.get_max_progress());
    if (quarter >= m_overall_next_quarter && quarter < 4) {
        lines.push_back(m_overall.to_string(plain));
        m_overall_next_quarter = quarter + 1;
    }
    return lines;
}

void MultiprocessProgressTracker::render_frame(bool is_final_render) {
    std::vector<std::string> logs;
    {
        const std::scoped_lock log_lock(m_log_mutex);
        while (!m_log_queue.empty()) {
            logs.push_back(std::move(m_log_queue.front()));
            m_log_queue.pop();
        }
    }

    const auto term = detail::terminal_size();
    // One spare column: a line that fills the last column would leave the
    // cursor in the pending-wrap state on some terminals.
    const int max_cols = std::max(term.cols - 1, 1);
    const RenderContext live{
        .live      = true,
        .color     = m_color,
        .bar_width = detail::kDefaultBarWidth,
    };

    std::vector<std::string> permanent;
    std::vector<std::string> frame;
    {
        const std::scoped_lock task_lock(m_tasks_mutex);
        std::vector<std::string> running;
        for (auto& [id, task] : m_tasks) {
            if (!task.active || task.committed) {
                continue;
            }
            const bool done = task.bar->is_completed();
            if (done || is_final_render) {
                // Completed (or abandoned at the end): transient bars
                // vanish, the others become permanent lines.
                if (!task.bar->is_transient()) {
                    permanent.push_back(task.bar->to_string(live, max_cols));
                }
                task.committed = true;
                continue;
            }
            running.push_back(task.bar->to_string(live, max_cols));
        }
        if (!is_final_render) {
            if (!m_standalone && m_tasks.size() > 1) {
                sync_overall_locked();
                frame.push_back(m_overall.to_string(live, max_cols));
            }
            // Never taller than the terminal: the cursor could not climb
            // back to the top of the region.
            const auto max_lines =
                static_cast<std::size_t>(std::max(term.rows - 2, 2));
            const std::size_t room =
                max_lines > frame.size() ? max_lines - frame.size() : 1;
            if (running.size() > room) {
                const auto hidden = running.size() - (room - 1);
                running.resize(room - 1);
                auto more = fmt::format("  … {} more running", hidden);
                running.push_back(
                    m_color ? fmt::format(
                                  "{}",
                                  fmt::styled(
                                      more, detail::fg(TermColor::kDarkOrange)))
                            : std::move(more));
            }
            frame.insert(frame.end(), std::make_move_iterator(running.begin()),
                         std::make_move_iterator(running.end()));
        }
    }

    std::string buf;
    // Back to the first column of the region's top line, then erase it and
    // everything below.
    if (m_last_rendered_lines > 0) {
        buf += '\r';
        if (m_last_rendered_lines > 1) {
            buf += fmt::format("\033[{}A", m_last_rendered_lines - 1);
        }
        buf += detail::kEraseBelow;
    }
    if (!logs.empty() || !permanent.empty()) {
        // Logs wrap as usual; the live region does not.
        buf += detail::kWrapOn;
        for (const auto& log : logs) {
            buf += log; // log messages end with a newline
        }
        for (const auto& line : permanent) {
            buf += line;
            buf += '\n';
        }
        buf += detail::kWrapOff;
    }
    for (std::size_t i = 0; i < frame.size(); ++i) {
        if (i > 0) {
            buf += '\n';
        }
        buf += frame[i];
    }
    m_last_rendered_lines = static_cast<int>(frame.size());
    detail::write_out(buf);
}

// --- ProgressTracker Implementations ---
ProgressTracker::ProgressTracker(std::string_view description,
                                 SizeType max_progress,
                                 bool show,
                                 ProgressBar::Kind kind,
                                 bool transient) {
    if (resolve_render(show) == Render::kOff) {
        return;
    }
    m_owned = std::make_unique<MultiprocessProgressTracker>(
        std::string(description), show);
    m_owned->m_standalone = true;
    m_task_id = m_owned->add_task(description, max_progress, transient, kind);
    m_owned->start();
    m_tracker = m_owned.get();
    // Shown at 0 % from the start, with the clock running.
    m_tracker->update_task(m_task_id, 0);
}

ProgressTracker::ProgressTracker(MultiprocessProgressTracker* tracker,
                                 int task_id)
    : m_tracker(tracker),
      m_task_id(task_id) {
    if (m_tracker != nullptr) {
        m_tracker->update_task(m_task_id, 0);
    }
}

// NOLINTNEXTLINE(bugprone-exception-escape) -- destructor may format strings.
ProgressTracker::~ProgressTracker() {
    if (std::uncaught_exceptions() > m_uncaught_exceptions) {
        // Unwinding: leave the bar unfinished. A transient bar vanishes, and
        // plain mode prints no completion line for work that did not end.
        m_task_id = -1;
        if (m_owned) {
            m_owned->stop();
        }
        return;
    }
    mark_as_completed();
}

void ProgressTracker::set_score(double value) { m_current_score = value; }

void ProgressTracker::set_leaves(double value) { m_current_leaves = value; }

void ProgressTracker::set_progress(SizeType value) {
    if (m_tracker == nullptr || m_task_id == -1) {
        return;
    }
    m_tracker->update_task(m_task_id, value, m_current_score, m_current_leaves);
    // Reset batched values
    m_current_score  = -1.0;
    m_current_leaves = -1.0;
}

void ProgressTracker::mark_as_completed() {
    if (m_tracker != nullptr && m_task_id != -1) {
        // Final update to mark as complete
        m_tracker->update_task(m_task_id, static_cast<SizeType>(-1),
                               m_current_score, m_current_leaves,
                               /*force_complete=*/true);
        m_task_id = -1; // Prevent double-completion
    }
    if (m_owned) {
        m_owned->stop();
    }
}

} // namespace loki::progress
