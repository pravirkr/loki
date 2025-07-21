#include "loki/progress.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>

#include <fmt/color.h>
#include <fmt/format.h>

namespace loki::progress {

namespace details {
namespace {

void show_console_cursor(bool const show) {
    std::fputs(show ? "\033[?25h" : "\033[?25l", stdout);
}

void erase_line() { std::fputs("\r\033[K", stdout); }

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
    auto ns        = std::chrono::duration_cast<std::chrono::nanoseconds>(dur);
    auto secs      = std::chrono::floor<std::chrono::seconds>(ns);
    const auto hms = std::chrono::hh_mm_ss{secs};
    auto days      = std::chrono::duration_cast<
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
} // namespace details

// --- Column Implementations ---
std::string TextColumn::render(const ProgressBar& /*bar*/) {
    return fmt::format("{}", fmt::styled(m_text, m_style.value));
}
std::string SpinnerColumn::render(const ProgressBar& bar) {
    if (bar.is_completed()) {
        const auto* completion_symbol = m_use_tva_frames ? "*" : "✔";
        return fmt::format(
            "{}", fmt::styled(completion_symbol,
                              fmt::fg(fmt::terminal_color::bright_green) |
                                  fmt::emphasis::bold));
    }

    constexpr double kIntervalMs = 150.0;
    auto frame_no =
        (static_cast<double>(bar.get_elapsed().count()) / 1e6) / kIntervalMs;
    if (m_use_tva_frames) {
        return fmt::format(
            "{}",
            fmt::styled(
                kTVAFrames[static_cast<SizeType>(frame_no) % kTVAFrames.size()],
                m_style.value));
    }
    return fmt::format(
        "{}",
        fmt::styled(kFrames[static_cast<SizeType>(frame_no) % kFrames.size()],
                    m_style.value));
}
std::string BarColumn::render_pulse(const ProgressBar& bar) const {
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
                                      m_style.value));
    result +=
        fmt::format("{}", fmt::styled(p1.substr(pulse_position + pulse_width),
                                      m_background_style.value));
    return result;
}

std::string BarColumn::render(const ProgressBar& bar) {
    if (bar.get_max_progress() == 0) { // Indeterminate
        return render_pulse(bar);
    }

    double progress_fraction = static_cast<double>(bar.get_progress()) /
                               static_cast<double>(bar.get_max_progress());
    progress_fraction = std::clamp(progress_fraction, 0.0, 1.0);

    const int completed_width = static_cast<int>(m_width * progress_fraction);
    const auto part1 = details::repeat_unicode(kBarChar, completed_width);
    const auto part2 =
        details::repeat_unicode(kBarChar, m_width - completed_width);

    return fmt::format("{}{}", fmt::styled(part1, m_style.value),
                       fmt::styled(part2, m_background_style.value));
}
std::string PercentageColumn::render(const ProgressBar& bar) {
    const auto progress     = bar.get_progress();
    const auto max_progress = bar.get_max_progress();
    if (max_progress == 0) {
        return "";
    }
    const auto percentage =
        static_cast<int>(static_cast<double>(progress) /
                         static_cast<double>(max_progress) * 100.0);
    return fmt::format(
        "{}", fmt::styled(fmt::format("{:02d}%", percentage), m_style.value));
}
std::string TimeStatsColumn::render(const ProgressBar& bar) {
    const auto elapsed      = bar.get_elapsed();
    const auto progress     = bar.get_progress();
    const auto max_progress = bar.get_max_progress();
    if (max_progress == 0) {
        return fmt::format("{}", fmt::styled("[??:??s]", m_style.value));
    }
    auto eta_str = [&]() -> std::string {
        if (progress == 0) {
            return "--:--s";
        }
        const auto estimated_total =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed * (static_cast<double>(max_progress) /
                           static_cast<double>(progress)));
        const auto remaining =
            std::max(estimated_total - elapsed, std::chrono::nanoseconds{0});
        return details::format_duration(remaining);
    };

    const auto rendered =
        fmt::format("[{}<{}]", details::format_duration(elapsed), eta_str());
    return fmt::format("{}", fmt::styled(rendered, m_style.value));
}

std::string ScoreColumn::render(const ProgressBar& bar) {
    return fmt::format(
        "{}", fmt::styled(fmt::format("Score: {:.2f}", bar.get_score()),
                          m_style.value));
}
std::string LeavesColumn::render(const ProgressBar& bar) {
    return fmt::format(
        "{}", fmt::styled(fmt::format("Leaves: {:.2f}", bar.get_leaves()),
                          m_style.value));
}

ProgressBar::ProgressBar(std::string_view prefix,
                         SizeType max_progress,
                         int bar_width,
                         bool transient,
                         bool managed)
    : m_max_progress(max_progress),
      m_transient(transient),
      m_is_managed(managed) {
    add_column(std::make_unique<TextColumn>(
        prefix,
        Style{.value = fmt::fg(tva_colors::kOrange) | fmt::emphasis::bold}));
    add_column(std::make_unique<TextColumn>(" "));
    add_column(std::make_unique<SpinnerColumn>(
        Style{.value = fmt::fg(tva_colors::kOrange)},
        /*use_tva_frames=*/true));
    add_column(std::make_unique<TextColumn>(" "));
    add_column(std::make_unique<BarColumn>(bar_width));
    add_column(std::make_unique<TextColumn>(" "));
    add_column(std::make_unique<PercentageColumn>());
    add_column(std::make_unique<TextColumn>(kSeparator));
    add_column(std::make_unique<TimeStatsColumn>());
}

ProgressBar::~ProgressBar() {
    if (!m_is_managed) {
        mark_as_completed();
    }
}
ProgressBar& ProgressBar::add_column(std::unique_ptr<Column> col) {
    m_columns.push_back(std::move(col));
    return *this;
}
ProgressBar& ProgressBar::add_score_column() {
    add_column(std::make_unique<TextColumn>(kSeparator));
    add_column(std::make_unique<ScoreColumn>());
    return *this;
}
ProgressBar& ProgressBar::add_leaves_column() {
    add_column(std::make_unique<TextColumn>(kSeparator));
    add_column(std::make_unique<LeavesColumn>());
    return *this;
}
void ProgressBar::set_progress(SizeType new_progress) {
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
void ProgressBar::set_score(double s) { m_score = s; }
void ProgressBar::set_leaves(double l) { m_leaves = l; }
void ProgressBar::tick() { set_progress(m_progress + 1); }
bool ProgressBar::is_completed() const { return m_completed.load(); }
bool ProgressBar::is_transient() const { return m_transient; }
void ProgressBar::mark_as_completed() {
    if (!is_completed()) {
        m_completed = true;
        if (!m_is_managed) {
            print_progress();
        }
    }
}
SizeType ProgressBar::get_progress() const { return m_progress.load(); }
SizeType ProgressBar::get_max_progress() const { return m_max_progress; }
double ProgressBar::get_score() const { return m_score.load(); }
double ProgressBar::get_leaves() const { return m_leaves.load(); }

std::chrono::nanoseconds ProgressBar::get_elapsed() const {
    if (is_completed()) {
        return m_elapsed;
    }
    return std::chrono::steady_clock::now() - m_start_time;
}

// Public method to render the bar state to a string
std::string ProgressBar::to_string() const {
    std::stringstream line_ss;
    for (const auto& column : m_columns) {
        line_ss << column->render(*this);
    }
    return line_ss.str();
}

void ProgressBar::print_progress() {
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
    std::lock_guard<Mutex> lock(this->mutex_);
    m_tracker = tracker;
}
template <class Mutex>
void ProgressTrackerSink<Mutex>::sink_it_(const spdlog::details::log_msg& msg) {
    if (m_tracker == nullptr) {
        return;
    }

    spdlog::memory_buf_t formatted;
    this->formatter_->format(msg, formatted);

    if (m_should_color && msg.color_range_end > msg.color_range_start) {
        std::string result;
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
        m_tracker->queue_log(std::move(result));
    } else {
        m_tracker->queue_log(fmt::to_string(formatted));
    }
}
template <class Mutex> void ProgressTrackerSink<Mutex>::flush_() {}

// Explicitly instantiate for std::mutex:
template class ProgressTrackerSink<std::mutex>;

// --- MultiprocessProgressTracker Implementations ---
MultiprocessProgressTracker::MultiprocessProgressTracker(
    std::string overall_description)
    : m_overall_description(std::move(overall_description)),
      m_running(false),
      m_permanently_stopped(false) {
    m_previous_logger = spdlog::default_logger();
    m_sink            = std::make_shared<ProgressTrackerSink<std::mutex>>(
        this, spdlog::color_mode::always);
    auto logger = std::make_shared<spdlog::logger>("multi_progress", m_sink);
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
}

MultiprocessProgressTracker::~MultiprocessProgressTracker() {
    stop();
    if (m_sink) {
        m_sink->set_tracker(nullptr);
    }
    if (m_previous_logger) {
        // Restores the original!
        spdlog::set_default_logger(m_previous_logger);
    }
}

void MultiprocessProgressTracker::start() {
    std::lock_guard<std::mutex> lock(m_control_mutex);
    if (!m_running.load() && !m_permanently_stopped.load()) {
        details::show_console_cursor(false);
        m_running.store(true);
        m_render_thread =
            std::thread(&MultiprocessProgressTracker::render_loop, this);
    }
}

void MultiprocessProgressTracker::stop() {
    std::unique_lock<std::mutex> lock(m_control_mutex);
    if (!m_running.load()) {
        return;
    }
    m_running.store(false);
    lock.unlock();

    m_cv.notify_one();
    if (m_render_thread.joinable()) {
        m_render_thread.join();
    }

    if (!m_permanently_stopped.load()) {
        render_frame(true); // Final render to flush logs
        clear_progress_lines();
        std::cerr << std::flush;
        details::show_console_cursor(true);
        m_permanently_stopped.store(true);
    }
}

int MultiprocessProgressTracker::add_task(std::string_view description,
                                          SizeType total,
                                          bool transient) {
    std::lock_guard<std::mutex> lock(m_tasks_mutex);
    int task_id = m_next_task_id++;
    auto bar    = make_pruning_bar(description, total, transient,
                                   /*managed=*/true);
    auto& state = m_tasks[task_id];
    state.bar   = std::move(bar);
    return task_id;
}

void MultiprocessProgressTracker::update_task(int task_id,
                                              SizeType completed,
                                              double score,
                                              double leaves,
                                              bool force_complete) {
    {
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
                if (it->second.bar->is_transient()) {
                    it->second.visible.store(false);
                }
            }
        }
    }
    m_cv.notify_one();
}

void MultiprocessProgressTracker::queue_log(std::string msg) {
    std::lock_guard<std::mutex> lock(m_log_mutex);
    m_log_queue.push(std::move(msg));
    m_cv.notify_one();
}

void MultiprocessProgressTracker::final_cleanup() {
    std::lock_guard<std::mutex> lock(m_output_mutex);
    clear_progress_lines();
    std::cerr << std::flush;
}

void MultiprocessProgressTracker::clear_progress_lines() const noexcept {
    if (m_last_rendered_lines > 0) {
        std::cerr << "\r\033[K";
        for (int i = 1; i < m_last_rendered_lines; ++i) {
            std::cerr << "\033[A\033[K";
        }
        std::cerr << "\r";
    }
}

void MultiprocessProgressTracker::render_loop() {
    while (m_running.load()) {
        render_frame(false);
        std::unique_lock<std::mutex> lock(m_control_mutex);
        m_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return !m_running.load() || !m_log_queue.empty();
        });
    }
}

void MultiprocessProgressTracker::render_frame(bool is_final_render) {
    std::vector<std::string> logs_to_print;
    {
        std::lock_guard<std::mutex> log_lock(m_log_mutex);
        while (!m_log_queue.empty()) {
            logs_to_print.push_back(std::move(m_log_queue.front()));
            m_log_queue.pop();
        }
    }

    std::vector<std::string> lines_to_render;
    {
        std::lock_guard<std::mutex> task_lock(m_tasks_mutex);
        std::vector<int> active_and_visible_task_ids;
        for (auto const& [id, task] : m_tasks) {
            if ((task.visible.load() || is_final_render) &&
                task.active.load()) {
                active_and_visible_task_ids.push_back(id);
            }
        }
        std::ranges::sort(active_and_visible_task_ids);
        for (int id : active_and_visible_task_ids) {
            lines_to_render.push_back(m_tasks.at(id).bar->to_string());
        }
    }

    std::lock_guard<std::mutex> output_lock(m_output_mutex);

    // --- Critical Rendering Section ---
    // 1. Clear only the old progress bar lines
    clear_progress_lines();

    // 2. Print any logs. This will scroll the terminal as needed.
    for (const auto& log : logs_to_print) {
        std::cerr << log; // Assume log messages have newlines
    }

    // 3. Render the new progress bars at the bottom.
    for (size_t i = 0; i < lines_to_render.size(); ++i) {
        std::cerr << lines_to_render[i]
                  << (i == lines_to_render.size() - 1 ? "" : "\n");
    }
    std::cerr << std::flush;
    m_last_rendered_lines = static_cast<int>(lines_to_render.size());
}

// --- ProgressGuard Implementations ---

ProgressGuard::ProgressGuard(bool show) : m_show(show) {
    if (m_show) {
        details::show_console_cursor(false);
    }
}

ProgressGuard::~ProgressGuard() {
    if (m_show) {
        details::show_console_cursor(true);
    }
}

// --- ProgressTracker Implementations ---
ProgressTracker::ProgressTracker(std::string_view description,
                                 SizeType max_progress,
                                 MultiprocessProgressTracker* tracker,
                                 int task_id)
    : m_tracker(tracker),
      m_task_id(task_id) {
    if (m_tracker == nullptr) {
        // Standalone mode: create and own a non-transient bar
        m_owned_bar =
            make_pruning_bar(description, max_progress, /*transient=*/true,
                             /*managed=*/false);
    }
}

ProgressTracker::~ProgressTracker() {
    // RAII cleanup: ensure the bar is marked as finished when the scope ends
    mark_as_completed();
}

void ProgressTracker::set_score(double value) {
    m_current_score = value;
    if (m_owned_bar) {
        m_owned_bar->set_score(value);
    }
}

void ProgressTracker::set_leaves(double value) {
    m_current_leaves = value;
    if (m_owned_bar) {
        m_owned_bar->set_leaves(value);
    }
}

void ProgressTracker::set_progress(SizeType value) {
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

void ProgressTracker::mark_as_completed() {
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

// --- Progress bar factory functions ---
ProgressBar make_standard_bar(std::string_view prefix,
                              SizeType max_progress,
                              bool transient) {
    return ProgressBar(prefix, max_progress, 40, transient, false);
}

std::unique_ptr<ProgressBar>
make_ffa_bar(std::string_view prefix, SizeType max_progress, bool transient) {
    auto bar = std::make_unique<ProgressBar>(prefix, max_progress, 40,
                                             transient, false);
    bar->add_leaves_column();
    return bar;
}

std::unique_ptr<ProgressBar> make_pruning_bar(std::string_view prefix,
                                              SizeType max_progress,
                                              bool transient,
                                              bool managed) {
    auto bar = std::make_unique<ProgressBar>(prefix, max_progress, 40,
                                             transient, managed);
    bar->add_score_column();
    bar->add_leaves_column();
    return bar;
}

} // namespace loki::progress