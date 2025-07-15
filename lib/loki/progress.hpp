#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include "loki/common/types.hpp"

namespace loki::progress {

struct ProgressState {
    std::string description;
    std::atomic<SizeType> completed;
    SizeType total{};
    std::atomic<double> score;
    std::atomic<double> leaves;
    std::atomic<bool> visible;
    std::chrono::steady_clock::time_point start_time;

    ProgressState() : completed(0), score(0.0), leaves(0.0), visible(true) {}
};

class MultiprocessProgressTracker {
public:
    explicit MultiprocessProgressTracker(
        std::string overall_description = "Working...")
        : m_overall_description(std::move(overall_description)),
          m_running(false),
          m_permanently_stopped(false) {
        auto stdout_sink =
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("logger", stdout_sink);
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
        if (m_last_rendered_lines > 0) {
            std::cout << "\033[" << m_last_rendered_lines << "A\r\033["
                      << m_last_rendered_lines << "M" << std::flush;
            m_last_rendered_lines = 0;
        }
        m_permanently_stopped.store(true);
    }

    int add_task(std::string description, SizeType total) {
        std::lock_guard<std::mutex> lock(m_tasks_mutex);
        int task_id                  = m_next_task_id++;
        m_tasks[task_id].description = std::move(description);
        m_tasks[task_id].total       = total;
        m_tasks[task_id].start_time  = std::chrono::steady_clock::now();
        return task_id;
    }

    void update_task(int task_id,
                     SizeType completed,
                     double score  = 0.0,
                     double leaves = 0.0) {
        std::lock_guard<std::mutex> lock(m_tasks_mutex);
        auto it = m_tasks.find(task_id);
        if (it != m_tasks.end()) {
            it->second.completed.store(completed);
            it->second.score.store(score);
            it->second.leaves.store(leaves);
            if (completed >= it->second.total) {
                it->second.visible.store(false);
            }
        }
    }

    template <typename... Args>
    void log(spdlog::level::level_enum level,
             fmt::format_string<Args...> fmt,
             Args&&... args) {
        bool was_running = false;
        {
            std::lock_guard<std::mutex> lock(m_control_mutex);
            was_running = m_running.exchange(false);
        }
        if (was_running && m_render_thread.joinable()) {
            m_render_thread.join();
        }
        if (m_last_rendered_lines > 0) {
            std::cout << "\033[" << m_last_rendered_lines << "A\r\033["
                      << m_last_rendered_lines << "M" << std::flush;
            m_last_rendered_lines = 0;
        }
        spdlog::default_logger()->log(level, fmt, std::forward<Args>(args)...);
        std::cout << std::flush;
        {
            std::lock_guard<std::mutex> lock(m_control_mutex);
            if (was_running && !m_permanently_stopped.load()) {
                m_running.store(true);
                m_render_thread = std::thread(
                    &MultiprocessProgressTracker::render_loop, this);
            }
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

    static constexpr std::array<std::string, 10> kSpinner = {
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};

    void render_loop() {
        while (m_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            std::lock_guard<std::mutex> lock(m_tasks_mutex);
            std::vector<int> visible_m_tasksids;
            for (const auto& pair : m_tasks) {
                if (pair.second.visible.load()) {
                    visible_m_tasksids.push_back(pair.first);
                }
            }

            std::ranges::sort(visible_m_tasksids);

            if (m_last_rendered_lines > 0) {
                std::cout << "\033[" << m_last_rendered_lines << "A\r";
            }

            int current_rendered_lines = 0;
            for (int task_id : visible_m_tasksids) {
                const auto& state = m_tasks[task_id];
                const auto progress_percent =
                    (static_cast<double>(state.completed.load()) /
                     static_cast<double>(state.total)) *
                    100.0;
                std::chrono::steady_clock::duration elapsed =
                    std::chrono::steady_clock::now() - state.start_time;

                std::stringstream line_ss;
                line_ss << "\033[34m" << std::left << std::setw(20)
                        << ("[" + state.description + "]") << "\033[0m" << " ";
                line_ss << "\033[33m" << get_spinner_frame() << "\033[0m"
                        << " ";

                std::string percentage_str =
                    std::to_string(static_cast<int>(progress_percent)) + "%";
                std::string eta_str =
                    format_eta(elapsed, state.completed.load(), state.total);
                std::string elapsed_str = format_elapsed(elapsed);
                std::string score_str =
                    "\033[31mScore: " +
                    std::to_string(static_cast<int>(state.score.load())) +
                    "\033[0m";
                std::string leaves_str =
                    "\033[32mLeaves: " +
                    std::to_string(static_cast<int>(state.leaves.load())) +
                    "\033[0m";

                const auto fixed_width = static_cast<int>(
                    20 + 1 + 2 + percentage_str.length() + 3 +
                    eta_str.length() + 3 + elapsed_str.length() + 3 +
                    score_str.length() + 3 + leaves_str.length());

                const auto terminal_width = get_terminal_width();
                const auto bar_width =
                    std::max(10, terminal_width - fixed_width - 5);

                line_ss << get_progress_bar(state.completed.load(), state.total,
                                            bar_width)
                        << " " << std::right << std::setw(3)
                        << static_cast<int>(progress_percent) << "% ";
                line_ss << "\033[2m| " << eta_str << "\033[0m";
                line_ss << " • " << elapsed_str;
                line_ss << " • " << score_str;
                line_ss << " • " << leaves_str;

                std::cout << line_ss.str() << "\033[K\n";
                current_rendered_lines++;
            }

            int extra = m_last_rendered_lines - current_rendered_lines;
            for (int i = 0; i < extra; ++i) {
                std::cout << "\033[2K\n";
            }

            if (extra > 0) {
                std::cout << "\033[" << extra << "A";
            }

            m_last_rendered_lines = current_rendered_lines;
            std::cout << std::flush;
        }
    }

    static std::string format_eta(std::chrono::steady_clock::duration elapsed,
                                  SizeType completed,
                                  SizeType total) {
        if (completed == 0) {
            return "ETA: --:--";
        }
        const auto progress =
            static_cast<double>(completed) / static_cast<double>(total);
        if (progress == 0) {
            return "ETA: --:--";
        }
        const auto remaining_time =
            std::chrono::duration_cast<std::chrono::seconds>(
                elapsed * ((1.0 / progress) - 1.0));
        const auto minutes = remaining_time.count() / 60;
        const auto seconds = remaining_time.count() % 60;
        std::stringstream ss;
        ss << "ETA: ";
        if (minutes < 10) {
            ss << "0";
        }
        ss << minutes << ":";
        if (seconds < 10) {
            ss << "0";
        }
        ss << seconds;
        return ss.str();
    }

    static std::string
    format_elapsed(std::chrono::steady_clock::duration elapsed) {
        const auto total_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        const auto minutes = total_seconds / 60;
        const auto seconds = total_seconds % 60;
        std::stringstream ss;
        ss << "Elapsed: ";
        if (minutes < 10) {
            ss << "0";
        }
        ss << minutes << ":";
        if (seconds < 10) {
            ss << "0";
        }
        ss << seconds;
        return ss.str();
    }

    static std::string get_spinner_frame() {
        static int frame = 0;
        return kSpinner[(frame++) % kSpinner.size()];
    }

    static std::string
    get_progress_bar(SizeType completed, SizeType total, int bar_width) {
        std::string bar;
        const auto progress =
            static_cast<double>(completed) / static_cast<double>(total);
        const int filled_width = static_cast<int>(bar_width * progress);

        for (int i = 0; i < filled_width; ++i) {
            bar += "\033[32m█\033[0m"; // Green block
        }
        bool partial_added = false;
        if (filled_width < bar_width && completed > 0) {
            double partial = (bar_width * progress) - filled_width;
            if (partial > 0) {
                partial_added = true;
                if (partial > 0.75) {
                    bar += "\033[32m█\033[0m";
                } else if (partial > 0.50) {
                    bar += "\033[32m▌\033[0m";
                } else if (partial > 0.25) {
                    bar += "\033[32m▎\033[0m";
                } else {
                    bar += "\033[32m░\033[0m";
                }
            }
        }
        for (int i = filled_width + (partial_added ? 1 : 0); i < bar_width;
             ++i) {
            bar += "░"; // Light gray block
        }
        return bar;
    }

    static int get_terminal_width() noexcept {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),
                                       &csbi)) {
            return csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
        return 80;
#else
        struct winsize w{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) { // NOLINT
            return w.ws_col;
        }
        return 80;
#endif
    }
};

} // namespace loki::progress