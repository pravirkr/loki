#pragma once

#include <chrono>
#include <string_view>
#include <utility>

#include <spdlog/spdlog.h>

namespace loki::timing {

namespace detail {

template <typename Fn> void swallow_exceptions(Fn&& fn) noexcept {
    try {
        std::forward<Fn>(fn)();
    } catch (...) { // NOLINT(bugprone-empty-catch)
    }
}

} // namespace detail

// ScopeTimer class to measure and log the time taken by a block of code
class ScopeTimer {
public:
    explicit ScopeTimer(std::string_view label)
        : m_label(label),
          m_start(std::chrono::steady_clock::now()) {
        if (s_enabled) {
            spdlog::info("{} started", m_label);
        }
    }

    // Destructor ends the timer and logs elapsed time
    ~ScopeTimer() noexcept {
        if (!s_enabled) {
            return;
        }

        detail::swallow_exceptions([&] {
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      m_start);
            spdlog::info("{} took {} ms", m_label, elapsed.count());
        });
    }

    ScopeTimer(const ScopeTimer&)            = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;
    ScopeTimer(ScopeTimer&&)                 = delete;
    ScopeTimer& operator=(ScopeTimer&&)      = delete;

    // Enable or disable timing globally
    static void set_enabled(bool enabled) { s_enabled = enabled; }

    // Check if timing is currently enabled
    static bool is_enabled() { return s_enabled; }

private:
    std::string_view m_label;
    std::chrono::steady_clock::time_point m_start;
    inline static bool s_enabled = true; // default: enabled
};

/**
 * @brief High-resolution timer for measuring code sections.
 *
 * Uses std::chrono::steady_clock for monotonic, high-resolution timing.
 * Not thread-safe - each thread should have its own instance.
 */
class SimpleTimer {
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

public:
    SimpleTimer() = default;

    /**
     * @brief Starts the timer.
     */
    void start() noexcept { m_start = Clock::now(); }

    /**
     * @brief Stops the timer and returns elapsed time in seconds.
     * @return Elapsed time in seconds since start()
     */
    [[nodiscard]] float stop() const {
        const auto end = Clock::now();
        return std::chrono::duration<float>(end - m_start).count();
    }

private:
    TimePoint m_start{Clock::now()};
};

// ScopedLogLevel class to temporarily set the log level
class ScopedLogLevel {
public:
    explicit ScopedLogLevel(bool quiet)
        : m_old_level(spdlog::get_level()),
          m_quiet(quiet) {
        if (m_quiet) {
            spdlog::set_level(spdlog::level::err);
        }
    }
    ~ScopedLogLevel() noexcept {
        if (m_quiet) {
            detail::swallow_exceptions([&] { spdlog::set_level(m_old_level); });
        }
    }

    ScopedLogLevel(const ScopedLogLevel&)            = delete;
    ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;
    ScopedLogLevel(ScopedLogLevel&&)                 = delete;
    ScopedLogLevel& operator=(ScopedLogLevel&&)      = delete;

private:
    spdlog::level::level_enum m_old_level;
    bool m_quiet;
};

} // namespace loki::timing
