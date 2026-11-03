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

namespace detail {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): per-thread state
inline thread_local int t_quiet_depth = 0;
} // namespace detail

// QuietScope marks the calling thread quiet while it lives (the `quiet`
// argument of the compute_ffa functions): engine detail lines that check
// quiet() stay silent. Thread-local, so the log level, the sinks and other
// threads' logging are untouched, and concurrent calls cannot leave the
// level changed. Scopes nest.
class QuietScope {
public:
    explicit QuietScope(bool quiet) : m_quiet(quiet) {
        if (m_quiet) {
            ++detail::t_quiet_depth;
        }
    }
    ~QuietScope() noexcept {
        if (m_quiet) {
            --detail::t_quiet_depth;
        }
    }

    QuietScope(const QuietScope&)            = delete;
    QuietScope& operator=(const QuietScope&) = delete;
    QuietScope(QuietScope&&)                 = delete;
    QuietScope& operator=(QuietScope&&)      = delete;

private:
    bool m_quiet;
};

// Whether a QuietScope is alive on this thread.
[[nodiscard]] inline bool quiet() noexcept { return detail::t_quiet_depth > 0; }

} // namespace loki::timing
