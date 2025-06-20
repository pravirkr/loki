#pragma once

#include <chrono>
#include <spdlog/spdlog.h>
#include <string_view>

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
    ~ScopeTimer() {
        if (!s_enabled) {
            return;
        }

        auto end     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - m_start);
        spdlog::info("{} took {} ms", m_label, elapsed.count());
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
