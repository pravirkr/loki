#pragma once

/**
 * @file interrupt.hpp
 * @brief Stopping long Loki calls cleanly on Ctrl-C.
 *
 * While an InterruptGuard is alive, SIGINT (and optionally SIGTERM) does not
 * kill the process: it asks the running Loki calls to stop. They throw
 * loki::Interrupted at their next checkpoint (between DTS stages, FFA
 * levels, sweep chunks and pruning iterations), so destructors run, files
 * are closed and the terminal is restored. Without a guard, signals keep
 * their usual behaviour and the checkpoints never throw.
 *
 * The Python module installs a guard around its long calls (from the main
 * thread only) and turns Interrupted into KeyboardInterrupt.
 */

#include <stdexcept>

namespace loki {

/// @brief Thrown by a Loki call stopped by an interrupt request.
class Interrupted : public std::runtime_error {
public:
    Interrupted() : std::runtime_error("interrupted") {}
};

/**
 * @brief Routes SIGINT (and SIGTERM with @p handle_terminate) to an
 * interrupt request for its lifetime.
 *
 * The first signal sets the request. Another one while the request is
 * pending terminates the process if the previous handler was the default
 * one (a second Ctrl-C at the command line), and is ignored otherwise.
 * Guards nest; the outermost one installs the handlers, and on destruction
 * restores the previous handlers and clears the request. A no-op on
 * platforms without POSIX signals.
 */
class InterruptGuard {
public:
    explicit InterruptGuard(bool handle_terminate = false);
    ~InterruptGuard();
    InterruptGuard(const InterruptGuard&)            = delete;
    InterruptGuard& operator=(const InterruptGuard&) = delete;
    InterruptGuard(InterruptGuard&&)                 = delete;
    InterruptGuard& operator=(InterruptGuard&&)      = delete;
};

/// @brief Whether an interrupt was requested and not yet cleared.
[[nodiscard]] bool interrupt_requested() noexcept;

/// @brief The signal behind the pending request (0 if none, or if it was
/// requested with request_interrupt()).
[[nodiscard]] int interrupt_signal() noexcept;

/// @brief Request an interrupt without a signal (e.g. from a GUI thread).
/// Cleared when the outermost InterruptGuard ends, or by clear_interrupt().
void request_interrupt() noexcept;

/// @brief Clear a pending request.
void clear_interrupt() noexcept;

/// @brief Checkpoint: throws Interrupted if a request is pending. Cheap (one
/// atomic load). Never call it inside an OpenMP parallel region.
void check_interrupt();

} // namespace loki
