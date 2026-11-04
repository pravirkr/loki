#include "loki/common/interrupt.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>

#ifndef _WIN32
#include <csignal>

#include <signal.h> // NOLINT(modernize-deprecated-headers): POSIX sigaction
#endif

#include "lib/detail/interrupt.hpp"

namespace loki {

namespace interrupt::detail {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): shared with a signal handler
std::atomic<bool> g_requested{false};
} // namespace interrupt::detail

namespace {

static_assert(std::atomic<bool>::is_always_lock_free &&
                  std::atomic<int>::is_always_lock_free,
              "the signal handler needs lock-free atomics");

struct InterruptState {
    std::atomic<int> signal{0};
    // Installed handlers: written by the outermost guard before it installs
    // them, read by the signal handler.
    std::mutex mutex;
    int depth{0};
#ifndef _WIN32
    static constexpr std::array<int, 2> kSignals = {SIGINT, SIGTERM};
    std::array<struct sigaction, 2> previous{};
    std::array<bool, 2> installed{};
#endif
};

InterruptState& state() {
    static InterruptState s;
    return s;
}

#ifndef _WIN32
void on_signal(int sig) {
    auto& s         = state();
    auto& requested = interrupt::detail::g_requested;
    if (!requested.load()) {
        s.signal.store(sig);
        requested.store(true);
        return;
    }
    // A second signal while the request is pending: a previous default
    // handler terminates the process, as it would have without the guard.
    for (std::size_t i = 0; i < InterruptState::kSignals.size(); ++i) {
        if (InterruptState::kSignals[i] == sig && s.installed[i] &&
            s.previous[i].sa_handler == SIG_DFL) {
            sigaction(sig, &s.previous[i], nullptr);
            std::raise(sig);
        }
    }
}
#endif

} // namespace

InterruptGuard::InterruptGuard(bool handle_terminate) {
    auto& s = state();
    const std::scoped_lock lock(s.mutex);
    if (s.depth++ > 0) {
        return;
    }
    clear_interrupt();
#ifndef _WIN32
    struct sigaction action{};
    action.sa_handler = &on_signal;
    action.sa_flags   = SA_RESTART;
    sigemptyset(&action.sa_mask);
    for (std::size_t i = 0; i < InterruptState::kSignals.size(); ++i) {
        const int sig = InterruptState::kSignals[i];
        if (sig == SIGTERM && !handle_terminate) {
            continue;
        }
        s.installed[i] = sigaction(sig, &action, &s.previous[i]) == 0;
    }
#else
    (void)handle_terminate;
#endif
}

InterruptGuard::~InterruptGuard() {
    auto& s = state();
    const std::scoped_lock lock(s.mutex);
    if (--s.depth > 0) {
        return;
    }
#ifndef _WIN32
    for (std::size_t i = 0; i < InterruptState::kSignals.size(); ++i) {
        if (s.installed[i]) {
            sigaction(InterruptState::kSignals[i], &s.previous[i], nullptr);
            s.installed[i] = false;
        }
    }
#endif
    clear_interrupt();
}

bool interrupt_requested() noexcept {
    return interrupt::detail::g_requested.load();
}

int interrupt_signal() noexcept {
    return interrupt_requested() ? state().signal.load() : 0;
}

void request_interrupt() noexcept {
    auto& s = state();
    s.signal.store(0);
    interrupt::detail::g_requested.store(true);
}

void clear_interrupt() noexcept {
    interrupt::detail::g_requested.store(false);
    state().signal.store(0);
}

void check_interrupt() {
    if (interrupt_requested()) {
        throw Interrupted();
    }
}

} // namespace loki
