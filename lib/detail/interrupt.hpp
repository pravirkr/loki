#pragma once

#include <atomic>

namespace loki::interrupt {

namespace detail {
// The flag behind loki::interrupt_requested(), set by the signal handler of
// loki::InterruptGuard (lib/common/interrupt.cpp).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): shared with a signal handler
extern std::atomic<bool> g_requested;
} // namespace detail

/// Inline poll for hot OpenMP loops: skip the remaining iterations when it
/// is set, then call loki::check_interrupt() after the parallel region (a
/// throw inside the region would terminate). One relaxed load.
inline bool requested() noexcept {
    return detail::g_requested.load(std::memory_order_relaxed);
}

} // namespace loki::interrupt
