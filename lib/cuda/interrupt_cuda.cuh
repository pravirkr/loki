#pragma once

// Device-side companion of lib/detail/interrupt.hpp: checkpoints for host
// loops that drive a CUDA stream.

#include <cuda_runtime.h>

#include "loki/common/interrupt.hpp"

#include "lib/detail/interrupt.hpp"

namespace loki::interrupt {

/// Interrupt checkpoint for host loops that queue GPU work on @p stream. On a
/// pending request it waits for the queued work, so no kernel still uses the
/// buffers the unwinding frees, and throws loki::Interrupted. One atomic load
/// otherwise. Call it on the host thread that drives the stream.
inline void check(cudaStream_t stream) {
    if (requested()) {
        (void)cudaStreamSynchronize(stream);
        loki::check_interrupt();
    }
}

} // namespace loki::interrupt
