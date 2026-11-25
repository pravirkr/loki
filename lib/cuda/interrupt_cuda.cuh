#pragma once

// Device-side companion of lib/detail/interrupt.hpp: checkpoints for host
// loops that drive a CUDA stream.

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "loki/common/interrupt.hpp"

#include "lib/detail/interrupt.hpp"

namespace loki::interrupt {

/// Interrupt checkpoint for host loops that queue GPU work on @p stream. On a
/// pending request it waits for the queued work, so no kernel still uses the
/// buffers the unwinding frees, and throws loki::Interrupted. One atomic load
/// otherwise. Call it on the host thread that drives the stream. An error the
/// queued work reports is logged and cleared, so the interrupt is what
/// propagates and the error is not reported later by an unrelated call (a
/// sticky error stays, and the next CUDA call reports it).
inline void check(cudaStream_t stream) {
    if (requested()) {
        const cudaError_t error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            (void)cudaGetLastError();
            spdlog::warn("CUDA error while stopping on an interrupt: {}",
                         cudaGetErrorString(error));
        }
        loki::check_interrupt();
    }
}

} // namespace loki::interrupt
