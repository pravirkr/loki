#pragma once

/**
 * @file fft_long_cuda.cuh
 * @brief Whole-series real FFTs on the GPU with cuFFT. Internal.
 *
 * Preprocessing transforms a whole series (up to 2^27 samples and beyond) a
 * few times, from the host. That is cuFFT's use case; the search kernels
 * keep the portable block FFT (fft_cuda.cuh), which fuses into device code.
 * This header exposes no cuFFT type: lib/cuda/fft_long_cuda.cu is the only
 * file that includes <cufft.h> (docs/fft.md, "Whole-series transforms").
 */

#include <memory>

#include <cuda_runtime.h>

#include "loki/common/types.hpp"

#include "lib/cuda/types_cuda.cuh"

namespace loki::math {

/**
 * @brief One real transform pair (R2C and C2R) of length n on one device.
 *
 * Both plans share one work area owned by this object, so the caller's
 * memory accounting sees it (workspace_bytes()). Transforms run in order on
 * the stream given at construction; two transforms of one object must not
 * overlap. Any n >= 2 (cuFFT picks the algorithm); sizes are 64-bit.
 */
class LongRealFFT {
public:
    /// Plans both directions for length @p n on @p device_id, ordered on
    /// @p stream.
    LongRealFFT(SizeType n, int device_id, cudaStream_t stream);
    ~LongRealFFT();

    LongRealFFT(LongRealFFT&&) noexcept;
    LongRealFFT& operator=(LongRealFFT&&) noexcept;
    LongRealFFT(const LongRealFFT&)            = delete;
    LongRealFFT& operator=(const LongRealFFT&) = delete;

    /// Work area of the plans for length @p n, in bytes, on the current
    /// device (no plan is kept).
    [[nodiscard]] static SizeType workspace_bytes(SizeType n);

    [[nodiscard]] SizeType size() const noexcept;

    /// Real-to-complex: @p x_d[n] -> @p spec_d[n / 2 + 1]. @p x_d is not
    /// modified.
    void rfft(const float* x_d, ComplexTypeCUDA* spec_d);

    /// Complex-to-real: @p spec_d[n / 2 + 1] -> @p x_d[n], scaled by 1 / n
    /// (the host convention, CpuFFT::irfft_batch). Overwrites @p spec_d.
    void irfft(ComplexTypeCUDA* spec_d, float* x_d);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::math
