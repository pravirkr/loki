#pragma once

/**
 * @file fft_cuda.cuh
 * @brief cuFFT plan cache and batched device transforms. Internal.
 */

#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "loki/common/types.hpp"
#include "loki/utils/fft.hpp"

#include "lib/cuda/types_cuda.cuh"
#include "lib/utils/fft_impl.hpp"

namespace loki::math {

inline constexpr int kCUFFTBatchSizeMax = 65536;

/**
 * @brief Owns cuFFT plans and a shared device work area for batched 1D R2C/C2R.
 *
 * Plans are bound to the CUDA device passed to the constructor.
 *
 * Empty manager (no prepare_plans / prepare_exact_plans): each execute lazily
 * creates and retains plans for the chunk sizes it needs.
 *
 * After prepare_plans: one R2C and one C2R plan are stored per n_real at a
 * workSize-capped max_batch; larger batches are chunked and remainders are
 * cached lazily.
 *
 * After prepare_exact_plans: no plans are pre-built. Each distinct C2R batch
 * size encountered at execute time is planned once and cached (chunked at
 * max_batch). rfft_batch is not supported in this mode. Suitable for fixed
 * n_real with varying batch sizes (e.g. EP pruning).
 *
 * Workspace is allocated once (grown as needed) and shared across all plans.
 * Concurrent execution of two plans that share this work area is undefined;
 * callers must run transforms sequentially (one stream at a time). The
 * stream argument on rfft_batch / irfft_batch orders this FFT against other
 * GPU work on that stream; it is not for overlapping two FFTs.
 *
 * Out-of-place C2R always overwrites the complex input buffer (cuFFT).
 */
class CUFFTManager {
public:
    explicit CUFFTManager(int device_id = 0);
    ~CUFFTManager();

    CUFFTManager(CUFFTManager&&) noexcept;
    CUFFTManager& operator=(CUFFTManager&&) noexcept;
    CUFFTManager(const CUFFTManager&)            = delete;
    CUFFTManager& operator=(const CUFFTManager&) = delete;

    /**
     * @brief Pre-build one R2C and one C2R plan per distinct n_real.
     *
     * @p max_batch is the execute chunk size (default 65536). It may be
     * reduced so cuFFT scratch fits a work-area budget. Re-preparing the
     * same n_real with the same (possibly clamped) max_batch is a no-op.
     * A larger max_batch rebuilds the chunk plans. Shrinking throws.
     * Conflicts with prepare_exact_plans for the same n_real.
     */
    void prepare_plans(std::span<const SizeType> n_reals,
                       SizeType max_batch = kCUFFTBatchSizeMax);

    /**
     * @brief Register @p n_real for lazy exact-batch IRFFT caching.
     *
     * Does not pre-build plans. At irfft_batch execute time each distinct
     * batch size (up to @p max_batch per chunk, possibly reduced so cuFFT
     * scratch fits the work-area budget) is planned once and retained on
     * this manager. rfft_batch is not supported in this mode. Conflicts
     * with prepare_plans for the same n_real.
     */
    void prepare_exact_plans(std::span<const SizeType> n_reals,
                             SizeType max_batch = kCUFFTBatchSizeMax);

    /**
     * @brief Batched real-to-complex FFT. Plans are reused for the lifetime of
     * this manager.
     *
     * @p stream orders this transform against other GPU work on the same
     * stream. Do not use it to overlap two FFTs on this manager: all plans
     * share one work area, so concurrent execution is undefined.
     */
    void rfft_batch(cuda::std::span<float> real_input,
                    cuda::std::span<ComplexTypeCUDA> complex_output,
                    SizeType batch_size,
                    SizeType n_real,
                    cudaStream_t stream = nullptr);

    /**
     * @brief Batched complex-to-real FFT. Overwrites @p complex_input.
     * Applies the 1/n_real normalization that cuFFT omits on C2R, unless
     * @p normalize is false; the caller then owes the 1/n_real factor.
     *
     * @p stream orders this transform against other GPU work on the same
     * stream. Do not use it to overlap two FFTs on this manager: all plans
     * share one work area, so concurrent execution is undefined.
     */
    void irfft_batch(cuda::std::span<ComplexTypeCUDA> complex_input,
                     cuda::std::span<float> real_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream = nullptr,
                     bool normalize      = true);

    [[nodiscard]] bool has_prepared(SizeType n_real) const noexcept;
    [[nodiscard]] SizeType n_cached_plans() const noexcept;
    [[nodiscard]] SizeType work_area_bytes() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Batched real-to-complex 1D FFT via an ephemeral CUFFTManager.
 *
 * Constructs an empty manager on @p device_id, runs one rfft_batch, and
 * destroys plans and the shared work area on return. Chunks batches larger
 * than the workSize-capped max_batch (default 65536).
 *
 * @param real_input Real input array [batch_size * n_real]
 * @param complex_output Complex output array [batch_size * (n_real/2+1)]
 * @param batch_size Number of transforms
 * @param n_real Length of each real transform
 * @param stream Orders this FFT against other GPU work on the same stream
 * @param device_id CUDA device that owns the ephemeral plans
 */
void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream = nullptr,
                     int device_id       = 0);

/**
 * @brief Batched complex-to-real 1D FFT via an ephemeral CUFFTManager.
 *
 * Same lifetime as rfft_batch_cuda. Applies the 1/n_real normalization that
 * cuFFT omits on C2R. Overwrites @p complex_input.
 *
 * @param complex_input Complex input array [batch_size * (n_real/2+1)]
 * @param real_output Real output array [batch_size * n_real]
 * @param batch_size Number of transforms
 * @param n_real Length of each real transform
 * @param stream Orders this FFT against other GPU work on the same stream
 * @param device_id CUDA device that owns the ephemeral plans
 */
void irfft_batch_cuda(cuda::std::span<ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream = nullptr,
                      int device_id       = 0);

/// GPU storage behind the public FFTManager handle.
class CUFFTPlans final : public DeviceFFTPlans {
public:
    explicit CUFFTPlans(int device_id) : manager(device_id) {}

    void prepare_plans(std::span<const SizeType> n_reals) override {
        manager.prepare_plans(n_reals);
    }
    void prepare_exact_plans(std::span<const SizeType> n_reals) override {
        manager.prepare_exact_plans(n_reals);
    }
    [[nodiscard]] bool has_prepared(SizeType n_real) const override {
        return manager.has_prepared(n_real);
    }
    [[nodiscard]] SizeType n_cached_plans() const override {
        return manager.n_cached_plans();
    }

    CUFFTManager manager;
};

namespace detail {

/// cuFFT plan cache behind @p fft. Throws if @p fft is not a GPU manager.
inline CUFFTManager& cuda_fft(FFTManager& fft, std::string_view what) {
    auto& impl = loki::detail::HandleAccess::impl(fft);
    if (!impl.device) {
        throw std::invalid_argument(
            std::format("{}: expected a GPU FFTManager", what));
    }
    return static_cast<CUFFTPlans&>(*impl.device).manager;
}

} // namespace detail

} // namespace loki::math
