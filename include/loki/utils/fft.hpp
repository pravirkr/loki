#pragma once

#include <memory>
#include <span>

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

#include "loki/common/types.hpp"

namespace loki::math {

inline constexpr int kFFTBatchSizeMax = 16384;

/**
 * @brief 2D FFT for circular convolution
 *
 * RAII wrapper for 2D transforms used in convolution operations.
 */
class FFT2D {
public:
    FFT2D(SizeType n1x, SizeType n2x, SizeType ny);
    ~FFT2D();

    FFT2D(const FFT2D&)            = delete;
    FFT2D& operator=(const FFT2D&) = delete;
    FFT2D(FFT2D&&)                 = delete;
    FFT2D& operator=(FFT2D&&)      = delete;

    void circular_convolve(std::span<float> n1,
                           std::span<float> n2,
                           std::span<float> out);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Owns optional FFTW plan caches and runs batched R2C / C2R 1D FFTs.
 *
 * Empty manager (default): each rfft_batch / irfft_batch call creates
 * 1–3 ephemeral plans, executes with OpenMP over balanced batch slices,
 * and destroys the plans before returning. Optimal for a one-shot FFA.
 *
 * After prepare_plans: a power-of-two howmany ladder (1, 2, …, 16384) is
 * stored per n_real. Subsequent calls reuse those plans and decompose
 * each thread's slice via its binary representation.
 *
 * After prepare_exact_plans: no ladder is built. Each distinct howmany
 * encountered at execute time is planned once and cached lazily (suitable
 * for fixed n_real with varying batch sizes, e.g. EP pruning).
 *
 * Planner create/destroy is serialized via a library-wide mutex.
 * Execute of a cached plan from multiple OpenMP threads is safe as long
 * as input/output slices do not overlap.
 */
class FFTWManager {
public:
    FFTWManager();
    ~FFTWManager();

    FFTWManager(FFTWManager&&) noexcept;
    FFTWManager& operator=(FFTWManager&&) noexcept;
    FFTWManager(const FFTWManager&)            = delete;
    FFTWManager& operator=(const FFTWManager&) = delete;

    /**
     * @brief Pre-build R2C and C2R plans for each distinct n_real.
     *
     * howmany values are 1, 2, 4, …, @p max_howmany. @p max_howmany must
     * be a positive power of two (default 16384). If @p n_real is already
     * prepared with the same @p max_howmany, the call is a no-op. If
     * @p max_howmany is larger than the stored ceiling, missing ladder
     * rungs are appended. Shrinking @p max_howmany throws.
     */
    void prepare_plans(std::span<const SizeType> n_reals,
                       SizeType max_howmany = kFFTBatchSizeMax);

    /**
     * @brief Register @p n_real for lazy exact-howmany IRFFT caching.
     *
     * Does not pre-build plans. At irfft_batch execute time each distinct
     * howmany value (up to @p max_howmany per chunk) is planned once and
     * retained on this manager. rfft_batch is not supported in this mode.
     * Use with @c nthreads=1 for workloads such as EP pruning where batch
     * size varies but n_real is fixed.
     */
    void prepare_exact_plans(std::span<const SizeType> n_reals,
                             SizeType max_howmany = kFFTBatchSizeMax);

    /**
     * @brief Batched real-to-complex 1D FFT with optional plan cache.
     *
     * 1D R2C preserves @p real_input by default.
     */
    void rfft_batch(std::span<float> real_input,
                    std::span<ComplexType> complex_output,
                    SizeType batch_size,
                    SizeType n_real,
                    int nthreads = 1);

    /**
     * @brief Batched complex-to-real 1D FFT with optional plan cache.
     *
     * On CPU, out-of-place FFTW C2R may overwrite @p complex_input; callers
     * that need the spectrum after the transform must copy first. Applies the
     * 1/n_real normalization that FFTW omits on C2R.
     */
    void irfft_batch(std::span<ComplexType> complex_input,
                     std::span<float> real_output,
                     SizeType batch_size,
                     SizeType n_real,
                     int nthreads = 1);

    [[nodiscard]] bool has_prepared(SizeType n_real) const noexcept;
    [[nodiscard]] SizeType n_cached_plans() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Helper functions for convenience

/**
 * @brief Batched real-to-complex 1D FFT via an ephemeral FFTWManager.
 *
 * Constructs an empty manager, runs one rfft_batch, and destroys plans on
 * return. Splits @p batch_size evenly across @p nthreads, then caps each
 * thread's slice into howmany ≤ 16384.
 *
 * @param real_input Real input array [batch_size * n_real]
 * @param complex_output Complex output array [batch_size * (n_real/2+1)]
 * @param batch_size Number of transforms (any positive integer)
 * @param n_real Length of each real transform (typically 32–1024)
 * @param nthreads Number of OpenMP threads (default: 1)
 */
void rfft_batch(std::span<float> real_input,
                std::span<ComplexType> complex_output,
                SizeType batch_size,
                SizeType n_real,
                int nthreads = 1);

/**
 * @brief Batched complex-to-real 1D FFT via an ephemeral FFTWManager.
 *
 * Same scheduling as rfft_batch. Applies the 1/n_real normalization that
 * FFTW omits on C2R. May overwrite @p complex_input; copy first if the
 * spectrum is needed afterward.
 *
 * @param complex_input Complex input array [batch_size * (n_real/2+1)]
 * @param real_output Real output array [batch_size * n_real]
 * @param batch_size Number of transforms (any positive integer)
 * @param n_real Length of each real transform (typically 32–1024)
 * @param nthreads Number of OpenMP threads (default: 1)
 */
void irfft_batch(std::span<ComplexType> complex_input,
                 std::span<float> real_output,
                 SizeType batch_size,
                 SizeType n_real,
                 int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA

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
     * Applies the 1/n_real normalization that cuFFT omits on C2R.
     *
     * @p stream orders this transform against other GPU work on the same
     * stream. Do not use it to overlap two FFTs on this manager: all plans
     * share one work area, so concurrent execution is undefined.
     */
    void irfft_batch(cuda::std::span<ComplexTypeCUDA> complex_input,
                     cuda::std::span<float> real_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream = nullptr);

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

#endif // LOKI_ENABLE_CUDA

} // namespace loki::math