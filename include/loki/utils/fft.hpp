#pragma once

#include <memory>
#include <span>

#ifdef LOKI_ENABLE_CUDA
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>
#endif // LOKI_ENABLE_CUDA

#include <fftw3.h>

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
    SizeType m_n1x;
    SizeType m_n2x;
    SizeType m_ny;
    SizeType m_fft_size;

    fftwf_complex* m_n1_fft;
    fftwf_complex* m_n2_fft;
    fftwf_complex* m_n1n2_fft;
    fftwf_plan m_plan_forward;
    fftwf_plan m_plan_inverse;
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

    void rfft_batch(std::span<const float> real_input,
                    std::span<ComplexType> complex_output,
                    SizeType batch_size,
                    SizeType n_real,
                    int nthreads = 1);

    /**
     * @brief Batched complex-to-real FFT with optional plan cache.
     *
     * 1D C2R defaults to destroying its input; this path plans with
     * FFTW_PRESERVE_INPUT so @p complex_input is left unchanged. Applies
     * the 1/n_real normalization that FFTW omits on C2R.
     */
    void irfft_batch(std::span<const ComplexType> complex_input,
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
 * @brief Batched real-to-complex FFT, self-contained and OpenMP-parallel.
 *
 * Splits @p batch_size evenly across @p nthreads, then caps each thread's
 * slice into howmany ≤ 16384. Creates one single-threaded FFTW plan per
 * distinct howmany, executes them in an OpenMP loop (same plan reused
 * concurrently on non-overlapping slices), and destroys every plan before
 * returning. Does not use a process-wide plan cache. Planner create/destroy
 * is serialized via a library-wide mutex. Execute is not locked.
 *
 * @param real_input Real input array [batch_size * n_real]
 * @param complex_output Complex output array [batch_size * (n_real/2+1)]
 * @param batch_size Number of transforms (any positive integer)
 * @param n_real Length of each real transform (typically 32–1024)
 * @param nthreads Number of OpenMP threads (default: 1)
 */
void rfft_batch(std::span<const float> real_input,
                std::span<ComplexType> complex_output,
                SizeType batch_size,
                SizeType n_real,
                int nthreads = 1);

/**
 * @brief Batched complex-to-real FFT, self-contained and OpenMP-parallel.
 *
 * Same scheduling as rfft_batch: splits @p batch_size evenly across
 * @p nthreads, caps each slice at howmany ≤ 16384, creates one
 * single-threaded FFTW plan per distinct howmany, executes in OpenMP,
 * and destroys every plan before returning. Applies the 1/n_real
 * normalization that FFTW omits on C2R. Plans with FFTW_PRESERVE_INPUT
 * so 1D C2R leaves @p complex_input unchanged.
 *
 * @param complex_input Complex input array [batch_size * (n_real/2+1)]
 * @param real_output Real output array [batch_size * n_real]
 * @param batch_size Number of transforms (any positive integer)
 * @param n_real Length of each real transform (typically 32–1024)
 * @param nthreads Number of OpenMP threads (default: 1)
 */
void irfft_batch(std::span<const ComplexType> complex_input,
                 std::span<float> real_output,
                 SizeType batch_size,
                 SizeType n_real,
                 int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA

struct PlanKeyDevice {
    int n_real;
    int batch_size;
    cudaStream_t stream;

    bool operator==(const PlanKeyDevice& other) const {
        return n_real == other.n_real && batch_size == other.batch_size &&
               stream == other.stream;
    }
};

struct PlanKeyHashDevice {
    SizeType operator()(const PlanKeyDevice& k) const {
        return std::hash<int>{}(k.n_real) ^
               (std::hash<int>{}(k.batch_size) << 1U) ^
               (std::hash<std::uintptr_t>{}(
                    reinterpret_cast<std::uintptr_t>(k.stream))
                << 2U);
    }
};

class IrfftExecutorCUDA {
public:
    explicit IrfftExecutorCUDA(int n_real);
    ~IrfftExecutorCUDA();

    IrfftExecutorCUDA(const IrfftExecutorCUDA&)            = delete;
    IrfftExecutorCUDA& operator=(const IrfftExecutorCUDA&) = delete;
    IrfftExecutorCUDA(IrfftExecutorCUDA&&)                 = delete;
    IrfftExecutorCUDA& operator=(IrfftExecutorCUDA&&)      = delete;

    void execute(cuda::std::span<const ComplexTypeCUDA> complex_input,
                 cuda::std::span<float> real_output,
                 int batch_size,
                 cudaStream_t stream);

private:
    int m_n_real;
    int m_n_complex;

    std::unordered_map<PlanKeyDevice, cufftHandle, PlanKeyHashDevice>
        m_plan_cache;
    std::mutex m_mutex;
    cufftHandle get_or_create_plan(int batch_size, cudaStream_t stream);
};

void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream = nullptr);

void irfft_batch_cuda(cuda::std::span<ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream = nullptr);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::math