#pragma once

/**
 * @file fft_impl.hpp
 * @brief FFTW plan caches, batched R2C/C2R transforms and FFT2D (host).
 * Internal; the public handle is loki::math::FFTManager.
 */

#include <memory>
#include <span>
#include <string_view>

#include "common/dispatch.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/fft.hpp"

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

/// GPU plan cache behind an FFTManager handle; implemented in lib/cuda/.
class DeviceFFTPlans : public loki::detail::DeviceStorage {
public:
    virtual void prepare_plans(std::span<const SizeType> n_reals)       = 0;
    virtual void prepare_exact_plans(std::span<const SizeType> n_reals) = 0;
    [[nodiscard]] virtual bool has_prepared(SizeType n_real) const      = 0;
    [[nodiscard]] virtual SizeType n_cached_plans() const               = 0;
};

// Public handle storage. Exactly one of `cpu` / `device` is set, matching
// `exec.backend`.
class FFTManager::Impl {
public:
    Exec exec;
    std::unique_ptr<FFTWManager> cpu;
    std::unique_ptr<DeviceFFTPlans> device;
};

namespace detail {

/// FFTW plan cache behind @p fft. Throws if @p fft is empty or not CPU.
FFTWManager& cpu_fft(FFTManager& fft, std::string_view what);

/// GPU storage factory, defined in cuda/fft_cuda.cu.
std::unique_ptr<DeviceFFTPlans> make_fft_manager_gpu(int device_id);

} // namespace detail

} // namespace loki::math
