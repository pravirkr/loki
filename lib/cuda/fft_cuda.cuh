#pragma once

/**
 * @file fft_cuda.cuh
 * @brief Batched real FFTs on the GPU (R2C and C2R) with the portable block
 * FFT (ffa_portable_fft.cuh): no vendor FFT library. Internal.
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

/// Longest real transform the GPU supports. Each row is transformed on
/// chip by one block: a power-of-two length directly, any other length by
/// Bluestein's algorithm through a power-of-two transform of at most 4096
/// points, which fits the 48 KiB of shared memory of every architecture.
inline constexpr SizeType kGpuFFTMaxN = 2048;

/// True if the GPU transforms support length @p n_real (2..kGpuFFTMaxN).
[[nodiscard]] constexpr bool gpu_fft_supported(SizeType n_real) noexcept {
    return n_real >= 2 && n_real <= kGpuFFTMaxN;
}

/// Rejects a Fourier-domain GPU search whose @p nbins the GPU FFTs cannot
/// transform, naming the caller @p what.
inline void check_gpu_fourier_nbins(SizeType nbins, std::string_view what) {
    if (!gpu_fft_supported(nbins)) {
        throw std::invalid_argument(std::format(
            "{}: Fourier-domain folds on the GPU support nbins in [2, {}] "
            "(got {}); lower nbins_max, use time-domain folds or the CPU "
            "backend",
            what, kGpuFFTMaxN, nbins));
    }
}

/**
 * @brief Batched 1D R2C and C2R transforms of rows of length n_real.
 *
 * Every row is transformed by its own block-level FFT, so a row's output
 * does not depend on the batch size or on the other rows. Transforms are
 * out of place and leave their input unchanged. There are no plans: the
 * only state is a Bluestein table per length that needs one (bound to the
 * device passed to the constructor), built by prepare_plans or on first
 * use.
 *
 * Transforms are ordered on the @p stream they are given; one manager may
 * be used from several streams.
 */
class GpuFFTManager {
public:
    explicit GpuFFTManager(int device_id = 0);
    ~GpuFFTManager();

    GpuFFTManager(GpuFFTManager&&) noexcept;
    GpuFFTManager& operator=(GpuFFTManager&&) noexcept;
    GpuFFTManager(const GpuFFTManager&)            = delete;
    GpuFFTManager& operator=(const GpuFFTManager&) = delete;

    /// Builds the tables of each length in @p n_reals (synchronous).
    /// @throws std::invalid_argument for an unsupported length.
    void prepare_plans(std::span<const SizeType> n_reals);

    /// Batched real-to-complex FFT: [batch][n_real] -> [batch][n_real/2+1].
    /// The imaginary parts of the DC and (even n_real) Nyquist bins are 0.
    void rfft_batch(cuda::std::span<const float> real_input,
                    cuda::std::span<ComplexTypeCUDA> complex_output,
                    SizeType batch_size,
                    SizeType n_real,
                    cudaStream_t stream = nullptr);

    /**
     * @brief Batched complex-to-real FFT: [batch][n_real/2+1] ->
     * [batch][n_real]. Applies 1/n_real unless @p normalize is false (the
     * caller then owes the factor). The imaginary parts of the DC and
     * Nyquist bins are ignored, as by every C2R.
     */
    void irfft_batch(cuda::std::span<const ComplexTypeCUDA> complex_input,
                     cuda::std::span<float> real_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream = nullptr,
                     bool normalize      = true);

    [[nodiscard]] bool has_prepared(SizeType n_real) const noexcept;
    /// Number of prepared lengths.
    [[nodiscard]] SizeType n_cached_plans() const noexcept;
    /// Device bytes of the cached tables.
    [[nodiscard]] SizeType table_bytes() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/// Batched R2C through a temporary GpuFFTManager on @p device_id.
void rfft_batch_cuda(cuda::std::span<const float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream = nullptr,
                     int device_id       = 0);

/// Batched C2R (normalized by 1/n_real) through a temporary GpuFFTManager.
void irfft_batch_cuda(cuda::std::span<const ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream = nullptr,
                      int device_id       = 0);

/// GPU storage behind the public FFTManager handle.
class GpuFFTPlans final : public DeviceFFTPlans {
public:
    explicit GpuFFTPlans(int device_id) : manager(device_id) {}

    void prepare_plans(std::span<const SizeType> n_reals) override {
        manager.prepare_plans(n_reals);
    }
    // No plans depend on the batch size: exact plans are the same tables.
    void prepare_exact_plans(std::span<const SizeType> n_reals) override {
        manager.prepare_plans(n_reals);
    }
    [[nodiscard]] bool has_prepared(SizeType n_real) const override {
        return manager.has_prepared(n_real);
    }
    [[nodiscard]] SizeType n_cached_plans() const override {
        return manager.n_cached_plans();
    }

    GpuFFTManager manager;
};

namespace detail {

/// GPU transforms behind @p fft. Throws if @p fft is not a GPU manager.
inline GpuFFTManager& cuda_fft(FFTManager& fft, std::string_view what) {
    auto& impl = loki::detail::HandleAccess::impl(fft);
    if (!impl.device) {
        throw std::invalid_argument(
            std::format("{}: expected a GPU FFTManager", what));
    }
    return static_cast<GpuFFTPlans&>(*impl.device).manager;
}

} // namespace detail

} // namespace loki::math
