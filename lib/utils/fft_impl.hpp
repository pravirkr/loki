#pragma once

/**
 * @file fft_impl.hpp
 * @brief Host batched R2C/C2R transforms and circular convolution.
 * Internal; the public handle is loki::math::FFTManager.
 */

#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string_view>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/fft.hpp"

#include "lib/common/dispatch.hpp"

namespace loki::math {

/// Longest row the lane FFT handles; longer rows go to pocketfft.
inline constexpr SizeType kLaneFFTMaxSize = 1024;

/**
 * @brief Per-row circular convolution of two sets of real rows.
 *
 * out[(i n2x + j) ny + k] = sum_m n1[i ny + m] n2[j ny + (k - m) mod ny] for
 * every row i < n1x of n1 and j < n2x of n2.
 */
class CircularConvolver {
public:
    CircularConvolver(SizeType n1x, SizeType n2x, SizeType ny);

    void convolve(std::span<const float> n1,
                  std::span<const float> n2,
                  std::span<float> out);

private:
    SizeType m_n1x;
    SizeType m_n2x;
    SizeType m_ny;
    std::vector<ComplexType> m_n1_fft;
    std::vector<ComplexType> m_n2_fft;
    std::vector<ComplexType> m_product;
};

/**
 * @brief Batched real-to-complex and complex-to-real 1D FFTs on the host.
 *
 * Rows of length 2..kLaneFFTMaxSize use the lane FFT (lib/cpu/lane_fft.hpp):
 * rows are transformed in pairs (2p, 2p + 1), two real rows per complex
 * transform, so a row's float rounding depends on its partner row. Other
 * lengths (e.g. the whole-series transforms of preprocessing) use pocketfft.
 * Transform tables are immutable and cached per length for the process; a
 * manager only records which lengths it prepared. The transforms never
 * modify their input and may run concurrently on one manager.
 */
class CpuFFT {
public:
    CpuFFT()  = default;
    ~CpuFFT() = default;
    // Movable for containers; the mutex is not moved (moves happen during
    // setup, never concurrently with transforms).
    CpuFFT(CpuFFT&& other) noexcept;
    CpuFFT& operator=(CpuFFT&& other) noexcept;
    CpuFFT(const CpuFFT&)            = delete;
    CpuFFT& operator=(const CpuFFT&) = delete;

    /// Register @p n_reals and build their transform tables now.
    void prepare_plans(std::span<const SizeType> n_reals);

    /// Register @p n_reals; their tables are built on first use.
    void prepare_exact_plans(std::span<const SizeType> n_reals);

    /// Batched real-to-complex FFT: `[batch_size][n_real]` reals into
    /// `[batch_size][n_real / 2 + 1]` complex bins.
    void rfft_batch(std::span<const float> real_input,
                    std::span<ComplexType> complex_output,
                    SizeType batch_size,
                    SizeType n_real,
                    int nthreads = 1);

    /// Batched complex-to-real FFT, normalised by 1 / n_real. The imaginary
    /// parts of the DC and (even n_real) Nyquist bins are ignored.
    void irfft_batch(std::span<const ComplexType> complex_input,
                     std::span<float> real_output,
                     SizeType batch_size,
                     SizeType n_real,
                     int nthreads = 1);

    [[nodiscard]] bool has_prepared(SizeType n_real) const;
    /// Number of lengths whose tables this manager has built.
    [[nodiscard]] SizeType n_cached_plans() const;

private:
    mutable std::mutex m_mutex;
    std::set<SizeType> m_registered;
    std::set<SizeType> m_built;

    void note_built(SizeType n_real);
};

/// Batched real-to-complex 1D FFT (see CpuFFT::rfft_batch).
void rfft_batch(std::span<const float> real_input,
                std::span<ComplexType> complex_output,
                SizeType batch_size,
                SizeType n_real,
                int nthreads = 1);

/// Batched complex-to-real 1D FFT, normalised (see CpuFFT::irfft_batch).
void irfft_batch(std::span<const ComplexType> complex_input,
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
    std::unique_ptr<CpuFFT> cpu;
    std::unique_ptr<DeviceFFTPlans> device;
};

namespace detail {

/// Host FFT behind @p fft. Throws if @p fft is empty or not CPU.
CpuFFT& cpu_fft(FFTManager& fft, std::string_view what);

/// GPU storage factory, defined in cuda/fft_cuda.cu.
std::unique_ptr<DeviceFFTPlans> make_fft_manager_gpu(int device_id);

} // namespace detail

} // namespace loki::math
