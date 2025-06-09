#pragma once

#include <cstddef>
#include <memory>
#include <span>

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

#include <fftw3.h>

#include "loki/common/types.hpp"

namespace loki::utils {

class FFT2D {
public:
    FFT2D(size_t n1x, size_t n2x, size_t ny);
    ~FFT2D();

    void circular_convolve(std::span<float> n1,
                           std::span<float> n2,
                           std::span<float> out);

private:
    size_t m_n1x;
    size_t m_n2x;
    size_t m_ny;

    size_t m_fft_size;
    fftwf_complex* m_n1_fft;
    fftwf_complex* m_n2_fft;
    fftwf_complex* m_n1n2_fft;
    fftwf_plan m_plan_forward;
    fftwf_plan m_plan_inverse;
};

void ensure_fftw_threading(int nthreads = 1);

void rfft_batch(std::span<float> real_input,
                std::span<ComplexType> complex_output,
                int batch_size,
                int n_real,
                int nthreads = 1);

void irfft_batch(std::span<ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size,
                 int n_real,
                 int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA
/**
 * @brief Manages FFT plans and execution for CUDA (cuFFT).
 *
 * Encapsulates the creation and management of FFT plans and provides
 * methods for forward and backward transforms. Performs the FFT using
 * cuFFT.
 */
class FFTManagerCUDA {
public:
    /**
     * @brief Construct for CUDA backend using cuFFT.
     * @param nfft Size of the FFT.
     * @param nsub Number of subbands (for potential batching or planning).
     * @param nbin Number of bins (for potential batching or planning).
     * @param mbin Another dimension (for potential batching or planning).
     * @param nchan Number of channels (for potential batching or planning).
     * @param device_id CUDA device ID.
     */
    FFTManagerCUDA(
        int nfft, int nsub, int nbin, int mbin, int nchan, int device_id = 0);

    ~FFTManagerCUDA();
    FFTManagerCUDA(FFTManagerCUDA&&) noexcept;
    FFTManagerCUDA& operator=(FFTManagerCUDA&&) noexcept;
    FFTManagerCUDA(const FFTManagerCUDA&)            = delete;
    FFTManagerCUDA& operator=(const FFTManagerCUDA&) = delete;

    void rfft(cuda::std::span<float> real_input,
              cuda::std::span<ComplexTypeCUDA> complex_output,
              cudaStream_t stream = nullptr) const;

    void irfft(cuda::std::span<ComplexTypeCUDA> complex_input,
               cuda::std::span<float> real_output,
               cudaStream_t stream = nullptr) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // LOKI_ENABLE_CUDA

} // namespace loki::utils