#pragma once

#include <cstddef>
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

void rfft_batch_inplace(std::span<ComplexType> inout_buffer,
                        int batch_size,
                        int n_real,
                        int nthreads = 1);

void irfft_batch(std::span<ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size,
                 int n_real,
                 int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA

void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     int batch_size,
                     int n_real,
                     cudaStream_t stream = nullptr);

void irfft_batch_cuda(cuda::std::span<ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      int batch_size,
                      int n_real,
                      cudaStream_t stream = nullptr);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::utils