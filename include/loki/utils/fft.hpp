#pragma once

#include <mutex>
#include <span>
#include <unordered_map>

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

#include <fftw3.h>

#include "loki/common/types.hpp"

namespace loki::utils {

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

class IrfftExecutor {
public:
    explicit IrfftExecutor(int n_real);
    ~IrfftExecutor() = default;

    IrfftExecutor(const IrfftExecutor&)            = delete;
    IrfftExecutor& operator=(const IrfftExecutor&) = delete;
    IrfftExecutor(IrfftExecutor&&)                 = delete;
    IrfftExecutor& operator=(IrfftExecutor&&)      = delete;

    void execute(std::span<const ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size);

private:
    int m_n_real;
    int m_n_complex;

    inline static std::unordered_map<int, fftwf_plan> s_plan_cache;
    inline static std::mutex s_mutex;
    inline static bool s_initialized{false};

    fftwf_plan get_plan(int batch_size);
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