#pragma once

#include <cstddef>
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

// Forward declare for global FFTW initialization
void ensure_fftw_threading(int nthreads = 1);

struct PlanKey {
    int n_real;
    int batch_size;

    bool operator==(const PlanKey& other) const {
        return n_real == other.n_real && batch_size == other.batch_size;
    }
};

struct PlanKeyHash {
    SizeType operator()(const PlanKey& k) const {
        return std::hash<int>{}(k.n_real) ^
               (std::hash<int>{}(k.batch_size) << 1U);
    }
};

/**
 * @brief RAII wrapper for FFTW forward transforms (R2C)
 *
 * Manages FFTW plans for real-to-complex transforms with automatic cleanup.
 * Plans are cached per chunk size for optimal reuse across batches.
 * Thread-safe for concurrent execution calls.
 */
class RfftExecutor {
public:
    /**
     * @brief Construct executor for given transform size
     * @param n_real Size of real input (output will be n_real/2+1 complex)
     * @param nthreads Number of threads to use for FFTW (default: 1)
     * @param max_chunk_size Maximum batch size for plan caching (default:
     * 16384)
     */
    explicit RfftExecutor(int n_real,
                          int nthreads       = 1,
                          int max_chunk_size = 16384);
    ~RfftExecutor() = default;

    RfftExecutor(const RfftExecutor&)            = delete;
    RfftExecutor& operator=(const RfftExecutor&) = delete;
    RfftExecutor(RfftExecutor&&)                 = delete;
    RfftExecutor& operator=(RfftExecutor&&)      = delete;

    /**
     * @brief Execute forward FFT on batch of real inputs
     * @param real_input Real input array [batch_size * n_real]
     * @param complex_output Complex output array [batch_size * n_complex]
     * @param batch_size Number of transforms to perform
     *
     * Automatically chunks large batches for optimal performance.
     * Plans are cached for the chunk size and reused.
     */
    void execute(std::span<const float> real_input,
                 std::span<ComplexType> complex_output,
                 int batch_size);

    inline static std::unordered_map<PlanKey, fftwf_plan, PlanKeyHash>
        s_plan_cache;
    inline static std::mutex s_mutex;

private:
    int m_n_real;
    int m_n_complex;
    int m_nthreads;
    int m_max_chunk_size;

    fftwf_plan get_or_create_plan(int batch_size);
};

/**
 * @brief RAII wrapper for FFTW inverse transforms (C2R)
 *
 * Manages FFTW plans for complex-to-real transforms with automatic cleanup.
 * Plans are cached per chunk size for optimal reuse across batches.
 * Thread-safe for concurrent execution calls.
 */
class IrfftExecutor {
public:
    /**
     * @brief Construct executor for given transform size
     * @param n_real Size of real output (input will be n_real/2+1 complex)
     * @param nthreads Number of threads to use for FFTW (default: 1)
     * @param max_chunk_size Maximum batch size for plan caching (default:
     * 16384)
     */
    explicit IrfftExecutor(int n_real,
                           int nthreads       = 1,
                           int max_chunk_size = 16384);
    ~IrfftExecutor() = default;

    IrfftExecutor(const IrfftExecutor&)            = delete;
    IrfftExecutor& operator=(const IrfftExecutor&) = delete;
    IrfftExecutor(IrfftExecutor&&)                 = delete;
    IrfftExecutor& operator=(IrfftExecutor&&)      = delete;

    /**
     * @brief Execute inverse FFT on batch of complex inputs
     * @param complex_input Complex input array [batch_size * n_complex]
     * @param real_output Real output array [batch_size * n_real]
     * @param batch_size Number of transforms to perform
     *
     * Automatically chunks large batches and normalizes output.
     * Plans are cached for the chunk size and reused.
     */
    void execute(std::span<const ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size);

    inline static std::unordered_map<PlanKey, fftwf_plan, PlanKeyHash>
        s_plan_cache;
    inline static std::mutex s_mutex;

private:
    int m_n_real;
    int m_n_complex;
    int m_nthreads;
    int m_max_chunk_size;

    fftwf_plan get_or_create_plan(int batch_size);
};

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

// Convenience functions (thin wrappers around executors)
/**
 * @brief Convenience function for one-off forward FFT batches
 *
 * Creates temporary RfftExecutor. For repeated calls with same n_real,
 * prefer creating RfftExecutor once and reusing it.
 */
void rfft_batch(std::span<const float> real_input,
                std::span<ComplexType> complex_output,
                int batch_size,
                int n_real,
                int nthreads = 1);

/**
 * @brief Convenience function for one-off inverse FFT batches
 *
 * Creates temporary IrfftExecutor. For repeated calls with same n_real,
 * prefer creating IrfftExecutor once and reusing it.
 */
void irfft_batch(std::span<const ComplexType> complex_input,
                 std::span<float> real_output,
                 int batch_size,
                 int n_real,
                 int nthreads = 1);

#ifdef LOKI_ENABLE_CUDA

class IrfftExecutorCUDA {
public:
    explicit IrfftExecutorCUDA(int n_real,
                               int batch_size      = 4096,
                               cudaStream_t stream = nullptr);
    ~IrfftExecutorCUDA();

    IrfftExecutorCUDA(const IrfftExecutorCUDA&)            = delete;
    IrfftExecutorCUDA& operator=(const IrfftExecutorCUDA&) = delete;
    IrfftExecutorCUDA(IrfftExecutorCUDA&&)                 = delete;
    IrfftExecutorCUDA& operator=(IrfftExecutorCUDA&&)      = delete;

    void execute(cuda::std::span<const ComplexTypeCUDA> complex_input,
                 std::span<float> real_output,
                 int batch_size);

private:
    int m_n_real;
    int m_n_complex;
    int m_batch_size;

    std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> m_plan_cache;
    std::mutex m_mutex;
    cudaStream_t m_stream;
    cufftHandle get_or_create_plan(int batch_size);
};

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