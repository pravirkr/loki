#include "loki/utils/fft.hpp"

#include <spdlog/spdlog.h>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::utils {

namespace {

// Custom kernel for fused normalization
__global__ void
normalize_kernel(float* __restrict__ data, int total_elements, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] *= norm;
    }
}

} // namespace

// IrfftExecutorCUDA implementation
IrfftExecutorCUDA::IrfftExecutorCUDA(int n_real,
                                     int batch_size,
                                     cudaStream_t stream)
    : m_n_real(n_real),
      m_n_complex(n_real / 2 + 1),
      m_batch_size(batch_size),
      m_stream(stream) {}

IrfftExecutorCUDA::~IrfftExecutorCUDA() {
    const size_t num_plans = m_plan_cache.size();
    for (auto& [key, plan] : s_plan_cache) {
        cuda_utils::check_cufft_call(cufftDestroy(plan),
                                     "IRFFT CUDA: cufftDestroy failed");
    }
    s_plan_cache.clear();
    spdlog::debug("IrfftExecutorCUDA destroyed {} cached plans", num_plans);
}

IrfftExecutorCUDA::execute(cuda::std::span<const ComplexTypeCUDA> complex_input,
                           cuda::std::span<float> real_output,
                           int batch_size) {
    // Input validation
    error_check::check_equal(
        real_output.size(), batch_size * m_n_real,
        "IrfftExecutorCUDA: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), batch_size * m_n_complex,
        "IrfftExecutorCUDA: complex_input size does not match batch size");

    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_input.data());
    auto* real_ptr    = real_output.data();
    cufftHandle plan  = get_or_create_plan(batch_size);
    cuda_utils::check_cufft_call(cufftExecC2R(plan, complex_ptr, real_ptr),
                                 "IrfftExecutorCUDA: cufftExecC2R failed");

    // Apply normalization (cuFFT C2R doesn't normalize automatically)
    const float norm         = 1.0F / static_cast<float>(m_n_real);
    const int total_elements = batch_size * m_n_real;

    const int threads_per_block = 256;
    const int num_blocks =
        (total_elements + threads_per_block - 1) / threads_per_block;

    normalize_kernel<<<num_blocks, threads_per_block, 0, m_stream>>>(
        real_ptr, total_elements, norm);
    cuda_utils::check_cuda_call(
        cudaGetLastError(),
        "IrfftExecutorCUDA: normalize_kernel launch failed");

    spdlog::debug(
        "IrfftExecutorCUDA: batch completed: {} transforms of size {}",
        batch_size, m_n_real);
}

cufftHandle IrfftExecutorCUDA::get_or_create_plan(int batch_size) {
    const PlanKey key{.n_real = m_n_real, .batch_size = batch_size};

    std::lock_guard<std::mutex> lock(s_mutex);

    auto it = s_plan_cache.find(key);
    if (it != s_plan_cache.end()) {
        spdlog::trace(
            "IrfftExecutor: Reusing cached plan for n_real={}, batch={}",
            m_n_real, batch_size);
        return it->second;
    }

    // Create cuFFT plan
    cufftHandle plan;
    cuda_utils::check_cufft_call(
        cufftPlan1d(&plan, m_n_real, CUFFT_C2R, batch_size),
        "IrfftExecutorCUDA: cufftPlan1d failed");
    if (m_stream != nullptr) {
        cuda_utils::check_cufft_call(cufftSetStream(plan, m_stream),
                                     "IRFFT CUDA: cufftSetStream failed");
    }

    // Insert into cache
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Check again in case another thread created it
        auto it = m_plan_cache.find(key);
        if (it != m_plan_cache.end()) {
            // Another thread beat us to it, destroy our plan and return theirs
            cufftDestroy(plan);
            return it->second;
        }

        m_plan_cache[key] = plan;
        spdlog::debug("IrfftExecutorCUDA: Created and cached plan for "
                      "n_real={}, batch={} (total plans: {})",
                      m_n_real, batch_size, m_plan_cache.size());
    }
    return plan;
}

void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     int batch_size,
                     int n_real,
                     cudaStream_t stream) {
    int n_complex = (n_real / 2) + 1;

    // Input validation
    error_check::check_equal(
        real_input.size(), batch_size * n_real,
        "RFFT CUDA batch: real_input size does not match batch size");
    error_check::check_equal(
        complex_output.size(), batch_size * n_complex,
        "RFFT CUDA batch: complex_output size does not match batch size");

    auto* real_ptr    = real_input.data();
    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_output.data());

    cufftHandle plan;
    cuda_utils::check_cufft_call(
        cufftPlan1d(&plan, n_real, CUFFT_R2C, batch_size),
        "RFFT CUDA: cufftPlan1d failed");
    if (stream != nullptr) {
        cuda_utils::check_cufft_call(cufftSetStream(plan, stream),
                                     "RFFT CUDA: cufftSetStream failed");
    }
    cuda_utils::check_cufft_call(cufftExecR2C(plan, real_ptr, complex_ptr),
                                 "RFFT CUDA: cufftExecR2C failed");
    cuda_utils::check_cufft_call(cufftDestroy(plan),
                                 "RFFT CUDA: cufftDestroy failed");
    spdlog::debug("RFFT CUDA batch completed: {} transforms of size {}",
                  batch_size, n_real);
}

void irfft_batch_cuda(cuda::std::span<ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      int batch_size,
                      int n_real,
                      cudaStream_t stream) {
    const int n_complex = (n_real / 2) + 1;

    // Input validation
    error_check::check_equal(
        real_output.size(), batch_size * n_real,
        "IRFFT CUDA batch: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), batch_size * n_complex,
        "IRFFT CUDA batch: complex_input size does not match batch size");

    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_input.data());
    auto* real_ptr    = real_output.data();

    // Create cuFFT plan
    cufftHandle plan;
    cuda_utils::check_cufft_call(
        cufftPlan1d(&plan, n_real, CUFFT_C2R, batch_size),
        "IRFFT CUDA: cufftPlan1d failed");
    if (stream != nullptr) {
        cuda_utils::check_cufft_call(cufftSetStream(plan, stream),
                                     "IRFFT CUDA: cufftSetStream failed");
    }
    cuda_utils::check_cufft_call(cufftExecC2R(plan, complex_ptr, real_ptr),
                                 "IRFFT CUDA: cufftExecC2R failed");
    cuda_utils::check_cufft_call(cufftDestroy(plan),
                                 "IRFFT CUDA: cufftDestroy failed");
    // Apply normalization (cuFFT C2R doesn't normalize automatically)
    const float norm         = 1.0F / static_cast<float>(n_real);
    const int total_elements = batch_size * n_real;

    // Use thrust for efficient GPU normalization
    if (stream != nullptr) {
        thrust::transform(thrust::cuda::par.on(stream), real_ptr,
                          real_ptr + total_elements, real_ptr,
                          [norm] __device__(float x) { return x * norm; });
    } else {
        thrust::transform(thrust::device, real_ptr, real_ptr + total_elements,
                          real_ptr,
                          [norm] __device__(float x) { return x * norm; });
    }

    spdlog::debug("IRFFT CUDA batch completed: {} transforms of size {}",
                  batch_size, n_real);
}

} // namespace loki::utils