#include "loki/utils/fft.hpp"

#include <format>
#include <spdlog/spdlog.h>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "loki/cuda_utils.cuh"

namespace loki::utils {

void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     int batch_size,
                     int n_real,
                     cudaStream_t stream) {
    int n_complex = (n_real / 2) + 1;

    // Input validation
    if (static_cast<int>(real_input.size()) != batch_size * n_real) {
        throw std::runtime_error(
            std::format("RFFT CUDA batch: real_input size mismatch. Expected "
                        "{}, got {}",
                        batch_size * n_real, real_input.size()));
    }
    if (static_cast<int>(complex_output.size()) != batch_size * n_complex) {
        throw std::runtime_error(std::format(
            "RFFT CUDA batch: complex_output size mismatch. Expected "
            "{}, got {}",
            batch_size * n_complex, complex_output.size()));
    }
    auto* real_ptr    = real_input.data();
    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_output.data());

    cufftHandle plan;
    cuda_utils::check_cuda_call(
        cufftPlan1d(&plan, n_real, CUFFT_R2C, batch_size),
        "RFFT CUDA: cufftPlan1d failed");
    if (stream != nullptr) {
        cuda_utils::check_cuda_call(cufftSetStream(plan, stream),
                                    "RFFT CUDA: cufftSetStream failed");
    }
    cuda_utils::check_cuda_call(cufftExecR2C(plan, real_ptr, complex_ptr),
                                "RFFT CUDA: cufftExecR2C failed");
    cuda_utils::check_cuda_call(cufftDestroy(plan),
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
    if (static_cast<int>(complex_input.size()) != batch_size * n_complex) {
        throw std::runtime_error(std::format(
            "IRFFT CUDA batch: complex_input size mismatch. Expected "
            "{}, got {}",
            batch_size * n_complex, complex_input.size()));
    }
    if (static_cast<int>(real_output.size()) != batch_size * n_real) {
        throw std::runtime_error(
            std::format("IRFFT CUDA batch: real_output size mismatch. Expected "
                        "{}, got {}",
                        batch_size * n_real, real_output.size()));
    }

    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_input.data());
    auto* real_ptr    = real_output.data();

    // Create cuFFT plan
    cufftHandle plan;
    cuda_utils::check_cuda_call(
        cufftPlan1d(&plan, n_real, CUFFT_C2R, batch_size),
        "IRFFT CUDA: cufftPlan1d failed");
    if (stream != nullptr) {
        cuda_utils::check_cuda_call(cufftSetStream(plan, stream),
                                    "IRFFT CUDA: cufftSetStream failed");
    }
    cuda_utils::check_cuda_call(cufftExecC2R(plan, complex_ptr, real_ptr),
                                "IRFFT CUDA: cufftExecC2R failed");
    cuda_utils::check_cuda_call(cufftDestroy(plan),
                                "IRFFT CUDA: cufftDestroy failed");
    // Apply normalization (cuFFT C2R doesn't normalize automatically)
    const float norm         = 1.0F / static_cast<float>(n_real);
    const int total_elements = batch_size * n_real;

    thrust::transform(thrust::device, real_ptr, real_ptr + total_elements,
                      real_ptr,
                      [norm] __device__(float x) { return x * norm; });

    spdlog::debug("IRFFT CUDA batch completed: {} transforms of size {}",
                  batch_size, n_real);
}

} // namespace loki::utils