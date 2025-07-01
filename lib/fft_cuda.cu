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
#include "loki/exceptions.hpp"

namespace loki::utils {

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

void rfft_batch_inplace_cuda(cuda::std::span<ComplexTypeCUDA> inout_buffer,
                             int batch_size,
                             int n_real,
                             cudaStream_t stream) {
    // Input validation
    if (batch_size <= 0 || n_real <= 0) {
        throw std::invalid_argument(
            std::format("RFFT CUDA in-place: batch_size ({}) and n_real ({}) "
                        "must be positive",
                        batch_size, n_real));
    }
    if (n_real % 2 != 0) {
        throw std::invalid_argument(std::format(
            "RFFT CUDA in-place: n_real ({}) must be even", n_real));
    }

    const int n_complex = (n_real / 2) + 1;

    // For in-place R2C, cuFFT needs space for n_real + 2 floats per batch
    const int required_floats  = batch_size * (n_real + 2);
    const int available_floats = static_cast<int>(inout_buffer.size()) * 2;
    if (available_floats < required_floats) {
        throw std::invalid_argument(
            std::format("RFFT CUDA in-place: buffer too small. Need {} floats "
                        "({} complex), "
                        "have {} floats ({} complex)",
                        required_floats, (required_floats + 1) / 2,
                        available_floats, inout_buffer.size()));
    }

    // Reinterpret buffer for cuFFT
    auto* real_ptr    = reinterpret_cast<float*>(inout_buffer.data());
    auto* complex_ptr = reinterpret_cast<cufftComplex*>(inout_buffer.data());

    // Create cuFFT plan for in-place R2C
    cufftHandle plan;
    cuda_utils::check_cuda_call(
        cufftPlan1d(&plan, n_real, CUFFT_R2C, batch_size),
        "RFFT CUDA in-place: cufftPlan1d failed");

    if (stream != nullptr) {
        cuda_utils::check_cuda_call(
            cufftSetStream(plan, stream),
            "RFFT CUDA in-place: cufftSetStream failed");
    }

    // Execute in-place R2C transform
    cuda_utils::check_cuda_call(cufftExecR2C(plan, real_ptr, complex_ptr),
                                "RFFT CUDA in-place: cufftExecR2C failed");

    cuda_utils::check_cuda_call(cufftDestroy(plan),
                                "RFFT CUDA in-place: cufftDestroy failed");

    spdlog::debug(
        "RFFT CUDA in-place batch completed: {} transforms of size {}",
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

void irfft_batch_inplace_cuda(cuda::std::span<ComplexTypeCUDA> inout_buffer,
                              int batch_size,
                              int n_real,
                              cudaStream_t stream) {
    // Input validation
    if (batch_size <= 0 || n_real <= 0) {
        throw std::invalid_argument(
            std::format("IRFFT CUDA in-place: batch_size ({}) and n_real ({}) "
                        "must be positive",
                        batch_size, n_real));
    }
    if (n_real % 2 != 0) {
        throw std::invalid_argument(std::format(
            "IRFFT CUDA in-place: n_real ({}) must be even", n_real));
    }

    const int n_complex = (n_real / 2) + 1;

    // Check minimum buffer size for complex input
    const int required_complex = batch_size * n_complex;
    if (static_cast<int>(inout_buffer.size()) < required_complex) {
        throw std::invalid_argument(
            std::format("IRFFT CUDA in-place: buffer too small. Need {} "
                        "complex elements, have {}",
                        required_complex, inout_buffer.size()));
    }

    // Reinterpret buffer for cuFFT
    auto* complex_ptr = reinterpret_cast<cufftComplex*>(inout_buffer.data());
    auto* real_ptr    = reinterpret_cast<float*>(inout_buffer.data());

    // Create cuFFT plan for in-place C2R
    cufftHandle plan;
    cuda_utils::check_cuda_call(
        cufftPlan1d(&plan, n_real, CUFFT_C2R, batch_size),
        "IRFFT CUDA in-place: cufftPlan1d failed");

    if (stream != nullptr) {
        cuda_utils::check_cuda_call(
            cufftSetStream(plan, stream),
            "IRFFT CUDA in-place: cufftSetStream failed");
    }

    // Execute in-place C2R transform
    cuda_utils::check_cuda_call(cufftExecC2R(plan, complex_ptr, real_ptr),
                                "IRFFT CUDA in-place: cufftExecC2R failed");

    cuda_utils::check_cuda_call(cufftDestroy(plan),
                                "IRFFT CUDA in-place: cufftDestroy failed");

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

    spdlog::debug(
        "IRFFT CUDA in-place batch completed: {} transforms of size {}",
        batch_size, n_real);
}

} // namespace loki::utils