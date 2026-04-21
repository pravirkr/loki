#include "loki/common/types.hpp"
#include "loki/utils/fft.hpp"

#include <spdlog/spdlog.h>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::math {

namespace {

// Custom kernel for fused normalization
__global__ void normalize_kernel(float* __restrict__ data,
                                 uint32_t total_elements,
                                 float norm) {
    const uint32_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < total_elements) {
        data[idx] *= norm;
    }
}
} // namespace

/*
// cuFFTDx descriptor
template <uint32_t N>
using C2R_FFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<float>() +
                         cufftdx::Type<cufftdx::fft_type::c2r>() +
                         cufftdx::SM<CUFFTDX_SM>() + cufftdx::Block());



template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void irfft_c2r_kernel(const ComplexTypeCUDA* __restrict__ complex_input,
                          float* __restrict__ real_output,
                          const uint32_t* __restrict__ batch_counter,
                          uint32_t max_batch) {
    constexpr uint32_t N       = cufftdx::size_of<FFT>::value;
    constexpr uint32_t in_len  = FFT::input_length;  // N/2+1
    constexpr uint32_t out_len = FFT::output_length; // N
    constexpr uint32_t stride  = FFT::stride;

    const uint32_t local_fft  = threadIdx.y;
    const uint32_t global_fft = blockIdx.x * FFT::ffts_per_block + local_fft;
    // nfft = 2 * batch_counter
    const uint32_t nfft_required = min(*batch_counter * 2, max_batch);

    if (global_fft >= nfft_required) {
        return;
    }

    // Register storage
    ComplexTypeCUDA thread_data[FFT::storage_size];

    // Load complex spectrum
    const uint32_t base_in = global_fft * in_len;
    for (uint32_t i = 0; i < FFT::input_ept; ++i) {
        const uint32_t pos = threadIdx.x + stride * i;
        if (pos < in_len) {
            thread_data[i] = reinterpret_cast<const ComplexTypeCUDA*>(
                complex_input)[base_in + pos];
        }
    }

    // Shared memory
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    auto* shared_mem = reinterpret_cast<ComplexTypeCUDA*>(smem);

    static_assert(!FFT::requires_workspace,
                  "Workspace-required FFT not supported");

    // Execute IRFFT
    FFT().execute(thread_data, shared_mem);

    // Store real output with normalization
    const float norm        = 1.0f / static_cast<float>(N);
    const uint32_t base_out = global_fft * out_len;
    const float* out        = reinterpret_cast<const float*>(thread_data);

    for (uint32_t i = 0; i < FFT::output_ept; ++i) {
        const uint32_t pos = threadIdx.x + stride * i;
        if (pos < out_len) {
            real_output[base_out + pos] = out[i] * norm;
        }
    }
}

template <unsigned int N> struct IrfftLauncher {
    using FFT = C2R_FFT<N>;

    static void launch(const ComplexTypeCUDA* in,
                       float* out,
                       const uint32_t* counter,
                       uint32_t max_batch,
                       cudaStream_t stream) {
        const uint32_t blocks =
            (max_batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;

        if (blocks == 0)
            return;

        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(irfft_c2r_kernel<FFT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FFT::shared_memory_size);
            configured = true;
        }

        irfft_c2r_kernel<FFT>
            <<<blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                reinterpret_cast<const ComplexTypeCUDA*>(in), out, counter,
                max_batch);
        cuda_utils::check_last_cuda_error(
            "IrfftLauncher: irfft_c2r_kernel launch failed");
    }
};

} // namespace
*/

// IrfftExecutorCUDA implementation
IrfftExecutorCUDA::IrfftExecutorCUDA(int n_real)
    : m_n_real(n_real),
      m_n_complex((n_real / 2) + 1) {}

IrfftExecutorCUDA::~IrfftExecutorCUDA() {
    const size_t num_plans = m_plan_cache.size();
    for (auto& [key, plan] : m_plan_cache) {
        cuda_utils::check_cufft_call(cufftDestroy(plan),
                                     "IRFFT CUDA: cufftDestroy failed");
    }
    m_plan_cache.clear();
    spdlog::debug("IrfftExecutorCUDA destroyed {} cached plans", num_plans);
}

void IrfftExecutorCUDA::execute(
    cuda::std::span<const ComplexTypeCUDA> complex_input,
    cuda::std::span<float> real_output,
    int batch_size,
    cudaStream_t stream) {
    // Input validation
    error_check::check_equal(
        real_output.size(), batch_size * m_n_real,
        "IrfftExecutorCUDA: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), batch_size * m_n_complex,
        "IrfftExecutorCUDA: complex_input size does not match batch size");

    auto* complex_ptr = reinterpret_cast<cufftComplex*>(
        const_cast<ComplexTypeCUDA*>(complex_input.data())); // NOLINT
    auto* real_ptr   = real_output.data();
    cufftHandle plan = get_or_create_plan(batch_size, stream);
    cuda_utils::check_cufft_call(cufftExecC2R(plan, complex_ptr, real_ptr),
                                 "IrfftExecutorCUDA: cufftExecC2R failed");

    // Apply normalization (cuFFT C2R doesn't normalize automatically)
    const int total_elements    = batch_size * m_n_real;
    const float norm            = 1.0F / static_cast<float>(m_n_real);
    const int threads_per_block = 256;
    const int blocks_per_grid =
        (total_elements + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    normalize_kernel<<<grid_dim, block_dim, 0, stream>>>(real_ptr,
                                                         total_elements, norm);
    cuda_utils::check_last_cuda_error(
        "IrfftExecutorCUDA: normalize_kernel launch failed");

    spdlog::debug(
        "IrfftExecutorCUDA: batch completed: {} transforms of size {}",
        batch_size, m_n_real);
}

cufftHandle IrfftExecutorCUDA::get_or_create_plan(int batch_size,
                                                  cudaStream_t stream) {
    const PlanKeyDevice key{
        .n_real = m_n_real, .batch_size = batch_size, .stream = stream};

    std::lock_guard<std::mutex> lock(m_mutex);

    auto it = m_plan_cache.find(key);
    if (it != m_plan_cache.end()) {
        return it->second;
    }

    // Create cuFFT plan
    cufftHandle plan;
    cuda_utils::check_cufft_call(
        cufftPlan1d(&plan, m_n_real, CUFFT_C2R, batch_size),
        "IrfftExecutorCUDA: cufftPlan1d failed");
    cuda_utils::check_cufft_call(cufftSetStream(plan, stream),
                                 "IrfftExecutorCUDA: cufftSetStream failed");

    // Insert into cache
    m_plan_cache.emplace(key, plan);
    spdlog::debug("IrfftExecutorCUDA: Created and cached plan for "
                  "n_real={}, batch={} (total plans: {})",
                  m_n_real, batch_size, m_plan_cache.size());

    return plan;
}

/*
// IrfftExecutorCUDADx implementation
IrfftExecutorCUDADx::IrfftExecutorCUDADx(int nbins, int max_leaves)
    : m_nbins(nbins),
      m_nbins_f(nbins / 2 + 1),
      m_max_batch(2 * max_leaves) {
    if (!is_supported(nbins)) {
        throw std::runtime_error(
            std::format("IrfftExecutorCUDADx: unsupported nbins={}", nbins));
    }
}

bool IrfftExecutorCUDADx::is_supported(int nbins) {

    switch (nbins) {
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
        return true;
    default:
        return false;
    }
}

void IrfftExecutorCUDADx::execute_async(
    cuda::std::span<const ComplexTypeCUDA> complex_input,
    cuda::std::span<float> real_output,
    const utils::DeviceCounter& batch_counter,
    cudaStream_t stream) const {
    const ComplexTypeCUDA* in = complex_input.data();
    float* out                = real_output.data();

    switch (m_nbins) {
    case 32:
        IrfftLauncher<32>::launch(in, out, batch_counter.data(), m_max_batch,
                                  stream);
        break;
    case 64:
        IrfftLauncher<64>::launch(in, out, batch_counter.data(), m_max_batch,
                                  stream);
        break;
    case 128:
        IrfftLauncher<128>::launch(in, out, batch_counter.data(), m_max_batch,
                                   stream);
        break;
    case 256:
        IrfftLauncher<256>::launch(in, out, batch_counter.data(), m_max_batch,
                                   stream);
        break;
    case 512:
        IrfftLauncher<512>::launch(in, out, batch_counter.data(), m_max_batch,
                                   stream);
        break;
    case 1024:
        IrfftLauncher<1024>::launch(in, out, batch_counter.data(), m_max_batch,
                                    stream);
        break;
    default:
        throw std::runtime_error(
            std::format("IrfftExecutorCUDADx: unsupported nbins={}", m_nbins));
    }
}
*/

void rfft_batch_cuda(cuda::std::span<float> real_input,
                     cuda::std::span<ComplexTypeCUDA> complex_output,
                     SizeType batch_size,
                     SizeType n_real,
                     cudaStream_t stream) {
    const SizeType n_complex = (n_real / 2) + 1;

    // Input validation
    error_check::check_equal(
        real_input.size(), batch_size * n_real,
        "RFFT CUDA batch: real_input size does not match batch size");
    error_check::check_equal(
        complex_output.size(), batch_size * n_complex,
        "RFFT CUDA batch: complex_output size does not match batch size");

    auto* real_ptr    = real_input.data();
    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_output.data());

    // ---- Chunking strategy ----
    const SizeType max_chunk_batch = 1'000'000;
    cufftHandle plan               = 0;
    SizeType current_plan_batch    = 0;

    for (SizeType offset = 0; offset < batch_size; offset += max_chunk_batch) {
        const SizeType remaining   = batch_size - offset;
        const SizeType chunk_batch = std::min(max_chunk_batch, remaining);

        // ---- cuFFT int limits ----
        if (chunk_batch >
            static_cast<SizeType>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Chunk batch exceeds cuFFT int limit");
        }
        if (n_real > static_cast<SizeType>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("n_real exceeds cuFFT int limit");
        }

        // ---- (Re)create plan if needed ----
        if (plan == 0 || current_plan_batch != chunk_batch) {
            if (plan != 0) {
                cuda_utils::check_cufft_call(cufftDestroy(plan),
                                             "RFFT CUDA: cufftDestroy failed");
            }

            cuda_utils::check_cufft_call(
                cufftPlan1d(&plan, static_cast<int>(n_real), CUFFT_R2C,
                            static_cast<int>(chunk_batch)),
                "RFFT CUDA: cufftPlan1d failed");

            if (stream != nullptr) {
                cuda_utils::check_cufft_call(
                    cufftSetStream(plan, stream),
                    "RFFT CUDA: cufftSetStream failed");
            }
            current_plan_batch = chunk_batch;
        }

        // ---- Compute offsets ----
        float* chunk_real           = real_ptr + offset * n_real;
        cufftComplex* chunk_complex = complex_ptr + offset * n_complex;
        // ---- Execute FFT ----
        cuda_utils::check_cufft_call(
            cufftExecR2C(plan, chunk_real, chunk_complex),
            "RFFT CUDA: cufftExecR2C failed");
    }

    if (plan != 0) {
        cuda_utils::check_cufft_call(cufftDestroy(plan),
                                     "RFFT CUDA: cufftDestroy failed");
    }

    spdlog::debug(
        "RFFT CUDA batch completed: {} transforms of size {} (chunked)",
        batch_size, n_real);
}

void irfft_batch_cuda(cuda::std::span<ComplexTypeCUDA> complex_input,
                      cuda::std::span<float> real_output,
                      SizeType batch_size,
                      SizeType n_real,
                      cudaStream_t stream) {
    const SizeType n_complex = (n_real / 2) + 1;

    // Input validation
    error_check::check_equal(
        real_output.size(), batch_size * n_real,
        "IRFFT CUDA batch: real_output size does not match batch size");
    error_check::check_equal(
        complex_input.size(), batch_size * n_complex,
        "IRFFT CUDA batch: complex_input size does not match batch size");

    auto* complex_ptr = reinterpret_cast<cufftComplex*>(complex_input.data());
    auto* real_ptr    = real_output.data();

    // ---- Chunking strategy ----
    const SizeType max_chunk_batch = 1'000'000;
    const float norm               = 1.0f / static_cast<float>(n_real);

    cufftHandle plan            = 0;
    SizeType current_plan_batch = 0;

    for (SizeType offset = 0; offset < batch_size; offset += max_chunk_batch) {
        const SizeType remaining   = batch_size - offset;
        const SizeType chunk_batch = std::min(max_chunk_batch, remaining);

        // cuFFT API uses int -> must check
        if (chunk_batch >
            static_cast<SizeType>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Chunk batch exceeds cuFFT int limit");
        }
        if (n_real > static_cast<SizeType>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("n_real exceeds cuFFT int limit");
        }

        // ---- (Re)create plan only if batch size changed ----
        if (plan == 0 || current_plan_batch != chunk_batch) {
            if (plan != 0) {
                cuda_utils::check_cufft_call(cufftDestroy(plan),
                                             "IRFFT CUDA: cufftDestroy failed");
            }

            cuda_utils::check_cufft_call(
                cufftPlan1d(&plan, static_cast<int>(n_real), CUFFT_C2R,
                            static_cast<int>(chunk_batch)),
                "IRFFT CUDA: cufftPlan1d failed");

            if (stream != nullptr) {
                cuda_utils::check_cufft_call(
                    cufftSetStream(plan, stream),
                    "IRFFT CUDA: cufftSetStream failed");
            }

            current_plan_batch = chunk_batch;
        }

        // ---- Compute offsets ----
        cufftComplex* chunk_complex = complex_ptr + offset * n_complex;

        float* chunk_real = real_ptr + offset * n_real;

        // ---- Execute FFT ----
        cuda_utils::check_cufft_call(
            cufftExecC2R(plan, chunk_complex, chunk_real),
            "IRFFT CUDA: cufftExecC2R failed");

        // ---- Normalize ----
        const SizeType chunk_elements = chunk_batch * n_real;

        if (stream != nullptr) {
            thrust::transform(thrust::cuda::par.on(stream), chunk_real,
                              chunk_real + chunk_elements, chunk_real,
                              [norm] __device__(float x) { return x * norm; });
        } else {
            thrust::transform(thrust::device, chunk_real,
                              chunk_real + chunk_elements, chunk_real,
                              [norm] __device__(float x) { return x * norm; });
        }
    }

    if (plan != 0) {
        cuda_utils::check_cufft_call(cufftDestroy(plan),
                                     "IRFFT CUDA: cufftDestroy failed");
    }

    spdlog::debug(
        "IRFFT CUDA batch completed: {} transforms of size {} (chunked)",
        batch_size, n_real);
}

} // namespace loki::math