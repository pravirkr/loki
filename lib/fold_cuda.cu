#include "loki/algorithms/fold.hpp"

#include <cmath>
#include <memory>

#include <spdlog/spdlog.h>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/kernel_utils.cuh"

namespace loki::algorithms {

namespace {

// Thrust functor for computing phase indices
struct ComputePhase {
    const double* __restrict__ freq_arr;
    double tsamp, t_ref;
    uint32_t segment_len, nbins;

    __device__ uint32_t operator()(uint32_t idx) const {
        const uint32_t ifreq     = idx / segment_len;
        const uint32_t isamp     = idx - (ifreq * segment_len);
        const double proper_time = (static_cast<double>(isamp) * tsamp) - t_ref;
        float phase = utils::get_phase_idx_device(proper_time, freq_arr[ifreq],
                                                  nbins, 0.0);
        uint32_t shift = __float2uint_rn(phase);
        return (shift >= nbins) ? 0 : shift;
    }
};

// CUDA kernel for folding operation with 1D block configuration
__global__ void kernel_fold_time_1d(const float* __restrict__ ts_e,
                                    const float* __restrict__ ts_v,
                                    const uint32_t* __restrict__ phase_map,
                                    float* __restrict__ fold,
                                    uint32_t nfreqs,
                                    uint32_t nsegments,
                                    uint32_t segment_len,
                                    uint32_t nbins) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Total (segment, sample) pairs
    const uint32_t total_work = nsegments * segment_len;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (segment, sample)
    const uint32_t iseg  = tid / segment_len;
    const uint32_t isamp = tid - (iseg * segment_len);

    // Process all frequencies for this (segment, sample) pair
    for (uint32_t ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const uint32_t phase_idx = (ifreq * segment_len) + isamp;
        const uint32_t phase_bin = phase_map[phase_idx];
        const uint32_t ts_idx    = (iseg * segment_len) + isamp;
        const uint32_t fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + (ifreq * 2 * nbins);

        // Atomic add (but much less contention now!)
        atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
        atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
    }
}

// CUDA kernel for folding operation with 2D block configuration
__global__ void kernel_fold_time_2d(const float* __restrict__ ts_e,
                                    const float* __restrict__ ts_v,
                                    const uint32_t* __restrict__ phase_map,
                                    float* __restrict__ fold,
                                    uint32_t nfreqs,
                                    uint32_t nsegments,
                                    uint32_t segment_len,
                                    uint32_t nbins) {
    const uint32_t isamp = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t ifreq = blockIdx.y;

    if (isamp >= segment_len || ifreq >= nfreqs) {
        return;
    }
    const uint32_t phase_idx   = (ifreq * segment_len) + isamp;
    const uint32_t phase_bin   = phase_map[phase_idx];
    const uint32_t freq_offset = ifreq * 2 * nbins;
    for (uint32_t iseg = 0; iseg < nsegments; ++iseg) {
        const uint32_t ts_idx = (iseg * segment_len) + isamp;
        const uint32_t fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + freq_offset;

        atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
        atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
    }
}

// Alternative: Use shared memory for even better performance
__global__ void kernel_fold_time_shmem(const float* __restrict__ ts_e,
                                       const float* __restrict__ ts_v,
                                       const uint32_t* __restrict__ phase_map,
                                       float* __restrict__ fold,
                                       uint32_t nfreqs,
                                       uint32_t nsegments,
                                       uint32_t segment_len,
                                       uint32_t nbins) {
    // One block per frequency, each block processes all samples for that
    // frequency
    extern __shared__ float shared_bins[];
    float* shared_e = shared_bins;
    float* shared_v = shared_bins + nbins;

    const uint32_t tid               = threadIdx.x;
    const uint32_t ifreq             = blockIdx.y;
    const uint32_t threads_per_block = blockDim.x;

    for (uint32_t iseg = 0; iseg < nsegments; ++iseg) {
        // Initialize shared memory for this segment
        for (uint32_t bin = tid; bin < nbins; bin += threads_per_block) {
            shared_e[bin] = 0.0F;
            shared_v[bin] = 0.0F;
        }
        __syncthreads();

        // Process samples for this frequency and segment
        for (uint32_t isamp = tid; isamp < segment_len;
             isamp += threads_per_block) {
            const uint32_t phase_idx = (ifreq * segment_len) + isamp;
            const uint32_t phase_bin = phase_map[phase_idx];
            const uint32_t ts_idx    = (iseg * segment_len) + isamp;

            // Accumulate in shared memory
            atomicAdd(&shared_e[phase_bin], ts_e[ts_idx]);
            atomicAdd(&shared_v[phase_bin], ts_v[ts_idx]);
        }
        __syncthreads();

        // Write shared memory results to global memory for this segment
        const uint32_t fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + (ifreq * 2 * nbins);
        for (uint32_t bin = tid; bin < nbins; bin += threads_per_block) {
            fold[fold_base_idx + bin]         = shared_e[bin];
            fold[fold_base_idx + nbins + bin] = shared_v[bin];
        }
        __syncthreads();
    }
}

// =============================================================================
// Optimized kernel for Complex BruteFold with small number of harmonics
// (num_harms <= blockDim.x) Each thread handles exactly one harmonic, better
// occupancy
// =============================================================================
__global__ void
kernel_fold_complex_one_harmonic_per_thread(const float* __restrict__ ts_e,
                                            const float* __restrict__ ts_v,
                                            ComplexTypeCUDA* __restrict__ fold,
                                            const double* __restrict__ freqs,
                                            uint32_t nfreqs,
                                            uint32_t nsegments,
                                            uint32_t segment_len,
                                            uint32_t nbins_f,
                                            double tsamp,
                                            double t_ref) {
    const uint32_t iseg      = blockIdx.x;
    const uint32_t ifreq     = blockIdx.y;
    const uint32_t tid       = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    if (iseg >= nsegments || ifreq >= nfreqs || tid >= nbins_f) {
        return;
    }
    extern __shared__ float sh[];
    float* sh_e = sh;
    float* sh_v = sh + segment_len;

    // Cooperative load - all threads participate
    const uint32_t start_idx = iseg * segment_len;
    for (uint32_t i = tid; i < segment_len; i += block_dim) {
        sh_e[i] = ts_e[start_idx + i];
        sh_v[i] = ts_v[start_idx + i];
    }
    __syncthreads();

    const auto base_offset =
        (iseg * nfreqs * 2 * nbins_f) + (ifreq * 2 * nbins_f);

    // Thread 0 handles DC via reduction
    if (tid == 0) {
        float sum_e = 0.0F, sum_v = 0.0F;
        for (uint32_t k = 0; k < segment_len; ++k) {
            sum_e += sh_e[k];
            sum_v += sh_v[k];
        }
        fold[base_offset]           = {sum_e, 0.0F};
        fold[base_offset + nbins_f] = {sum_v, 0.0F};
    }

    // Threads 1..nbins_f-1 handle AC harmonics
    if (tid >= 1) {
        // Compute AC for this harmonic
        const double phase_factor =
            2.0 * kPI * freqs[ifreq] * static_cast<double>(tid);
        const double init_phase  = phase_factor * t_ref;
        const double delta_phase = -phase_factor * tsamp;
        // Fast sincos computation
        float ph_r, ph_i, step_r, step_i;
        __sincosf(static_cast<float>(init_phase), &ph_i, &ph_r);
        __sincosf(static_cast<float>(delta_phase), &step_i, &step_r);
        float acc_e_r = 0.0F, acc_e_i = 0.0F;
        float acc_v_r = 0.0F, acc_v_i = 0.0F;

        for (uint32_t k = 0; k < segment_len; ++k) {
            acc_e_r = fmaf(sh_e[k], ph_r, acc_e_r);
            acc_e_i = fmaf(sh_e[k], ph_i, acc_e_i);
            acc_v_r = fmaf(sh_v[k], ph_r, acc_v_r);
            acc_v_i = fmaf(sh_v[k], ph_i, acc_v_i);

            const float new_r = (ph_r * step_r) - (ph_i * step_i);
            const float new_i = (ph_r * step_i) + (ph_i * step_r);
            ph_r              = new_r;
            ph_i              = new_i;
        }

        fold[base_offset + tid]           = {acc_e_r, acc_e_i};
        fold[base_offset + nbins_f + tid] = {acc_v_r, acc_v_i};
    }
}

template <bool UseShared>
__global__ void kernel_fold_complex_unified(const float* __restrict__ ts_e,
                                            const float* __restrict__ ts_v,
                                            ComplexTypeCUDA* __restrict__ fold,
                                            const double* __restrict__ freqs,
                                            uint32_t nfreqs,
                                            uint32_t nsegments,
                                            uint32_t segment_len,
                                            uint32_t nbins_f,
                                            double tsamp,
                                            double t_ref) {
    const uint32_t iseg      = blockIdx.x;
    const uint32_t ifreq     = blockIdx.y;
    const uint32_t tid       = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    if (iseg >= nsegments || ifreq >= nfreqs || tid >= nbins_f) {
        return;
    }

    const float* ts_e_seg;
    const float* ts_v_seg;
    const uint32_t start_idx = iseg * segment_len;
    if constexpr (UseShared) {
        extern __shared__ float sh[];
        float* sh_e = sh;
        float* sh_v = sh + segment_len;

        // Cooperative load - all threads participate
        for (uint32_t i = tid; i < segment_len; i += block_dim) {
            sh_e[i] = ts_e[start_idx + i];
            sh_v[i] = ts_v[start_idx + i];
        }
        __syncthreads();
        ts_e_seg = sh_e;
        ts_v_seg = sh_v;
    } else {
        ts_e_seg = ts_e + start_idx;
        ts_v_seg = ts_v + start_idx;
    }

    const auto base_offset =
        (iseg * nfreqs * 2 * nbins_f) + (ifreq * 2 * nbins_f);

    // DC Component: parallel reduction from global memory
    float sum_e = 0.0F, sum_v = 0.0F;
    for (uint32_t i = tid; i < segment_len; i += block_dim) {
        sum_e += ts_e_seg[i];
        sum_v += ts_v_seg[i];
    }

    // Warp reduction
    for (uint32_t off = 16; off > 0; off >>= 1U) {
        sum_e += __shfl_down_sync(0xffffffff, sum_e, off);
        sum_v += __shfl_down_sync(0xffffffff, sum_v, off);
    }

    __shared__ float warp_e[32];
    __shared__ float warp_v[32];

    const uint32_t warp_id   = tid >> 5U;
    const uint32_t lane_id   = tid & 31U;
    const uint32_t num_warps = (block_dim + 31U) >> 5U;

    if (lane_id == 0) {
        warp_e[warp_id] = sum_e;
        warp_v[warp_id] = sum_v;
    }
    __syncthreads();

    if (tid < 32) {
        float e = (tid < num_warps) ? warp_e[tid] : 0.0F;
        float v = (tid < num_warps) ? warp_v[tid] : 0.0F;

        for (uint32_t off = 16; off > 0; off >>= 1U) {
            e += __shfl_down_sync(0xffffffff, e, off);
            v += __shfl_down_sync(0xffffffff, v, off);
        }

        if (tid == 0) {
            fold[base_offset]           = {e, 0.0F};
            fold[base_offset + nbins_f] = {v, 0.0F};
        }
    }
    __syncthreads();

    // AC Components
    const double phase_factor = 2.0 * kPI * freqs[ifreq];
    for (uint32_t m = tid + 1; m < nbins_f; m += block_dim) {
        float ph_r, ph_i, step_r, step_i;
        __sincosf(static_cast<float>(phase_factor * m * t_ref), &ph_i, &ph_r);
        __sincosf(static_cast<float>(-phase_factor * m * tsamp), &step_i,
                  &step_r);
        float acc_e_r = 0.0F, acc_e_i = 0.0F;
        float acc_v_r = 0.0F, acc_v_i = 0.0F;

        for (uint32_t k = 0; k < segment_len; ++k) {
            acc_e_r = fmaf(ts_e_seg[k], ph_r, acc_e_r);
            acc_e_i = fmaf(ts_e_seg[k], ph_i, acc_e_i);
            acc_v_r = fmaf(ts_v_seg[k], ph_r, acc_v_r);
            acc_v_i = fmaf(ts_v_seg[k], ph_i, acc_v_i);

            const float new_r = (ph_r * step_r) - (ph_i * step_i);
            const float new_i = (ph_r * step_i) + (ph_i * step_r);
            ph_r              = new_r;
            ph_i              = new_i;
        }

        fold[base_offset + m]           = {acc_e_r, acc_e_i};
        fold[base_offset + nbins_f + m] = {acc_v_r, acc_v_i};
    }
}

} // namespace

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class BruteFoldCUDA<FoldTypeCUDA>::Impl {
public:
    using HostFoldType   = FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = FoldTypeTraits<FoldTypeCUDA>::DeviceType;

    Impl(std::span<const double> freq_arr,
         SizeType segment_len,
         SizeType nbins,
         SizeType nsamps,
         double tsamp,
         double t_ref,
         int device_id)
        : m_freq_arr(freq_arr.begin(), freq_arr.end()),
          m_segment_len(segment_len),
          m_nbins(nbins),
          m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_t_ref(t_ref),
          m_device_id(device_id) {
        error_check::check(!m_freq_arr.empty(),
                           "BruteFoldCUDA::Impl: Frequency array is empty");
        error_check::check(m_nsamps % m_segment_len == 0,
                           "BruteFoldCUDA::Impl: Number of samples is not a "
                           "multiple of segment length");
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        m_nfreqs     = m_freq_arr.size();
        m_nsegments  = m_nsamps / m_segment_len;
        m_freq_arr_d = thrust::device_vector<double>(m_freq_arr);
        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            // Time domain - allocate phase map
            m_phase_map_d.resize(m_nfreqs * m_segment_len, 0U);
            compute_phase();
        } else {
            // Fourier domain
            m_nbins_f = (nbins / 2) + 1;
        }
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_fold_size() const {
        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            return m_nsegments * m_nfreqs * 2 * m_nbins;
        } else {
            return m_nsegments * m_nfreqs * 2 * m_nbins_f;
        }
    }

    void execute_h(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   std::span<HostFoldType> fold) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());

        // Resize buffers only if needed
        if (m_ts_e_d.size() < ts_e.size()) {
            m_ts_e_d.resize(ts_e.size());
            m_ts_v_d.resize(ts_v.size());
        }
        if (m_fold_d.size() < fold.size()) {
            m_fold_d.resize(fold.size());
        }

        // Copy input data to device
        cudaStream_t stream = nullptr;
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_e_d.data()),
                            ts_e.data(), ts_e.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync ts_e failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_v_d.data()),
                            ts_v.data(), ts_v.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync ts_v failed");

        // Execute folding kernel on device using persistent buffers
        execute_d(cuda_utils::as_span(m_ts_e_d), cuda_utils::as_span(m_ts_v_d),
                  cuda_utils::as_span(m_fold_d), stream);

        // Copy result back to host
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(fold.data(),
                            thrust::raw_pointer_cast(m_fold_d.data()),
                            fold.size() * sizeof(HostFoldType),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync fold failed");
        // Synchronize stream before returning to host
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
        spdlog::debug("BruteFoldCUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e,
                   cuda::std::span<const float> ts_v,
                   cuda::std::span<DeviceFoldType> fold,
                   cudaStream_t stream) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());
        // Ensure output fold is zeroed
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(fold.data(), 0,
                            fold.size() * sizeof(DeviceFoldType), stream),
            "cudaMemsetAsync fold failed");
        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            execute_device_float(ts_e.data(), ts_v.data(), fold.data(), stream);
        } else {
            execute_device_complex(ts_e.data(), ts_v.data(), fold.data(),
                                   stream);
        }
        // Execution complete
        spdlog::debug(
            "BruteFoldCUDA::Impl: Device execution complete on stream");
    }

private:
    std::vector<double> m_freq_arr;
    SizeType m_segment_len;
    SizeType m_nbins;
    SizeType m_nsamps;
    double m_tsamp;
    double m_t_ref;
    int m_device_id;

    SizeType m_nfreqs;
    SizeType m_nsegments;
    thrust::device_vector<double> m_freq_arr_d;

    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<DeviceFoldType> m_fold_d;

    // Time domain only
    thrust::device_vector<uint32_t> m_phase_map_d;

    // Fourier domain only
    SizeType m_nbins_f;

    void check_inputs(SizeType ts_e_size,
                      SizeType ts_v_size,
                      SizeType fold_size) const {
        error_check::check_equal(
            ts_e_size, m_nsamps,
            "BruteFoldCUDA::Impl: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v_size, ts_e_size,
            "BruteFoldCUDA::Impl: ts_v must have size nsamps");
        error_check::check_equal(
            fold_size, get_fold_size(),
            "BruteFoldCUDA::Impl: fold must have size fold_size");
    }

    void compute_phase() {
        auto first = thrust::counting_iterator<uint32_t>(0);
        auto last =
            thrust::counting_iterator<uint32_t>(m_nfreqs * m_segment_len);
        ComputePhase functor{
            .freq_arr    = thrust::raw_pointer_cast(m_freq_arr_d.data()),
            .tsamp       = m_tsamp,
            .t_ref       = m_t_ref,
            .segment_len = static_cast<uint32_t>(m_segment_len),
            .nbins       = static_cast<uint32_t>(m_nbins),
        };
        thrust::transform(thrust::device, first, last, m_phase_map_d.begin(),
                          functor);
        cuda_utils::check_last_cuda_error("thrust::transform failed");
        spdlog::debug("BruteFoldCUDA::Impl: Phase map computed");
    }

    void execute_device_float(const float* __restrict__ ts_e_d,
                              const float* __restrict__ ts_v_d,
                              float* __restrict__ fold_d,
                              cudaStream_t stream) {
        // Use 1D block configuration for small nfreqs
        if (m_nfreqs <= 64) {
            const auto total_work =
                static_cast<int>(m_nsegments * m_segment_len);
            const dim3 block_dim(512);
            const dim3 grid_dim((total_work + block_dim.x - 1) / block_dim.x);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
            kernel_fold_time_1d<<<grid_dim, block_dim, 0, stream>>>(
                ts_e_d, ts_v_d, thrust::raw_pointer_cast(m_phase_map_d.data()),
                fold_d, m_nfreqs, m_nsegments, m_segment_len, m_nbins);
        } else if (m_nbins <= 512 && m_nfreqs <= 65535) {
            // Use shared memory for small bin counts
            const dim3 block_dim(256);
            const dim3 grid_dim(1, m_nfreqs);
            const auto shmem_size = 2 * m_nbins * sizeof(float);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                                   shmem_size);
            kernel_fold_time_shmem<<<grid_dim, block_dim, shmem_size, stream>>>(
                ts_e_d, ts_v_d, thrust::raw_pointer_cast(m_phase_map_d.data()),
                fold_d, m_nfreqs, m_nsegments, m_segment_len, m_nbins);
        } else {
            // Use 2D block configuration for large nfreqs
            const dim3 block_dim(256);
            const dim3 grid_dim((m_segment_len + block_dim.x - 1) / block_dim.x,
                                static_cast<int>(m_nfreqs), 1);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
            kernel_fold_time_2d<<<grid_dim, block_dim, 0, stream>>>(
                ts_e_d, ts_v_d, thrust::raw_pointer_cast(m_phase_map_d.data()),
                fold_d, m_nfreqs, m_nsegments, m_segment_len, m_nbins);
        }
        cuda_utils::check_last_cuda_error("kernel_fold launch failed");
    }

    void execute_device_complex(const float* __restrict__ ts_e_d,
                                const float* __restrict__ ts_v_d,
                                ComplexTypeCUDA* __restrict__ fold_d,
                                cudaStream_t stream) {
        const dim3 grid(m_nsegments, m_nfreqs);
        const auto max_shmem  = cuda_utils::get_max_shared_memory();
        const auto shmem_size = 2 * m_segment_len * sizeof(float);
        // Strategy selection based on problem size and hardware constraints
        if (shmem_size <= max_shmem) {
            if (m_nbins_f <= 1024) {
                const int block_size = m_nbins_f;
                cuda_utils::check_kernel_launch_params(grid, block_size,
                                                       shmem_size);
                kernel_fold_complex_one_harmonic_per_thread<<<
                    grid, block_size, shmem_size, stream>>>(
                    ts_e_d, ts_v_d, fold_d,
                    thrust::raw_pointer_cast(m_freq_arr_d.data()), m_nfreqs,
                    m_nsegments, m_segment_len, m_nbins_f, m_tsamp, m_t_ref);
            } else {
                // Strided approach for larger number of harmonics
                const int block_size = 256;
                cuda_utils::check_kernel_launch_params(grid, block_size,
                                                       shmem_size);
                kernel_fold_complex_unified<true>
                    <<<grid, block_size, shmem_size, stream>>>(
                        ts_e_d, ts_v_d, fold_d,
                        thrust::raw_pointer_cast(m_freq_arr_d.data()), m_nfreqs,
                        m_nsegments, m_segment_len, m_nbins_f, m_tsamp,
                        m_t_ref);
            }
        } else {
            // Fallback: segment too large for shared memory
            const int block_size = 256;
            cuda_utils::check_kernel_launch_params(grid, block_size);
            kernel_fold_complex_unified<false><<<grid, block_size, 0, stream>>>(
                ts_e_d, ts_v_d, fold_d,
                thrust::raw_pointer_cast(m_freq_arr_d.data()), m_nfreqs,
                m_nsegments, m_segment_len, m_nbins_f, m_tsamp, m_t_ref);
        }

        cuda_utils::check_last_cuda_error("execute_device_complex failed");
    }

}; // End BruteFoldCUDA::Impl definition

template <SupportedFoldTypeCUDA FoldTypeCUDA>
BruteFoldCUDA<FoldTypeCUDA>::BruteFoldCUDA(std::span<const double> freq_arr,
                                           SizeType segment_len,
                                           SizeType nbins,
                                           SizeType nsamps,
                                           double tsamp,
                                           double t_ref,
                                           int device_id)
    : m_impl(std::make_unique<Impl>(
          freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
BruteFoldCUDA<FoldTypeCUDA>::~BruteFoldCUDA() = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
BruteFoldCUDA<FoldTypeCUDA>::BruteFoldCUDA(BruteFoldCUDA&& other) noexcept =
    default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
BruteFoldCUDA<FoldTypeCUDA>& BruteFoldCUDA<FoldTypeCUDA>::operator=(
    BruteFoldCUDA<FoldTypeCUDA>&& other) noexcept = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType BruteFoldCUDA<FoldTypeCUDA>::get_fold_size() const {
    return m_impl->get_fold_size();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BruteFoldCUDA<FoldTypeCUDA>::execute(std::span<const float> ts_e,
                                          std::span<const float> ts_v,
                                          std::span<HostFoldType> fold) {
    m_impl->execute_h(ts_e, ts_v, fold);
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BruteFoldCUDA<FoldTypeCUDA>::execute(cuda::std::span<const float> ts_e,
                                          cuda::std::span<const float> ts_v,
                                          cuda::std::span<DeviceFoldType> fold,
                                          cudaStream_t stream) {
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::vector<typename FoldTypeTraits<FoldTypeCUDA>::HostType>
compute_brute_fold_cuda(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        std::span<const double> freq_arr,
                        SizeType segment_len,
                        SizeType nbins,
                        double tsamp,
                        double t_ref,
                        int device_id) {
    using HostFoldType    = FoldTypeTraits<FoldTypeCUDA>::HostType;
    const SizeType nsamps = ts_e.size();
    BruteFoldCUDA<FoldTypeCUDA> bf(freq_arr, segment_len, nbins, nsamps, tsamp,
                                   t_ref, device_id);
    std::vector<HostFoldType> fold(bf.get_fold_size(), HostFoldType{});
    bf.execute(ts_e, ts_v, std::span<HostFoldType>(fold));
    return fold;
}

// Explicit instantiation
template class BruteFoldCUDA<float>;
template class BruteFoldCUDA<ComplexTypeCUDA>;

template std::vector<float>
compute_brute_fold_cuda<float>(std::span<const float>,
                               std::span<const float>,
                               std::span<const double>,
                               SizeType,
                               SizeType,
                               double,
                               double,
                               int);
template std::vector<ComplexType>
compute_brute_fold_cuda<ComplexTypeCUDA>(std::span<const float>,
                                         std::span<const float>,
                                         std::span<const double>,
                                         SizeType,
                                         SizeType,
                                         double,
                                         double,
                                         int);

} // namespace loki::algorithms