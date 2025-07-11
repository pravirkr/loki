#include "loki/algorithms/fold.hpp"

#include <cmath>
#include <memory>

#include <spdlog/spdlog.h>

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::algorithms {

namespace {

// CUDA device function to calculate phase index
__device__ int get_phase_idx_device(double proper_time,
                                    double freq,
                                    int nbins,
                                    double delay = 0.0) {
    const double phase      = fmod((proper_time + delay) * freq, 1.0);
    const double norm_phase = phase < 0.0 ? phase + 1.0 : phase;
    const auto iphase =
        static_cast<int>(round(norm_phase * static_cast<double>(nbins))) %
        nbins;
    return iphase;
}

// Thrust functor for computing phase indices
struct ComputePhase {
    const double* freq_arr;
    double tsamp, t_ref;
    int segment_len, nbins;

    __device__ uint32_t operator()(int idx) const {
        const int ifreq          = idx / segment_len;
        const int isamp          = idx % segment_len;
        const double proper_time = (static_cast<double>(isamp) * tsamp) - t_ref;
        return static_cast<uint32_t>(
            get_phase_idx_device(proper_time, freq_arr[ifreq], nbins, 0.0));
    }
};

// CUDA kernel for folding operation with 1D block configuration
__global__ void kernel_fold_segments(const float* __restrict__ ts_e,
                                     const float* __restrict__ ts_v,
                                     const uint32_t* __restrict__ phase_map,
                                     float* __restrict__ fold,
                                     int nfreqs,
                                     int segment_len,
                                     int nbins,
                                     int nsegments) {
    const auto tid = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    // Total (segment, sample) pairs
    const auto total_work = nsegments * segment_len;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (segment, sample)
    const int iseg  = tid / segment_len;
    const int isamp = tid % segment_len;

    // Process all frequencies for this (segment, sample) pair
    for (int ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const auto phase_idx = (ifreq * segment_len) + isamp;
        const auto phase_bin = phase_map[phase_idx];
        const auto ts_idx    = (iseg * segment_len) + isamp;
        const auto fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + (ifreq * 2 * nbins);

        // Atomic add (but much less contention now!)
        atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
        atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
    }
}

// CUDA kernel for folding operation with 2D block configuration
__global__ void kernel_fold_2d(const float* __restrict__ ts_e,
                               const float* __restrict__ ts_v,
                               const uint32_t* __restrict__ phase_map,
                               float* __restrict__ fold,
                               int nfreqs,
                               int segment_len,
                               int nbins,
                               int nsegments) {
    const auto isamp =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto ifreq = static_cast<int>(blockIdx.y);

    if (isamp >= segment_len || ifreq >= nfreqs) {
        return;
    }
    const auto phase_idx  = (ifreq * segment_len) + isamp;
    const auto phase_bin  = phase_map[phase_idx];
    const int freq_offset = ifreq * 2 * nbins;
    for (int iseg = 0; iseg < nsegments; ++iseg) {
        const int ts_idx        = (iseg * segment_len) + isamp;
        const int fold_base_idx = (iseg * nfreqs * 2 * nbins) + freq_offset;

        atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
        atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
    }
}

// Alternative: Use shared memory for even better performance
__global__ void kernel_fold_shared_mem(const float* __restrict__ ts_e,
                                       const float* __restrict__ ts_v,
                                       const uint32_t* __restrict__ phase_map,
                                       float* __restrict__ fold,
                                       int nfreqs,
                                       int segment_len,
                                       int nbins,
                                       int nsegments) {
    // One block per frequency, each block processes all samples for that
    // frequency
    extern __shared__ float shared_bins[]; // NOLINT
    float* shared_e = shared_bins;         // NOLINT
    float* shared_v = shared_bins + nbins; // NOLINT

    const auto tid               = static_cast<int>(threadIdx.x);
    const auto ifreq             = static_cast<int>(blockIdx.y);
    const auto threads_per_block = static_cast<int>(blockDim.x);

    for (int iseg = 0; iseg < nsegments; ++iseg) {
        // Initialize shared memory for this segment
        for (int bin = tid; bin < nbins; bin += threads_per_block) {
            shared_e[bin] = 0.0F;
            shared_v[bin] = 0.0F;
        }
        __syncthreads();

        // Process samples for this frequency and segment
        for (int isamp = tid; isamp < segment_len; isamp += threads_per_block) {
            const auto phase_idx = (ifreq * segment_len) + isamp;
            const auto phase_bin = phase_map[phase_idx];
            const auto ts_idx    = (iseg * segment_len) + isamp;

            // Accumulate in shared memory
            atomicAdd(&shared_e[phase_bin], ts_e[ts_idx]);
            atomicAdd(&shared_v[phase_bin], ts_v[ts_idx]);
        }
        __syncthreads();

        // Write shared memory results to global memory for this segment
        const auto fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + (ifreq * 2 * nbins);
        for (int bin = tid; bin < nbins; bin += threads_per_block) {
            atomicAdd(&fold[fold_base_idx + bin], shared_e[bin]);
            atomicAdd(&fold[fold_base_idx + nbins + bin], shared_v[bin]);
        }
        __syncthreads();
    }
}

} // namespace

class BruteFoldCUDA::Impl {
public:
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
        cuda_utils::set_device(m_device_id);
        m_nfreqs    = m_freq_arr.size();
        m_nsegments = m_nsamps / m_segment_len;
        // Allocate and compute phase map
        m_phase_map_d.resize(m_nfreqs * m_segment_len, 0U);
        compute_phase();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_fold_size() const {
        return m_nsegments * m_nfreqs * 2 * m_nbins;
    }

    void execute_h(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   std::span<float> fold) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());

        // Copy input data to device
        cudaStream_t stream = nullptr;
        thrust::device_vector<float> ts_e_d(ts_e.begin(), ts_e.end());
        thrust::device_vector<float> ts_v_d(ts_v.begin(), ts_v.end());
        thrust::device_vector<float> fold_d(fold.size(), 0.0F);

        // Execute folding kernel
        execute_d(cuda::std::span<const float>(
                      thrust::raw_pointer_cast(ts_e_d.data()), ts_e_d.size()),
                  cuda::std::span<const float>(
                      thrust::raw_pointer_cast(ts_v_d.data()), ts_v_d.size()),
                  cuda::std::span<float>(
                      thrust::raw_pointer_cast(fold_d.data()), fold_d.size()),
                  stream);

        // Copy result back to host
        thrust::copy(fold_d.begin(), fold_d.end(), fold.begin());
        spdlog::debug("BruteFoldCUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e,
                   cuda::std::span<const float> ts_v,
                   cuda::std::span<float> fold,
                   cudaStream_t stream) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());
        // Ensure output fold is zeroed
        thrust::fill(thrust::device, fold.begin(), fold.end(), 0.0F);
        execute_device(ts_e.data(), ts_v.data(), fold.data(), stream);
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
    thrust::device_vector<uint32_t> m_phase_map_d;

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
        thrust::device_vector<double> freq_arr_d(m_freq_arr);
        const int total_elements = static_cast<int>(m_nfreqs * m_segment_len);
        auto first               = thrust::counting_iterator<int>(0);
        auto last = thrust::counting_iterator<int>(total_elements);
        ComputePhase functor{
            .freq_arr    = thrust::raw_pointer_cast(freq_arr_d.data()),
            .tsamp       = m_tsamp,
            .t_ref       = m_t_ref,
            .segment_len = static_cast<int>(m_segment_len),
            .nbins       = static_cast<int>(m_nbins),
        };
        thrust::transform(thrust::device, first, last, m_phase_map_d.begin(),
                          functor);
        cuda_utils::check_last_cuda_error("thrust::transform failed");
        spdlog::debug("BruteFoldCUDA::Impl: Phase map computed");
    }

    void execute_device(const float* __restrict__ ts_e_d,
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
            kernel_fold_segments<<<grid_dim, block_dim, 0, stream>>>(
                ts_e_d, ts_v_d, thrust::raw_pointer_cast(m_phase_map_d.data()),
                fold_d, static_cast<int>(m_nfreqs),
                static_cast<int>(m_segment_len), static_cast<int>(m_nbins),
                static_cast<int>(m_nsegments));
        } else if (m_nbins <= 512) {
            // Use shared memory for small bin counts
            const dim3 block_dim(256);
            const dim3 grid_dim(1, m_nfreqs);
            const auto shared_mem_size =
                (2 * m_nbins * static_cast<int>(sizeof(float)));

            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
            kernel_fold_shared_mem<<<grid_dim, block_dim, shared_mem_size,
                                     stream>>>(
                ts_e_d, ts_v_d, thrust::raw_pointer_cast(m_phase_map_d.data()),
                fold_d, static_cast<int>(m_nfreqs),
                static_cast<int>(m_segment_len), static_cast<int>(m_nbins),
                static_cast<int>(m_nsegments));
        } else {
            // Use 2D block configuration for large nfreqs
            const dim3 block_dim(256);
            const dim3 grid_dim((m_segment_len + block_dim.x - 1) / block_dim.x,
                                static_cast<int>(m_nfreqs), 1);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
            kernel_fold_2d<<<grid_dim, block_dim, 0, stream>>>(
                ts_e_d, ts_v_d, thrust::raw_pointer_cast(m_phase_map_d.data()),
                fold_d, static_cast<int>(m_nfreqs),
                static_cast<int>(m_segment_len), static_cast<int>(m_nbins),
                static_cast<int>(m_nsegments));
        }
        cuda_utils::check_last_cuda_error("kernel_fold launch failed");
    }
}; // End BruteFoldCUDA::Impl definition

BruteFoldCUDA::BruteFoldCUDA(std::span<const double> freq_arr,
                             SizeType segment_len,
                             SizeType nbins,
                             SizeType nsamps,
                             double tsamp,
                             double t_ref,
                             int device_id)
    : m_impl(std::make_unique<Impl>(
          freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, device_id)) {}

BruteFoldCUDA::~BruteFoldCUDA()                              = default;
BruteFoldCUDA::BruteFoldCUDA(BruteFoldCUDA&& other) noexcept = default;
BruteFoldCUDA&
BruteFoldCUDA::operator=(BruteFoldCUDA&& other) noexcept = default;
SizeType BruteFoldCUDA::get_fold_size() const {
    return m_impl->get_fold_size();
}
void BruteFoldCUDA::execute(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            std::span<float> fold) {
    m_impl->execute_h(ts_e, ts_v, fold);
}
void BruteFoldCUDA::execute(cuda::std::span<const float> ts_e,
                            cuda::std::span<const float> ts_v,
                            cuda::std::span<float> fold,
                            cudaStream_t stream) {
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

std::vector<float> compute_brute_fold_cuda(std::span<const float> ts_e,
                                           std::span<const float> ts_v,
                                           std::span<const double> freq_arr,
                                           SizeType segment_len,
                                           SizeType nbins,
                                           double tsamp,
                                           double t_ref,
                                           int device_id) {
    const SizeType nsamps = ts_e.size();
    BruteFoldCUDA bf(freq_arr, segment_len, nbins, nsamps, tsamp, t_ref,
                     device_id);
    std::vector<float> fold(bf.get_fold_size(), 0.0F);
    bf.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}

} // namespace loki::algorithms