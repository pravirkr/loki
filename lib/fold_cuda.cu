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
    double norm_phase = fmod((proper_time - delay) * freq, 1.0);
    if (norm_phase < 0.0) {
        norm_phase += 1.0;
    }
    auto scaled_phase =
        static_cast<float>(norm_phase * static_cast<double>(nbins));
    if (scaled_phase >= static_cast<float>(nbins)) {
        scaled_phase = 0.0F;
    }
    const auto final_idx = __float2int_rn(scaled_phase);
    return final_idx;
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

// CUDA kernel for Complex BruteFold
__global__ void kernel_fold_segments_complex(const float* __restrict__ ts_e,
                                             const float* __restrict__ ts_v,
                                             ComplexTypeCUDA* __restrict__ fold,
                                             const double* __restrict__ freqs,
                                             int nfreqs,
                                             int nsegments,
                                             int segment_len,
                                             int nbins,
                                             double tsamp,
                                             double t_ref) {
    const auto iseg  = static_cast<int>(blockIdx.x);
    const auto ifreq = static_cast<int>(blockIdx.y);
    if (iseg >= nsegments || ifreq >= nfreqs) {
        return;
    }

    const auto tid      = static_cast<int>(threadIdx.x);
    const int nbins_f   = (nbins / 2) + 1;
    const int num_harms = nbins_f - 1;
    const int m         = tid + 1;
    if (m > num_harms) {
        return;
    }

    extern __shared__ float sh[];
    float* sh_e     = sh;
    float* sh_v     = sh + segment_len;
    float* sh_red_e = sh + (2 * segment_len);
    float* sh_red_v = sh + (2 * segment_len) + blockDim.x;

    const auto start_idx = iseg * segment_len;
    const auto block_dim = static_cast<int>(blockDim.x);
    for (int i = tid; i < segment_len; i += block_dim) {
        sh_e[i] = ts_e[start_idx + i];
        sh_v[i] = ts_v[start_idx + i];
    }
    __syncthreads();

    // Compute AC for this harmonic
    const double two_pi      = 2.0 * M_PI;
    const double freq        = freqs[ifreq];
    const double harm        = static_cast<double>(m);
    const double fm          = freq * harm;
    const double delta_phase = -two_pi * fm * tsamp;
    const double init_phase  = two_pi * fm * t_ref;
    double init_phase_mod    = fmod(init_phase, two_pi);
    if (init_phase_mod < -M_PI) {
        init_phase_mod += two_pi;
    } else if (init_phase_mod > M_PI) {
        init_phase_mod -= two_pi;
    }
    float ph_r         = cosf(static_cast<float>(init_phase_mod));
    float ph_i         = sinf(static_cast<float>(init_phase_mod));
    const float step_r = cosf(static_cast<float>(delta_phase));
    const float step_i = sinf(static_cast<float>(delta_phase));

    float acc_e_r = 0.0F;
    float acc_e_i = 0.0F;
    float acc_v_r = 0.0F;
    float acc_v_i = 0.0F;

    for (int k = 0; k < segment_len; ++k) {
        acc_e_r = fmaf(sh_e[k], ph_r, acc_e_r);
        acc_e_i = fmaf(sh_e[k], ph_i, acc_e_i);
        acc_v_r = fmaf(sh_v[k], ph_r, acc_v_r);
        acc_v_i = fmaf(sh_v[k], ph_i, acc_v_i);

        const float new_r = (ph_r * step_r) - (ph_i * step_i);
        const float new_i = (ph_r * step_i) + (ph_i * step_r);
        ph_r              = new_r;
        ph_i              = new_i;
    }

    const auto seg_offset                     = iseg * nfreqs * (2 * nbins_f);
    const auto freq_offset_out                = ifreq * (2 * nbins_f);
    ComplexTypeCUDA* __restrict__ fold_seg    = fold + seg_offset;
    ComplexTypeCUDA* __restrict__ fold_e_base = fold_seg + freq_offset_out;
    ComplexTypeCUDA* __restrict__ fold_v_base = fold_e_base + nbins_f;
    fold_e_base[m] = ComplexTypeCUDA(acc_e_r, acc_e_i);
    fold_v_base[m] = ComplexTypeCUDA(acc_v_r, acc_v_i);

    // Compute DC sum
    float local_e = 0.0F;
    float local_v = 0.0F;
    for (int i = tid; i < segment_len; i += block_dim) {
        local_e += sh_e[i];
        local_v += sh_v[i];
    }
    sh_red_e[tid] = local_e;
    sh_red_v[tid] = local_v;
    __syncthreads();

    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_red_e[tid] += sh_red_e[tid + s];
            sh_red_v[tid] += sh_red_v[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        fold_e_base[0] = ComplexTypeCUDA(sh_red_e[0], 0.0F);
        fold_v_base[0] = ComplexTypeCUDA(sh_red_v[0], 0.0F);
    }
}

} // namespace

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class BruteFoldCUDA<FoldTypeCUDA>::Impl {
public:
    using HostFoldType   = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = typename FoldTypeTraits<FoldTypeCUDA>::DeviceType;

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
        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            // Time domain - allocate phase map
            m_phase_map_d.resize(m_nfreqs * m_segment_len, 0U);
            compute_phase();
        } else {
            // Fourier domain
            m_nbins_f    = (nbins / 2) + 1;
            m_freq_arr_d = thrust::device_vector<double>(m_freq_arr);
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

        // Copy input data to device
        cudaStream_t stream = nullptr;
        thrust::device_vector<float> ts_e_d(ts_e.begin(), ts_e.end());
        thrust::device_vector<float> ts_v_d(ts_v.begin(), ts_v.end());

        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            thrust::device_vector<float> fold_d(fold.size(), 0.0F);
            // Execute folding kernel
            execute_d(
                cuda::std::span<const float>(
                    thrust::raw_pointer_cast(ts_e_d.data()), ts_e_d.size()),
                cuda::std::span<const float>(
                    thrust::raw_pointer_cast(ts_v_d.data()), ts_v_d.size()),
                cuda::std::span<float>(thrust::raw_pointer_cast(fold_d.data()),
                                       fold_d.size()),
                stream);
            // Copy result back to host
            thrust::copy(fold_d.begin(), fold_d.end(), fold.begin());
        } else {
            thrust::device_vector<ComplexTypeCUDA> fold_d(
                fold.size(), ComplexTypeCUDA(0.0F, 0.0F));
            // Execute folding kernel
            execute_d(
                cuda::std::span<const float>(
                    thrust::raw_pointer_cast(ts_e_d.data()), ts_e_d.size()),
                cuda::std::span<const float>(
                    thrust::raw_pointer_cast(ts_v_d.data()), ts_v_d.size()),
                cuda::std::span<ComplexTypeCUDA>(
                    thrust::raw_pointer_cast(fold_d.data()), fold_d.size()),
                stream);
            // Copy result back to host
            thrust::copy(fold_d.begin(), fold_d.end(), fold.begin());
        }
        spdlog::debug("BruteFoldCUDA::Impl: Host execution complete");
    }

    void execute_d(cuda::std::span<const float> ts_e,
                   cuda::std::span<const float> ts_v,
                   cuda::std::span<DeviceFoldType> fold,
                   cudaStream_t stream) {
        check_inputs(ts_e.size(), ts_v.size(), fold.size());
        // Ensure output fold is zeroed
        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            thrust::fill(thrust::device, fold.begin(), fold.end(), 0.0F);
            execute_device_float(ts_e.data(), ts_v.data(), fold.data(), stream);
        } else {
            thrust::fill(thrust::device, fold.begin(), fold.end(),
                         ComplexTypeCUDA(0.0F, 0.0F));
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

    // Time domain only
    thrust::device_vector<uint32_t> m_phase_map_d;

    // Fourier domain only
    SizeType m_nbins_f;
    thrust::device_vector<double> m_freq_arr_d;

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

    void execute_device_complex(const float* __restrict__ ts_e_d,
                                const float* __restrict__ ts_v_d,
                                ComplexTypeCUDA* __restrict__ fold_d,
                                cudaStream_t stream) {
        const int num_harms = static_cast<int>(m_nbins_f) - 1;
        const dim3 grid(m_nsegments, m_nfreqs);
        const int block_size = num_harms;
        const size_t shmem_size =
            (2 * m_segment_len + (2 * block_size)) * sizeof(float);
        kernel_fold_segments_complex<<<grid, block_size, shmem_size, stream>>>(
            ts_e_d, ts_v_d, fold_d, thrust::raw_pointer_cast(m_freq_arr_d.data()),
            static_cast<int>(m_nfreqs), static_cast<int>(m_nsegments),
            static_cast<int>(m_segment_len), static_cast<int>(m_nbins), m_tsamp,
            m_t_ref);
        cuda_utils::check_last_cuda_error(
            "kernel_fold_segments_complex launch failed");
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
    using HostType        = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    const SizeType nsamps = ts_e.size();
    BruteFoldCUDA<FoldTypeCUDA> bf(freq_arr, segment_len, nbins, nsamps, tsamp,
                                   t_ref, device_id);
    std::vector<HostType> fold(bf.get_fold_size());
    if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
        std::fill(fold.begin(), fold.end(), 0.0F);
    } else {
        std::fill(fold.begin(), fold.end(), HostType(0.0F, 0.0F));
    }
    bf.execute(ts_e, ts_v, std::span<HostType>(fold));
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