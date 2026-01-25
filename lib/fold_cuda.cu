#include "loki/algorithms/fold.hpp"

#include <memory>

#include <spdlog/spdlog.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/kernel_utils.cuh"
#include "loki/kernels_cuda.cuh"

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
            kernels::brute_fold_ts_cuda(
                ts_e.data(), ts_v.data(), fold.data(),
                thrust::raw_pointer_cast(m_phase_map_d.data()), m_nsegments,
                m_nfreqs, m_segment_len, m_nbins, stream);
        } else {
            kernels::brute_fold_ts_complex_cuda(
                ts_e.data(), ts_v.data(), fold.data(),
                thrust::raw_pointer_cast(m_freq_arr_d.data()), m_nfreqs,
                m_nsegments, m_segment_len, m_nbins_f, m_tsamp, m_t_ref,
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