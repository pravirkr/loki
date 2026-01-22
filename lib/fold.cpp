#include "loki/algorithms/fold.hpp"

#include <algorithm>
#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
#include "loki/psr_utils.hpp"

namespace loki::algorithms {

template <SupportedFoldType FoldType> class BruteFold<FoldType>::Impl {
public:
    Impl(std::span<const double> freq_arr,
         SizeType segment_len,
         SizeType nbins,
         SizeType nsamps,
         double tsamp,
         double t_ref,
         int nthreads)
        : m_freq_arr(freq_arr.begin(), freq_arr.end()),
          m_segment_len(segment_len),
          m_nbins(nbins),
          m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_t_ref(t_ref),
          m_nthreads(nthreads) {
        error_check::check(!m_freq_arr.empty(),
                           "BruteFold::Impl: Frequency array is empty");
        error_check::check_equal(m_nsamps % m_segment_len, 0,
                                 "BruteFold::Impl: Number of samples is not a "
                                 "multiple of segment length");
        m_nthreads  = std::clamp(m_nthreads, 1, omp_get_max_threads());
        m_nfreqs    = m_freq_arr.size();
        m_nsegments = m_nsamps / m_segment_len;
        m_nbins_f   = (nbins / 2) + 1;
        if constexpr (std::is_same_v<FoldType, float>) {
            // Allocate and compute phase map for time domain
            m_phase_map.resize(m_nfreqs * m_segment_len);
            m_bucket_indices.resize(m_nfreqs * m_segment_len);
            m_offsets.resize((m_nfreqs * m_nbins) + 1);
            compute_phase_time_domain();
        }
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_fold_size() const {
        if constexpr (std::is_same_v<FoldType, float>) {
            return m_nsegments * m_nfreqs * 2 * m_nbins;
        } else {
            return m_nsegments * m_nfreqs * 2 * m_nbins_f;
        }
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) {
        error_check::check_equal(
            ts_e.size(), m_nsamps,
            "BruteFold::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "BruteFold::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), get_fold_size(),
            "BruteFold::Impl::execute: fold must have size fold_size");

        // Ensure output fold is zeroed
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            std::ranges::fill(fold, ComplexType(0.0F, 0.0F));
        } else {
            std::ranges::fill(fold, 0.0F);
        }
        if constexpr (std::is_same_v<FoldType, float>) {
            kernels::brute_fold_ts(ts_e.data(), ts_v.data(), fold.data(),
                                   m_bucket_indices.data(), m_offsets.data(),
                                   m_nsegments, m_nfreqs, m_segment_len,
                                   m_nbins, m_nthreads);

        } else {
            kernels::brute_fold_ts_complex(
                ts_e.data(), ts_v.data(), fold.data(), m_freq_arr.data(),
                m_nfreqs, m_nsegments, m_segment_len, m_nbins, m_tsamp, m_t_ref,
                m_nthreads);
        }
    }

private:
    std::vector<double> m_freq_arr;
    SizeType m_segment_len;
    SizeType m_nbins;
    SizeType m_nsamps;
    double m_tsamp;
    double m_t_ref;
    int m_nthreads;

    SizeType m_nfreqs;
    SizeType m_nsegments;
    SizeType m_nbins_f;

    // Time domain only
    std::vector<uint32_t> m_phase_map;
    std::vector<uint32_t> m_bucket_indices;
    std::vector<SizeType> m_offsets;

    // Time domain only
    void compute_phase_time_domain() {
        const SizeType total_buckets = m_nfreqs * m_nbins;
        std::vector<SizeType> counts(total_buckets, 0);
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            const auto freq_offset_in = ifreq * m_segment_len;
            for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
                const auto proper_time =
                    (static_cast<double>(isamp) * m_tsamp) - m_t_ref;
                const auto iphase = psr_utils::get_phase_idx(
                    proper_time, m_freq_arr[ifreq], m_nbins, 0.0);
                const auto iphase_int =
                    static_cast<uint32_t>(std::nearbyint(iphase)) % m_nbins;
                m_phase_map[freq_offset_in + isamp] = iphase_int;
                const auto bucket_idx = (ifreq * m_nbins) + iphase_int;
                ++counts[bucket_idx];
            }
        }
        // Compute prefix sum for offsets
        for (SizeType i = 1; i <= total_buckets; ++i) {
            m_offsets[i] = m_offsets[i - 1] + counts[i - 1];
        }
        // Second pass: fill the indices
        std::vector<SizeType> writers = m_offsets; // Copy for writing positions
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            const auto freq_offset_in = ifreq * m_segment_len;
            for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
                const auto iphase     = m_phase_map[freq_offset_in + isamp];
                const auto bucket_idx = (ifreq * m_nbins) + iphase;
                m_bucket_indices[writers[bucket_idx]++] = isamp;
            }
        }
    }

}; // End BruteFold::Impl definition

template <SupportedFoldType FoldType>
BruteFold<FoldType>::BruteFold(std::span<const double> freq_arr,
                               SizeType segment_len,
                               SizeType nbins,
                               SizeType nsamps,
                               double tsamp,
                               double t_ref,
                               int nthreads)
    : m_impl(std::make_unique<Impl>(
          freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, nthreads)) {}
template <SupportedFoldType FoldType>
BruteFold<FoldType>::~BruteFold() = default;
template <SupportedFoldType FoldType>
BruteFold<FoldType>::BruteFold(BruteFold&& other) noexcept = default;
template <SupportedFoldType FoldType>
BruteFold<FoldType>&
BruteFold<FoldType>::operator=(BruteFold&& other) noexcept = default;
template <SupportedFoldType FoldType>
SizeType BruteFold<FoldType>::get_fold_size() const {
    return m_impl->get_fold_size();
}

template <SupportedFoldType FoldType>
void BruteFold<FoldType>::execute(std::span<const float> ts_e,
                                  std::span<const float> ts_v,
                                  std::span<FoldType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}

template <SupportedFoldType FoldType>
std::vector<FoldType> compute_brute_fold(std::span<const float> ts_e,
                                         std::span<const float> ts_v,
                                         std::span<const double> freq_arr,
                                         SizeType segment_len,
                                         SizeType nbins,
                                         double tsamp,
                                         double t_ref,
                                         int nthreads) {
    const SizeType nsamps = ts_e.size();
    BruteFold<FoldType> bf(freq_arr, segment_len, nbins, nsamps, tsamp, t_ref,
                           nthreads);
    std::vector<FoldType> fold(bf.get_fold_size(), FoldType{});
    bf.execute(ts_e, ts_v, std::span<FoldType>(fold));
    return fold;
}

// Explicit instantiation
template class BruteFold<float>;
template class BruteFold<ComplexType>;

template std::vector<float> compute_brute_fold<float>(std::span<const float>,
                                                      std::span<const float>,
                                                      std::span<const double>,
                                                      SizeType,
                                                      SizeType,
                                                      double,
                                                      double,
                                                      int);
template std::vector<ComplexType>
compute_brute_fold<ComplexType>(std::span<const float>,
                                std::span<const float>,
                                std::span<const double>,
                                SizeType,
                                SizeType,
                                double,
                                double,
                                int);

} // namespace loki::algorithms