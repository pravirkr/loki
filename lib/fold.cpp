#include "loki/algorithms/fold.hpp"

#include <algorithm>
#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"

namespace loki::algorithms {

class BruteFold::Impl {
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
        error_check::check(m_nsamps % m_segment_len == 0,
                           "BruteFold::Impl: Number of samples is not a "
                           "multiple of segment length");
        m_nthreads  = std::clamp(m_nthreads, 1, omp_get_max_threads());
        m_nfreqs    = m_freq_arr.size();
        m_nsegments = m_nsamps / m_segment_len;
        // Allocate and compute phase map
        m_phase_map.resize(m_nfreqs * m_segment_len);
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

    SizeType get_fold_size_stride() const {
        // Extra padding for in-place RFFT
        return m_nsegments * m_nfreqs * 2 * (m_nbins + 2);
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold) {
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
        std::ranges::fill(fold, 0.0F);

#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(ts_e, ts_v, fold, m_segment_len, m_nfreqs, m_nbins, m_nsegments)
        for (SizeType iseg = 0; iseg < m_nsegments; ++iseg) {
            const auto ts_e_seg =
                ts_e.subspan(iseg * m_segment_len, m_segment_len);
            const auto ts_v_seg =
                ts_v.subspan(iseg * m_segment_len, m_segment_len);
            auto fold_seg = fold.subspan(iseg * m_nfreqs * 2 * m_nbins,
                                         m_nfreqs * 2 * m_nbins);
            execute_segment(ts_e_seg.data(), ts_v_seg.data(), fold_seg.data(),
                            m_segment_len, m_nbins);
        }
    }

    void execute_stride(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        std::span<float> fold) {
        error_check::check_equal(
            ts_e.size(), m_nsamps,
            "BruteFold::Impl::execute_stride: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "BruteFold::Impl::execute_stride: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), get_fold_size_stride(),
            "BruteFold::Impl::execute_stride: fold must have size "
            "fold_size_stride");
        // Ensure output fold is zeroed
        std::ranges::fill(fold, 0.0F);
        const auto profile_stride = m_nbins + 2;

#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(ts_e, ts_v, fold, m_segment_len, m_nfreqs, m_nsegments,             \
               profile_stride)
        for (SizeType iseg = 0; iseg < m_nsegments; ++iseg) {
            const auto ts_e_seg =
                ts_e.subspan(iseg * m_segment_len, m_segment_len);
            const auto ts_v_seg =
                ts_v.subspan(iseg * m_segment_len, m_segment_len);
            auto fold_seg = fold.subspan(iseg * m_nfreqs * 2 * profile_stride,
                                         m_nfreqs * 2 * profile_stride);
            execute_segment(ts_e_seg.data(), ts_v_seg.data(), fold_seg.data(),
                            m_segment_len, profile_stride);
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
    std::vector<uint32_t> m_phase_map;

    void compute_phase() {
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            const auto freq_offset_in = ifreq * m_segment_len;
            for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
                const auto proper_time =
                    (static_cast<double>(isamp) * m_tsamp) - m_t_ref;
                m_phase_map[freq_offset_in + isamp] =
                    static_cast<uint32_t>(psr_utils::get_phase_idx_int(
                        proper_time, m_freq_arr[ifreq], m_nbins, 0.0));
            }
        }
    }

    void execute_segment(const float* __restrict__ ts_e_seg,
                         const float* __restrict__ ts_v_seg,
                         float* __restrict__ fold_seg,
                         SizeType segment_len_act,
                         SizeType profile_stride) noexcept {
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            const auto freq_offset_in          = ifreq * m_segment_len;
            const auto freq_offset_out         = ifreq * 2 * profile_stride;
            const auto* __restrict__ phase_ptr = &m_phase_map[freq_offset_in];
            float* __restrict__ fold_e_base    = fold_seg + freq_offset_out;
            float* __restrict__ fold_v_base    = fold_e_base + profile_stride;

            const SizeType main_loop =
                segment_len_act - (segment_len_act % kUnrollFactor);
            for (SizeType isamp = 0; isamp < main_loop;
                 isamp += kUnrollFactor) {
                UNROLL_VECTORIZE
                for (SizeType j = 0; j < kUnrollFactor; ++j) {
                    const auto idx    = isamp + j;
                    const auto iphase = phase_ptr[idx];
                    fold_e_base[iphase] += ts_e_seg[idx];
                    fold_v_base[iphase] += ts_v_seg[idx];
                }
            }
            for (SizeType isamp = main_loop; isamp < segment_len_act; ++isamp) {
                const auto iphase = phase_ptr[isamp];
                fold_e_base[iphase] += ts_e_seg[isamp];
                fold_v_base[iphase] += ts_v_seg[isamp];
            }
        }
    }

}; // End BruteFold::Impl definition

BruteFold::BruteFold(std::span<const double> freq_arr,
                     SizeType segment_len,
                     SizeType nbins,
                     SizeType nsamps,
                     double tsamp,
                     double t_ref,
                     int nthreads)
    : m_impl(std::make_unique<Impl>(
          freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, nthreads)) {}
BruteFold::~BruteFold()                                     = default;
BruteFold::BruteFold(BruteFold&& other) noexcept            = default;
BruteFold& BruteFold::operator=(BruteFold&& other) noexcept = default;
SizeType BruteFold::get_fold_size() const { return m_impl->get_fold_size(); }
SizeType BruteFold::get_fold_size_stride() const {
    return m_impl->get_fold_size_stride();
}

void BruteFold::execute(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        std::span<float> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}
void BruteFold::execute_stride(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<float> fold) {
    m_impl->execute_stride(ts_e, ts_v, fold);
}

std::vector<float> compute_brute_fold(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      std::span<const double> freq_arr,
                                      SizeType segment_len,
                                      SizeType nbins,
                                      double tsamp,
                                      double t_ref,
                                      int nthreads) {
    const SizeType nsamps = ts_e.size();
    BruteFold bf(freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, nthreads);
    std::vector<float> fold(bf.get_fold_size(), 0.0F);
    bf.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}

} // namespace loki::algorithms