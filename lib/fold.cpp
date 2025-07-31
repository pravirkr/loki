#include "loki/algorithms/fold.hpp"

#include <algorithm>
#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
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
        error_check::check_equal(m_nsamps % m_segment_len, 0,
                                 "BruteFold::Impl: Number of samples is not a "
                                 "multiple of segment length");
        m_nthreads  = std::clamp(m_nthreads, 1, omp_get_max_threads());
        m_nfreqs    = m_freq_arr.size();
        m_nsegments = m_nsamps / m_segment_len;
        // Allocate and compute phase map
        m_phase_map.resize(m_nfreqs * m_segment_len);
        m_bucket_indices.resize(m_nfreqs * m_segment_len);
        m_offsets.resize((m_nfreqs * m_nbins) + 1);
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

        const auto* __restrict__ ts_e_ptr = ts_e.data();
        const auto* __restrict__ ts_v_ptr = ts_v.data();
        auto* __restrict__ fold_ptr       = fold.data();

        kernels::brute_fold_ts(ts_e_ptr, ts_v_ptr, fold_ptr,
                               m_bucket_indices.data(), m_offsets.data(),
                               m_nsegments, m_nfreqs, m_segment_len, m_nbins,
                               m_nthreads);
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
    std::vector<uint32_t> m_bucket_indices;
    std::vector<SizeType> m_offsets;

    void compute_phase() {
        const SizeType total_buckets = m_nfreqs * m_nbins;
        std::vector<SizeType> counts(total_buckets, 0);
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            const auto freq_offset_in = ifreq * m_segment_len;
            for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
                const auto proper_time =
                    (static_cast<double>(isamp) * m_tsamp) - m_t_ref;
                const auto iphase =
                    static_cast<uint32_t>(psr_utils::get_phase_idx_int(
                        proper_time, m_freq_arr[ifreq], m_nbins, 0.0));
                m_phase_map[freq_offset_in + isamp] = iphase;
                const auto bucket_idx = (ifreq * m_nbins) + iphase;
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
        // Sort each bucket's indices for monotonic access (better
        // cache/prefetch)
        for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
            for (SizeType iphase = 0; iphase < m_nbins; ++iphase) {
                const auto bucket_idx = (ifreq * m_nbins) + iphase;
                const auto start      = m_offsets[bucket_idx];
                const auto end        = m_offsets[bucket_idx + 1];
                std::sort(m_bucket_indices.data() + start,
                          m_bucket_indices.data() + end);
            }
        }
    }

}; // End BruteFold::Impl definition

class BruteFoldComplex::Impl {
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
        m_nbins_f   = (nbins / 2) + 1;
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_fold_size() const {
        return m_nsegments * m_nfreqs * 2 * m_nbins_f;
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<ComplexType> fold) {
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
        std::ranges::fill(fold, ComplexType(0.0F, 0.0F));

        const auto* __restrict__ ts_e_ptr = ts_e.data();
        const auto* __restrict__ ts_v_ptr = ts_v.data();
        auto* __restrict__ fold_ptr       = fold.data();

        kernels::brute_fold_ts_complex(
            ts_e_ptr, ts_v_ptr, fold_ptr, m_freq_arr.data(), m_nfreqs,
            m_nsegments, m_segment_len, m_nbins, m_tsamp, m_t_ref, m_nthreads);
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

}; // End BruteFoldComplex::Impl definition

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

void BruteFold::execute(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        std::span<float> fold) {
    m_impl->execute(ts_e, ts_v, fold);
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

BruteFoldComplex::BruteFoldComplex(std::span<const double> freq_arr,
                                   SizeType segment_len,
                                   SizeType nbins,
                                   SizeType nsamps,
                                   double tsamp,
                                   double t_ref,
                                   int nthreads)
    : m_impl(std::make_unique<Impl>(
          freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, nthreads)) {}
BruteFoldComplex::~BruteFoldComplex()                                 = default;
BruteFoldComplex::BruteFoldComplex(BruteFoldComplex&& other) noexcept = default;
BruteFoldComplex&
BruteFoldComplex::operator=(BruteFoldComplex&& other) noexcept = default;
SizeType BruteFoldComplex::get_fold_size() const {
    return m_impl->get_fold_size();
}

void BruteFoldComplex::execute(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<ComplexType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}

std::vector<ComplexType>
compute_brute_fold_complex(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           std::span<const double> freq_arr,
                           SizeType segment_len,
                           SizeType nbins,
                           double tsamp,
                           double t_ref,
                           int nthreads) {
    const SizeType nsamps = ts_e.size();
    BruteFoldComplex bf(freq_arr, segment_len, nbins, nsamps, tsamp, t_ref,
                        nthreads);
    std::vector<ComplexType> fold(bf.get_fold_size(), 0.0F);
    bf.execute(ts_e, ts_v, std::span<ComplexType>(fold));
    return fold;
}

} // namespace loki::algorithms