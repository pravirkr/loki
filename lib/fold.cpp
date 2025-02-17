#include <loki/fold.hpp>

#include <utility>

#include <loki/loki_types.hpp>
#include <loki/psr_utils.hpp>
#include <loki/utils.hpp>

BruteFold::BruteFold(std::vector<float> freq_arr,
                     SizeType segment_len,
                     SizeType nbins,
                     SizeType nsamps,
                     float tsamp,
                     float t_ref)
    : m_freq_arr(std::move(freq_arr)),
      m_segment_len(segment_len),
      m_nbins(nbins),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_t_ref(t_ref) {
    if (m_freq_arr.empty()) {
        throw std::runtime_error("Frequency array is empty");
    }
    if (m_nsamps % m_segment_len != 0) {
        throw std::runtime_error(
            "Number of samples is not a multiple of segment length");
    }
    m_nfreqs    = m_freq_arr.size();
    m_nsegments = m_nsamps / m_segment_len;
    m_phase_map.resize(m_nfreqs * m_segment_len);
    compute_phase();
}

SizeType BruteFold::get_fold_size() const {
    return m_nsegments * m_nfreqs * m_nbins;
}

void BruteFold::execute(std::span<const float> ts, std::span<float> fold) {
    if (ts.size() != m_nsamps) {
        throw std::runtime_error("Input array has wrong size");
    }
    if (fold.size() != get_fold_size()) {
        throw std::runtime_error("Output array has wrong size");
    }
#pragma omp parallel for
    for (SizeType iseg = 0; iseg < m_nsegments; ++iseg) {
        const auto ts_segment = ts.subspan(iseg * m_segment_len, m_segment_len);
        auto fold_segment =
            fold.subspan(iseg * m_nfreqs * m_nbins, m_nfreqs * m_nbins);
        execute_segment(ts_segment, fold_segment);
    }
}

void BruteFold::compute_phase() {
    for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
        for (SizeType i = 0; i < m_segment_len; ++i) {
            const float proper_time =
                (static_cast<float>(i) * m_tsamp) - m_t_ref;
            m_phase_map[(ifreq * m_segment_len) + i] =
                loki::utils::get_phase_idx(proper_time, m_freq_arr[ifreq],
                                           m_nbins, 0);
        }
    }
}

void BruteFold::execute_segment(std::span<const float> ts_segment,
                                std::span<float> fold_segment) {
    for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
        for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
            const auto& iphase = m_phase_map[(ifreq * m_segment_len) + isamp];
            const auto out_idx = (ifreq * m_nbins) + iphase;
#pragma omp atomic
            fold_segment[out_idx] += ts_segment[isamp];
        }
    }
}

void brute_fold(std::span<const float> ts,
                std::span<float> fold,
                std::vector<float> freq_arr,
                SizeType segment_len,
                SizeType nbins,
                SizeType nsamps,
                float tsamp,
                float t_ref) {
    BruteFold bf(std::move(freq_arr), segment_len, nbins, nsamps, tsamp, t_ref);
    bf.execute(ts, fold);
}