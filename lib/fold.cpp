
#include <loki/fold.hpp>
#include <loki/utils.hpp>
#include <utility>

BruteFold::BruteFold(std::vector<float> freq_arr,
                     SizeType nbin,
                     SizeType segment_len,
                     float tsamp,
                     float t_ref)
    : m_freq_arr(std::move(freq_arr)),
      m_nbin(nbin),
      m_segment_len(segment_len),
      m_tsamp(tsamp),
      m_t_ref(t_ref) {
    if (m_freq_arr.empty()) {
        throw std::runtime_error("Frequency array is empty");
    }
    m_nfreqs = m_freq_arr.size();
    m_phase_map.resize(m_nfreqs * m_segment_len);
    compute_phase();
}

void BruteFold::execute(std::span<const float> ts, std::span<float> fold) {
    const auto nsamps = ts.size();
    if (nsamps % m_segment_len != 0) {
        throw std::runtime_error(
            "Number of samples is not a multiple of segment length");
    }
    const auto nsegments = nsamps / m_segment_len;
    if (fold.size() != nsegments * m_nfreqs * m_nbin) {
        throw std::runtime_error("Output array has wrong size");
    }
#pragma omp parallel for
    for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
        const auto ts_segment = ts.subspan(iseg * m_segment_len, m_segment_len);
        auto fold_segment =
            fold.subspan(iseg * m_nfreqs * m_nbin, m_nfreqs * m_nbin);
        execute_segment(ts_segment, fold_segment);
    }
}

void BruteFold::compute_phase() {
#pragma omp parallel for collapse(2)
    for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
        for (SizeType i = 0; i < m_segment_len; ++i) {
            const float proper_time = static_cast<float>(i) * m_tsamp - m_t_ref;
            m_phase_map[ifreq * m_segment_len + i] =
                loki::get_phase_idx(proper_time, m_freq_arr[ifreq], m_nbin, 0);
        }
    }
}

void BruteFold::execute_segment(std::span<const float> ts_segment,
                                std::span<float> fold_segment) {
#pragma omp parallel for collapse(2)
    for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
        for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
            const auto& iphase = m_phase_map[ifreq * m_segment_len + isamp];
            const auto out_idx = ifreq * m_nbin + iphase;
#pragma omp atomic
            fold_segment[out_idx] += ts_segment[isamp];
        }
    }
}