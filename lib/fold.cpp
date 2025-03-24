#include <loki/fold.hpp>

#include <algorithm>
#include <thread>

#include <spdlog/spdlog.h>

#include <loki/loki_types.hpp>
#include <loki/psr_utils.hpp>
#include <loki/utils.hpp>

BruteFold::BruteFold(std::span<const FloatType> freq_arr,
                     SizeType segment_len,
                     SizeType nbins,
                     SizeType nsamps,
                     FloatType tsamp,
                     FloatType t_ref,
                     SizeType nthreads)
    : m_freq_arr(freq_arr.begin(), freq_arr.end()),
      m_segment_len(segment_len),
      m_nbins(nbins),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_t_ref(t_ref),
      m_nthreads(nthreads) {
    if (m_freq_arr.empty()) {
        throw std::runtime_error("Frequency array is empty");
    }
    if (m_nsamps % m_segment_len != 0) {
        throw std::runtime_error(
            "Number of samples is not a multiple of segment length");
    }
    m_nthreads = std::max<SizeType>(m_nthreads, 1);
    m_nthreads =
        std::min<SizeType>(m_nthreads, std::thread::hardware_concurrency());
    m_nfreqs    = m_freq_arr.size();
    m_nsegments = m_nsamps / m_segment_len;
    m_phase_map.resize(m_nfreqs * m_segment_len);
    compute_phase();
}

SizeType BruteFold::get_fold_size() const {
    return m_nsegments * m_nfreqs * 2 * m_nbins;
}

void BruteFold::execute(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        std::span<float> fold) {
    if (ts_e.size() != m_nsamps) {
        throw std::runtime_error("Input array has wrong size");
    }
    if (ts_v.size() != ts_e.size()) {
        throw std::runtime_error("Input variance array has wrong size");
    }
    if (fold.size() != get_fold_size()) {
        throw std::runtime_error("Output array has wrong size");
    }
#pragma omp parallel for num_threads(m_nthreads)
    for (SizeType iseg = 0; iseg < m_nsegments; ++iseg) {
        const auto ts_e_seg = ts_e.subspan(iseg * m_segment_len, m_segment_len);
        const auto ts_v_seg = ts_v.subspan(iseg * m_segment_len, m_segment_len);
        auto fold_seg =
            fold.subspan(iseg * m_nfreqs * 2 * m_nbins, m_nfreqs * 2 * m_nbins);
        execute_segment(ts_e_seg, ts_v_seg, fold_seg);
    }
}

void BruteFold::compute_phase() {
    for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
        const auto freq_offset_in = ifreq * m_segment_len;
        for (SizeType isamp = 0; isamp < m_segment_len; ++isamp) {
            const auto proper_time =
                (static_cast<FloatType>(isamp) * m_tsamp) - m_t_ref;
            m_phase_map[freq_offset_in + isamp] =
                static_cast<uint32_t>(loki::utils::get_phase_idx(
                    proper_time, m_freq_arr[ifreq], m_nbins, 0.0));
        }
    }
}

void BruteFold::execute_segment(std::span<const float> ts_e_seg,
                                std::span<const float> ts_v_seg,
                                std::span<float> fold_seg) {
    for (SizeType ifreq = 0; ifreq < m_nfreqs; ++ifreq) {
        const auto freq_offset_in  = ifreq * m_segment_len;
        const auto freq_offset_out = ifreq * 2 * m_nbins;
        const auto segment_len_act = ts_e_seg.size();
        for (SizeType isamp = 0; isamp < segment_len_act; ++isamp) {
            const auto iphase    = m_phase_map[freq_offset_in + isamp];
            const auto out_idx_e = freq_offset_out + iphase;
            const auto out_idx_v = freq_offset_out + iphase + m_nbins;
            fold_seg[out_idx_e] += ts_e_seg[isamp];
            fold_seg[out_idx_v] += ts_v_seg[isamp];
        }
    }
}

std::vector<float> compute_brute_fold(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      std::span<const FloatType> freq_arr,
                                      SizeType segment_len,
                                      SizeType nbins,
                                      SizeType nsamps,
                                      FloatType tsamp,
                                      FloatType t_ref,
                                      SizeType nthreads) {
    BruteFold bf(freq_arr, segment_len, nbins, nsamps, tsamp, t_ref, nthreads);
    std::vector<float> fold(bf.get_fold_size(), 0.0F);
    bf.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}