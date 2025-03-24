#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <loki/loki_types.hpp>

/**
 * @brief Fold time series using brute-force method
 *
 */
class BruteFold {
public:
    BruteFold(std::span<const FloatType> freq_arr,
              SizeType segment_len,
              SizeType nbins,
              SizeType nsamps,
              FloatType tsamp,
              FloatType t_ref   = 0.0F,
              SizeType nthreads = 1);
    BruteFold(const BruteFold&)            = delete;
    BruteFold& operator=(const BruteFold&) = delete;
    BruteFold(BruteFold&&)                 = delete;
    BruteFold& operator=(BruteFold&&)      = delete;
    ~BruteFold()                           = default;

    SizeType get_fold_size() const;
    /**
     * @brief Fold time series using brute-force method
     *
     * @param ts_e Time series signal
     * @param ts_v Time series variance
     * @param fold  Folded time series with shape [nsegments, nfreqs, 2, nbins]
     */
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);

private:
    std::vector<FloatType> m_freq_arr;
    SizeType m_segment_len;
    SizeType m_nbins;
    SizeType m_nsamps;
    FloatType m_tsamp;
    FloatType m_t_ref;
    SizeType m_nthreads;

    SizeType m_nfreqs;
    SizeType m_nsegments;
    std::vector<uint32_t> m_phase_map;

    void compute_phase();
    void execute_segment(std::span<const float> ts_e_seg,
                         std::span<const float> ts_v_seg,
                         std::span<float> fold_seg);
};

/* Convenience function to fold time series using brute-force method */
std::vector<float> compute_brute_fold(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      std::span<const FloatType> freq_arr,
                                      SizeType segment_len,
                                      SizeType nbins,
                                      SizeType nsamps,
                                      FloatType tsamp,
                                      FloatType t_ref   = 0.0,
                                      SizeType nthreads = 1);