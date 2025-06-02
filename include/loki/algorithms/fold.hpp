#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::algorithms {

/**
 * @brief Fold time series using brute-force method
 *
 */
class BruteFold {
public:
    BruteFold(std::span<const double> freq_arr,
              SizeType segment_len,
              SizeType nbins,
              SizeType nsamps,
              double tsamp,
              double t_ref = 0.0,
              int nthreads = 1);
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

    void compute_phase();
    void execute_segment(const float* __restrict__ ts_e_seg,
                         const float* __restrict__ ts_v_seg,
                         float* __restrict__ fold_seg,
                         SizeType segment_len_act) noexcept;
};

/* Convenience function to fold time series using brute-force method */
std::vector<float> compute_brute_fold(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      std::span<const double> freq_arr,
                                      SizeType segment_len,
                                      SizeType nbins,
                                      double tsamp,
                                      double t_ref = 0.0,
                                      int nthreads = 1);

} // namespace loki::algorithms