#pragma once

#include <cstddef>
#include <span>

#include <loki/loki_types.hpp>

/**
 * @brief Fold time series using brute-force method
 * 
 */
class BruteFold {
public:
    BruteFold(std::vector<float> freq_arr,
              SizeType nbin,
              SizeType segment_len,
              float tsamp,
              float t_ref);
    BruteFold(const BruteFold&)            = delete;
    BruteFold& operator=(const BruteFold&) = delete;
    BruteFold(BruteFold&&)                 = delete;
    BruteFold& operator=(BruteFold&&)      = delete;
    ~BruteFold();

    void execute(std::span<const float> ts, std::span<float> fold);

private:
    std::vector<float> m_freq_arr;
    SizeType m_nbin;
    SizeType m_segment_len;
    float m_tsamp;
    float m_t_ref;

    SizeType m_nfreqs;
    std::vector<SizeType> m_phase_map;

    void compute_phase();
    void execute_segment(std::span<const float> ts_segment,
                         std::span<float> fold_segment);
};