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
              SizeType segment_len,
              SizeType nbins,
              SizeType nsamps,
              float tsamp,
              float t_ref = 0.0F);
    BruteFold(const BruteFold&)            = delete;
    BruteFold& operator=(const BruteFold&) = delete;
    BruteFold(BruteFold&&)                 = delete;
    BruteFold& operator=(BruteFold&&)      = delete;
    ~BruteFold();

    SizeType get_fold_size() const;
    void execute(std::span<const float> ts, std::span<float> fold);

private:
    std::vector<float> m_freq_arr;
    SizeType m_segment_len;
    SizeType m_nbins;
    SizeType m_nsamps;
    float m_tsamp;
    float m_t_ref;

    SizeType m_nfreqs;
    SizeType m_nsegments;
    std::vector<SizeType> m_phase_map;

    void compute_phase();
    void execute_segment(std::span<const float> ts_segment,
                         std::span<float> fold_segment);
};


/* Convenience function to fold time series using brute-force method */
void brute_fold(std::span<const float> ts,
                std::span<float> fold,
                std::vector<float> freq_arr,
                SizeType segment_len,
                SizeType nbins,
                SizeType nsamps,
                float tsamp,
                float t_ref = 0.0F);