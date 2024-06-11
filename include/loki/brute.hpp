#pragma once

#include <span>

class BruteFold {
public:
    BruteFold(std::size_t nsamps,
              std::size_t nfreqs,
              std::size_t nbins,
              std::size_t nsubints,
              std::size_t chunk_len,
              float dt,
              float t_ref);

    void fold_ts(std::span<const float> ts,
                 std::span<const size_t> ind_arrs,
                 std::span<float> fold);

    void fold_brute_start(std::span<const float> ts,
                          std::span<const float> freq_arr,
                          std::span<float> fold);
};