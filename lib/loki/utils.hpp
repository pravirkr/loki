#pragma once

#include <span>

namespace loki {

constexpr float kCval = 299792458.0F;

// out = x + scalar
void add_scalar(std::span<const float> x, float scalar, std::span<float> out);

// return max(x[i] - y[i])
float diff_max(std::span<const float> x, std::span<const float> y);

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out);

// return index of nearest value in sorted array
template <typename T>
size_t find_nearest_sorted_idx(std::span<const T> arr_sorted, T val);

// generate consistent grid points for range [vmin, vmax] with step dv
std::vector<float> range_param(float vmin, float vmax, float dv);

// return index of phase bin
size_t get_phase_idx(float proper_time, float freq, size_t nbins, float delay);

// fold time series and variance for given phase bins indices
void fold_ts(std::span<const float> ts,
             std::span<const size_t> ind_arrs,
             std::span<float> fold,
             size_t nbins,
             size_t nsubints);

// fold time series and variance for given frequency array
void fold_brute_start(std::span<const float> ts,
                      std::span<const float> freq_arr,
                      std::span<float> fold,
                      size_t chunk_len,
                      size_t nbins,
                      float dt,
                      float t_ref);

} // namespace loki