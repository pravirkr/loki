#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

inline constexpr double kCval = 299792458.0;

// Return the next power of two greater than or equal to n
SizeType next_power_of_two(SizeType n) noexcept;

// return max(x[i] - y[i])
float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size);

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out);

// return index of nearest value in sorted array
SizeType find_nearest_sorted_idx(std::span<const double> arr_sorted,
                                 double val);

SizeType find_nearest_sorted_idx_scan(std::span<const double> arr_sorted,
                                      double val,
                                      SizeType& hint_idx);

std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num);

std::vector<double> linspace(double start,
                             double stop,
                             SizeType num_samples = 50,
                             bool endpoint        = true) noexcept;

} // namespace loki::utils
