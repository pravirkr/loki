#pragma once

#include <span>
#include <vector>

#include "loki/loki_types.hpp"

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
SizeType find_nearest_sorted_idx(std::span<const T> arr_sorted, T val);

std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num);

} // namespace loki