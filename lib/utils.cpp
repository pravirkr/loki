#include "loki/utils.hpp"

#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>

#include <omp.h>

#include "loki/common/types.hpp"

namespace loki::utils {

float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size) noexcept {
    float max_diff = std::numeric_limits<float>::lowest();
#pragma omp simd simdlen(16) reduction(max : max_diff)
    for (SizeType i = 0; i < size; ++i) {
        max_diff = std::max(max_diff, x[i] - y[i]);
    }
    return max_diff;
}

void circular_prefix_sum(std::span<const float> x, std::span<float> out) {
    const auto nbins = x.size();
    const auto nsum  = out.size();
    if (nbins == 0 || nsum == 0) {
        return;
    }
    // Compute the initial prefix sum
    const auto initial_count = std::min(nbins, nsum);
    std::inclusive_scan(x.begin(),
                        x.begin() + static_cast<IndexType>(initial_count),
                        out.begin());
    if (nsum <= nbins) {
        return;
    }
    // Wrap around
    const float last_sum = out[nbins - 1];
    for (SizeType i = nbins; i < nsum; ++i) {
        const auto wrap_count   = i / nbins;
        const auto pos_in_cycle = i % nbins;
        out[i] = out[pos_in_cycle] + static_cast<float>(wrap_count) * last_sum;
    }
}

SizeType find_nearest_sorted_idx(std::span<const double> arr_sorted,
                                 double val) {
    if (arr_sorted.empty()) {
        throw std::invalid_argument("find_nearest_sorted_idx: array is empty");
    }
    if (std::isnan(val)) {
        throw std::invalid_argument("find_nearest_sorted_idx: val is NaN");
    }
    const auto it = std::ranges::lower_bound(arr_sorted, val);
    auto idx = static_cast<SizeType>(std::distance(arr_sorted.begin(), it));

    // Handle case where val is larger than all elements
    if (it == arr_sorted.end()) {
        return arr_sorted.size() - 1;
    }
    // Check if previous element is closer
    if (it != arr_sorted.begin() && val - *(it - 1) <= *it - val) {
        --idx;
    }
    return idx;
}

SizeType find_nearest_sorted_idx_scan(std::span<const double> arr_sorted,
                                      double val,
                                      SizeType& hint_idx) {
    const auto n = arr_sorted.size();
    if (n == 0) {
        throw std::invalid_argument(
            "find_nearest_sorted_idx_scan: array is empty");
    }
    if (std::isnan(val)) {
        throw std::invalid_argument("find_nearest_sorted_idx_scan: val is NaN");
    }

    // Clamp the hint to a valid range.
    hint_idx = std::min(hint_idx, n);

    // Scan forward from the last known position (the hint).
    while (hint_idx < n && arr_sorted[hint_idx] < val) {
        ++hint_idx;
    }
    // Scan backward in case the `val` sequence isn't perfectly monotonic.
    while (hint_idx > 0 && arr_sorted[hint_idx - 1] >= val) {
        --hint_idx;
    }
    // `hint_idx` is now our lower bound index.
    SizeType idx = hint_idx;
    if (idx == n) {
        idx = n - 1; // past the end
    } else if (idx > 0 &&
               (val - arr_sorted[idx - 1]) <= (arr_sorted[idx] - val)) {
        --idx; // predecessor is closer (or tie)
    }

    hint_idx = idx; // keep hint consistent for next call
    return idx;
}

std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num) {

    if (indices.empty()) {
        throw std::invalid_argument("find_neighbouring_indices: indices empty");
    }
    if (num == 0) {
        throw std::invalid_argument("find_neighbouring_indices: num == 0");
    }

    // Find the index of target_idx in indices
    const auto it  = std::ranges::lower_bound(indices, target_idx);
    const auto pos = static_cast<SizeType>(std::distance(indices.begin(), it));

    // Calculate the window around the target
    const auto half  = num / 2;
    auto left        = (pos > half) ? pos - half : 0;
    const auto right = std::min(indices.size(), left + num);
    if (right - left < num && right == indices.size()) {
        left = right > num ? right - num : 0; // adjust when at right edge
    }

    std::vector<SizeType> result;
    result.reserve(right - left);
    result.assign(indices.begin() + static_cast<IndexType>(left),
                  indices.begin() + static_cast<IndexType>(right));
    return result;
}

std::vector<double> linspace(double start,
                             double stop,
                             SizeType num_samples,
                             bool endpoint) noexcept {
    std::vector<double> result(num_samples);

    if (num_samples == 0) {
        return result;
    }
    if (num_samples == 1) {
        result[0] = start;
        return result;
    }
    const auto step =
        (stop - start) /
        std::fmax(1.0, static_cast<double>(num_samples - (endpoint ? 1 : 0)));

    for (SizeType i = 0; i < num_samples; ++i) {
        result[i] = start + step * static_cast<double>(i);
    }
    // Correct the last element to be exactly stop if endpoint is true
    if (endpoint && num_samples > 1) {
        result[num_samples - 1] = stop;
    }

    return result;
}
} // namespace loki::utils