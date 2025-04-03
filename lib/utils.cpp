#include "loki/utils.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>

void loki::add_scalar(std::span<const float> x,
                      float scalar,
                      std::span<float> out) {
    std::ranges::transform(x, out.begin(),
                           [scalar](float xi) { return xi + scalar; });
}

float loki::diff_max(std::span<const float> x, std::span<const float> y) {
    float max_diff = -std::numeric_limits<float>::max();
    for (SizeType i = 0; i < x.size(); ++i) {
        max_diff = std::max(max_diff, x[i] - y[i]);
    }
    return max_diff;
}

void loki::circular_prefix_sum(std::span<const float> x, std::span<float> out) {
    const SizeType nbins = x.size();
    const SizeType nsum  = out.size();

    if (nbins == 0 || nsum == 0) {
        return;
    }

    // Compute the initial prefix sum
    std::inclusive_scan(x.begin(),
                        x.begin() + static_cast<int>(std::min(nbins, nsum)),
                        out.begin());
    if (nsum <= nbins) {
        return;
    }
    // Wrap around
    const SizeType n_wraps = nsum / nbins;
    const SizeType extra   = nsum % nbins;
    const float last_sum   = out[nbins - 1];
    for (SizeType i = 1; i < n_wraps; ++i) {
        add_scalar(out.subspan(0, nbins), static_cast<float>(i) * last_sum,
                   out.subspan(i * nbins, nbins));
    }
    add_scalar(out.subspan(0, extra), static_cast<float>(n_wraps) * last_sum,
               out.subspan(n_wraps * nbins, extra));
}

SizeType loki::find_nearest_sorted_idx(std::span<const FloatType> arr_sorted,
                                       FloatType val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it      = std::ranges::lower_bound(arr_sorted, val);
    SizeType idx = std::distance(arr_sorted.begin(), it);

    if (it != arr_sorted.end()) {
        if (it != arr_sorted.begin() && val - *(it - 1) < *it - val) {
            idx--;
        }
    } else {
        idx = arr_sorted.size() - 1;
    }
    return idx;
}

std::vector<SizeType> loki::find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num) {

    if (indices.empty()) {
        throw std::invalid_argument("indices cannot be empty");
    }
    if (num == 0) {
        throw std::invalid_argument("num must be greater than 0");
    }

    // Find the index of target_idx in indices
    const auto target_it = std::ranges::lower_bound(indices, target_idx);
    const auto target_idx_pos =
        static_cast<SizeType>(std::distance(indices.begin(), target_it));

    // Calculate the window around the target
    auto left = (target_idx_pos > num / 2) ? target_idx_pos - (num / 2) : 0;
    const auto right = std::min(indices.size(), left + num);
    // Adjust left if we're at the right edge
    left = (right > num) ? right - num : 0;

    // Return the slice of indices
    return {indices.begin() + static_cast<int>(left),
            indices.begin() + static_cast<int>(right)};
}
