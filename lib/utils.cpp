#include "loki/utils.hpp"

#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

#include <omp.h>
#include <spdlog/spdlog.h>

namespace loki::utils {

SizeType next_power_of_two(SizeType n) noexcept {
    return 1U << static_cast<SizeType>(std::ceil(std::log2(n)));
}

float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size) {
    float max_diff = std::numeric_limits<float>::lowest();
#pragma omp simd simdlen(16) reduction(max : max_diff)
    for (SizeType i = 0; i < size; ++i) {
        max_diff = std::max(max_diff, x[i] - y[i]);
    }
    return max_diff;
}

void circular_prefix_sum(std::span<const float> x, std::span<float> out) {
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
    const float last_sum   = out[nbins - 1];
    const SizeType n_wraps = nsum / nbins;
    const SizeType extra   = nsum % nbins;
    for (SizeType i = 1; i < n_wraps; ++i) {
        const float offset_sum = static_cast<float>(i) * last_sum;
        const SizeType offset  = i * nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            out[offset + j] = out[j] + offset_sum;
        }
    }
    if (extra > 0) {
        const float final_offset = static_cast<float>(n_wraps) * last_sum;
        const SizeType offset    = n_wraps * nbins;

        for (SizeType j = 0; j < extra; ++j) {
            out[offset + j] = out[j] + final_offset;
        }
    }
}

SizeType find_nearest_sorted_idx(std::span<const double> arr_sorted,
                                 double val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it      = std::ranges::lower_bound(arr_sorted, val);
    SizeType idx = std::distance(arr_sorted.begin(), it);

    if (it != arr_sorted.end()) {
        if (it != arr_sorted.begin() && val - *(it - 1) <= *it - val) {
            idx--;
        }
    } else {
        idx = arr_sorted.size() - 1;
    }
    return idx;
}

std::vector<SizeType> find_neighbouring_indices(
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

void debug_tensor(const xt::xtensor<double, 3>& leaf_batch, SizeType n_slices) {
    xt::print_options::set_line_width(120);
    xt::print_options::set_threshold(1000);
    xt::print_options::set_edge_items(10);

    size_t num_slices = std::min<size_t>(n_slices, leaf_batch.shape()[0]);

    for (size_t i = 0; i < num_slices; ++i) {
        auto slice = xt::view(leaf_batch, i, xt::all(), xt::all());

        std::ostringstream oss;
        oss << slice;

        spdlog::info("leaf_batch[{}] =\n{}", i, oss.str());
    }
}

void debug_tensor(const xt::xtensor<double, 2>& leaf_batch, SizeType n_slices) {
    xt::print_options::set_line_width(120);
    xt::print_options::set_threshold(1000);
    xt::print_options::set_edge_items(10);

    size_t num_slices = std::min<size_t>(n_slices, leaf_batch.shape()[0]);

    for (size_t i = 0; i < num_slices; ++i) {
        auto slice = xt::view(leaf_batch, i, xt::all());

        std::ostringstream oss;
        oss << slice;

        spdlog::info("leaf_batch[{}] =\n{}", i, oss.str());
    }
}
} // namespace loki::utils