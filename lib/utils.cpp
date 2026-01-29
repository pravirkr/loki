#include "loki/utils.hpp"

#include <algorithm>
#include <format>
#include <optional>
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

void circular_prefix_sum(const float* __restrict__ x,
                         float* __restrict__ out,
                         SizeType nbins,
                         SizeType nsum) noexcept {
    if (nbins == 0 || nsum == 0) {
        return;
    }
    // Initial prefix sum over the base cycle (as nbins < nsum)
    out[0] = x[0];
    for (SizeType i = 1; i < nbins; ++i) {
        out[i] = out[i - 1] + x[i];
    }
    if (nsum <= nbins) [[unlikely]] {
        return;
    }

    // Wrap around - optimized for the common case where wmax < nbins
    const float last_sum = out[nbins - 1];

    // First wrap (wrap_count = 1): most common case
    const SizeType first_wrap_end = std::min(2 * nbins, nsum);
    for (SizeType i = nbins; i < first_wrap_end; ++i) {
        out[i] = out[i - nbins] + last_sum;
    }

    // Additional wraps if needed (rare)
    if (nsum > 2 * nbins) [[unlikely]] {
        for (SizeType i = 2 * nbins; i < nsum; ++i) {
            const auto wrap_count   = i / nbins;
            const auto pos_in_cycle = i % nbins;
            out[i] =
                out[pos_in_cycle] + (static_cast<float>(wrap_count) * last_sum);
        }
    }
}

SizeType find_nearest_sorted_idx(std::span<const double> arr_sorted,
                                 double val,
                                 double rtol,
                                 double atol) {
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
    if (it != arr_sorted.begin()) {
        const auto diff_prev = std::abs(val - *(it - 1));
        const auto diff_curr = std::abs(*it - val);
        if (diff_prev <= (diff_curr * (1.0 + rtol)) + atol) {
            --idx;
        }
    }
    return idx;
}

SizeType find_nearest_sorted_idx_scan(std::span<const double> arr_sorted,
                                      double val,
                                      SizeType& hint_idx,
                                      double rtol,
                                      double atol) {
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
    } else if (idx > 0) {
        double diff_prev = std::abs(val - arr_sorted[idx - 1]);
        double diff_curr = std::abs(arr_sorted[idx] - val);
        if (diff_prev <= (diff_curr * (1.0 + rtol)) + atol) {
            --idx; // predecessor is closer (or tie)
        }
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
    const auto divisor = endpoint ? (num_samples - 1) : num_samples;
    const auto step    = (stop - start) / static_cast<double>(divisor);
    for (SizeType i = 0; i < num_samples; ++i) {
        result[i] = start + (step * static_cast<double>(i));
    }
    // Correct the last element to be exactly stop if endpoint is true
    if (endpoint && num_samples > 1) {
        result[num_samples - 1] = stop;
    }

    return result;
}

std::vector<SizeType>
determine_ref_segs(SizeType nsegments,
                   std::optional<SizeType> n_runs,
                   std::optional<std::vector<SizeType>> ref_segs) {
    if (n_runs.has_value()) {
        // n_runs takes precedence over ref_segs
        const auto n_runs_val = n_runs.value();
        if (n_runs_val < 1 || n_runs_val > nsegments) {
            throw std::runtime_error(
                std::format("n_runs must be between 1 and {}, got {}",
                            nsegments, n_runs_val));
        }
        std::vector<SizeType> ref_segs_val(n_runs_val);
        if (n_runs_val == 1) {
            ref_segs_val[0] = 0;
        } else {
            const SizeType max   = nsegments - 1;
            const SizeType denom = n_runs_val - 1;
            for (SizeType i = 0; i < n_runs_val; ++i) {
                ref_segs_val[i] = (i * max) / denom; // integer division
            }
        }
        return ref_segs_val;
    }
    if (ref_segs.has_value()) {
        return ref_segs.value();
    }
    throw std::runtime_error("Either n_runs or ref_segs must be provided");
}

} // namespace loki::utils