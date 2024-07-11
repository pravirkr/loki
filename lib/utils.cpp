#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>

#include <Eigen/Dense>

#include <loki/utils.hpp>

void loki::add_scalar(std::span<const float> x,
                      float scalar,
                      std::span<float> out) {
    std::transform(x.begin(), x.end(), out.begin(),
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

template <typename T>
SizeType loki::find_nearest_sorted_idx(std::span<const T> arr_sorted, T val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it      = std::lower_bound(arr_sorted.begin(), arr_sorted.end(), val);
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

std::vector<float> range_param(float vmin, float vmax, float dv) {
    if (vmin > vmax) {
        throw std::invalid_argument("vmin must be less than or equal to vmax");
    }
    if (dv <= 0) {
        throw std::invalid_argument("dv must be positive");
    }
    // Check if step size is larger than half the range
    if (dv > (vmax - vmin) / 2.0F) {
        return {(vmax + vmin) / 2.0F};
    }
    const auto npoints = static_cast<int>((vmax - vmin) / dv);
    Eigen::VectorXf grid_points_eigen =
        Eigen::VectorXf::LinSpaced(npoints + 2, vmin, vmax);
    std::vector<float> grid_points(grid_points_eigen.data() + 1,
                                   grid_points_eigen.data() + 1 + npoints);
    return grid_points;
}

SizeType loki::get_phase_idx(float proper_time,
                             float freq,
                             SizeType nbins,
                             float delay) {
    const auto phase      = std::fmod((proper_time + delay) * freq, 1.0F);
    const auto phase_bins = phase * static_cast<float>(nbins);
    return static_cast<SizeType>(std::round(phase_bins)) % nbins;
}
