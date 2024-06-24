#include <algorithm>
#include <arm_neon.h>
#include <cstddef>
#include <numeric>
#include <span>

#include <Eigen/Dense>

#include <loki/utils.hpp>
#include <stdexcept>

void loki::add_scalar(std::span<const float> x,
                      float scalar,
                      std::span<float> out) {
    std::transform(x.begin(), x.end(), out.begin(),
                   [scalar](float xi) { return xi + scalar; });
}

float loki::diff_max(std::span<const float> x, std::span<const float> y) {
    float max_diff = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < x.size(); ++i) {
        max_diff = std::max(max_diff, x[i] - y[i]);
    }
    return max_diff;
}

void loki::circular_prefix_sum(std::span<const float> x, std::span<float> out) {
    const size_t nbins = x.size();
    const size_t nsum  = out.size();

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
    const size_t n_wraps = nsum / nbins;
    const size_t extra   = nsum % nbins;
    const float last_sum = out[nbins - 1];
    for (size_t i = 1; i < n_wraps; ++i) {
        add_scalar(out.subspan(0, nbins), static_cast<float>(i) * last_sum,
                   out.subspan(i * nbins, nbins));
    }
    add_scalar(out.subspan(0, extra), static_cast<float>(n_wraps) * last_sum,
               out.subspan(n_wraps * nbins, extra));
}

template <typename T>
size_t loki::find_nearest_sorted_idx(std::span<const T> arr_sorted, T val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it    = std::lower_bound(arr_sorted.begin(), arr_sorted.end(), val);
    size_t idx = std::distance(arr_sorted.begin(), it);

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

size_t
loki::get_phase_idx(float proper_time, float freq, size_t nbins, float delay) {
    const auto phase      = std::fmod((proper_time + delay) * freq, 1.0);
    const auto phase_bins = phase * static_cast<float>(nbins);
    return static_cast<size_t>(std::round(phase_bins)) % nbins;
}

void loki::fold_ts(std::span<const float> ts,
                   std::span<const size_t> ind_arrs,
                   std::span<float> fold,
                   size_t nbins,
                   size_t nsubints) {
    const auto nsamps          = ts.size();
    const auto nfreqs          = ind_arrs.size() / nbins;
    const auto samp_per_subint = nsamps / nsubints;
    const auto samps_final     = nsubints * samp_per_subint;

    if (fold.size() != nfreqs * nsubints * nbins) {
        throw std::runtime_error("Output array has wrong size");
    }
    std::vector<size_t> subint_idxs(samps_final);
    for (size_t i = 0; i < samps_final; ++i) {
        subint_idxs[i] = i / samp_per_subint;
    }
    for (size_t ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const auto& ind_arr = ind_arrs.subspan(ifreq * nbins, nbins);
        for (size_t isamp = 0; isamp < samps_final; ++isamp) {
            const auto isubint = subint_idxs[isamp];
            const auto iphase  = ind_arr[isamp];
            const auto out_idx =
                ifreq * nsubints * nbins + isubint * nbins + iphase;
            fold[out_idx] += ts[isamp];
        }
    }
}

void loki::fold_brute_start(std::span<const float> ts,
                            std::span<const float> freq_arr,
                            std::span<float> fold,
                            size_t chunk_len,
                            size_t nbins,
                            float dt,
                            float t_ref) {
    const auto nsamps          = ts.size();
    const auto nfreqs          = freq_arr.size();
    const auto fold_seg_stride = nfreqs * 2 * nbins;
    const auto nchunks         = fold.size() / fold_seg_stride;
    std::vector<size_t> segment_idxs;
    for (size_t i = 0; i < nsamps; i += chunk_len) {
        segment_idxs.push_back(i);
    }

    if (segment_idxs.size() != nchunks) {
        throw std::runtime_error("Number of segments does not match");
    }

    std::vector<float> proper_time(chunk_len);
    for (size_t i = 0; i < chunk_len; ++i) {
        proper_time[i] = static_cast<float>(i) * dt;
    }
    std::vector<size_t> phase_idx_arrs(nfreqs * chunk_len);
    for (size_t ifreq = 0; ifreq < freq_arr.size(); ++ifreq) {
        const auto freq = freq_arr[ifreq];
        for (size_t i = 0; i < chunk_len; ++i) {
            phase_idx_arrs[ifreq * chunk_len + i] =
                get_phase_idx(proper_time[i] - t_ref, freq, nbins, 0);
        }
    }
    for (size_t iseg = 0; iseg < segment_idxs.size(); ++iseg) {
        const auto segment_idx = segment_idxs[iseg];
        const auto segment_len = std::min(chunk_len, nsamps - segment_idx);
        const auto ts_segment  = ts.subspan(segment_idx, segment_len);
        auto fold_span = fold.subspan(iseg * fold_seg_stride, fold_seg_stride);
        fold_ts(ts_segment, phase_idx_arrs, fold_span, nbins, 1);
    }
}