#pragma once

/**
 * @file kadane_device.cuh
 * @brief The warp-level Kadane scorer (approximate boxcar S/N), shared by the
 * standalone kernels (kadane_cuda.cu) and the fused EP kernels
 * (ep_score_*_cuda.cu), so that they share one arithmetic. CUDA only.
 *
 * A profile is scored with `nbiases` (2 or 3) biases: lane b of a group of
 * `nbiases` lanes scans the profile once, for bias b, tracking the maximum and
 * the minimum contiguous subarray of (x - mean - bias) with their widths; the
 * S/N of the best (possibly wrapped) boxcar follows, and the group's maximum
 * is the profile's S/N. A warp holds 32 / nbiases such groups: 10 profiles
 * (3 biases) or 16 (2 biases) scanned at once.
 */

#include <cstdint>

#include <cuda/std/limits>
#include <cuda_runtime.h>

namespace loki::detection::detail {

inline constexpr int kKadaneWarpSize = 32;

/// Lane layout of a warp scoring several profiles at once, one lane per bias.
/// Lanes that belong to no profile (32 % nbiases of them) get in_group false
/// and read their own lane in the shuffles.
struct KadaneLanes {
    int profile; // index of the profile within the warp
    int bias;    // index of the bias this lane scans
    int base;    // first lane of the group
    bool in_group;
};

__host__ __device__ constexpr int kadane_profiles_per_warp(int nbiases) {
    return kKadaneWarpSize / nbiases;
}

__device__ __forceinline__ KadaneLanes kadane_lanes(int lane, int nbiases) {
    const int profile = nbiases == 3 ? lane / 3 : lane / 2;
    const int base    = profile * nbiases;
    const bool in     = profile < kadane_profiles_per_warp(nbiases);
    return {profile, in ? lane - base : 0, in ? base : lane, in};
}

/// S/N of the best boxcar from the Kadane results of one bias: the maximum
/// subarray (sum @p max_sum, width @p best_w) or the wrap-around of the
/// minimum one (sum @p min_sum, width @p best_wm).
__device__ __forceinline__ float kadane_snr(float max_sum,
                                            float best_w,
                                            float min_sum,
                                            float best_wm,
                                            int nbins,
                                            float bias) {
    // Candidate 1: standard, non-wrapped.
    float best_biased_sum = max_sum;
    int best_width        = static_cast<int>(best_w);

    // Candidate 2: wrapped maximum.
    const int excluded_width = static_cast<int>(best_wm);
    const int wrapped_width  = nbins - excluded_width;

    const float wrapped_sum = (-static_cast<float>(nbins) * bias) - min_sum;
    const bool use_wrapped =
        (wrapped_width > 0) && (wrapped_sum > best_biased_sum);
    best_biased_sum = use_wrapped ? wrapped_sum : best_biased_sum;
    best_width      = use_wrapped ? wrapped_width : best_width;

    float snr = cuda::std::numeric_limits<float>::lowest();
    if (best_width > 0 && best_width < nbins) {
        const float unbiased_sum =
            best_biased_sum + (static_cast<float>(best_width) * bias);
        const float scale =
            sqrtf(static_cast<float>(nbins) /
                  static_cast<float>(best_width * (nbins - best_width)));
        snr = unbiased_sum * scale;
    }
    return snr;
}

/// S/N of one normalized profile @p row (shared or global memory) for one
/// bias, scanned from @p shift = mean + bias.
__device__ __forceinline__ float kadane_scan_snr(const float* __restrict__ row,
                                                 int nbins,
                                                 float shift,
                                                 float bias) {
    float max_sum = cuda::std::numeric_limits<float>::lowest();
    float cur_max = 0.0F;
    float w_cur   = 0.0F;
    float best_w  = 1.0F;

    float min_sum = cuda::std::numeric_limits<float>::max();
    float cur_min = 0.0F;
    float wc_min  = 0.0F;
    float best_wm = 1.0F;

    for (int j = 0; j < nbins; ++j) {
        const float val = row[j] - shift;

        // Max contiguous subarray.
        cur_max += val;
        w_cur += 1.0F;

        const bool update_max = (cur_max > max_sum);
        max_sum               = update_max ? cur_max : max_sum;
        best_w                = update_max ? w_cur : best_w;

        const bool reset_max = (cur_max < 0.0F);
        cur_max              = reset_max ? 0.0F : cur_max;
        w_cur                = reset_max ? 0.0F : w_cur;

        // Min contiguous subarray (for the wrapped candidate).
        cur_min += val;
        wc_min += 1.0F;

        const bool update_min = (cur_min < min_sum);
        min_sum               = update_min ? cur_min : min_sum;
        best_wm               = update_min ? wc_min : best_wm;

        const bool reset_min = (cur_min > 0.0F);
        cur_min              = reset_min ? 0.0F : cur_min;
        wc_min               = reset_min ? 0.0F : wc_min;
    }
    return kadane_snr(max_sum, best_w, min_sum, best_wm, nbins, bias);
}

/**
 * Max S/N over the biases of one profile per lane group. Every lane of the
 * warp calls it. @p row is the group's normalized profile (read only when
 * @p active), @p bias this lane's bias. The mean is summed in `nbiases`
 * parts, one per lane, and combined by shuffles. Returns the group's maximum
 * on every lane of the group (the lowest float when not @p active).
 */
__device__ __forceinline__ float
kadane_group_max_snr(const float* __restrict__ row,
                     bool active,
                     int nbins,
                     float bias,
                     int nbiases,
                     const KadaneLanes& lanes) {
    constexpr uint32_t kMask = 0xFFFFFFFFU;
    const int b              = lanes.bias;
    int next1                = b + 1;
    next1                    = next1 >= nbiases ? next1 - nbiases : next1;

    float partial = 0.0F;
    if (active) {
        const int chunk = (nbins + nbiases - 1) / nbiases;
        const int start = b * chunk;
        const int end   = min(nbins, start + chunk);
        for (int j = start; j < end; ++j) {
            partial += row[j];
        }
    }
    float total = partial;
    total += __shfl_sync(kMask, partial, lanes.base + next1);
    if (nbiases == 3) {
        const int next2 = b + 2 >= 3 ? b - 1 : b + 2;
        total += __shfl_sync(kMask, partial, lanes.base + next2);
    }
    const float mean = total / static_cast<float>(nbins);

    float snr = cuda::std::numeric_limits<float>::lowest();
    if (active) {
        snr = kadane_scan_snr(row, nbins, mean + bias, bias);
    }
    float best = fmaxf(snr, __shfl_sync(kMask, snr, lanes.base + next1));
    if (nbiases == 3) {
        const int next2 = b + 2 >= 3 ? b - 1 : b + 2;
        best = fmaxf(best, __shfl_sync(kMask, snr, lanes.base + next2));
    }
    return best;
}

} // namespace loki::detection::detail
