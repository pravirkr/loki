#pragma once

#include <cstdint>
#include <cuda/std/utility>

#include <cuda_runtime.h>

#include "loki/utils.hpp"

namespace loki::utils {

__device__ __forceinline__ double get_param_val_at_idx_device(double vmin,
                                                              double vmax,
                                                              uint32_t count,
                                                              uint32_t i) {
    const double step = (vmax - vmin) / static_cast<double>(count + 1);
    return vmin + (step * static_cast<double>(i + 1));
}

__device__ __forceinline__ uint32_t get_nearest_idx_analytical_device(
    double val, double vmin, double vmax, uint32_t count) {
    const double step_inv = static_cast<double>(count + 1) / (vmax - vmin);
    const double raw_idx  = ((val - vmin) * step_inv) - 1.0;
    const auto idx        = static_cast<uint32_t>(nearbyint(raw_idx));

    if (idx < 0) {
        return 0;
    }
    if (idx >= count) {
        return count - 1;
    }
    return idx;
}

// Nearest linear scan
__device__ __forceinline__ uint32_t
nearest_linear_scan(const float* __restrict__ arr, uint32_t n, float val) {
    uint32_t best = 0;
    float best_d  = fabsf(arr[0] - val);

#pragma unroll
    for (uint32_t i = 1; i < n; ++i) {
        float d = fabsf(arr[i] - val);
        if (d < best_d) {
            best_d = d;
            best   = i;
        }
    }
    return best;
}

__device__ __forceinline__ uint32_t
lower_bound_scanf(const float* __restrict__ arr, uint32_t n, float val) {
    uint32_t l = 0, r = n;
    while (l < r) {
        uint32_t m = (l + r) >> 1U;
        if (arr[m] < val) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (l == n) {
        return n - 1;
    }
    if (l > 0) {
        const float dp = fabsf(val - arr[l - 1]);
        const float dc = fabsf(arr[l] - val);
        if (dp <= dc) {
            --l;
        }
    }
    return l;
}

__device__ __forceinline__ uint32_t
lower_bound_scan(const double* __restrict__ arr, uint32_t n, double val) {
    uint32_t l = 0, r = n;
    while (l < r) {
        uint32_t m = (l + r) >> 1U;
        if (arr[m] < val) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    if (l == n) {
        return n - 1;
    }
    if (l > 0) {
        const double dp = fabs(val - arr[l - 1]);
        const double dc = fabs(arr[l] - val);
        if (dp <= dc) {
            --l;
        }
    }
    return l;
}

// Nearest binary scan
__device__ __forceinline__ uint32_t
binary_search_device(const float* __restrict__ arr, uint32_t n, float target) {
    if (n == 0) {
        return 0;
    }
    uint32_t left   = 0;
    uint32_t right  = n - 1;
    uint32_t best   = 0;
    float best_dist = fabsf(arr[0] - target);

    while (left <= right) {
        uint32_t mid = (left + right) >> 1U;
        float dist   = fabsf(arr[mid] - target);

        if (dist < best_dist) {
            best      = mid;
            best_dist = dist;
        }

        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return best;
}

__device__ __forceinline__ uint32_t find_ffa_level(
    const uint32_t* __restrict__ offsets, uint32_t tid, uint32_t n_levels) {
    // offsets has size n_levels + 1
    // offsets[0] = 0
    // offsets[n_levels] = total_coords

    uint32_t lo = 0;
    uint32_t hi = n_levels; // invariant: tid < offsets[hi]

    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) >> 1U;
        if (tid < offsets[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ float get_phase_idx_device(double proper_time,
                                                      double freq,
                                                      uint32_t nbins,
                                                      double delay) {
    const double total_phase = (proper_time - delay) * freq;
    double ipart;
    double norm_phase = modf(total_phase, &ipart);
    if (norm_phase < 0.0) {
        norm_phase += 1.0;
    }
    double iphase = norm_phase * static_cast<double>(nbins);
    if (iphase >= static_cast<double>(nbins)) {
        iphase = 0.0;
    }
    return static_cast<float>(iphase);
}

// Device helper: Branch a single parameter
__device__ __forceinline__ cuda::std::pair<double, uint32_t>
branch_param_padded_device(double* __restrict__ out_values,
                           uint32_t out_size,
                           double param_cur,
                           double dparam_cur,
                           double dparam_new,
                           double param_min,
                           double param_max) {
    const double param_range = (param_max - param_min) * 0.5;
    if (dparam_new > (param_range + utils::kEps)) {
        out_values[0] = param_cur;
        return {dparam_new, static_cast<uint32_t>(1)};
    }

    const auto num_points = static_cast<uint32_t>(
        ceil(((dparam_cur + utils::kEps) / dparam_new) - utils::kEps));

    const double confidence_const =
        0.5 + (0.5 / static_cast<double>(num_points));
    const double half_range = confidence_const * dparam_cur;
    const double start      = param_cur - half_range;
    const double step =
        (2.0 * half_range) / static_cast<double>(num_points + 1);

    const uint32_t count = min(num_points, out_size);
#pragma unroll 4
    for (uint32_t i = 0; i < count; ++i) {
        out_values[i] = fma(static_cast<double>(i + 1), step, start);
    }

    return {dparam_cur / static_cast<double>(num_points), count};
}

__device__ __forceinline__ void
branch_one_param_padded_device(uint32_t p,
                               double cur,
                               double sig_cur,
                               double sig_new,
                               double pmin,
                               double pmax,
                               double eta,
                               double shift,
                               double* __restrict__ scratch_params,
                               double* __restrict__ scratch_dparams,
                               uint32_t* __restrict__ scratch_counts,
                               uint32_t flat_base,
                               uint32_t branch_max) {
    const uint32_t pad_offset = (flat_base + p) * branch_max;
    if (shift >= (eta - utils::kEps)) {
        auto [dparam_act, count] =
            branch_param_padded_device(scratch_params + pad_offset, branch_max,
                                       cur, sig_cur, sig_new, pmin, pmax);
        scratch_dparams[flat_base + p] = dparam_act;
        scratch_counts[flat_base + p]  = count;
    } else {
        scratch_params[pad_offset]     = cur;
        scratch_dparams[flat_base + p] = sig_cur;
        scratch_counts[flat_base + p]  = 1;
    }
};

} // namespace loki::utils