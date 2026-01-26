#pragma once

#include <format>
#include <string_view>

#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::utils {

// Nearest linear scan
__device__ __forceinline__ SizeType
nearest_linear_scan(const float* __restrict__ arr, SizeType n, float val) {
    SizeType best = 0;
    float best_d  = fabsf(arr[0] - val);

#pragma unroll
    for (SizeType i = 1; i < n; ++i) {
        float d = fabsf(arr[i] - val);
        if (d < best_d) {
            best_d = d;
            best   = i;
        }
    }
    return best;
}

__device__ __forceinline__ SizeType
lower_bound_scan(const float* __restrict__ arr, SizeType n, float val) {
    SizeType l = 0, r = n;
    while (l < r) {
        SizeType m = (l + r) >> 1U;
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

// Nearest binary scan
__device__ __forceinline__ SizeType
binary_search_device(const float* __restrict__ arr, SizeType n, float target) {
    if (n == 0) {
        return 0;
    }
    SizeType left   = 0;
    SizeType right  = n - 1;
    SizeType best   = 0;
    float best_dist = fabsf(arr[0] - target);

    while (left <= right) {
        SizeType mid = (left + right) >> 1U;
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
__device__ __forceinline__ cuda::std::pair<double, SizeType>
branch_param_padded_device(double* __restrict__ out_values,
                           uint32_t out_size,
                           double param_cur,
                           double dparam_cur,
                           double dparam_new,
                           double param_min,
                           double param_max) {
    constexpr double kEps    = 1e-12;
    const double param_range = (param_max - param_min) * 0.5;
    if (dparam_new > (param_range + kEps)) {
        out_values[0] = param_cur;
        return {dparam_new, static_cast<SizeType>(1)};
    }

    const auto num_points = static_cast<uint32_t>(
        std::ceil(((dparam_cur + kEps) / dparam_new) - kEps));

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

// Helper: Compute branch count for a single parameter
__device__ inline int compute_branch_count_device(double param_cur,
                                                  double dparam_cur,
                                                  double dparam_new,
                                                  double param_min,
                                                  double param_max,
                                                  int branch_max) {
    constexpr double kEps = 1e-12;

    if (dparam_cur <= kEps || dparam_new <= kEps) {
        return 1;
    }

    const double param_range = (param_max - param_min) / 2.0;
    if (dparam_new > (param_range + kEps)) {
        return 1;
    }

    const int num_points =
        static_cast<int>(ceilf(((dparam_cur + kEps) / dparam_new) - kEps));

    if (num_points <= 0) {
        return 1;
    }

    return cuda::std::min(num_points, branch_max);
}

struct CubScanWorkspace {
    void* temp_storage = nullptr;
    size_t temp_bytes  = 0;
    SizeType capacity  = 0;

    void init(SizeType max_n) {
        capacity = max_n;

        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (SizeType*)nullptr,
                                      (SizeType*)nullptr, max_n);

        cudaMalloc(&temp_storage, temp_bytes);
    }

    void ensure(SizeType n) {
        if (n <= capacity)
            return;

        if (temp_storage)
            cudaFree(temp_storage);

        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (SizeType*)nullptr,
                                      (SizeType*)nullptr, n);

        cudaMalloc(&temp_storage, temp_bytes);
        capacity = n;
    }

    ~CubScanWorkspace() {
        if (temp_storage)
            cudaFree(temp_storage);
    }
};

} // namespace loki::utils