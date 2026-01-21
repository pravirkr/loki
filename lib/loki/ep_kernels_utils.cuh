#pragma once

#include <format>
#include <string_view>

#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

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

namespace loki::ep_kernels {

enum class PolyBasisType : uint8_t { kTaylor = 0, kChebyshev = 1 };

PolyBasisType parse_basis(std::string_view s) {
    if (s == "taylor") {
        return PolyBasisType::kTaylor;
    }
    if (s == "chebyshev") {
        return PolyBasisType::kChebyshev;
    }
    throw std::invalid_argument(std::format("Unknown basis: {}", s));
}

namespace detail {

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
                                                      SizeType nbins,
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
__device__ __forceinline__ SizeType
branch_param_padded_device(double* __restrict__ out_values,
                           double param_cur,
                           double dparam_cur,
                           double dparam_new,
                           double param_min,
                           double param_max,
                           int branch_max) {
    constexpr double kEps = 1e-12;

    if (dparam_cur <= kEps || dparam_new <= kEps) {
        return 0; // Error condition
    }

    if (param_max <= param_min + kEps) {
        return 0; // Error condition
    }

    const double param_range = (param_max - param_min) / 2.0;
    if (dparam_new > (param_range + kEps)) {
        // Step size too large, fallback to current value
        out_values[0] = param_cur;
        return 1;
    }

    const int num_points =
        static_cast<int>(ceil(((dparam_cur + kEps) / dparam_new) - kEps));

    if (num_points <= 0) {
        return 0; // Error condition
    }

    const int n = num_points + 2;
    const double confidence_const =
        0.5 * (1.0 + 1.0 / static_cast<double>(num_points));
    const double half_range = confidence_const * dparam_cur;
    const double start      = param_cur - half_range;
    const double stop       = param_cur + half_range;
    const int num_intervals = n - 1;
    const double step = (stop - start) / static_cast<double>(num_intervals);

    const int count = cuda::std::min(num_points, branch_max);
    for (int i = 0; i < count; ++i) {
        out_values[i] = start + (static_cast<double>(i + 1) * step);
    }

    return count;
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

} // namespace detail

__global__ void shift_add_kernel(const float* __restrict__ tree_folds,
                                 const SizeType* __restrict__ tree_indices,
                                 const float* __restrict__ ffa_fold_segment,
                                 const SizeType* __restrict__ param_idx,
                                 const float* __restrict__ phase_shift,
                                 float* __restrict__ out_folds,
                                 uint32_t n_leaves,
                                 uint32_t nbins) {

    constexpr uint32_t kWarpSize = 32;

    // 1D thread mapping with optimal work distribution
    const uint32_t tid    = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto total_work = n_leaves * nbins;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, ibin)
    // Fastest varying (best for coalescing)
    const auto ibin  = tid % nbins;
    const auto ileaf = tid / nbins;

    uint32_t shift = 0;
    // We can only use warp-level broadcast if each warp maps to exactly one
    // leaf. This is true iff nbins is a multiple of warpSize.
    if (nbins >= kWarpSize && (nbins & (kWarpSize - 1)) == 0) {
        // warp maps to a single leaf → safe to broadcast
        const auto lane = threadIdx.x & (kWarpSize - 1);
        if (lane == 0) {
            shift = __float2int_rn(phase_shift[ileaf]) % nbins;
        }
        const auto mask = __activemask();
        shift           = __shfl_sync(mask, shift, 0);
    } else {
        // warp spans multiple leaves, must compute per-thread
        shift = __float2int_rn(phase_shift[ileaf]) % nbins;
    }
    const auto idx_add =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    // Calculate offsets
    const auto total_size  = 2 * nbins;
    const auto tree_offset = tree_indices[ileaf] * total_size;
    const auto ffa_offset  = param_idx[ileaf] * total_size;
    const auto out_offset  = ileaf * total_size;

    // Process both e and v components
    out_folds[out_offset + ibin] =
        tree_folds[tree_offset + ibin] + ffa_fold_segment[ffa_offset + idx_add];
    out_folds[out_offset + ibin + nbins] =
        tree_folds[tree_offset + ibin + nbins] +
        ffa_fold_segment[ffa_offset + idx_add + nbins];
}

__global__ void
shift_add_complex_kernel(const ComplexTypeCUDA* __restrict__ tree_folds,
                         const SizeType* __restrict__ tree_indices,
                         const ComplexTypeCUDA* __restrict__ ffa_fold_segment,
                         const SizeType* __restrict__ param_idx,
                         const float* __restrict__ phase_shift,
                         ComplexTypeCUDA* __restrict__ out_folds,
                         uint32_t n_leaves,
                         uint32_t nbins,
                         uint32_t nbins_f) {

    // 1D thread mapping with optimal work distribution
    const uint32_t tid    = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto total_work = n_leaves * nbins_f;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, k)
    const auto k     = tid % nbins_f;
    const auto ileaf = tid / nbins_f;

    // Precompute phase factors: exp(-2πi * k * shift / nbins)
    const auto phase_factor =
        static_cast<float>(-2.0F * M_PI * k * phase_shift[ileaf] / nbins);
    // Fast sincos computation
    float cos_phase, sin_phase;
    __sincosf(phase_factor, &sin_phase, &cos_phase);

    // Calculate offsets
    const auto total_size  = 2 * nbins_f;
    const auto tree_offset = tree_indices[ileaf] * total_size;
    const auto ffa_offset  = param_idx[ileaf] * total_size;
    const auto out_offset  = ileaf * total_size;

    // Load complex values for both e and v components
    const ComplexTypeCUDA ffa_e  = ffa_fold_segment[ffa_offset + k];
    const ComplexTypeCUDA ffa_v  = ffa_fold_segment[ffa_offset + k + nbins_f];
    const ComplexTypeCUDA tree_e = tree_folds[tree_offset + k];
    const ComplexTypeCUDA tree_v = tree_folds[tree_offset + k + nbins_f];

    // OPTIMIZED complex multiplication using fmaf
    // ffa_shifted_e = ffa_e * exp(-2πi * k * shift / nbins)
    const float real_ffa_e =
        fmaf(ffa_e.real(), cos_phase, -ffa_e.imag() * sin_phase);
    const float imag_ffa_e =
        fmaf(ffa_e.real(), sin_phase, ffa_e.imag() * cos_phase);
    const float real_ffa_v =
        fmaf(ffa_v.real(), cos_phase, -ffa_v.imag() * sin_phase);
    const float imag_ffa_v =
        fmaf(ffa_v.real(), sin_phase, ffa_v.imag() * cos_phase);

    // Process both e and v components
    out_folds[out_offset + k] =
        ComplexTypeCUDA(tree_e.real() + real_ffa_e, tree_e.imag() + imag_ffa_e);
    out_folds[out_offset + k + nbins_f] =
        ComplexTypeCUDA(tree_v.real() + real_ffa_v, tree_v.imag() + imag_ffa_v);
}

template <uint32_t BlockThreads>
__global__ void score_and_filter_kernel(const float* __restrict__ tree_folds,
                                        float* __restrict__ tree_scores,
                                        SizeType* __restrict__ tree_indices,
                                        const SizeType* __restrict__ widths,
                                        uint32_t nwidths,
                                        float threshold,
                                        uint32_t n_leaves,
                                        uint32_t nbins,
                                        uint32_t* n_leaves_passing) {
    // Kernel Configuration & Indexing
    constexpr uint32_t kWarpSize       = 32;
    constexpr uint32_t kLeavesPerBlock = BlockThreads / kWarpSize;

    const auto warp_id = threadIdx.x / kWarpSize;
    const auto lane_id = threadIdx.x % kWarpSize;
    const auto ileaf   = (blockIdx.x * kLeavesPerBlock) + warp_id;

    if (ileaf >= n_leaves) {
        return;
    }
    // Dynamic Shared Memory
    extern __shared__ float s_mem_block[];
    float* s_warp_data = &s_mem_block[static_cast<SizeType>(warp_id * nbins)];

    using WarpScan   = cub::WarpScan<float, kWarpSize>;
    using WarpReduce = cub::WarpReduce<float, kWarpSize>;
    __shared__ typename WarpScan::TempStorage temp_scan[kLeavesPerBlock];
    __shared__ typename WarpReduce::TempStorage temp_reduce[kLeavesPerBlock];
    __shared__ typename WarpReduce::TempStorage temp_final[kLeavesPerBlock];

    // Load data from global to shared memory
    for (uint32_t j = lane_id; j < nbins; j += kWarpSize) {
        const auto idx_e = (ileaf * 2 * nbins) + j;
        const auto idx_v = (ileaf * 2 * nbins) + j + nbins;
        s_warp_data[j]   = tree_folds[idx_e] / sqrtf(tree_folds[idx_v]);
    }

    // Perform warp-level complicated inclusive prefix sum
    float running_sum     = 0.0F;
    const auto num_chunks = (nbins + kWarpSize - 1) / kWarpSize;
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const auto idx = (chunk * kWarpSize) + lane_id;
        float val      = (idx < nbins) ? s_warp_data[idx] : 0.0F;
        WarpScan(temp_scan[warp_id]).InclusiveSum(val, val);
        if (idx < nbins) {
            s_warp_data[idx] = val + running_sum;
        }
        float chunk_sum = __shfl_sync(0xFFFFFFFF, val, kWarpSize - 1);
        if (lane_id == 0) {
            running_sum += chunk_sum;
        }
        running_sum = __shfl_sync(0xFFFFFFFF, running_sum, 0);
    }

    // Find max SNR across all widths
    const float total_sum = s_warp_data[nbins - 1];
    float thread_max_snr  = -CUDART_INF_F;

    for (uint32_t iw = 0; iw < nwidths; ++iw) {
        const int w   = static_cast<int>(widths[iw]);
        const float h = sqrtf(static_cast<float>(nbins - w) /
                              static_cast<float>(nbins * w));
        const float b =
            static_cast<float>(w) * h / static_cast<float>(nbins - w);

        float thread_max_diff = -CUDART_INF_F;
        for (uint32_t j = lane_id; j < nbins; j += kWarpSize) {
            const auto sum_before_start = (j > 0) ? s_warp_data[j - 1] : 0.0F;
            float current_sum;
            const auto end_idx = j + w - 1;
            if (end_idx < nbins) {
                // Normal case: sum from j to j+w-1
                current_sum = s_warp_data[end_idx] - sum_before_start;
            } else {
                // Circular case: sum wraps around
                current_sum = (total_sum - sum_before_start) +
                              s_warp_data[end_idx % nbins];
            }
            thread_max_diff = fmaxf(thread_max_diff, current_sum);
        }
        const float max_diff = WarpReduce(temp_reduce[warp_id])
                                   .Reduce(thread_max_diff, CubMaxOp<float>());

        if (lane_id == 0) {
            const float snr = ((h + b) * max_diff) - (b * total_sum);
            thread_max_snr  = fmaxf(thread_max_snr, snr);
        }
    }

    // Final reduction to get max SNR across all widths for this warp
    float final_max_snr = WarpReduce(temp_final[warp_id])
                              .Reduce(thread_max_snr, CubMaxOp<float>());
    if (lane_id == 0) {
        tree_scores[ileaf] = final_max_snr;
        if (final_max_snr > threshold) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> counter(
                *n_leaves_passing);
            const uint32_t idx =
                counter.fetch_add(1, cuda::std::memory_order_relaxed);
            tree_indices[idx] = ileaf;
        }
    }
}

} // namespace loki::ep_kernels