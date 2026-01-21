#pragma once

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
#include "loki/ep_kernels_utils.cuh"
#include "loki/utils.hpp"

namespace loki::ep_kernels {

namespace detail {

__device__ inline void
branch_and_validate_taylor_accel(const double* __restrict__ leaves_tree,
                                 cuda::std::pair<double, double> coord_cur,
                                 cuda::std::pair<double, double> coord_prev,
                                 double* __restrict__ branched_leaves,
                                 SizeType* __restrict__ branched_indices,
                                 SizeType n_leaves,
                                 SizeType& n_leaves_after_branching,
                                 SizeType& n_leaves_after_validation,
                                 SizeType branch_max,
                                 SizeType nbins,
                                 double eta,
                                 const double* __restrict__ param_limits_d2,
                                 const double* __restrict__ param_limits_d1,
                                 double* __restrict__ scratch_params,
                                 double* __restrict__ scratch_dparams,
                                 SizeType* __restrict__ scratch_counts) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;
    constexpr double kEps            = 1e-12;
    constexpr double kCval           = 299792458.0; // Speed of light

    const auto [_, dt]         = coord_cur;
    const double dt2           = dt * dt;
    const double inv_dt        = 1.0 / dt;
    const double inv_dt2       = inv_dt * inv_dt;
    const double nbins_d       = static_cast<double>(nbins);
    const double dphi          = eta / nbins_d;
    const double eta_threshold = eta - kEps;

    // Thread indexing
    const int tid        = threadIdx.x;
    const int bid        = blockIdx.x;
    const int block_size = blockDim.x;

    // Shared memory for per-block intermediate results
    __shared__ double s_dparam_new[2];
    __shared__ double s_shift_bins[2];
    __shared__ SizeType s_branch_counts[2];
    __shared__ SizeType s_block_offset;
    __shared__ bool s_needs_branching;

    // Each block processes one leaf
    if (bid >= n_leaves)
        return;

    const SizeType leaf_idx    = bid;
    const SizeType leaf_offset = leaf_idx * kLeavesStride;
    const SizeType flat_base   = leaf_idx * kParams;

    // --- PHASE 1: Compute step and shift (parallel across 2 params) ---
    if (tid < kParams) {
        const SizeType param_idx = tid;
        const double sig_cur =
            leaves_tree[leaf_offset + (param_idx * kParamStride) + 1];
        const double f0      = leaves_tree[leaf_offset + 6];
        const double dfactor = kCval / f0;

        // Compute step size
        double sig_new;
        double shift_factor;
        if (param_idx == 0) { // d2
            sig_new      = dphi * dfactor * 4.0 * inv_dt2;
            shift_factor = dt2 * nbins_d / (4.0 * dfactor);
        } else { // d1
            sig_new      = dphi * dfactor * 1.0 * inv_dt;
            shift_factor = dt * nbins_d / (1.0 * dfactor);
        }

        s_dparam_new[param_idx] = sig_new;
        s_shift_bins[param_idx] = (sig_cur - sig_new) * shift_factor;
    }
    __syncthreads();

    // Check if any parameter needs branching
    if (tid == 0) {
        s_needs_branching = (s_shift_bins[0] >= eta_threshold) ||
                            (s_shift_bins[1] >= eta_threshold);
    }
    __syncthreads();

    // Early exit: no branching needed for this leaf
    if (!s_needs_branching) {
        if (tid == 0) {
            // Copy leaf directly to output
            const SizeType out_offset = atomicAdd(&n_leaves_after_branching, 1);
            const SizeType out_leaf_offset = out_offset * kLeavesStride;

            // Copy entire leaf
            for (SizeType i = 0; i < kLeavesStride; ++i) {
                branched_leaves[out_leaf_offset + i] =
                    leaves_tree[leaf_offset + i];
            }
            branched_indices[out_offset] = leaf_idx;

            atomicAdd(&n_leaves_after_validation, 1);
        }
        return;
    }

    // --- PHASE 2: Branch parameters (parallel across 2 params) ---
    if (tid < kParams) {
        const SizeType param_idx    = tid;
        const SizeType param_offset = leaf_offset + (param_idx * kParamStride);
        const double param_cur      = leaves_tree[param_offset + 0];
        const double dparam_cur     = leaves_tree[param_offset + 1];
        const double dparam_new     = s_dparam_new[param_idx];
        const SizeType pad_offset   = (flat_base + param_idx) * branch_max;

        if (s_shift_bins[param_idx] >= eta_threshold) {
            // Needs branching
            double param_min, param_max;

            if (param_idx == 0) { // d2
                param_min = param_limits_d2[0];
                param_max = param_limits_d2[1];
            } else { // d1
                const double f0 = leaves_tree[leaf_offset + 6];
                param_min       = (1.0 - param_limits_d1[1] / f0) * kCval;
                param_max       = (1.0 - param_limits_d1[0] / f0) * kCval;
            }

            SizeType count = branch_param_padded_device(
                scratch_params + pad_offset, param_cur, dparam_cur, dparam_new,
                param_min, param_max, branch_max);

            s_branch_counts[param_idx] = count;

            // Compute actual dparam
            const double dparam_act = dparam_cur / static_cast<double>(count);
            scratch_dparams[flat_base + param_idx] = dparam_act;
        } else {
            // No branching
            scratch_params[pad_offset]             = param_cur;
            scratch_dparams[flat_base + param_idx] = dparam_cur;
            s_branch_counts[param_idx]             = 1;
        }
    }
    __syncthreads();

    // --- PHASE 3: Cartesian product and output ---
    const SizeType n_d2_branches  = s_branch_counts[0];
    const SizeType n_d1_branches  = s_branch_counts[1];
    const SizeType total_branches = n_d2_branches * n_d1_branches;

    // Allocate output space atomically (once per block)
    if (tid == 0) {
        s_block_offset = atomicAdd(&n_leaves_after_branching, total_branches);
    }
    __syncthreads();

    const SizeType d2_offset = (flat_base + 0) * branch_max;
    const SizeType d1_offset = (flat_base + 1) * branch_max;

    // Parallel write of Cartesian product
    // Each thread handles multiple combinations if total_branches > block_size
    for (SizeType combo_idx = tid; combo_idx < total_branches;
         combo_idx += block_size) {
        const SizeType a = combo_idx / n_d1_branches;
        const SizeType b = combo_idx % n_d1_branches;

        const SizeType out_idx       = s_block_offset + combo_idx;
        const SizeType branch_offset = out_idx * kLeavesStride;

        // Fill parameters
        branched_leaves[branch_offset + 0] = scratch_params[d2_offset + a];
        branched_leaves[branch_offset + 1] = scratch_dparams[flat_base + 0];
        branched_leaves[branch_offset + 2] = scratch_params[d1_offset + b];
        branched_leaves[branch_offset + 3] = scratch_dparams[flat_base + 1];

        // Copy d0 and f0 (4 doubles)
        branched_leaves[branch_offset + 4] = leaves_tree[leaf_offset + 4];
        branched_leaves[branch_offset + 5] = leaves_tree[leaf_offset + 5];
        branched_leaves[branch_offset + 6] = leaves_tree[leaf_offset + 6];
        branched_leaves[branch_offset + 7] = leaves_tree[leaf_offset + 7];

        branched_indices[out_idx] = leaf_idx;
    }

    // Update validation count (all branches are valid for Taylor)
    if (tid == 0) {
        atomicAdd(&n_leaves_after_validation, total_branches);
    }
}

template <bool UseShared>
__device__ __forceinline__ void
resolve_load_shared(const float* __restrict__ accel_grid,
                    SizeType n_accel,
                    const float* __restrict__ freq_grid,
                    SizeType n_freq,
                    const float*& accel_arr,
                    const float*& freq_arr) {
    if constexpr (UseShared) {
        extern __shared__ float s_mem[];
        float* s_accel = s_mem;
        float* s_freq  = s_mem + n_accel;
        // load grids
        for (uint32_t i = threadIdx.x; i < n_accel; i += blockDim.x) {
            s_accel[i] = accel_grid[i];
        }
        for (uint32_t i = threadIdx.x; i < n_freq; i += blockDim.x) {
            s_freq[i] = freq_grid[i];
        }
        __syncthreads(); // Ensure load completes
        accel_arr = s_accel;
        freq_arr  = s_freq;
    }
}

template <bool UseShared>
__device__ __forceinline__ void
resolve_taylor_accel(const double* __restrict__ leaves,
                     SizeType n_leaves,
                     cuda::std::pair<double, double> coord_add,
                     cuda::std::pair<double, double> coord_cur,
                     cuda::std::pair<double, double> coord_init,
                     const float* __restrict__ accel_grid,
                     SizeType n_accel,
                     const float* __restrict__ freq_grid,
                     SizeType n_freq,
                     SizeType* __restrict__ param_idx,
                     float* __restrict__ phase_shift,
                     SizeType nbins) {
    constexpr SizeType kLeavesStride = 8;
    const float* accel_arr           = accel_grid;
    const float* freq_arr            = freq_grid;

    // Load grids into shared memory
    resolve_load_shared<UseShared>(accel_grid, n_accel, freq_grid, n_freq,
                                   accel_arr, freq_arr);

    // Compute locally
    const double dt_add     = coord_add.first - coord_cur.first;
    const double dt_init    = coord_init.first - coord_cur.first;
    const double dt_rel     = dt_add - dt_init;
    const double half_dt_sq = 0.5 * (dt_add * dt_add - dt_init * dt_init);

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double* leaf = leaves + (tid * kLeavesStride);
    const double a_cur = leaf[0];
    const double v_cur = leaf[2];
    const double f0    = leaf[6];

    const double delta_v   = a_cur * dt_rel;
    const double delta_d   = (v_cur * dt_rel) + (a_cur * half_dt_sq);
    const double f_new     = f0 * (1.0 - delta_v * utils::kInvCval);
    const double delay_rel = delta_d * utils::kInvCval;

    const SizeType idx_a =
        nearest_linear_scan(accel_arr, n_accel, static_cast<float>(a_cur));
    const SizeType idx_f =
        lower_bound_scan(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid]   = idx_a * n_freq + idx_f;
    phase_shift[tid] = get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
}

template <bool UseShared>
__device__ __forceinline__ void
resolve_taylor_jerk(const double* __restrict__ leaves,
                    SizeType n_leaves,
                    cuda::std::pair<double, double> coord_add,
                    cuda::std::pair<double, double> coord_cur,
                    cuda::std::pair<double, double> coord_init,
                    const float* __restrict__ accel_grid,
                    SizeType n_accel,
                    const float* __restrict__ freq_grid,
                    SizeType n_freq,
                    SizeType* __restrict__ param_idx,
                    float* __restrict__ phase_shift,
                    SizeType nbins) {
    constexpr SizeType kLeavesStride = 10;
    const float* accel_arr           = accel_grid;
    const float* freq_arr            = freq_grid;

    // Load grids into shared memory
    resolve_load_shared<UseShared>(accel_grid, n_accel, freq_grid, n_freq,
                                   accel_arr, freq_arr);

    // Compute locally
    const double dt_add     = coord_add.first - coord_cur.first;
    const double dt_init    = coord_init.first - coord_cur.first;
    const double dt_rel     = dt_add - dt_init;
    const double dt_add_sq  = dt_add * dt_add;
    const double dt_init_sq = dt_init * dt_init;
    const double half_dt_sq = 0.5 * (dt_add_sq - dt_init_sq);
    const double sixth_dt_cubed =
        (dt_add_sq * dt_add - dt_init_sq * dt_init) / 6.0;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double* leaf = leaves + (tid * kLeavesStride);
    const double j_cur = leaf[0];
    const double a_cur = leaf[2];
    const double v_cur = leaf[4];
    const double f0    = leaf[8];

    const double a_new   = a_cur + (j_cur * dt_add);
    const double delta_v = (a_cur * dt_rel) + (j_cur * half_dt_sq);
    const double delta_d =
        (v_cur * dt_rel) + (a_cur * half_dt_sq) + (j_cur * sixth_dt_cubed);
    const double f_new     = f0 * (1.0 - delta_v * utils::kInvCval);
    const double delay_rel = delta_d * utils::kInvCval;

    const SizeType idx_a =
        nearest_linear_scan(accel_arr, n_accel, static_cast<float>(a_new));
    const SizeType idx_f =
        lower_bound_scan(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid]   = idx_a * n_freq + idx_f;
    phase_shift[tid] = get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
}

template <bool UseShared>
__device__ __forceinline__ void
resolve_taylor_snap(const double* __restrict__ leaves,
                    SizeType n_leaves,
                    cuda::std::pair<double, double> coord_add,
                    cuda::std::pair<double, double> coord_cur,
                    cuda::std::pair<double, double> coord_init,
                    const float* __restrict__ accel_grid,
                    SizeType n_accel,
                    const float* __restrict__ freq_grid,
                    SizeType n_freq,
                    SizeType* __restrict__ param_idx,
                    float* __restrict__ phase_shift,
                    SizeType nbins) {
    constexpr SizeType kLeavesStride = 12;
    const float* accel_arr           = accel_grid;
    const float* freq_arr            = freq_grid;

    // Load grids into shared memory
    resolve_load_shared<UseShared>(accel_grid, n_accel, freq_grid, n_freq,
                                   accel_arr, freq_arr);

    // Compute locally
    const double dt_add     = coord_add.first - coord_cur.first;
    const double dt_init    = coord_init.first - coord_cur.first;
    const double dt_rel     = dt_add - dt_init;
    const double dt_add_sq  = dt_add * dt_add;
    const double dt_init_sq = dt_init * dt_init;
    const double half_dt_sq = 0.5 * (dt_add_sq - dt_init_sq);
    const double sixth_dt_cubed =
        (dt_add_sq * dt_add - dt_init_sq * dt_init) / 6.0;
    const double twenty_fourth_dt_fourth =
        (dt_add_sq * dt_add_sq - dt_init_sq * dt_init_sq) / 24.0;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double* leaf = leaves + (tid * kLeavesStride);
    const double s_cur = leaf[0];
    const double j_cur = leaf[2];
    const double a_cur = leaf[4];
    const double v_cur = leaf[6];
    const double f0    = leaf[10];

    const double a_new = a_cur + (j_cur * dt_add) + (s_cur * 0.5 * dt_add_sq);
    const double delta_v =
        (a_cur * dt_rel) + (j_cur * half_dt_sq) + (s_cur * sixth_dt_cubed);
    const double delta_d = (v_cur * dt_rel) + (a_cur * half_dt_sq) +
                           (j_cur * sixth_dt_cubed) +
                           (s_cur * twenty_fourth_dt_fourth);
    const double f_new     = f0 * (1.0 - delta_v * utils::kInvCval);
    const double delay_rel = delta_d * utils::kInvCval;

    const SizeType idx_a =
        nearest_linear_scan(accel_arr, n_accel, static_cast<float>(a_new));
    const SizeType idx_f =
        lower_bound_scan(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid]   = idx_a * n_freq + idx_f;
    phase_shift[tid] = get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
}

template <bool UseConservativeTile>
__device__ __forceinline__ void
trnasform_taylor_accel(double* __restrict__ leaves,
                       SizeType n_leaves,
                       cuda::std::pair<double, double> coord_next,
                       cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 8;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double dt         = coord_next.second - coord_cur.second;
    const double half_dt_sq = 0.5 * dt * dt;

    double* leaf          = leaves + (tid * kLeavesStride);
    const double d2_val_i = leaf[0];
    const double d2_err_i = leaf[1];
    const double d1_val_i = leaf[2];
    const double d1_err_i = leaf[3];
    const double d0_val_i = leaf[4];
    const double d0_err_i = leaf[5];
    const double d2_val_j = d2_val_i;
    const double d1_val_j = d1_val_i + (d2_val_i * dt);
    const double d0_val_j =
        d0_val_i + (d1_val_i * dt) + (d2_val_i * half_dt_sq);

    double d2_err_j, d1_err_j;
    if constexpr (UseConservativeTile) {
        d2_err_j = d2_err_i;
        d1_err_j =
            sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt * dt));
    } else {
        d2_err_j = d2_err_i;
        d1_err_j = d1_err_i;
    }
    // Write back values
    leaf[0] = d2_val_j;
    leaf[1] = d2_err_j;
    leaf[2] = d1_val_j;
    leaf[3] = d1_err_j;
    leaf[4] = d0_val_j;
    leaf[5] = d0_err_i;
}

template <bool UseConservativeTile>
__device__ __forceinline__ void
trnasform_taylor_jerk(double* __restrict__ leaves,
                      SizeType n_leaves,
                      cuda::std::pair<double, double> coord_next,
                      cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 10;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double dt             = coord_next.second - coord_cur.second;
    const double dt_sq          = dt * dt;
    const double half_dt_sq     = 0.5 * dt_sq;
    const double sixth_dt_cubed = (dt_sq * dt) / 6.0;

    double* leaf          = leaves + (tid * kLeavesStride);
    const double d3_val_i = leaf[0];
    const double d3_err_i = leaf[1];
    const double d2_val_i = leaf[2];
    const double d2_err_i = leaf[3];
    const double d1_val_i = leaf[4];
    const double d1_err_i = leaf[5];
    const double d0_val_i = leaf[6];
    const double d0_err_i = leaf[7];
    const double d3_val_j = d3_val_i;
    const double d2_val_j = d2_val_i + (d3_val_i * dt);
    const double d1_val_j =
        d1_val_i + (d2_val_i * dt) + (d3_val_i * half_dt_sq);
    const double d0_val_j = d0_val_i + (d1_val_i * dt) +
                            (d2_val_i * half_dt_sq) +
                            (d3_val_i * sixth_dt_cubed);

    double d3_err_j, d2_err_j, d1_err_j;
    if constexpr (UseConservativeTile) {
        d3_err_j = d3_err_i;
        d2_err_j = sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt_sq));
        d1_err_j = sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt_sq) +
                        (d3_err_i * d3_err_i * half_dt_sq * half_dt_sq));
    } else {
        d3_err_j = d3_err_i;
        d2_err_j = d2_err_i;
        d1_err_j = d1_err_i;
    }
    // Write back values
    leaf[0] = d3_val_j;
    leaf[1] = d3_err_j;
    leaf[2] = d2_val_j;
    leaf[3] = d2_err_j;
    leaf[4] = d1_val_j;
    leaf[5] = d1_err_j;
    leaf[6] = d0_val_j;
    leaf[7] = d0_err_i;
}

template <bool UseConservativeTile>
__device__ __forceinline__ void
trnasform_taylor_snap(double* __restrict__ leaves,
                      SizeType n_leaves,
                      cuda::std::pair<double, double> coord_next,
                      cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 12;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double dt                      = coord_next.second - coord_cur.second;
    const double dt_sq                   = dt * dt;
    const double half_dt_sq              = 0.5 * dt_sq;
    const double sixth_dt_cubed          = (dt_sq * dt) / 6.0;
    const double twenty_fourth_dt_fourth = (dt_sq * dt_sq) / 24.0;

    double* leaf          = leaves + (tid * kLeavesStride);
    const double d4_val_i = leaf[0];
    const double d4_err_i = leaf[1];
    const double d3_val_i = leaf[2];
    const double d3_err_i = leaf[3];
    const double d2_val_i = leaf[4];
    const double d2_err_i = leaf[5];
    const double d1_val_i = leaf[6];
    const double d1_err_i = leaf[7];
    const double d0_val_i = leaf[8];
    const double d0_err_i = leaf[9];
    const double d4_val_j = d4_val_i;
    const double d3_val_j = d3_val_i + (d4_val_i * dt);
    const double d2_val_j =
        d2_val_i + (d3_val_i * dt) + (d4_val_i * half_dt_sq);
    const double d1_val_j = d1_val_i + (d2_val_i * dt) +
                            (d3_val_i * half_dt_sq) +
                            (d4_val_i * sixth_dt_cubed);
    const double d0_val_j =
        d0_val_i + (d1_val_i * dt) + (d2_val_i * half_dt_sq) +
        (d3_val_i * sixth_dt_cubed) + (d4_val_i * twenty_fourth_dt_fourth);

    double d4_err_j, d3_err_j, d2_err_j, d1_err_j;
    if constexpr (UseConservativeTile) {
        d4_err_j = d4_err_i;
        d3_err_j = sqrt((d3_err_i * d3_err_i) + (d4_err_i * d4_err_i * dt_sq));
        d2_err_j = sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt_sq) +
                        (d4_err_i * d4_err_i * half_dt_sq * half_dt_sq));
        d1_err_j =
            sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt_sq) +
                 (d3_err_i * d3_err_i * half_dt_sq * half_dt_sq) +
                 (d4_err_i * d4_err_i * sixth_dt_cubed * sixth_dt_cubed));
    } else {
        d4_err_j = d4_err_i;
        d3_err_j = d3_err_i;
        d2_err_j = d2_err_i;
        d1_err_j = d1_err_i;
    }
    // Write back values
    leaf[0] = d4_val_j;
    leaf[1] = d4_err_j;
    leaf[2] = d3_val_j;
    leaf[3] = d3_err_j;
    leaf[4] = d2_val_j;
    leaf[5] = d2_err_j;
    leaf[6] = d1_val_j;
    leaf[7] = d1_err_j;
    leaf[8] = d0_val_j;
    leaf[9] = d0_err_i;
}

// ------------------------------
// Compile-time dispatch
// ------------------------------

// Resolve kernel specialization
template <int N, PolyBasisType B> struct KernelBranchAndValidateImpl;
template <> struct KernelBranchAndValidateImpl<2, PolyBasisType::kTaylor> {
    __device__ __forceinline__ static void
    execute(const double* __restrict__ leaves_tree,
            cuda::std::pair<double, double> coord_cur,
            cuda::std::pair<double, double> coord_prev,
            const double* __restrict__ branched_leaves,
            const SizeType* __restrict__ branched_indices,
            SizeType n_leaves,
            SizeType n_leaves_after_branching,
            SizeType& n_leaves_after_validation,
            double* __restrict__ scratch_params,
            double* __restrict__ scratch_dparams,
            SizeType* __restrict__ scratch_counts) {
        branch_and_validate_taylor_accel(
            leaves_tree, coord_cur, coord_prev, branched_leaves,
            branched_indices, n_leaves, n_leaves_after_branching,
            n_leaves_after_validation, scratch_params, scratch_dparams,
            scratch_counts);
    }
};

// Resolve kernel
template <PolyBasisType Basis> struct ResolveBasisDispatcher;

template <> struct ResolveBasisDispatcher<PolyBasisType::kTaylor> {

    template <int NPARAMS, bool UseShared>
    __device__ __forceinline__ static void
    execute(const double* __restrict__ leaves,
            SizeType n_leaves,
            cuda::std::pair<double, double> coord_add,
            cuda::std::pair<double, double> coord_cur,
            cuda::std::pair<double, double> coord_init,
            const float* __restrict__ accel_grid,
            SizeType n_accel,
            const float* __restrict__ freq_grid,
            SizeType n_freq,
            SizeType* __restrict__ param_idx,
            float* __restrict__ phase_shift,
            SizeType nbins) {
        if constexpr (NPARAMS == 2) {
            resolve_taylor_accel<UseShared>(
                leaves, n_leaves, coord_add, coord_cur, coord_init, accel_grid,
                n_accel, freq_grid, n_freq, param_idx, phase_shift, nbins);
        } else if constexpr (NPARAMS == 3) {
            resolve_taylor_jerk<UseShared>(
                leaves, n_leaves, coord_add, coord_cur, coord_init, accel_grid,
                n_accel, freq_grid, n_freq, param_idx, phase_shift, nbins);
        } else if constexpr (NPARAMS == 4) {
            resolve_taylor_snap<UseShared>(
                leaves, n_leaves, coord_add, coord_cur, coord_init, accel_grid,
                n_accel, freq_grid, n_freq, param_idx, phase_shift, nbins);
        } else {
            static_assert(NPARAMS <= 4, "Unsupported Taylor order");
        }
    }
};

/*
template <> struct ResolveBasisDispatcher<PolyBasisType::kChebyshev> {

    template <int NPARAMS, bool UseShared>
    __device__ __forceinline__ static void
    execute(const double* __restrict__ leaves,
            SizeType n_leaves,
            cuda::std::pair<double, double> coord_add,
            cuda::std::pair<double, double> coord_cur,
            cuda::std::pair<double, double> coord_init,
            const float* __restrict__ accel_grid,
            SizeType n_accel,
            const float* __restrict__ freq_grid,
            SizeType n_freq,
            SizeType* __restrict__ param_idx,
            float* __restrict__ phase_shift,
            SizeType nbins) {
        static_assert(NPARAMS > 0, "Not implemented yet");
    }
};
*/

// Transform kernel
template <PolyBasisType Basis> struct TransformBasisDispatcher;

template <> struct TransformBasisDispatcher<PolyBasisType::kTaylor> {

    template <int NPARAMS, bool UseConservativeTile>
    __device__ __forceinline__ static void
    execute(const double* __restrict__ leaves,
            SizeType n_leaves,
            cuda::std::pair<double, double> coord_next,
            cuda::std::pair<double, double> coord_cur) {
        if constexpr (NPARAMS == 2) {
            transform_taylor_accel<UseConservativeTile>(leaves, n_leaves,
                                                        coord_next, coord_cur);
        } else if constexpr (NPARAMS == 3) {
            transform_taylor_jerk<UseConservativeTile>(leaves, n_leaves,
                                                       coord_next, coord_cur);
        } else if constexpr (NPARAMS == 4) {
            transform_taylor_snap<UseConservativeTile>(leaves, n_leaves,
                                                       coord_next, coord_cur);
        } else {
            static_assert(NPARAMS <= 4, "Unsupported Taylor order");
        }
    }
};

// Branch and Validate Kernel dispatch
template <int NPARAMS, PolyBasisType BASIS>
__device__ __forceinline__ void
branch_and_validate_kernel_impl(const double* __restrict__ leaves_tree,
                                cuda::std::pair<double, double> coord_cur,
                                cuda::std::pair<double, double> coord_prev,
                                const double* __restrict__ branched_leaves,
                                const SizeType* __restrict__ branched_indices,
                                SizeType n_leaves,
                                SizeType n_leaves_after_branching,
                                SizeType& n_leaves_after_validation,
                                double* __restrict__ scratch_params,
                                double* __restrict__ scratch_dparams,
                                SizeType* __restrict__ scratch_counts) {
    KernelBranchAndValidateImpl<NPARAMS, BASIS>::execute(
        leaves_tree, coord_cur, coord_prev, branched_leaves, branched_indices,
        n_leaves, n_leaves_after_branching, n_leaves_after_validation,
        scratch_params, scratch_dparams, scratch_counts);
}

// Resolve kernel dispatch
template <int NPARAMS, PolyBasisType Basis, bool UseShared>
__device__ __forceinline__ void
resolve_kernel_impl(const double* __restrict__ leaves,
                    SizeType n_leaves,
                    cuda::std::pair<double, double> coord_add,
                    cuda::std::pair<double, double> coord_cur,
                    cuda::std::pair<double, double> coord_init,
                    const float* __restrict__ accel_grid,
                    SizeType n_accel,
                    const float* __restrict__ freq_grid,
                    SizeType n_freq,
                    SizeType* __restrict__ param_idx,
                    float* __restrict__ phase_shift,
                    SizeType nbins) {
    ResolveBasisDispatcher<Basis>::template execute<NPARAMS, UseShared>(
        leaves, n_leaves, coord_add, coord_cur, coord_init, accel_grid, n_accel,
        freq_grid, n_freq, param_idx, phase_shift, nbins);
}

// Transform kernel dispatch
template <int NPARAMS, PolyBasisType Basis, bool UseConservativeTile>
__device__ __forceinline__ void
transform_kernel_impl(const double* __restrict__ leaves,
                      SizeType n_leaves,
                      cuda::std::pair<double, double> coord_next,
                      cuda::std::pair<double, double> coord_cur) {
    TransformBasisDispatcher<Basis>::template execute<NPARAMS,
                                                      UseConservativeTile>(
        leaves, n_leaves, coord_next, coord_cur);
}

} // namespace detail

// Branch and Validate Kernel (Generic)
template <int NPARAMS, PolyBasisType BASIS>
__global__ __launch_bounds__(128) void branch_and_validate_kernel(
    const double* __restrict__ leaves_tree,
    cuda::std::pair<double, double> coord_cur,
    cuda::std::pair<double, double> coord_prev,
    const double* __restrict__ branched_leaves,
    const SizeType* __restrict__ branched_indices,
    SizeType n_leaves,
    SizeType n_leaves_after_branching,
    SizeType& n_leaves_after_validation,
    double* __restrict__ scratch_params,
    double* __restrict__ scratch_dparams,
    SizeType* __restrict__ scratch_counts) {
    detail::branch_and_validate_kernel_impl<NPARAMS>(
        leaves_tree, coord_cur, coord_prev, branched_leaves, branched_indices,
        n_leaves, n_leaves_after_branching, n_leaves_after_validation,
        scratch_params, scratch_dparams, scratch_counts);
}

// Resolve Kernel (Generic)
template <int NPARAMS, PolyBasisType Basis, bool UseShared>
__global__ void resolve_kernel(const double* __restrict__ leaves,
                               SizeType n_leaves,
                               cuda::std::pair<double, double> coord_add,
                               cuda::std::pair<double, double> coord_cur,
                               cuda::std::pair<double, double> coord_init,
                               const float* __restrict__ accel_grid,
                               SizeType n_accel,
                               const float* __restrict__ freq_grid,
                               SizeType n_freq,
                               SizeType* __restrict__ param_idx,
                               float* __restrict__ phase_shift,
                               SizeType nbins) {
    detail::resolve_kernel_impl<NPARAMS, Basis, UseShared>(
        leaves, n_leaves, coord_add, coord_cur, coord_init, accel_grid, n_accel,
        freq_grid, n_freq, param_idx, phase_shift, nbins);
}

// Transform Kernel (Generic)
template <int NPARAMS, PolyBasisType Basis, bool UseConservativeTile>
__global__ void transform_kernel(const double* __restrict__ leaves,
                                 SizeType n_leaves,
                                 cuda::std::pair<double, double> coord_next,
                                 cuda::std::pair<double, double> coord_cur) {
    detail::transform_kernel_impl<NPARAMS, Basis, UseConservativeTile>(
        leaves, n_leaves, coord_next, coord_cur);
}

} // namespace loki::ep_kernels
