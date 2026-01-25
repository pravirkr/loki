#include "loki/core/taylor.hpp"

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/kernel_utils.cuh"
#include "loki/utils.hpp"

namespace loki::core {

namespace {

template <SupportedFoldTypeCUDA FoldTypeCUDA>
__global__ void seed_kernel(const FoldTypeCUDA* __restrict__ ffa_fold,
                            cuda::std::pair<double, double> coord_init,
                            const double* __restrict__ param_arr,
                            const double* __restrict__ dparams,
                            int poly_order,
                            int nbins,
                            double* __restrict__ tree_leaves,
                            FoldTypeCUDA* __restrict__ tree_folds,
                            float* __restrict__ tree_scores,
                            int n_leaves) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_leaves) {
        return;
    }
    int n_param_sets = 1;
    for (int i = 0; i < poly_order; ++i) {
        n_param_sets *= param_arr[(i * 2)];
    }
    const auto param_set = idx * n_param_sets;
    for (int i = 0; i < n_param_sets; ++i) {
        tree_leaves[idx * n_param_sets + i] = param_arr[i][param_set + i];
    }
    for (int i = 0; i < nbins; ++i) {
        tree_folds[(idx * nbins) + i] = ffa_fold[(idx * nbins) + i];
    }
    tree_scores[idx] = 0.0F;
}

// Phase 1a: Classify leaves into branching vs non-branching
__global__ void
compute_shift_bins_kernel(const double* __restrict__ leaves_tree,
                          cuda::std::pair<double, double> coord_cur,
                          double eta,
                          SizeType nbins,
                          SizeType n_leaves,
                          const double* __restrict__ param_limits_d2,
                          const double* __restrict__ param_limits_d1,
                          uint8_t* __restrict__ branch_flags,
                          SizeType* __restrict__ branch_counts) {

    constexpr double kCval           = 299792458.0;
    constexpr double kEps            = 1e-12;
    constexpr SizeType kLeavesStride = 8;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaves)
        return;

    // Load leaf data
    const double* leaf  = &leaves_tree[tid * kLeavesStride];
    const double d2_err = leaf[1];
    const double d1_err = leaf[3];
    const double f0     = leaf[6];

    // Compute shift bins
    const double dt      = coord_cur.second;
    const double dt2     = dt * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double dphi    = eta / static_cast<double>(nbins);
    const double dfactor = kCval / f0;

    // New step sizes
    const double d2_step = dphi * dfactor * 4.0 * inv_dt2;
    const double d1_step = dphi * dfactor * 1.0 * inv_dt;

    // Shift bins
    const double shift_d2 = (d2_err - d2_step) * dt2 * nbins / (4.0 * dfactor);
    const double shift_d1 = (d1_err - d1_step) * dt * nbins / (1.0 * dfactor);

    const double eta_threshold = eta - kEps;
    const bool needs_d2_branch = shift_d2 >= eta_threshold;
    const bool needs_d1_branch = shift_d1 >= eta_threshold;
    const bool needs_branching = needs_d2_branch || needs_d1_branch;

    branch_flags[tid] = needs_branching ? 1 : 0;

    if (needs_branching) {
        // Compute branch counts
        const double d2_val = leaf[0];
        const double d1_val = leaf[2];

        int n_d2 = 1;
        if (needs_d2_branch) {
            n_d2 = detail::compute_branch_count_device(d2_val, d2_err, d2_step,
                                                       param_limits_d2[0],
                                                       param_limits_d2[1], 16);
        }

        int n_d1 = 1;
        if (needs_d1_branch) {
            const double d1_min = (1.0 - param_limits_d1[1] / f0) * kCval;
            const double d1_max = (1.0 - param_limits_d1[0] / f0) * kCval;
            n_d1 = detail::compute_branch_count_device(d1_val, d1_err, d1_step,
                                                       d1_min, d1_max, 16);
        }

        branch_counts[tid] = n_d2 * n_d1;
    } else {
        branch_counts[tid] = 1;
    }
}

// Phase 2a: Copy non-branching leaves
__global__ void copy_leaves_kernel(const double* __restrict__ leaves_tree,
                                   const uint8_t* __restrict__ branch_flags,
                                   const SizeType* __restrict__ copy_offsets,
                                   SizeType n_leaves,
                                   double* __restrict__ branched_leaves,
                                   SizeType* __restrict__ branched_indices) {

    constexpr SizeType kLeavesStride = 8;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaves)
        return;
    if (branch_flags[tid] == 1)
        return; // Skip branching leaves

    const SizeType out_idx = copy_offsets[tid];
    const double* src      = &leaves_tree[tid * kLeavesStride];
    double* dst            = &branched_leaves[out_idx * kLeavesStride];

// Unrolled copy (compiler optimizes to vector loads/stores)
#pragma unroll
    for (int i = 0; i < kLeavesStride; ++i) {
        dst[i] = src[i];
    }

    branched_indices[out_idx] = tid;
}

// Helper: Compact branching leaf IDs
__global__ void
compact_branching_ids_kernel(const uint8_t* __restrict__ branch_flags,
                             SizeType n_leaves,
                             SizeType* __restrict__ branching_leaf_ids,
                             SizeType* __restrict__ n_branching) {

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= static_cast<int>(n_leaves))
        return;
    if (branch_flags[tid] == 0)
        return;

    const SizeType idx      = atomicAdd(n_branching, 1);
    branching_leaf_ids[idx] = static_cast<SizeType>(tid);
}

// Phase 2b: Branch leaves that need branching
__global__ void
branch_leaves_kernel(const double* __restrict__ leaves_tree,
                     const SizeType* __restrict__ branching_leaf_ids,
                     const SizeType* __restrict__ branch_offsets,
                     cuda::std::pair<double, double> coord_cur,
                     double eta,
                     SizeType nbins,
                     SizeType n_branching_leaves,
                     SizeType branch_max,
                     const double* __restrict__ param_limits_d2,
                     const double* __restrict__ param_limits_d1,
                     double* __restrict__ branched_leaves,
                     SizeType* __restrict__ branched_indices,
                     SizeType n_copy_offset) {

    constexpr double kCval           = 299792458.0;
    constexpr SizeType kLeavesStride = 8;

    __shared__ double s_d2_vals[16];
    __shared__ double s_d1_vals[16];
    __shared__ double s_d2_err, s_d1_err;
    __shared__ double s_d0_val, s_d0_err, s_f0, s_basis;
    __shared__ int s_n_d2, s_n_d1;
    __shared__ SizeType s_out_base;
    __shared__ SizeType s_leaf_idx;

    const int bid        = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_size = blockDim.x;

    if (bid >= n_branching_leaves)
        return;

    // Load leaf index
    if (tid == 0) {
        s_leaf_idx = branching_leaf_ids[bid];
    }
    __syncthreads();

    const double* leaf = &leaves_tree[s_leaf_idx * kLeavesStride];

    // Thread 0: Branch both parameters
    if (tid == 0) {
        const double d2_val      = leaf[0];
        const double d2_err_orig = leaf[1];
        const double d1_val      = leaf[2];
        const double d1_err_orig = leaf[3];
        const double f0          = leaf[6];

        const double dt      = coord_cur.second;
        const double dt2     = dt * dt;
        const double inv_dt  = 1.0 / dt;
        const double inv_dt2 = inv_dt * inv_dt;
        const double dphi    = eta / static_cast<double>(nbins);
        const double dfactor = kCval / f0;

        // Compute new step sizes
        const double d2_step = dphi * dfactor * 4.0 * inv_dt2;
        const double d1_step = dphi * dfactor * 1.0 * inv_dt;

        // Branch d2
        s_n_d2 = detail::branch_param_padded_device(
            s_d2_vals, d2_val, d2_err_orig, d2_step, param_limits_d2[0],
            param_limits_d2[1], static_cast<int>(branch_max));
        s_d2_err = d2_err_orig / static_cast<double>(s_n_d2);

        // Branch d1
        const double d1_min = (1.0 - param_limits_d1[1] / f0) * kCval;
        const double d1_max = (1.0 - param_limits_d1[0] / f0) * kCval;
        s_n_d1              = detail::branch_param_padded_device(
            s_d1_vals, d1_val, d1_err_orig, d1_step, d1_min, d1_max,
            static_cast<int>(branch_max));
        s_d1_err = d1_err_orig / static_cast<double>(s_n_d1);

        // Copy non-branching parameters
        s_d0_val = leaf[4];
        s_d0_err = leaf[5];
        s_f0     = leaf[6];
        s_basis  = leaf[7];

        // Compute output base
        s_out_base = n_copy_offset + branch_offsets[s_leaf_idx];
    }
    __syncthreads();

    // All threads: Cartesian product
    const int total = s_n_d2 * s_n_d1;
    for (int idx = tid; idx < total; idx += block_size) {
        const int i = idx / s_n_d1;
        const int j = idx % s_n_d1;

        const SizeType out_offset       = (s_out_base + idx) * kLeavesStride;
        branched_leaves[out_offset + 0] = s_d2_vals[i];
        branched_leaves[out_offset + 1] = s_d2_err;
        branched_leaves[out_offset + 2] = s_d1_vals[j];
        branched_leaves[out_offset + 3] = s_d1_err;
        branched_leaves[out_offset + 4] = s_d0_val;
        branched_leaves[out_offset + 5] = s_d0_err;
        branched_leaves[out_offset + 6] = s_f0;
        branched_leaves[out_offset + 7] = s_basis;

        branched_indices[s_out_base + idx] = s_leaf_idx;
    }
}

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

__device__ __forceinline__ void
resolve_load_shared(const float* __restrict__ accel_grid,
                    SizeType n_accel,
                    const float* __restrict__ freq_grid,
                    SizeType n_freq,
                    const float*& accel_arr,
                    const float*& freq_arr) {
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
    const float* accel_arr;
    const float* freq_arr;
    if constexpr (UseShared) {
        // Load grids into shared memory
        resolve_load_shared(accel_grid, n_accel, freq_grid, n_freq, accel_arr,
                            freq_arr);
    } else {
        accel_arr = accel_grid;
        freq_arr  = freq_grid;
    }

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

    const SizeType idx_a = utils::nearest_linear_scan(
        accel_arr, n_accel, static_cast<float>(a_cur));
    const SizeType idx_f =
        utils::lower_bound_scan(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid] = idx_a * n_freq + idx_f;
    phase_shift[tid] =
        utils::get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
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
    const float* accel_arr;
    const float* freq_arr;
    if constexpr (UseShared) {
        // Load grids into shared memory
        resolve_load_shared(accel_grid, n_accel, freq_grid, n_freq, accel_arr,
                            freq_arr);
    } else {
        accel_arr = accel_grid;
        freq_arr  = freq_grid;
    }

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

    const SizeType idx_a = utils::nearest_linear_scan(
        accel_arr, n_accel, static_cast<float>(a_new));
    const SizeType idx_f =
        utils::lower_bound_scan(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid] = idx_a * n_freq + idx_f;
    phase_shift[tid] =
        utils::get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
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
    const float* accel_arr;
    const float* freq_arr;
    if constexpr (UseShared) {
        // Load grids into shared memory
        resolve_load_shared(accel_grid, n_accel, freq_grid, n_freq, accel_arr,
                            freq_arr);
    } else {
        accel_arr = accel_grid;
        freq_arr  = freq_grid;
    }

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

    const SizeType idx_a = utils::nearest_linear_scan(
        accel_arr, n_accel, static_cast<float>(a_new));
    const SizeType idx_f =
        utils::lower_bound_scan(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid] = idx_a * n_freq + idx_f;
    phase_shift[tid] =
        utils::get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
}

template <bool UseConservativeTile>
__device__ __forceinline__ void
transform_taylor_accel(double* __restrict__ leaves,
                       SizeType n_leaves,
                       cuda::std::pair<double, double> coord_next,
                       cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 8;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double dt       = coord_next.second - coord_cur.second;
    const double half_dt2 = 0.5 * dt * dt;

    double* leaf          = leaves + (tid * kLeavesStride);
    const double d2_val_i = leaf[0];
    const double d2_err_i = leaf[1];
    const double d1_val_i = leaf[2];
    const double d1_err_i = leaf[3];
    const double d0_val_i = leaf[4];
    const double d0_err_i = leaf[5];
    const double d2_val_j = d2_val_i;
    const double d1_val_j = d1_val_i + (d2_val_i * dt);
    const double d0_val_j = d0_val_i + (d1_val_i * dt) + (d2_val_i * half_dt2);

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
transform_taylor_jerk(double* __restrict__ leaves,
                      SizeType n_leaves,
                      cuda::std::pair<double, double> coord_next,
                      cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 10;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double dt        = coord_next.second - coord_cur.second;
    const double dt2       = dt * dt;
    const double half_dt2  = 0.5 * dt2;
    const double sixth_dt3 = (dt2 * dt) / 6.0;

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
    const double d1_val_j = d1_val_i + (d2_val_i * dt) + (d3_val_i * half_dt2);
    const double d0_val_j = d0_val_i + (d1_val_i * dt) + (d2_val_i * half_dt2) +
                            (d3_val_i * sixth_dt3);

    double d3_err_j, d2_err_j, d1_err_j;
    if constexpr (UseConservativeTile) {
        d3_err_j = d3_err_i;
        d2_err_j = sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt2));
        d1_err_j = sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt2) +
                        (d3_err_i * d3_err_i * half_dt2 * half_dt2));
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
transform_taylor_snap(double* __restrict__ leaves,
                      SizeType n_leaves,
                      cuda::std::pair<double, double> coord_next,
                      cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 12;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double dt                = coord_next.second - coord_cur.second;
    const double dt2               = dt * dt;
    const double half_dt2          = 0.5 * dt2;
    const double sixth_dt3         = (dt2 * dt) / 6.0;
    const double twenty_fourth_dt4 = (dt2 * dt2) / 24.0;

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
    const double d2_val_j = d2_val_i + (d3_val_i * dt) + (d4_val_i * half_dt2);
    const double d1_val_j = d1_val_i + (d2_val_i * dt) + (d3_val_i * half_dt2) +
                            (d4_val_i * sixth_dt3);
    const double d0_val_j = d0_val_i + (d1_val_i * dt) + (d2_val_i * half_dt2) +
                            (d3_val_i * sixth_dt3) +
                            (d4_val_i * twenty_fourth_dt4);

    double d4_err_j, d3_err_j, d2_err_j, d1_err_j;
    if constexpr (UseConservativeTile) {
        d4_err_j = d4_err_i;
        d3_err_j = sqrt((d3_err_i * d3_err_i) + (d4_err_i * d4_err_i * dt2));
        d2_err_j = sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt2) +
                        (d4_err_i * d4_err_i * half_dt2 * half_dt2));
        d1_err_j = sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt2) +
                        (d3_err_i * d3_err_i * half_dt2 * half_dt2) +
                        (d4_err_i * d4_err_i * sixth_dt3 * sixth_dt3));
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

// Branch and Validate Taylor Kernel
template <int NPARAMS>
__global__ void
branch_and_validate_taylor_kernel(const double* __restrict__ leaves_tree,
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
    if constexpr (NPARAMS == 2) {
        branch_and_validate_taylor_accel(
            leaves_tree, coord_cur, coord_prev, branched_leaves,
            branched_indices, n_leaves, n_leaves_after_branching,
            n_leaves_after_validation, scratch_params, scratch_dparams,
            scratch_counts);
    } else if constexpr (NPARAMS == 3) {
        branch_and_validate_taylor_jerk(
            leaves_tree, coord_cur, coord_prev, branched_leaves,
            branched_indices, n_leaves, n_leaves_after_branching,
            n_leaves_after_validation, scratch_params, scratch_dparams,
            scratch_counts);
    } else if constexpr (NPARAMS == 4) {
        branch_and_validate_taylor_snap(
            leaves_tree, coord_cur, coord_prev, branched_leaves,
            branched_indices, n_leaves, n_leaves_after_branching,
            n_leaves_after_validation, scratch_params, scratch_dparams,
            scratch_counts);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Taylor order");
    }
}

// Resolve Kernel
template <int NPARAMS, bool UseShared>
__global__ void
resolve_taylor_kernel(const double* __restrict__ leaves,
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

// Transform Kernel
template <int NPARAMS, bool UseConservativeTile>
__global__ void
transform_taylor_kernel(double* __restrict__ leaves_tree,
                        SizeType n_leaves,
                        cuda::std::pair<double, double> coord_next,
                        cuda::std::pair<double, double> coord_cur) {
    if constexpr (NPARAMS == 2) {
        transform_taylor_accel<UseConservativeTile>(leaves_tree, n_leaves,
                                                    coord_next, coord_cur);
    } else if constexpr (NPARAMS == 3) {
        transform_taylor_jerk<UseConservativeTile>(leaves_tree, n_leaves,
                                                   coord_next, coord_cur);
    } else if constexpr (NPARAMS == 4) {
        transform_taylor_snap<UseConservativeTile>(leaves_tree, n_leaves,
                                                   coord_next, coord_cur);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Taylor order");
    }
}

} // namespace

std::tuple<SizeType, SizeType> poly_taylor_branch_and_validate_cuda(
    cuda::std::span<const double> leaves_tree,
    std::pair<double, double> coord_cur,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<SizeType> leaves_origins,
    SizeType n_leaves,
    SizeType n_params,
    SizeType nbins,
    double eta,
    const std::vector<ParamLimitType>& param_limits,
    SizeType branch_max,
    cuda::std::span<double> scratch_params,
    cuda::std::span<double> scratch_dparams,
    cuda::std::span<SizeType> scratch_counts,
    cudaStream_t stream,
    CudaDeviceContext& ctx) {

    if (n_leaves == 0) {
        return std::make_tuple(0, 0);
    }
    // Better occupancy than 512 for many kernels
    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((n_leaves + kBlockSize - 1) / kBlockSize);
    ctx.check_kernel_launch_params(grid, block);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<int N>() {
        branch_and_validate_taylor_kernel<N><<<grid, block, 0, stream>>>(
            leaves_tree.data(), coord_cur, coord_prev, leaves_branch.data(),
            leaves_origins.data(), n_leaves, n_leaves_after_branching,
            n_leaves_after_validation, scratch_params.data(),
            scratch_dparams.data(), scratch_counts.data());
    };

    // Fully specialized dispatch
    switch (n_params) {
    case 2:
        dispatch.template operator()<2>();
        break;
    case 3:
        dispatch.template operator()<3>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported n_params");
    }
    cuda_utils::check_last_cuda_error(
        "Taylor branch and validate kernel launch failed");
    return std::make_tuple(n_leaves_after_branching, n_leaves_after_validation);
}

void poly_taylor_resolve_cuda(cuda::std::span<const double> leaves_branch,
                              cuda::std::span<const float> accel_grid,
                              cuda::std::span<const float> freq_grid,
                              cuda::std::span<SizeType> param_indices,
                              cuda::std::span<float> phase_shift,
                              std::pair<double, double> coord_add,
                              std::pair<double, double> coord_cur,
                              std::pair<double, double> coord_init,
                              SizeType nbins,
                              SizeType n_leaves,
                              SizeType n_params,
                              cudaStream_t stream,
                              CudaDeviceContext& ctx) {
    if (n_leaves == 0) {
        return;
    }
    const SizeType n_accel = accel_grid.size();
    const SizeType n_freq  = freq_grid.size();

    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((n_leaves + kBlockSize - 1) / kBlockSize);

    // Calculate Shared Memory Strategy
    const SizeType max_shmem = ctx.get_max_shared_memory();
    SizeType shmem_bytes     = (n_accel + n_freq) * sizeof(float);
    const bool use_smem      = (shmem_bytes <= max_shmem);
    if (!use_smem) {
        shmem_bytes = 0; // No shared memory needed
    }
    ctx.check_kernel_launch_params(grid, block, shmem_bytes);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<int N, bool S>() {
        resolve_taylor_kernel<N, S><<<grid, block, shmem_bytes, stream>>>(
            leaves_branch.data(), n_leaves, coord_add, coord_cur, coord_init,
            accel_grid.data(), n_accel, freq_grid.data(), n_freq,
            param_indices.data(), phase_shift.data(), nbins);
    };

    // Fully specialized dispatch
    if (use_smem) {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, true>();
            break;
        case 3:
            dispatch.template operator()<3, true>();
            break;
        case 4:
            dispatch.template operator()<4, true>();
            break;
        default:
            throw std::invalid_argument("Unsupported n_params");
        }
    } else {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, false>();
            break;
        case 3:
            dispatch.template operator()<3, false>();
            break;
        case 4:
            dispatch.template operator()<4, false>();
            break;
        default:
            throw std::invalid_argument("Unsupported n_params");
        }
    }
    cuda_utils::check_last_cuda_error("Taylor resolve kernel launch failed");
}

void poly_taylor_transform_cuda(cuda::std::span<double> leaves_tree,
                                std::pair<double, double> coord_next,
                                std::pair<double, double> coord_cur,
                                SizeType n_leaves,
                                SizeType n_params,
                                bool use_conservative_tile,
                                cudaStream_t stream,
                                CudaDeviceContext& ctx) {
    if (n_leaves == 0) {
        return;
    }
    // Better occupancy than 512 for many kernels
    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((n_leaves + kBlockSize - 1) / kBlockSize);
    ctx.check_kernel_launch_params(grid, block);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<int N, bool C>() {
        transform_taylor_kernel<N, C><<<grid, block, 0, stream>>>(
            leaves_tree.data(), n_leaves, coord_next, coord_cur);
    };

    // Fully specialized dispatch
    if (use_conservative_tile) {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, true>();
            break;
        case 3:
            dispatch.template operator()<3, true>();
            break;
        case 4:
            dispatch.template operator()<4, true>();
            break;
        default:
            throw std::invalid_argument("Unsupported n_params");
        }
    } else {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, false>();
            break;
        case 3:
            dispatch.template operator()<3, false>();
            break;
        case 4:
            dispatch.template operator()<4, false>();
            break;
        default:
            throw std::invalid_argument("Unsupported n_params");
        }
    }
    cuda_utils::check_last_cuda_error("Taylor transform kernel launch failed");
}

} // namespace loki::core