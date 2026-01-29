#include "loki/core/taylor.hpp"

#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/kernel_utils.cuh"
#include "loki/utils.hpp"

namespace loki::core {

namespace {

__global__ void
kernel_poly_taylor_seed(const SizeType* __restrict__ param_grid_count_init,
                        const double* __restrict__ dparams_init,
                        const ParamLimit* __restrict__ param_limits,
                        double* __restrict__ seed_leaves,
                        uint32_t n_leaves,
                        uint32_t n_params) {
    constexpr uint32_t kParamStride = 2;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const auto n_accel_init  = param_grid_count_init[n_params - 2];
    const auto n_freq_init   = param_grid_count_init[n_params - 1];
    const auto d_freq_cur    = dparams_init[n_params - 1];
    const uint32_t accel_idx = ileaf / n_freq_init;
    const uint32_t freq_idx  = ileaf % n_freq_init;

    const double a_cur = utils::get_param_val_at_idx_device(
        param_limits[n_params - 2].min, param_limits[n_params - 2].max,
        n_accel_init, accel_idx);
    const double f_cur = utils::get_param_val_at_idx_device(
        param_limits[n_params - 1].min, param_limits[n_params - 1].max,
        n_freq_init, freq_idx);

    const uint32_t lo = (n_params + 2) * kParamStride * ileaf;
    // Copy till d2 (acceleration)
    for (uint32_t j = 0; j < n_params - 1; ++j) {
        seed_leaves[lo + (j * kParamStride) + 0] = 0.0;
        seed_leaves[lo + (j * kParamStride) + 1] = dparams_init[j];
    }
    seed_leaves[lo + ((n_params - 2) * kParamStride) + 0] = a_cur;
    // Update frequency to velocity
    // f = f0(1 - v / C) => dv = -(C/f0) * df
    seed_leaves[lo + ((n_params - 1) * kParamStride) + 0] = 0;
    seed_leaves[lo + ((n_params - 1) * kParamStride) + 1] =
        d_freq_cur * (utils::kCval / f_cur);
    // intialize d0 (measure from t=t_init) and store f0
    seed_leaves[lo + ((n_params + 0) * kParamStride) + 0] = 0;
    seed_leaves[lo + ((n_params + 0) * kParamStride) + 1] = 0;
    seed_leaves[lo + ((n_params + 1) * kParamStride) + 0] = f_cur;
    // Store basis flag (0: Polynomial, 1: Physical)
    seed_leaves[lo + ((n_params + 1) * kParamStride) + 1] = 0;
}

__global__ void kernel_analyze_and_branch_accel(
    const double* __restrict__ leaves_tree,
    uint32_t n_leaves,
    cuda::std::pair<double, double> coord_cur,
    uint32_t nbins,
    double eta,
    const ParamLimit* __restrict__ param_limits,
    uint32_t branch_max,
    utils::BranchingWorkspaceCUDAView ws,
    int* __restrict__ global_branch_flag) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t leaf_offset = ileaf * kLeavesStride;
    const uint32_t flat_base   = ileaf * kParams;

    const auto [_, dt]   = coord_cur;
    const double dt2     = dt * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    const double d2_cur     = leaves_tree[leaf_offset + 0];
    const double d2_sig_cur = leaves_tree[leaf_offset + 1];
    const double d1_cur     = leaves_tree[leaf_offset + 2];
    const double d1_sig_cur = leaves_tree[leaf_offset + 3];
    const double f0         = leaves_tree[leaf_offset + 6];

    const double dfactor    = utils::kCval / f0;
    const double d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
    const double d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);

    bool branched = false;

    utils::branch_one_param_padded_device(
        0, d2_cur, d2_sig_cur, d2_sig_new, param_limits[0].min,
        param_limits[1].max, eta, shift_d2, ws.scratch_params,
        ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
    const double d1_min = (1 - param_limits[1].max / f0) * utils::kCval;
    const double d1_max = (1 - param_limits[1].min / f0) * utils::kCval;
    utils::branch_one_param_padded_device(
        1, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta, shift_d1,
        ws.scratch_params, ws.scratch_dparams, ws.scratch_counts, flat_base,
        branch_max);

    uint32_t count = 1;
    count *= ws.scratch_counts[flat_base + 0];
    count *= ws.scratch_counts[flat_base + 1];

    ws.leaf_branch_count[ileaf] = count;

    if (branched) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(
            *global_branch_flag);
        flag.fetch_or(1, cuda::memory_order_relaxed);
    }
}

__global__ void kernel_analyze_and_branch_jerk(
    const double* __restrict__ leaves_tree,
    uint32_t n_leaves,
    cuda::std::pair<double, double> coord_cur,
    uint32_t nbins,
    double eta,
    const ParamLimit* __restrict__ param_limits,
    uint32_t branch_max,
    utils::BranchingWorkspaceCUDAView ws,
    int* __restrict__ global_branch_flag) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t leaf_offset = ileaf * kLeavesStride;
    const uint32_t flat_base   = ileaf * kParams;

    const auto [_, dt]   = coord_cur;
    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    const double d3_cur     = leaves_tree[leaf_offset + 0];
    const double d3_sig_cur = leaves_tree[leaf_offset + 1];
    const double d2_cur     = leaves_tree[leaf_offset + 2];
    const double d2_sig_cur = leaves_tree[leaf_offset + 3];
    const double d1_cur     = leaves_tree[leaf_offset + 4];
    const double d1_sig_cur = leaves_tree[leaf_offset + 5];
    const double f0         = leaves_tree[leaf_offset + 8];

    const double dfactor    = utils::kCval / f0;
    const double d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
    const double d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
    const double d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

    const double shift_d3 =
        (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);

    bool branched = false;

    utils::branch_one_param_padded_device(
        0, d3_cur, d3_sig_cur, d3_sig_new, param_limits[0].min,
        param_limits[1].max, eta, shift_d3, ws.scratch_params,
        ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
    utils::branch_one_param_padded_device(
        1, d2_cur, d2_sig_cur, d2_sig_new, param_limits[1].min,
        param_limits[2].max, eta, shift_d2, ws.scratch_params,
        ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
    const double d1_min = (1 - param_limits[2].max / f0) * utils::kCval;
    const double d1_max = (1 - param_limits[2].min / f0) * utils::kCval;
    utils::branch_one_param_padded_device(
        2, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta, shift_d1,
        ws.scratch_params, ws.scratch_dparams, ws.scratch_counts, flat_base,
        branch_max);

    uint32_t count = 1;
    count *= ws.scratch_counts[flat_base + 0];
    count *= ws.scratch_counts[flat_base + 1];
    count *= ws.scratch_counts[flat_base + 2];

    ws.leaf_branch_count[ileaf] = count;

    if (branched) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(
            *global_branch_flag);
        flag.fetch_or(1, cuda::memory_order_relaxed);
    }
}

__global__ void kernel_analyze_and_branch_snap(
    const double* __restrict__ leaves_tree,
    uint32_t n_leaves,
    cuda::std::pair<double, double> coord_cur,
    uint32_t nbins,
    double eta,
    const ParamLimit* __restrict__ param_limits,
    uint32_t branch_max,
    utils::BranchingWorkspaceCUDAView ws,
    int* __restrict__ global_branch_flag) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t leaf_offset = ileaf * kLeavesStride;
    const uint32_t flat_base   = ileaf * kParams;

    const auto [_, dt]   = coord_cur;
    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double dt4     = dt2 * dt2;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const double inv_dt4 = inv_dt2 * inv_dt2;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    const double d4_cur     = leaves_tree[leaf_offset + 0];
    const double d4_sig_cur = leaves_tree[leaf_offset + 1];
    const double d3_cur     = leaves_tree[leaf_offset + 2];
    const double d3_sig_cur = leaves_tree[leaf_offset + 3];
    const double d2_cur     = leaves_tree[leaf_offset + 4];
    const double d2_sig_cur = leaves_tree[leaf_offset + 5];
    const double d1_cur     = leaves_tree[leaf_offset + 6];
    const double d1_sig_cur = leaves_tree[leaf_offset + 7];
    const double f0         = leaves_tree[leaf_offset + 10];

    const double dfactor    = utils::kCval / f0;
    const double d4_sig_new = dphi * dfactor * 192.0 * inv_dt4;
    const double d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
    const double d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
    const double d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

    const double shift_d4 =
        (d4_sig_cur - d4_sig_new) * dt4 * nbins_d / (192.0 * dfactor);
    const double shift_d3 =
        (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);

    bool branched = false;

    utils::branch_one_param_padded_device(
        0, d4_cur, d4_sig_cur, d4_sig_new, param_limits[0].min,
        param_limits[0].max, eta, shift_d4, ws.scratch_params,
        ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
    utils::branch_one_param_padded_device(
        1, d3_cur, d3_sig_cur, d3_sig_new, param_limits[1].min,
        param_limits[1].max, eta, shift_d3, ws.scratch_params,
        ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
    utils::branch_one_param_padded_device(
        2, d2_cur, d2_sig_cur, d2_sig_new, param_limits[2].min,
        param_limits[2].max, eta, shift_d2, ws.scratch_params,
        ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
    const double d1_min = (1 - param_limits[3].max / f0) * utils::kCval;
    const double d1_max = (1 - param_limits[3].min / f0) * utils::kCval;
    utils::branch_one_param_padded_device(
        3, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta, shift_d1,
        ws.scratch_params, ws.scratch_dparams, ws.scratch_counts, flat_base,
        branch_max);

    uint32_t count = 1;
    count *= ws.scratch_counts[flat_base + 0];
    count *= ws.scratch_counts[flat_base + 1];
    count *= ws.scratch_counts[flat_base + 2];
    count *= ws.scratch_counts[flat_base + 3];

    ws.leaf_branch_count[ileaf] = count;

    if (branched) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(
            *global_branch_flag);
        flag.fetch_or(1, cuda::memory_order_relaxed);
    }
}

__global__ void
kernel_materialize_branches_accel(const double* __restrict__ leaves_tree,
                                  double* __restrict__ leaves_branch,
                                  uint32_t* __restrict__ leaves_origins,
                                  uint32_t n_leaves,
                                  uint32_t branch_max,
                                  utils::BranchingWorkspaceCUDAView ws) {
    constexpr uint32_t kParams       = 2;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t leaf_offset   = ileaf * kLeavesStride;
    const uint32_t flat_base     = ileaf * kParams;
    const uint32_t n_d2_branches = ws.scratch_counts[flat_base + 1];
    const uint32_t n_d1_branches = ws.scratch_counts[flat_base + 2];
    const uint32_t off2          = (flat_base + 0) * branch_max;
    const uint32_t off1          = (flat_base + 1) * branch_max;

    uint32_t out = ws.leaf_output_offset[ileaf];

    for (uint32_t a = 0; a < n_d2_branches; ++a) {
        for (uint32_t b = 0; b < n_d1_branches; ++b) {
            const uint32_t bo = out * kLeavesStride;

            leaves_branch[bo + 0] = ws.scratch_params[off2 + a];
            leaves_branch[bo + 1] = ws.scratch_dparams[flat_base + 0];
            leaves_branch[bo + 2] = ws.scratch_params[off1 + b];
            leaves_branch[bo + 3] = ws.scratch_dparams[flat_base + 1];

// copy d0 + f0 + flag (4 doubles)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                leaves_branch[bo + 4 + k] = leaves_tree[leaf_offset + 4 + k];
            }

            leaves_origins[out] = ileaf;
            ++out;
        }
    }
}

__global__ void
kernel_materialize_branches_jerk(const double* __restrict__ leaves_tree,
                                 double* __restrict__ leaves_branch,
                                 uint32_t* __restrict__ leaves_origins,
                                 uint32_t n_leaves,
                                 uint32_t branch_max,
                                 utils::BranchingWorkspaceCUDAView ws) {
    constexpr uint32_t kParams       = 3;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t leaf_offset   = ileaf * kLeavesStride;
    const uint32_t flat_base     = ileaf * kParams;
    const uint32_t n_d3_branches = ws.scratch_counts[flat_base + 1];
    const uint32_t n_d2_branches = ws.scratch_counts[flat_base + 2];
    const uint32_t n_d1_branches = ws.scratch_counts[flat_base + 3];
    const uint32_t off3          = (flat_base + 0) * branch_max;
    const uint32_t off2          = (flat_base + 1) * branch_max;
    const uint32_t off1          = (flat_base + 2) * branch_max;

    uint32_t out = ws.leaf_output_offset[ileaf];

    for (uint32_t a = 0; a < n_d3_branches; ++a) {
        for (uint32_t b = 0; b < n_d2_branches; ++b) {
            for (uint32_t c = 0; c < n_d1_branches; ++c) {
                const uint32_t bo = out * kLeavesStride;

                leaves_branch[bo + 0] = ws.scratch_params[off3 + a];
                leaves_branch[bo + 1] = ws.scratch_dparams[flat_base + 0];
                leaves_branch[bo + 2] = ws.scratch_params[off2 + b];
                leaves_branch[bo + 3] = ws.scratch_dparams[flat_base + 1];
                leaves_branch[bo + 4] = ws.scratch_params[off1 + c];
                leaves_branch[bo + 5] = ws.scratch_dparams[flat_base + 2];

// copy d0 + f0 + flag (4 doubles)
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    leaves_branch[bo + 6 + k] =
                        leaves_tree[leaf_offset + 6 + k];
                }

                leaves_origins[out] = ileaf;
                ++out;
            }
        }
    }
}

__global__ void
kernel_materialize_branches_snap(const double* __restrict__ leaves_tree,
                                 double* __restrict__ leaves_branch,
                                 uint32_t* __restrict__ leaves_origins,
                                 uint32_t n_leaves,
                                 uint32_t branch_max,
                                 utils::BranchingWorkspaceCUDAView ws) {
    constexpr uint32_t kParams       = 4;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t leaf_offset   = ileaf * kLeavesStride;
    const uint32_t flat_base     = ileaf * kParams;
    const uint32_t n_d4_branches = ws.scratch_counts[flat_base + 0];
    const uint32_t n_d3_branches = ws.scratch_counts[flat_base + 1];
    const uint32_t n_d2_branches = ws.scratch_counts[flat_base + 2];
    const uint32_t n_d1_branches = ws.scratch_counts[flat_base + 3];
    const uint32_t off4          = (flat_base + 0) * branch_max;
    const uint32_t off3          = (flat_base + 1) * branch_max;
    const uint32_t off2          = (flat_base + 2) * branch_max;
    const uint32_t off1          = (flat_base + 3) * branch_max;

    uint32_t out = ws.leaf_output_offset[ileaf];

    for (uint32_t a = 0; a < n_d4_branches; ++a) {
        for (uint32_t b = 0; b < n_d3_branches; ++b) {
            for (uint32_t c = 0; c < n_d2_branches; ++c) {
                for (uint32_t d = 0; d < n_d1_branches; ++d) {
                    const uint32_t bo = out * kLeavesStride;

                    leaves_branch[bo + 0] = ws.scratch_params[off4 + a];
                    leaves_branch[bo + 1] = ws.scratch_dparams[flat_base + 0];
                    leaves_branch[bo + 2] = ws.scratch_params[off3 + b];
                    leaves_branch[bo + 3] = ws.scratch_dparams[flat_base + 1];
                    leaves_branch[bo + 4] = ws.scratch_params[off2 + c];
                    leaves_branch[bo + 5] = ws.scratch_dparams[flat_base + 2];
                    leaves_branch[bo + 6] = ws.scratch_params[off1 + d];
                    leaves_branch[bo + 7] = ws.scratch_dparams[flat_base + 3];

// copy d0 + f0 + flag (4 doubles)
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        leaves_branch[bo + 8 + k] =
                            leaves_tree[leaf_offset + 8 + k];
                    }

                    leaves_origins[out] = ileaf;
                    ++out;
                }
            }
        }
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
                     uint32_t n_leaves,
                     const float* __restrict__ accel_grid,
                     uint32_t n_accel,
                     const float* __restrict__ freq_grid,
                     uint32_t n_freq,
                     uint32_t* __restrict__ param_idx,
                     float* __restrict__ phase_shift,
                     cuda::std::pair<double, double> coord_add,
                     cuda::std::pair<double, double> coord_cur,
                     cuda::std::pair<double, double> coord_init,
                     uint32_t nbins) {
    constexpr uint32_t kLeavesStride = 8;
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

    const double* leaf = leaves + static_cast<IndexType>((tid * kLeavesStride));
    const double a_cur = leaf[0];
    const double v_cur = leaf[2];
    const double f0    = leaf[6];

    const double delta_v   = a_cur * dt_rel;
    const double delta_d   = (v_cur * dt_rel) + (a_cur * half_dt_sq);
    const double f_new     = f0 * (1.0 - delta_v * utils::kInvCval);
    const double delay_rel = delta_d * utils::kInvCval;

    const uint32_t idx_a = utils::nearest_linear_scan(
        accel_arr, n_accel, static_cast<float>(a_cur));
    const uint32_t idx_f =
        utils::lower_bound_scanf(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid] = idx_a * n_freq + idx_f;
    phase_shift[tid] =
        utils::get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
}

template <bool UseShared>
__device__ __forceinline__ void
resolve_taylor_jerk(const double* __restrict__ leaves,
                    uint32_t n_leaves,
                    const float* __restrict__ accel_grid,
                    uint32_t n_accel,
                    const float* __restrict__ freq_grid,
                    uint32_t n_freq,
                    uint32_t* __restrict__ param_idx,
                    float* __restrict__ phase_shift,
                    cuda::std::pair<double, double> coord_add,
                    cuda::std::pair<double, double> coord_cur,
                    cuda::std::pair<double, double> coord_init,
                    uint32_t nbins) {
    constexpr uint32_t kLeavesStride = 10;
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

    const double* leaf = leaves + static_cast<IndexType>(tid * kLeavesStride);
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

    const uint32_t idx_a = utils::nearest_linear_scan(
        accel_arr, n_accel, static_cast<float>(a_new));
    const uint32_t idx_f =
        utils::lower_bound_scanf(freq_arr, n_freq, static_cast<float>(f_new));
    param_idx[tid] = idx_a * n_freq + idx_f;
    phase_shift[tid] =
        utils::get_phase_idx_device(dt_rel, f0, nbins, delay_rel);
}

template <bool UseShared>
__device__ __forceinline__ void
resolve_taylor_snap(const double* __restrict__ leaves,
                    uint32_t n_leaves,
                    const float* __restrict__ accel_grid,
                    uint32_t n_accel,
                    const float* __restrict__ freq_grid,
                    uint32_t n_freq,
                    uint32_t* __restrict__ param_idx,
                    float* __restrict__ phase_shift,
                    cuda::std::pair<double, double> coord_add,
                    cuda::std::pair<double, double> coord_cur,
                    cuda::std::pair<double, double> coord_init,
                    uint32_t nbins) {
    constexpr uint32_t kLeavesStride = 12;
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

    const double* leaf = leaves + static_cast<IndexType>(tid * kLeavesStride);
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

    const uint32_t idx_a = utils::nearest_linear_scan(
        accel_arr, n_accel, static_cast<float>(a_new));
    const uint32_t idx_f =
        utils::lower_bound_scanf(freq_arr, n_freq, static_cast<float>(f_new));
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

// Resolve Kernel
template <int NPARAMS, bool UseShared>
__global__ void
kernel_poly_taylor_resolve_batch(const double* __restrict__ leaves,
                                 SizeType n_leaves,
                                 const float* __restrict__ accel_grid,
                                 SizeType n_accel,
                                 const float* __restrict__ freq_grid,
                                 SizeType n_freq,
                                 uint32_t* __restrict__ param_idx,
                                 float* __restrict__ phase_shift,
                                 cuda::std::pair<double, double> coord_add,
                                 cuda::std::pair<double, double> coord_cur,
                                 cuda::std::pair<double, double> coord_init,
                                 SizeType nbins) {
    if constexpr (NPARAMS == 2) {
        resolve_taylor_accel<UseShared>(
            leaves, n_leaves, accel_grid, n_accel, freq_grid, n_freq, param_idx,
            phase_shift, coord_add, coord_cur, coord_init, nbins);
    } else if constexpr (NPARAMS == 3) {
        resolve_taylor_jerk<UseShared>(
            leaves, n_leaves, accel_grid, n_accel, freq_grid, n_freq, param_idx,
            phase_shift, coord_add, coord_cur, coord_init, nbins);
    } else if constexpr (NPARAMS == 4) {
        resolve_taylor_snap<UseShared>(
            leaves, n_leaves, accel_grid, n_accel, freq_grid, n_freq, param_idx,
            phase_shift, coord_add, coord_cur, coord_init, nbins);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Taylor order");
    }
}

// Transform Kernel
template <int NPARAMS, bool UseConservativeTile>
__global__ void
kernel_poly_taylor_transform_batch(double* __restrict__ leaves_tree,
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

SizeType poly_taylor_branch_accel_cuda(
    cuda::std::span<const double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    std::pair<double, double> coord_cur,
    SizeType nbins,
    double eta,
    cuda::std::span<const ParamLimit> param_limits,
    SizeType branch_max,
    SizeType n_leaves,
    utils::BranchingWorkspaceCUDAView ws,
    cudaStream_t stream) {
    constexpr SizeType kLeavesStride = 8;

    int h_global_branch_flag = 0;
    int* d_global_branch_flag;
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&d_global_branch_flag, sizeof(int), stream),
        "cudaMallocAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_global_branch_flag, 0, sizeof(int), stream),
        "cudaMemsetAsync failed");

    // ---- Kernel 1: analyze + branch enumeration ----
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    kernel_analyze_and_branch_accel<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), n_leaves, coord_cur, nbins, eta,
        param_limits.data(), branch_max, ws, d_global_branch_flag);
    cuda_utils::check_last_cuda_error("Kernel 1 launch failed");

    // ---- check global flag ----
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&h_global_branch_flag, d_global_branch_flag,
                        sizeof(int), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    if (h_global_branch_flag == 0) {
        // FAST PATH: no branching
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_branch.data(), leaves_tree.data(),
                            n_leaves * kLeavesStride * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync failed");
        thrust::sequence(thrust::cuda::par.on(stream), leaves_origins.data(),
                         leaves_origins.data() + n_leaves,
                         static_cast<uint32_t>(0));

        return n_leaves;
    }

    // scan for output offsets (leaf_output_offset)
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(ws.cub_temp_storage, ws.cub_temp_bytes,
                                      ws.leaf_branch_count,
                                      ws.leaf_output_offset, n_leaves, stream),
        "cub::DeviceScan::ExclusiveSum failed");

    // ---- compute output size ----
    SizeType last_offset, last_count;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_offset, ws.leaf_output_offset + (n_leaves - 1),
                        sizeof(SizeType), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_count, ws.leaf_branch_count + (n_leaves - 1),
                        sizeof(SizeType), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    SizeType out_leaves = last_offset + last_count;

    // ---- Kernel 2: materialize ----
    kernel_materialize_branches_accel<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
        n_leaves, branch_max, ws);
    cuda_utils::check_last_cuda_error("Kernel 2 launch failed");

    return out_leaves;
}

SizeType poly_taylor_branch_jerk_cuda(
    cuda::std::span<const double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    std::pair<double, double> coord_cur,
    SizeType nbins,
    double eta,
    cuda::std::span<const ParamLimit> param_limits,
    SizeType branch_max,
    SizeType n_leaves,
    utils::BranchingWorkspaceCUDAView ws,
    cudaStream_t stream) {
    constexpr SizeType kLeavesStride = 10;

    int h_global_branch_flag = 0;
    int* d_global_branch_flag;
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&d_global_branch_flag, sizeof(int), stream),
        "cudaMallocAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_global_branch_flag, 0, sizeof(int), stream),
        "cudaMemsetAsync failed");

    // ---- Kernel 1: analyze + branch enumeration ----
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    kernel_analyze_and_branch_jerk<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), n_leaves, coord_cur, nbins, eta,
        param_limits.data(), branch_max, ws, d_global_branch_flag);
    cuda_utils::check_last_cuda_error("Kernel 1 launch failed");

    // ---- check global flag ----
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&h_global_branch_flag, d_global_branch_flag,
                        sizeof(int), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    if (h_global_branch_flag == 0) {
        // FAST PATH: no branching
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_branch.data(), leaves_tree.data(),
                            n_leaves * kLeavesStride * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync failed");
        thrust::sequence(thrust::cuda::par.on(stream), leaves_origins.data(),
                         leaves_origins.data() + n_leaves,
                         static_cast<uint32_t>(0));

        return n_leaves;
    }

    // scan for output offsets (leaf_output_offset)
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(ws.cub_temp_storage, ws.cub_temp_bytes,
                                      ws.leaf_branch_count,
                                      ws.leaf_output_offset, n_leaves, stream),
        "cub::DeviceScan::ExclusiveSum failed");

    // ---- compute output size ----
    SizeType last_offset, last_count;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_offset, ws.leaf_output_offset + (n_leaves - 1),
                        sizeof(SizeType), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_count, ws.leaf_branch_count + (n_leaves - 1),
                        sizeof(SizeType), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    SizeType out_leaves = last_offset + last_count;

    // ---- Kernel 2: materialize ----
    kernel_materialize_branches_jerk<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
        n_leaves, branch_max, ws);
    cuda_utils::check_last_cuda_error("Kernel 2 launch failed");

    return out_leaves;
}

SizeType poly_taylor_branch_snap_cuda(
    cuda::std::span<const double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    std::pair<double, double> coord_cur,
    SizeType nbins,
    double eta,
    cuda::std::span<const ParamLimit> param_limits,
    SizeType branch_max,
    SizeType n_leaves,
    utils::BranchingWorkspaceCUDAView ws,
    cudaStream_t stream) {
    constexpr SizeType kLeavesStride = 12;

    int h_global_branch_flag = 0;
    int* d_global_branch_flag;
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&d_global_branch_flag, sizeof(int), stream),
        "cudaMallocAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_global_branch_flag, 0, sizeof(int), stream),
        "cudaMemsetAsync failed");

    // ---- Kernel 1: analyze + branch enumeration ----
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    kernel_analyze_and_branch_snap<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), n_leaves, coord_cur, nbins, eta,
        param_limits.data(), branch_max, ws, d_global_branch_flag);
    cuda_utils::check_last_cuda_error("Kernel 1 launch failed");

    // ---- check global flag ----
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&h_global_branch_flag, d_global_branch_flag,
                        sizeof(int), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    if (h_global_branch_flag == 0) {
        // FAST PATH: no branching
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_branch.data(), leaves_tree.data(),
                            n_leaves * kLeavesStride * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync failed");
        thrust::sequence(thrust::cuda::par.on(stream), leaves_origins.data(),
                         leaves_origins.data() + n_leaves,
                         static_cast<uint32_t>(0));

        return n_leaves;
    }

    // scan for output offsets (leaf_output_offset)
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(ws.cub_temp_storage, ws.cub_temp_bytes,
                                      ws.leaf_branch_count,
                                      ws.leaf_output_offset, n_leaves, stream),
        "cub::DeviceScan::ExclusiveSum failed");

    // ---- compute output size ----
    SizeType last_offset, last_count;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_offset, ws.leaf_output_offset + (n_leaves - 1),
                        sizeof(SizeType), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_count, ws.leaf_branch_count + (n_leaves - 1),
                        sizeof(SizeType), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    SizeType out_leaves = last_offset + last_count;

    // ---- Kernel 2: materialize ----
    kernel_materialize_branches_snap<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
        n_leaves, branch_max, ws);
    cuda_utils::check_last_cuda_error("Kernel 2 launch failed");

    return out_leaves;
}

} // namespace

SizeType
poly_taylor_seed_cuda(cuda::std::span<const SizeType> param_grid_count_init,
                      cuda::std::span<const double> dparams_init,
                      cuda::std::span<const ParamLimit> param_limits,
                      cuda::std::span<double> seed_leaves,
                      SizeType n_params,
                      cudaStream_t stream) {
    SizeType n_leaves = 1;
    for (const auto count : param_grid_count_init) {
        n_leaves *= count;
    }
    constexpr SizeType kThreadPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadPerBlock - 1) / kThreadPerBlock;

    const dim3 block_dim(kThreadPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_poly_taylor_seed<<<grid_dim, block_dim>>>(
        param_grid_count_init.data(), dparams_init.data(), param_limits.data(),
        seed_leaves.data(), n_leaves, n_params);
    cuda_utils::check_last_cuda_error("Taylor seed kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
    return n_leaves;
}

SizeType poly_taylor_branch_batch_cuda(
    cuda::std::span<const double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    std::pair<double, double> coord_cur,
    SizeType nbins,
    double eta,
    cuda::std::span<const ParamLimit> param_limits,
    SizeType branch_max,
    SizeType n_leaves,
    SizeType n_params,
    utils::BranchingWorkspaceCUDAView ws,
    cudaStream_t stream) {
    if (n_params == 2) {
        return poly_taylor_branch_accel_cuda(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws, stream);
    } else if (n_params == 3) {
        return poly_taylor_branch_jerk_cuda(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws, stream);
    } else if (n_params == 4) {
        return poly_taylor_branch_snap_cuda(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws, stream);
    } else {
        throw std::invalid_argument("Unsupported n_params");
    }
}

void poly_taylor_resolve_batch_cuda(cuda::std::span<const double> leaves_tree,
                                    cuda::std::span<const float> accel_grid,
                                    cuda::std::span<const float> freq_grid,
                                    cuda::std::span<uint32_t> param_indices,
                                    cuda::std::span<float> phase_shift,
                                    std::pair<double, double> coord_add,
                                    std::pair<double, double> coord_cur,
                                    std::pair<double, double> coord_init,
                                    SizeType nbins,
                                    SizeType n_leaves,
                                    SizeType n_params,
                                    cudaStream_t stream) {
    if (n_leaves == 0) {
        return;
    }
    const SizeType n_accel = accel_grid.size();
    const SizeType n_freq  = freq_grid.size();

    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((n_leaves + kBlockSize - 1) / kBlockSize);

    // Calculate Shared Memory Strategy
    const SizeType max_shmem = cuda_utils::get_max_shared_memory();
    SizeType shmem_bytes     = (n_accel + n_freq) * sizeof(float);
    const bool use_smem      = (shmem_bytes <= max_shmem);
    if (!use_smem) {
        shmem_bytes = 0; // No shared memory needed
    }
    cuda_utils::check_kernel_launch_params(grid, block, shmem_bytes);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<int N, bool S>() {
        kernel_poly_taylor_resolve_batch<N, S>
            <<<grid, block, shmem_bytes, stream>>>(
                leaves_tree.data(), n_leaves, accel_grid.data(), n_accel,
                freq_grid.data(), n_freq, param_indices.data(),
                phase_shift.data(), coord_add, coord_cur, coord_init, nbins);
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

void poly_taylor_transform_batch_cuda(cuda::std::span<double> leaves_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool use_conservative_tile,
                                      cudaStream_t stream) {
    // Better occupancy than 512 for many kernels
    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((n_leaves + kBlockSize - 1) / kBlockSize);
    cuda_utils::check_kernel_launch_params(grid, block);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<int N, bool C>() {
        kernel_poly_taylor_transform_batch<N, C><<<grid, block, 0, stream>>>(
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