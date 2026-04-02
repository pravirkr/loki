#include "loki/core/circular.hpp"

#include <cstdint>
#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <sys/types.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/kernel_utils.cuh"
#include "loki/utils.hpp"

namespace loki::core {

namespace {

// 0: Taylor, 1: Circular Snap, 2: Circular Crackle
__device__ __forceinline__ uint32_t
get_circ_taylor_mask_device(double crackle,
                            double dcrackle,
                            double snap,
                            double dsnap,
                            double jerk,
                            double accel,
                            double minimum_snap_cells) {
    const bool is_sig_snap =
        std::abs(snap) > (minimum_snap_cells * (dsnap + utils::kEps));
    if (!is_sig_snap) {
        return 0;
    }
    const bool is_physical_snap = ((-snap * accel) > 0.0) &&
                                  (std::abs(accel) > utils::kEps) &&
                                  (std::abs(snap) > utils::kEps);
    if (is_sig_snap && is_physical_snap) {
        return 1;
    }
    const bool is_sig_crackle =
        std::abs(crackle) > (minimum_snap_cells * (dcrackle + utils::kEps));
    const bool is_physical_crackle = ((-crackle * jerk) > 0.0) &&
                                     (std::abs(jerk) > utils::kEps) &&
                                     (std::abs(crackle) > utils::kEps);
    if (is_sig_snap && !is_physical_snap && is_sig_crackle &&
        is_physical_crackle) {
        return 2;
    }
    return 0;
}

__device__ __forceinline__ bool
is_in_hole_device(double snap,
                  double dsnap,
                  double jerk,
                  double accel,
                  double minimum_snap_cells) noexcept {
    const bool is_sig_snap =
        abs(snap) > (minimum_snap_cells * (dsnap + utils::kEps));
    const bool is_stable_snap =
        (abs(accel) > utils::kEps) && (abs(snap) > utils::kEps);
    const bool is_stable_jerk = abs(jerk) > utils::kEps;
    return is_sig_snap && (!is_stable_snap) && is_stable_jerk;
}

__global__ void
kernel_analyze_and_branch_circular(const double* __restrict__ leaves_tree,
                                   uint32_t n_leaves,
                                   double dt,
                                   double nbins,
                                   double eta,
                                   const ParamLimit* __restrict__ param_limits,
                                   uint32_t branch_max,
                                   memory::BranchingWorkspaceCUDAView branch_ws,
                                   memory::CUBScratchArena& scratch_ws) {
    constexpr SizeType kParams       = 5;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t lo = ileaf * kLeavesStride;
    const uint32_t fb = ileaf * kParams;

    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double dt4     = dt2 * dt2;
    const double dt5     = dt4 * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const double inv_dt4 = inv_dt2 * inv_dt2;
    const double inv_dt5 = inv_dt4 * inv_dt;
    const double dphi    = eta / nbins;

    const double d5_cur     = leaves_tree[lo + 0];
    const double d5_sig_cur = leaves_tree[lo + 1];
    const double d4_cur     = leaves_tree[lo + 2];
    const double d4_sig_cur = leaves_tree[lo + 3];
    const double d3_cur     = leaves_tree[lo + 4];
    const double d3_sig_cur = leaves_tree[lo + 5];
    const double d2_cur     = leaves_tree[lo + 6];
    const double d2_sig_cur = leaves_tree[lo + 7];
    const double d1_cur     = leaves_tree[lo + 8];
    const double d1_sig_cur = leaves_tree[lo + 9];
    const double f0         = leaves_tree[lo + 12];

    const double dfactor  = utils::kCval / f0;
    const double d5_range = param_limits[0].max - param_limits[0].min;
    const double d4_range = param_limits[1].max - param_limits[1].min;
    const double d3_range = param_limits[2].max - param_limits[2].min;
    const double d2_range = param_limits[3].max - param_limits[3].min;
    const double d1_range =
        dfactor * (param_limits[4].max - param_limits[4].min);

    const double d5_sig_new =
        cuda::std::min(dphi * dfactor * 1920.0 * inv_dt5, d5_range);
    const double d4_sig_new =
        cuda::std::min(dphi * dfactor * 192.0 * inv_dt4, d4_range);
    const double d3_sig_new =
        cuda::std::min(dphi * dfactor * 24.0 * inv_dt3, d3_range);
    const double d2_sig_new =
        cuda::std::min(dphi * dfactor * 4.0 * inv_dt2, d2_range);
    const double d1_sig_new =
        cuda::std::min(dphi * dfactor * 1.0 * inv_dt, d1_range);

    const double shift_d5 =
        (d5_sig_cur - d5_sig_new) * dt5 * nbins / (1920.0 * dfactor);
    const double shift_d4 =
        (d4_sig_cur - d4_sig_new) * dt4 * nbins / (192.0 * dfactor);
    const double shift_d3 =
        (d3_sig_cur - d3_sig_new) * dt3 * nbins / (24.0 * dfactor);
    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins / (1.0 * dfactor);

    // Store d5 as single entry in scratch (count=1)
    const uint32_t pad_offset            = (fb + 0) * branch_max;
    branch_ws.scratch_params[pad_offset] = d5_cur;
    branch_ws.scratch_dparams[fb + 0]    = d5_sig_cur;
    branch_ws.scratch_counts[fb + 0]     = 1;

    // Branch d4–d1 into scratch workspace
    utils::branch_one_param_padded_device(
        1, d4_cur, d4_sig_cur, d4_sig_new, eta, shift_d4,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        2, d3_cur, d3_sig_cur, d3_sig_new, eta, shift_d3,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        3, d2_cur, d2_sig_cur, d2_sig_new, eta, shift_d2,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        4, d1_cur, d1_sig_cur, d1_sig_new, eta, shift_d1,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);

    const uint32_t c1 = branch_ws.scratch_counts[fb + 1];
    const uint32_t c2 = branch_ws.scratch_counts[fb + 2];
    const uint32_t c3 = branch_ws.scratch_counts[fb + 3];
    const uint32_t c4 = branch_ws.scratch_counts[fb + 4];

    branch_ws.leaf_branch_count[ileaf] = c1 * c2 * c3 * c4;
    if (shift_d5 >= (eta - utils::kEps)) {
        atomicOr(scratch_ws.d_reduce_out, 1U);
    }
}

template <uint32_t KThreadsPerBlock>
__global__ void kernel_materialize_branches_circular(
    const double* __restrict__ leaves_tree,
    double* __restrict__ leaves_branch,
    uint32_t* __restrict__ leaves_origins,
    uint32_t n_leaves,
    uint32_t n_leaves_branched,
    SizeType branch_max,
    memory::BranchingWorkspaceCUDAView branch_ws) {
    constexpr uint32_t kParams       = 5;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    // Strategy: Block-Cooperative Search
    __shared__ uint32_t smem_offsets[KThreadsPerBlock + 2];
    __shared__ uint32_t base_leaf_idx;

    utils::load_block_cooperative_map_device<KThreadsPerBlock>(
        smem_offsets, base_leaf_idx, branch_ws.leaf_output_offset, n_leaves);

    const uint32_t tid          = threadIdx.x;
    const uint32_t branch_ileaf = (blockIdx.x * blockDim.x) + tid;

    if (branch_ileaf >= n_leaves_branched) {
        return;
    }

    // Phase 3: SMEM Search
    const uint32_t search_len =
        min(KThreadsPerBlock + 1, n_leaves - base_leaf_idx);
    const uint32_t local_leaf_idx =
        utils::binary_search_cartesian(smem_offsets, search_len, branch_ileaf);

    // Resolve actual leaf index
    const uint32_t tree_ileaf        = base_leaf_idx + local_leaf_idx;
    const uint32_t leaf_start_offset = smem_offsets[local_leaf_idx];
    const uint32_t flat_base         = tree_ileaf * kParams;

    const uint32_t n_d3 = branch_ws.scratch_counts[flat_base + 2];
    const uint32_t n_d2 = branch_ws.scratch_counts[flat_base + 3];
    const uint32_t n_d1 = branch_ws.scratch_counts[flat_base + 4];

    // Inverse Mapping (Div/Mod)
    uint32_t rem     = branch_ileaf - leaf_start_offset;
    const uint32_t e = rem % n_d1;
    rem /= n_d1; // d1
    const uint32_t d = rem % n_d2;
    rem /= n_d2; // d2
    const uint32_t c = rem % n_d3;
    rem /= n_d3;            // d3
    const uint32_t b = rem; // d4 (rem < n_d4 by construction)

    const uint32_t bo = branch_ileaf * kLeavesStride;
    const uint32_t lo = tree_ileaf * kLeavesStride;

    // Write d5 as single value (count=1, no branching yet)
    leaves_branch[bo + 0] =
        branch_ws.scratch_params[(flat_base + 0) * branch_max];
    leaves_branch[bo + 1] =
        branch_ws.scratch_dparams[flat_base + 0]; // d5_sig_cur
    leaves_branch[bo + 2] =
        branch_ws.scratch_params[((flat_base + 1) * branch_max) + b];
    leaves_branch[bo + 3] = branch_ws.scratch_dparams[flat_base + 1];
    leaves_branch[bo + 4] =
        branch_ws.scratch_params[((flat_base + 2) * branch_max) + c];
    leaves_branch[bo + 5] = branch_ws.scratch_dparams[flat_base + 2];
    leaves_branch[bo + 6] =
        branch_ws.scratch_params[((flat_base + 3) * branch_max) + d];
    leaves_branch[bo + 7] = branch_ws.scratch_dparams[flat_base + 3];
    leaves_branch[bo + 8] =
        branch_ws.scratch_params[((flat_base + 4) * branch_max) + e];
    leaves_branch[bo + 9] = branch_ws.scratch_dparams[flat_base + 4];

// copy d0 + f0 + flag (4 doubles) from parent tree
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        leaves_branch[bo + 10 + k] = leaves_tree[lo + 10 + k];
    }
    leaves_origins[branch_ileaf] = tree_ileaf;
}

__global__ void
kernel_expand_crackle_holes(double* __restrict__ leaves_branch,
                            uint32_t* __restrict__ leaves_origins,
                            uint32_t n_leaves_branched,
                            double dt,
                            double nbins,
                            double eta,
                            const ParamLimit* __restrict__ param_limits,
                            double minimum_snap_cells,
                            uint32_t* __restrict__ out_count) {
    constexpr uint32_t kParams       = 5;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;
    constexpr uint32_t kMaxCrackle   = 16;

    const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n_leaves_branched) {
        return;
    }

    const uint32_t bo         = i * kLeavesStride;
    double* __restrict__ leaf = leaves_branch + bo;

    if (!is_in_hole_device(leaf[2], leaf[3], leaf[4], leaf[6],
                           minimum_snap_cells)) {
        return;
    }

    const uint32_t origin   = leaves_origins[i];
    const double d5_cur     = leaf[0];
    const double d5_sig_cur = leaf[1];

    // Recompute d5_sig_new and shift_d5
    const double f0       = leaf[12];
    const double dt5      = dt * dt * dt * dt * dt;
    const double inv_dt5  = 1.0 / dt5;
    const double dphi     = eta / nbins;
    const double dfactor  = utils::kCval / f0;
    const double d5_range = param_limits[0].max - param_limits[0].min;
    const double d5_sig_new =
        cuda::std::min(dphi * dfactor * 1920.0 * inv_dt5, d5_range);
    const double shift_d5 =
        (d5_sig_cur - d5_sig_new) * dt5 * nbins / (1920.0 * dfactor);

    // Per-thread register buffer — no scratch, no race
    double local_d5_vals[kMaxCrackle];
    uint32_t n_d5;
    double dparam_act;
    if (shift_d5 >= (eta - utils::kEps)) {
        auto [act, count] = utils::branch_param_generate_points_device(
            local_d5_vals, kMaxCrackle, d5_cur, d5_sig_cur, d5_sig_new);
        dparam_act = act;
        n_d5       = count;
    } else {
        local_d5_vals[0] = d5_cur;
        dparam_act       = d5_sig_cur;
        n_d5             = 1;
    }
    assert(n_d5 <= kMaxCrackle); // catches misconfiguration early

    // Overwrite slot i in-place
    leaf[0] = local_d5_vals[0];
    leaf[1] = dparam_act;

    if (n_d5 > 1) {
        const uint32_t extra      = n_d5 - 1;
        const uint32_t tail_start = atomicAdd(out_count, extra);

        for (uint32_t a = 1; a < n_d5; ++a) {
            double* __restrict__ tout =
                leaves_branch +
                static_cast<IndexType>((tail_start + a - 1) * kLeavesStride);
#pragma unroll
            for (uint32_t k = 0; k < kLeavesStride; ++k) {
                tout[k] = leaf[k];
            }
            tout[0]                            = local_d5_vals[a];
            leaves_origins[tail_start + a - 1] = origin;
        }
    }
}

__global__ void
kernel_validate_branches_circular(const double* __restrict__ leaves_branch,
                                  uint8_t* __restrict__ validation_mask,
                                  uint32_t n_leaves,
                                  double p_orb_min,
                                  double x_mass_const,
                                  double minimum_snap_cells) {
    constexpr uint32_t kLeavesStride = 14;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const uint32_t lo  = tid * kLeavesStride;
    const double snap  = leaves_branch[lo + 2];
    const double dsnap = leaves_branch[lo + 3];
    const double jerk  = leaves_branch[lo + 4];
    const double accel = leaves_branch[lo + 6];

    const double omega_max_sq = pow(2.0 * M_PI / p_orb_min, 2);
    const bool is_sig_snap =
        fabs(snap) > (minimum_snap_cells * (dsnap + utils::kEps));
    const bool snap_possible =
        (fabs(accel) > utils::kEps) && (fabs(snap) > utils::kEps);
    const bool sign_valid  = (-snap * accel) > 0.0;
    const bool snap_region = is_sig_snap && snap_possible && sign_valid;
    if (!snap_region) {
        validation_mask[tid] = 1U;
        return;
    }
    // Inside snap_region: omega_sq is guaranteed positive
    const double omega_sq = -snap / accel;
    const double limit_accel =
        x_mass_const * (pow(omega_sq, 2.0 / 3.0) + utils::kEps);
    const bool valid_omega = omega_sq < omega_max_sq;
    // |d2| < x * omega^(4/3)
    const bool valid_accel = fabs(accel) <= limit_accel;
    // |d3| < |d2| * omega  =>  d3^2 < d2^2 * omega^2
    const bool valid_jerk = (jerk * jerk) <= (accel * accel * omega_sq);
    validation_mask[tid]  = (valid_omega && valid_accel && valid_jerk) ? 1 : 0;
}

__global__ void
kernel_circ_taylor_resolve_batch(const double* __restrict__ leaves_tree,
                                 const uint8_t* __restrict__ validation_mask,
                                 uint32_t* __restrict__ param_indices,
                                 float* __restrict__ phase_shift,
                                 const ParamLimit* __restrict__ param_limits,
                                 cuda::std::pair<double, double> coord_add,
                                 cuda::std::pair<double, double> coord_cur,
                                 cuda::std::pair<double, double> coord_init,
                                 uint32_t n_accel_init,
                                 uint32_t n_freq_init,
                                 uint32_t nbins,
                                 uint32_t n_leaves,
                                 double minimum_snap_cells) {
    constexpr uint32_t kLeavesStride = 14;

    // Compute locally
    const double dt_add         = coord_add.first - coord_cur.first;
    const double dt_init        = coord_init.first - coord_cur.first;
    const double half_dt2_add   = 0.5 * (dt_add * dt_add);
    const double half_dt2_init  = 0.5 * (dt_init * dt_init);
    const double dt             = dt_add - dt_init;
    const double sixth_dt3_add  = half_dt2_add * dt_add / 3.0;
    const double sixth_dt3_init = half_dt2_init * dt_init / 3.0;
    const double half_dt2       = half_dt2_add - half_dt2_init;
    const double sixth_dt3      = sixth_dt3_add - sixth_dt3_init;
    const double twenty_fourth_dt4 =
        ((sixth_dt3_add * dt_add) - (sixth_dt3_init * dt_init)) / 4.0;
    const double onehundred_twenty_dt5 =
        ((sixth_dt3_add * half_dt2_add) - (sixth_dt3_init * half_dt2_init)) /
        10.0;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const uint32_t lo     = tid * kLeavesStride;
    const double c_t_cur  = leaves_tree[lo + 0];
    const double dc_t_cur = leaves_tree[lo + 1];
    const double s_t_cur  = leaves_tree[lo + 2];
    const double ds_t_cur = leaves_tree[lo + 3];
    const double j_t_cur  = leaves_tree[lo + 4];
    const double a_t_cur  = leaves_tree[lo + 6];
    const double v_t_cur  = leaves_tree[lo + 8];
    const double f0       = leaves_tree[lo + 12];

    double a_new, delta_v, delta_d;
    const uint32_t mask_circular =
        get_circ_taylor_mask_device(c_t_cur, dc_t_cur, s_t_cur, ds_t_cur,
                                    j_t_cur, a_t_cur, minimum_snap_cells);
    if (mask_circular == 0) {
        a_new   = a_t_cur + (j_t_cur * dt_add) + (s_t_cur * half_dt2_add) +
                  (c_t_cur * sixth_dt3_add);
        delta_v = (a_t_cur * dt) + (j_t_cur * half_dt2) +
                  (s_t_cur * sixth_dt3) + (c_t_cur * twenty_fourth_dt4);
        delta_d = (v_t_cur * dt) + (a_t_cur * half_dt2) +
                  (j_t_cur * sixth_dt3) + (s_t_cur * twenty_fourth_dt4) +
                  (c_t_cur * onehundred_twenty_dt5);
    } else {
        const double omega_orb_sq =
            mask_circular == 1 ? -s_t_cur / a_t_cur : -c_t_cur / a_t_cur;
        const double omega_orb = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const double omega_dt_add = omega_orb * dt_add;
        const double cos_odt_add  = std::cos(omega_dt_add);
        const double sin_odt_add  = std::sin(omega_dt_add);
        const double a_t_add =
            (a_t_cur * cos_odt_add) + ((j_t_cur / omega_orb) * sin_odt_add);
        const double j_t_add =
            (j_t_cur * cos_odt_add) - ((a_t_cur * omega_orb) * sin_odt_add);

        const double omega_dt_init = omega_orb * dt_init;
        const double cos_odt_init  = std::cos(omega_dt_init);
        const double sin_odt_init  = std::sin(omega_dt_init);
        const double a_t_init =
            (a_t_cur * cos_odt_init) + ((j_t_cur / omega_orb) * sin_odt_init);
        const double j_t_init =
            (j_t_cur * cos_odt_init) - ((a_t_cur * omega_orb) * sin_odt_init);
        a_new   = a_t_add;
        delta_v = (-j_t_add / omega_orb_sq) - (-j_t_init / omega_orb_sq);
        delta_d = (-a_t_add / omega_orb_sq) - (-a_t_init / omega_orb_sq) +
                  ((v_t_cur + (j_t_cur / omega_orb_sq)) * dt);
    }
    const double f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
    const double delay_rel = delta_d * utils::kInvCval;

    const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
        a_new, param_limits[3].min, param_limits[3].max, n_accel_init);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_new, param_limits[4].min, param_limits[4].max, n_freq_init);
    param_indices[tid] = (idx_a * n_freq_init) + idx_f;
    phase_shift[tid]   = utils::get_phase_idx_device(dt, f0, nbins, delay_rel);
}

template <bool UseConservativeTile>
__global__ void
kernel_circ_taylor_transform_batch(double* __restrict__ leaves_tree,
                                   const uint8_t* __restrict__ validation_mask,
                                   uint32_t n_leaves,
                                   cuda::std::pair<double, double> coord_next,
                                   cuda::std::pair<double, double> coord_cur,
                                   double minimum_snap_cells) {
    constexpr uint32_t kLeavesStride = 14;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const double dt                    = coord_next.first - coord_cur.first;
    const double dt2                   = dt * dt;
    const double half_dt2              = 0.5 * dt2;
    const double sixth_dt3             = dt2 * dt / 6.0;
    const double twenty_fourth_dt4     = dt2 * dt2 / 24.0;
    const double onehundred_twenty_dt5 = (dt2 * dt2 * dt) / 120.0;

    const uint32_t lo     = tid * kLeavesStride;
    const double d5_val_i = leaves_tree[lo + 0];
    const double d5_err_i = leaves_tree[lo + 1];
    const double d4_val_i = leaves_tree[lo + 2];
    const double d4_err_i = leaves_tree[lo + 3];
    const double d3_val_i = leaves_tree[lo + 4];
    const double d3_err_i = leaves_tree[lo + 5];
    const double d2_val_i = leaves_tree[lo + 6];
    const double d2_err_i = leaves_tree[lo + 7];
    const double d1_val_i = leaves_tree[lo + 8];
    const double d1_err_i = leaves_tree[lo + 9];
    const double d0_val_i = leaves_tree[lo + 10];
    const double d0_err_i = leaves_tree[lo + 11];

    const bool mask_circular =
        get_circ_taylor_mask_device(d5_val_i, d5_err_i, d4_val_i, d4_err_i,
                                    d3_val_i, d2_val_i, minimum_snap_cells);

    if (mask_circular == 0) {
        leaves_tree[lo + 0] = d5_val_i;
        leaves_tree[lo + 2] = d4_val_i + (d5_val_i * dt);
        leaves_tree[lo + 4] =
            d3_val_i + (d4_val_i * dt) + (d5_val_i * half_dt2);
        leaves_tree[lo + 6]  = d2_val_i + (d3_val_i * dt) +
                               (d4_val_i * half_dt2) + (d5_val_i * sixth_dt3);
        leaves_tree[lo + 8]  = d1_val_i + (d2_val_i * dt) +
                               (d3_val_i * half_dt2) + (d4_val_i * sixth_dt3) +
                               (d5_val_i * twenty_fourth_dt4);
        leaves_tree[lo + 10] = d0_val_i + (d1_val_i * dt) +
                               (d2_val_i * half_dt2) + (d3_val_i * sixth_dt3) +
                               (d4_val_i * twenty_fourth_dt4) +
                               (d5_val_i * onehundred_twenty_dt5);
    } else {
        const double omega_orb_sq =
            mask_circular == 1 ? -d4_val_i / d2_val_i : -d5_val_i / d3_val_i;
        const double omega_orb = std::sqrt(omega_orb_sq);
        const double omega_dt  = omega_orb * dt;
        const double cos_odt   = std::cos(omega_dt);
        const double sin_odt   = std::sin(omega_dt);
        // Pin-down {s, j, a}
        const double d2_j =
            (d2_val_i * cos_odt) + (d3_val_i * sin_odt / omega_orb);
        const double d3_j =
            (d3_val_i * cos_odt) - (d2_val_i * sin_odt * omega_orb);
        const double d4_j = -omega_orb_sq * d2_j;
        const double d5_j = -omega_orb_sq * d3_j;
        // Integrate to get {v, d}
        const double v_circ_i = -d3_val_i / omega_orb_sq;
        const double v_circ_j = -d3_j / omega_orb_sq;
        const double d1_diff  = d1_val_i - v_circ_i;
        const double d1_j     = v_circ_j + d1_diff;
        const double d_circ_j = -d2_j / omega_orb_sq;
        const double d_circ_i = -d2_val_i / omega_orb_sq;
        const double d0_j   = d_circ_j + (d0_val_i - d_circ_i) + (d1_diff * dt);
        leaves_tree[lo + 0] = d5_j;
        leaves_tree[lo + 2] = d4_j;
        leaves_tree[lo + 4] = d3_j;
        leaves_tree[lo + 6] = d2_j;
        leaves_tree[lo + 8] = d1_j;
        leaves_tree[lo + 10] = d0_j;
    }

    // Process error leaves
    double d5_err_j, d4_err_j, d3_err_j, d2_err_j, d1_err_j;
    if constexpr (UseConservativeTile) {
        d5_err_j = d5_err_i;
        d4_err_j =
            std::sqrt((d4_err_i * d4_err_i) + (d5_err_i * d5_err_i * dt2));
        d3_err_j =
            std::sqrt((d3_err_i * d3_err_i) + (d4_err_i * d4_err_i * dt2) +
                      (d5_err_i * d5_err_i * half_dt2 * half_dt2));
        d2_err_j =
            std::sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt2) +
                      (d4_err_i * d4_err_i * half_dt2 * half_dt2) +
                      (d5_err_i * d5_err_i * sixth_dt3 * sixth_dt3));
        d1_err_j = std::sqrt(
            (d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt2) +
            (d3_err_i * d3_err_i * half_dt2 * half_dt2) +
            (d4_err_i * d4_err_i * sixth_dt3 * sixth_dt3) +
            (d5_err_i * d5_err_i * twenty_fourth_dt4 * twenty_fourth_dt4));
    } else {
        d5_err_j = d5_err_i;
        d4_err_j = d4_err_i;
        d3_err_j = d3_err_i;
        d2_err_j = d2_err_i;
        d1_err_j = d1_err_i;
    }
    leaves_tree[lo + 1]  = d5_err_j;
    leaves_tree[lo + 3]  = d4_err_j;
    leaves_tree[lo + 5]  = d3_err_j;
    leaves_tree[lo + 7]  = d2_err_j;
    leaves_tree[lo + 9]  = d1_err_j;
    leaves_tree[lo + 11] = d0_err_i;
}

} // namespace

SizeType
circ_taylor_branch_batch_cuda(cuda::std::span<const double> leaves_tree,
                              cuda::std::span<double> leaves_branch,
                              cuda::std::span<uint32_t> leaves_origins,
                              cuda::std::span<uint8_t> validation_mask,
                              std::pair<double, double> coord_cur,
                              SizeType nbins,
                              double eta,
                              cuda::std::span<const ParamLimit> param_limits,
                              SizeType branch_max,
                              SizeType n_leaves,
                              double minimum_snap_cells,
                              memory::BranchingWorkspaceCUDAView branch_ws,
                              memory::CUBScratchArena& scratch_ws,
                              cudaStream_t stream) {
    constexpr SizeType kLeavesStride = 14;

    // Check if crackle branching is needed (reuse scratch_ws.d_reduce_out)
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(scratch_ws.d_reduce_out, 0, sizeof(uint32_t), stream),
        "cudaMemsetAsync failed");

    // ---- Kernel 1: analyze + branch enumeration ----
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    kernel_analyze_and_branch_circular<<<grid_dim, block_dim, 0, stream>>>(
        leaves_tree.data(), n_leaves, coord_cur.second,
        static_cast<double>(nbins), eta, param_limits.data(), branch_max,
        branch_ws, scratch_ws);
    cuda_utils::check_last_cuda_error("Kernel 1 launch failed");

    // compute output size and offsets (leaf_output_offset)
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(
            scratch_ws.cub_temp_storage, scratch_ws.cub_temp_bytes,
            branch_ws.leaf_branch_count, branch_ws.leaf_output_offset, n_leaves,
            stream),
        "cub::DeviceScan::ExclusiveSum failed");
    uint32_t last_offset, last_count, has_crackle;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_offset,
                        branch_ws.leaf_output_offset + (n_leaves - 1),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_count,
                        branch_ws.leaf_branch_count + (n_leaves - 1),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&has_crackle, scratch_ws.d_reduce_out, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    // Unavoidable sync
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize branch kernel failed");
    const SizeType n_leaves_branched = last_offset + last_count;
    if (n_leaves_branched == 0) {
        return n_leaves_branched;
    }

    // Set validation mask for produced outputs
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(validation_mask.data(), 1,
                        n_leaves_branched * sizeof(uint8_t), stream),
        "cudaMemsetAsync failed");

    if (n_leaves_branched == n_leaves) {
        // FAST PATH: no branching
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_branch.data(), leaves_tree.data(),
                            n_leaves * kLeavesStride * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync failed");
        thrust::sequence(thrust::cuda::par.on(stream), leaves_origins.data(),
                         leaves_origins.data() + n_leaves,
                         static_cast<uint32_t>(0));

        return n_leaves_branched;
    }

    // ---- Kernel 2: materialize ----
    const auto blocks_per_grid_out =
        (n_leaves_branched + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 grid_dim_out(blocks_per_grid_out);
    cuda_utils::check_kernel_launch_params(grid_dim_out, block_dim);
    kernel_materialize_branches_circular<kThreadsPerBlock>
        <<<grid_dim_out, block_dim, 0, stream>>>(
            leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
            n_leaves, n_leaves_branched, branch_max, branch_ws);
    cuda_utils::check_last_cuda_error("Kernel 2 launch failed");

    // ---- Kernel 3: expand crackle holes ----
    if (has_crackle == 0) {
        return n_leaves_branched;
    }
    // Init out_count = n_leaves_branched
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(scratch_ws.d_reduce_out, &n_leaves_branched,
                        sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync failed");

    const SizeType blocks_per_grid_3 =
        (n_leaves_branched + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 grid_dim_3(blocks_per_grid_3);
    cuda_utils::check_kernel_launch_params(grid_dim_3, block_dim);
    kernel_expand_crackle_holes<<<grid_dim_3, block_dim, 0, stream>>>(
        leaves_branch.data(), leaves_origins.data(), n_leaves_branched,
        coord_cur.second, static_cast<double>(nbins), eta, param_limits.data(),
        minimum_snap_cells, scratch_ws.d_reduce_out);
    cuda_utils::check_last_cuda_error("Kernel 3 launch failed");

    // Sync + read back final count
    uint32_t final_count;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&final_count, scratch_ws.d_reduce_out, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    // Unavoidable sync
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize kernel 3 failed");
    return final_count;
}

SizeType
circ_taylor_validate_batch_cuda(cuda::std::span<const double> leaves_branch,
                                cuda::std::span<uint8_t> validation_mask,
                                SizeType n_leaves,
                                double p_orb_min,
                                double x_mass_const,
                                double minimum_snap_cells,
                                memory::CUBScratchArena& scratch_ws,
                                cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_validate_branches_circular<<<grid_dim, block_dim, 0, stream>>>(
        leaves_branch.data(), validation_mask.data(), n_leaves, p_orb_min,
        x_mass_const, minimum_snap_cells);
    cuda_utils::check_last_cuda_error(
        "kernel_validate_branches_circular launch failed");

    // Count number of passing profiles
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Sum(
            scratch_ws.cub_temp_storage, scratch_ws.cub_temp_bytes,
            validation_mask.data(), scratch_ws.d_reduce_out, n_leaves, stream),
        "cub::DeviceReduce::Sum failed");

    // Copy result back
    uint32_t n_leaves_passing = 0;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&n_leaves_passing, scratch_ws.d_reduce_out,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");

    // You already sync elsewhere usually, but if needed:
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "stream sync failed");
    return n_leaves_passing;
}

void circ_taylor_resolve_batch_cuda(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    cuda::std::span<const ParamLimit> param_limits,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    double minimum_snap_cells,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    kernel_circ_taylor_resolve_batch<<<grid_dim, block_dim, 0, stream>>>(
        leaves_branch.data(), validation_mask.data(), param_indices.data(),
        phase_shift.data(), param_limits.data(), coord_add, coord_cur,
        coord_init, n_accel_init, n_freq_init, nbins, n_leaves,
        minimum_snap_cells);
    cuda_utils::check_last_cuda_error(
        "Circular Taylor resolve kernel launch failed");
    // No need to sync, the next kernel will do it
}

void circ_taylor_transform_batch_cuda(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    bool use_conservative_tile,
    double minimum_snap_cells,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;
    const SizeType blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    if (use_conservative_tile) {
        kernel_circ_taylor_transform_batch<true>
            <<<grid_dim, block_dim, 0, stream>>>(
                leaves_tree.data(), validation_mask.data(), n_leaves,
                coord_next, coord_cur, minimum_snap_cells);
    } else {
        kernel_circ_taylor_transform_batch<false>
            <<<grid_dim, block_dim, 0, stream>>>(
                leaves_tree.data(), validation_mask.data(), n_leaves,
                coord_next, coord_cur, minimum_snap_cells);
    }
    cuda_utils::check_last_cuda_error(
        "Circular Taylor transform kernel launch failed");
    // No need to sync, the next kernel will do it
}

} // namespace loki::core