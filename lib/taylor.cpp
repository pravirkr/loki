#include "loki/core/taylor.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/cartesian.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"
#include "loki/utils.hpp"

namespace loki::core {

namespace {

template <int LATTER>
void ffa_taylor_resolve_accel_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins) {
    constexpr SizeType kParams = 2;
    error_check::check_equal(param_grid_count_cur.size(), kParams,
                             "param_grid_count_cur should have 2 elements");
    error_check::check_equal(param_grid_count_prev.size(), kParams,
                             "param_grid_count_prev should have 2 elements");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits should have 2 elements");

    const SizeType n_accel_cur  = param_grid_count_cur[0];
    const SizeType n_freq_cur   = param_grid_count_cur[1];
    const SizeType n_accel_prev = param_grid_count_prev[0];
    const SizeType n_freq_prev  = param_grid_count_prev[1];
    const ParamLimit& lim_accel = param_limits[0];
    const ParamLimit& lim_freq  = param_limits[1];
    const auto ncoords          = n_accel_cur * n_freq_cur;
    error_check::check_equal(coords.size(), ncoords, "coords size mismatch");

    const double tsegment =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));
    const auto delta_t = (static_cast<double>(LATTER) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto half_delta_t_sq = 0.5 * delta_t * delta_t;

    for (SizeType accel_idx = 0; accel_idx < n_accel_cur; ++accel_idx) {
        // Generate parameters on the fly
        const auto a_cur =
            psr_utils::get_param_val_at_idx(lim_accel, n_accel_cur, accel_idx);
        const auto a_new = a_cur;
        const auto v_new = a_cur * delta_t;
        const auto d_new = a_cur * half_delta_t_sq;
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_prev);
        const auto coord_a_offset = accel_idx * n_freq_cur;
        const auto idx_a_offset   = idx_a * n_freq_prev;

        for (SizeType freq_idx = 0; freq_idx < n_freq_cur; ++freq_idx) {
            const auto f_cur =
                psr_utils::get_param_val_at_idx(lim_freq, n_freq_cur, freq_idx);
            const auto f_new     = f_cur * (1.0 - v_new * utils::kInvCval);
            const auto delay_rel = d_new * utils::kInvCval;

            const auto relative_phase =
                psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);
            const auto idx_f = psr_utils::get_nearest_idx_analytical(
                f_new, lim_freq, n_freq_prev);

            const auto final_idx = static_cast<uint32_t>(idx_a_offset + idx_f);
            const auto coord_idx = coord_a_offset + freq_idx;
            if constexpr (LATTER == 0) {
                coords[coord_idx].i_tail     = final_idx;
                coords[coord_idx].shift_tail = relative_phase;
            } else {
                coords[coord_idx].i_head     = final_idx;
                coords[coord_idx].shift_head = relative_phase;
            }
        }
    }
}

template <int LATTER>
void ffa_taylor_resolve_jerk_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins,
    SizeType param_stride) {
    constexpr SizeType kParams = 3;
    error_check::check_equal(param_grid_count_cur.size(), kParams,
                             "param_grid_count_cur should have 3 elements");
    error_check::check_equal(param_grid_count_prev.size(), kParams,
                             "param_grid_count_prev should have 3 elements");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits should have 3 elements");
    error_check::check_greater_equal(
        param_stride, kParams,
        "param_stride should be greater than or equal to 3");

    const SizeType po           = param_stride - kParams;
    const SizeType n_jerk_cur   = param_grid_count_cur[po + 0];
    const SizeType n_accel_cur  = param_grid_count_cur[po + 1];
    const SizeType n_freq_cur   = param_grid_count_cur[po + 2];
    const SizeType n_jerk_prev  = param_grid_count_prev[po + 0];
    const SizeType n_accel_prev = param_grid_count_prev[po + 1];
    const SizeType n_freq_prev  = param_grid_count_prev[po + 2];
    const ParamLimit& lim_jerk  = param_limits[po + 0];
    const ParamLimit& lim_accel = param_limits[po + 1];
    const ParamLimit& lim_freq  = param_limits[po + 2];
    const auto ncoords          = n_jerk_cur * n_accel_cur * n_freq_cur;
    error_check::check_equal(coords.size(), ncoords, "coords size mismatch");

    const double tsegment =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));
    const auto delta_t = (static_cast<double>(LATTER) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_sq          = delta_t * delta_t;
    const auto delta_t_cubed       = delta_t_sq * delta_t;
    const auto half_delta_t_sq     = 0.5 * delta_t_sq;
    const auto sixth_delta_t_cubed = delta_t_cubed / 6.0;

    for (SizeType jerk_idx = 0; jerk_idx < n_jerk_cur; ++jerk_idx) {
        const auto j_cur =
            psr_utils::get_param_val_at_idx(lim_jerk, n_jerk_cur, jerk_idx);
        const auto j_new = j_cur; // No transformation needed

        // Pre-compute jerk-related terms for this jerk value
        const auto j_delta_t             = j_cur * delta_t;
        const auto half_j_delta_t_sq     = 0.5 * j_cur * delta_t_sq;
        const auto j_sixth_delta_t_cubed = j_cur * sixth_delta_t_cubed;

        const auto idx_j =
            psr_utils::get_nearest_idx_analytical(j_new, lim_jerk, n_jerk_prev);
        const auto coord_j_offset = jerk_idx * n_accel_cur * n_freq_cur;
        const auto idx_j_offset   = idx_j * n_accel_prev * n_freq_prev;

        for (SizeType accel_idx = 0; accel_idx < n_accel_cur; ++accel_idx) {
            const auto a_cur = psr_utils::get_param_val_at_idx(
                lim_accel, n_accel_cur, accel_idx);
            const auto a_new = a_cur + j_delta_t;
            const auto v_new = (a_cur * delta_t) + half_j_delta_t_sq;
            const auto d_new =
                (a_cur * half_delta_t_sq) + j_sixth_delta_t_cubed;

            // Find accel index once per (jerk_idx, accel_idx) pair
            const auto idx_a = psr_utils::get_nearest_idx_analytical(
                a_new, lim_accel, n_accel_prev);
            const auto coord_a_offset =
                coord_j_offset + (accel_idx * n_freq_cur);
            const auto idx_a_offset = idx_j_offset + (idx_a * n_freq_prev);

            for (SizeType freq_idx = 0; freq_idx < n_freq_cur; ++freq_idx) {
                const auto f_cur = psr_utils::get_param_val_at_idx(
                    lim_freq, n_freq_cur, freq_idx);
                const auto f_new     = f_cur * (1.0 - v_new * utils::kInvCval);
                const auto delay_rel = d_new * utils::kInvCval;

                const auto relative_phase =
                    psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);
                const auto idx_f = psr_utils::get_nearest_idx_analytical(
                    f_new, lim_freq, n_freq_prev);

                const auto final_idx =
                    static_cast<uint32_t>(idx_a_offset + idx_f);
                const auto coord_idx = coord_a_offset + freq_idx;
                if constexpr (LATTER == 0) {
                    coords[coord_idx].i_tail     = final_idx;
                    coords[coord_idx].shift_tail = relative_phase;
                } else {
                    coords[coord_idx].i_head     = final_idx;
                    coords[coord_idx].shift_head = relative_phase;
                }
            }
        }
    }
}

SizeType
poly_taylor_branch_accel_batch(std::span<const double> leaves_tree,
                               std::span<double> leaves_branch,
                               std::span<SizeType> leaves_origins,
                               std::pair<double, double> coord_cur,
                               SizeType nbins,
                               double eta,
                               std::span<const ParamLimit> param_limits,
                               SizeType branch_max,
                               SizeType n_leaves,
                               utils::BranchingWorkspaceView ws) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");
    const auto [_, dt]   = coord_cur; // t_obs_minus_t_ref
    const double dt2     = dt * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ dparam_new_ptr        = dparam_new.data();
    double* __restrict__ shift_bins_ptr        = shift_bins.data();

    // --- Loop 1: step + shift (vectorizable) ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto f0         = leaves_tree_ptr[leaf_offset + 6];

        // Compute steps
        const auto dfactor    = utils::kCval / f0;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        dparam_new_ptr[flat_base + 0] = d2_sig_new;
        dparam_new_ptr[flat_base + 1] = d1_sig_new;

        // Compute shift bins
        shift_bins_ptr[flat_base + 0] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[flat_base + 1] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // --- Early Exit: Check if any leaf needs branching ---
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - utils::kEps)) {
            any_branching = true;
            break;
        }
    }
    // Fast path: no branching at all
    if (!any_branching) {
        std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                    n_leaves * kLeavesStride * sizeof(double));
        for (SizeType i = 0; i < n_leaves; ++i) {
            leaves_origins_ptr[i] = i;
        }
        return n_leaves;
    }

    // --- Loop 2: branching ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d2_cur     = leaves_tree_ptr[leaf_offset + 0];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d1_cur     = leaves_tree_ptr[leaf_offset + 2];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto f0         = leaves_tree_ptr[leaf_offset + 6];
        const auto d2_sig_new = dparam_new_ptr[flat_base + 0];
        const auto d1_sig_new = dparam_new_ptr[flat_base + 1];

        // Branch d2-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d2_cur, d2_sig_cur, d2_sig_new, param_limits[0].min,
            param_limits[0].max, eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        const double d1_min = (1 - param_limits[1].max / f0) * utils::kCval;
        const double d1_max = (1 - param_limits[1].min / f0) * utils::kCval;
        psr_utils::branch_one_param_padded(
            1, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta,
            shift_bins_ptr, ws.scratch_params, ws.scratch_dparams,
            ws.scratch_counts, flat_base, branch_max);
    }

    // --- Loop 3: Fill leaves_origins ---
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset   = i * kLeavesStride;
        const SizeType flat_base     = i * kParams;
        const SizeType n_d2_branches = ws.scratch_counts[flat_base + 0];
        const SizeType n_d1_branches = ws.scratch_counts[flat_base + 1];
        const SizeType d2_offset     = (flat_base + 0) * branch_max;
        const SizeType d1_offset     = (flat_base + 1) * branch_max;

        for (SizeType a = 0; a < n_d2_branches; ++a) {
            for (SizeType b = 0; b < n_d1_branches; ++b) {
                const SizeType branch_offset = out_leaves * kLeavesStride;
                leaves_branch_ptr[branch_offset + 0] =
                    ws.scratch_params[d2_offset + a];
                leaves_branch_ptr[branch_offset + 1] =
                    ws.scratch_dparams[flat_base + 0];
                leaves_branch_ptr[branch_offset + 2] =
                    ws.scratch_params[d1_offset + b];
                leaves_branch_ptr[branch_offset + 3] =
                    ws.scratch_dparams[flat_base + 1];
                // Fill d0 and f0 directly from leaves_tree
                std::memcpy(leaves_branch_ptr + branch_offset + 4,
                            leaves_tree_ptr + leaf_offset + 4,
                            4 * sizeof(double));

                leaves_origins_ptr[out_leaves] = i;
                ++out_leaves;
            }
        }
    }

    return out_leaves;
}

SizeType poly_taylor_branch_jerk_batch(std::span<const double> leaves_tree,
                                       std::span<double> leaves_branch,
                                       std::span<SizeType> leaves_origins,
                                       std::pair<double, double> coord_cur,
                                       SizeType nbins,
                                       double eta,
                                       std::span<const ParamLimit> param_limits,
                                       SizeType branch_max,
                                       SizeType n_leaves,
                                       utils::BranchingWorkspaceView ws) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");
    const auto [_, dt]   = coord_cur; // t_obs_minus_t_ref
    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ dparam_new_ptr        = dparam_new.data();
    double* __restrict__ shift_bins_ptr        = shift_bins.data();

    // --- Loop 1: step + shift (vectorizable) ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d3_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 5];
        const auto f0         = leaves_tree_ptr[leaf_offset + 8];

        const auto dfactor    = utils::kCval / f0;
        const auto d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        dparam_new_ptr[flat_base + 0] = d3_sig_new;
        dparam_new_ptr[flat_base + 1] = d2_sig_new;
        dparam_new_ptr[flat_base + 2] = d1_sig_new;

        shift_bins_ptr[flat_base + 0] =
            (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
        shift_bins_ptr[flat_base + 1] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[flat_base + 2] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // --- Early Exit: Check if any leaf needs branching ---
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - utils::kEps)) {
            any_branching = true;
            break;
        }
    }
    // Fast path: no branching at all
    if (!any_branching) {
        std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                    n_leaves * kLeavesStride * sizeof(double));
        for (SizeType i = 0; i < n_leaves; ++i) {
            leaves_origins_ptr[i] = i;
        }
        return n_leaves;
    }

    // --- Loop 2: branching ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d3_cur     = leaves_tree_ptr[leaf_offset + 0];
        const auto d3_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d2_cur     = leaves_tree_ptr[leaf_offset + 2];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto d1_cur     = leaves_tree_ptr[leaf_offset + 4];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 5];
        const auto f0         = leaves_tree_ptr[leaf_offset + 8];
        const auto d3_sig_new = dparam_new_ptr[flat_base + 0];
        const auto d2_sig_new = dparam_new_ptr[flat_base + 1];
        const auto d1_sig_new = dparam_new_ptr[flat_base + 2];

        // Branch d3-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d3_cur, d3_sig_cur, d3_sig_new, param_limits[0].min,
            param_limits[0].max, eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        psr_utils::branch_one_param_padded(
            1, d2_cur, d2_sig_cur, d2_sig_new, param_limits[1].min,
            param_limits[1].max, eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        const double d1_min = (1 - param_limits[2].max / f0) * utils::kCval;
        const double d1_max = (1 - param_limits[2].min / f0) * utils::kCval;
        psr_utils::branch_one_param_padded(
            2, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta,
            shift_bins_ptr, ws.scratch_params, ws.scratch_dparams,
            ws.scratch_counts, flat_base, branch_max);
    }

    // --- Loop 3: Fill leaves_origins (3D Cartesian product) ---
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset   = i * kLeavesStride;
        const SizeType flat_base     = i * kParams;
        const SizeType n_d3_branches = ws.scratch_counts[flat_base + 0];
        const SizeType n_d2_branches = ws.scratch_counts[flat_base + 1];
        const SizeType n_d1_branches = ws.scratch_counts[flat_base + 2];
        const SizeType d3_offset     = (flat_base + 0) * branch_max;
        const SizeType d2_offset     = (flat_base + 1) * branch_max;
        const SizeType d1_offset     = (flat_base + 2) * branch_max;

        for (SizeType a = 0; a < n_d3_branches; ++a) {
            for (SizeType b = 0; b < n_d2_branches; ++b) {
                for (SizeType c = 0; c < n_d1_branches; ++c) {
                    const SizeType branch_offset = out_leaves * kLeavesStride;
                    leaves_branch_ptr[branch_offset + 0] =
                        ws.scratch_params[d3_offset + a];
                    leaves_branch_ptr[branch_offset + 1] =
                        ws.scratch_dparams[flat_base + 0];
                    leaves_branch_ptr[branch_offset + 2] =
                        ws.scratch_params[d2_offset + b];
                    leaves_branch_ptr[branch_offset + 3] =
                        ws.scratch_dparams[flat_base + 1];
                    leaves_branch_ptr[branch_offset + 4] =
                        ws.scratch_params[d1_offset + c];
                    leaves_branch_ptr[branch_offset + 5] =
                        ws.scratch_dparams[flat_base + 2];
                    // Fill d0 and f0 directly from leaves_tree
                    std::memcpy(leaves_branch_ptr + branch_offset + 6,
                                leaves_tree_ptr + leaf_offset + 6,
                                4 * sizeof(double));

                    leaves_origins_ptr[out_leaves] = i;
                    ++out_leaves;
                }
            }
        }
    }

    return out_leaves;
}

SizeType poly_taylor_branch_snap_batch(std::span<const double> leaves_tree,
                                       std::span<double> leaves_branch,
                                       std::span<SizeType> leaves_origins,
                                       std::pair<double, double> coord_cur,
                                       SizeType nbins,
                                       double eta,
                                       std::span<const ParamLimit> param_limits,
                                       SizeType branch_max,
                                       SizeType n_leaves,
                                       utils::BranchingWorkspaceView ws) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");
    const auto [_, dt]   = coord_cur; // t_obs_minus_t_ref
    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double dt4     = dt2 * dt2;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const double inv_dt4 = inv_dt2 * inv_dt2;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ dparam_new_ptr        = dparam_new.data();
    double* __restrict__ shift_bins_ptr        = shift_bins.data();

    // --- Loop 1: step + shift (vectorizable) ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d4_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d3_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 5];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 7];
        const auto f0         = leaves_tree_ptr[leaf_offset + 10];

        const auto dfactor    = utils::kCval / f0;
        const auto d4_sig_new = dphi * dfactor * 192.0 * inv_dt4;
        const auto d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        dparam_new_ptr[flat_base + 0] = d4_sig_new;
        dparam_new_ptr[flat_base + 1] = d3_sig_new;
        dparam_new_ptr[flat_base + 2] = d2_sig_new;
        dparam_new_ptr[flat_base + 3] = d1_sig_new;

        shift_bins_ptr[flat_base + 0] =
            (d4_sig_cur - d4_sig_new) * dt4 * nbins_d / (192.0 * dfactor);
        shift_bins_ptr[flat_base + 1] =
            (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
        shift_bins_ptr[flat_base + 2] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[flat_base + 3] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // --- Early Exit: Check if any leaf needs branching ---
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - utils::kEps)) {
            any_branching = true;
            break;
        }
    }
    // Fast path: no branching at all
    if (!any_branching) {
        std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                    n_leaves * kLeavesStride * sizeof(double));
        for (SizeType i = 0; i < n_leaves; ++i) {
            leaves_origins_ptr[i] = i;
        }
        return n_leaves;
    }

    // --- Loop 2: branching ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d4_cur     = leaves_tree_ptr[leaf_offset + 0];
        const auto d4_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d3_cur     = leaves_tree_ptr[leaf_offset + 2];
        const auto d3_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto d2_cur     = leaves_tree_ptr[leaf_offset + 4];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 5];
        const auto d1_cur     = leaves_tree_ptr[leaf_offset + 6];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 7];
        const auto f0         = leaves_tree_ptr[leaf_offset + 10];
        const auto d4_sig_new = dparam_new_ptr[flat_base + 0];
        const auto d3_sig_new = dparam_new_ptr[flat_base + 1];
        const auto d2_sig_new = dparam_new_ptr[flat_base + 2];
        const auto d1_sig_new = dparam_new_ptr[flat_base + 3];

        // Branch d4-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d4_cur, d4_sig_cur, d4_sig_new, param_limits[0].min,
            param_limits[0].max, eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        psr_utils::branch_one_param_padded(
            1, d3_cur, d3_sig_cur, d3_sig_new, param_limits[1].min,
            param_limits[1].max, eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        psr_utils::branch_one_param_padded(
            2, d2_cur, d2_sig_cur, d2_sig_new, param_limits[2].min,
            param_limits[2].max, eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        const double d1_min = (1 - param_limits[3].max / f0) * utils::kCval;
        const double d1_max = (1 - param_limits[3].min / f0) * utils::kCval;
        psr_utils::branch_one_param_padded(
            3, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta,
            shift_bins_ptr, ws.scratch_params, ws.scratch_dparams,
            ws.scratch_counts, flat_base, branch_max);
    }

    // --- Loop 3: Fill leaves_origins (4D Cartesian product) ---
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset   = i * kLeavesStride;
        const SizeType flat_base     = i * kParams;
        const SizeType n_d4_branches = ws.scratch_counts[flat_base + 0];
        const SizeType n_d3_branches = ws.scratch_counts[flat_base + 1];
        const SizeType n_d2_branches = ws.scratch_counts[flat_base + 2];
        const SizeType n_d1_branches = ws.scratch_counts[flat_base + 3];
        const SizeType d4_offset     = (flat_base + 0) * branch_max;
        const SizeType d3_offset     = (flat_base + 1) * branch_max;
        const SizeType d2_offset     = (flat_base + 2) * branch_max;
        const SizeType d1_offset     = (flat_base + 3) * branch_max;

        for (SizeType a = 0; a < n_d4_branches; ++a) {
            for (SizeType b = 0; b < n_d3_branches; ++b) {
                for (SizeType c = 0; c < n_d2_branches; ++c) {
                    for (SizeType d = 0; d < n_d1_branches; ++d) {
                        const SizeType branch_offset =
                            out_leaves * kLeavesStride;
                        leaves_branch_ptr[branch_offset + 0] =
                            ws.scratch_params[d4_offset + a];
                        leaves_branch_ptr[branch_offset + 1] =
                            ws.scratch_dparams[flat_base + 0];
                        leaves_branch_ptr[branch_offset + 2] =
                            ws.scratch_params[d3_offset + b];
                        leaves_branch_ptr[branch_offset + 3] =
                            ws.scratch_dparams[flat_base + 1];
                        leaves_branch_ptr[branch_offset + 4] =
                            ws.scratch_params[d2_offset + c];
                        leaves_branch_ptr[branch_offset + 5] =
                            ws.scratch_dparams[flat_base + 2];
                        leaves_branch_ptr[branch_offset + 6] =
                            ws.scratch_params[d1_offset + d];
                        leaves_branch_ptr[branch_offset + 7] =
                            ws.scratch_dparams[flat_base + 3];
                        // Fill d0 and f0 directly from leaves_tree
                        std::memcpy(leaves_branch_ptr + branch_offset + 8,
                                    leaves_tree_ptr + leaf_offset + 8,
                                    4 * sizeof(double));

                        leaves_origins_ptr[out_leaves] = i;
                        ++out_leaves;
                    }
                }
            }
        }
    }

    return out_leaves;
}

void poly_taylor_resolve_accel_batch(std::span<const double> leaves_tree,
                                     std::span<SizeType> param_indices,
                                     std::span<float> phase_shift,
                                     std::span<const ParamLimit> param_limits,
                                     std::pair<double, double> coord_add,
                                     std::pair<double, double> coord_cur,
                                     std::pair<double, double> coord_init,
                                     SizeType n_accel_init,
                                     SizeType n_freq_init,
                                     SizeType nbins,
                                     SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[0];
    const auto& lim_freq  = param_limits[1];

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_add  = t0_add - t0_cur;
    const auto delta_t_init = t0_init - t0_cur;
    const auto delta_t      = delta_t_add - delta_t_init;
    const auto half_delta_t_sq =
        0.5 * (delta_t_add * delta_t_add - delta_t_init * delta_t_init);

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto a_t_cur     = leaves_tree[leaf_offset + (0 * kParamStride)];
        const auto v_t_cur     = leaves_tree[leaf_offset + (1 * kParamStride)];
        const auto f0          = leaves_tree[leaf_offset + (3 * kParamStride)];
        const auto a_new       = a_t_cur;
        const auto delta_v_new = a_t_cur * delta_t;
        const auto delta_d_new =
            (v_t_cur * delta_t) + (a_t_cur * half_delta_t_sq);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - delta_v_new * utils::kInvCval);
        const auto delay_rel = delta_d_new * utils::kInvCval;

        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
        phase_shift[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);
    }
}

void poly_taylor_resolve_jerk_batch(std::span<const double> leaves_tree,
                                    std::span<SizeType> param_indices,
                                    std::span<float> phase_shift,
                                    std::span<const ParamLimit> param_limits,
                                    std::pair<double, double> coord_add,
                                    std::pair<double, double> coord_cur,
                                    std::pair<double, double> coord_init,
                                    SizeType n_accel_init,
                                    SizeType n_freq_init,
                                    SizeType nbins,
                                    SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[1];
    const auto& lim_freq  = param_limits[2];

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_add          = t0_add - t0_cur;
    const auto delta_t_init         = t0_init - t0_cur;
    const auto half_delta_t_add_sq  = 0.5 * delta_t_add * delta_t_add;
    const auto half_delta_t_init_sq = 0.5 * delta_t_init * delta_t_init;
    const auto delta_t              = delta_t_add - delta_t_init;
    const auto half_delta_t_sq     = half_delta_t_add_sq - half_delta_t_init_sq;
    const auto sixth_delta_t_cubed = (half_delta_t_add_sq * delta_t_add -
                                      half_delta_t_init_sq * delta_t_init) /
                                     3.0;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto j_t_cur = leaves_tree[leaf_offset + (0 * kParamStride)];
        const auto a_t_cur = leaves_tree[leaf_offset + (1 * kParamStride)];
        const auto v_t_cur = leaves_tree[leaf_offset + (2 * kParamStride)];
        const auto f0      = leaves_tree[leaf_offset + (4 * kParamStride)];
        const auto a_new   = a_t_cur + (j_t_cur * delta_t_add);
        const auto delta_v_new =
            (a_t_cur * delta_t) + (j_t_cur * half_delta_t_sq);
        const auto delta_d_new = (v_t_cur * delta_t) +
                                 (a_t_cur * half_delta_t_sq) +
                                 (j_t_cur * sixth_delta_t_cubed);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - delta_v_new * utils::kInvCval);
        const auto delay_rel = delta_d_new * utils::kInvCval;

        // Calculate relative phase
        phase_shift[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);
        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
    }
}

void poly_taylor_resolve_snap_batch(std::span<const double> leaves_tree,
                                    std::span<SizeType> param_indices,
                                    std::span<float> phase_shift,
                                    std::span<const ParamLimit> param_limits,
                                    std::pair<double, double> coord_add,
                                    std::pair<double, double> coord_cur,
                                    std::pair<double, double> coord_init,
                                    SizeType n_accel_init,
                                    SizeType n_freq_init,
                                    SizeType nbins,
                                    SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[2];
    const auto& lim_freq  = param_limits[3];

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_add          = t0_add - t0_cur;
    const auto delta_t_init         = t0_init - t0_cur;
    const auto half_delta_t_add_sq  = 0.5 * delta_t_add * delta_t_add;
    const auto half_delta_t_init_sq = 0.5 * delta_t_init * delta_t_init;
    const auto delta_t              = delta_t_add - delta_t_init;
    const auto half_delta_t_sq     = half_delta_t_add_sq - half_delta_t_init_sq;
    const auto sixth_delta_t_cubed = (half_delta_t_add_sq * delta_t_add -
                                      half_delta_t_init_sq * delta_t_init) /
                                     3.0;
    const auto twenty_fourth_delta_t_fourth =
        (half_delta_t_add_sq * half_delta_t_add_sq -
         half_delta_t_init_sq * half_delta_t_init_sq) /
        6.0;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto s_t_cur = leaves_tree[leaf_offset + (0 * kParamStride)];
        const auto j_t_cur = leaves_tree[leaf_offset + (1 * kParamStride)];
        const auto a_t_cur = leaves_tree[leaf_offset + (2 * kParamStride)];
        const auto v_t_cur = leaves_tree[leaf_offset + (3 * kParamStride)];
        const auto f0      = leaves_tree[leaf_offset + (5 * kParamStride)];
        const auto a_new =
            a_t_cur + (j_t_cur * delta_t_add) + (s_t_cur * half_delta_t_add_sq);
        const auto delta_v_new = (a_t_cur * delta_t) +
                                 (j_t_cur * half_delta_t_sq) +
                                 (s_t_cur * sixth_delta_t_cubed);
        const auto delta_d_new = (v_t_cur * delta_t) +
                                 (a_t_cur * half_delta_t_sq) +
                                 (j_t_cur * sixth_delta_t_cubed) +
                                 (s_t_cur * twenty_fourth_delta_t_fourth);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - delta_v_new * utils::kInvCval);
        const auto delay_rel = delta_d_new * utils::kInvCval;

        // Calculate relative phase
        phase_shift[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);
        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
    }
}

template <bool UseConservativeTile>
void poly_taylor_transform_accel_batch(std::span<double> leaves_tree,
                                       std::pair<double, double> coord_next,
                                       std::pair<double, double> coord_cur,
                                       SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto delta_t         = t0_next - t0_cur;
    const auto half_delta_t_sq = 0.5 * (delta_t * delta_t);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto leaf_offset = i * kLeavesStride;

        const auto a_val_i = leaves_tree[leaf_offset + 0];
        const auto a_err_i = leaves_tree[leaf_offset + 1];
        const auto v_val_i = leaves_tree[leaf_offset + 2];
        const auto v_err_i = leaves_tree[leaf_offset + 3];
        const auto d_val_i = leaves_tree[leaf_offset + 4];
        const auto d_err_i = leaves_tree[leaf_offset + 5];

        const auto a_val_j = a_val_i;
        const auto v_val_j = v_val_i + (a_val_i * delta_t);
        const auto d_val_j =
            d_val_i + (v_val_i * delta_t) + (a_val_i * half_delta_t_sq);

        double a_err_j, v_err_j;
        if constexpr (UseConservativeTile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            a_err_j = a_err_i;
            v_err_j = std::sqrt((v_err_i * v_err_i) +
                                (a_err_i * a_err_i * delta_t * delta_t));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            a_err_j = a_err_i;
            v_err_j = v_err_i;
        }

        // Write back transformed values
        leaves_tree[leaf_offset + 0] = a_val_j;
        leaves_tree[leaf_offset + 1] = a_err_j;
        leaves_tree[leaf_offset + 2] = v_val_j;
        leaves_tree[leaf_offset + 3] = v_err_j;
        leaves_tree[leaf_offset + 4] = d_val_j;
        leaves_tree[leaf_offset + 5] = d_err_i;
    }
}

template <bool UseConservativeTile>
void poly_taylor_transform_jerk_batch(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto delta_t             = t0_next - t0_cur;
    const auto delta_t_sq          = delta_t * delta_t;
    const auto half_delta_t_sq     = 0.5 * (delta_t * delta_t);
    const auto sixth_delta_t_cubed = delta_t_sq * delta_t / 6.0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto leaf_offset = i * kLeavesStride;

        const auto j_val_i = leaves_tree[leaf_offset + 0];
        const auto j_err_i = leaves_tree[leaf_offset + 1];
        const auto a_val_i = leaves_tree[leaf_offset + 2];
        const auto a_err_i = leaves_tree[leaf_offset + 3];
        const auto v_val_i = leaves_tree[leaf_offset + 4];
        const auto v_err_i = leaves_tree[leaf_offset + 5];
        const auto d_val_i = leaves_tree[leaf_offset + 6];
        const auto d_err_i = leaves_tree[leaf_offset + 7];

        const auto j_val_j = j_val_i;
        const auto a_val_j = a_val_i + (j_val_i * delta_t);
        const auto v_val_j =
            v_val_i + (a_val_i * delta_t) + (j_val_i * half_delta_t_sq);
        const auto d_val_j = d_val_i + (v_val_i * delta_t) +
                             (a_val_i * half_delta_t_sq) +
                             (j_val_i * sixth_delta_t_cubed);

        double j_err_j, a_err_j, v_err_j;
        if constexpr (UseConservativeTile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            j_err_j = j_err_i;
            a_err_j = std::sqrt((a_err_i * a_err_i) +
                                (j_err_i * j_err_i * delta_t_sq));
            v_err_j = std::sqrt(
                (v_err_i * v_err_i) + (a_err_i * a_err_i * delta_t_sq) +
                (j_err_i * j_err_i * half_delta_t_sq * half_delta_t_sq));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
        }

        // Write back transformed values
        leaves_tree[leaf_offset + 0] = j_val_j;
        leaves_tree[leaf_offset + 1] = j_err_j;
        leaves_tree[leaf_offset + 2] = a_val_j;
        leaves_tree[leaf_offset + 3] = a_err_j;
        leaves_tree[leaf_offset + 4] = v_val_j;
        leaves_tree[leaf_offset + 5] = v_err_j;
        leaves_tree[leaf_offset + 6] = d_val_j;
        leaves_tree[leaf_offset + 7] = d_err_i;
    }
}

template <bool UseConservativeTile>
void poly_taylor_transform_snap_batch(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto delta_t                      = t0_next - t0_cur;
    const auto delta_t_sq                   = delta_t * delta_t;
    const auto half_delta_t_sq              = 0.5 * (delta_t * delta_t);
    const auto sixth_delta_t_cubed          = delta_t_sq * delta_t / 6.0;
    const auto twenty_fourth_delta_t_fourth = delta_t_sq * delta_t_sq / 24.0;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto leaf_offset = i * kLeavesStride;

        const auto s_val_i = leaves_tree[leaf_offset + 0];
        const auto s_err_i = leaves_tree[leaf_offset + 1];
        const auto j_val_i = leaves_tree[leaf_offset + 2];
        const auto j_err_i = leaves_tree[leaf_offset + 3];
        const auto a_val_i = leaves_tree[leaf_offset + 4];
        const auto a_err_i = leaves_tree[leaf_offset + 5];
        const auto v_val_i = leaves_tree[leaf_offset + 6];
        const auto v_err_i = leaves_tree[leaf_offset + 7];
        const auto d_val_i = leaves_tree[leaf_offset + 8];
        const auto d_err_i = leaves_tree[leaf_offset + 9];

        const auto s_val_j = s_val_i;
        const auto j_val_j = j_val_i + (s_val_i * delta_t);
        const auto a_val_j =
            a_val_i + (j_val_i * delta_t) + (s_val_i * half_delta_t_sq);
        const auto v_val_j = v_val_i + (a_val_i * delta_t) +
                             (j_val_i * half_delta_t_sq) +
                             (s_val_i * sixth_delta_t_cubed);
        const auto d_val_j = d_val_i + (v_val_i * delta_t) +
                             (a_val_i * half_delta_t_sq) +
                             (j_val_i * sixth_delta_t_cubed) +
                             (s_val_i * twenty_fourth_delta_t_fourth);

        double s_err_j, j_err_j, a_err_j, v_err_j;
        if constexpr (UseConservativeTile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            s_err_j = s_err_i;
            j_err_j = std::sqrt((j_err_i * j_err_i) +
                                (s_err_i * s_err_i * delta_t_sq));
            a_err_j = std::sqrt(
                (a_err_i * a_err_i) + (j_err_i * j_err_i * delta_t_sq) +
                (s_err_i * s_err_i * half_delta_t_sq * half_delta_t_sq));
            v_err_j = std::sqrt(
                (v_err_i * v_err_i) + (a_err_i * a_err_i * delta_t_sq) +
                (j_err_i * j_err_i * half_delta_t_sq * half_delta_t_sq) +
                (s_err_i * s_err_i * sixth_delta_t_cubed *
                 sixth_delta_t_cubed));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            s_err_j = s_err_i;
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
        }

        // Write back transformed values
        leaves_tree[leaf_offset + 0] = s_val_j;
        leaves_tree[leaf_offset + 1] = s_err_j;
        leaves_tree[leaf_offset + 2] = j_val_j;
        leaves_tree[leaf_offset + 3] = j_err_j;
        leaves_tree[leaf_offset + 4] = a_val_j;
        leaves_tree[leaf_offset + 5] = a_err_j;
        leaves_tree[leaf_offset + 6] = v_val_j;
        leaves_tree[leaf_offset + 7] = v_err_j;
        leaves_tree[leaf_offset + 8] = d_val_j;
        leaves_tree[leaf_offset + 9] = d_err_i;
    }
}

template <SizeType NPARAMS>
SizeType poly_taylor_branch_batch_impl(std::span<const double> leaves_tree,
                                       std::span<double> leaves_branch,
                                       std::span<SizeType> leaves_origins,
                                       std::pair<double, double> coord_cur,
                                       SizeType nbins,
                                       double eta,
                                       std::span<const ParamLimit> param_limits,
                                       SizeType branch_max,
                                       SizeType n_leaves,
                                       utils::BranchingWorkspaceView ws) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Taylor order");
    if constexpr (NPARAMS == 2) {
        return poly_taylor_branch_accel_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws);
    } else if constexpr (NPARAMS == 3) {
        return poly_taylor_branch_jerk_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws);
    } else if constexpr (NPARAMS == 4) {
        return poly_taylor_branch_snap_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws);
    }
}

template <SizeType NPARAMS, int LATTER>
void ffa_taylor_resolve_poly_batch_impl(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins) {
    static_assert(NPARAMS > 1 && NPARAMS <= 5 && LATTER >= 0 && LATTER <= 1,
                  "Unsupported number of parameters or latter");

    if constexpr (NPARAMS == 2) {
        ffa_taylor_resolve_accel_batch<LATTER>(
            param_grid_count_cur, param_grid_count_prev, param_limits, coords,
            ffa_level, tseg_brute, nbins);
    } else {
        ffa_taylor_resolve_jerk_batch<LATTER>(
            param_grid_count_cur, param_grid_count_prev, param_limits, coords,
            ffa_level, tseg_brute, nbins, NPARAMS);
    }
}

template <SizeType NPARAMS>
void poly_taylor_resolve_batch_impl(std::span<const double> leaves_tree,
                                    std::span<SizeType> param_indices,
                                    std::span<float> phase_shift,
                                    std::span<const ParamLimit> param_limits,
                                    std::pair<double, double> coord_add,
                                    std::pair<double, double> coord_cur,
                                    std::pair<double, double> coord_init,
                                    SizeType n_accel_init,
                                    SizeType n_freq_init,
                                    SizeType nbins,
                                    SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Taylor order");
    if constexpr (NPARAMS == 2) {
        poly_taylor_resolve_accel_batch(
            leaves_tree, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_taylor_resolve_jerk_batch(
            leaves_tree, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_taylor_resolve_snap_batch(
            leaves_tree, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    }
}

template <SizeType NPARAMS, bool UseConservativeTile>
void poly_taylor_transform_batch_impl(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Taylor order");
    if constexpr (NPARAMS == 2) {
        poly_taylor_transform_accel_batch<UseConservativeTile>(
            leaves_tree, coord_next, coord_cur, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_taylor_transform_jerk_batch<UseConservativeTile>(
            leaves_tree, coord_next, coord_cur, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_taylor_transform_snap_batch<UseConservativeTile>(
            leaves_tree, coord_next, coord_cur, n_leaves);
    }
}

} // namespace

std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve_generic(std::span<const double> pset_cur,
                           std::span<const SizeType> param_grid_count_prev,
                           std::span<const ParamLimit> param_limits,
                           SizeType ffa_level,
                           SizeType latter,
                           double tseg_brute,
                           SizeType nbins) {
    const auto nparams = pset_cur.size();
    std::vector<double> pset_prev(nparams, 0.0);
    const double tsegment =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));
    double delta_t{}, delay{};

    if (nparams == 1) {
        delta_t      = static_cast<double>(latter) * tsegment;
        pset_prev[0] = pset_cur[0];
        delay        = 0.0;
    } else {
        delta_t = (static_cast<double>(latter) - 0.5) * tsegment;
        std::tie(pset_prev, delay) =
            transforms::shift_taylor_params_d_f(pset_cur, delta_t);
    }
    const auto relative_phase =
        psr_utils::get_phase_idx(delta_t, pset_cur[nparams - 1], nbins, delay);

    std::vector<SizeType> pindex_prev(nparams);
    for (SizeType ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] = psr_utils::get_nearest_idx_analytical(
            pset_prev[ip], param_limits[ip], param_grid_count_prev[ip]);
    }
    return {pindex_prev, relative_phase};
}

void ffa_taylor_resolve_freq_batch(SizeType n_freqs_cur,
                                   SizeType n_freqs_prev,
                                   const ParamLimit& lim_freq,
                                   std::span<coord::FFACoordFreq> coords,
                                   SizeType ffa_level,
                                   double tseg_brute,
                                   SizeType nbins) {
    error_check::check_equal(coords.size(), n_freqs_cur,
                             "coords size mismatch");

    const double delta_t =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));

    // Calculate relative phases and flattened parameter indices
    for (SizeType i = 0; i < n_freqs_cur; ++i) {
        const double f_cur =
            psr_utils::get_param_val_at_idx(lim_freq, n_freqs_cur, i);
        const SizeType idx_f = psr_utils::get_nearest_idx_analytical(
            f_cur, lim_freq, n_freqs_prev);
        coords[i].idx   = static_cast<uint32_t>(idx_f);
        coords[i].shift = psr_utils::get_phase_idx(delta_t, f_cur, nbins, 0.0);
    }
}

void ffa_taylor_resolve_poly_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins,
    SizeType n_params) {
    auto dispatch = [&]<SizeType N, int L>() {
        return ffa_taylor_resolve_poly_batch_impl<N, L>(
            param_grid_count_cur, param_grid_count_prev, param_limits, coords,
            ffa_level, tseg_brute, nbins);
    };
    if (latter == 0) {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, 0>();
            break;
        case 3:
            dispatch.template operator()<3, 0>();
            break;
        case 4:
            dispatch.template operator()<4, 0>();
            break;
        case 5:
            dispatch.template operator()<5, 0>();
            break;
        default:
            throw std::invalid_argument("Unsupported Taylor order");
        }
    } else {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, 1>();
            break;
        case 3:
            dispatch.template operator()<3, 1>();
            break;
        case 4:
            dispatch.template operator()<4, 1>();
            break;
        case 5:
            dispatch.template operator()<5, 1>();
            break;
        default:
            throw std::invalid_argument("Unsupported Taylor order");
        }
    }
}

SizeType poly_taylor_seed(std::span<const SizeType> param_grid_count_init,
                          std::span<const double> dparams_init,
                          std::span<const ParamLimit> param_limits,
                          std::span<double> seed_leaves,
                          std::pair<double, double> /*coord_init*/,
                          SizeType n_params) {
    constexpr SizeType kParamStride = 2U;
    error_check::check_equal(param_grid_count_init.size(), n_params,
                             "param_grid_count_init size mismatch");
    error_check::check_equal(dparams_init.size(), n_params,
                             "dparams_init size mismatch");
    error_check::check_equal(param_limits.size(), n_params,
                             "param_limits size mismatch");
    const SizeType leaves_stride = (n_params + 2) * kParamStride;
    SizeType n_leaves            = 1;
    for (const auto count : param_grid_count_init) {
        n_leaves *= count;
    }
    const auto n_accel_init = param_grid_count_init[n_params - 2];
    const auto n_freq_init  = param_grid_count_init[n_params - 1];
    const auto& lim_accel   = param_limits[n_params - 2];
    const auto& lim_freq    = param_limits[n_params - 1];
    const auto d_freq_cur   = dparams_init[n_params - 1];
    error_check::check_equal(n_leaves, n_accel_init * n_freq_init,
                             "n_leaves mismatch");

    // Create parameter sets tensor: (n_param_sets, poly_order + 2, 2)
    error_check::check_greater_equal(seed_leaves.size(),
                                     n_leaves * leaves_stride,
                                     "seed_leaves size mismatch");
    SizeType leaf_idx = 0;
    for (SizeType accel_idx = 0; accel_idx < n_accel_init; ++accel_idx) {
        const auto accel_cur =
            psr_utils::get_param_val_at_idx(lim_accel, n_accel_init, accel_idx);
        for (SizeType freq_idx = 0; freq_idx < n_freq_init; ++freq_idx) {
            const auto freq_cur = psr_utils::get_param_val_at_idx(
                lim_freq, n_freq_init, freq_idx);
            const auto lo = leaf_idx * leaves_stride;
            // Copy till d2 (acceleration)
            for (SizeType j = 0; j < n_params - 1; ++j) {
                seed_leaves[lo + (j * kParamStride) + 0] = 0;
                seed_leaves[lo + (j * kParamStride) + 1] = dparams_init[j];
            }
            seed_leaves[lo + ((n_params - 2) * kParamStride) + 0] = accel_cur;
            // Update frequency to velocity
            // f = f0(1 - v / C) => dv = -(C/f0) * df
            seed_leaves[lo + ((n_params - 1) * kParamStride) + 0] = 0;
            seed_leaves[lo + ((n_params - 1) * kParamStride) + 1] =
                d_freq_cur * (utils::kCval / freq_cur);
            // intialize d0 (measure from t=t_init) and store f0
            seed_leaves[lo + ((n_params + 0) * kParamStride) + 0] = 0;
            seed_leaves[lo + ((n_params + 0) * kParamStride) + 1] = 0;
            seed_leaves[lo + ((n_params + 1) * kParamStride) + 0] = freq_cur;
            // Store basis flag (0: Polynomial, 1: Physical)
            seed_leaves[lo + ((n_params + 1) * kParamStride) + 1] = 0;
            ++leaf_idx;
        }
    }
    error_check::check_equal(leaf_idx, n_leaves, "n_leaves mismatch");
    return n_leaves;
}

std::vector<SizeType>
poly_taylor_branch_batch_generic(std::span<const double> leaves_batch,
                                 std::pair<double, double> coord_cur,
                                 std::span<double> leaves_branch_batch,
                                 SizeType nbins,
                                 double eta,
                                 std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 SizeType n_params) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (n_params + 2) * kParamStride;
    error_check::check_equal(leaves_batch.size(), n_leaves * leaves_stride,
                             "leaves_batch size mismatch");
    error_check::check_greater_equal(leaves_branch_batch.size(),
                                     n_leaves * branch_max * leaves_stride,
                                     "leaves_branch_batch size mismatch");

    const auto [t0_cur, t_obs_minus_t_ref] = coord_cur;

    // Use leaves_branch_batch memory as workspace. Partition workspace into
    // sections:
    const SizeType workspace_size      = leaves_branch_batch.size();
    const SizeType single_batch_params = n_leaves * n_params;

    // Get spans from workspace + other vector allocations
    std::span<double> dparam_cur_batch =
        leaves_branch_batch.subspan(0, single_batch_params);
    std::span<double> dparam_new_batch =
        leaves_branch_batch.subspan(single_batch_params, single_batch_params);
    std::span<double> shift_bins_batch = leaves_branch_batch.subspan(
        single_batch_params * 2, single_batch_params);
    std::span<double> f0_batch =
        leaves_branch_batch.subspan(single_batch_params * 3, n_leaves);
    std::span<double> pad_branched_params = leaves_branch_batch.subspan(
        (single_batch_params * 3) + n_leaves, n_leaves * n_params * branch_max);
    const auto workspace_acquired_size = (single_batch_params * 3) + n_leaves +
                                         (n_leaves * n_params * branch_max);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType param_offset = i * leaves_stride;
        for (SizeType j = 0; j < n_params; ++j) {
            dparam_cur_batch[(i * n_params) + j] =
                leaves_batch[(param_offset + (j * kParamStride)) + 1];
        }
        f0_batch[i] =
            leaves_batch[param_offset + ((n_params + 1) * kParamStride)];
    }

    psr_utils::poly_taylor_step_d_vec(n_params, t_obs_minus_t_ref, nbins, eta,
                                      f0_batch, dparam_new_batch, 0);
    psr_utils::poly_taylor_shift_d_vec(dparam_cur_batch, dparam_new_batch,
                                       t_obs_minus_t_ref, nbins, f0_batch, 0,
                                       shift_bins_batch, n_leaves, n_params);

    std::vector<double> pad_branched_dparams(n_leaves * n_params);
    std::vector<SizeType> branched_counts(n_leaves * n_params);
    // Optimized branching loop - same logic as original but vectorized access
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * leaves_stride;
        const SizeType flat_base   = i * n_params;

        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType flat_idx     = flat_base + j;
            const SizeType param_offset = leaf_offset + (j * kParamStride);
            const double param_cur_val  = leaves_batch[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];

            if (shift_bins_batch[flat_idx] >= (eta - utils::kEps)) {
                double param_min = std::numeric_limits<double>::lowest();
                double param_max = std::numeric_limits<double>::max();
                if (j == n_params - 1) {
                    param_min =
                        (1 - param_limits[j].max / f0_batch[i]) * utils::kCval;
                    param_max =
                        (1 - param_limits[j].min / f0_batch[i]) * utils::kCval;
                } else {
                    param_min = param_limits[j].min;
                    param_max = param_limits[j].max;
                }
                const SizeType pad_offset =
                    (i * n_params * branch_max) + (j * branch_max);
                std::span<double> slice_span =
                    pad_branched_params.subspan(pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, param_cur_val, dparam_cur_val,
                    dparam_new_batch[flat_idx], param_min, param_max);

                pad_branched_dparams[flat_idx] = dparam_act;
                branched_counts[flat_idx]      = count;
            } else {
                // No branching: only use current value
                const SizeType pad_offset =
                    (i * n_params * branch_max) + (j * branch_max);
                pad_branched_params[pad_offset] = param_cur_val;
                pad_branched_dparams[flat_idx]  = dparam_cur_val;
                branched_counts[flat_idx]       = 1;
            }
        }
    }

    // Use the existing robust Cartesian product function
    const auto [leaves_branch_taylor_batch, batch_origins] =
        utils::cartesian_prod_padded(pad_branched_params, branched_counts,
                                     n_leaves, n_params, branch_max);
    const SizeType total_leaves = batch_origins.size();

    // Fill dparams and other parameters using the same logic as original
    for (SizeType i = 0; i < total_leaves; ++i) {
        const SizeType origin        = batch_origins[i];
        const SizeType branch_offset = i * leaves_stride;
        const SizeType leaf_offset   = origin * leaves_stride;
        const SizeType d0_branch_offset =
            branch_offset + (n_params * kParamStride);
        const SizeType f0_branch_offset = d0_branch_offset + kParamStride;
        const SizeType d0_leaf_offset = leaf_offset + (n_params * kParamStride);
        const SizeType f0_leaf_offset = d0_leaf_offset + kParamStride;

        // Fill parameters and dparams
        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType leaf_offset_j = branch_offset + (j * kParamStride);
            leaves_branch_batch[leaf_offset_j + 0] =
                leaves_branch_taylor_batch[(i * n_params) + j];
            leaves_branch_batch[leaf_offset_j + 1] =
                pad_branched_dparams[(origin * n_params) + j];
        }
        // Fill d0 and f0 directly from leaves_batch
        leaves_branch_batch[d0_branch_offset + 0] =
            leaves_batch[d0_leaf_offset + 0];
        leaves_branch_batch[d0_branch_offset + 1] =
            leaves_batch[d0_leaf_offset + 1];
        leaves_branch_batch[f0_branch_offset + 0] =
            leaves_batch[f0_leaf_offset + 0];
        leaves_branch_batch[f0_branch_offset + 1] =
            leaves_batch[f0_leaf_offset + 1];
    }

    return batch_origins;
}

SizeType poly_taylor_branch_batch(std::span<const double> leaves_tree,
                                  std::span<double> leaves_branch,
                                  std::span<SizeType> leaves_origins,
                                  std::pair<double, double> coord_cur,
                                  SizeType nbins,
                                  double eta,
                                  std::span<const ParamLimit> param_limits,
                                  SizeType branch_max,
                                  SizeType n_leaves,
                                  SizeType n_params,
                                  utils::BranchingWorkspaceView ws) {

    auto dispatch = [&]<SizeType N>() {
        return poly_taylor_branch_batch_impl<N>(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, ws);
    };
    switch (n_params) {
    case 2:
        return dispatch.template operator()<2>();
        break;
    case 3:
        return dispatch.template operator()<3>();
        break;
    case 4:
        return dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported Taylor order");
    }
}

void poly_taylor_resolve_batch(std::span<const double> leaves_branch,
                               std::span<SizeType> param_indices,
                               std::span<float> phase_shift,
                               std::span<const ParamLimit> param_limits,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_cur,
                               std::pair<double, double> coord_init,
                               SizeType n_accel_init,
                               SizeType n_freq_init,
                               SizeType nbins,
                               SizeType n_leaves,
                               SizeType n_params) {
    auto dispatch = [&]<SizeType N>() {
        return poly_taylor_resolve_batch_impl<N>(
            leaves_branch, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    };
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
        throw std::invalid_argument("Unsupported Taylor order");
    }
}

void poly_taylor_transform_batch(std::span<double> leaves_tree,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 bool use_conservative_tile) {
    auto dispatch = [&]<SizeType N, bool C>() {
        return poly_taylor_transform_batch_impl<N, C>(leaves_tree, coord_next,
                                                      coord_cur, n_leaves);
    };
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
            throw std::invalid_argument("Unsupported Taylor order");
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
            throw std::invalid_argument("Unsupported Taylor order");
        }
    }
}

void poly_taylor_report_batch(std::span<double> leaves_tree,
                              std::pair<double, double> /*coord_report*/,
                              SizeType n_leaves,
                              SizeType n_params) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = n_params * kParamStride;
    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * leaves_stride,
                                     "leaves_tree size not enough");
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto leaf_offset = i * leaves_stride;
        const auto v_final =
            leaves_tree[leaf_offset + ((n_params - 1) * kParamStride) + 0];
        const auto dv_final =
            leaves_tree[leaf_offset + ((n_params - 1) * kParamStride) + 1];
        const auto f0_batch =
            leaves_tree[leaf_offset + ((n_params + 1) * kParamStride) + 0];
        const auto s_factor = 1.0 - (v_final * utils::kInvCval);
        // Gauge transform + error propagation
        for (SizeType j = 0; j < n_params - 1; ++j) {
            const auto param_offset       = leaf_offset + (j * kParamStride);
            const auto param_val          = leaves_tree[param_offset + 0];
            const auto param_err          = leaves_tree[param_offset + 1];
            leaves_tree[param_offset + 0] = param_val / s_factor;
            leaves_tree[param_offset + 1] = std::sqrt(
                ((param_err / s_factor) * (param_err / s_factor)) +
                ((param_val * utils::kInvCval / (s_factor * s_factor)) *
                 (param_val * utils::kInvCval / (s_factor * s_factor)) *
                 (dv_final * dv_final)));
        }
        leaves_tree[leaf_offset + ((n_params - 1) * kParamStride) + 0] =
            f0_batch * s_factor;
        leaves_tree[leaf_offset + ((n_params - 1) * kParamStride) + 1] =
            f0_batch * dv_final * utils::kInvCval;
    }
}

std::vector<double>
poly_taylor_branch_generic(std::span<const double> leaf,
                           std::pair<double, double> coord_cur,
                           SizeType nbins,
                           double eta,
                           std::span<const ParamLimit> param_limits,
                           SizeType n_params) {
    const auto branch_max    = 100;
    const auto leaves_stride = (n_params + 2) * 2;
    std::vector<double> branch_leaves(branch_max * leaves_stride);
    const auto batch_origins = poly_taylor_branch_batch_generic(
        leaf, coord_cur, branch_leaves, nbins, eta, param_limits, branch_max, 1,
        n_params);
    return {branch_leaves.begin(),
            branch_leaves.begin() +
                static_cast<IndexType>(batch_origins.size() * leaves_stride)};
}

std::vector<double>
generate_bp_poly_taylor_approx(std::span<const SizeType> param_grid_count_init,
                               std::span<const double> dparams_init,
                               std::span<const ParamLimit> param_limits,
                               double tseg_ffa,
                               SizeType nsegments,
                               SizeType nbins,
                               double eta,
                               SizeType ref_seg,
                               IndexType isuggest,
                               bool use_conservative_tile) {
    error_check::check_equal(param_grid_count_init.size(), param_limits.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_grid_count_init.size(), param_limits.size(),
        "param_grid_count_init and param_limits must have the same size");
    std::vector<double> branching_pattern(nsegments - 1);
    const auto n_params      = param_grid_count_init.size();
    const auto leaves_stride = (n_params + 2) * 2;

    psr_utils::MiddleOutScheme snail_scheme(nsegments, ref_seg, tseg_ffa);
    const auto coord_init = snail_scheme.get_coord(0);
    SizeType n_leaves     = 1;
    for (const auto count : param_grid_count_init) {
        n_leaves *= count;
    }
    std::vector<double> seed_leaves(n_leaves * leaves_stride);
    const auto n_leaves_seed =
        poly_taylor_seed(param_grid_count_init, dparams_init, param_limits,
                         seed_leaves, coord_init, n_params);
    error_check::check_equal(n_leaves_seed, n_leaves, "n_leaves mismatch");
    // Get isuggest-th leaf
    if (isuggest < 0) { // Negative index
        isuggest = static_cast<IndexType>(n_leaves + isuggest);
    }
    error_check::check_greater_equal(isuggest, 0,
                                     "isuggest must be non-negative");
    error_check::check_less(isuggest, n_leaves,
                            "isuggest must be less than n_leaves");
    auto leaf = std::span(seed_leaves)
                    .subspan((leaves_stride * isuggest), leaves_stride);
    std::vector<double> leaf_data(leaves_stride);
    std::ranges::copy(leaf, leaf_data.begin());
    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = snail_scheme.get_coord(prune_level);
        const auto coord_cur  = snail_scheme.get_current_coord(prune_level);
        auto leaves_arr       = poly_taylor_branch_generic(
            leaf_data, coord_cur, nbins, eta, param_limits, n_params);
        const auto n_leaves_branch = leaves_arr.size() / leaves_stride;
        branching_pattern[prune_level - 1] =
            static_cast<double>(n_leaves_branch);
        poly_taylor_transform_batch(leaves_arr, coord_next, coord_cur,
                                    n_leaves_branch, n_params,
                                    use_conservative_tile);
        const auto leaf_start = leaves_stride * (n_leaves_branch - 1);
        std::ranges::copy(
            leaves_arr.begin() + static_cast<IndexType>(leaf_start),
            leaves_arr.begin() +
                static_cast<IndexType>(leaf_start + leaves_stride),
            leaf_data.begin());
    }
    return branching_pattern;
}

std::vector<double>
generate_bp_poly_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        std::span<const ParamLimit> param_limits,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        bool use_conservative_tile) {
    error_check::check_equal(param_arr.size(), dparams.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    const auto n_params  = dparams.size();
    const auto& f0_batch = param_arr.back(); // Last array is frequency
    const auto n_freqs   = f0_batch.size();  // Number of frequency bins

    // Snail Scheme
    psr_utils::MiddleOutScheme snail_scheme(nsegments, ref_seg, tseg_ffa);
    std::vector<double> weights(n_freqs, 1.0);
    std::vector<double> branching_pattern(nsegments - 1);

    // Initialize dparam_cur_batch - each frequency gets the same dparams
    std::vector<double> dparam_cur_batch(n_freqs * n_params);
    for (SizeType i = 0; i < n_freqs; ++i) {
        std::ranges::copy(dparams, dparam_cur_batch.begin() +
                                       static_cast<IndexType>(i * n_params));
    }
    // f = f0(1 - v / C) => dv = -(C/f0) * df
    for (SizeType i = 0; i < n_freqs; ++i) {
        dparam_cur_batch[(i * n_params) + n_params - 1] =
            dparam_cur_batch[(i * n_params) + n_params - 1] *
            (utils::kCval / f0_batch[i]);
    }

    // Pre-compute parameter ranges
    std::vector<double> param_ranges(n_freqs * n_params);
    for (SizeType i = 0; i < n_freqs; ++i) {
        for (SizeType j = 0; j < n_params; ++j) {
            if (j == n_params - 1) {
                const auto param_min =
                    (1 - param_limits[j].max / f0_batch[i]) * utils::kCval;
                const auto param_max =
                    (1 - param_limits[j].min / f0_batch[i]) * utils::kCval;
                param_ranges[(i * n_params) + j] =
                    (param_max - param_min) / 2.0;
            } else {
                param_ranges[(i * n_params) + j] =
                    (param_limits[j].max - param_limits[j].min) / 2.0;
            }
        }
    }

    std::vector<double> dparam_new_batch(n_freqs * n_params, 0.0);
    std::vector<double> shift_bins_batch(n_freqs * n_params, 0.0);
    std::vector<double> dparam_cur_next(n_freqs * n_params, 0.0);
    std::vector<double> n_branches(n_freqs, 1);
    const auto n_params_d = n_params + 1;
    std::vector<double> dparam_d_vec(n_freqs * n_params_d, 0.0);

    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = snail_scheme.get_coord(prune_level);
        const auto coord_cur  = snail_scheme.get_current_coord(prune_level);
        const auto [_, t_obs_minus_t_ref] = coord_cur;

        // Calculate optimal parameter steps and shift bins
        psr_utils::poly_taylor_step_d_vec(n_params, t_obs_minus_t_ref, nbins,
                                          eta, f0_batch, dparam_new_batch, 0);
        psr_utils::poly_taylor_shift_d_vec(
            dparam_cur_batch, dparam_new_batch, t_obs_minus_t_ref, nbins,
            f0_batch, 0, shift_bins_batch, n_freqs, n_params);

        std::ranges::fill(n_branches, 1.0);
        // Determine branching needs
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < n_params; ++j) {
                const auto idx = (i * n_params) + j;
                const auto needs_branching =
                    shift_bins_batch[idx] >= (eta - utils::kEps);
                const auto too_large_step =
                    dparam_new_batch[idx] > (param_ranges[idx] + utils::kEps);

                if (!needs_branching || too_large_step) {
                    dparam_cur_next[idx] = dparam_cur_batch[idx];
                    continue;
                }
                const auto ratio = (dparam_cur_batch[idx] + utils::kEps) /
                                   (dparam_new_batch[idx]);
                const auto num_points = std::max(
                    1, static_cast<int>(std::ceil(ratio - utils::kEps)));
                n_branches[i] *= static_cast<double>(num_points);
                dparam_cur_next[idx] =
                    dparam_cur_batch[idx] / static_cast<double>(num_points);
            }
        }

        // Compute average branching factor and update weights
        double children = 0.0;
        double parents  = 0.0;
        for (SizeType i = 0; i < n_freqs; ++i) {
            children += weights[i] * n_branches[i];
            parents += weights[i];
            weights[i] *= n_branches[i];
        }
        branching_pattern[prune_level - 1] = children / parents;

        // Transform dparams to the next segment
        const auto delta_t = coord_next.first - coord_cur.first;
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < n_params; ++j) {
                dparam_d_vec[(i * n_params_d) + j] =
                    dparam_cur_next[(i * n_params) + j];
            }
        }
        auto dparam_d_vec_new = transforms::shift_taylor_errors_batch(
            dparam_d_vec, delta_t, use_conservative_tile, n_freqs, n_params_d);
        // Copy back to dparam_cur_batch (excluding last dimension)
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < n_params; ++j) {
                dparam_cur_batch[(i * n_params) + j] =
                    dparam_d_vec_new[(i * n_params_d) + j];
            }
        }
    }
    return branching_pattern;
}

} // namespace loki::core
