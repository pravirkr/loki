#include "loki/core/taylor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"
#include "loki/utils.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::core {

namespace {

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
                               memory::BranchingWorkspace& branch_ws) {
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
    const auto [_, dt]    = coord_cur; // t_obs_minus_t_ref
    const double dt2      = dt * dt;
    const double inv_dt   = 1.0 / dt;
    const double inv_dt2  = inv_dt * inv_dt;
    const auto nbins_d    = static_cast<double>(nbins);
    const double dphi     = eta / nbins_d;
    const double d2_range = param_limits[0].max - param_limits[0].min;
    const double f0_range = param_limits[1].max - param_limits[1].min;

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
    double* __restrict__ scratch_params   = branch_ws.scratch_params.data();
    double* __restrict__ scratch_dparams  = branch_ws.scratch_dparams.data();
    SizeType* __restrict__ scratch_counts = branch_ws.scratch_counts.data();

    // Step + Shift
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto d2_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 3];
        const auto f0         = leaves_tree_ptr[lo + 6];

        // Compute steps
        const auto dfactor    = utils::kCval / f0;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        // Compute new dparams with limited range
        const auto d1_range    = dfactor * f0_range;
        dparam_new_ptr[fb + 0] = std::min(d2_sig_new, d2_range);
        dparam_new_ptr[fb + 1] = std::min(d1_sig_new, d1_range);

        // Compute shift bins
        shift_bins_ptr[fb + 0] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[fb + 1] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // Early Exit: Check if any leaf needs branching
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

    // Branching
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo         = i * kLeavesStride;
        const auto fb         = i * kParams;
        const auto d2_cur     = leaves_tree_ptr[lo + 0];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d1_cur     = leaves_tree_ptr[lo + 2];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 3];

        // Branch d2-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            1, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
    }

    // Fill leaves_origins
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo            = i * kLeavesStride;
        const SizeType fb            = i * kParams;
        const SizeType n_d2_branches = scratch_counts[fb + 0];
        const SizeType n_d1_branches = scratch_counts[fb + 1];
        const SizeType d2_offset     = (fb + 0) * branch_max;
        const SizeType d1_offset     = (fb + 1) * branch_max;

        for (SizeType a = 0; a < n_d2_branches; ++a) {
            for (SizeType b = 0; b < n_d1_branches; ++b) {
                const SizeType bo         = out_leaves * kLeavesStride;
                leaves_branch_ptr[bo + 0] = scratch_params[d2_offset + a];
                leaves_branch_ptr[bo + 1] = scratch_dparams[fb + 0];
                leaves_branch_ptr[bo + 2] = scratch_params[d1_offset + b];
                leaves_branch_ptr[bo + 3] = scratch_dparams[fb + 1];
                // Fill d0 and f0 directly from leaves_tree
                std::memcpy(leaves_branch_ptr + bo + 4,
                            leaves_tree_ptr + lo + 4, 4 * sizeof(double));

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
                                       memory::BranchingWorkspace& branch_ws) {
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
    const auto [_, dt]    = coord_cur; // t_obs_minus_t_ref
    const double dt2      = dt * dt;
    const double dt3      = dt2 * dt;
    const double inv_dt   = 1.0 / dt;
    const double inv_dt2  = inv_dt * inv_dt;
    const double inv_dt3  = inv_dt2 * inv_dt;
    const auto nbins_d    = static_cast<double>(nbins);
    const double dphi     = eta / nbins_d;
    const double d3_range = param_limits[0].max - param_limits[0].min;
    const double d2_range = param_limits[1].max - param_limits[1].min;
    const double f0_range = param_limits[2].max - param_limits[2].min;

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
    double* __restrict__ scratch_params   = branch_ws.scratch_params.data();
    double* __restrict__ scratch_dparams  = branch_ws.scratch_dparams.data();
    SizeType* __restrict__ scratch_counts = branch_ws.scratch_counts.data();

    // Step + Shift
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto d3_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 5];
        const auto f0         = leaves_tree_ptr[lo + 8];

        const auto dfactor    = utils::kCval / f0;
        const auto d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        // Compute new dparams with limited range
        const auto d1_range    = dfactor * f0_range;
        dparam_new_ptr[fb + 0] = std::min(d3_sig_new, d3_range);
        dparam_new_ptr[fb + 1] = std::min(d2_sig_new, d2_range);
        dparam_new_ptr[fb + 2] = std::min(d1_sig_new, d1_range);

        shift_bins_ptr[fb + 0] =
            (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
        shift_bins_ptr[fb + 1] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[fb + 2] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // Early Exit: Check if any leaf needs branching
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

    // Branching
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo         = i * kLeavesStride;
        const auto fb         = i * kParams;
        const auto d3_cur     = leaves_tree_ptr[lo + 0];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d2_cur     = leaves_tree_ptr[lo + 2];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d1_cur     = leaves_tree_ptr[lo + 4];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 5];

        // Branch d3-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d3_cur, d3_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            1, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            2, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
    }

    // Fill leaves_origins
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo            = i * kLeavesStride;
        const SizeType fb            = i * kParams;
        const SizeType n_d3_branches = scratch_counts[fb + 0];
        const SizeType n_d2_branches = scratch_counts[fb + 1];
        const SizeType n_d1_branches = scratch_counts[fb + 2];
        const SizeType d3_offset     = (fb + 0) * branch_max;
        const SizeType d2_offset     = (fb + 1) * branch_max;
        const SizeType d1_offset     = (fb + 2) * branch_max;

        for (SizeType a = 0; a < n_d3_branches; ++a) {
            for (SizeType b = 0; b < n_d2_branches; ++b) {
                for (SizeType c = 0; c < n_d1_branches; ++c) {
                    const SizeType bo         = out_leaves * kLeavesStride;
                    leaves_branch_ptr[bo + 0] = scratch_params[d3_offset + a];
                    leaves_branch_ptr[bo + 1] = scratch_dparams[fb + 0];
                    leaves_branch_ptr[bo + 2] = scratch_params[d2_offset + b];
                    leaves_branch_ptr[bo + 3] = scratch_dparams[fb + 1];
                    leaves_branch_ptr[bo + 4] = scratch_params[d1_offset + c];
                    leaves_branch_ptr[bo + 5] = scratch_dparams[fb + 2];
                    // Fill d0 and f0 directly from leaves_tree
                    std::memcpy(leaves_branch_ptr + bo + 6,
                                leaves_tree_ptr + lo + 6, 4 * sizeof(double));

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
                                       memory::BranchingWorkspace& branch_ws) {
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
    const auto [_, dt]    = coord_cur; // t_obs_minus_t_ref
    const double dt2      = dt * dt;
    const double dt3      = dt2 * dt;
    const double dt4      = dt2 * dt2;
    const double inv_dt   = 1.0 / dt;
    const double inv_dt2  = inv_dt * inv_dt;
    const double inv_dt3  = inv_dt2 * inv_dt;
    const double inv_dt4  = inv_dt2 * inv_dt2;
    const auto nbins_d    = static_cast<double>(nbins);
    const double dphi     = eta / nbins_d;
    const double d4_range = param_limits[0].max - param_limits[0].min;
    const double d3_range = param_limits[1].max - param_limits[1].min;
    const double d2_range = param_limits[2].max - param_limits[2].min;
    const double f0_range = param_limits[3].max - param_limits[3].min;

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
    double* __restrict__ scratch_params   = branch_ws.scratch_params.data();
    double* __restrict__ scratch_dparams  = branch_ws.scratch_dparams.data();
    SizeType* __restrict__ scratch_counts = branch_ws.scratch_counts.data();

    // Step + Shift
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto d4_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 5];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 7];
        const auto f0         = leaves_tree_ptr[lo + 10];

        const auto dfactor    = utils::kCval / f0;
        const auto d4_sig_new = dphi * dfactor * 192.0 * inv_dt4;
        const auto d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        // Compute new dparams with limited range
        const auto d1_range    = dfactor * f0_range;
        dparam_new_ptr[fb + 0] = std::min(d4_sig_new, d4_range);
        dparam_new_ptr[fb + 1] = std::min(d3_sig_new, d3_range);
        dparam_new_ptr[fb + 2] = std::min(d2_sig_new, d2_range);
        dparam_new_ptr[fb + 3] = std::min(d1_sig_new, d1_range);

        shift_bins_ptr[fb + 0] =
            (d4_sig_cur - d4_sig_new) * dt4 * nbins_d / (192.0 * dfactor);
        shift_bins_ptr[fb + 1] =
            (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
        shift_bins_ptr[fb + 2] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[fb + 3] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // Early Exit: Check if any leaf needs branching
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

    // Loop 2: branching
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo     = i * kLeavesStride;
        const SizeType fb     = i * kParams;
        const auto d4_cur     = leaves_tree_ptr[lo + 0];
        const auto d4_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d3_cur     = leaves_tree_ptr[lo + 2];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d2_cur     = leaves_tree_ptr[lo + 4];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 5];
        const auto d1_cur     = leaves_tree_ptr[lo + 6];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 7];

        // Branch d4-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d4_cur, d4_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            1, d3_cur, d3_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            2, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            3, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
    }

    // Fill leaves_origins
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo            = i * kLeavesStride;
        const SizeType fb            = i * kParams;
        const SizeType n_d4_branches = scratch_counts[fb + 0];
        const SizeType n_d3_branches = scratch_counts[fb + 1];
        const SizeType n_d2_branches = scratch_counts[fb + 2];
        const SizeType n_d1_branches = scratch_counts[fb + 3];
        const SizeType d4_offset     = (fb + 0) * branch_max;
        const SizeType d3_offset     = (fb + 1) * branch_max;
        const SizeType d2_offset     = (fb + 2) * branch_max;
        const SizeType d1_offset     = (fb + 3) * branch_max;

        for (SizeType a = 0; a < n_d4_branches; ++a) {
            for (SizeType b = 0; b < n_d3_branches; ++b) {
                for (SizeType c = 0; c < n_d2_branches; ++c) {
                    for (SizeType d = 0; d < n_d1_branches; ++d) {
                        const SizeType bo = out_leaves * kLeavesStride;
                        leaves_branch_ptr[bo + 0] =
                            scratch_params[d4_offset + a];
                        leaves_branch_ptr[bo + 1] = scratch_dparams[fb + 0];
                        leaves_branch_ptr[bo + 2] =
                            scratch_params[d3_offset + b];
                        leaves_branch_ptr[bo + 3] = scratch_dparams[fb + 1];
                        leaves_branch_ptr[bo + 4] =
                            scratch_params[d2_offset + c];
                        leaves_branch_ptr[bo + 5] = scratch_dparams[fb + 2];
                        leaves_branch_ptr[bo + 6] =
                            scratch_params[d1_offset + d];
                        leaves_branch_ptr[bo + 7] = scratch_dparams[fb + 3];
                        // Fill d0 and f0 directly from leaves_tree
                        std::memcpy(leaves_branch_ptr + bo + 8,
                                    leaves_tree_ptr + lo + 8,
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
    const auto dt_add   = t0_add - t0_cur;
    const auto dt_init  = t0_init - t0_cur;
    const auto dt       = dt_add - dt_init;
    const auto half_dt2 = 0.5 * ((dt_add * dt_add) - (dt_init * dt_init));

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo      = i * kLeavesStride;
        const auto a_cur   = leaves_tree[lo + 0];
        const auto v_cur   = leaves_tree[lo + 2];
        const auto f0      = leaves_tree[lo + 6];
        const auto a_new   = a_cur;
        const auto delta_v = a_cur * dt;
        const auto delta_d = (v_cur * dt) + (a_cur * half_dt2);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
        const auto delay_rel = delta_d * utils::kInvCval;

        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
        phase_shift[i]   = psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
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
    const auto dt_add        = t0_add - t0_cur;
    const auto dt_init       = t0_init - t0_cur;
    const auto half_dt2_add  = 0.5 * (dt_add * dt_add);
    const auto half_dt2_init = 0.5 * (dt_init * dt_init);
    const auto dt            = dt_add - dt_init;
    const auto half_dt2      = half_dt2_add - half_dt2_init;
    const auto sixth_dt3 =
        ((half_dt2_add * dt_add) - (half_dt2_init * dt_init)) / 3.0;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo      = i * kLeavesStride;
        const auto j_cur   = leaves_tree[lo + 0];
        const auto a_cur   = leaves_tree[lo + 2];
        const auto v_cur   = leaves_tree[lo + 4];
        const auto f0      = leaves_tree[lo + 8];
        const auto a_new   = a_cur + (j_cur * dt_add);
        const auto delta_v = (a_cur * dt) + (j_cur * half_dt2);
        const auto delta_d =
            (v_cur * dt) + (a_cur * half_dt2) + (j_cur * sixth_dt3);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
        const auto delay_rel = delta_d * utils::kInvCval;

        // Calculate relative phase
        phase_shift[i] = psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
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
    const auto dt_add        = t0_add - t0_cur;
    const auto dt_init       = t0_init - t0_cur;
    const auto half_dt2_add  = 0.5 * (dt_add * dt_add);
    const auto half_dt2_init = 0.5 * (dt_init * dt_init);
    const auto dt            = dt_add - dt_init;
    const auto half_dt2      = half_dt2_add - half_dt2_init;
    const auto sixth_dt3 =
        ((half_dt2_add * dt_add) - (half_dt2_init * dt_init)) / 3.0;
    const auto twenty_fourth_dt4 =
        ((half_dt2_add * half_dt2_add) - (half_dt2_init * half_dt2_init)) / 6.0;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo    = i * kLeavesStride;
        const auto s_cur = leaves_tree[lo + 0];
        const auto j_cur = leaves_tree[lo + 2];
        const auto a_cur = leaves_tree[lo + 4];
        const auto v_cur = leaves_tree[lo + 6];
        const auto f0    = leaves_tree[lo + 10];
        const auto a_new = a_cur + (j_cur * dt_add) + (s_cur * half_dt2_add);
        const auto delta_v =
            (a_cur * dt) + (j_cur * half_dt2) + (s_cur * sixth_dt3);
        const auto delta_d = (v_cur * dt) + (a_cur * half_dt2) +
                             (j_cur * sixth_dt3) + (s_cur * twenty_fourth_dt4);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
        const auto delay_rel = delta_d * utils::kInvCval;

        // Calculate relative phase
        phase_shift[i] = psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
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
                                       std::span<SizeType> indices_tree,
                                       std::pair<double, double> coord_next,
                                       std::pair<double, double> coord_cur,
                                       SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto dt       = t0_next - t0_cur;
    const auto half_dt2 = 0.5 * (dt * dt);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo       = indices_tree[i] * kLeavesStride;
        const auto d2_val_i = leaves_tree[lo + 0];
        const auto d2_err_i = leaves_tree[lo + 1];
        const auto d1_val_i = leaves_tree[lo + 2];
        const auto d1_err_i = leaves_tree[lo + 3];
        const auto d0_val_i = leaves_tree[lo + 4];
        const auto d0_err_i = leaves_tree[lo + 5];

        const auto d2_val_j = d2_val_i;
        const auto d1_val_j = d1_val_i + (d2_val_i * dt);
        const auto d0_val_j =
            d0_val_i + (d1_val_i * dt) + (d2_val_i * half_dt2);

        double d2_err_j, d1_err_j;
        if constexpr (UseConservativeTile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            d2_err_j = d2_err_i;
            d1_err_j = std::sqrt((d1_err_i * d1_err_i) +
                                 (d2_err_i * d2_err_i * dt * dt));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            d2_err_j = d2_err_i;
            d1_err_j = d1_err_i;
        }

        // Write back transformed values
        leaves_tree[lo + 0] = d2_val_j;
        leaves_tree[lo + 1] = d2_err_j;
        leaves_tree[lo + 2] = d1_val_j;
        leaves_tree[lo + 3] = d1_err_j;
        leaves_tree[lo + 4] = d0_val_j;
        leaves_tree[lo + 5] = d0_err_i;
    }
}

template <bool UseConservativeTile>
void poly_taylor_transform_jerk_batch(std::span<double> leaves_tree,
                                      std::span<SizeType> indices_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto dt        = t0_next - t0_cur;
    const auto dt2       = dt * dt;
    const auto half_dt2  = 0.5 * (dt2);
    const auto sixth_dt3 = dt2 * dt / 6.0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo       = indices_tree[i] * kLeavesStride;
        const auto d3_val_i = leaves_tree[lo + 0];
        const auto d3_err_i = leaves_tree[lo + 1];
        const auto d2_val_i = leaves_tree[lo + 2];
        const auto d2_err_i = leaves_tree[lo + 3];
        const auto d1_val_i = leaves_tree[lo + 4];
        const auto d1_err_i = leaves_tree[lo + 5];
        const auto d0_val_i = leaves_tree[lo + 6];
        const auto d0_err_i = leaves_tree[lo + 7];

        const auto d3_val_j = d3_val_i;
        const auto d2_val_j = d2_val_i + (d3_val_i * dt);
        const auto d1_val_j =
            d1_val_i + (d2_val_i * dt) + (d3_val_i * half_dt2);
        const auto d0_val_j = d0_val_i + (d1_val_i * dt) +
                              (d2_val_i * half_dt2) + (d3_val_i * sixth_dt3);

        double d3_err_j, d2_err_j, d1_err_j;
        if constexpr (UseConservativeTile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            d3_err_j = d3_err_i;
            d2_err_j =
                std::sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt2));
            d1_err_j =
                std::sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt2) +
                          (d3_err_i * d3_err_i * half_dt2 * half_dt2));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            d3_err_j = d3_err_i;
            d2_err_j = d2_err_i;
            d1_err_j = d1_err_i;
        }

        // Write back transformed values
        leaves_tree[lo + 0] = d3_val_j;
        leaves_tree[lo + 1] = d3_err_j;
        leaves_tree[lo + 2] = d2_val_j;
        leaves_tree[lo + 3] = d2_err_j;
        leaves_tree[lo + 4] = d1_val_j;
        leaves_tree[lo + 5] = d1_err_j;
        leaves_tree[lo + 6] = d0_val_j;
        leaves_tree[lo + 7] = d0_err_i;
    }
}

template <bool UseConservativeTile>
void poly_taylor_transform_snap_batch(std::span<double> leaves_tree,
                                      std::span<SizeType> indices_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto dt                = t0_next - t0_cur;
    const auto dt2               = dt * dt;
    const auto half_dt2          = 0.5 * (dt2);
    const auto sixth_dt3         = dt2 * dt / 6.0;
    const auto twenty_fourth_dt4 = dt2 * dt2 / 24.0;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo       = indices_tree[i] * kLeavesStride;
        const auto d4_val_i = leaves_tree[lo + 0];
        const auto d4_err_i = leaves_tree[lo + 1];
        const auto d3_val_i = leaves_tree[lo + 2];
        const auto d3_err_i = leaves_tree[lo + 3];
        const auto d2_val_i = leaves_tree[lo + 4];
        const auto d2_err_i = leaves_tree[lo + 5];
        const auto d1_val_i = leaves_tree[lo + 6];
        const auto d1_err_i = leaves_tree[lo + 7];
        const auto d0_val_i = leaves_tree[lo + 8];
        const auto d0_err_i = leaves_tree[lo + 9];

        const auto d4_val_j = d4_val_i;
        const auto d3_val_j = d3_val_i + (d4_val_i * dt);
        const auto d2_val_j =
            d2_val_i + (d3_val_i * dt) + (d4_val_i * half_dt2);
        const auto d1_val_j = d1_val_i + (d2_val_i * dt) +
                              (d3_val_i * half_dt2) + (d4_val_i * sixth_dt3);
        const auto d0_val_j = d0_val_i + (d1_val_i * dt) +
                              (d2_val_i * half_dt2) + (d3_val_i * sixth_dt3) +
                              (d4_val_i * twenty_fourth_dt4);

        double d4_err_j, d3_err_j, d2_err_j, d1_err_j;
        if constexpr (UseConservativeTile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            d4_err_j = d4_err_i;
            d3_err_j =
                std::sqrt((d3_err_i * d3_err_i) + (d4_err_i * d4_err_i * dt2));
            d2_err_j =
                std::sqrt((d2_err_i * d2_err_i) + (d3_err_i * d3_err_i * dt2) +
                          (d4_err_i * d4_err_i * half_dt2 * half_dt2));
            d1_err_j =
                std::sqrt((d1_err_i * d1_err_i) + (d2_err_i * d2_err_i * dt2) +
                          (d3_err_i * d3_err_i * half_dt2 * half_dt2) +
                          (d4_err_i * d4_err_i * sixth_dt3 * sixth_dt3));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            d4_err_j = d4_err_i;
            d3_err_j = d3_err_i;
            d2_err_j = d2_err_i;
            d1_err_j = d1_err_i;
        }

        // Write back transformed values
        leaves_tree[lo + 0] = d4_val_j;
        leaves_tree[lo + 1] = d4_err_j;
        leaves_tree[lo + 2] = d3_val_j;
        leaves_tree[lo + 3] = d3_err_j;
        leaves_tree[lo + 4] = d2_val_j;
        leaves_tree[lo + 5] = d2_err_j;
        leaves_tree[lo + 6] = d1_val_j;
        leaves_tree[lo + 7] = d1_err_j;
        leaves_tree[lo + 8] = d0_val_j;
        leaves_tree[lo + 9] = d0_err_i;
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
                                       memory::BranchingWorkspace& branch_ws) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Taylor order");
    if constexpr (NPARAMS == 2) {
        return poly_taylor_branch_accel_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, branch_ws);
    } else if constexpr (NPARAMS == 3) {
        return poly_taylor_branch_jerk_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, branch_ws);
    } else if constexpr (NPARAMS == 4) {
        return poly_taylor_branch_snap_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, branch_ws);
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
                                      std::span<SizeType> indices_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Taylor order");
    if constexpr (NPARAMS == 2) {
        poly_taylor_transform_accel_batch<UseConservativeTile>(
            leaves_tree, indices_tree, coord_next, coord_cur, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_taylor_transform_jerk_batch<UseConservativeTile>(
            leaves_tree, indices_tree, coord_next, coord_cur, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_taylor_transform_snap_batch<UseConservativeTile>(
            leaves_tree, indices_tree, coord_next, coord_cur, n_leaves);
    }
}

} // namespace

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

    psr_utils::poly_taylor_step_d_vec_limited(
        n_params, t_obs_minus_t_ref, nbins, eta, f0_batch, param_limits,
        dparam_new_batch, 0);
    psr_utils::poly_taylor_shift_d_vec(dparam_cur_batch, dparam_new_batch,
                                       t_obs_minus_t_ref, nbins, f0_batch, 0,
                                       shift_bins_batch, n_leaves, n_params);

    std::vector<double> pad_branched_dparams(n_leaves * n_params);
    std::vector<SizeType> branched_counts(n_leaves * n_params);
    // Optimized branching loop - same logic as original but vectorized access
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * leaves_stride;
        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType flat_idx     = (i * n_params) + j;
            const SizeType param_offset = lo + (j * kParamStride);
            const double param_cur_val  = leaves_batch[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];
            const SizeType pad_offset =
                (i * n_params * branch_max) + (j * branch_max);

            if (shift_bins_batch[flat_idx] >= (eta - utils::kEps)) {
                std::span<double> slice_span =
                    pad_branched_params.subspan(pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, param_cur_val, dparam_cur_val,
                    dparam_new_batch[flat_idx]);

                pad_branched_dparams[flat_idx] = dparam_act;
                branched_counts[flat_idx]      = count;
            } else {
                // No branching: only use current value
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
                                  memory::BranchingWorkspace& branch_ws) {

    auto dispatch = [&]<SizeType N>() {
        return poly_taylor_branch_batch_impl<N>(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, nbins, eta,
            param_limits, branch_max, n_leaves, branch_ws);
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
                                 std::span<SizeType> indices_tree,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 bool use_conservative_tile) {
    auto dispatch = [&]<SizeType N, bool C>() {
        return poly_taylor_transform_batch_impl<N, C>(
            leaves_tree, indices_tree, coord_next, coord_cur, n_leaves);
    };
    auto launch = [&](bool conservative) {
        switch (n_params) {
        case 2:
            conservative ? dispatch.template operator()<2, true>()
                         : dispatch.template operator()<2, false>();
            break;
        case 3:
            conservative ? dispatch.template operator()<3, true>()
                         : dispatch.template operator()<3, false>();
            break;
        case 4:
            conservative ? dispatch.template operator()<4, true>()
                         : dispatch.template operator()<4, false>();
            break;
        default:
            throw std::invalid_argument("Unsupported Taylor order");
        }
    };
    launch(use_conservative_tile);
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
generate_bp_poly_taylor_approx(std::span<const SizeType> param_grid_count_init,
                               std::span<const double> dparams_init,
                               std::span<const ParamLimit> param_limits,
                               double tseg_ffa,
                               SizeType nsegments,
                               SizeType nbins,
                               double eta,
                               SizeType ref_seg,
                               IndexType isuggest,
                               bool use_conservative_tile,
                               SizeType branch_max) {
    error_check::check_equal(param_grid_count_init.size(), param_limits.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_grid_count_init.size(), param_limits.size(),
        "param_grid_count_init and param_limits must have the same size");
    std::vector<double> branching_pattern(nsegments - 1);
    const auto n_params      = param_grid_count_init.size();
    const auto leaves_stride = (n_params + 2) * 2;
    std::vector<double> branch_leaves(branch_max * leaves_stride);
    std::vector<double> leaf_data(leaves_stride);

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
    // Copy isuggest-th leaf to leaf_data
    auto leaf = std::span(seed_leaves)
                    .subspan((leaves_stride * isuggest), leaves_stride);
    std::ranges::copy(leaf, leaf_data.begin());
    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next    = snail_scheme.get_coord(prune_level);
        const auto coord_cur     = snail_scheme.get_current_coord(prune_level);
        const auto batch_origins = poly_taylor_branch_batch_generic(
            leaf_data, coord_cur, branch_leaves, nbins, eta, param_limits,
            branch_max, 1, n_params);
        const auto n_leaves_branch = batch_origins.size();
        auto leaves_span =
            std::span(branch_leaves).first(n_leaves_branch * leaves_stride);
        std::vector<SizeType> indices_branch(n_leaves_branch);
        std::iota(indices_branch.begin(), indices_branch.end(), 0U);
        branching_pattern[prune_level - 1] =
            static_cast<double>(n_leaves_branch);
        poly_taylor_transform_batch(leaves_span, indices_branch, coord_next,
                                    coord_cur, n_leaves_branch, n_params,
                                    use_conservative_tile);
        // Copy first leaf to leaf_data
        auto first_leaf_span = std::span(branch_leaves).first(leaves_stride);
        std::ranges::copy(first_leaf_span, leaf_data.begin());
    }
    // Check if any branches is truncated due to branch_max
    if (std::ranges::any_of(branching_pattern, [branch_max](double value) {
            return static_cast<SizeType>(value) == branch_max;
        })) {
        throw std::runtime_error("Branching pattern is truncated due to "
                                 "branch_max. Increase branch_max.");
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
        psr_utils::poly_taylor_step_d_vec_limited(
            n_params, t_obs_minus_t_ref, nbins, eta, f0_batch, param_limits,
            dparam_new_batch, 0);
        psr_utils::poly_taylor_shift_d_vec(
            dparam_cur_batch, dparam_new_batch, t_obs_minus_t_ref, nbins,
            f0_batch, 0, shift_bins_batch, n_freqs, n_params);

        std::ranges::fill(n_branches, 1.0);
        // Determine branching needs
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < n_params; ++j) {
                const auto idx = (i * n_params) + j;
                if (shift_bins_batch[idx] < (eta - utils::kEps)) {
                    dparam_cur_next[idx] = dparam_cur_batch[idx];
                    continue;
                }
                const auto ratio      = (dparam_cur_batch[idx] + utils::kEps) /
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
