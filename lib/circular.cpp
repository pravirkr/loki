#include "loki/core/circular.hpp"

#include <algorithm>
#include <numbers>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"
#include "loki/utils.hpp"

namespace loki::core {

std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask(std::span<const double> leaves_batch,
                     SizeType n_leaves,
                     SizeType n_params,
                     double minimum_snap_cells) {
    constexpr SizeType kParamsExpected = 5U;
    constexpr SizeType kParamStride    = 2U;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;
    constexpr double kEps              = 1e-12;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 5 for circular orbit resolve");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    std::vector<SizeType> idx_circular_snap;
    std::vector<SizeType> idx_circular_crackle;
    std::vector<SizeType> idx_taylor;

    // Reserve space for efficiency
    idx_circular_snap.reserve(n_leaves / 3);
    idx_circular_crackle.reserve(n_leaves / 3);
    idx_taylor.reserve(n_leaves / 3);

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        // Extract values from leaves_batch
        const auto crackle  = leaves_batch[leaf_offset + 0];
        const auto dcrackle = leaves_batch[leaf_offset + 1];
        const auto snap     = leaves_batch[leaf_offset + 2];
        const auto dsnap    = leaves_batch[leaf_offset + 3];
        const auto jerk     = leaves_batch[leaf_offset + 4];
        const auto accel    = leaves_batch[leaf_offset + 6];

        // Compute significance tests
        const double snap_threshold    = minimum_snap_cells * (dsnap + kEps);
        const double crackle_threshold = minimum_snap_cells * (dcrackle + kEps);

        const bool is_sig_snap    = std::abs(snap) > snap_threshold;
        const bool is_sig_crackle = std::abs(crackle) > crackle_threshold;

        // Snap-Dominated Region: Check if implied Omega^2 = -snap/accel is
        // physical
        const bool is_physical_snap =
            ((-snap * accel) > 0.0) && (std::abs(accel) > kEps);
        const bool mask_circular_snap = is_sig_snap && is_physical_snap;

        // Crackle-Dominated Region (The Hole): snap weak, crackle strong
        const bool in_the_hole = (!is_sig_snap) && is_sig_crackle;

        // Check if implied Omega^2 = -crackle/jerk is physical
        const bool is_physical_crackle =
            ((-crackle * jerk) > 0.0) && (std::abs(jerk) > kEps);
        const bool mask_circular_crackle = in_the_hole && is_physical_crackle;

        // Classify
        if (mask_circular_snap) {
            idx_circular_snap.push_back(i);
        } else if (mask_circular_crackle) {
            idx_circular_crackle.push_back(i);
        } else {
            idx_taylor.push_back(i);
        }
    }
    return {std::move(idx_circular_snap), std::move(idx_circular_crackle),
            std::move(idx_taylor)};
}

namespace {
inline bool is_in_hole(double d5,
                       double d4,
                       double d3,
                       double d5_sig,
                       double d4_sig,
                       double minimum_snap_cells) noexcept {
    constexpr double kEps          = 1e-12;
    const double snap_threshold    = minimum_snap_cells * (d4_sig + kEps);
    const double crackle_threshold = minimum_snap_cells * (d5_sig + kEps);
    const bool is_sig_snap         = std::abs(d4) > snap_threshold;
    const bool is_sig_crackle      = std::abs(d5) > crackle_threshold;
    bool in_the_hole               = (!is_sig_snap) && is_sig_crackle;
    const bool is_physical_crackle =
        ((-d5 * d3) > 0.0) && (std::abs(d3) > kEps);
    return in_the_hole && is_physical_crackle;
}

} // namespace

SizeType
circ_taylor_branch_batch(std::span<const double> leaves_tree,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch,
                         std::span<SizeType> leaves_origins,
                         SizeType n_leaves,
                         SizeType nbins,
                         double eta,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max,
                         double minimum_snap_cells,
                         utils::BranchingWorkspaceView ws) {
    constexpr SizeType kParams       = 5U;
    constexpr SizeType kParamStride  = 2U;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;
    constexpr double kEps            = 1e-12;

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
    const double dt5     = dt4 * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const double inv_dt4 = inv_dt2 * inv_dt2;
    const double inv_dt5 = inv_dt4 * inv_dt;
    const auto nbins_d   = static_cast<double>(nbins);
    const double dphi    = eta / nbins_d;

    // Use batch_leaves memory as workspace. Partition workspace into sections:
    const SizeType workspace_size = leaves_branch.size();
    const SizeType batch_size     = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new = leaves_branch.subspan(0, batch_size);
    std::span<double> shift_bins =
        leaves_branch.subspan(batch_size, batch_size);
    const auto workspace_acquired_size = (batch_size * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    const double* __restrict leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict leaves_origins_ptr  = leaves_origins.data();
    double* __restrict leaves_branch_ptr     = leaves_branch.data();
    double* __restrict dparam_new_ptr        = dparam_new.data();
    double* __restrict shift_bins_ptr        = shift_bins.data();

    // --- Loop 1: step + shift (vectorizable) ---
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * kParams;

        const auto d5_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d4_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto d3_sig_cur = leaves_tree_ptr[leaf_offset + 5];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 7];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 9];
        const auto f0         = leaves_tree_ptr[leaf_offset + 12];

        const auto dfactor    = utils::kCval / f0;
        const auto d5_sig_new = dphi * dfactor * 1920.0 * inv_dt5;
        const auto d4_sig_new = dphi * dfactor * 192.0 * inv_dt4;
        const auto d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        dparam_new_ptr[flat_base + 0] = d5_sig_new;
        dparam_new_ptr[flat_base + 1] = d4_sig_new;
        dparam_new_ptr[flat_base + 2] = d3_sig_new;
        dparam_new_ptr[flat_base + 3] = d2_sig_new;
        dparam_new_ptr[flat_base + 4] = d1_sig_new;

        shift_bins_ptr[flat_base + 0] =
            (d5_sig_cur - d5_sig_new) * dt5 * nbins_d / (1920.0 * dfactor);
        shift_bins_ptr[flat_base + 1] =
            (d4_sig_cur - d4_sig_new) * dt4 * nbins_d / (192.0 * dfactor);
        shift_bins_ptr[flat_base + 2] =
            (d3_sig_cur - d3_sig_new) * dt3 * nbins_d / (24.0 * dfactor);
        shift_bins_ptr[flat_base + 3] =
            (d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[flat_base + 4] =
            (d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // --- Early Exit: Check if any leaf needs branching ---
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - kEps)) {
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

        const auto d5_cur     = leaves_tree_ptr[leaf_offset + 0];
        const auto d5_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d4_cur     = leaves_tree_ptr[leaf_offset + 2];
        const auto d4_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto d3_cur     = leaves_tree_ptr[leaf_offset + 4];
        const auto d3_sig_cur = leaves_tree_ptr[leaf_offset + 5];
        const auto d2_cur     = leaves_tree_ptr[leaf_offset + 6];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 7];
        const auto d1_cur     = leaves_tree_ptr[leaf_offset + 8];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 9];
        const auto f0         = leaves_tree_ptr[leaf_offset + 12];
        const auto d4_sig_new = dparam_new_ptr[flat_base + 1];
        const auto d3_sig_new = dparam_new_ptr[flat_base + 2];
        const auto d2_sig_new = dparam_new_ptr[flat_base + 3];
        const auto d1_sig_new = dparam_new_ptr[flat_base + 4];

        // Branch d5 parameter (no branching as of yet)
        {
            const SizeType pad_offset         = (flat_base + 0) * branch_max;
            ws.scratch_params[pad_offset]     = d5_cur;
            ws.scratch_dparams[flat_base + 0] = d5_sig_cur;
            ws.scratch_counts[flat_base + 0]  = 1;
        }

        // Branch d4-d1 parameters
        psr_utils::branch_one_param_padded(
            1, d4_cur, d4_sig_cur, d4_sig_new, param_limits[1][0],
            param_limits[1][1], eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        psr_utils::branch_one_param_padded(
            2, d3_cur, d3_sig_cur, d3_sig_new, param_limits[2][0],
            param_limits[2][1], eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);
        psr_utils::branch_one_param_padded(
            3, d2_cur, d2_sig_cur, d2_sig_new, param_limits[3][0],
            param_limits[3][1], eta, shift_bins_ptr, ws.scratch_params,
            ws.scratch_dparams, ws.scratch_counts, flat_base, branch_max);

        const double d1_min = (1 - param_limits[4][1] / f0) * utils::kCval;
        const double d1_max = (1 - param_limits[4][0] / f0) * utils::kCval;
        psr_utils::branch_one_param_padded(
            4, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max, eta,
            shift_bins_ptr, ws.scratch_params, ws.scratch_dparams,
            ws.scratch_counts, flat_base, branch_max);
    }

    // --- Check if crackle branching is needed ---
    bool any_crackle_branching = false;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (shift_bins_ptr[(i * kParams) + 0] >= (eta - kEps)) {
            any_crackle_branching = true;
            break;
        }
    }

    SizeType out_leaves = 0;
    if (!any_crackle_branching) {
        // FAST PATH: No crackle branching, simple 4-nested loops
        for (SizeType i = 0; i < n_leaves; ++i) {
            const SizeType leaf_offset = i * kLeavesStride;
            const SizeType flat_base   = i * kParams;
            // n_d5_branches = 1;
            const SizeType n_d4_branches = ws.scratch_counts[flat_base + 1];
            const SizeType n_d3_branches = ws.scratch_counts[flat_base + 2];
            const SizeType n_d2_branches = ws.scratch_counts[flat_base + 3];
            const SizeType n_d1_branches = ws.scratch_counts[flat_base + 4];

            const SizeType d5_offset = (flat_base + 0) * branch_max;
            const SizeType d4_offset = (flat_base + 1) * branch_max;
            const SizeType d3_offset = (flat_base + 2) * branch_max;
            const SizeType d2_offset = (flat_base + 3) * branch_max;
            const SizeType d1_offset = (flat_base + 4) * branch_max;

            for (SizeType b = 0; b < n_d4_branches; ++b) {
                for (SizeType c = 0; c < n_d3_branches; ++c) {
                    for (SizeType d = 0; d < n_d2_branches; ++d) {
                        for (SizeType e = 0; e < n_d1_branches; ++e) {
                            const SizeType bo = out_leaves * kLeavesStride;
                            leaves_branch_ptr[bo + 0] =
                                ws.scratch_params[d5_offset];
                            leaves_branch_ptr[bo + 1] =
                                ws.scratch_dparams[flat_base + 0];
                            leaves_branch_ptr[bo + 2] =
                                ws.scratch_params[d4_offset + b];
                            leaves_branch_ptr[bo + 3] =
                                ws.scratch_dparams[flat_base + 1];
                            leaves_branch_ptr[bo + 4] =
                                ws.scratch_params[d3_offset + c];
                            leaves_branch_ptr[bo + 5] =
                                ws.scratch_dparams[flat_base + 2];
                            leaves_branch_ptr[bo + 6] =
                                ws.scratch_params[d2_offset + d];
                            leaves_branch_ptr[bo + 7] =
                                ws.scratch_dparams[flat_base + 3];
                            leaves_branch_ptr[bo + 8] =
                                ws.scratch_params[d1_offset + e];
                            leaves_branch_ptr[bo + 9] =
                                ws.scratch_dparams[flat_base + 4];
                            // Copy d0 and f0
                            std::memcpy(leaves_branch_ptr + bo + 10,
                                        leaves_tree_ptr + leaf_offset + 10,
                                        4 * sizeof(double));

                            leaves_origins_ptr[out_leaves] = i;
                            ++out_leaves;
                        }
                    }
                }
            }
        }
    } else {
        // COMPLEX PATH: Crackle branching needed, inline hole checking
        for (SizeType i = 0; i < n_leaves; ++i) {
            const SizeType leaf_offset = i * kLeavesStride;
            const SizeType flat_base   = i * kParams;

            const bool needs_crackle =
                shift_bins_ptr[flat_base + 0] >= (eta - kEps);

            const SizeType n_d4_branches = ws.scratch_counts[flat_base + 1];
            const SizeType n_d3_branches = ws.scratch_counts[flat_base + 2];
            const SizeType n_d2_branches = ws.scratch_counts[flat_base + 3];
            const SizeType n_d1_branches = ws.scratch_counts[flat_base + 4];

            const SizeType d5_offset = (flat_base + 0) * branch_max;
            const SizeType d4_offset = (flat_base + 1) * branch_max;
            const SizeType d3_offset = (flat_base + 2) * branch_max;
            const SizeType d2_offset = (flat_base + 3) * branch_max;
            const SizeType d1_offset = (flat_base + 4) * branch_max;

            if (!needs_crackle) {
                for (SizeType b = 0; b < n_d4_branches; ++b) {
                    for (SizeType c = 0; c < n_d3_branches; ++c) {
                        for (SizeType d = 0; d < n_d2_branches; ++d) {
                            for (SizeType e = 0; e < n_d1_branches; ++e) {
                                const SizeType bo = out_leaves * kLeavesStride;
                                leaves_branch_ptr[bo + 0] =
                                    ws.scratch_params[d5_offset];
                                leaves_branch_ptr[bo + 1] =
                                    ws.scratch_dparams[flat_base + 0];
                                leaves_branch_ptr[bo + 2] =
                                    ws.scratch_params[d4_offset + b];
                                leaves_branch_ptr[bo + 3] =
                                    ws.scratch_dparams[flat_base + 1];
                                leaves_branch_ptr[bo + 4] =
                                    ws.scratch_params[d3_offset + c];
                                leaves_branch_ptr[bo + 5] =
                                    ws.scratch_dparams[flat_base + 2];
                                leaves_branch_ptr[bo + 6] =
                                    ws.scratch_params[d2_offset + d];
                                leaves_branch_ptr[bo + 7] =
                                    ws.scratch_dparams[flat_base + 3];
                                leaves_branch_ptr[bo + 8] =
                                    ws.scratch_params[d1_offset + e];
                                leaves_branch_ptr[bo + 9] =
                                    ws.scratch_dparams[flat_base + 4];
                                std::memcpy(leaves_branch_ptr + bo + 10,
                                            leaves_tree_ptr + leaf_offset + 10,
                                            4 * sizeof(double));

                                leaves_origins_ptr[out_leaves] = i;
                                ++out_leaves;
                            }
                        }
                    }
                }
            } else {
                for (SizeType b = 0; b < n_d4_branches; ++b) {
                    for (SizeType c = 0; c < n_d3_branches; ++c) {
                        for (SizeType d = 0; d < n_d2_branches; ++d) {
                            for (SizeType e = 0; e < n_d1_branches; ++e) {
                                // Extract current combination
                                const double d5 = ws.scratch_params[d5_offset];
                                const double d4 =
                                    ws.scratch_params[d4_offset + b];
                                const double d3 =
                                    ws.scratch_params[d3_offset + c];
                                const double d2 =
                                    ws.scratch_params[d2_offset + d];
                                const double d1 =
                                    ws.scratch_params[d1_offset + e];
                                const double d5_sig =
                                    ws.scratch_dparams[flat_base + 0];
                                const double d4_sig =
                                    ws.scratch_dparams[flat_base + 1];
                                const double d3_sig =
                                    ws.scratch_dparams[flat_base + 2];
                                const double d2_sig =
                                    ws.scratch_dparams[flat_base + 3];
                                const double d1_sig =
                                    ws.scratch_dparams[flat_base + 4];

                                // Check if this combination is in "the hole"
                                bool in_hole =
                                    is_in_hole(d5, d4, d3, d5_sig, d4_sig,
                                               minimum_snap_cells);

                                if (in_hole) [[unlikely]] {
                                    // Branch d5 (crackle) for this combination
                                    const double d5_sig_new =
                                        dparam_new_ptr[flat_base + 0];
                                    const auto [pmin, pmax] = param_limits[0];
                                    auto slice_span         = std::span<double>(
                                        ws.scratch_params + d5_offset,
                                        branch_max);
                                    auto [dparam_act, count] =
                                        psr_utils::branch_param_padded(
                                            slice_span, d5, d5_sig, d5_sig_new,
                                            pmin, pmax);
                                    for (SizeType a = 0; a < count; ++a) {
                                        const SizeType bo =
                                            out_leaves * kLeavesStride;

                                        leaves_branch_ptr[bo + 0] =
                                            ws.scratch_params[d5_offset + a];
                                        leaves_branch_ptr[bo + 1] = dparam_act;
                                        leaves_branch_ptr[bo + 2] = d4;
                                        leaves_branch_ptr[bo + 3] = d4_sig;
                                        leaves_branch_ptr[bo + 4] = d3;
                                        leaves_branch_ptr[bo + 5] = d3_sig;
                                        leaves_branch_ptr[bo + 6] = d2;
                                        leaves_branch_ptr[bo + 7] = d2_sig;
                                        leaves_branch_ptr[bo + 8] = d1;
                                        leaves_branch_ptr[bo + 9] = d1_sig;

                                        std::memcpy(leaves_branch_ptr + bo + 10,
                                                    leaves_tree_ptr +
                                                        leaf_offset + 10,
                                                    4 * sizeof(double));

                                        leaves_origins_ptr[out_leaves] = i;
                                        ++out_leaves;
                                    }
                                } else [[likely]] {
                                    // Not in hole, write single leaf
                                    const SizeType bo =
                                        out_leaves * kLeavesStride;

                                    leaves_branch_ptr[bo + 0] = d5;
                                    leaves_branch_ptr[bo + 1] = d5_sig;
                                    leaves_branch_ptr[bo + 2] = d4;
                                    leaves_branch_ptr[bo + 3] = d4_sig;
                                    leaves_branch_ptr[bo + 4] = d3;
                                    leaves_branch_ptr[bo + 5] = d3_sig;
                                    leaves_branch_ptr[bo + 6] = d2;
                                    leaves_branch_ptr[bo + 7] = d2_sig;
                                    leaves_branch_ptr[bo + 8] = d1;
                                    leaves_branch_ptr[bo + 9] = d1_sig;
                                    std::memcpy(leaves_branch_ptr + bo + 10,
                                                leaves_tree_ptr + leaf_offset +
                                                    10,
                                                4 * sizeof(double));

                                    leaves_origins_ptr[out_leaves] = i;
                                    ++out_leaves;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    error_check::check_less_equal(out_leaves, n_leaves * branch_max,
                                  "out_leaves size mismatch");

    return out_leaves;
}

SizeType circ_taylor_validate_batch(std::span<double> leaves_branch,
                                    std::span<SizeType> leaves_origins,
                                    SizeType n_leaves,
                                    double p_orb_min,
                                    double x_mass_const,
                                    double minimum_snap_cells) {
    constexpr SizeType kParams       = 5U;
    constexpr SizeType kParamStride  = 2U;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;
    constexpr double kEps            = 1e-12;
    constexpr double kTwoThirds      = 2.0 / 3.0;

    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_branch size mismatch");

    const double omega_max_sq = std::pow(2.0 * std::numbers::pi / p_orb_min, 2);
    std::vector<bool> mask_keep(n_leaves, false);

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        // Extract values
        const double crackle  = leaves_branch[leaf_offset + 0];
        const double dcrackle = leaves_branch[leaf_offset + 1];
        const double snap     = leaves_branch[leaf_offset + 2];
        const double dsnap    = leaves_branch[leaf_offset + 3];
        const double jerk     = leaves_branch[leaf_offset + 4];
        const double accel    = leaves_branch[leaf_offset + 6];

        // Classification thresholds
        const double snap_threshold    = minimum_snap_cells * (dsnap + kEps);
        const double crackle_threshold = minimum_snap_cells * (dcrackle + kEps);

        const bool is_sig_snap    = std::abs(snap) > snap_threshold;
        const bool is_sig_crackle = std::abs(crackle) > crackle_threshold;

        // 1. Noise (Unresolved Taylor cells) - Always keep
        const bool is_noise = (!is_sig_snap) && (!is_sig_crackle);
        if (is_noise) {
            mask_keep[i] = true;
            continue;
        }

        // 2. Snap-Dominated Region
        if (is_sig_snap) {
            const double omega_sq = -snap / (accel + kEps);

            // Check: Physical Sign (-d4/d2 > 0)
            const bool valid_sign =
                (omega_sq > 0.0) && (std::abs(accel) > kEps);

            // Check: Max Orbital Frequency
            const bool valid_omega = omega_sq < omega_max_sq;

            // |d2| < x * omega^(4/3)
            const double omega_sq_safe = std::abs(omega_sq);
            const double limit_accel =
                x_mass_const * std::pow(omega_sq_safe, kTwoThirds);
            const bool valid_accel = std::abs(accel) < limit_accel;

            // |d3| < |d2| * omega  =>  d3^2 < d2^2 * omega^2
            const bool valid_jerk =
                (jerk * jerk) < (accel * accel * omega_sq_safe);

            mask_keep[i] =
                valid_sign && valid_omega && valid_accel && valid_jerk;
            continue;
        }

        // 3. Crackle-Dominated Region (The Hole)
        // Only if snap is NOT significant but crackle IS significant
        const bool is_hole = (!is_sig_snap) && is_sig_crackle;
        if (is_hole) {
            const double omega_sq = -crackle / (jerk + kEps);

            // Check: Physical Sign (-d5/d3 > 0)
            const bool valid_sign = (omega_sq > 0.0) && (std::abs(jerk) > kEps);

            // Check: Max Orbital Frequency
            const bool valid_omega = omega_sq < omega_max_sq;

            // |d2| < x * omega^(4/3)
            const double omega_sq_safe = std::abs(omega_sq);
            const double limit_accel =
                x_mass_const * std::pow(omega_sq_safe, kTwoThirds);
            const bool valid_accel = std::abs(accel) < limit_accel;

            // |d3| < limit_accel * omega  =>  d3^2 < limit_accel^2 * omega^2
            const bool valid_jerk =
                (jerk * jerk) < (limit_accel * limit_accel * omega_sq_safe);

            mask_keep[i] =
                valid_sign && valid_omega && valid_accel && valid_jerk;
        }
    }

    // Compact arrays in-place
    SizeType write_idx = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (mask_keep[i]) {
            if (write_idx != i) {
                // Copy batch data
                const SizeType src_offset = i * kLeavesStride;
                const SizeType dst_offset = write_idx * kLeavesStride;
                std::copy_n(
                    leaves_branch.begin() + static_cast<IndexType>(src_offset),
                    kLeavesStride,
                    leaves_branch.begin() + static_cast<IndexType>(dst_offset));
                // Copy origin
                leaves_origins[write_idx] = leaves_origins[i];
            }
            ++write_idx;
        }
    }
    // write_idx is the new number of valid leaves
    return write_idx;
}

void circ_taylor_resolve_batch(std::span<const double> leaves_branch,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_cur,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               std::span<SizeType> param_indices,
                               std::span<float> phase_shift,
                               SizeType nbins,
                               SizeType n_leaves,
                               double minimum_snap_cells) {
    constexpr SizeType kParams       = 5;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(param_arr.size(), kParams,
                             "param_arr should have 5 parameters");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access to parameter grids
    const auto& accel_arr_grid = param_arr[3];
    const auto& freq_arr_grid  = param_arr[4];
    const auto n_freq          = param_arr[4].size();

    const double inv_c_val  = 1.0 / utils::kCval;
    const auto delta_t_add  = t0_add - t0_cur;
    const auto delta_t_init = t0_init - t0_cur;
    const auto delta_t      = delta_t_add - delta_t_init;

    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circ_taylor_mask(leaves_branch, n_leaves, kParams,
                             minimum_snap_cells);

    SizeType hint_a = 0, hint_f = 0;
    // Process circular indices
    for (SizeType i : idx_circular_snap) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto s_t_cur = leaves_branch[leaf_offset + (1 * kParamStride)];
        const auto j_t_cur = leaves_branch[leaf_offset + (2 * kParamStride)];
        const auto a_t_cur = leaves_branch[leaf_offset + (3 * kParamStride)];
        const auto v_t_cur = leaves_branch[leaf_offset + (4 * kParamStride)];
        const auto f0      = leaves_branch[leaf_offset + (6 * kParamStride)];

        // Circular orbit mask condition
        const auto omega_orb_sq = -s_t_cur / a_t_cur;
        const auto omega_orb    = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const auto omega_dt_add = omega_orb * delta_t_add;
        const auto cos_odt_add  = std::cos(omega_dt_add);
        const auto sin_odt_add  = std::sin(omega_dt_add);
        const auto a_t_add =
            (a_t_cur * cos_odt_add) + ((j_t_cur / omega_orb) * sin_odt_add);
        const auto j_t_add =
            (j_t_cur * cos_odt_add) - ((a_t_cur * omega_orb) * sin_odt_add);

        const auto omega_dt_init = omega_orb * delta_t_init;
        const auto cos_odt_init  = std::cos(omega_dt_init);
        const auto sin_odt_init  = std::sin(omega_dt_init);
        const auto a_t_init =
            (a_t_cur * cos_odt_init) + ((j_t_cur / omega_orb) * sin_odt_init);
        const auto j_t_init =
            (j_t_cur * cos_odt_init) - ((a_t_cur * omega_orb) * sin_odt_init);

        const auto a_new = a_t_add;
        const auto delta_v_new =
            (-j_t_add / omega_orb_sq) - (-j_t_init / omega_orb_sq);
        const auto delta_d_new = (-a_t_add / omega_orb_sq) -
                                 (-a_t_init / omega_orb_sq) +
                                 ((v_t_cur + j_t_cur / omega_orb_sq) * delta_t);
        const auto f_new     = f0 * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        phase_shift[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices (only need accel and freq)
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);

        param_indices[i] = (idx_a * n_freq) + idx_f;
    }

    // Reset hints
    hint_a = 0;
    hint_f = 0;
    // Process circular crackle indices
    for (SizeType i : idx_circular_crackle) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto c_t_cur = leaves_branch[leaf_offset + (0 * kParamStride)];
        const auto j_t_cur = leaves_branch[leaf_offset + (2 * kParamStride)];
        const auto a_t_cur = leaves_branch[leaf_offset + (3 * kParamStride)];
        const auto v_t_cur = leaves_branch[leaf_offset + (4 * kParamStride)];
        const auto f0      = leaves_branch[leaf_offset + (6 * kParamStride)];

        // Circular orbit mask condition
        const auto omega_orb_sq = -c_t_cur / a_t_cur;
        const auto omega_orb    = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const auto omega_dt_add = omega_orb * delta_t_add;
        const auto cos_odt_add  = std::cos(omega_dt_add);
        const auto sin_odt_add  = std::sin(omega_dt_add);
        const auto a_t_add =
            (a_t_cur * cos_odt_add) + ((j_t_cur / omega_orb) * sin_odt_add);
        const auto j_t_add =
            (j_t_cur * cos_odt_add) - ((a_t_cur * omega_orb) * sin_odt_add);

        const auto omega_dt_init = omega_orb * delta_t_init;
        const auto cos_odt_init  = std::cos(omega_dt_init);
        const auto sin_odt_init  = std::sin(omega_dt_init);
        const auto a_t_init =
            (a_t_cur * cos_odt_init) + ((j_t_cur / omega_orb) * sin_odt_init);
        const auto j_t_init =
            (j_t_cur * cos_odt_init) - ((a_t_cur * omega_orb) * sin_odt_init);

        const auto a_new = a_t_add;
        const auto delta_v_new =
            (-j_t_add / omega_orb_sq) - (-j_t_init / omega_orb_sq);
        const auto delta_d_new = (-a_t_add / omega_orb_sq) -
                                 (-a_t_init / omega_orb_sq) +
                                 ((v_t_cur + j_t_cur / omega_orb_sq) * delta_t);
        const auto f_new     = f0 * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        phase_shift[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices (only need accel and freq)
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);

        param_indices[i] = (idx_a * n_freq) + idx_f;
    }

    // Reset hints
    hint_a = 0;
    hint_f = 0;

    // Pre-compute constants to avoid repeated calculations
    const auto half_delta_t_add_sq  = 0.5 * delta_t_add * delta_t_add;
    const auto half_delta_t_init_sq = 0.5 * delta_t_init * delta_t_init;
    const auto sixth_delta_t_add_cubed =
        half_delta_t_add_sq * delta_t_add / 3.0;
    const auto sixth_delta_t_init_cubed =
        half_delta_t_init_sq * delta_t_init / 3.0;
    const auto half_delta_t_sq = half_delta_t_add_sq - half_delta_t_init_sq;
    const auto sixth_delta_t_cubed =
        sixth_delta_t_add_cubed - sixth_delta_t_init_cubed;
    const auto twentyfourth_delta_t_fourth =
        (sixth_delta_t_add_cubed * delta_t_add -
         sixth_delta_t_init_cubed * delta_t_init) /
        4.0;
    const auto onehundredtwenty_delta_t_fifth =
        (sixth_delta_t_add_cubed * half_delta_t_add_sq -
         sixth_delta_t_init_cubed * half_delta_t_init_sq) /
        10.0;
    // Process taylor indices
    for (SizeType i : idx_taylor) {
        const SizeType batch_offset = i * kLeavesStride;

        const auto c_t_cur = leaves_branch[batch_offset + (0 * kParamStride)];
        const auto s_t_cur = leaves_branch[batch_offset + (1 * kParamStride)];
        const auto j_t_cur = leaves_branch[batch_offset + (2 * kParamStride)];
        const auto a_t_cur = leaves_branch[batch_offset + (3 * kParamStride)];
        const auto v_t_cur = leaves_branch[batch_offset + (4 * kParamStride)];
        const auto f0      = leaves_branch[batch_offset + (6 * kParamStride)];
        const auto a_new   = a_t_cur + (j_t_cur * delta_t_add) +
                           (s_t_cur * half_delta_t_add_sq) +
                           (c_t_cur * sixth_delta_t_add_cubed);
        const auto delta_v_new = (a_t_cur * delta_t) +
                                 (j_t_cur * half_delta_t_sq) +
                                 (s_t_cur * sixth_delta_t_cubed) +
                                 (c_t_cur * twentyfourth_delta_t_fourth);
        const auto delta_d_new = (v_t_cur * delta_t) +
                                 (a_t_cur * half_delta_t_sq) +
                                 (j_t_cur * sixth_delta_t_cubed) +
                                 (s_t_cur * twentyfourth_delta_t_fourth) +
                                 (c_t_cur * onehundredtwenty_delta_t_fifth);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        phase_shift[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        param_indices[i] = (idx_a * n_freq) + idx_f;
    }
}

void circ_taylor_transform_batch(std::span<double> leaves_tree,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 bool use_conservative_tile,
                                 double minimum_snap_cells) {
    constexpr SizeType kParams       = 5;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto delta_t               = t0_next - t0_cur;

    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circ_taylor_mask(leaves_tree, n_leaves, kParams,
                             minimum_snap_cells);

    // Process circular indices
    for (SizeType i : idx_circular_snap) {
        const SizeType leaf_offset = i * kLeavesStride;

        const double d4_i     = leaves_tree[leaf_offset + 2];
        const double sig_d4_i = leaves_tree[leaf_offset + 3];
        const double d3_i     = leaves_tree[leaf_offset + 4];
        const double sig_d3_i = leaves_tree[leaf_offset + 5];
        const double d2_i     = leaves_tree[leaf_offset + 6];
        const double sig_d2_i = leaves_tree[leaf_offset + 7];
        const double d1_i     = leaves_tree[leaf_offset + 8];
        const double sig_d1_i = leaves_tree[leaf_offset + 9];
        const double d0_i     = leaves_tree[leaf_offset + 10];
        const double sig_d0_i = leaves_tree[leaf_offset + 11];

        const double omega_orb_sq = -d4_i / d2_i;
        const double omega_orb    = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const double omega_dt = omega_orb * delta_t;
        const double cos_odt  = std::cos(omega_dt);
        const double sin_odt  = std::sin(omega_dt);

        // Precompute some constants for efficiency
        const double inv_omega_orb          = 1.0 / omega_orb;
        const double inv_omega_orb_sq       = 1.0 / omega_orb_sq;
        const double sin_odt_inv_omega      = sin_odt * inv_omega_orb;
        const double d3_i_sin_odt_inv_omega = d3_i * sin_odt_inv_omega;
        const double d2_i_omega_sin_odt     = d2_i * omega_orb * sin_odt;

        // Pin-down omega using {d4, d2}
        const double d2_j = (d2_i * cos_odt) + d3_i_sin_odt_inv_omega;
        const double d3_j = (d3_i * cos_odt) - d2_i_omega_sin_odt;
        const double d4_j = -omega_orb_sq * d2_j;
        const double d5_j = -omega_orb_sq * d3_j;
        // Integrate to get {v, d}
        const double v_circ_i = -d3_i / omega_orb_sq;
        const double v_circ_j = -d3_j / omega_orb_sq;
        const double d1_diff  = d1_i - v_circ_i;
        const double d1_j     = v_circ_j + d1_diff;
        const double d_circ_j = -d2_j / omega_orb_sq;
        const double d_circ_i = -d2_i / omega_orb_sq;
        const double d0_j = d_circ_j + (d0_i - d_circ_i) + (d1_diff * delta_t);

        // Transform errors
        double sig_d5_j, sig_d4_j, sig_d3_j, sig_d2_j, sig_d1_j;
        if (use_conservative_tile) {
            const double omega_cu     = omega_orb_sq * omega_orb;
            const double inv_omega_cu = inv_omega_orb * inv_omega_orb_sq;
            const double var_d1_i     = sig_d1_i * sig_d1_i;
            const double var_d2_i     = sig_d2_i * sig_d2_i;
            const double var_d3_i     = sig_d3_i * sig_d3_i;
            const double var_d4_i     = sig_d4_i * sig_d4_i;

            const double u2 = (omega_dt * d3_j * inv_omega_orb_sq) -
                              (d3_i_sin_odt_inv_omega * inv_omega_orb);
            const double u3 = -(omega_dt * d2_j) - (d2_i * sin_odt);
            const double u4 = -(2 * omega_orb * d2_j) - (omega_orb_sq * u2);
            const double u5 = -(2 * omega_orb * d3_j) - (omega_orb_sq * u3);
            const double u1 =
                (2 * (d3_j - d3_i) * inv_omega_cu) - (u3 * inv_omega_orb_sq);
            const double v2 = -omega_orb / (2 * d2_i);
            const double v4 = omega_orb / (2 * d4_i);

            const double j52 = (omega_cu * sin_odt) + (u5 * v2);
            const double j53 = -omega_orb_sq * cos_odt;
            const double j54 = u5 * v4;
            sig_d5_j =
                std::sqrt((j52 * j52 * var_d2_i) + (j53 * j53 * var_d3_i) +
                          (j54 * j54 * var_d4_i));

            const double j42 = -(omega_orb_sq * cos_odt) + (u4 * v2);
            const double j43 = -omega_orb * sin_odt;
            const double j44 = u4 * v4;
            sig_d4_j =
                std::sqrt((j42 * j42 * var_d2_i) + (j43 * j43 * var_d3_i) +
                          (j44 * j44 * var_d4_i));

            const double j32 = -(omega_orb * sin_odt) + (u3 * v2);
            const double j33 = cos_odt;
            const double j34 = u3 * v4;
            sig_d3_j =
                std::sqrt((j32 * j32 * var_d2_i) + (j33 * j33 * var_d3_i) +
                          (j34 * j34 * var_d4_i));

            const double j22 = cos_odt + (u2 * v2);
            const double j23 = sin_odt / omega_orb;
            const double j24 = u2 * v4;
            sig_d2_j =
                std::sqrt((j22 * j22 * var_d2_i) + (j23 * j23 * var_d3_i) +
                          (j24 * j24 * var_d4_i));

            const double j11 = 1.0;
            const double j12 = sin_odt_inv_omega + (u1 * v2);
            const double j13 = (1 - cos_odt) * inv_omega_orb_sq;
            const double j14 = u1 * v4;
            sig_d1_j =
                std::sqrt((j11 * j11 * var_d1_i) + (j12 * j12 * var_d2_i) +
                          (j13 * j13 * var_d3_i) + (j14 * j14 * var_d4_i));

        } else {
            const double sig_d2_i_cos = cos_odt * sig_d2_i;
            const double sig_d3_i_cos = cos_odt * sig_d3_i;
            const double sig_d2_i_sin = sin_odt * sig_d2_i;
            const double sig_d3_i_sin = sin_odt * sig_d3_i;
            const double sig_d3_i_1mincos =
                (1 - cos_odt) * sig_d3_i / omega_orb_sq;
            sig_d2_j =
                std::sqrt((sig_d2_i_cos * sig_d2_i_cos) +
                          ((sig_d3_i_sin * sig_d3_i_sin) / omega_orb_sq));
            sig_d3_j =
                std::sqrt((sig_d3_i_cos * sig_d3_i_cos) +
                          ((sig_d2_i_sin * sig_d2_i_sin) * omega_orb_sq));
            sig_d1_j = std::sqrt(
                (sig_d1_i * sig_d1_i) + (sig_d3_i_1mincos * sig_d3_i_1mincos) +
                ((sig_d2_i_sin * sig_d2_i_sin) / omega_orb_sq));
            sig_d5_j = omega_orb_sq * sig_d3_j;
            sig_d4_j = omega_orb_sq * sig_d2_j;
        }

        // Write back transformed values
        leaves_tree[leaf_offset + 0]  = d5_j;
        leaves_tree[leaf_offset + 1]  = sig_d5_j;
        leaves_tree[leaf_offset + 2]  = d4_j;
        leaves_tree[leaf_offset + 3]  = sig_d4_j;
        leaves_tree[leaf_offset + 4]  = d3_j;
        leaves_tree[leaf_offset + 5]  = sig_d3_j;
        leaves_tree[leaf_offset + 6]  = d2_j;
        leaves_tree[leaf_offset + 7]  = sig_d2_j;
        leaves_tree[leaf_offset + 8]  = d1_j;
        leaves_tree[leaf_offset + 9]  = sig_d1_j;
        leaves_tree[leaf_offset + 10] = d0_j;
        leaves_tree[leaf_offset + 11] = sig_d0_i;
    }

    // Process circular crackle indices
    for (SizeType i : idx_circular_crackle) {
        const SizeType leaf_offset = i * kLeavesStride;

        const double d5_i     = leaves_tree[leaf_offset + 0];
        const double sig_d5_i = leaves_tree[leaf_offset + 1];
        const double d3_i     = leaves_tree[leaf_offset + 4];
        const double sig_d3_i = leaves_tree[leaf_offset + 5];
        const double d2_i     = leaves_tree[leaf_offset + 6];
        const double sig_d2_i = leaves_tree[leaf_offset + 7];
        const double d1_i     = leaves_tree[leaf_offset + 8];
        const double sig_d1_i = leaves_tree[leaf_offset + 9];
        const double d0_i     = leaves_tree[leaf_offset + 10];
        const double sig_d0_i = leaves_tree[leaf_offset + 11];

        const double omega_orb_sq = -d5_i / d3_i;
        const double omega_orb    = std::sqrt(omega_orb_sq);
        const double omega_dt     = omega_orb * delta_t;
        const double cos_odt      = std::cos(omega_dt);
        const double sin_odt      = std::sin(omega_dt);

        const double inv_omega_orb          = 1.0 / omega_orb;
        const double inv_omega_orb_sq       = 1.0 / omega_orb_sq;
        const double sin_odt_inv_omega      = sin_odt * inv_omega_orb;
        const double d3_i_sin_odt_inv_omega = d3_i * sin_odt_inv_omega;
        const double d2_i_omega_sin_odt     = d2_i * omega_orb * sin_odt;

        // Pin-down {s, j, a}
        const double d2_j = (d2_i * cos_odt) + d3_i_sin_odt_inv_omega;
        const double d3_j = (d3_i * cos_odt) - d2_i_omega_sin_odt;
        const double d4_j = -omega_orb_sq * d2_j;
        const double d5_j = -omega_orb_sq * d3_j;
        // Integrate to get {v, d}
        const double v_circ_i = -d3_i / omega_orb_sq;
        const double v_circ_j = -d3_j / omega_orb_sq;
        const double d1_diff  = d1_i - v_circ_i;
        const double d1_j     = v_circ_j + d1_diff;
        const double d_circ_j = -d2_j / omega_orb_sq;
        const double d_circ_i = -d2_i / omega_orb_sq;
        const double d0_j = d_circ_j + (d0_i - d_circ_i) + (d1_diff * delta_t);

        // Transform errors
        double sig_d5_j, sig_d4_j, sig_d3_j, sig_d2_j, sig_d1_j;
        if (use_conservative_tile) {
            const double omega_cu     = omega_orb_sq * omega_orb;
            const double inv_omega_cu = inv_omega_orb * inv_omega_orb_sq;
            const double var_d1_i     = sig_d1_i * sig_d1_i;
            const double var_d2_i     = sig_d2_i * sig_d2_i;
            const double var_d3_i     = sig_d3_i * sig_d3_i;
            const double var_d5_i     = sig_d5_i * sig_d5_i;

            const double u2 = (omega_dt * d3_j * inv_omega_orb_sq) -
                              (d3_i_sin_odt_inv_omega * inv_omega_orb);
            const double u3 = -(omega_dt * d2_j) - (d2_i * sin_odt);
            const double u4 = -(2 * omega_orb * d2_j) - (omega_orb_sq * u2);
            const double u5 = -(2 * omega_orb * d3_j) - (omega_orb_sq * u3);
            const double u1 =
                (2 * (d3_j - d3_i) * inv_omega_cu) - (u3 * inv_omega_orb_sq);
            const double v3 = -omega_orb / (2 * d3_i);
            const double v5 = omega_orb / (2 * d5_i);

            const double j52 = omega_cu * sin_odt;
            const double j53 = -(omega_orb_sq * cos_odt) + (u5 * v3);
            const double j55 = u5 * v5;
            sig_d5_j =
                std::sqrt((j52 * j52 * var_d2_i) + (j53 * j53 * var_d3_i) +
                          (j55 * j55 * var_d5_i));

            const double j42 = -omega_orb_sq * cos_odt;
            const double j43 = -(omega_orb * sin_odt) + (u4 * v3);
            const double j45 = u4 * v5;
            sig_d4_j =
                std::sqrt((j42 * j42 * var_d2_i) + (j43 * j43 * var_d3_i) +
                          (j45 * j45 * var_d5_i));

            const double j32 = -omega_orb * sin_odt;
            const double j33 = cos_odt + (u3 * v3);
            const double j35 = u3 * v5;
            sig_d3_j =
                std::sqrt((j32 * j32 * var_d2_i) + (j33 * j33 * var_d3_i) +
                          (j35 * j35 * var_d5_i));

            const double j22 = cos_odt;
            const double j23 = (sin_odt / omega_orb) + (u2 * v3);
            const double j25 = u2 * v5;
            sig_d2_j =
                std::sqrt((j22 * j22 * var_d2_i) + (j23 * j23 * var_d3_i) +
                          (j25 * j25 * var_d5_i));

            const double j11 = 1.0;
            const double j12 = sin_odt_inv_omega;
            const double j13 = ((1 - cos_odt) * inv_omega_orb_sq) + (u1 * v3);
            const double j15 = u1 * v5;
            sig_d1_j =
                std::sqrt((j11 * j11 * var_d1_i) + (j12 * j12 * var_d2_i) +
                          (j13 * j13 * var_d3_i) + (j15 * j15 * var_d5_i));

        } else {
            const double sig_d2_i_cos = cos_odt * sig_d2_i;
            const double sig_d3_i_cos = cos_odt * sig_d3_i;
            const double sig_d2_i_sin = sin_odt * sig_d2_i;
            const double sig_d3_i_sin = sin_odt * sig_d3_i;
            const double sig_d3_i_1mincos =
                (1 - cos_odt) * sig_d3_i / omega_orb_sq;
            sig_d2_j =
                std::sqrt((sig_d2_i_cos * sig_d2_i_cos) +
                          ((sig_d3_i_sin * sig_d3_i_sin) / omega_orb_sq));
            sig_d3_j =
                std::sqrt((sig_d3_i_cos * sig_d3_i_cos) +
                          ((sig_d2_i_sin * sig_d2_i_sin) * omega_orb_sq));
            sig_d1_j = std::sqrt(
                (sig_d1_i * sig_d1_i) + (sig_d3_i_1mincos * sig_d3_i_1mincos) +
                ((sig_d2_i_sin * sig_d2_i_sin) / omega_orb_sq));
            sig_d5_j = omega_orb_sq * sig_d3_j;
            sig_d4_j = omega_orb_sq * sig_d2_j;
        }

        // Write back transformed values
        leaves_tree[leaf_offset + 0]  = d5_j;
        leaves_tree[leaf_offset + 1]  = sig_d5_j;
        leaves_tree[leaf_offset + 2]  = d4_j;
        leaves_tree[leaf_offset + 3]  = sig_d4_j;
        leaves_tree[leaf_offset + 4]  = d3_j;
        leaves_tree[leaf_offset + 5]  = sig_d3_j;
        leaves_tree[leaf_offset + 6]  = d2_j;
        leaves_tree[leaf_offset + 7]  = sig_d2_j;
        leaves_tree[leaf_offset + 8]  = d1_j;
        leaves_tree[leaf_offset + 9]  = sig_d1_j;
        leaves_tree[leaf_offset + 10] = d0_j;
        leaves_tree[leaf_offset + 11] = sig_d0_i;
    }

    // Pre-compute constants to avoid repeated calculations
    const double delta_t_sq                   = delta_t * delta_t;
    const double delta_t_cubed                = delta_t_sq * delta_t;
    const double half_delta_t_sq              = 0.5 * delta_t_sq;
    const double sixth_delta_t_cubed          = delta_t_cubed / 6.0;
    const double twenty_fourth_delta_t_fourth = delta_t_cubed * delta_t / 24.0;
    const double onehundred_twenty_delta_t_fifth =
        delta_t_cubed * delta_t_sq / 120.0;
    // Process normal indices
    for (SizeType i : idx_taylor) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto c_val_i = leaves_tree[leaf_offset + 0];
        const auto c_err_i = leaves_tree[leaf_offset + 1];
        const auto s_val_i = leaves_tree[leaf_offset + 2];
        const auto s_err_i = leaves_tree[leaf_offset + 3];
        const auto j_val_i = leaves_tree[leaf_offset + 4];
        const auto j_err_i = leaves_tree[leaf_offset + 5];
        const auto a_val_i = leaves_tree[leaf_offset + 6];
        const auto a_err_i = leaves_tree[leaf_offset + 7];
        const auto v_val_i = leaves_tree[leaf_offset + 8];
        const auto v_err_i = leaves_tree[leaf_offset + 9];
        const auto d_val_i = leaves_tree[leaf_offset + 10];
        const auto d_err_i = leaves_tree[leaf_offset + 11];

        const auto c_val_j = c_val_i;
        const auto s_val_j = s_val_i + (c_val_i * delta_t);
        const auto j_val_j =
            j_val_i + (s_val_i * delta_t) + (c_val_i * half_delta_t_sq);
        const auto a_val_j = a_val_i + (j_val_i * delta_t) +
                             (s_val_i * half_delta_t_sq) +
                             (c_val_i * sixth_delta_t_cubed);
        const auto v_val_j = v_val_i + (a_val_i * delta_t) +
                             (j_val_i * half_delta_t_sq) +
                             (s_val_i * sixth_delta_t_cubed) +
                             (c_val_i * twenty_fourth_delta_t_fourth);
        const auto d_val_j = d_val_i + (v_val_i * delta_t) +
                             (a_val_i * half_delta_t_sq) +
                             (j_val_i * sixth_delta_t_cubed) +
                             (s_val_i * twenty_fourth_delta_t_fourth) +
                             (c_val_i * onehundred_twenty_delta_t_fifth);

        // Transform errors
        double c_err_j, s_err_j, j_err_j, a_err_j, v_err_j;
        if (use_conservative_tile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            c_err_j = c_err_i;
            s_err_j = std::sqrt((s_err_i * s_err_i) +
                                (c_err_i * c_err_i * delta_t_sq));
            j_err_j = std::sqrt(
                (j_err_i * j_err_i) + (s_err_i * s_err_i * delta_t_sq) +
                (c_err_i * c_err_i * half_delta_t_sq * half_delta_t_sq));
            a_err_j = std::sqrt(
                (a_err_i * a_err_i) + (j_err_i * j_err_i * delta_t_sq) +
                (s_err_i * s_err_i * half_delta_t_sq * half_delta_t_sq) +
                (c_err_i * c_err_i * sixth_delta_t_cubed *
                 sixth_delta_t_cubed));
            v_err_j = std::sqrt(
                (v_err_i * v_err_i) + (a_err_i * a_err_i * delta_t_sq) +
                (j_err_i * j_err_i * half_delta_t_sq * half_delta_t_sq) +
                (s_err_i * s_err_i * sixth_delta_t_cubed *
                 sixth_delta_t_cubed) +
                (c_err_i * c_err_i * twenty_fourth_delta_t_fourth *
                 twenty_fourth_delta_t_fourth));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            c_err_j = c_err_i;
            s_err_j = s_err_i;
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
        }

        // Write back transformed values
        leaves_tree[leaf_offset + 0]  = c_val_j;
        leaves_tree[leaf_offset + 1]  = c_err_j;
        leaves_tree[leaf_offset + 2]  = s_val_j;
        leaves_tree[leaf_offset + 3]  = s_err_j;
        leaves_tree[leaf_offset + 4]  = j_val_j;
        leaves_tree[leaf_offset + 5]  = j_err_j;
        leaves_tree[leaf_offset + 6]  = a_val_j;
        leaves_tree[leaf_offset + 7]  = a_err_j;
        leaves_tree[leaf_offset + 8]  = v_val_j;
        leaves_tree[leaf_offset + 9]  = v_err_j;
        leaves_tree[leaf_offset + 10] = d_val_j;
        leaves_tree[leaf_offset + 11] = d_err_i;
    }
}

std::vector<double>
generate_bp_circ_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        const std::vector<ParamLimitType>& param_limits,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        double p_orb_min,
                        double minimum_snap_cells,
                        bool use_conservative_tile) {
    constexpr double kEps              = 1e-12;
    constexpr SizeType kParamsExpected = 5;
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr must have 5 parameters");
    error_check::check_equal(dparams.size(), kParamsExpected,
                             "dparams must have 5 parameters");
    error_check::check_equal(param_limits.size(), kParamsExpected,
                             "param_limits must have 5 parameters");
    const auto poly_order = dparams.size();
    const auto& f0_batch  = param_arr.back(); // Last array is frequency
    const auto n_freqs    = f0_batch.size();  // Number of frequency bins

    // Snail Scheme
    psr_utils::MiddleOutScheme scheme(nsegments, ref_seg, tseg_ffa);
    std::vector<double> weights(n_freqs, 1.0);
    std::vector<double> branching_pattern(nsegments - 1);

    // Initialize dparam_cur_batch - each frequency gets the same dparams
    std::vector<double> dparam_cur_batch(n_freqs * poly_order);
    for (SizeType i = 0; i < n_freqs; ++i) {
        std::ranges::copy(dparams, dparam_cur_batch.begin() +
                                       static_cast<IndexType>(i * poly_order));
    }
    // f = f0(1 - v / C) => dv = -(C/f0) * df
    for (SizeType i = 0; i < n_freqs; ++i) {
        dparam_cur_batch[(i * poly_order) + poly_order - 1] =
            dparam_cur_batch[(i * poly_order) + poly_order - 1] *
            (utils::kCval / f0_batch[i]);
    }

    // Pre-compute parameter ranges
    std::vector<double> param_ranges(n_freqs * poly_order);
    for (SizeType i = 0; i < n_freqs; ++i) {
        for (SizeType j = 0; j < poly_order; ++j) {
            if (j == poly_order - 1) {
                const auto param_min =
                    (1 - param_limits[j][1] / f0_batch[i]) * utils::kCval;
                const auto param_max =
                    (1 - param_limits[j][0] / f0_batch[i]) * utils::kCval;
                param_ranges[(i * poly_order) + j] =
                    (param_max - param_min) / 2.0;
            } else {
                param_ranges[(i * poly_order) + j] =
                    (param_limits[j][1] - param_limits[j][0]) / 2.0;
            }
        }
    }
    // Track when first snap branching occurs for each frequency
    std::vector<bool> snap_first_branched(n_freqs, false);
    std::vector<bool> snap_active_mask(n_freqs, false);
    std::vector<double> dparam_new_batch(n_freqs * poly_order, 0.0);
    std::vector<double> shift_bins_batch(n_freqs * poly_order, 0.0);
    std::vector<double> dparam_cur_next(n_freqs * poly_order, 0.0);
    std::vector<double> n_branches(n_freqs, 1.0);
    const auto n_params = poly_order + 1;
    std::vector<double> dparam_d_vec(n_freqs * n_params, 0.0);
    std::vector<SizeType> idx_circular_snap;
    std::vector<SizeType> idx_taylor;
    idx_circular_snap.reserve(n_freqs);
    idx_taylor.reserve(n_freqs);

    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = scheme.get_coord(prune_level);
        const auto coord_cur  = scheme.get_current_coord(prune_level);
        const auto [t0_cur, t_obs_minus_t_ref] = coord_cur;

        // Calculate optimal parameter steps and shift bins
        psr_utils::poly_taylor_step_d_vec(poly_order, t_obs_minus_t_ref, nbins,
                                          eta, f0_batch, dparam_new_batch, 0);
        psr_utils::poly_taylor_shift_d_vec(
            dparam_cur_batch, dparam_new_batch, t_obs_minus_t_ref, nbins,
            f0_batch, 0, shift_bins_batch, n_freqs, poly_order);

        // Determine branching needs
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 1; j < poly_order; ++j) {
                const auto idx = (i * poly_order) + j;
                const auto needs_branching =
                    shift_bins_batch[idx] >= (eta - kEps);
                const auto too_large_step =
                    dparam_new_batch[idx] > (param_ranges[idx] + kEps);

                if (!needs_branching || too_large_step) {
                    dparam_cur_next[idx] = dparam_cur_batch[idx];
                    continue;
                }
                const auto ratio =
                    (dparam_cur_batch[idx] + kEps) / (dparam_new_batch[idx]);
                const auto num_points =
                    std::max(1, static_cast<int>(std::ceil(ratio - kEps)));
                n_branches[i] *= static_cast<double>(num_points);
                dparam_cur_next[idx] =
                    dparam_cur_batch[idx] / static_cast<double>(num_points);
            }
        }
        // Determine validation fraction
        for (SizeType i = 0; i < n_freqs; ++i) {
            const auto snap_val = param_limits[1][1];
            const auto dsnap    = dparam_cur_next[(i * poly_order) + 1];
            const auto snap_active =
                std::abs(snap_val) > (minimum_snap_cells * (dsnap + kEps));
            // Apply 0.5x if this is the first time snap becomes active
            const bool just_active = snap_active && !snap_first_branched[i];
            n_branches[i] *= just_active ? 0.5 : 1.0;
            snap_first_branched[i] = snap_first_branched[i] || just_active;
            snap_active_mask[i]    = snap_active;
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
        const auto delta_t  = coord_next.first - coord_cur.first;
        const auto n_params = poly_order + 1;
        std::vector<double> dparam_d_vec(n_freqs * n_params, 0.0);
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < poly_order; ++j) {
                dparam_d_vec[(i * n_params) + j] =
                    dparam_cur_next[(i * poly_order) + j];
            }
        }

        for (SizeType i = 0; i < n_freqs; ++i) {
            if (snap_active_mask[i]) {
                idx_circular_snap.push_back(i);
            } else {
                idx_taylor.push_back(i);
            }
        }

        // Process circular snap subset
        if (!idx_circular_snap.empty()) {
            const auto n_snap = idx_circular_snap.size();

            // Extract subset for circular snap processing
            std::vector<double> dparam_snap_subset(n_snap * n_params);
            for (SizeType i = 0; i < n_snap; ++i) {
                const auto src_idx = idx_circular_snap[i];
                for (SizeType j = 0; j < n_params; ++j) {
                    dparam_snap_subset[(i * n_params) + j] =
                        dparam_d_vec[(src_idx * n_params) + j];
                }
            }

            // Apply circular transformation
            std::vector<double> dparam_snap_new =
                transforms::shift_taylor_circular_errors_batch(
                    dparam_snap_subset, delta_t, p_orb_min,
                    use_conservative_tile, n_snap, n_params);

            // Copy back to dparam_cur_batch (excluding last dimension)
            for (SizeType i = 0; i < n_snap; ++i) {
                const auto dst_idx = idx_circular_snap[i];
                for (SizeType j = 0; j < poly_order; ++j) {
                    dparam_cur_batch[(dst_idx * poly_order) + j] =
                        dparam_snap_new[(i * n_params) + j];
                }
            }
        }

        // Process taylor subset
        if (!idx_taylor.empty()) {
            const auto n_taylor = idx_taylor.size();

            // Extract subset for taylor processing
            std::vector<double> dparam_taylor_subset(n_taylor * n_params);
            for (SizeType i = 0; i < n_taylor; ++i) {
                const auto src_idx = idx_taylor[i];
                for (SizeType j = 0; j < n_params; ++j) {
                    dparam_taylor_subset[(i * n_params) + j] =
                        dparam_d_vec[(src_idx * n_params) + j];
                }
            }

            // Apply taylor transformation
            std::vector<double> dparam_taylor_new =
                transforms::shift_taylor_errors_batch(
                    dparam_taylor_subset, delta_t, use_conservative_tile,
                    n_taylor, n_params);

            // Copy back to dparam_cur_batch (excluding last dimension)
            for (SizeType i = 0; i < n_taylor; ++i) {
                const auto dst_idx = idx_taylor[i];
                for (SizeType j = 0; j < poly_order; ++j) {
                    dparam_cur_batch[(dst_idx * poly_order) + j] =
                        dparam_taylor_new[(i * n_params) + j];
                }
            }
        }
    }

    return branching_pattern;
}

} // namespace loki::core