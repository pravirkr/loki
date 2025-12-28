#include "loki/core/circular.hpp"

#include <algorithm>
#include <numbers>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/cartesian.hpp"
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
std::pair<std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask_branch(std::span<const double> leaves_params_batch,
                            std::span<const double> leaves_dparams_batch,
                            std::span<const uint8_t> batch_needs_crackle,
                            SizeType n_leaves,
                            SizeType n_params,
                            double minimum_snap_cells) {
    constexpr double kEps = 1e-12;
    std::vector<SizeType> idx_expand_crackle;
    std::vector<SizeType> idx_keep;

    // Early exit: if no leaves need crackle branching, keep all
    bool any_needs_crackle = false;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (batch_needs_crackle[i] != 0U) {
            any_needs_crackle = true;
            break;
        }
    }
    if (!any_needs_crackle) {
        idx_keep.resize(n_leaves);
        std::iota(idx_keep.begin(), idx_keep.end(), SizeType{0});
        return {std::move(idx_expand_crackle), std::move(idx_keep)};
    }

    // Reserve reasonable capacity (crackle branching is rare)
    idx_expand_crackle.reserve(n_leaves / 10); // Conservative estimate
    idx_keep.reserve(n_leaves);

    for (SizeType i = 0; i < n_leaves; ++i) {
        // Quick filter: if doesn't need crackle branching, keep it
        if (batch_needs_crackle[i] == 0U) {
            idx_keep.push_back(i);
            continue;
        }
        const SizeType leaf_offset = i * n_params;
        // Extract values (assuming contiguous layout)
        const double crackle  = leaves_params_batch[leaf_offset + 0];
        const double snap     = leaves_params_batch[leaf_offset + 1];
        const double jerk     = leaves_params_batch[leaf_offset + 2];
        const double dcrackle = leaves_dparams_batch[leaf_offset + 0];
        const double dsnap    = leaves_dparams_batch[leaf_offset + 1];

        // Compute significance tests
        const double snap_threshold    = minimum_snap_cells * (dsnap + kEps);
        const double crackle_threshold = minimum_snap_cells * (dcrackle + kEps);

        const bool is_sig_snap    = std::abs(snap) > snap_threshold;
        const bool is_sig_crackle = std::abs(crackle) > crackle_threshold;
        // Crackle-Dominated Region (The Hole): snap weak, crackle strong
        const bool in_the_hole = (!is_sig_snap) && is_sig_crackle;

        // Physical crackle: -crackle * jerk > 0 and jerk is non-zero
        const bool is_physical_crackle =
            ((-crackle * jerk) > 0.0) && (std::abs(jerk) > kEps);

        // Circular crackle candidate that needs branching
        const bool mask_expand_crackle = in_the_hole && is_physical_crackle;

        if (mask_expand_crackle) {
            idx_expand_crackle.push_back(i);
        } else {
            idx_keep.push_back(i);
        }
    }
    return {std::move(idx_expand_crackle), std::move(idx_keep)};
}
} // namespace

std::vector<SizeType>
circ_taylor_branch_batch(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch_batch,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType fold_bins,
                         double tol_bins,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max,
                         double minimum_snap_cells) {
    constexpr SizeType kParamsExpected = 5U;
    constexpr SizeType kParamStride    = 2U;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;
    constexpr double kEps              = 1e-6;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 5 for circular orbit branch");
    error_check::check_equal(leaves_batch.size(), n_batch * kLeavesStride,
                             "leaves_batch size mismatch");
    error_check::check_greater_equal(leaves_branch_batch.size(),
                                     n_batch * branch_max * kLeavesStride,
                                     "leaves_branch_batch size mismatch");

    const auto [t0_cur, t_obs_minus_t_ref] = coord_cur;

    // Use batch_leaves memory as workspace. Partition workspace into sections:
    const SizeType workspace_size      = leaves_branch_batch.size();
    const SizeType single_batch_params = n_batch * n_params;

    // Get spans from workspace + other vector allocations
    std::span<double> dparam_cur_batch =
        leaves_branch_batch.subspan(0, single_batch_params);
    std::span<double> dparam_new_batch =
        leaves_branch_batch.subspan(single_batch_params, single_batch_params);
    std::span<double> shift_bins_batch = leaves_branch_batch.subspan(
        single_batch_params * 2, single_batch_params);
    std::span<double> f0_batch =
        leaves_branch_batch.subspan(single_batch_params * 3, n_batch);
    std::span<double> pad_branched_params = leaves_branch_batch.subspan(
        (single_batch_params * 3) + n_batch, n_batch * n_params * branch_max);
    const auto workspace_acquired_size =
        (single_batch_params * 3) + n_batch + (n_batch * n_params * branch_max);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    // Extract dparam_cur and f0 from leaves_batch
    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType param_offset = i * kLeavesStride;
        for (SizeType j = 0; j < n_params; ++j) {
            dparam_cur_batch[(i * n_params) + j] =
                leaves_batch[(param_offset + (j * kParamStride)) + 1];
        }
        f0_batch[i] =
            leaves_batch[param_offset + ((n_params + 1) * kParamStride)];
    }

    psr_utils::poly_taylor_step_d_vec(n_params, t_obs_minus_t_ref, fold_bins,
                                      tol_bins, f0_batch, dparam_new_batch, 0);
    psr_utils::poly_taylor_shift_d_vec(dparam_cur_batch, dparam_new_batch,
                                       t_obs_minus_t_ref, fold_bins, f0_batch,
                                       0, shift_bins_batch, n_batch, n_params);

    // --- Vectorized Padded Branching (All params except crackle) ---
    std::vector<double> pad_branched_dparams(n_batch * n_params);
    std::vector<SizeType> branched_counts(n_batch * n_params);

    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * n_params;

        for (SizeType j = 1; j < n_params; ++j) { // Skip crackle (j = 0)
            const SizeType flat_idx     = flat_base + j;
            const SizeType param_offset = leaf_offset + (j * kParamStride);
            const double param_cur_val  = leaves_batch[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];

            double param_min = std::numeric_limits<double>::lowest();
            double param_max = std::numeric_limits<double>::max();
            if (j == n_params - 1) {
                param_min =
                    (1 - param_limits[j][1] / f0_batch[i]) * utils::kCval;
                param_max =
                    (1 - param_limits[j][0] / f0_batch[i]) * utils::kCval;
            } else {
                const auto [pmin, pmax] = param_limits[j];
                param_min               = pmin;
                param_max               = pmax;
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
        }
    }

    // --- Vectorized Selection (mask non-crackle branched params) ---
    std::vector<uint8_t> batch_needs_crackle(n_batch);

    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;
        const SizeType flat_base   = i * n_params;

        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType flat_idx     = flat_base + j;
            const SizeType param_offset = leaf_offset + (j * kParamStride);
            const double param_cur_val  = leaves_batch[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];

            const bool needs_branching =
                shift_bins_batch[flat_idx] >= (tol_bins - kEps);

            if (j == 0) {
                // Track which batches need crackle branching
                batch_needs_crackle[i] = needs_branching ? 1U : 0U;
            }
            if (!needs_branching || j == 0) { // crackle - don't branch yet
                const SizeType pad_offset =
                    (i * n_params * branch_max) + (j * branch_max);
                pad_branched_params[pad_offset] = param_cur_val;
                pad_branched_dparams[flat_idx]  = dparam_cur_val;
                branched_counts[flat_idx]       = 1;
            }
        }
    }

    // --- First Cartesian Product (All Except Crackle) ---
    const auto [leaves_branch_taylor_batch, batch_origins] =
        utils::cartesian_prod_padded(pad_branched_params, branched_counts,
                                     n_batch, n_params, branch_max);
    const SizeType total_intermediate_leaves = batch_origins.size();

    // Create batch_needs_crackle_expanded for intermediate leaves
    std::vector<uint8_t> batch_needs_crackle_expanded(
        total_intermediate_leaves);
    for (SizeType i = 0; i < total_intermediate_leaves; ++i) {
        batch_needs_crackle_expanded[i] = batch_needs_crackle[batch_origins[i]];
    }

    // --- Classify Intermediate Leaves ---
    const auto [idx_expand_crackle, idx_keep] = get_circ_taylor_mask_branch(
        leaves_branch_taylor_batch, pad_branched_dparams,
        batch_needs_crackle_expanded, total_intermediate_leaves, n_params,
        minimum_snap_cells);

    const SizeType n_keep           = idx_keep.size();
    const SizeType n_crackle_expand = idx_expand_crackle.size();

    // Early exit: No crackle branching needed
    if (n_crackle_expand == 0) {
        // Copy all intermediate leaves to output
        for (SizeType i = 0; i < total_intermediate_leaves; ++i) {
            const SizeType origin        = batch_origins[i];
            const SizeType branch_offset = i * kLeavesStride;
            const SizeType leaf_offset   = origin * kLeavesStride;

            // Fill parameters and dparams
            for (SizeType j = 0; j < n_params; ++j) {
                const SizeType leaf_offset_j =
                    branch_offset + (j * kParamStride);
                leaves_branch_batch[leaf_offset_j + 0] =
                    leaves_branch_taylor_batch[(i * n_params) + j];
                leaves_branch_batch[leaf_offset_j + 1] =
                    pad_branched_dparams[(origin * n_params) + j];
            }

            // Fill d0 and f0
            const SizeType d0_branch_offset =
                branch_offset + (n_params * kParamStride);
            const SizeType f0_branch_offset = d0_branch_offset + kParamStride;
            const SizeType d0_leaf_offset =
                leaf_offset + (n_params * kParamStride);
            const SizeType f0_leaf_offset = d0_leaf_offset + kParamStride;

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

    // --- Branch crackle for idx_expand_crackle cases ---
    std::vector<double> crackle_branched_params(n_crackle_expand * branch_max);
    std::vector<double> crackle_branched_dparams(n_crackle_expand);
    std::vector<SizeType> crackle_branched_counts(n_crackle_expand);

    for (SizeType i = 0; i < n_crackle_expand; ++i) {
        const SizeType leaf_idx       = idx_expand_crackle[i];
        const SizeType orig_batch_idx = batch_origins[leaf_idx];

        const double crackle_cur =
            leaves_branch_taylor_batch[(leaf_idx * n_params) + 0];
        const double crackle_dparam =
            pad_branched_dparams[(orig_batch_idx * n_params) + 0];

        const auto [pmin, pmax] = param_limits[0]; // crackle limits
        std::span<double> slice_span(
            crackle_branched_params.data() + (i * branch_max), branch_max);
        auto [dparam_act, count] = psr_utils::branch_param_padded(
            slice_span, crackle_cur, crackle_dparam,
            dparam_new_batch[(orig_batch_idx * n_params) + 0], pmin, pmax);

        crackle_branched_dparams[i] = dparam_act;
        crackle_branched_counts[i]  = count;
    }

    // --- Construct Final Array ---
    SizeType total_crackle_branches = 0;
    for (SizeType count : crackle_branched_counts) {
        total_crackle_branches += count;
    }
    const SizeType total_leaves = n_keep + total_crackle_branches;
    std::vector<SizeType> origins_final(total_leaves);

    // Copy non-crackle-branching leaves (idx_keep)
    for (SizeType i = 0; i < n_keep; ++i) {
        const SizeType keep_idx      = idx_keep[i];
        const SizeType origin        = batch_origins[keep_idx];
        const SizeType branch_offset = i * kLeavesStride;
        const SizeType leaf_offset   = origin * kLeavesStride;

        // Fill parameters and dparams
        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType leaf_offset_j = branch_offset + (j * kParamStride);
            leaves_branch_batch[leaf_offset_j + 0] =
                leaves_branch_taylor_batch[(keep_idx * n_params) + j];
            leaves_branch_batch[leaf_offset_j + 1] =
                pad_branched_dparams[(origin * n_params) + j];
        }

        // Fill d0 and f0
        const SizeType d0_branch_offset =
            branch_offset + (n_params * kParamStride);
        const SizeType f0_branch_offset = d0_branch_offset + kParamStride;
        const SizeType d0_leaf_offset = leaf_offset + (n_params * kParamStride);
        const SizeType f0_leaf_offset = d0_leaf_offset + kParamStride;

        leaves_branch_batch[d0_branch_offset + 0] =
            leaves_batch[d0_leaf_offset + 0];
        leaves_branch_batch[d0_branch_offset + 1] =
            leaves_batch[d0_leaf_offset + 1];
        leaves_branch_batch[f0_branch_offset + 0] =
            leaves_batch[f0_leaf_offset + 0];
        leaves_branch_batch[f0_branch_offset + 1] =
            leaves_batch[f0_leaf_offset + 1];

        origins_final[i] = origin;
    }

    // Add crackle-branched leaves
    SizeType current_idx = n_keep;
    for (SizeType i = 0; i < n_crackle_expand; ++i) {
        const SizeType count_i        = crackle_branched_counts[i];
        const SizeType orig_leaf_idx  = idx_expand_crackle[i];
        const SizeType orig_batch_idx = batch_origins[orig_leaf_idx];
        const SizeType leaf_offset    = orig_batch_idx * kLeavesStride;

        for (SizeType b = 0; b < count_i; ++b) {
            const SizeType dst_offset = (current_idx + b) * kLeavesStride;

            // Fill parameters (non-crackle from original leaf)
            for (SizeType j = 0; j < n_params; ++j) {
                const SizeType leaf_offset_j = dst_offset + (j * kParamStride);
                leaves_branch_batch[leaf_offset_j + 0] =
                    leaves_branch_taylor_batch[(orig_leaf_idx * n_params) + j];
                leaves_branch_batch[leaf_offset_j + 1] =
                    pad_branched_dparams[(orig_batch_idx * n_params) + j];
            }

            // Override crackle parameter with branched value
            leaves_branch_batch[dst_offset + 0] =
                crackle_branched_params[(i * branch_max) + b];
            leaves_branch_batch[dst_offset + 1] = crackle_branched_dparams[i];

            // Fill d0 and f0
            const SizeType d0_dst_offset =
                dst_offset + (n_params * kParamStride);
            const SizeType f0_dst_offset = d0_dst_offset + kParamStride;
            const SizeType d0_leaf_offset =
                leaf_offset + (n_params * kParamStride);
            const SizeType f0_leaf_offset = d0_leaf_offset + kParamStride;

            leaves_branch_batch[d0_dst_offset + 0] =
                leaves_batch[d0_leaf_offset + 0];
            leaves_branch_batch[d0_dst_offset + 1] =
                leaves_batch[d0_leaf_offset + 1];
            leaves_branch_batch[f0_dst_offset + 0] =
                leaves_batch[f0_leaf_offset + 0];
            leaves_branch_batch[f0_dst_offset + 1] =
                leaves_batch[f0_leaf_offset + 1];

            origins_final[current_idx + b] = orig_batch_idx;
        }
        current_idx += count_i;
    }

    return origins_final;
}

SizeType circ_taylor_validate_batch(std::span<double> leaves_batch,
                                    std::span<SizeType> leaves_origins,
                                    SizeType n_leaves,
                                    SizeType n_params,
                                    double p_orb_min,
                                    double x_mass_const,
                                    double minimum_snap_cells) {
    constexpr SizeType kParamsExpected = 5U;
    constexpr SizeType kParamStride    = 2U;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;
    constexpr double kEps              = 1e-12;
    constexpr double kTwoThirds        = 2.0 / 3.0;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 5 for circular orbit resolve");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double omega_max_sq = std::pow(2.0 * std::numbers::pi / p_orb_min, 2);
    std::vector<bool> mask_keep(n_leaves, false);

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        // Extract values
        const double crackle  = leaves_batch[leaf_offset + 0];
        const double dcrackle = leaves_batch[leaf_offset + 1];
        const double snap     = leaves_batch[leaf_offset + 2];
        const double dsnap    = leaves_batch[leaf_offset + 3];
        const double jerk     = leaves_batch[leaf_offset + 4];
        const double accel    = leaves_batch[leaf_offset + 6];

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
                    leaves_batch.begin() + static_cast<IndexType>(src_offset),
                    kLeavesStride,
                    leaves_batch.begin() + static_cast<IndexType>(dst_offset));
                // Copy origin
                leaves_origins[write_idx] = leaves_origins[i];
            }
            ++write_idx;
        }
    }
    // write_idx is the new number of valid leaves
    return write_idx;
}

void circ_taylor_resolve_batch(std::span<const double> leaves_batch,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_cur,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               std::span<SizeType> pindex_flat_batch,
                               std::span<float> relative_phase_batch,
                               SizeType nbins,
                               SizeType n_leaves,
                               SizeType n_params,
                               double minimum_snap_cells) {
    constexpr SizeType kParamsExpected = 5;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 5 for circular orbit resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 5 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "pindex_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");

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
        get_circ_taylor_mask(leaves_batch, n_leaves, n_params,
                             minimum_snap_cells);

    SizeType hint_a = 0, hint_f = 0;
    // Process circular indices
    for (SizeType i : idx_circular_snap) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto s_t_cur = leaves_batch[leaf_offset + (1 * kParamStride)];
        const auto j_t_cur = leaves_batch[leaf_offset + (2 * kParamStride)];
        const auto a_t_cur = leaves_batch[leaf_offset + (3 * kParamStride)];
        const auto v_t_cur = leaves_batch[leaf_offset + (4 * kParamStride)];
        const auto f0      = leaves_batch[leaf_offset + (6 * kParamStride)];

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
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices (only need accel and freq)
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);

        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
    }

    // Reset hints
    hint_a = 0;
    hint_f = 0;
    // Process circular crackle indices
    for (SizeType i : idx_circular_crackle) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto c_t_cur = leaves_batch[leaf_offset + (0 * kParamStride)];
        const auto j_t_cur = leaves_batch[leaf_offset + (2 * kParamStride)];
        const auto a_t_cur = leaves_batch[leaf_offset + (3 * kParamStride)];
        const auto v_t_cur = leaves_batch[leaf_offset + (4 * kParamStride)];
        const auto f0      = leaves_batch[leaf_offset + (6 * kParamStride)];

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
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices (only need accel and freq)
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);

        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
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

        const auto c_t_cur = leaves_batch[batch_offset + (0 * kParamStride)];
        const auto s_t_cur = leaves_batch[batch_offset + (1 * kParamStride)];
        const auto j_t_cur = leaves_batch[batch_offset + (2 * kParamStride)];
        const auto a_t_cur = leaves_batch[batch_offset + (3 * kParamStride)];
        const auto v_t_cur = leaves_batch[batch_offset + (4 * kParamStride)];
        const auto f0      = leaves_batch[batch_offset + (6 * kParamStride)];
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
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
    }
}

void circ_taylor_transform_batch(std::span<double> leaves_batch,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 bool use_conservative_tile,
                                 double minimum_snap_cells) {
    constexpr SizeType kParamsExpected = 5;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 4 for circular transform");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto delta_t               = t0_next - t0_cur;

    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circ_taylor_mask(leaves_batch, n_leaves, n_params,
                             minimum_snap_cells);

    // Process circular indices
    for (SizeType i : idx_circular_snap) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto s_val_i = leaves_batch[leaf_offset + 2];
        const auto j_val_i = leaves_batch[leaf_offset + 4];
        const auto a_val_i = leaves_batch[leaf_offset + 6];
        const auto v_val_i = leaves_batch[leaf_offset + 8];
        const auto d_val_i = leaves_batch[leaf_offset + 10];

        const double omega_orb_sq = -s_val_i / a_val_i;
        const double omega_orb    = std::sqrt(omega_orb_sq);
        const double omega_dt     = omega_orb * delta_t;
        const double cos_odt      = std::cos(omega_dt);
        const double sin_odt      = std::sin(omega_dt);

        // Pin-down {s, j, a}
        const double a_val_j =
            (a_val_i * cos_odt) + ((j_val_i / omega_orb) * sin_odt);
        const double j_val_j =
            (j_val_i * cos_odt) - ((a_val_i * omega_orb) * sin_odt);
        const double s_val_j = -omega_orb_sq * a_val_j;
        const double c_val_j = -omega_orb_sq * j_val_j;
        // Integrate to get {v, d}
        const double v_circ_i = -j_val_i / omega_orb_sq;
        const double v_circ_j = -j_val_j / omega_orb_sq;
        const double v_val_j  = v_circ_j + (v_val_i - v_circ_i);
        const double d_circ_j = -a_val_j / omega_orb_sq;
        const double d_circ_i = -a_val_i / omega_orb_sq;
        const double d_val_j =
            d_circ_j + (d_val_i - d_circ_i) + ((v_val_i - v_circ_i) * delta_t);

        // Write back transformed values (errors unchanged)
        leaves_batch[leaf_offset + 0]  = c_val_j;
        leaves_batch[leaf_offset + 2]  = s_val_j;
        leaves_batch[leaf_offset + 4]  = j_val_j;
        leaves_batch[leaf_offset + 6]  = a_val_j;
        leaves_batch[leaf_offset + 8]  = v_val_j;
        leaves_batch[leaf_offset + 10] = d_val_j;
    }

    // Process circular crackle indices
    for (SizeType i : idx_circular_crackle) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto c_val_i = leaves_batch[leaf_offset + 0];
        const auto j_val_i = leaves_batch[leaf_offset + 4];
        const auto a_val_i = leaves_batch[leaf_offset + 6];
        const auto v_val_i = leaves_batch[leaf_offset + 8];
        const auto d_val_i = leaves_batch[leaf_offset + 10];

        const double omega_orb_sq = -c_val_i / j_val_i;
        const double omega_orb    = std::sqrt(omega_orb_sq);
        const double omega_dt     = omega_orb * delta_t;
        const double cos_odt      = std::cos(omega_dt);
        const double sin_odt      = std::sin(omega_dt);

        // Pin-down {s, j, a}
        const double a_val_j =
            (a_val_i * cos_odt) + ((j_val_i / omega_orb) * sin_odt);
        const double j_val_j =
            (j_val_i * cos_odt) - ((a_val_i * omega_orb) * sin_odt);
        const double s_val_j = -omega_orb_sq * a_val_j;
        const double c_val_j = -omega_orb_sq * j_val_j;
        // Integrate to get {v, d}
        const double v_circ_i = -j_val_i / omega_orb_sq;
        const double v_circ_j = -j_val_j / omega_orb_sq;
        const double v_val_j  = v_circ_j + (v_val_i - v_circ_i);
        const double d_circ_j = -a_val_j / omega_orb_sq;
        const double d_circ_i = -a_val_i / omega_orb_sq;
        const double d_val_j =
            d_circ_j + (d_val_i - d_circ_i) + ((v_val_i - v_circ_i) * delta_t);

        // Write back transformed values (errors unchanged)
        leaves_batch[leaf_offset + 0]  = c_val_j;
        leaves_batch[leaf_offset + 2]  = s_val_j;
        leaves_batch[leaf_offset + 4]  = j_val_j;
        leaves_batch[leaf_offset + 6]  = a_val_j;
        leaves_batch[leaf_offset + 8]  = v_val_j;
        leaves_batch[leaf_offset + 10] = d_val_j;
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

        const auto c_val_i = leaves_batch[leaf_offset + 0];
        const auto c_err_i = leaves_batch[leaf_offset + 1];
        const auto s_val_i = leaves_batch[leaf_offset + 2];
        const auto s_err_i = leaves_batch[leaf_offset + 3];
        const auto j_val_i = leaves_batch[leaf_offset + 4];
        const auto j_err_i = leaves_batch[leaf_offset + 5];
        const auto a_val_i = leaves_batch[leaf_offset + 6];
        const auto a_err_i = leaves_batch[leaf_offset + 7];
        const auto v_val_i = leaves_batch[leaf_offset + 8];
        const auto v_err_i = leaves_batch[leaf_offset + 9];
        const auto d_val_i = leaves_batch[leaf_offset + 10];
        const auto d_err_i = leaves_batch[leaf_offset + 11];

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
        double c_err_j, s_err_j, j_err_j, a_err_j, v_err_j, d_err_j;
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
            d_err_j = std::sqrt(
                (d_err_i * d_err_i) + (v_err_i * v_err_i * delta_t_sq) +
                (a_err_i * a_err_i * half_delta_t_sq * half_delta_t_sq) +
                (j_err_i * j_err_i * sixth_delta_t_cubed *
                 sixth_delta_t_cubed) +
                (s_err_i * s_err_i * twenty_fourth_delta_t_fourth *
                 twenty_fourth_delta_t_fourth) +
                (c_err_i * c_err_i * onehundred_twenty_delta_t_fifth *
                 onehundred_twenty_delta_t_fifth));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            c_err_j = c_err_i;
            s_err_j = s_err_i;
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
            d_err_j = d_err_i;
        }

        // Write back transformed values
        leaves_batch[leaf_offset + 0]  = c_val_j;
        leaves_batch[leaf_offset + 1]  = c_err_j;
        leaves_batch[leaf_offset + 2]  = s_val_j;
        leaves_batch[leaf_offset + 3]  = s_err_j;
        leaves_batch[leaf_offset + 4]  = j_val_j;
        leaves_batch[leaf_offset + 5]  = j_err_j;
        leaves_batch[leaf_offset + 6]  = a_val_j;
        leaves_batch[leaf_offset + 7]  = a_err_j;
        leaves_batch[leaf_offset + 8]  = v_val_j;
        leaves_batch[leaf_offset + 9]  = v_err_j;
        leaves_batch[leaf_offset + 10] = d_val_j;
        leaves_batch[leaf_offset + 11] = d_err_j;
    }
}

std::vector<double>
generate_bp_taylor_circular(std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            const std::vector<ParamLimitType>& param_limits,
                            double tseg_ffa,
                            SizeType nsegments,
                            SizeType fold_bins,
                            double tol_bins,
                            SizeType ref_seg,
                            bool use_conservative_tile) {
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
    std::vector<bool> snap_first_branched(n_freqs, false);
    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = scheme.get_coord(prune_level);
        const auto coord_cur  = scheme.get_current_coord(prune_level);
        const auto [t0_cur, t_obs_minus_t_ref] = coord_cur;

        // Calculate optimal parameter steps
        std::vector<double> dparam_new_batch(n_freqs * poly_order);
        psr_utils::poly_taylor_step_d_vec(
            poly_order, t_obs_minus_t_ref, fold_bins, tol_bins,
            std::span<const double>(f0_batch),
            std::span<double>(dparam_new_batch), 0);

        // Calculate shift bins
        std::vector<double> shift_bins_batch(n_freqs * poly_order);
        psr_utils::poly_taylor_shift_d_vec(
            std::span<const double>(dparam_cur_batch),
            std::span<const double>(dparam_new_batch), t_obs_minus_t_ref,
            fold_bins, std::span<const double>(f0_batch), 0,
            std::span<double>(shift_bins_batch), n_freqs, poly_order);

        // Initialize arrays for next iteration
        std::vector<double> dparam_cur_next(n_freqs * poly_order);
        std::vector<SizeType> n_branch_accel(n_freqs, 1);
        std::vector<SizeType> n_branch_snap(n_freqs, 1);
        std::vector<double> n_branches(n_freqs, 1);
        std::vector<double> validation_fractions(n_freqs, 1.0);

        // Determine branching needs
        constexpr double kEps = 1e-12;
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 1; j < poly_order; ++j) {
                const auto idx = (i * poly_order) + j;
                const auto needs_branching =
                    shift_bins_batch[idx] >= (tol_bins - kEps);
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
                if (j == 1) {
                    n_branch_snap[i] *= static_cast<SizeType>(num_points);
                } else if (j == 3) {
                    n_branch_accel[i] *= static_cast<SizeType>(num_points);
                }
            }
            // Determine validation fraction
            const bool snap_branches_now  = n_branch_snap[i] > 1;
            const bool accel_branches_now = n_branch_accel[i] > 1;
            if (snap_branches_now && !snap_first_branched[i]) {
                if (accel_branches_now ||
                    dparam_cur_batch[(i * poly_order) + 3] > 0.0) {
                    validation_fractions[i] = 0.5;
                } else {
                    validation_fractions[i] = 1.0;
                }
                snap_first_branched[i] = true;
            } else {
                validation_fractions[i] = 1.0;
            }
            n_branches[i] *= validation_fractions[i];
        }
        double total_branches = 0;
        for (SizeType i = 0; i < n_freqs; ++i) {
            total_branches += n_branches[i];
        }
        branching_pattern[prune_level - 1] =
            total_branches / static_cast<double>(n_freqs);

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
        auto dparam_d_vec_new = transforms::shift_taylor_errors_batch(
            dparam_d_vec, delta_t, use_conservative_tile, n_freqs, n_params);
        // Copy back to dparam_cur_batch (excluding last dimension)
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < poly_order; ++j) {
                dparam_cur_batch[(i * poly_order) + j] =
                    dparam_d_vec_new[(i * n_params) + j];
            }
        }
    }

    return branching_pattern;
}

} // namespace loki::core