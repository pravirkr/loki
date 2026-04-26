#include "loki/core/circular.hpp"

#include <algorithm>
#include <cstring>
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
#include "loki/utils/workspace.hpp"

namespace loki::core {

std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask_scattered(std::span<const double> leaves_batch,
                               std::span<SizeType> indices_batch,
                               SizeType n_leaves,
                               SizeType n_params,
                               double minimum_snap_cells) {
    constexpr SizeType kParamsExpected = 5U;
    constexpr SizeType kParamStride    = 2U;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

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
        const auto leaf_idx = indices_batch[i];
        const auto lo       = leaf_idx * kLeavesStride;
        const auto crackle  = leaves_batch[lo + 0];
        const auto dcrackle = leaves_batch[lo + 1];
        const auto snap     = leaves_batch[lo + 2];
        const auto dsnap    = leaves_batch[lo + 3];
        const auto jerk     = leaves_batch[lo + 4];
        const auto accel    = leaves_batch[lo + 6];

        // Determine whether snap is significantly measured
        const bool is_sig_snap = std::abs(snap) > (minimum_snap_cells * dsnap);
        // Snap-Dominated Region: Check if implied Omega^2 = -snap/accel is
        // physical
        const bool is_physical_snap =
            ((-snap * accel) > 0.0) && (std::abs(accel) > utils::kZeroEps);

        if (is_sig_snap && is_physical_snap) {
            idx_circular_snap.push_back(leaf_idx);
            continue;
        }

        // Crackle-Dominated Region: Check if crackle is significantly measured
        const bool is_sig_crackle =
            std::abs(crackle) > (minimum_snap_cells * dcrackle);
        // Check if implied Omega^2 = -crackle/jerk is physical
        const bool is_physical_crackle =
            ((-crackle * jerk) > 0.0) && (std::abs(jerk) > utils::kZeroEps);
        if (is_sig_crackle && is_physical_crackle) {
            idx_circular_crackle.push_back(leaf_idx);
        } else {
            idx_taylor.push_back(leaf_idx);
        }
    }
    return {std::move(idx_circular_snap), std::move(idx_circular_crackle),
            std::move(idx_taylor)};
}

std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask(std::span<const double> leaves_batch,
                     SizeType n_leaves,
                     SizeType n_params,
                     double minimum_snap_cells) {
    std::vector<SizeType> indices_batch(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        indices_batch[i] = i;
    }
    return get_circ_taylor_mask_scattered(leaves_batch, indices_batch, n_leaves,
                                          n_params, minimum_snap_cells);
}

namespace {
inline bool is_in_hole(double snap,
                       double dsnap,
                       double jerk,
                       double accel,
                       double minimum_snap_cells) noexcept {
    const bool is_sig_snap = std::abs(snap) > (minimum_snap_cells * dsnap);
    const bool is_physical_snap =
        ((-snap * accel) > 0.0) && (std::abs(accel) > utils::kZeroEps);
    const bool is_stable_jerk = std::abs(jerk) > utils::kZeroEps;
    return is_sig_snap && (!is_physical_snap) && is_stable_jerk;
}

} // namespace

SizeType circ_taylor_branch_batch(std::span<const double> leaves_tree,
                                  std::span<double> leaves_branch,
                                  std::span<SizeType> leaves_origins,
                                  std::pair<double, double> coord_cur,
                                  SizeType nbins,
                                  double eta,
                                  SizeType branch_max,
                                  SizeType n_leaves,
                                  double minimum_snap_cells,
                                  memory::BranchingWorkspace& branch_ws) {
    constexpr SizeType kParams       = 5U;
    constexpr SizeType kParamStride  = 2U;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");

    error_check::check_greater_equal(branch_ws.scratch_params.size(),
                                     n_leaves * kParams * branch_max,
                                     "Workspace scratch_params too small");
    error_check::check_greater_equal(branch_ws.scratch_shifts.size(),
                                     n_leaves * kParams,
                                     "Workspace scratch_shifts too small");
    const auto [_, dt]   = coord_cur; // t_obs - t_ref
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

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;
    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    error_check::check_less_equal(single_batch_params, workspace_size,
                                  "workspace size mismatch");

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ scratch_params   = branch_ws.scratch_params.data();
    double* __restrict__ scratch_dparams  = branch_ws.scratch_dparams.data();
    SizeType* __restrict__ scratch_counts = branch_ws.scratch_counts.data();
    double* __restrict__ shift_bins_ptr   = branch_ws.scratch_shifts.data();
    double* __restrict__ dparam_new_ptr   = dparam_new.data();

    // Loop 1: step + shift (vectorizable)
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto d5_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d4_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 5];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 7];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 9];
        const auto f0         = leaves_tree_ptr[lo + 12];

        const auto dfactor    = utils::kCval / f0;
        const auto d5_sig_new = dphi * dfactor * 1920.0 * inv_dt5;
        const auto d4_sig_new = dphi * dfactor * 192.0 * inv_dt4;
        const auto d3_sig_new = dphi * dfactor * 24.0 * inv_dt3;
        const auto d2_sig_new = dphi * dfactor * 4.0 * inv_dt2;
        const auto d1_sig_new = dphi * dfactor * 1.0 * inv_dt;

        dparam_new_ptr[fb + 0] = d5_sig_new;
        dparam_new_ptr[fb + 1] = d4_sig_new;
        dparam_new_ptr[fb + 2] = d3_sig_new;
        dparam_new_ptr[fb + 3] = d2_sig_new;
        dparam_new_ptr[fb + 4] = d1_sig_new;

        shift_bins_ptr[fb + 0] = std::abs(d5_sig_cur - d5_sig_new) * dt5 *
                                 nbins_d / (1920.0 * dfactor);
        shift_bins_ptr[fb + 1] = std::abs(d4_sig_cur - d4_sig_new) * dt4 *
                                 nbins_d / (192.0 * dfactor);
        shift_bins_ptr[fb + 2] = std::abs(d3_sig_cur - d3_sig_new) * dt3 *
                                 nbins_d / (24.0 * dfactor);
        shift_bins_ptr[fb + 3] =
            std::abs(d2_sig_cur - d2_sig_new) * dt2 * nbins_d / (4.0 * dfactor);
        shift_bins_ptr[fb + 4] =
            std::abs(d1_sig_cur - d1_sig_new) * dt * nbins_d / (1.0 * dfactor);
    }

    // EXIT A — no branching at all: fast memcpy path
    {
        bool any_branching = false;
        for (SizeType i = 0; i < n_leaves * kParams; ++i) {
            if (shift_bins_ptr[i] >= (eta - utils::kFloatEps)) {
                any_branching = true;
                break;
            }
        }
        if (!any_branching) {
            std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                        n_leaves * kLeavesStride * sizeof(double));
            for (SizeType i = 0; i < n_leaves; ++i) {
                leaves_origins_ptr[i] = i;
            }
            return n_leaves;
        }
    }

    // Loop 2: branching, populate scratch workspace
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto d5_cur     = leaves_tree_ptr[lo + 0];
        const auto d5_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d4_cur     = leaves_tree_ptr[lo + 2];
        const auto d4_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d3_cur     = leaves_tree_ptr[lo + 4];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 5];
        const auto d2_cur     = leaves_tree_ptr[lo + 6];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 7];
        const auto d1_cur     = leaves_tree_ptr[lo + 8];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 9];

        // Branch d5 parameter (no branching as of yet)
        psr_utils::branch_one_param_padded_crackle(
            0, d5_cur, d5_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        // Branch d4-d1 parameters
        psr_utils::branch_one_param_padded(
            1, d4_cur, d4_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            2, d3_cur, d3_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            3, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            4, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            scratch_params, scratch_dparams, scratch_counts, fb, branch_max);
    }

    // Loop 3: Branching d4-d1, write every (d4×d3×d2×d1) combo as a complete
    // output leaf. Ignore n_d5
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const double d5_sig   = scratch_dparams[fb + 0];
        const double d4_sig   = scratch_dparams[fb + 1];
        const double d3_sig   = scratch_dparams[fb + 2];
        const double d2_sig   = scratch_dparams[fb + 3];
        const double d1_sig   = scratch_dparams[fb + 4];
        const SizeType n_d4   = scratch_counts[fb + 1];
        const SizeType n_d3   = scratch_counts[fb + 2];
        const SizeType n_d2   = scratch_counts[fb + 3];
        const SizeType n_d1   = scratch_counts[fb + 4];
        const SizeType d5_off = (fb + 0) * branch_max;
        const SizeType d4_off = (fb + 1) * branch_max;
        const SizeType d3_off = (fb + 2) * branch_max;
        const SizeType d2_off = (fb + 3) * branch_max;
        const SizeType d1_off = (fb + 4) * branch_max;

        for (SizeType b = 0; b < n_d4; ++b) {
            for (SizeType c = 0; c < n_d3; ++c) {
                for (SizeType d = 0; d < n_d2; ++d) {
                    for (SizeType e = 0; e < n_d1; ++e) {
                        const SizeType bo = out_leaves * kLeavesStride;
                        double* __restrict__ out_ptr = leaves_branch_ptr + bo;

                        out_ptr[0] = scratch_params[d5_off];
                        out_ptr[1] = d5_sig;
                        out_ptr[2] = scratch_params[d4_off + b];
                        out_ptr[3] = d4_sig;
                        out_ptr[4] = scratch_params[d3_off + c];
                        out_ptr[5] = d3_sig;
                        out_ptr[6] = scratch_params[d2_off + d];
                        out_ptr[7] = d2_sig;
                        out_ptr[8] = scratch_params[d1_off + e];
                        out_ptr[9] = d1_sig;
                        // Copy d0 and f0
                        std::memcpy(out_ptr + 10, leaves_tree_ptr + lo + 10,
                                    4 * sizeof(double));

                        leaves_origins_ptr[out_leaves] = i;
                        ++out_leaves;
                    }
                }
            }
        }
    }

    // EXIT B — no crackle branching needed (stages 1–30 always exit here)
    {
        bool any_crackle = false;
        for (SizeType i = 0; i < n_leaves; ++i) {
            if (shift_bins_ptr[(i * kParams) + 0] >= (eta - utils::kFloatEps)) {
                any_crackle = true;
                break;
            }
        }
        if (!any_crackle) {
            error_check::check_less_equal(out_leaves, n_leaves * branch_max,
                                          "out_leaves size mismatch");
            return out_leaves;
        }
    }

    // Loop 4: Hole expansion. For each existing output leaf that falls in
    // the hole region, replace its d5 value in-place with the first crackle
    // child and append the remaining (n_d5 - 1) crackle children at the tail.
    const SizeType base_out = out_leaves; // snapshot — do not modify in loop
    auto slice_span         = std::span<double>(scratch_params, branch_max);
    for (SizeType i = 0; i < base_out; ++i) {
        const SizeType lo         = i * kLeavesStride;
        double* __restrict__ leaf = leaves_branch_ptr + lo;

        // Retrieve origin leaf index to look up d5_sig_new
        const SizeType origin    = leaves_origins_ptr[i];
        const SizeType origin_lo = origin * kLeavesStride;
        const SizeType fb        = origin * kParams;
        const SizeType n_d5      = scratch_counts[fb + 0];

        const bool in_hole =
            is_in_hole(leaf[2], leaf[3], leaf[4], leaf[6], minimum_snap_cells);
        const bool need_branching =
            shift_bins_ptr[fb + 0] >= (eta - utils::kFloatEps);
        if (!in_hole || !need_branching || n_d5 == 1) [[likely]] {
            continue;
        }

        const double d5_val_parent = leaves_tree_ptr[origin_lo + 0];
        const double d5_sig_parent = leaves_tree_ptr[origin_lo + 1];
        psr_utils::branch_crackle_padded(slice_span, d5_val_parent,
                                         d5_sig_parent, n_d5);

        // Overwrite slot i with first crackle branch, dparam already
        // computed in Loop 3. Append remaining crackle branches at the tail
        leaf[0] = slice_span[0];
        for (SizeType a = 1; a < n_d5; ++a) [[unlikely]] {
            const SizeType bo        = out_leaves * kLeavesStride;
            double* __restrict__ out = leaves_branch_ptr + bo;

            // Full copy of this leaf, then patch d5
            std::memcpy(out, leaf, kLeavesStride * sizeof(double));
            out[0] = slice_span[a];

            leaves_origins_ptr[out_leaves] = origin;
            ++out_leaves;
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
    constexpr double kTwoThirds      = 2.0 / 3.0;

    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_branch size mismatch");

    const double omega_max_sq = std::pow(2.0 * std::numbers::pi / p_orb_min, 2);
    std::vector<bool> mask_keep(n_leaves, false);

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo  = i * kLeavesStride;
        const double snap  = leaves_branch[lo + 2];
        const double dsnap = leaves_branch[lo + 3];
        const double accel = leaves_branch[lo + 6];

        // Determine whether snap is significantly measured
        const bool is_sig_snap = std::abs(snap) > (minimum_snap_cells * dsnap);
        const bool snap_possible = (std::abs(accel) > utils::kZeroEps);
        const bool sign_valid    = (-snap * accel) > 0.0;
        const bool snap_unphysical =
            is_sig_snap && snap_possible && !sign_valid;
        const bool snap_region = is_sig_snap && snap_possible && sign_valid;
        if (snap_unphysical) {
            mask_keep[i] = false; // kill outright
            continue;
        }
        if (!snap_region) {
            mask_keep[i] = true; // hole region — always preserve
            continue;
        }
        // snap_region: apply physical validity checks
        const double omega_sq = -snap / accel;
        const double limit_accel =
            x_mass_const * std::pow(omega_sq, kTwoThirds);

        const bool valid_omega = omega_sq < omega_max_sq;
        // |d2| < x * omega^(4/3)
        const bool valid_accel = std::abs(accel) <= limit_accel;

        mask_keep[i] = valid_omega && valid_accel;
    }

    // Compact arrays in-place
    SizeType write_idx = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (!mask_keep[i]) {
            continue;
        }
        if (write_idx != i) {
            std::copy_n(leaves_branch.begin() +
                            static_cast<IndexType>(i * kLeavesStride),
                        kLeavesStride,
                        leaves_branch.begin() +
                            static_cast<IndexType>(write_idx * kLeavesStride));
            // Copy origin
            leaves_origins[write_idx] = leaves_origins[i];
        }
        ++write_idx;
    }
    // write_idx is the new number of valid leaves
    return write_idx;
}

void circ_taylor_resolve_batch(std::span<const double> leaves_tree,
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
                               double minimum_snap_cells) {
    constexpr SizeType kParams       = 5;
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
                             "param_limits size should be 5");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[3];
    const auto& lim_freq  = param_limits[4];

    // Pre-compute constants to avoid repeated calculations
    const auto dt_add         = t0_add - t0_cur;
    const auto dt_init        = t0_init - t0_cur;
    const auto half_dt2_add   = 0.5 * (dt_add * dt_add);
    const auto half_dt2_init  = 0.5 * (dt_init * dt_init);
    const auto dt             = dt_add - dt_init;
    const auto sixth_dt3_add  = half_dt2_add * dt_add / 3.0;
    const auto sixth_dt3_init = half_dt2_init * dt_init / 3.0;
    const auto half_dt2       = half_dt2_add - half_dt2_init;
    const auto sixth_dt3      = sixth_dt3_add - sixth_dt3_init;
    const auto twenty_fourth_dt4 =
        ((sixth_dt3_add * dt_add) - (sixth_dt3_init * dt_init)) / 4.0;
    const auto onehundred_twenty_dt5 =
        ((sixth_dt3_add * half_dt2_add) - (sixth_dt3_init * half_dt2_init)) /
        10.0;

    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circ_taylor_mask(leaves_tree, n_leaves, kParams,
                             minimum_snap_cells);

    // Process circular indices
    for (SizeType i : idx_circular_snap) {
        const auto lo      = i * kLeavesStride;
        const auto s_t_cur = leaves_tree[lo + 2];
        const auto j_t_cur = leaves_tree[lo + 4];
        const auto a_t_cur = leaves_tree[lo + 6];
        const auto v_t_cur = leaves_tree[lo + 8];
        const auto f0      = leaves_tree[lo + 12];

        // Circular orbit mask condition
        const auto omega_orb_sq = -s_t_cur / a_t_cur;
        const auto omega_orb    = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const auto omega_dt_add = omega_orb * dt_add;
        const auto cos_odt_add  = std::cos(omega_dt_add);
        const auto sin_odt_add  = std::sin(omega_dt_add);
        const auto a_t_add =
            (a_t_cur * cos_odt_add) + ((j_t_cur / omega_orb) * sin_odt_add);
        const auto j_t_add =
            (j_t_cur * cos_odt_add) - ((a_t_cur * omega_orb) * sin_odt_add);

        const auto omega_dt_init = omega_orb * dt_init;
        const auto cos_odt_init  = std::cos(omega_dt_init);
        const auto sin_odt_init  = std::sin(omega_dt_init);
        const auto a_t_init =
            (a_t_cur * cos_odt_init) + ((j_t_cur / omega_orb) * sin_odt_init);
        const auto j_t_init =
            (j_t_cur * cos_odt_init) - ((a_t_cur * omega_orb) * sin_odt_init);

        const auto a_new = a_t_add;
        const auto delta_v =
            (-j_t_add / omega_orb_sq) - (-j_t_init / omega_orb_sq);
        const auto delta_d   = (-a_t_add / omega_orb_sq) -
                               (-a_t_init / omega_orb_sq) +
                               ((v_t_cur + (j_t_cur / omega_orb_sq)) * dt);
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

    // Process circular crackle indices
    for (SizeType i : idx_circular_crackle) {
        const auto lo      = i * kLeavesStride;
        const auto c_t_cur = leaves_tree[lo + 0];
        const auto j_t_cur = leaves_tree[lo + 4];
        const auto a_t_cur = leaves_tree[lo + 6];
        const auto v_t_cur = leaves_tree[lo + 8];
        const auto f0      = leaves_tree[lo + 12];

        // Circular orbit mask condition
        const auto omega_orb_sq = -c_t_cur / j_t_cur;
        const auto omega_orb    = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const auto omega_dt_add = omega_orb * dt_add;
        const auto cos_odt_add  = std::cos(omega_dt_add);
        const auto sin_odt_add  = std::sin(omega_dt_add);
        const auto a_t_add =
            (a_t_cur * cos_odt_add) + ((j_t_cur / omega_orb) * sin_odt_add);
        const auto j_t_add =
            (j_t_cur * cos_odt_add) - ((a_t_cur * omega_orb) * sin_odt_add);

        const auto omega_dt_init = omega_orb * dt_init;
        const auto cos_odt_init  = std::cos(omega_dt_init);
        const auto sin_odt_init  = std::sin(omega_dt_init);
        const auto a_t_init =
            (a_t_cur * cos_odt_init) + ((j_t_cur / omega_orb) * sin_odt_init);
        const auto j_t_init =
            (j_t_cur * cos_odt_init) - ((a_t_cur * omega_orb) * sin_odt_init);

        const auto a_new = a_t_add;
        const auto delta_v =
            (-j_t_add / omega_orb_sq) - (-j_t_init / omega_orb_sq);
        const auto delta_d   = (-a_t_add / omega_orb_sq) -
                               (-a_t_init / omega_orb_sq) +
                               ((v_t_cur + (j_t_cur / omega_orb_sq)) * dt);
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

    // Process taylor indices
    for (SizeType i : idx_taylor) {
        const auto lo      = i * kLeavesStride;
        const auto c_t_cur = leaves_tree[lo + 0];
        const auto s_t_cur = leaves_tree[lo + 2];
        const auto j_t_cur = leaves_tree[lo + 4];
        const auto a_t_cur = leaves_tree[lo + 6];
        const auto v_t_cur = leaves_tree[lo + 8];
        const auto f0      = leaves_tree[lo + 12];
        const auto a_new = a_t_cur + (j_t_cur * dt_add) +
                           (s_t_cur * half_dt2_add) + (c_t_cur * sixth_dt3_add);
        const auto delta_v = (a_t_cur * dt) + (j_t_cur * half_dt2) +
                             (s_t_cur * sixth_dt3) +
                             (c_t_cur * twenty_fourth_dt4);
        const auto delta_d =
            (v_t_cur * dt) + (a_t_cur * half_dt2) + (j_t_cur * sixth_dt3) +
            (s_t_cur * twenty_fourth_dt4) + (c_t_cur * onehundred_twenty_dt5);
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

void circ_taylor_ascend_resolve_batch(
    std::span<const double> leaves_tree,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    std::span<const ParamLimit> param_limits,
    std::span<const std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_segments,
    double minimum_snap_cells) {
    constexpr SizeType kParams       = 5;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(param_indices.size(),
                                     n_leaves * n_segments,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves * n_segments,
                                     "phase_shift size mismatch");
    error_check::check_equal(coord_segments.size(), n_segments,
                             "coord_segments size mismatch");
    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size should be 5");

    const auto& lim_accel = param_limits[3];
    const auto& lim_freq  = param_limits[4];
    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circ_taylor_mask(leaves_tree, n_leaves, kParams,
                             minimum_snap_cells);

    for (SizeType i = 0; i < n_segments; ++i) {
        auto param_indices_seg = param_indices.subspan(i * n_leaves, n_leaves);
        auto phase_shift_seg   = phase_shift.subspan(i * n_leaves, n_leaves);

        const auto [t0_cur, scale_cur] = coord_cur;
        const auto [t0_seg, scale_seg] = coord_segments[i];

        // Pre-compute constants to avoid repeated calculations
        const double dt                    = t0_seg - t0_cur;
        const double dt2                   = dt * dt;
        const double half_dt2              = 0.5 * dt2;
        const double sixth_dt3             = dt2 * dt / 6.0;
        const double twenty_fourth_dt4     = dt2 * dt2 / 24.0;
        const double onehundred_twenty_dt5 = (dt2 * dt2 * dt) / 120.0;

        // Process circular indices
        for (SizeType i : idx_circular_snap) {
            const SizeType lo  = i * kLeavesStride;
            const double s_cur = leaves_tree[lo + 2];
            const double j_cur = leaves_tree[lo + 4];
            const double a_cur = leaves_tree[lo + 6];
            const double v_cur = leaves_tree[lo + 8];
            const double d_cur = leaves_tree[lo + 10];
            const double f0    = leaves_tree[lo + 12];

            // Circular orbit mask condition
            const double omega_orb_sq = -s_cur / a_cur;
            const double omega_orb    = std::sqrt(omega_orb_sq);
            // Evolve the phase to the new time t_j = t_i + delta_t
            const double omega_dt = omega_orb * dt;
            const double cos_odt  = std::cos(omega_dt);
            const double sin_odt  = std::sin(omega_dt);
            const double a_new =
                (a_cur * cos_odt) + (j_cur * sin_odt / omega_orb);
            const double j_new =
                (j_cur * cos_odt) - (a_cur * sin_odt * omega_orb);
            const double v_circ_cur = -j_cur / omega_orb_sq;
            const double v_circ_new = -j_new / omega_orb_sq;
            const double d_circ_new = -a_new / omega_orb_sq;
            const double d_circ_cur = -a_cur / omega_orb_sq;
            const double v_new      = v_circ_new + (v_cur - v_circ_cur);
            const double d_new =
                d_circ_new + (d_cur - d_circ_cur) + ((v_cur - v_circ_cur) * dt);
            const double f_new     = f0 * (1.0 - (v_new * utils::kInvCval));
            const double delay_rel = d_new * utils::kInvCval;

            // Calculate relative phase
            phase_shift_seg[i] =
                psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
            // Find nearest grid indices
            const auto idx_a = psr_utils::get_nearest_idx_analytical(
                a_new, lim_accel, n_accel_init);
            const auto idx_f = psr_utils::get_nearest_idx_analytical(
                f_new, lim_freq, n_freq_init);
            param_indices_seg[i] = (idx_a * n_freq_init) + idx_f;
        }

        // Process circular crackle indices
        for (SizeType i : idx_circular_crackle) {
            const SizeType lo  = i * kLeavesStride;
            const double c_cur = leaves_tree[lo + 0];
            const double j_cur = leaves_tree[lo + 4];
            const double a_cur = leaves_tree[lo + 6];
            const double v_cur = leaves_tree[lo + 8];
            const double d_cur = leaves_tree[lo + 10];
            const double f0    = leaves_tree[lo + 12];

            // Circular orbit mask condition
            const double omega_orb_sq = -c_cur / j_cur;
            const double omega_orb    = std::sqrt(omega_orb_sq);
            // Evolve the phase to the new time t_j = t_i + delta_t
            const double omega_dt = omega_orb * dt;
            const double cos_odt  = std::cos(omega_dt);
            const double sin_odt  = std::sin(omega_dt);
            const double a_new =
                (a_cur * cos_odt) + (j_cur * sin_odt / omega_orb);
            const double j_new =
                (j_cur * cos_odt) - (a_cur * sin_odt * omega_orb);
            const double v_circ_cur = -j_cur / omega_orb_sq;
            const double v_circ_new = -j_new / omega_orb_sq;
            const double d_circ_new = -a_new / omega_orb_sq;
            const double d_circ_cur = -a_cur / omega_orb_sq;
            const double v_new      = v_circ_new + (v_cur - v_circ_cur);
            const double d_new =
                d_circ_new + (d_cur - d_circ_cur) + ((v_cur - v_circ_cur) * dt);
            const double f_new     = f0 * (1.0 - (v_new * utils::kInvCval));
            const double delay_rel = d_new * utils::kInvCval;

            // Calculate relative phase
            phase_shift_seg[i] =
                psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
            // Find nearest grid indices
            const auto idx_a = psr_utils::get_nearest_idx_analytical(
                a_new, lim_accel, n_accel_init);
            const auto idx_f = psr_utils::get_nearest_idx_analytical(
                f_new, lim_freq, n_freq_init);
            param_indices_seg[i] = (idx_a * n_freq_init) + idx_f;
        }

        // Process taylor indices
        for (SizeType i : idx_taylor) {
            const SizeType lo  = i * kLeavesStride;
            const double c_cur = leaves_tree[lo + 0];
            const double s_cur = leaves_tree[lo + 2];
            const double j_cur = leaves_tree[lo + 4];
            const double a_cur = leaves_tree[lo + 6];
            const double v_cur = leaves_tree[lo + 8];
            const double d_cur = leaves_tree[lo + 10];
            const double f0    = leaves_tree[lo + 12];
            const double a_new =
                a_cur + (j_cur * dt) + (s_cur * half_dt2) + (c_cur * sixth_dt3);
            const double v_new = v_cur + (a_cur * dt) + (j_cur * half_dt2) +
                                 (s_cur * sixth_dt3) +
                                 (c_cur * twenty_fourth_dt4);
            const double d_new = d_cur + (v_cur * dt) + (a_cur * half_dt2) +
                                 (j_cur * sixth_dt3) +
                                 (s_cur * twenty_fourth_dt4) +
                                 (c_cur * onehundred_twenty_dt5);
            // Calculates new frequency based on the first-order Doppler
            // approximation:
            const double f_new     = f0 * (1.0 - (v_new * utils::kInvCval));
            const double delay_rel = d_new * utils::kInvCval;

            // Calculate relative phase
            phase_shift_seg[i] =
                psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
            // Find nearest grid indices
            const auto idx_a = psr_utils::get_nearest_idx_analytical(
                a_new, lim_accel, n_accel_init);
            const auto idx_f = psr_utils::get_nearest_idx_analytical(
                f_new, lim_freq, n_freq_init);
            param_indices_seg[i] = (idx_a * n_freq_init) + idx_f;
        }
    }
}

void circ_taylor_transform_batch(std::span<double> leaves_tree,
                                 std::span<SizeType> indices_tree,
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
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const double dt                    = t0_next - t0_cur;
    const double dt2                   = dt * dt;
    const double half_dt2              = 0.5 * dt2;
    const double sixth_dt3             = dt2 * dt / 6.0;
    const double twenty_fourth_dt4     = dt2 * dt2 / 24.0;
    const double onehundred_twenty_dt5 = (dt2 * dt2 * dt) / 120.0;

    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circ_taylor_mask_scattered(leaves_tree, indices_tree, n_leaves,
                                       kParams, minimum_snap_cells);

    // Process circular indices
    for (SizeType i : idx_circular_snap) {
        const SizeType lo = i * kLeavesStride;
        const double d4_i = leaves_tree[lo + 2];
        const double d3_i = leaves_tree[lo + 4];
        const double d2_i = leaves_tree[lo + 6];
        const double d1_i = leaves_tree[lo + 8];
        const double d0_i = leaves_tree[lo + 10];

        const double omega_orb_sq = -d4_i / d2_i;
        const double omega_orb    = std::sqrt(omega_orb_sq);
        // Evolve the phase to the new time t_j = t_i + delta_t
        const double omega_dt = omega_orb * dt;
        const double cos_odt  = std::cos(omega_dt);
        const double sin_odt  = std::sin(omega_dt);
        // Pin-down omega using {d4, d2}
        const double d2_j = (d2_i * cos_odt) + (d3_i * sin_odt / omega_orb);
        const double d3_j = (d3_i * cos_odt) - (d2_i * sin_odt * omega_orb);
        const double d4_j = -omega_orb_sq * d2_j;
        const double d5_j = -omega_orb_sq * d3_j;
        // Integrate to get {v, d}
        const double v_circ_i = -d3_i / omega_orb_sq;
        const double v_circ_j = -d3_j / omega_orb_sq;
        const double d1_diff  = d1_i - v_circ_i;
        const double d1_j     = v_circ_j + d1_diff;
        const double d_circ_j = -d2_j / omega_orb_sq;
        const double d_circ_i = -d2_i / omega_orb_sq;
        const double d0_j     = d_circ_j + (d0_i - d_circ_i) + (d1_diff * dt);

        // Write back transformed values
        leaves_tree[lo + 0]  = d5_j;
        leaves_tree[lo + 2]  = d4_j;
        leaves_tree[lo + 4]  = d3_j;
        leaves_tree[lo + 6]  = d2_j;
        leaves_tree[lo + 8]  = d1_j;
        leaves_tree[lo + 10] = d0_j;
    }

    // Process circular crackle indices
    for (SizeType i : idx_circular_crackle) {
        const SizeType lo = i * kLeavesStride;
        const double d5_i = leaves_tree[lo + 0];
        const double d3_i = leaves_tree[lo + 4];
        const double d2_i = leaves_tree[lo + 6];
        const double d1_i = leaves_tree[lo + 8];
        const double d0_i = leaves_tree[lo + 10];

        const double omega_orb_sq = -d5_i / d3_i;
        const double omega_orb    = std::sqrt(omega_orb_sq);
        const double omega_dt     = omega_orb * dt;
        const double cos_odt      = std::cos(omega_dt);
        const double sin_odt      = std::sin(omega_dt);
        // Pin-down {s, j, a}
        const double d2_j = (d2_i * cos_odt) + (d3_i * sin_odt / omega_orb);
        const double d3_j = (d3_i * cos_odt) - (d2_i * sin_odt * omega_orb);
        const double d4_j = -omega_orb_sq * d2_j;
        const double d5_j = -omega_orb_sq * d3_j;
        // Integrate to get {v, d}
        const double v_circ_i = -d3_i / omega_orb_sq;
        const double v_circ_j = -d3_j / omega_orb_sq;
        const double d1_diff  = d1_i - v_circ_i;
        const double d1_j     = v_circ_j + d1_diff;
        const double d_circ_j = -d2_j / omega_orb_sq;
        const double d_circ_i = -d2_i / omega_orb_sq;
        const double d0_j     = d_circ_j + (d0_i - d_circ_i) + (d1_diff * dt);

        // Write back transformed values
        leaves_tree[lo + 0]  = d5_j;
        leaves_tree[lo + 2]  = d4_j;
        leaves_tree[lo + 4]  = d3_j;
        leaves_tree[lo + 6]  = d2_j;
        leaves_tree[lo + 8]  = d1_j;
        leaves_tree[lo + 10] = d0_j;
    }

    // Process normal indices
    for (SizeType i : idx_taylor) {
        const SizeType lo     = i * kLeavesStride;
        const double d5_val_i = leaves_tree[lo + 0];
        const double d4_val_i = leaves_tree[lo + 2];
        const double d3_val_i = leaves_tree[lo + 4];
        const double d2_val_i = leaves_tree[lo + 6];
        const double d1_val_i = leaves_tree[lo + 8];
        const double d0_val_i = leaves_tree[lo + 10];

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
    }

    // Process error leaves
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo      = indices_tree[i] * kLeavesStride;
        const auto c_err_i = leaves_tree[lo + 1];
        const auto s_err_i = leaves_tree[lo + 3];
        const auto j_err_i = leaves_tree[lo + 5];
        const auto a_err_i = leaves_tree[lo + 7];
        const auto v_err_i = leaves_tree[lo + 9];
        const auto d_err_i = leaves_tree[lo + 11];

        // Transform errors
        double c_err_j, s_err_j, j_err_j, a_err_j, v_err_j;
        if (use_conservative_tile) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            c_err_j = c_err_i;
            s_err_j =
                std::sqrt((s_err_i * s_err_i) + (c_err_i * c_err_i * dt2));
            j_err_j =
                std::sqrt((j_err_i * j_err_i) + (s_err_i * s_err_i * dt2) +
                          (c_err_i * c_err_i * half_dt2 * half_dt2));
            a_err_j =
                std::sqrt((a_err_i * a_err_i) + (j_err_i * j_err_i * dt2) +
                          (s_err_i * s_err_i * half_dt2 * half_dt2) +
                          (c_err_i * c_err_i * sixth_dt3 * sixth_dt3));
            v_err_j = std::sqrt(
                (v_err_i * v_err_i) + (a_err_i * a_err_i * dt2) +
                (j_err_i * j_err_i * half_dt2 * half_dt2) +
                (s_err_i * s_err_i * sixth_dt3 * sixth_dt3) +
                (c_err_i * c_err_i * twenty_fourth_dt4 * twenty_fourth_dt4));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            c_err_j = c_err_i;
            s_err_j = s_err_i;
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
        }

        // Write back transformed values
        leaves_tree[lo + 1]  = c_err_j;
        leaves_tree[lo + 3]  = s_err_j;
        leaves_tree[lo + 5]  = j_err_j;
        leaves_tree[lo + 7]  = a_err_j;
        leaves_tree[lo + 9]  = v_err_j;
        leaves_tree[lo + 11] = d_err_i;
    }
}

std::vector<double>
generate_bp_circ_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        bool use_conservative_tile) {
    constexpr SizeType kParamsExpected = 5;
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr must have 5 parameters");
    error_check::check_equal(dparams.size(), kParamsExpected,
                             "dparams must have 5 parameters");
    const auto n_params  = dparams.size();
    const auto& f0_batch = param_arr.back(); // Last array is frequency
    const auto n_freqs   = f0_batch.size();  // Number of frequency bins

    // Snail Scheme
    psr_utils::MiddleOutScheme scheme(nsegments, ref_seg, tseg_ffa);
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

    // Track when first snap branching occurs for each frequency
    std::vector<bool> snap_first_branched(n_freqs, false);
    std::vector<bool> snap_active_mask(n_freqs, false);
    std::vector<double> dparam_new_batch(n_freqs * n_params, 0.0);
    std::vector<double> shift_bins_batch(n_freqs * n_params, 0.0);
    std::vector<double> dparam_cur_next(n_freqs * n_params, 0.0);
    std::vector<double> n_branches(n_freqs, 1.0);
    std::vector<double> n_branches_snap(n_freqs, 1.0);
    const auto n_params_d = n_params + 1;
    std::vector<double> dparam_d_vec(n_freqs * n_params_d, 0.0);

    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = scheme.get_coord(prune_level);
        const auto coord_cur  = scheme.get_current_coord(prune_level);
        const auto [t0_cur, t_obs_minus_t_ref] = coord_cur;

        // Calculate optimal parameter steps and shift bins
        psr_utils::poly_taylor_step_d_vec(n_params, t_obs_minus_t_ref, nbins,
                                          eta, f0_batch, dparam_new_batch, 0);
        psr_utils::poly_taylor_shift_d_vec(
            dparam_cur_batch, dparam_new_batch, t_obs_minus_t_ref, nbins,
            f0_batch, 0, shift_bins_batch, n_freqs, n_params);

        std::ranges::fill(n_branches, 1.0);
        std::ranges::fill(n_branches_snap, 1.0);
        // Determine branching needs
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 1; j < n_params; ++j) {
                const auto idx = (i * n_params) + j;
                if (shift_bins_batch[idx] < (eta - utils::kFloatEps)) {
                    dparam_cur_next[idx] = dparam_cur_batch[idx];
                    continue;
                }
                const auto ratio =
                    (dparam_cur_batch[idx]) / (dparam_new_batch[idx]);
                const SizeType num_points = std::max(
                    1UL,
                    static_cast<SizeType>(std::ceil(ratio - utils::kFloatEps)));
                n_branches[i] *= static_cast<double>(num_points);
                dparam_cur_next[idx] =
                    dparam_cur_batch[idx] / static_cast<double>(num_points);
                if (j == 1) {
                    n_branches_snap[i] *= static_cast<double>(num_points);
                }
            }
        }
        // Determine validation fraction
        for (SizeType i = 0; i < n_freqs; ++i) {
            const auto snap_active = n_branches_snap[i] > 1.0;
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