#include "loki/core/taylor.hpp"

#include <algorithm>
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

std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve(std::span<const double> pset_cur,
                   std::span<const std::vector<double>> param_arr,
                   SizeType ffa_level,
                   SizeType latter,
                   double tseg_brute,
                   SizeType nbins) {
    const auto nparams = pset_cur.size();
    std::vector<double> pset_prev(nparams, 0.0);
    const auto tsegment = std::pow(2.0, ffa_level - 1) * tseg_brute;
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
    SizeType hint_idx{};
    for (SizeType ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] = utils::find_nearest_sorted_idx_scan(
            std::span(param_arr[ip]), pset_prev[ip], hint_idx);
    }
    return {pindex_prev, relative_phase};
}

void ffa_taylor_resolve_freq_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins) {
    const auto nparams = param_arr_prev.size();
    error_check::check_equal(
        nparams, param_arr_cur.size(),
        "param_arr_cur and param_arr_prev should have the same size");
    error_check::check_equal(nparams, 1U,
                             "nparams should be 1 for frequency resolve");
    const auto& freq_arr = param_arr_cur[0];
    const auto n_freq    = freq_arr.size();
    error_check::check_equal(pindex_prev_flat_batch.size(), n_freq,
                             "pindex_prev_flat size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_freq,
                             "relative_phase_batch size mismatch");

    const auto delta_t = std::pow(2.0, ffa_level - 1) * tseg_brute;

    // Calculate relative phases and flattened parameter indices
    SizeType hint_f = 0;
    for (SizeType i = 0; i < n_freq; ++i) {
        const auto idx_f = utils::find_nearest_sorted_idx_scan(
            std::span(param_arr_prev[0]), freq_arr[i], hint_f);
        pindex_prev_flat_batch[i] = static_cast<uint32_t>(idx_f);
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, freq_arr[i], nbins, 0.0);
    }
}

void ffa_taylor_resolve_accel_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins) {
    const auto nparams = param_arr_prev.size();
    error_check::check_equal(
        param_arr_cur.size(), nparams,
        "param_arr_cur and param_arr_prev should have the same size");
    error_check::check_equal(nparams, 2U,
                             "nparams should be 2 for accel resolve");

    // parameter access patterns
    const auto& accel_arr_cur  = param_arr_cur[0];
    const auto& freq_arr_cur   = param_arr_cur[1];
    const auto& accel_arr_prev = param_arr_prev[0];
    const auto& freq_arr_prev  = param_arr_prev[1];
    const auto n_accel         = param_arr_cur[0].size();
    const auto n_freq          = param_arr_cur[1].size();
    const auto n_freq_prev     = param_arr_prev[1].size();
    const auto ncoords         = n_accel * n_freq;
    error_check::check_equal(pindex_prev_flat_batch.size(), ncoords,
                             "pindex_prev_flat size mismatch");
    error_check::check_equal(relative_phase_batch.size(), ncoords,
                             "relative_phase_batch size mismatch");

    const auto tsegment = std::pow(2.0, ffa_level - 1) * tseg_brute;
    const auto delta_t  = (static_cast<double>(latter) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto half_delta_t_sq = 0.5 * delta_t * delta_t;

    SizeType hint_a = 0;
    for (SizeType accel_idx = 0; accel_idx < n_accel; ++accel_idx) {
        const auto a_cur = accel_arr_cur[accel_idx];
        const auto a_new = a_cur;
        const auto v_new = a_cur * delta_t;
        const auto d_new = a_cur * half_delta_t_sq;
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_prev, a_new, hint_a);

        SizeType hint_f = 0;
        for (SizeType freq_idx = 0; freq_idx < n_freq; ++freq_idx) {
            const auto coord_idx = (accel_idx * n_freq) + freq_idx;
            const auto f_cur     = freq_arr_cur[freq_idx];
            const auto f_new     = f_cur * (1.0 - v_new * utils::kInvCval);
            const auto delay_rel = d_new * utils::kInvCval;

            relative_phase_batch[coord_idx] =
                psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);
            const auto idx_f = utils::find_nearest_sorted_idx_scan(
                freq_arr_prev, f_new, hint_f);

            pindex_prev_flat_batch[coord_idx] =
                static_cast<uint32_t>((idx_a * n_freq_prev) + idx_f);
        }
    }
}

void ffa_taylor_resolve_jerk_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins) {
    const auto nparams = param_arr_prev.size();
    error_check::check_equal(
        param_arr_cur.size(), nparams,
        "param_arr_cur and param_arr_prev should have the same size");
    error_check::check_equal(nparams, 3U,
                             "nparams should be 3 for jerk resolve");

    const auto n_jerk       = param_arr_cur[0].size();
    const auto n_accel      = param_arr_cur[1].size();
    const auto n_freq       = param_arr_cur[2].size();
    const auto ncoords      = n_jerk * n_accel * n_freq;
    const auto n_accel_prev = param_arr_prev[1].size();
    const auto n_freq_prev  = param_arr_prev[2].size();

    error_check::check_equal(pindex_prev_flat_batch.size(), ncoords,
                             "pindex_prev_flat size mismatch");
    error_check::check_equal(relative_phase_batch.size(), ncoords,
                             "relative_phase_batch size mismatch");

    const auto tsegment = std::pow(2.0, ffa_level - 1) * tseg_brute;
    const auto delta_t  = (static_cast<double>(latter) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_sq          = delta_t * delta_t;
    const auto delta_t_cubed       = delta_t_sq * delta_t;
    const auto half_delta_t_sq     = 0.5 * delta_t_sq;
    const auto sixth_delta_t_cubed = delta_t_cubed / 6.0;

    // Cache-friendly access patterns
    const auto& jerk_arr_cur   = param_arr_cur[0];
    const auto& accel_arr_cur  = param_arr_cur[1];
    const auto& freq_arr_cur   = param_arr_cur[2];
    const auto& jerk_arr_prev  = param_arr_prev[0];
    const auto& accel_arr_prev = param_arr_prev[1];
    const auto& freq_arr_prev  = param_arr_prev[2];

    const auto jerk_stride_prev  = n_accel_prev * n_freq_prev;
    const auto accel_stride_prev = n_freq_prev;

    // Separate hints for better search performance
    SizeType hint_j = 0;

    for (SizeType jerk_idx = 0; jerk_idx < n_jerk; ++jerk_idx) {
        const auto j_cur = jerk_arr_cur[jerk_idx];
        const auto j_new = j_cur; // No transformation needed

        // Pre-compute jerk-related terms for this jerk value
        const auto j_delta_t             = j_cur * delta_t;
        const auto half_j_delta_t_sq     = 0.5 * j_cur * delta_t_sq;
        const auto j_sixth_delta_t_cubed = j_cur * sixth_delta_t_cubed;

        // Find jerk index once per jerk_idx
        const auto idx_j = utils::find_nearest_sorted_idx_scan(
            std::span(jerk_arr_prev), j_new, hint_j);

        SizeType hint_a = 0;

        for (SizeType accel_idx = 0; accel_idx < n_accel; ++accel_idx) {
            const auto a_cur = accel_arr_cur[accel_idx];

            // Compute acceleration-related terms
            const auto a_new = a_cur + j_delta_t;
            const auto v_new = (a_cur * delta_t) + half_j_delta_t_sq;
            const auto d_new =
                (a_cur * half_delta_t_sq) + j_sixth_delta_t_cubed;

            // Find accel index once per (jerk_idx, accel_idx) pair
            const auto idx_a = utils::find_nearest_sorted_idx_scan(
                std::span(accel_arr_prev), a_new, hint_a);

            // Pre-compute stride calculation
            const auto jerk_accel_offset =
                (idx_j * jerk_stride_prev) + (idx_a * accel_stride_prev);

            SizeType hint_f = 0;

            for (SizeType freq_idx = 0; freq_idx < n_freq; ++freq_idx) {
                const auto coord_idx =
                    ((jerk_idx * n_accel + accel_idx) * n_freq) + freq_idx;
                const auto f_cur = freq_arr_cur[freq_idx];

                // Frequency-specific calculations
                const auto f_new     = f_cur * (1.0 - v_new * utils::kInvCval);
                const auto delay_rel = d_new * utils::kInvCval;
                const auto relative_phase =
                    psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);

                const auto idx_f = utils::find_nearest_sorted_idx_scan(
                    std::span(freq_arr_prev), f_new, hint_f);

                pindex_prev_flat_batch[coord_idx] =
                    static_cast<uint32_t>(jerk_accel_offset + idx_f);
                relative_phase_batch[coord_idx] = relative_phase;
            }
        }
    }
}

void ffa_taylor_resolve_snap_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins) {
    const auto nparams = param_arr_prev.size();
    error_check::check_equal(
        param_arr_cur.size(), nparams,
        "param_arr_cur and param_arr_prev should have the same size");
    error_check::check_equal(nparams, 4U,
                             "nparams should be 4 for snap resolve");

    // Since snap is never used, we simply adapt the call to
    // jerk resolve
    const auto nparams_jerk = nparams - 1;
    std::vector<std::vector<double>> param_arr_cur_jerk(nparams_jerk);
    std::vector<std::vector<double>> param_arr_prev_jerk(nparams_jerk);
    for (SizeType i = 0; i < nparams_jerk; ++i) {
        param_arr_cur_jerk[i]  = param_arr_cur[1 + i];
        param_arr_prev_jerk[i] = param_arr_prev[1 + i];
    }
    ffa_taylor_resolve_jerk_batch(param_arr_cur_jerk, param_arr_prev_jerk,
                                  pindex_prev_flat_batch, relative_phase_batch,
                                  ffa_level, latter, tseg_brute, nbins);
}

void ffa_taylor_resolve_crackle_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins) {
    const auto nparams = param_arr_prev.size();
    error_check::check_equal(
        param_arr_cur.size(), nparams,
        "param_arr_cur and param_arr_prev should have the same size");
    error_check::check_equal(nparams, 5U,
                             "nparams should be 5 for crackle resolve");

    // Since crackle is never used, we simply adapt the call to
    // jerk resolve
    const auto nparams_jerk = nparams - 2;
    std::vector<std::vector<double>> param_arr_cur_jerk(nparams_jerk);
    std::vector<std::vector<double>> param_arr_prev_jerk(nparams_jerk);
    for (SizeType i = 0; i < nparams_jerk; ++i) {
        param_arr_cur_jerk[i]  = param_arr_cur[2 + i];
        param_arr_prev_jerk[i] = param_arr_prev[2 + i];
    }
    ffa_taylor_resolve_jerk_batch(param_arr_cur_jerk, param_arr_prev_jerk,
                                  pindex_prev_flat_batch, relative_phase_batch,
                                  ffa_level, latter, tseg_brute, nbins);
}

SizeType poly_taylor_seed(std::span<const std::vector<double>> param_arr,
                          std::span<const double> dparams,
                          SizeType poly_order,
                          std::pair<double, double> /*coord_init*/,
                          std::span<double> seed_leaves) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (poly_order + 2) * kParamStride;
    const auto n_params             = param_arr.size();
    SizeType n_leaves               = 1;
    for (const auto& arr : param_arr) {
        n_leaves *= arr.size();
    }
    error_check::check_equal(n_params, poly_order,
                             "n_params should be equal to poly_order");
    // Create parameter sets tensor: (n_param_sets, poly_order + 2, 2)
    error_check::check_greater_equal(seed_leaves.size(),
                                     n_leaves * leaves_stride,
                                     "seed_leaves size mismatch");
    SizeType leaf_idx = 0;
    for (const auto& p_set_view : utils::cartesian_product_view(param_arr)) {
        const auto leaf_offset = leaf_idx * leaves_stride;
        const auto f0          = p_set_view[n_params - 1];
        const auto df          = dparams[n_params - 1];
        // Copy till d2 (acceleration)
        for (SizeType j = 0; j < n_params - 1; ++j) {
            seed_leaves[leaf_offset + (j * kParamStride) + 0] = p_set_view[j];
            seed_leaves[leaf_offset + (j * kParamStride) + 1] = dparams[j];
        }
        // Update frequency to velocity
        // f = f0(1 - v / C) => dv = -(C/f0) * df
        seed_leaves[leaf_offset + ((n_params - 1) * kParamStride) + 0] = 0;
        seed_leaves[leaf_offset + ((n_params - 1) * kParamStride) + 1] =
            df * (utils::kCval / f0);
        // intialize d0 (measure from t=t_init) and store f0
        seed_leaves[leaf_offset + ((n_params + 0) * kParamStride) + 0] = 0;
        seed_leaves[leaf_offset + ((n_params + 0) * kParamStride) + 1] = 0;
        seed_leaves[leaf_offset + ((n_params + 1) * kParamStride) + 0] = f0;
        // Store basis flag (0: Polynomial, 1: Physical)
        seed_leaves[leaf_offset + ((n_params + 1) * kParamStride) + 1] = 0;
        ++leaf_idx;
    }
    error_check::check_equal(leaf_idx, n_leaves, "n_leaves mismatch");
    return n_leaves;
}

std::vector<SizeType>
poly_taylor_branch_batch(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch_batch,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType nbins,
                         double eta,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (n_params + 2) * kParamStride;
    error_check::check_equal(leaves_batch.size(), n_batch * leaves_stride,
                             "leaves_batch size mismatch");
    error_check::check_greater_equal(leaves_branch_batch.size(),
                                     n_batch * branch_max * leaves_stride,
                                     "leaves_branch_batch size mismatch");

    const auto [t0_cur, t_obs_minus_t_ref] = coord_cur;

    // Use leaves_branch_batch memory as workspace. Partition workspace into
    // sections:
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

    for (SizeType i = 0; i < n_batch; ++i) {
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
                                       shift_bins_batch, n_batch, n_params);

    std::vector<double> pad_branched_dparams(n_batch * n_params);
    std::vector<SizeType> branched_counts(n_batch * n_params);
    // Optimized branching loop - same logic as original but vectorized access
    constexpr double kEps = 1e-12;
    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType leaf_offset = i * leaves_stride;
        const SizeType flat_base   = i * n_params;

        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType flat_idx     = flat_base + j;
            const SizeType param_offset = leaf_offset + (j * kParamStride);
            const double param_cur_val  = leaves_batch[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];

            if (shift_bins_batch[flat_idx] >= (eta - kEps)) {
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
                                     n_batch, n_params, branch_max);
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

SizeType
poly_taylor_branch_accel_batch(std::span<const double> leaves_tree,
                               std::pair<double, double> coord_cur,
                               std::span<double> leaves_branch,
                               std::span<SizeType> leaves_origins,
                               SizeType n_leaves,
                               SizeType n_params,
                               SizeType nbins,
                               double eta,
                               const std::vector<ParamLimitType>& param_limits,
                               SizeType branch_max,
                               std::span<double> scratch_params,
                               std::span<double> scratch_dparams,
                               std::span<SizeType> scratch_counts) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(n_params, kParams,
                             "nparams should be 2 for accel branch");
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
    constexpr double kEps = 1e-12;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * n_params;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");
    error_check::check_greater_equal(scratch_params.size(),
                                     n_leaves * n_params * branch_max,
                                     "scratch_params size mismatch");
    error_check::check_greater_equal(scratch_dparams.size(),
                                     n_leaves * n_params,
                                     "scratch_dparams size mismatch");
    error_check::check_greater_equal(scratch_counts.size(), n_leaves * n_params,
                                     "scratch_counts size mismatch");

    const double* __restrict leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict leaves_origins_ptr  = leaves_origins.data();
    double* __restrict leaves_branch_ptr     = leaves_branch.data();
    double* __restrict dparam_new_ptr        = dparam_new.data();
    double* __restrict shift_bins_ptr        = shift_bins.data();
    double* __restrict scratch_params_ptr    = scratch_params.data();
    double* __restrict scratch_dparams_ptr   = scratch_dparams.data();
    SizeType* __restrict scratch_counts_ptr  = scratch_counts.data();

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

        const auto d2_cur     = leaves_tree_ptr[leaf_offset + 0];
        const auto d2_sig_cur = leaves_tree_ptr[leaf_offset + 1];
        const auto d1_cur     = leaves_tree_ptr[leaf_offset + 2];
        const auto d1_sig_cur = leaves_tree_ptr[leaf_offset + 3];
        const auto f0         = leaves_tree_ptr[leaf_offset + 6];
        const auto d2_sig_new = dparam_new_ptr[flat_base + 0];
        const auto d1_sig_new = dparam_new_ptr[flat_base + 1];

        // Branch d2 parameter
        {
            const SizeType pad_offset = (flat_base + 0) * branch_max;
            if (shift_bins_ptr[flat_base + 0] >= (eta - kEps)) {
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d2_cur, d2_sig_cur, d2_sig_new,
                    param_limits[0][0], param_limits[0][1]);
                scratch_dparams_ptr[flat_base] = dparam_act;
                scratch_counts_ptr[flat_base]  = count;
            } else {
                // No branching: only use current value
                scratch_params_ptr[pad_offset] = d2_cur;
                scratch_dparams_ptr[flat_base] = d2_sig_cur;
                scratch_counts_ptr[flat_base]  = 1;
            }
        }

        // Branch d1 parameter
        {

            const SizeType pad_offset = (flat_base + 1) * branch_max;
            if (shift_bins_ptr[flat_base + 1] >= (eta - kEps)) {
                const double d1_min =
                    (1 - param_limits[1][1] / f0) * utils::kCval;
                const double d1_max =
                    (1 - param_limits[1][0] / f0) * utils::kCval;
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max);
                scratch_dparams_ptr[flat_base + 1] = dparam_act;
                scratch_counts_ptr[flat_base + 1]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d1_cur;
                scratch_dparams_ptr[flat_base + 1] = d1_sig_cur;
                scratch_counts_ptr[flat_base + 1]  = 1;
            }
        }
    }

    // --- Loop 3: Fill leaves_origins ---
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset   = i * kLeavesStride;
        const SizeType flat_base     = i * kParams;
        const SizeType n_d2_branches = scratch_counts_ptr[flat_base + 0];
        const SizeType n_d1_branches = scratch_counts_ptr[flat_base + 1];
        const SizeType d2_offset     = (flat_base + 0) * branch_max;
        const SizeType d1_offset     = (flat_base + 1) * branch_max;

        for (SizeType a = 0; a < n_d2_branches; ++a) {
            for (SizeType b = 0; b < n_d1_branches; ++b) {
                const SizeType branch_offset = out_leaves * kLeavesStride;
                leaves_branch_ptr[branch_offset + 0] =
                    scratch_params_ptr[d2_offset + a];
                leaves_branch_ptr[branch_offset + 1] =
                    scratch_dparams_ptr[flat_base + 0];
                leaves_branch_ptr[branch_offset + 2] =
                    scratch_params_ptr[d1_offset + b];
                leaves_branch_ptr[branch_offset + 3] =
                    scratch_dparams_ptr[flat_base + 1];
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

SizeType
poly_taylor_branch_jerk_batch(std::span<const double> leaves_tree,
                              std::pair<double, double> coord_cur,
                              std::span<double> leaves_branch,
                              std::span<SizeType> leaves_origins,
                              SizeType n_leaves,
                              SizeType n_params,
                              SizeType nbins,
                              double eta,
                              const std::vector<ParamLimitType>& param_limits,
                              SizeType branch_max,
                              std::span<double> scratch_params,
                              std::span<double> scratch_dparams,
                              std::span<SizeType> scratch_counts) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(n_params, kParams,
                             "nparams should be 3 for jerk branch");
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
    constexpr double kEps = 1e-12;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * n_params;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");
    error_check::check_greater_equal(scratch_params.size(),
                                     n_leaves * n_params * branch_max,
                                     "scratch_params size mismatch");
    error_check::check_greater_equal(scratch_dparams.size(),
                                     n_leaves * n_params,
                                     "scratch_dparams size mismatch");
    error_check::check_greater_equal(scratch_counts.size(), n_leaves * n_params,
                                     "scratch_counts size mismatch");

    const double* __restrict leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict leaves_origins_ptr  = leaves_origins.data();
    double* __restrict leaves_branch_ptr     = leaves_branch.data();
    double* __restrict dparam_new_ptr        = dparam_new.data();
    double* __restrict shift_bins_ptr        = shift_bins.data();
    double* __restrict scratch_params_ptr    = scratch_params.data();
    double* __restrict scratch_dparams_ptr   = scratch_dparams.data();
    SizeType* __restrict scratch_counts_ptr  = scratch_counts.data();

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

        // Branch d3 parameter
        {
            const SizeType pad_offset = (flat_base + 0) * branch_max;
            if (shift_bins_ptr[flat_base + 0] >= (eta - kEps)) {
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d3_cur, d3_sig_cur, d3_sig_new,
                    param_limits[0][0], param_limits[0][1]);
                scratch_dparams_ptr[flat_base + 0] = dparam_act;
                scratch_counts_ptr[flat_base + 0]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d3_cur;
                scratch_dparams_ptr[flat_base + 0] = d3_sig_cur;
                scratch_counts_ptr[flat_base + 0]  = 1;
            }
        }

        // Branch d2 parameter
        {
            const SizeType pad_offset = (flat_base + 1) * branch_max;
            if (shift_bins_ptr[flat_base + 1] >= (eta - kEps)) {
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d2_cur, d2_sig_cur, d2_sig_new,
                    param_limits[1][0], param_limits[1][1]);
                scratch_dparams_ptr[flat_base + 1] = dparam_act;
                scratch_counts_ptr[flat_base + 1]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d2_cur;
                scratch_dparams_ptr[flat_base + 1] = d2_sig_cur;
                scratch_counts_ptr[flat_base + 1]  = 1;
            }
        }

        // Branch d1 parameter
        {
            const SizeType pad_offset = (flat_base + 2) * branch_max;
            if (shift_bins_ptr[flat_base + 2] >= (eta - kEps)) {
                const double d1_min =
                    (1 - param_limits[2][1] / f0) * utils::kCval;
                const double d1_max =
                    (1 - param_limits[2][0] / f0) * utils::kCval;
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max);
                scratch_dparams_ptr[flat_base + 2] = dparam_act;
                scratch_counts_ptr[flat_base + 2]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d1_cur;
                scratch_dparams_ptr[flat_base + 2] = d1_sig_cur;
                scratch_counts_ptr[flat_base + 2]  = 1;
            }
        }
    }

    // --- Loop 3: Fill leaves_origins (3D Cartesian product) ---
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset   = i * kLeavesStride;
        const SizeType flat_base     = i * kParams;
        const SizeType n_d3_branches = scratch_counts_ptr[flat_base + 0];
        const SizeType n_d2_branches = scratch_counts_ptr[flat_base + 1];
        const SizeType n_d1_branches = scratch_counts_ptr[flat_base + 2];
        const SizeType d3_offset     = (flat_base + 0) * branch_max;
        const SizeType d2_offset     = (flat_base + 1) * branch_max;
        const SizeType d1_offset     = (flat_base + 2) * branch_max;

        for (SizeType a = 0; a < n_d3_branches; ++a) {
            for (SizeType b = 0; b < n_d2_branches; ++b) {
                for (SizeType c = 0; c < n_d1_branches; ++c) {
                    const SizeType branch_offset = out_leaves * kLeavesStride;
                    leaves_branch_ptr[branch_offset + 0] =
                        scratch_params_ptr[d3_offset + a];
                    leaves_branch_ptr[branch_offset + 1] =
                        scratch_dparams_ptr[flat_base + 0];
                    leaves_branch_ptr[branch_offset + 2] =
                        scratch_params_ptr[d2_offset + b];
                    leaves_branch_ptr[branch_offset + 3] =
                        scratch_dparams_ptr[flat_base + 1];
                    leaves_branch_ptr[branch_offset + 4] =
                        scratch_params_ptr[d1_offset + c];
                    leaves_branch_ptr[branch_offset + 5] =
                        scratch_dparams_ptr[flat_base + 2];
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

SizeType
poly_taylor_branch_snap_batch(std::span<const double> leaves_tree,
                              std::pair<double, double> coord_cur,
                              std::span<double> leaves_branch,
                              std::span<SizeType> leaves_origins,
                              SizeType n_leaves,
                              SizeType n_params,
                              SizeType nbins,
                              double eta,
                              const std::vector<ParamLimitType>& param_limits,
                              SizeType branch_max,
                              std::span<double> scratch_params,
                              std::span<double> scratch_dparams,
                              std::span<SizeType> scratch_counts) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(n_params, kParams,
                             "nparams should be 4 for snap branch");
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
    constexpr double kEps = 1e-12;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * n_params;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");
    error_check::check_greater_equal(scratch_params.size(),
                                     n_leaves * n_params * branch_max,
                                     "scratch_params size mismatch");
    error_check::check_greater_equal(scratch_dparams.size(),
                                     n_leaves * n_params,
                                     "scratch_dparams size mismatch");
    error_check::check_greater_equal(scratch_counts.size(), n_leaves * n_params,
                                     "scratch_counts size mismatch");

    const double* __restrict leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict leaves_origins_ptr  = leaves_origins.data();
    double* __restrict leaves_branch_ptr     = leaves_branch.data();
    double* __restrict dparam_new_ptr        = dparam_new.data();
    double* __restrict shift_bins_ptr        = shift_bins.data();
    double* __restrict scratch_params_ptr    = scratch_params.data();
    double* __restrict scratch_dparams_ptr   = scratch_dparams.data();
    SizeType* __restrict scratch_counts_ptr  = scratch_counts.data();

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

        // Branch d4 parameter
        {
            const SizeType pad_offset = (flat_base + 0) * branch_max;
            if (shift_bins_ptr[flat_base + 0] >= (eta - kEps)) {
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d4_cur, d4_sig_cur, d4_sig_new,
                    param_limits[0][0], param_limits[0][1]);
                scratch_dparams_ptr[flat_base + 0] = dparam_act;
                scratch_counts_ptr[flat_base + 0]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d4_cur;
                scratch_dparams_ptr[flat_base + 0] = d4_sig_cur;
                scratch_counts_ptr[flat_base + 0]  = 1;
            }
        }

        // Branch d3 parameter
        {
            const SizeType pad_offset = (flat_base + 1) * branch_max;
            if (shift_bins_ptr[flat_base + 1] >= (eta - kEps)) {
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d3_cur, d3_sig_cur, d3_sig_new,
                    param_limits[1][0], param_limits[1][1]);
                scratch_dparams_ptr[flat_base + 1] = dparam_act;
                scratch_counts_ptr[flat_base + 1]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d3_cur;
                scratch_dparams_ptr[flat_base + 1] = d3_sig_cur;
                scratch_counts_ptr[flat_base + 1]  = 1;
            }
        }

        // Branch d2 parameter
        {
            const SizeType pad_offset = (flat_base + 2) * branch_max;
            if (shift_bins_ptr[flat_base + 2] >= (eta - kEps)) {
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d2_cur, d2_sig_cur, d2_sig_new,
                    param_limits[2][0], param_limits[2][1]);
                scratch_dparams_ptr[flat_base + 2] = dparam_act;
                scratch_counts_ptr[flat_base + 2]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d2_cur;
                scratch_dparams_ptr[flat_base + 2] = d2_sig_cur;
                scratch_counts_ptr[flat_base + 2]  = 1;
            }
        }

        // Branch d1 parameter
        {
            const SizeType pad_offset = (flat_base + 3) * branch_max;
            if (shift_bins_ptr[flat_base + 3] >= (eta - kEps)) {
                const double d1_min =
                    (1 - param_limits[3][1] / f0) * utils::kCval;
                const double d1_max =
                    (1 - param_limits[3][0] / f0) * utils::kCval;
                auto slice_span = std::span<double>(
                    scratch_params_ptr + pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, d1_cur, d1_sig_cur, d1_sig_new, d1_min, d1_max);
                scratch_dparams_ptr[flat_base + 3] = dparam_act;
                scratch_counts_ptr[flat_base + 3]  = count;
            } else {
                scratch_params_ptr[pad_offset]     = d1_cur;
                scratch_dparams_ptr[flat_base + 3] = d1_sig_cur;
                scratch_counts_ptr[flat_base + 3]  = 1;
            }
        }
    }

    // --- Loop 3: Fill leaves_origins (4D Cartesian product) ---
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset   = i * kLeavesStride;
        const SizeType flat_base     = i * kParams;
        const SizeType n_d4_branches = scratch_counts_ptr[flat_base + 0];
        const SizeType n_d3_branches = scratch_counts_ptr[flat_base + 1];
        const SizeType n_d2_branches = scratch_counts_ptr[flat_base + 2];
        const SizeType n_d1_branches = scratch_counts_ptr[flat_base + 3];
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
                            scratch_params_ptr[d4_offset + a];
                        leaves_branch_ptr[branch_offset + 1] =
                            scratch_dparams_ptr[flat_base + 0];
                        leaves_branch_ptr[branch_offset + 2] =
                            scratch_params_ptr[d3_offset + b];
                        leaves_branch_ptr[branch_offset + 3] =
                            scratch_dparams_ptr[flat_base + 1];
                        leaves_branch_ptr[branch_offset + 4] =
                            scratch_params_ptr[d2_offset + c];
                        leaves_branch_ptr[branch_offset + 5] =
                            scratch_dparams_ptr[flat_base + 2];
                        leaves_branch_ptr[branch_offset + 6] =
                            scratch_params_ptr[d1_offset + d];
                        leaves_branch_ptr[branch_offset + 7] =
                            scratch_dparams_ptr[flat_base + 3];
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

void poly_taylor_resolve_accel_batch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params) {
    constexpr SizeType kParamsExpected = 2;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 2 for accel resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 2 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(pindex_flat_batch.size(), n_leaves,
                                     "param_idx_flat_batch size mismatch");
    error_check::check_greater_equal(relative_phase_batch.size(), n_leaves,
                                     "relative_phase_batch size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[0];
    const auto& freq_arr_grid  = param_arr[1];
    const auto n_freq          = param_arr[1].size();

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_add  = t0_add - t0_cur;
    const auto delta_t_init = t0_init - t0_cur;
    const auto delta_t      = delta_t_add - delta_t_init;
    const auto half_delta_t_sq =
        0.5 * (delta_t_add * delta_t_add - delta_t_init * delta_t_init);

    SizeType hint_a = 0, hint_f = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto a_t_cur     = leaves_batch[leaf_offset + (0 * kParamStride)];
        const auto v_t_cur     = leaves_batch[leaf_offset + (1 * kParamStride)];
        const auto f0          = leaves_batch[leaf_offset + (3 * kParamStride)];
        const auto a_new       = a_t_cur;
        const auto delta_v_new = a_t_cur * delta_t;
        const auto delta_d_new =
            (v_t_cur * delta_t) + (a_t_cur * half_delta_t_sq);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - delta_v_new * utils::kInvCval);
        const auto delay_rel = delta_d_new * utils::kInvCval;

        // Calculate relative phase
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
    }
}

void poly_taylor_resolve_jerk_batch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params) {
    constexpr SizeType kParamsExpected = 3;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 3 for jerk resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 3 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(pindex_flat_batch.size(), n_leaves,
                                     "param_idx_flat_batch size mismatch");
    error_check::check_greater_equal(relative_phase_batch.size(), n_leaves,
                                     "relative_phase_batch size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[1];
    const auto& freq_arr_grid  = param_arr[2];
    const auto n_freq          = param_arr[2].size();

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

    SizeType hint_a = 0, hint_f = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto j_t_cur = leaves_batch[leaf_offset + (0 * kParamStride)];
        const auto a_t_cur = leaves_batch[leaf_offset + (1 * kParamStride)];
        const auto v_t_cur = leaves_batch[leaf_offset + (2 * kParamStride)];
        const auto f0      = leaves_batch[leaf_offset + (4 * kParamStride)];
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
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
    }
}

void poly_taylor_resolve_snap_batch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params) {
    constexpr SizeType kParamsExpected = 4;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 4 for snap resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 4 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(pindex_flat_batch.size(), n_leaves,
                                     "param_idx_flat_batch size mismatch");
    error_check::check_greater_equal(relative_phase_batch.size(), n_leaves,
                                     "relative_phase_batch size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[2];
    const auto& freq_arr_grid  = param_arr[3];
    const auto n_freq          = param_arr[3].size();

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

    SizeType hint_a = 0, hint_f = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto s_t_cur = leaves_batch[leaf_offset + (0 * kParamStride)];
        const auto j_t_cur = leaves_batch[leaf_offset + (1 * kParamStride)];
        const auto a_t_cur = leaves_batch[leaf_offset + (2 * kParamStride)];
        const auto v_t_cur = leaves_batch[leaf_offset + (3 * kParamStride)];
        const auto f0      = leaves_batch[leaf_offset + (5 * kParamStride)];
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
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f0, nbins, delay_rel);

        // Find nearest grid indices
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
    }
}

void poly_taylor_transform_accel_batch(std::span<double> leaves_batch,
                                       std::pair<double, double> coord_next,
                                       std::pair<double, double> coord_cur,
                                       SizeType n_leaves,
                                       SizeType n_params,
                                       bool use_conservative_tile) {
    constexpr SizeType kParamsExpected = 2;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 2 for accel transform");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto delta_t         = t0_next - t0_cur;
    const auto half_delta_t_sq = 0.5 * (delta_t * delta_t);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto leaf_offset = i * kLeavesStride;

        const auto a_val_i = leaves_batch[leaf_offset + 0];
        const auto a_err_i = leaves_batch[leaf_offset + 1];
        const auto v_val_i = leaves_batch[leaf_offset + 2];
        const auto v_err_i = leaves_batch[leaf_offset + 3];
        const auto d_val_i = leaves_batch[leaf_offset + 4];
        const auto d_err_i = leaves_batch[leaf_offset + 5];

        const auto a_val_j = a_val_i;
        const auto v_val_j = v_val_i + (a_val_i * delta_t);
        const auto d_val_j =
            d_val_i + (v_val_i * delta_t) + (a_val_i * half_delta_t_sq);

        double a_err_j, v_err_j;
        if (use_conservative_tile) {
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
        leaves_batch[leaf_offset + 0] = a_val_j;
        leaves_batch[leaf_offset + 1] = a_err_j;
        leaves_batch[leaf_offset + 2] = v_val_j;
        leaves_batch[leaf_offset + 3] = v_err_j;
        leaves_batch[leaf_offset + 4] = d_val_j;
        leaves_batch[leaf_offset + 5] = d_err_i;
    }
}

void poly_taylor_transform_jerk_batch(std::span<double> leaves_batch,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool use_conservative_tile) {
    constexpr SizeType kParamsExpected = 3;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 3 for jerk transform");
    error_check::check_greater_equal(leaves_batch.size(),
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

        const auto j_val_i = leaves_batch[leaf_offset + 0];
        const auto j_err_i = leaves_batch[leaf_offset + 1];
        const auto a_val_i = leaves_batch[leaf_offset + 2];
        const auto a_err_i = leaves_batch[leaf_offset + 3];
        const auto v_val_i = leaves_batch[leaf_offset + 4];
        const auto v_err_i = leaves_batch[leaf_offset + 5];
        const auto d_val_i = leaves_batch[leaf_offset + 6];
        const auto d_err_i = leaves_batch[leaf_offset + 7];

        const auto j_val_j = j_val_i;
        const auto a_val_j = a_val_i + (j_val_i * delta_t);
        const auto v_val_j =
            v_val_i + (a_val_i * delta_t) + (j_val_i * half_delta_t_sq);
        const auto d_val_j = d_val_i + (v_val_i * delta_t) +
                             (a_val_i * half_delta_t_sq) +
                             (j_val_i * sixth_delta_t_cubed);

        double j_err_j, a_err_j, v_err_j;
        if (use_conservative_tile) {
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
        leaves_batch[leaf_offset + 0] = j_val_j;
        leaves_batch[leaf_offset + 1] = j_err_j;
        leaves_batch[leaf_offset + 2] = a_val_j;
        leaves_batch[leaf_offset + 3] = a_err_j;
        leaves_batch[leaf_offset + 4] = v_val_j;
        leaves_batch[leaf_offset + 5] = v_err_j;
        leaves_batch[leaf_offset + 6] = d_val_j;
        leaves_batch[leaf_offset + 7] = d_err_i;
    }
}

void poly_taylor_transform_snap_batch(std::span<double> leaves_batch,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool use_conservative_tile) {
    constexpr SizeType kParamsExpected = 4;
    constexpr SizeType kParamStride    = 2;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 4 for snap transform");
    error_check::check_greater_equal(leaves_batch.size(),
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

        const auto s_val_i = leaves_batch[leaf_offset + 0];
        const auto s_err_i = leaves_batch[leaf_offset + 1];
        const auto j_val_i = leaves_batch[leaf_offset + 2];
        const auto j_err_i = leaves_batch[leaf_offset + 3];
        const auto a_val_i = leaves_batch[leaf_offset + 4];
        const auto a_err_i = leaves_batch[leaf_offset + 5];
        const auto v_val_i = leaves_batch[leaf_offset + 6];
        const auto v_err_i = leaves_batch[leaf_offset + 7];
        const auto d_val_i = leaves_batch[leaf_offset + 8];
        const auto d_err_i = leaves_batch[leaf_offset + 9];

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
        if (use_conservative_tile) {
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
        leaves_batch[leaf_offset + 0] = s_val_j;
        leaves_batch[leaf_offset + 1] = s_err_j;
        leaves_batch[leaf_offset + 2] = j_val_j;
        leaves_batch[leaf_offset + 3] = j_err_j;
        leaves_batch[leaf_offset + 4] = a_val_j;
        leaves_batch[leaf_offset + 5] = a_err_j;
        leaves_batch[leaf_offset + 6] = v_val_j;
        leaves_batch[leaf_offset + 7] = v_err_j;
        leaves_batch[leaf_offset + 8] = d_val_j;
        leaves_batch[leaf_offset + 9] = d_err_i;
    }
}

void report_leaves_taylor_batch(std::span<double> leaves_tree,
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
                std::pow(param_err / s_factor, 2) +
                (std::pow(param_val * utils::kInvCval / (s_factor * s_factor), 2) *
                 std::pow(dv_final, 2)));
        }
        leaves_tree[leaf_offset + ((n_params - 1) * kParamStride) + 0] =
            f0_batch * s_factor;
        leaves_tree[leaf_offset + ((n_params - 1) * kParamStride) + 1] =
            f0_batch * dv_final * utils::kInvCval;
    }
}

std::vector<double>
poly_taylor_branch(std::span<const double> leaf,
                   std::pair<double, double> coord_cur,
                   SizeType n_params,
                   SizeType nbins,
                   double eta,
                   const std::vector<ParamLimitType>& param_limits) {
    const auto branch_max    = 100;
    const auto leaves_stride = (n_params + 2) * 2;
    std::vector<double> branch_leaves(branch_max * leaves_stride);
    const auto batch_origins =
        poly_taylor_branch_batch(leaf, coord_cur, branch_leaves, 1, n_params,
                                 nbins, eta, param_limits, branch_max);
    return {branch_leaves.begin(),
            branch_leaves.begin() +
                static_cast<IndexType>(batch_origins.size() * leaves_stride)};
}

std::vector<double>
generate_bp_poly_taylor_approx(std::span<const std::vector<double>> param_arr,
                               std::span<const double> dparams_lim,
                               const std::vector<ParamLimitType>& param_limits,
                               double tseg_ffa,
                               SizeType nsegments,
                               SizeType nbins,
                               double eta,
                               SizeType ref_seg,
                               IndexType isuggest,
                               bool use_conservative_tile) {
    error_check::check_equal(param_arr.size(), dparams_lim.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    std::vector<double> branching_pattern(nsegments - 1);
    const auto poly_order = dparams_lim.size();
    psr_utils::MiddleOutScheme snail_scheme(nsegments, ref_seg, tseg_ffa);
    const auto leaves_stride = (poly_order + 2) * 2;
    const auto coord_init    = snail_scheme.get_coord(0);
    SizeType n_leaves        = 1;
    for (const auto& arr : param_arr) {
        n_leaves *= arr.size();
    }
    std::vector<double> seed_leaves(n_leaves * leaves_stride);
    const auto n_leaves_seed = poly_taylor_seed(
        param_arr, dparams_lim, poly_order, coord_init, seed_leaves);
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
        auto leaves_arr = poly_taylor_branch(leaf_data, coord_cur, poly_order,
                                             nbins, eta, param_limits);
        const auto n_leaves_branch = leaves_arr.size() / leaves_stride;
        branching_pattern[prune_level - 1] =
            static_cast<double>(n_leaves_branch);

        if (poly_order == 2) {
            poly_taylor_transform_accel_batch(leaves_arr, coord_next, coord_cur,
                                              n_leaves_branch, poly_order,
                                              use_conservative_tile);
        } else if (poly_order == 3) {
            poly_taylor_transform_jerk_batch(leaves_arr, coord_next, coord_cur,
                                             n_leaves_branch, poly_order,
                                             use_conservative_tile);
        } else if (poly_order == 4) {
            poly_taylor_transform_snap_batch(leaves_arr, coord_next, coord_cur,
                                             n_leaves_branch, poly_order,
                                             use_conservative_tile);
        } else {
            throw std::invalid_argument("poly_order must be 2, 3, or 4 for "
                                        "branching pattern generation");
        }
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
                        const std::vector<ParamLimitType>& param_limits,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        bool use_conservative_tile) {
    constexpr double kEps = 1e-12;
    error_check::check_equal(param_arr.size(), dparams.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    const auto poly_order = dparams.size();
    const auto& f0_batch  = param_arr.back(); // Last array is frequency
    const auto n_freqs    = f0_batch.size();  // Number of frequency bins

    // Snail Scheme
    psr_utils::MiddleOutScheme snail_scheme(nsegments, ref_seg, tseg_ffa);
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

    std::vector<double> dparam_new_batch(n_freqs * poly_order, 0.0);
    std::vector<double> shift_bins_batch(n_freqs * poly_order, 0.0);
    std::vector<double> dparam_cur_next(n_freqs * poly_order, 0.0);
    std::vector<double> n_branches(n_freqs, 1);
    const auto n_params = poly_order + 1;
    std::vector<double> dparam_d_vec(n_freqs * n_params, 0.0);

    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = snail_scheme.get_coord(prune_level);
        const auto coord_cur  = snail_scheme.get_current_coord(prune_level);
        const auto [_, t_obs_minus_t_ref] = coord_cur;

        // Calculate optimal parameter steps and shift bins
        psr_utils::poly_taylor_step_d_vec(poly_order, t_obs_minus_t_ref, nbins,
                                          eta, f0_batch, dparam_new_batch, 0);
        psr_utils::poly_taylor_shift_d_vec(
            dparam_cur_batch, dparam_new_batch, t_obs_minus_t_ref, nbins,
            f0_batch, 0, shift_bins_batch, n_freqs, poly_order);

        std::ranges::fill(n_branches, 1.0);
        // Determine branching needs
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < poly_order; ++j) {
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
