#include "loki/core/taylor.hpp"

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
#include "loki/detection/score.hpp"
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
    error_check::check_equal(n_freq, pindex_prev_flat_batch.size(),
                             "pindex_prev_flat size mismatch");
    error_check::check_equal(n_freq, relative_phase_batch.size(),
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
        nparams, param_arr_cur.size(),
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
    error_check::check_equal(ncoords, pindex_prev_flat_batch.size(),
                             "pindex_prev_flat size mismatch");
    error_check::check_equal(ncoords, relative_phase_batch.size(),
                             "relative_phase_batch size mismatch");

    const auto tsegment = std::pow(2.0, ffa_level - 1) * tseg_brute;
    const auto delta_t  = (static_cast<double>(latter) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val       = 1.0 / utils::kCval;
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
            const auto f_new     = f_cur * (1.0 - v_new * inv_c_val);
            const auto delay_rel = d_new * inv_c_val;

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
        nparams, param_arr_cur.size(),
        "param_arr_cur and param_arr_prev should have the same size");
    error_check::check_equal(nparams, 3U,
                             "nparams should be 3 for jerk resolve");

    const auto n_jerk       = param_arr_cur[0].size();
    const auto n_accel      = param_arr_cur[1].size();
    const auto n_freq       = param_arr_cur[2].size();
    const auto ncoords      = n_jerk * n_accel * n_freq;
    const auto n_accel_prev = param_arr_prev[1].size();
    const auto n_freq_prev  = param_arr_prev[2].size();

    error_check::check_equal(ncoords, pindex_prev_flat_batch.size(),
                             "pindex_prev_flat size mismatch");
    error_check::check_equal(ncoords, relative_phase_batch.size(),
                             "relative_phase_batch size mismatch");

    const auto tsegment = std::pow(2.0, ffa_level - 1) * tseg_brute;
    const auto delta_t  = (static_cast<double>(latter) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val           = 1.0 / utils::kCval;
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
                const auto f_new     = f_cur * (1.0 - v_new * inv_c_val);
                const auto delay_rel = d_new * inv_c_val;
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
        nparams, param_arr_cur.size(),
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
        nparams, param_arr_cur.size(),
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

std::vector<double>
poly_taylor_leaves(std::span<const std::vector<double>> param_arr,
                   std::span<const double> dparams,
                   SizeType poly_order,
                   std::pair<double, double> /*coord_init*/) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (poly_order + 2) * kParamStride;
    const auto n_params             = param_arr.size();
    error_check::check_equal(n_params, poly_order,
                             "n_params should be equal to poly_order");

    SizeType n_leaves = 1;
    for (const auto& arr : param_arr) {
        n_leaves *= arr.size();
    }
    // Create parameter sets tensor: (n_param_sets, poly_order + 2, 2)
    std::vector<double> param_sets(n_leaves * leaves_stride);

    SizeType leaf_idx = 0;
    for (const auto& p_set_view : utils::cartesian_product_view(param_arr)) {
        const auto leaf_offset = leaf_idx * leaves_stride;
        const auto f0          = p_set_view[n_params - 1];
        const auto df          = dparams[n_params - 1];
        // Copy till d2 (acceleration)
        for (SizeType j = 0; j < n_params - 1; ++j) {
            param_sets[leaf_offset + (j * kParamStride) + 0] = p_set_view[j];
            param_sets[leaf_offset + (j * kParamStride) + 1] = dparams[j];
        }
        // Update frequency to velocity
        // f = f0(1 - v / C) => dv = -(C/f0) * df
        param_sets[leaf_offset + ((n_params - 1) * kParamStride) + 0] = 0;
        param_sets[leaf_offset + ((n_params - 1) * kParamStride) + 1] =
            df * (utils::kCval / f0);
        // intialize d0 (measure from t=t_init) and store f0
        param_sets[leaf_offset + ((n_params + 0) * kParamStride) + 0] = 0;
        param_sets[leaf_offset + ((n_params + 0) * kParamStride) + 1] = 0;
        param_sets[leaf_offset + ((n_params + 1) * kParamStride) + 0] = f0;
        param_sets[leaf_offset + ((n_params + 1) * kParamStride) + 1] = 0;
        ++leaf_idx;
    }
    return param_sets;
}

template <typename FoldType>
void poly_taylor_suggest(
    std::span<const FoldType> fold_segment,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType poly_order,
    SizeType nbins,
    const detection::ScoringFunction<FoldType>& scoring_func,
    detection::BoxcarWidthsCache& boxcar_widths_cache,
    utils::SuggestionTree<FoldType>& sugg_tree) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (poly_order + 2) * kParamStride;
    const auto n_params             = param_arr.size();
    error_check::check_equal(n_params, poly_order,
                             "n_params should be equal to poly_order");
    error_check::check_equal(
        leaves_stride, sugg_tree.get_leaves_stride(),
        "leaves_stride should be equal to sugg_tree.get_leaves_stride()");

    SizeType n_leaves = 1;
    for (const auto& arr : param_arr) {
        n_leaves *= arr.size();
    }
    const auto param_sets =
        poly_taylor_leaves(param_arr, dparams, poly_order, coord_init);

    // Fold segment is (n_leaves, 2, nbins)
    error_check::check_equal(fold_segment.size(), n_leaves * 2 * nbins,
                             "fold_segment size mismatch");

    // Calculate scores
    std::vector<float> scores(n_leaves);
    scoring_func(fold_segment, scores, n_leaves, boxcar_widths_cache);
    // Initialize the SuggestionStruct with the generated data
    sugg_tree.add_initial(param_sets, fold_segment, scores, n_leaves);
}

std::vector<SizeType>
poly_taylor_branch_batch(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch_batch,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType fold_bins,
                         double tol_bins,
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

    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType param_offset = i * leaves_stride;
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

    std::vector<double> pad_branched_dparams(n_batch * n_params);
    std::vector<SizeType> branched_counts(n_batch * n_params);
    // Optimized branching loop - same logic as original but vectorized access
    constexpr double kEps = 1e-6;
    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType leaf_offset = i * leaves_stride;
        const SizeType flat_base   = i * n_params;

        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType flat_idx     = flat_base + j;
            const SizeType param_offset = leaf_offset + (j * kParamStride);
            const double param_cur_val  = leaves_batch[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];

            if (shift_bins_batch[flat_idx] >= (tol_bins - kEps)) {
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

std::vector<SizeType> poly_taylor_branch_circular_batch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_cur,
    std::span<double> leaves_branch_batch,
    SizeType n_batch,
    SizeType n_params,
    SizeType fold_bins,
    double tol_bins,
    const std::vector<ParamLimitType>& param_limits,
    SizeType branch_max,
    double snap_threshold) {
    constexpr SizeType kParamsExpected = 5U;
    constexpr SizeType kParamStride    = 2U;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;
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
    constexpr double kEps = 1e-6;
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

    // Create intermediate leaves
    std::vector<double> leaves_intermediate(total_intermediate_leaves *
                                            kLeavesStride);
    for (SizeType i = 0; i < total_intermediate_leaves; ++i) {
        const SizeType origin        = batch_origins[i];
        const SizeType branch_offset = i * kLeavesStride;
        const SizeType leaf_offset   = origin * kLeavesStride;
        const SizeType d0_branch_offset =
            branch_offset + (n_params * kParamStride);
        const SizeType f0_branch_offset = d0_branch_offset + kParamStride;
        const SizeType d0_leaf_offset = leaf_offset + (n_params * kParamStride);
        const SizeType f0_leaf_offset = d0_leaf_offset + kParamStride;

        // Fill parameters and dparams
        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType leaf_offset_j = branch_offset + (j * kParamStride);
            leaves_intermediate[leaf_offset_j + 0] =
                leaves_branch_taylor_batch[(i * n_params) + j];
            leaves_intermediate[leaf_offset_j + 1] =
                pad_branched_dparams[(origin * n_params) + j];
        }
        // Fill d0 and f0
        leaves_intermediate[d0_branch_offset + 0] =
            leaves_batch[d0_leaf_offset + 0];
        leaves_intermediate[d0_branch_offset + 1] =
            leaves_batch[d0_leaf_offset + 1];
        leaves_intermediate[f0_branch_offset + 0] =
            leaves_batch[f0_leaf_offset + 0];
        leaves_intermediate[f0_branch_offset + 1] =
            leaves_batch[f0_leaf_offset + 1];
    }

    // --- Classify Intermediate Leaves into Circular and Non-Circular ---
    const auto [idx_circular_snap, idx_circular_crackle, idx_taylor] =
        get_circular_mask(leaves_intermediate, total_intermediate_leaves,
                          n_params, snap_threshold);

    if (idx_circular_crackle.empty()) {
        // No crackle branching needed, return intermediate leaves
        std::ranges::copy(leaves_intermediate, leaves_branch_batch.begin());
        return batch_origins;
    }

    // Branch crackle for idx_circular_crackle cases
    const SizeType n_crackle_branch = idx_circular_crackle.size();
    std::vector<double> crackle_branched_params(n_crackle_branch * branch_max);
    std::vector<double> crackle_branched_dparams(n_crackle_branch);
    std::vector<SizeType> crackle_branched_counts(n_crackle_branch);

    for (SizeType i = 0; i < n_crackle_branch; ++i) {
        const SizeType leaf_idx       = idx_circular_crackle[i];
        const SizeType orig_batch_idx = batch_origins[leaf_idx];
        const SizeType leaf_offset    = leaf_idx * kLeavesStride;

        const double crackle_cur    = leaves_intermediate[leaf_offset + 0];
        const double crackle_dparam = leaves_intermediate[leaf_offset + 1];

        const auto [pmin, pmax] = param_limits[0]; // crackle limits
        std::span<double> slice_span(
            crackle_branched_params.data() + (i * branch_max), branch_max);
        auto [dparam_act, count] = psr_utils::branch_param_padded(
            slice_span, crackle_cur, crackle_dparam,
            dparam_new_batch[(orig_batch_idx * n_params) + 0], pmin, pmax);

        crackle_branched_dparams[i] = dparam_act;
        crackle_branched_counts[i]  = count;
    }

    // Check if crackle actually needs branching
    for (SizeType i = 0; i < n_crackle_branch; ++i) {
        const SizeType leaf_idx       = idx_circular_crackle[i];
        const SizeType orig_batch_idx = batch_origins[leaf_idx];
        const SizeType leaf_offset    = leaf_idx * kLeavesStride;

        const bool needs_branching =
            shift_bins_batch[(orig_batch_idx * n_params) + 0] >=
            (tol_bins - kEps);
        if (!needs_branching) {
            crackle_branched_params[i * branch_max] =
                leaves_intermediate[leaf_offset + 0];
            crackle_branched_dparams[i] = leaves_intermediate[leaf_offset + 1];
            crackle_branched_counts[i]  = 1;
        }
    }

    // Create final leaves with branched crackle
    SizeType total_crackle_branches = 0;
    for (SizeType count : crackle_branched_counts) {
        total_crackle_branches += count;
    }

    // Combine keep indices
    std::vector<SizeType> keep_indices;
    keep_indices.reserve(idx_circular_snap.size() + idx_taylor.size());
    keep_indices.insert(keep_indices.end(), idx_circular_snap.begin(),
                        idx_circular_snap.end());
    keep_indices.insert(keep_indices.end(), idx_taylor.begin(),
                        idx_taylor.end());

    const SizeType n_keep       = keep_indices.size();
    const SizeType total_leaves = n_keep + total_crackle_branches;
    std::vector<SizeType> origins_final(total_leaves);

    // Copy non-crackle-branching leaves
    for (SizeType i = 0; i < n_keep; ++i) {
        const SizeType keep_idx   = keep_indices[i];
        const SizeType src_offset = keep_idx * kLeavesStride;
        const SizeType dst_offset = i * kLeavesStride;

        std::copy_n(leaves_intermediate.data() + src_offset, kLeavesStride,
                    leaves_branch_batch.data() + dst_offset);
        origins_final[i] = batch_origins[keep_idx];
    }
    // Add crackle-branched leaves
    SizeType current_idx = n_keep;
    for (SizeType i = 0; i < n_crackle_branch; ++i) {
        const SizeType count_i          = crackle_branched_counts[i];
        const SizeType orig_leaf_idx    = idx_circular_crackle[i];
        const SizeType orig_batch_idx   = batch_origins[orig_leaf_idx];
        const SizeType orig_leaf_offset = orig_leaf_idx * kLeavesStride;

        for (SizeType b = 0; b < count_i; ++b) {
            const SizeType dst_offset = (current_idx + b) * kLeavesStride;

            // Copy original leaf
            std::copy_n(leaves_intermediate.data() + orig_leaf_offset,
                        kLeavesStride, leaves_branch_batch.data() + dst_offset);

            // Update crackle parameter
            leaves_branch_batch[dst_offset + 0] =
                crackle_branched_params[(i * branch_max) + b];
            leaves_branch_batch[dst_offset + 1] = crackle_branched_dparams[i];

            origins_final[current_idx + b] = orig_batch_idx;
        }
        current_idx += count_i;
    }
    return origins_final;
}

SizeType poly_taylor_validate_circular_batch(std::span<double> leaves_batch,
                                             std::span<SizeType> leaves_origins,
                                             SizeType n_leaves,
                                             SizeType n_params,
                                             double p_orb_min,
                                             double snap_threshold) {
    constexpr SizeType kParamsExpected = 5U;
    constexpr SizeType kParamStride    = 2U;
    constexpr SizeType kLeavesStride   = (kParamsExpected + 2) * kParamStride;
    constexpr double kEps              = 1e-12;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 5 for circular orbit resolve");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double omega_orb_max_sq =
        std::pow(2.0 * std::numbers::pi / p_orb_min, 2);
    std::vector<bool> circular_mask(n_leaves, false);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType leaf_offset = i * kLeavesStride;

        const auto snap  = leaves_batch[leaf_offset + (1 * kParamStride)];
        const auto dsnap = leaves_batch[leaf_offset + (1 * kParamStride) + 1];
        const auto accel = leaves_batch[leaf_offset + (3 * kParamStride)];
        const auto omega_sq = -snap / (accel + kEps);
        // Physically usable via s/a
        const bool is_usable = (std::abs(accel) > kEps) &&
                               (std::abs(snap) > kEps) && (omega_sq > 0.0);
        // Numerically degenerate but not obviously unphysical
        const bool is_zero =
            (std::abs(snap) <= kEps) || (std::abs(accel) <= kEps);
        // Within maximum orbital frequency limit
        const bool is_within_omega_limit = omega_sq <= omega_orb_max_sq;
        // Delays circular validation until snap is well-measured.
        const bool is_significant =
            std::abs(snap / (dsnap + kEps)) > snap_threshold;
        circular_mask[i] = !is_significant ||
                           (is_zero || (is_usable && is_within_omega_limit));
    }

    SizeType write_idx = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        if (circular_mask[i]) {
            if (write_idx != i) {
                // Copy batch data
                const SizeType src_offset = i * kLeavesStride;
                const SizeType dst_offset = write_idx * kLeavesStride;
                std::copy(
                    leaves_batch.begin() + static_cast<IndexType>(src_offset),
                    leaves_batch.begin() +
                        static_cast<IndexType>(src_offset + kLeavesStride),
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
    SizeType n_params,
    double /*snap_threshold*/) {
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
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "param_idx_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[0];
    const auto& freq_arr_grid  = param_arr[1];
    const auto n_freq          = param_arr[1].size();

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val    = 1.0 / utils::kCval;
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
        const auto f_new     = f0 * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

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
    SizeType n_params,
    double /*snap_threshold*/) {
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
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "param_idx_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[1];
    const auto& freq_arr_grid  = param_arr[2];
    const auto n_freq          = param_arr[2].size();

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val            = 1.0 / utils::kCval;
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
        const auto f_new     = f0 * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

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
    SizeType n_params,
    double /*snap_threshold*/) {
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
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "param_idx_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[2];
    const auto& freq_arr_grid  = param_arr[3];
    const auto n_freq          = param_arr[3].size();

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val            = 1.0 / utils::kCval;
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
        const auto f_new     = f0 * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

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

std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circular_mask(std::span<const double> leaves_batch,
                  SizeType n_leaves,
                  SizeType n_params,
                  double snap_threshold) {
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

        const auto crackle = leaves_batch[leaf_offset + 0];
        const auto snap    = leaves_batch[leaf_offset + 2];
        const auto dsnap   = leaves_batch[leaf_offset + 3];
        const auto jerk    = leaves_batch[leaf_offset + 4];
        const auto accel   = leaves_batch[leaf_offset + 6];

        // Delays circular classification until snap is well-measured
        const bool is_significant =
            std::abs(snap / (dsnap + kEps)) > snap_threshold;

        // Numerical Stability - acceleration and snap must be significantly
        // non-zero
        const bool is_stable =
            (std::abs(accel) > kEps) && (std::abs(snap) > kEps);

        // Physicality - for Omega^2 = -s/a to be positive, s and a must have
        // opposite signs
        const auto omega_sq    = -snap / (accel + kEps);
        const bool is_physical = omega_sq > 0.0;

        // Classification logic
        const bool stable_and_significant = is_stable && is_significant;

        // idx_circular_snap: stable, significant, and physical
        const bool mask_circular_snap = stable_and_significant && is_physical;
        if (mask_circular_snap) {
            idx_circular_snap.push_back(i);
        } else {
            // For unstable but significant cases, check crackle/jerk
            // approximation
            const bool unstable_and_significant =
                (!is_stable) && is_significant;
            const bool crackle_jerk_nonzero =
                (std::abs(crackle) > kEps) && (std::abs(jerk) > kEps);

            if (unstable_and_significant && crackle_jerk_nonzero) {
                const auto omega_sq_crackle_jerk = -crackle / (jerk + kEps);
                const bool crackle_jerk_physical = omega_sq_crackle_jerk > 0.0;

                if (crackle_jerk_physical) {
                    idx_circular_crackle.push_back(i);
                } else {
                    idx_taylor.push_back(i);
                }
            } else {
                // idx_taylor: everything else (no hope for circularity)
                idx_taylor.push_back(i);
            }
        }
    }
    return {std::move(idx_circular_snap), std::move(idx_circular_crackle),
            std::move(idx_taylor)};
}

void poly_taylor_resolve_circular_batch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params,
    double snap_threshold) {
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
        get_circular_mask(leaves_batch, n_leaves, n_params, snap_threshold);

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

void poly_taylor_transform_accel_batch(std::span<double> leaves_batch,
                                       std::pair<double, double> coord_next,
                                       std::pair<double, double> coord_cur,
                                       SizeType n_leaves,
                                       SizeType n_params,
                                       bool conservative_errors,
                                       double /*snap_threshold*/) {
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

        double a_err_j, v_err_j, d_err_j;
        if (conservative_errors) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            a_err_j = a_err_i;
            v_err_j = std::sqrt((v_err_i * v_err_i) +
                                (a_err_i * a_err_i * delta_t * delta_t));
            d_err_j = std::sqrt(
                (d_err_i * d_err_i) + (v_err_i * v_err_i * delta_t * delta_t) +
                (a_err_i * a_err_i * half_delta_t_sq * half_delta_t_sq));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            a_err_j = a_err_i;
            v_err_j = v_err_i;
            d_err_j = d_err_i;
        }

        // Write back transformed values
        leaves_batch[leaf_offset + 0] = a_val_j;
        leaves_batch[leaf_offset + 1] = a_err_j;
        leaves_batch[leaf_offset + 2] = v_val_j;
        leaves_batch[leaf_offset + 3] = v_err_j;
        leaves_batch[leaf_offset + 4] = d_val_j;
        leaves_batch[leaf_offset + 5] = d_err_j;
    }
}

void poly_taylor_transform_jerk_batch(std::span<double> leaves_batch,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool conservative_errors,
                                      double /*snap_threshold*/) {
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

        double j_err_j, a_err_j, v_err_j, d_err_j;
        if (conservative_errors) {
            // Conservative: sqrt(errors^2 @ T^2.T)
            j_err_j = j_err_i;
            a_err_j = std::sqrt((a_err_i * a_err_i) +
                                (j_err_i * j_err_i * delta_t_sq));
            v_err_j = std::sqrt(
                (v_err_i * v_err_i) + (a_err_i * a_err_i * delta_t_sq) +
                (j_err_i * j_err_i * half_delta_t_sq * half_delta_t_sq));
            d_err_j = std::sqrt(
                (d_err_i * d_err_i) + (v_err_i * v_err_i * delta_t_sq) +
                (a_err_i * a_err_i * half_delta_t_sq * half_delta_t_sq) +
                (j_err_i * j_err_i * sixth_delta_t_cubed *
                 sixth_delta_t_cubed));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
            d_err_j = d_err_i;
        }

        // Write back transformed values
        leaves_batch[leaf_offset + 0] = j_val_j;
        leaves_batch[leaf_offset + 1] = j_err_j;
        leaves_batch[leaf_offset + 2] = a_val_j;
        leaves_batch[leaf_offset + 3] = a_err_j;
        leaves_batch[leaf_offset + 4] = v_val_j;
        leaves_batch[leaf_offset + 5] = v_err_j;
        leaves_batch[leaf_offset + 6] = d_val_j;
        leaves_batch[leaf_offset + 7] = d_err_j;
    }
}

void poly_taylor_transform_snap_batch(std::span<double> leaves_batch,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool conservative_errors,
                                      double /*snap_threshold*/) {
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

        double s_err_j, j_err_j, a_err_j, v_err_j, d_err_j;
        if (conservative_errors) {
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
            d_err_j = std::sqrt(
                (d_err_i * d_err_i) + (v_err_i * v_err_i * delta_t_sq) +
                (a_err_i * a_err_i * half_delta_t_sq * half_delta_t_sq) +
                (j_err_i * j_err_i * sixth_delta_t_cubed *
                 sixth_delta_t_cubed) +
                (s_err_i * s_err_i * twenty_fourth_delta_t_fourth *
                 twenty_fourth_delta_t_fourth));
        } else {
            // Non-conservative: errors * |diag(T)| = errors * 1
            s_err_j = s_err_i;
            j_err_j = j_err_i;
            a_err_j = a_err_i;
            v_err_j = v_err_i;
            d_err_j = d_err_i;
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
        leaves_batch[leaf_offset + 9] = d_err_j;
    }
}

void poly_taylor_transform_circular_batch(std::span<double> leaves_batch,
                                          std::pair<double, double> coord_next,
                                          std::pair<double, double> coord_cur,
                                          SizeType n_leaves,
                                          SizeType n_params,
                                          bool conservative_errors,
                                          double snap_threshold) {
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
        get_circular_mask(leaves_batch, n_leaves, n_params, snap_threshold);

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
        if (conservative_errors) {
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
poly_taylor_branch(std::span<const double> leaf,
                   std::pair<double, double> coord_cur,
                   SizeType n_params,
                   SizeType fold_bins,
                   double tol_bins,
                   const std::vector<ParamLimitType>& param_limits) {
    const auto branch_max    = 100;
    const auto leaves_stride = (n_params + 2) * 2;
    std::vector<double> branch_leaves(branch_max * leaves_stride);
    const auto batch_origins =
        poly_taylor_branch_batch(leaf, coord_cur, branch_leaves, 1, n_params,
                                 fold_bins, tol_bins, param_limits, branch_max);
    return {branch_leaves.begin(),
            branch_leaves.begin() +
                static_cast<IndexType>(batch_origins.size() * leaves_stride)};
}

std::vector<double>
generate_bp_taylor_approx(std::span<const std::vector<double>> param_arr,
                          std::span<const double> dparams_lim,
                          const std::vector<ParamLimitType>& param_limits,
                          double tseg_ffa,
                          SizeType nsegments,
                          SizeType fold_bins,
                          double tol_bins,
                          SizeType ref_seg,
                          bool use_conservative_errors) {
    error_check::check_equal(param_arr.size(), dparams_lim.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    std::vector<double> branching_pattern;
    const auto poly_order = dparams_lim.size();
    psr_utils::SnailScheme scheme(nsegments, ref_seg, tseg_ffa);
    branching_pattern.reserve(nsegments - 1);
    const auto leaves_stride = (poly_order + 2) * 2;
    const auto coord_init    = scheme.get_coord(0);
    const auto leaves =
        poly_taylor_leaves(param_arr, dparams_lim, poly_order, coord_init);
    const auto n_leaves = leaves.size() / leaves_stride;
    // Get last leaf
    auto leaf = std::span(leaves).subspan((leaves_stride * (n_leaves - 1)),
                                          leaves_stride);
    std::vector<double> leaf_data(leaves_stride);
    std::ranges::copy(leaf, leaf_data.begin());

    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_next = scheme.get_coord(prune_level);
        const auto coord_cur  = scheme.get_current_coord(prune_level);
        auto leaves_arr = poly_taylor_branch(leaf_data, coord_cur, poly_order,
                                             fold_bins, tol_bins, param_limits);
        const auto n_leaves_branch = leaves_arr.size() / leaves_stride;
        branching_pattern.push_back(static_cast<double>(n_leaves_branch));
        if (poly_order == 2) {
            poly_taylor_transform_accel_batch(leaves_arr, coord_next, coord_cur,
                                              n_leaves_branch, poly_order,
                                              use_conservative_errors, 0.0);
        } else if (poly_order == 3) {
            poly_taylor_transform_jerk_batch(leaves_arr, coord_next, coord_cur,
                                             n_leaves_branch, poly_order,
                                             use_conservative_errors, 0.0);
        } else {
            throw std::invalid_argument(
                "poly_order must be 2, or 3 for branching pattern generation");
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
generate_bp_taylor(std::span<const std::vector<double>> param_arr,
                   std::span<const double> dparams,
                   const std::vector<ParamLimitType>& param_limits,
                   double tseg_ffa,
                   SizeType nsegments,
                   SizeType fold_bins,
                   double tol_bins,
                   SizeType ref_seg,
                   bool use_conservative_errors) {
    error_check::check_equal(param_arr.size(), dparams.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    const auto poly_order = dparams.size();
    const auto& f0_batch  = param_arr.back(); // Last array is frequency
    const auto n_freqs    = f0_batch.size();  // Number of frequency bins

    // Snail Scheme
    psr_utils::SnailScheme scheme(nsegments, ref_seg, tseg_ffa);
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
        std::vector<SizeType> n_branches(n_freqs, 1);

        // Determine branching needs
        constexpr double kEps = 1e-12;
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < poly_order; ++j) {
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
                n_branches[i] *= static_cast<SizeType>(num_points);
                dparam_cur_next[idx] =
                    dparam_cur_batch[idx] / static_cast<double>(num_points);
            }
        }
        SizeType total_branches = 0;
        for (SizeType i = 0; i < n_freqs; ++i) {
            total_branches += n_branches[i];
        }
        branching_pattern[prune_level - 1] =
            static_cast<double>(total_branches) / static_cast<double>(n_freqs);

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
            dparam_d_vec, delta_t, use_conservative_errors, n_freqs, n_params);
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

std::vector<double>
generate_bp_taylor_circular(std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            const std::vector<ParamLimitType>& param_limits,
                            double tseg_ffa,
                            SizeType nsegments,
                            SizeType fold_bins,
                            double tol_bins,
                            SizeType ref_seg,
                            bool use_conservative_errors) {
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
    psr_utils::SnailScheme scheme(nsegments, ref_seg, tseg_ffa);
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
            dparam_d_vec, delta_t, use_conservative_errors, n_freqs, n_params);
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

template void poly_taylor_suggest<float>(
    std::span<const float> fold_segment,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType poly_order,
    SizeType nbins,
    const detection::ScoringFunction<float>& scoring_func,
    detection::BoxcarWidthsCache& boxcar_widths_cache,
    utils::SuggestionTree<float>& sugg_tree);

template void poly_taylor_suggest<ComplexType>(
    std::span<const ComplexType> fold_segment,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType poly_order,
    SizeType nbins,
    const detection::ScoringFunction<ComplexType>& scoring_func,
    detection::BoxcarWidthsCache& boxcar_widths_cache,
    utils::SuggestionTree<ComplexType>& sugg_tree);

} // namespace loki::core
