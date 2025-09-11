#include "loki/core/taylor.hpp"

#include <algorithm>
#include <span>
#include <vector>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
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
        std::tie(pset_prev, delay) = psr_utils::shift_params(pset_cur, delta_t);
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

std::vector<double>
poly_taylor_leaves(std::span<const std::vector<double>> param_arr,
                   std::span<const double> dparams,
                   std::pair<double, double> coord_init,
                   SizeType leaves_stride) {
    const auto nparams = param_arr.size();
    SizeType n_leaves  = 1;
    for (const auto& arr : param_arr) {
        n_leaves *= arr.size();
    }
    const auto [t0, scale] = coord_init;

    // Create parameter sets tensor: (n_param_sets, poly_order + 2, 2)
    std::vector<double> param_sets(n_leaves * leaves_stride);

    SizeType leaf_idx = 0;
    for (const auto& p_set_view : utils::cartesian_product_view(param_arr)) {
        const auto leaves_offset = leaf_idx * leaves_stride;
        // Fill first nparams dimensions with parameter values and dparams
        for (SizeType j = 0; j < nparams; ++j) {
            param_sets[leaves_offset + (j * 2) + 0] = p_set_view[j];
            param_sets[leaves_offset + (j * 2) + 1] = dparams[j];
        }
        param_sets[leaves_offset + (nparams * 2) + 0] = p_set_view[nparams - 1];
        param_sets[leaves_offset + ((nparams + 1) * 2) + 0] = t0;
        param_sets[leaves_offset + ((nparams + 1) * 2) + 1] = scale;
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
    const auto nparams = param_arr.size();
    error_check::check_equal(nparams, poly_order,
                             "nparams should be equal to poly_order");
    SizeType n_leaves = 1;
    for (const auto& arr : param_arr) {
        n_leaves *= arr.size();
    }
    error_check::check_equal(fold_segment.size(), n_leaves * 2 * nbins,
                             "fold_segment size mismatch");
    const auto leaves_stride = sugg_tree.get_leaves_stride();
    const auto param_sets =
        poly_taylor_leaves(param_arr, dparams, coord_init, leaves_stride);
    // Calculate scores
    std::vector<float> scores(n_leaves);
    scoring_func(fold_segment, scores, n_leaves, boxcar_widths_cache);
    // Initialize the SuggestionStruct with the generated data
    sugg_tree.add_initial(param_sets, fold_segment, scores, n_leaves);
}

std::vector<SizeType>
poly_taylor_branch_batch(std::span<const double> batch_psets,
                         std::pair<double, double> coord_cur,
                         std::span<double> batch_leaves,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType fold_bins,
                         double tol_bins,
                         SizeType poly_order,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max) {
    const SizeType leaves_stride_param = 2;
    const SizeType leaves_stride_batch = (n_params + 2) * leaves_stride_param;
    error_check::check_equal(n_params, poly_order,
                             "n_params should be equal to poly_order");
    error_check::check_equal(batch_psets.size(), n_batch * leaves_stride_batch,
                             "batch_psets size mismatch");
    error_check::check_greater_equal(batch_leaves.size(),
                                     n_batch * branch_max * leaves_stride_batch,
                                     "batch_leaves size mismatch");

    const auto [_, scale_cur] = coord_cur;
    const double tseg_cur     = 2.0 * scale_cur;
    const double t_ref        = tseg_cur / 2.0;

    // Use batch_leaves memory as workspace. Partition workspace into sections:
    const SizeType workspace_size      = batch_leaves.size();
    const SizeType single_batch_params = n_batch * n_params;

    // Get spans from workspace + other vector allocations
    std::span<double> dparam_cur_batch =
        batch_leaves.subspan(0, single_batch_params);
    std::span<double> dparam_opt_batch =
        batch_leaves.subspan(single_batch_params, single_batch_params);
    std::span<double> shift_bins_batch =
        batch_leaves.subspan(single_batch_params * 2, single_batch_params);
    std::span<double> f_max_batch =
        batch_leaves.subspan(single_batch_params * 3, n_batch);
    std::span<double> pad_branched_params = batch_leaves.subspan(
        (single_batch_params * 3) + n_batch, n_batch * n_params * branch_max);
    const auto workspace_acquired_size =
        (single_batch_params * 3) + n_batch + (n_batch * n_params * branch_max);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType param_offset = i * leaves_stride_batch;
        for (SizeType j = 0; j < n_params; ++j) {
            dparam_cur_batch[(i * n_params) + j] =
                batch_psets[(param_offset + (j * leaves_stride_param)) + 1];
        }
        f_max_batch[i] =
            batch_psets[param_offset + ((n_params - 1) * leaves_stride_param)];
    }

    psr_utils::poly_taylor_step_d_vec(n_params, tseg_cur, fold_bins, tol_bins,
                                      f_max_batch, dparam_opt_batch, t_ref);
    psr_utils::poly_taylor_shift_d_vec(dparam_cur_batch, dparam_opt_batch,
                                       tseg_cur, fold_bins, f_max_batch, t_ref,
                                       shift_bins_batch, n_batch, n_params);

    std::vector<double> pad_branched_dparams(n_batch * n_params);
    std::vector<SizeType> branched_counts(n_batch * n_params);
    // Optimized branching loop - same logic as original but vectorized access
    for (SizeType i = 0; i < n_batch; ++i) {
        const SizeType batch_offset = i * leaves_stride_batch;
        const SizeType flat_base    = i * n_params;

        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType flat_idx = flat_base + j;
            const SizeType param_offset =
                batch_offset + (j * leaves_stride_param);
            const double param_cur_val  = batch_psets[param_offset + 0];
            const double dparam_cur_val = dparam_cur_batch[flat_idx];

            if (shift_bins_batch[flat_idx] >= tol_bins) {
                const auto [p_min, p_max] = param_limits[j];
                const SizeType pad_offset =
                    (i * n_params * branch_max) + (j * branch_max);
                std::span<double> slice_span =
                    pad_branched_params.subspan(pad_offset, branch_max);
                auto [dparam_act, count] = psr_utils::branch_param_padded(
                    slice_span, param_cur_val, dparam_cur_val,
                    dparam_opt_batch[flat_idx], p_min, p_max);

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
    const auto [batch_leaves_taylor, batch_origins] =
        utils::cartesian_prod_padded(pad_branched_params, branched_counts,
                                     n_batch, n_params, branch_max);
    const SizeType total_leaves = batch_origins.size();

    // Fill dparams and other parameters using the same logic as original
    for (SizeType i = 0; i < total_leaves; ++i) {
        const SizeType origin        = batch_origins[i];
        const SizeType leaves_offset = i * leaves_stride_batch;
        const SizeType param_offset  = origin * leaves_stride_batch;
        const SizeType f0_leaves_offset =
            leaves_offset + (n_params * leaves_stride_param);
        const SizeType t0_leaves_offset =
            f0_leaves_offset + leaves_stride_param;
        const SizeType f0_param_offset =
            param_offset + (n_params * leaves_stride_param);
        const SizeType t0_param_offset = f0_param_offset + leaves_stride_param;

        // Fill parameters and dparams
        for (SizeType j = 0; j < n_params; ++j) {
            const SizeType leaves_offset_j =
                leaves_offset + (j * leaves_stride_param);
            batch_leaves[leaves_offset_j + 0] =
                batch_leaves_taylor[(i * n_params) + j];
            batch_leaves[leaves_offset_j + 1] =
                pad_branched_dparams[(origin * n_params) + j];
        }
        // Fill f0, t0, and scale directly from batch_psets
        batch_leaves[f0_leaves_offset + 0] = batch_psets[f0_param_offset + 0];
        batch_leaves[f0_leaves_offset + 1] = batch_psets[f0_param_offset + 1];
        batch_leaves[t0_leaves_offset + 0] = batch_psets[t0_param_offset + 0];
        batch_leaves[t0_leaves_offset + 1] = batch_psets[t0_param_offset + 1];
    }

    return batch_origins;
}

SizeType poly_taylor_validate_batch(std::span<double> leaves_batch,
                                    std::span<SizeType> leaves_origins,
                                    SizeType n_leaves,
                                    SizeType n_params) {
    constexpr SizeType kParamsExpected    = 4;
    constexpr SizeType kLeavesStrideParam = 2;
    constexpr SizeType kLeavesStrideBatch =
        (kParamsExpected + 2) * kLeavesStrideParam;
    constexpr double kEps = 1e-15;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 4 for circular orbit resolve");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStrideBatch,
                                     "batch_leaves size mismatch");
    SizeType write_idx = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType batch_offset = i * kLeavesStrideBatch;

        const auto snap  = leaves_batch[batch_offset + 0];
        const auto accel = leaves_batch[batch_offset + 4];
        bool zero_case   = (std::abs(snap) < kEps) || (std::abs(accel) < kEps);
        bool real_omega  = (-snap / accel) >= 0.0;
        if (zero_case || real_omega) {
            // Copy this leaf to the write position if needed
            if (write_idx != i) {
                // Copy batch
                for (SizeType j = 0; j < kLeavesStrideBatch; ++j) {
                    leaves_batch[(write_idx * kLeavesStrideBatch) + j] =
                        leaves_batch[batch_offset + j];
                }
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
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params) {
    constexpr SizeType kParamsExpected    = 2;
    constexpr SizeType kLeavesStrideParam = 2;
    constexpr SizeType kLeavesStrideBatch =
        (kParamsExpected + 2) * kLeavesStrideParam;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 2 for accel resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 2 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStrideBatch,
                                     "batch_leaves size mismatch");
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "param_idx_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");

    const double delta_t = coord_add.first - coord_init.first;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[0];
    const auto& freq_arr_grid  = param_arr[1];
    const auto n_freq          = param_arr[1].size();

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val       = 1.0 / utils::kCval;
    const auto half_delta_t_sq = 0.5 * delta_t * delta_t;

    SizeType hint_a = 0, hint_f = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType batch_offset = i * kLeavesStrideBatch;

        const auto a_cur       = leaves_batch[batch_offset + 0];
        const auto f_cur       = leaves_batch[batch_offset + 2];
        const auto a_new       = a_cur;
        const auto delta_v_new = a_cur * delta_t;
        const auto delta_d_new = a_cur * half_delta_t_sq;
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f_cur * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);

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
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params) {
    constexpr SizeType kParamsExpected    = 3;
    constexpr SizeType kLeavesStrideParam = 2;
    constexpr SizeType kLeavesStrideBatch =
        (kParamsExpected + 2) * kLeavesStrideParam;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 3 for jerk resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 3 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kLeavesStrideBatch,
                                     "batch_leaves size mismatch");
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "param_idx_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");

    const double delta_t = coord_add.first - coord_init.first;

    // Cache-friendly access patterns
    const auto& accel_arr_grid = param_arr[1];
    const auto& freq_arr_grid  = param_arr[2];
    const auto n_freq          = param_arr[2].size();

    // Pre-compute constants to avoid repeated calculations
    const auto inv_c_val           = 1.0 / utils::kCval;
    const auto delta_t_sq          = delta_t * delta_t;
    const auto delta_t_cubed       = delta_t_sq * delta_t;
    const auto half_delta_t_sq     = 0.5 * delta_t_sq;
    const auto sixth_delta_t_cubed = delta_t_cubed / 6.0;

    SizeType hint_a = 0, hint_f = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType batch_offset = i * kLeavesStrideBatch;

        const auto j_cur       = leaves_batch[batch_offset + 0];
        const auto a_cur       = leaves_batch[batch_offset + 2];
        const auto f_cur       = leaves_batch[batch_offset + 4];
        const auto a_new       = a_cur + (j_cur * delta_t);
        const auto delta_v_new = (a_cur * delta_t) + (j_cur * half_delta_t_sq);
        const auto delta_d_new =
            (a_cur * half_delta_t_sq) + (j_cur * sixth_delta_t_cubed);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f_cur * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);

        // Find nearest grid indices
        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
    }
}

void poly_taylor_resolve_circular_batch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> pindex_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params) {
    constexpr SizeType kParamsExpected = 4;
    constexpr SizeType kParamsStride   = 2;
    constexpr SizeType kBatchStride    = (kParamsExpected + 2) * kParamsStride;

    error_check::check_equal(n_params, kParamsExpected,
                             "nparams should be 4 for circular orbit resolve");
    error_check::check_equal(param_arr.size(), kParamsExpected,
                             "param_arr should have 4 parameters");
    error_check::check_greater_equal(leaves_batch.size(),
                                     n_leaves * kBatchStride,
                                     "batch_leaves size mismatch");
    error_check::check_equal(pindex_flat_batch.size(), n_leaves,
                             "pindex_flat_batch size mismatch");
    error_check::check_equal(relative_phase_batch.size(), n_leaves,
                             "relative_phase_batch size mismatch");
    const double delta_t   = coord_add.first - coord_init.first;
    const double inv_c_val = 1.0 / utils::kCval;

    // Cache-friendly access to parameter grids
    const auto& accel_arr_grid = param_arr[2];
    const auto& freq_arr_grid  = param_arr[3];
    const auto n_freq          = param_arr[3].size();

    // Categorize leaves into circular vs normal
    std::vector<SizeType> idx_circular, idx_normal;
    idx_circular.reserve(n_leaves / 2);
    idx_normal.reserve(n_leaves / 2);

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType batch_offset = i * kBatchStride;

        const auto snap  = leaves_batch[batch_offset + (0 * kParamsStride)];
        const auto dsnap = leaves_batch[batch_offset + (0 * kParamsStride) + 1];
        const auto accel = leaves_batch[batch_offset + (2 * kParamsStride)];

        // Circular orbit mask condition
        const bool is_circular = (accel != 0.0) && (snap != 0.0) &&
                                 ((-snap / accel) > 0.0) &&
                                 (std::abs(snap / dsnap) > 5.0);
        if (is_circular) {
            idx_circular.push_back(i);
        } else {
            idx_normal.push_back(i);
        }
    }

    SizeType hint_a = 0, hint_f = 0;
    // Process circular indices
    for (SizeType i : idx_circular) {
        const SizeType batch_offset = i * kBatchStride;

        const auto s_cur = leaves_batch[batch_offset + (0 * kParamsStride)];
        const auto j_cur = leaves_batch[batch_offset + (1 * kParamsStride)];
        const auto a_cur = leaves_batch[batch_offset + (2 * kParamsStride)];
        const auto f_cur = leaves_batch[batch_offset + (3 * kParamsStride)];

        // Circular orbit mask condition
        const auto minus_omega_sq  = s_cur / a_cur;
        const auto omega_orb       = std::sqrt(-minus_omega_sq);
        const auto omega_orb_sq    = -minus_omega_sq;
        const auto omega_orb_cubed = omega_orb * omega_orb_sq;
        // Evolve the phase to the new time t_j = t_i + delta_t
        const auto omega_dt = omega_orb * delta_t;
        const auto cos_odt  = std::cos(omega_dt);
        const auto sin_odt  = std::sin(omega_dt);
        const auto a_new = (a_cur * cos_odt) + ((j_cur / omega_orb) * sin_odt);
        const auto delta_v_new = ((a_cur / omega_orb) * sin_odt) -
                                 ((j_cur / omega_orb_sq) * (cos_odt - 1.0));
        const auto delta_d_new = -((a_cur / omega_orb_sq) * (cos_odt - 1.0)) -
                                 ((j_cur / omega_orb_cubed) * sin_odt) +
                                 ((j_cur * delta_t) / omega_orb_sq);
        const auto f_new     = f_cur * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);

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
    const auto delta_t_sq                  = delta_t * delta_t;
    const auto delta_t_cubed               = delta_t_sq * delta_t;
    const auto delta_t_fourth              = delta_t_cubed * delta_t;
    const auto half_delta_t_sq             = 0.5 * delta_t_sq;
    const auto sixth_delta_t_cubed         = delta_t_cubed / 6.0;
    const auto twentyfourth_delta_t_fourth = delta_t_fourth / 24.0;
    // Process normal indices
    for (SizeType i : idx_normal) {
        const SizeType batch_offset = i * kBatchStride;

        const auto s_cur = leaves_batch[batch_offset + 0];
        const auto j_cur = leaves_batch[batch_offset + 2];
        const auto a_cur = leaves_batch[batch_offset + 4];
        const auto f_cur = leaves_batch[batch_offset + 6];
        const auto a_new =
            a_cur + (j_cur * delta_t) + (s_cur * half_delta_t_sq);
        const auto delta_v_new = (a_cur * delta_t) + (j_cur * half_delta_t_sq) +
                                 (s_cur * sixth_delta_t_cubed);
        const auto delta_d_new = (a_cur * half_delta_t_sq) +
                                 (j_cur * sixth_delta_t_cubed) +
                                 (s_cur * twentyfourth_delta_t_fourth);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f_cur * (1.0 - delta_v_new * inv_c_val);
        const auto delay_rel = delta_d_new * inv_c_val;

        // Calculate relative phase
        relative_phase_batch[i] =
            psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);

        const auto idx_a =
            utils::find_nearest_sorted_idx_scan(accel_arr_grid, a_new, hint_a);
        const auto idx_f =
            utils::find_nearest_sorted_idx_scan(freq_arr_grid, f_new, hint_f);
        pindex_flat_batch[i] = (idx_a * n_freq) + idx_f;
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
    const auto poly_order    = n_params;
    const auto batch_origins = poly_taylor_branch_batch(
        leaf, coord_cur, branch_leaves, 1, n_params, fold_bins, tol_bins,
        poly_order, param_limits, branch_max);
    return {branch_leaves.begin(),
            branch_leaves.begin() +
                static_cast<IndexType>(batch_origins.size() * leaves_stride)};
}

std::vector<SizeType>
poly_taylor_branching_pattern(std::span<const std::vector<double>> param_arr,
                              std::span<const double> dparams_lim,
                              const std::vector<ParamLimitType>& param_limits,
                              SizeType nsegments,
                              double tsegment,
                              SizeType fold_bins,
                              double tol_bins) {
    std::vector<SizeType> branching_pattern;
    const auto n_params = param_limits.size();
    psr_utils::SnailScheme scheme(nsegments, 0, tsegment);
    branching_pattern.reserve(nsegments - 1);
    const auto leaves_stride = (n_params + 2) * 2;
    const auto coord         = scheme.get_coord(0);
    const auto leaves =
        core::poly_taylor_leaves(param_arr, dparams_lim, coord, leaves_stride);
    const auto n_leaves = leaves.size() / leaves_stride;
    // Get last leaf
    auto leaf = std::span(leaves).subspan((leaves_stride * (n_leaves - 1)),
                                          leaves_stride);
    std::vector<double> leaf_data(leaves_stride);
    std::ranges::copy(leaf, leaf_data.begin());

    for (SizeType prune_level = 1; prune_level < nsegments - 1; ++prune_level) {
        const auto coord_cur  = scheme.get_coord(prune_level);
        const auto leaves_arr = poly_taylor_branch(
            leaf_data, coord_cur, n_params, fold_bins, tol_bins, param_limits);
        const auto n_leaves_branch = leaves_arr.size() / leaves_stride;
        branching_pattern.push_back(n_leaves_branch);
        const auto leaf_start = leaves_stride * (n_leaves_branch - 1);
        std::ranges::copy(
            leaves_arr.begin() + static_cast<IndexType>(leaf_start),
            leaves_arr.begin() +
                static_cast<IndexType>(leaf_start + leaves_stride),
            leaf_data.begin());
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
