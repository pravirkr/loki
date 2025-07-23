#include "loki/core/taylor.hpp"

#include <span>
#include <vector>

#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/utils.hpp"

namespace loki::core {

std::tuple<std::vector<SizeType>, double>
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
    for (SizeType ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] = utils::find_nearest_sorted_idx(
            std::span(param_arr[ip]), pset_prev[ip]);
    }
    return {pindex_prev, relative_phase};
}

std::tuple<std::vector<SizeType>, std::vector<double>>
poly_taylor_resolve_batch(std::span<const double> batch_leaves,
                          std::pair<double, double> coord_add,
                          std::pair<double, double> coord_init,
                          std::span<const std::vector<double>> param_arr,
                          SizeType fold_bins,
                          SizeType n_leaves,
                          SizeType n_params) {
    const SizeType leaves_stride_param = 2;
    const SizeType leaves_stride_batch = (n_params + 2) * leaves_stride_param;
    error_check::check_equal(n_params, param_arr.size(),
                             "nparams should be equal to param_arr size");
    error_check::check_greater_equal(batch_leaves.size(),
                                     n_leaves * leaves_stride_batch,
                                     "batch_leaves size mismatch");
    const double delta_t = coord_add.first - coord_init.first;

    // Allocate working memory for transformed parameters and delays
    std::vector<double> kvec_new_batch(n_leaves * n_params);
    std::vector<double> delay_batch(n_leaves);
    psr_utils::shift_params_batch(batch_leaves, delta_t, n_leaves, n_params,
                                  leaves_stride_batch, kvec_new_batch,
                                  delay_batch);

    // Calculate relative phases
    std::vector<double> relative_phase_batch(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto freq_old =
            batch_leaves[(i * leaves_stride_batch) + ((n_params - 1) * 2)];
        relative_phase_batch[i] = psr_utils::get_phase_idx(
            delta_t, freq_old, fold_bins, delay_batch[i]);
    }
    // Calculate flattened parameter indices
    std::vector<SizeType> param_idx_flat(n_leaves);
    const SizeType f_size = param_arr[n_params - 1].size();
    SizeType hint_a{}, hint_f{};
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType kvec_offset = i * n_params;
        const auto f_val = kvec_new_batch[kvec_offset + (n_params - 1)];
        const auto a_val = kvec_new_batch[kvec_offset + (n_params - 2)];
        const auto idx_f = utils::find_nearest_sorted_idx_scan(
            param_arr[n_params - 1], f_val, hint_f);
        const auto idx_a = utils::find_nearest_sorted_idx_scan(
            param_arr[n_params - 2], a_val, hint_a);
        param_idx_flat[i] = idx_a * f_size + idx_f;
    }

    return {std::move(param_idx_flat), std::move(relative_phase_batch)};
}

std::tuple<std::vector<SizeType>, std::vector<double>>
poly_taylor_resolve_snap_batch(std::span<const double> batch_leaves,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               SizeType fold_bins,
                               SizeType n_leaves,
                               SizeType n_params) {
    const SizeType leaves_stride_param = 2;
    const SizeType leaves_stride_batch = (n_params + 2) * leaves_stride_param;
    error_check::check_equal(n_params, 4U,
                             "nparams should be 4 for circular orbit resolve");
    error_check::check_equal(n_params, param_arr.size(),
                             "nparams should be equal to param_arr size");
    error_check::check_greater_equal(batch_leaves.size(),
                                     n_leaves * leaves_stride_batch,
                                     "batch_leaves size mismatch");
    const double delta_t = coord_add.first - coord_init.first;

    // Create mask for circular orbit conditions
    std::vector<bool> mask(n_leaves);
    std::vector<SizeType> idx_circular, idx_normal;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType batch_offset = i * leaves_stride_batch;
        const double snap           = batch_leaves[batch_offset];
        const double dsnap          = batch_leaves[batch_offset + 1];
        const double accel          = batch_leaves[batch_offset + 4];

        mask[i] = (accel != 0.0) && (snap != 0.0) && ((-snap / accel) > 0.0) &&
                  (std::abs(snap / dsnap) > 5.0);

        if (mask[i]) {
            idx_circular.push_back(i);
        } else {
            idx_normal.push_back(i);
        }
    }

    // Allocate working memory for transformed parameters and delays
    std::vector<double> kvec_new_batch(n_leaves * n_params);
    std::vector<double> delay_batch(n_leaves);

    // Process circular indices
    if (!idx_circular.empty()) {
        std::vector<double> param_vec_circular_data(idx_circular.size() *
                                                    leaves_stride_batch);
        // Extract subset for circular processing
        for (SizeType i = 0; i < idx_circular.size(); ++i) {
            const SizeType orig_idx   = idx_circular[i];
            const SizeType src_offset = orig_idx * leaves_stride_batch;
            const SizeType dst_offset = i * leaves_stride_batch;
            std::copy(
                batch_leaves.begin() + static_cast<IndexType>(src_offset),
                batch_leaves.begin() +
                    static_cast<IndexType>(src_offset + leaves_stride_batch),
                param_vec_circular_data.begin() +
                    static_cast<IndexType>(dst_offset));
        }

        std::vector<double> kvec_circ(idx_circular.size() * n_params);
        std::vector<double> delay_circ(idx_circular.size());
        psr_utils::shift_params_circular_batch(
            param_vec_circular_data, delta_t, idx_circular.size(), n_params,
            leaves_stride_batch, kvec_circ, delay_circ);

        // Copy results back to main arrays
        for (SizeType i = 0; i < idx_circular.size(); ++i) {
            const SizeType orig_idx   = idx_circular[i];
            const SizeType src_offset = i * n_params;
            const SizeType dst_offset = orig_idx * n_params;
            std::copy(kvec_circ.begin() + static_cast<IndexType>(src_offset),
                      kvec_circ.begin() +
                          static_cast<IndexType>(src_offset + n_params),
                      kvec_new_batch.begin() +
                          static_cast<IndexType>(dst_offset));
            delay_batch[orig_idx] = delay_circ[i];
        }
    }

    // Process normal indices
    if (!idx_normal.empty()) {
        std::vector<double> param_vec_normal_data(idx_normal.size() *
                                                  leaves_stride_batch);
        // Extract normal data
        for (SizeType i = 0; i < idx_normal.size(); ++i) {
            const SizeType orig_idx   = idx_normal[i];
            const SizeType src_offset = orig_idx * leaves_stride_batch;
            const SizeType dst_offset = i * leaves_stride_batch;
            std::copy(
                batch_leaves.begin() + static_cast<IndexType>(src_offset),
                batch_leaves.begin() +
                    static_cast<IndexType>(src_offset + leaves_stride_batch),
                param_vec_normal_data.begin() +
                    static_cast<IndexType>(dst_offset));
        }

        std::vector<double> kvec_norm(idx_normal.size() * n_params);
        std::vector<double> delay_norm(idx_normal.size());

        psr_utils::shift_params_batch(
            param_vec_normal_data, delta_t, idx_normal.size(), n_params,
            leaves_stride_batch, kvec_norm, delay_norm);

        // Copy results back to main arrays
        for (SizeType i = 0; i < idx_normal.size(); ++i) {
            const SizeType orig_idx   = idx_normal[i];
            const SizeType src_offset = i * n_params;
            const SizeType dst_offset = orig_idx * n_params;
            std::copy(kvec_norm.begin() + static_cast<IndexType>(src_offset),
                      kvec_norm.begin() +
                          static_cast<IndexType>(src_offset + n_params),
                      kvec_new_batch.begin() +
                          static_cast<IndexType>(dst_offset));
            delay_batch[orig_idx] = delay_norm[i];
        }
    }

    // Calculate relative phases
    std::vector<double> relative_phase_batch(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto freq_old =
            batch_leaves[(i * leaves_stride_batch) + ((n_params - 1) * 2)];
        relative_phase_batch[i] = psr_utils::get_phase_idx(
            delta_t, freq_old, fold_bins, delay_batch[i]);
    }

    // Calculate flattened parameter indices (same as previous function)
    std::vector<SizeType> param_idx_flat(n_leaves);
    const SizeType f_size = param_arr[n_params - 1].size();
    SizeType hint_a{}, hint_f{};
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType kvec_offset = i * n_params;
        const auto f_val = kvec_new_batch[kvec_offset + (n_params - 1)];
        const auto a_val = kvec_new_batch[kvec_offset + (n_params - 2)];
        const auto idx_f = utils::find_nearest_sorted_idx_scan(
            param_arr[n_params - 1], f_val, hint_f);
        const auto idx_a = utils::find_nearest_sorted_idx_scan(
            param_arr[n_params - 2], a_val, hint_a);
        param_idx_flat[i] = idx_a * f_size + idx_f;
    }

    return {std::move(param_idx_flat), std::move(relative_phase_batch)};
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

            if (shift_bins_batch[flat_idx] <= tol_bins) {
                // Mask triggered: only use current value
                const SizeType pad_offset =
                    (i * n_params * branch_max) + (j * branch_max);
                pad_branched_params[pad_offset] = param_cur_val;
                pad_branched_dparams[flat_idx]  = dparam_cur_val;
                branched_counts[flat_idx]       = 1;
            } else {
                // Normal branching - use the existing robust function
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
    const auto [t0, scale]   = coord_init;
    const auto leaves_stride = sugg_tree.get_leaves_stride();

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
    // Calculate scores
    std::vector<float> scores(n_leaves);
    scoring_func(fold_segment, scores, n_leaves, boxcar_widths_cache);
    // Initialize the SuggestionStruct with the generated data
    sugg_tree.add_initial(param_sets, fold_segment, scores, n_leaves);
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
