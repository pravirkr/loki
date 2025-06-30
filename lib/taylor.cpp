#include "loki/core/taylor.hpp"

#include <span>
#include <vector>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

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
poly_taylor_resolve_batch(const xt::xtensor<double, 3>& leaf_batch,
                          std::pair<double, double> coord_add,
                          std::pair<double, double> coord_init,
                          std::span<const std::vector<double>> param_arr,
                          SizeType fold_bins) {

    const SizeType n_leaves = leaf_batch.shape(0);
    const SizeType nparams  = param_arr.size();
    const double delta_t    = coord_add.first - coord_init.first;

    auto param_vec_batch =
        xt::view(leaf_batch, xt::all(), xt::range(0, nparams), xt::all());
    auto freq_old_batch = xt::view(leaf_batch, xt::all(), nparams - 1, 0);
    auto [kvec_new_batch, delay_batch] =
        psr_utils::shift_params_batch(param_vec_batch, delta_t);

    // Calculate relative phases
    std::vector<double> relative_phase_batch(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        relative_phase_batch[i] = psr_utils::get_phase_idx(
            delta_t, freq_old_batch[i], fold_bins, delay_batch[i]);
    }
    // Calculate flattened parameter indices
    std::vector<SizeType> param_idx_flat(n_leaves);
    const SizeType f_size = param_arr[nparams - 1].size();

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto f_val = kvec_new_batch(i, nparams - 1, 0);
        const auto a_val = kvec_new_batch(i, nparams - 2, 0);
        const auto idx_f =
            utils::find_nearest_sorted_idx(param_arr[nparams - 1], f_val);
        const auto idx_a =
            utils::find_nearest_sorted_idx(param_arr[nparams - 2], a_val);
        param_idx_flat[i] = idx_a * f_size + idx_f;
    }

    return {std::move(param_idx_flat), std::move(relative_phase_batch)};
}

std::tuple<std::vector<SizeType>, std::vector<double>>
poly_taylor_resolve_snap_batch(const xt::xtensor<double, 3>& leaf_batch,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               SizeType fold_bins) {

    const SizeType n_leaves = leaf_batch.shape(0);
    const SizeType nparams  = param_arr.size();
    const double delta_t    = coord_add.first - coord_init.first;

    const auto param_vec_batch =
        xt::view(leaf_batch, xt::all(), xt::range(0, nparams), xt::all());
    const auto freq_old_batch = xt::view(leaf_batch, xt::all(), nparams - 1, 0);
    const auto snap_old_batch = xt::view(leaf_batch, xt::all(), 0, 0);
    const auto dsnap_old_batch = xt::view(leaf_batch, xt::all(), 0, 1);
    const auto accel_old_batch = xt::view(leaf_batch, xt::all(), 2, 0);

    // Create mask for circular orbit conditions
    std::vector<bool> mask(n_leaves);
    std::vector<SizeType> idx_circular, idx_normal;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto accel = accel_old_batch[i];
        const auto snap  = snap_old_batch[i];
        const auto dsnap = dsnap_old_batch[i];

        mask[i] = (accel != 0.0) && (snap != 0.0) && ((-snap / accel) > 0.0) &&
                  (std::abs(snap / dsnap) > 5.0);

        if (mask[i]) {
            idx_circular.push_back(i);
        } else {
            idx_normal.push_back(i);
        }
    }

    // Initialize output arrays
    xt::xtensor<double, 3> kvec_new_batch({n_leaves, nparams, 2}, 0.0);
    xt::xtensor<double, 1> delay_batch({n_leaves}, 0.0);

    // Process circular indices
    if (!idx_circular.empty()) {
        // Extract subset for circular processing
        xt::xtensor<double, 3> param_vec_circular(
            {idx_circular.size(), nparams, 2});
        for (SizeType i = 0; i < idx_circular.size(); ++i) {
            const SizeType orig_idx = idx_circular[i];
            xt::view(param_vec_circular, i, xt::all(), xt::all()) =
                xt::view(param_vec_batch, orig_idx, xt::all(), xt::all());
        }

        auto [kvec_new_circ, delay_circ] =
            psr_utils::shift_params_circular_batch(param_vec_circular, delta_t);

        // Copy results back to main arrays
        for (SizeType i = 0; i < idx_circular.size(); ++i) {
            const SizeType orig_idx = idx_circular[i];
            xt::view(kvec_new_batch, orig_idx, xt::all(), xt::all()) =
                xt::view(kvec_new_circ, i, xt::all(), xt::all());
            delay_batch[orig_idx] = delay_circ[i];
        }
    }

    // Process normal indices
    if (!idx_normal.empty()) {
        // Extract subset for normal processing
        xt::xtensor<double, 3> param_vec_normal(
            {idx_normal.size(), nparams, 2});
        for (SizeType i = 0; i < idx_normal.size(); ++i) {
            const SizeType orig_idx = idx_normal[i];
            xt::view(param_vec_normal, i, xt::all(), xt::all()) =
                xt::view(param_vec_batch, orig_idx, xt::all(), xt::all());
        }

        auto [kvec_new_norm, delay_norm] =
            psr_utils::shift_params_batch(param_vec_normal, delta_t);

        // Copy results back to main arrays
        for (SizeType i = 0; i < idx_normal.size(); ++i) {
            const SizeType orig_idx = idx_normal[i];
            xt::view(kvec_new_batch, orig_idx, xt::all(), xt::all()) =
                xt::view(kvec_new_norm, i, xt::all(), xt::all());
            delay_batch[orig_idx] = delay_norm[i];
        }
    }

    // Calculate relative phases
    std::vector<double> relative_phase_batch(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        relative_phase_batch[i] = psr_utils::get_phase_idx(
            delta_t, freq_old_batch[i], fold_bins, delay_batch[i]);
    }

    // Calculate flattened parameter indices (same as previous function)
    std::vector<SizeType> param_idx_flat(n_leaves);
    const SizeType f_size = param_arr[nparams - 1].size();

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto f_val = kvec_new_batch(i, nparams - 1, 0);
        const auto a_val = kvec_new_batch(i, nparams - 2, 0);
        const auto idx_f =
            utils::find_nearest_sorted_idx(param_arr[nparams - 1], f_val);
        const auto idx_a =
            utils::find_nearest_sorted_idx(param_arr[nparams - 2], a_val);
        param_idx_flat[i] = idx_a * f_size + idx_f;
    }

    return {std::move(param_idx_flat), std::move(relative_phase_batch)};
}

std::tuple<xt::xtensor<double, 3>, std::vector<SizeType>>
poly_taylor_branch_batch(const xt::xtensor<double, 3>& param_set_batch,
                         std::pair<double, double> coord_cur,
                         SizeType fold_bins,
                         double tol_bins,
                         SizeType poly_order,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max) {

    const SizeType n_batch = param_set_batch.shape(0);
    const SizeType nparams = param_set_batch.shape(1) - 2;
    error_check::check_equal(nparams, poly_order,
                             "nparams should be equal to poly_order");

    const auto [_, scale_cur] = coord_cur;

    // Extract parameter arrays
    const xt::xtensor<double, 2> param_cur_batch =
        xt::view(param_set_batch, xt::all(), xt::range(0, nparams), 0);
    const xt::xtensor<double, 2> dparam_cur_batch =
        xt::view(param_set_batch, xt::all(), xt::range(0, nparams), 1);
    const xt::xtensor<double, 1> f0_batch =
        xt::view(param_set_batch, xt::all(), nparams, 0);
    const xt::xtensor<double, 1> t0_batch =
        xt::view(param_set_batch, xt::all(), nparams + 1, 0);
    const xt::xtensor<double, 1> scale_batch =
        xt::view(param_set_batch, xt::all(), nparams + 1, 1);
    const xt::xtensor<double, 1> f_max_batch =
        xt::view(param_cur_batch, xt::all(), nparams - 1);

    // Calculate optimal parameters
    const double tseg_cur = 2.0 * scale_cur;
    const double t_ref    = tseg_cur / 2.0;

    const auto dparam_opt_batch = psr_utils::poly_taylor_step_d_vec(
        nparams, tseg_cur, fold_bins, tol_bins, f_max_batch, t_ref);

    const auto shift_bins_batch = psr_utils::poly_taylor_shift_d_vec(
        dparam_cur_batch, dparam_opt_batch, tseg_cur, fold_bins, f_max_batch,
        t_ref);

    // Vectorized padded branching
    xt::xtensor<double, 3> pad_branched_params({n_batch, nparams, branch_max},
                                               0.0);
    xt::xtensor<double, 2> pad_branched_dparams({n_batch, nparams}, 0.0);
    xt::xtensor<SizeType, 2> branched_counts({n_batch, nparams}, 0);

    for (SizeType i = 0; i < n_batch; ++i) {
        for (SizeType j = 0; j < nparams; ++j) {
            const auto [p_min, p_max] = param_limits[j];
            auto slice_span = xt::view(pad_branched_params, i, j, xt::all());
            auto [dparam_act, count] = psr_utils::branch_param_padded(
                slice_span, param_cur_batch(i, j), dparam_cur_batch(i, j),
                dparam_opt_batch(i, j), p_min, p_max);
            pad_branched_dparams(i, j) = dparam_act;
            branched_counts(i, j)      = count;
        }
    }

    // Vectorized selection based on mask
    for (SizeType i = 0; i < n_batch; ++i) {
        for (SizeType j = 0; j < nparams; ++j) {
            if (shift_bins_batch(i, j) <= tol_bins) {
                // Reset branched params and use current value
                for (SizeType k = 0; k < branch_max; ++k) {
                    pad_branched_params(i, j, k) = 0.0;
                }
                pad_branched_params(i, j, 0) = param_cur_batch(i, j);
                pad_branched_dparams(i, j)   = dparam_cur_batch(i, j);
                branched_counts(i, j)        = 1;
            }
        }
    }

    // Optimized padded Cartesian product
    auto [batch_leaves_taylor, batch_origins] = utils::cartesian_prod_padded(
        pad_branched_params, branched_counts, n_batch, nparams);

    // Construct final output
    const SizeType total_leaves = batch_origins.size();
    xt::xtensor<double, 3> batch_leaves({total_leaves, poly_order + 2, 2}, 0.0);
    xt::view(batch_leaves, xt::all(), xt::range(0, nparams), 0) =
        batch_leaves_taylor;

    // Fill dparams using advanced indexing
    for (SizeType i = 0; i < total_leaves; ++i) {
        const SizeType origin = batch_origins[i];
        for (SizeType j = 0; j < nparams; ++j) {
            batch_leaves(i, j, 1) = pad_branched_dparams(origin, j);
        }
        batch_leaves(i, poly_order, 0)     = f0_batch(origin);
        batch_leaves(i, poly_order + 1, 0) = t0_batch(origin);
        batch_leaves(i, poly_order + 1, 1) = scale_batch(origin);
    }

    return {std::move(batch_leaves), std::move(batch_origins)};
}

template <typename FoldType>
void poly_taylor_suggest(std::span<const FoldType> fold_segment,
                         std::pair<double, double> coord_init,
                         std::span<const std::vector<double>> param_arr,
                         std::span<const double> dparams,
                         SizeType poly_order,
                         std::span<const SizeType> score_widths,
                         detection::ScoringFunction<FoldType> scoring_func,
                         utils::SuggestionStruct<FoldType>& sugg_struct) {
    const auto nparams = param_arr.size();
    error_check::check_equal(nparams, poly_order,
                             "nparams should be equal to poly_order");
    SizeType n_param_sets = 1;
    for (const auto& arr : param_arr) {
        n_param_sets *= arr.size();
    }
    const auto nbins       = fold_segment.size() / (n_param_sets * 2);
    const auto [t0, scale] = coord_init;

    // Create parameter sets tensor: (n_param_sets, poly_order + 2, 2)
    xt::xtensor<double, 3> param_sets({n_param_sets, poly_order + 2, 2}, 0.0);

    SizeType param_idx = 0;
    for (const auto& p_set_view : utils::cartesian_product_view(param_arr)) {
        // Fill first nparams dimensions with parameter values and dparams
        for (SizeType j = 0; j < nparams; ++j) {
            param_sets(param_idx, j, 0) = p_set_view[j];
            param_sets(param_idx, j, 1) = dparams[j];
        }
        param_sets(param_idx, poly_order, 0)     = p_set_view[nparams - 1];
        param_sets(param_idx, poly_order + 1, 0) = t0;
        param_sets(param_idx, poly_order + 1, 1) = scale;
        ++param_idx;
    }

    // Create folds tensor: (n_param_sets, 2, nbins)
    auto folds = xt::adapt(fold_segment.data(), {n_param_sets, 2, nbins});

    // Calculate scores
    std::vector<float> scores(n_param_sets);
    scoring_func(folds, score_widths, scores);

    // Create backtracks tensor: (n_param_sets, 2 + nparams)
    xt::xtensor<SizeType, 2> backtracks({n_param_sets, 2 + nparams}, 0);

    // Initialize the SuggestionStruct with the generated data
    sugg_struct.add_initial(param_sets, folds, scores, backtracks);
}
} // namespace loki::core
