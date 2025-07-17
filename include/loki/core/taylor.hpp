#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/utils/suggestions.hpp"

namespace loki::core {

std::tuple<std::vector<SizeType>, double>
ffa_taylor_resolve(std::span<const double> pset_cur,
                   std::span<const std::vector<double>> param_arr,
                   SizeType ffa_level,
                   SizeType latter,
                   double tseg_brute,
                   SizeType nbins);

std::tuple<std::vector<SizeType>, std::vector<double>>
poly_taylor_resolve_batch(std::span<const double> batch_leaves,
                          std::pair<double, double> coord_add,
                          std::pair<double, double> coord_init,
                          std::span<const std::vector<double>> param_arr,
                          SizeType fold_bins,
                          SizeType n_leaves,
                          SizeType n_params);

std::tuple<std::vector<SizeType>, std::vector<double>>
poly_taylor_resolve_snap_batch(std::span<const double> batch_leaves,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               SizeType fold_bins,
                               SizeType n_leaves,
                               SizeType n_params);

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
                         SizeType branch_max = 16U);

template <typename FoldType>
void poly_taylor_suggest(
    std::span<const FoldType> fold_segment,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType poly_order,
    std::span<const SizeType> score_widths,
    const detection::ScoringFunction<FoldType>& scoring_func,
    utils::SuggestionStruct<FoldType>& sugg_struct);

} // namespace loki::core
