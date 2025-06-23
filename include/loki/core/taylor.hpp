#pragma once

#include <span>
#include <tuple>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

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
poly_taylor_resolve_batch(const xt::xtensor<double, 3>& leaf_batch,
                          std::pair<double, double> coord_add,
                          std::pair<double, double> coord_init,
                          std::span<const std::vector<double>> param_arr,
                          SizeType fold_bins);

std::tuple<std::vector<SizeType>, std::vector<double>>
poly_taylor_resolve_snap_batch(const xt::xtensor<double, 3>& leaf_batch,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               SizeType fold_bins);

std::tuple<xt::xtensor<double, 3>, std::vector<SizeType>>
poly_taylor_branch_batch(const xt::xtensor<double, 3>& param_set_batch,
                         std::pair<double, double> coord_cur,
                         SizeType fold_bins,
                         double tol_bins,
                         SizeType poly_order,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max = 16U);

template <typename FoldType>
void poly_taylor_suggest(std::span<const FoldType> fold_segment,
                         std::pair<double, double> coord_init,
                         std::span<const std::vector<double>> param_arr,
                         std::span<const double> dparams,
                         SizeType poly_order,
                         std::span<const SizeType> score_widths,
                         detection::ScoringFunction<FoldType> scoring_func,
                         utils::SuggestionStruct<FoldType>& sugg_struct);

void poly_shift_add_batch(const xt::xtensor<float, 3>& segment_batch,
                          std::span<const SizeType> shift_batch,
                          const xt::xtensor<float, 3>& folds,
                          std::span<const SizeType> isuggest_batch,
                          xt::xtensor<float, 3>& out);

void poly_shift_add_complex_batch(
    const xt::xtensor<ComplexType, 3>& segment_batch,
    std::span<const double> shift_batch,
    const xt::xtensor<ComplexType, 3>& folds,
    std::span<const SizeType> isuggest_batch,
    xt::xtensor<ComplexType, 3>& out);

} // namespace loki::core
