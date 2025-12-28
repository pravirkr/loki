#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/utils/suggestions.hpp"

namespace loki::core {

// Old method (not used anymore)
std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve(std::span<const double> pset_cur,
                   std::span<const std::vector<double>> param_arr,
                   SizeType ffa_level,
                   SizeType latter,
                   double tseg_brute,
                   SizeType nbins);

void ffa_taylor_resolve_freq_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_accel_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_jerk_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_snap_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_crackle_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

std::vector<double>
poly_taylor_leaves(std::span<const std::vector<double>> param_arr,
                   std::span<const double> dparams,
                   SizeType poly_order,
                   std::pair<double, double> coord_init);

template <SupportedFoldType FoldType>
void poly_taylor_suggest(
    std::span<const FoldType> fold_segment,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType poly_order,
    SizeType nbins,
    const detection::ScoringFunction<FoldType>& scoring_func,
    detection::BoxcarWidthsCache& boxcar_widths_cache,
    utils::SuggestionTree<FoldType>& sugg_tree);

std::vector<SizeType>
poly_taylor_branch_batch(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch_batch,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType fold_bins,
                         double tol_bins,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max);

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
    SizeType n_params);

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
    SizeType n_params);

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
    SizeType n_params);

void poly_taylor_transform_accel_batch(std::span<double> leaves_batch,
                                       std::pair<double, double> coord_next,
                                       std::pair<double, double> coord_cur,
                                       SizeType n_leaves,
                                       SizeType n_params,
                                       bool conservative_errors,
                                       double snap_threshold);

void poly_taylor_transform_jerk_batch(std::span<double> leaves_batch,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool conservative_errors,
                                      double snap_threshold);

void poly_taylor_transform_snap_batch(std::span<double> leaves_batch,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool conservative_errors,
                                      double snap_threshold);

std::vector<double>
poly_taylor_branch(std::span<const double> leaf,
                   std::pair<double, double> coord_cur,
                   SizeType n_params,
                   SizeType fold_bins,
                   double tol_bins,
                   const std::vector<ParamLimitType>& param_limits);

// Generate an approximate branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_poly_taylor_approx(std::span<const std::vector<double>> param_arr,
                               std::span<const double> dparams,
                               const std::vector<ParamLimitType>& param_limits,
                               double tseg_ffa,
                               SizeType nsegments,
                               SizeType fold_bins,
                               double tol_bins,
                               SizeType ref_seg,
                               bool use_conservative_tile = false);

// Generate an exact branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_poly_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        const std::vector<ParamLimitType>& param_limits,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType fold_bins,
                        double tol_bins,
                        SizeType ref_seg,
                        bool use_conservative_tile = false);

} // namespace loki::core
