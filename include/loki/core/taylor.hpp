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
    double snap_threshold);

SizeType poly_taylor_validate_circular_batch(std::span<double> leaves_batch,
                                             std::span<SizeType> leaves_origins,
                                             SizeType n_leaves,
                                             SizeType n_params,
                                             double p_orb_min,
                                             double snap_threshold = 5.0);

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
    double snap_threshold);

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
    double snap_threshold);

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
    double snap_threshold);

/**
 * @brief Generate robust masks to identify circular orbit candidates.
 *
 * Returns three vector of indices
 *   - idx_circular_snap: stable, significant, and physical (high-quality
 * circular orbits)
 *   - idx_circular_crackle: unstable snap/accel but stable crackle/jerk
 * (secondary circular candidates)
 *   - idx_taylor: everything else (not circular)
 *
 * @param leaves_batch   Flat span of leaves (size: n_leaves * (n_params + 2) *
 * 2)
 * @param n_leaves      Number of leaves (batches)
 * @param n_params      Number of Taylor parameters
 * @param snap_threshold Threshold for significant snap (default: 5.0)
 * @return std::tuple<std::vector<SizeType>, std::vector<SizeType>,
 * std::vector<SizeType>> (idx_circular_snap, idx_circular_crackle, idx_taylor)
 */
std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circular_mask(std::span<const double> leaves_batch,
                  SizeType n_leaves,
                  SizeType n_params,
                  double snap_threshold = 5.0);

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
    double snap_threshold);

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

void poly_taylor_transform_circular_batch(std::span<double> leaves_batch,
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
generate_bp_taylor_approx(std::span<const std::vector<double>> param_arr,
                          std::span<const double> dparams,
                          const std::vector<ParamLimitType>& param_limits,
                          double tseg_ffa,
                          SizeType nsegments,
                          SizeType fold_bins,
                          double tol_bins,
                          SizeType ref_seg,
                          bool use_conservative_errors = false);

// Generate an exact branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_taylor(std::span<const std::vector<double>> param_arr,
                   std::span<const double> dparams,
                   const std::vector<ParamLimitType>& param_limits,
                   double tseg_ffa,
                   SizeType nsegments,
                   SizeType fold_bins,
                   double tol_bins,
                   SizeType ref_seg,
                   bool use_conservative_errors = false);

std::vector<double>
generate_bp_taylor_circular(std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            const std::vector<ParamLimitType>& param_limits,
                            double tseg_ffa,
                            SizeType nsegments,
                            SizeType fold_bins,
                            double tol_bins,
                            SizeType ref_seg,
                            bool use_conservative_errors = false);
} // namespace loki::core
