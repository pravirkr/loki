#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::core {

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
 * @param minimum_snap_cells Threshold for significant snap (default: 5.0)
 * @return std::tuple<std::vector<SizeType>, std::vector<SizeType>,
 * std::vector<SizeType>> (idx_circular_snap, idx_circular_crackle, idx_taylor)
 */
std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask(std::span<const double> leaves_batch,
                     SizeType n_leaves,
                     SizeType n_params,
                     double minimum_snap_cells);

std::vector<SizeType>
circ_taylor_branch_batch(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch_batch,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType fold_bins,
                         double tol_bins,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max,
                         double minimum_snap_cells);

SizeType circ_taylor_validate_batch(std::span<double> leaves_batch,
                                    std::span<SizeType> leaves_origins,
                                    SizeType n_leaves,
                                    SizeType n_params,
                                    double p_orb_min,
                                    double x_mass_const,
                                    double minimum_snap_cells);

void circ_taylor_resolve_batch(std::span<const double> leaves_batch,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_cur,
                               std::pair<double, double> coord_init,
                               std::span<const std::vector<double>> param_arr,
                               std::span<SizeType> pindex_flat_batch,
                               std::span<float> relative_phase_batch,
                               SizeType nbins,
                               SizeType n_leaves,
                               SizeType n_params,
                               double minimum_snap_cells);

void circ_taylor_transform_batch(std::span<double> leaves_batch,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 bool use_conservative_tile,
                                 double minimum_snap_cells);

std::vector<double>
generate_bp_taylor_circular(std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            const std::vector<ParamLimitType>& param_limits,
                            double tseg_ffa,
                            SizeType nsegments,
                            SizeType fold_bins,
                            double tol_bins,
                            SizeType ref_seg,
                            bool use_conservative_tile = false);

} // namespace loki::core