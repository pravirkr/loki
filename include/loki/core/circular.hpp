#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

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
 * @param leaves_batch   Flat span of leaves (size:> n_leaves * (n_params + 2) *
 * 2)
 * @param indices_batch  Flat span of indices (size: n_leaves)
 * @param n_leaves      Number of leaves (batches)
 * @param n_params      Number of Taylor parameters
 * @param minimum_snap_cells Threshold for significant snap (default: 5.0)
 * @return std::tuple<std::vector<SizeType>, std::vector<SizeType>,
 * std::vector<SizeType>> (idx_circular_snap, idx_circular_crackle, idx_taylor)
 */
std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask_scattered(std::span<const double> leaves_batch,
                               std::span<SizeType> indices_batch,
                               SizeType n_leaves,
                               SizeType n_params,
                               double minimum_snap_cells);

std::tuple<std::vector<SizeType>, std::vector<SizeType>, std::vector<SizeType>>
get_circ_taylor_mask(std::span<const double> leaves_batch,
                     SizeType n_leaves,
                     SizeType n_params,
                     double minimum_snap_cells);

SizeType circ_taylor_branch_batch(std::span<const double> leaves_tree,
                                  std::span<double> leaves_branch,
                                  std::span<SizeType> leaves_origins,
                                  std::pair<double, double> coord_cur,
                                  SizeType nbins,
                                  double eta,
                                  SizeType branch_max,
                                  SizeType n_leaves,
                                  double minimum_snap_cells,
                                  memory::BranchingWorkspace& branch_ws);

SizeType circ_taylor_validate_batch(std::span<double> leaves_branch,
                                    std::span<SizeType> leaves_origins,
                                    SizeType n_leaves,
                                    double p_orb_min,
                                    double x_mass_const,
                                    double minimum_snap_cells);

void circ_taylor_resolve_batch(std::span<const double> leaves_tree,
                               std::span<SizeType> param_indices,
                               std::span<float> phase_shift,
                               std::span<const ParamLimit> param_limits,
                               std::pair<double, double> coord_add,
                               std::pair<double, double> coord_cur,
                               std::pair<double, double> coord_init,
                               SizeType n_accel_init,
                               SizeType n_freq_init,
                               SizeType nbins,
                               SizeType n_leaves,
                               double minimum_snap_cells);

void circ_taylor_ascend_resolve_batch(
    std::span<const double> leaves_tree,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    std::span<const ParamLimit> param_limits,
    std::span<const std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_segments,
    double minimum_snap_cells);

void circ_taylor_transform_batch(std::span<double> leaves_tree,
                                 std::span<SizeType> indices_tree,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 bool use_conservative_tile,
                                 double minimum_snap_cells);

std::vector<double>
generate_bp_circ_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        bool use_conservative_tile = false);

#ifdef LOKI_ENABLE_CUDA

SizeType
circ_taylor_branch_batch_cuda(cuda::std::span<const double> leaves_tree,
                              cuda::std::span<double> leaves_branch,
                              cuda::std::span<uint32_t> leaves_origins,
                              cuda::std::span<uint8_t> validation_mask,
                              std::pair<double, double> coord_cur,
                              SizeType nbins,
                              double eta,
                              SizeType branch_max,
                              SizeType n_leaves,
                              double minimum_snap_cells,
                              memory::BranchingWorkspaceCUDAView branch_ws,
                              memory::CUBScratchArena& scratch_ws,
                              cudaStream_t stream);

SizeType
circ_taylor_validate_batch_cuda(cuda::std::span<const double> leaves_branch,
                                cuda::std::span<uint8_t> validation_mask,
                                SizeType n_leaves,
                                double p_orb_min,
                                double x_mass_const,
                                double minimum_snap_cells,
                                memory::CUBScratchArena& scratch_ws,
                                cudaStream_t stream);

void circ_taylor_resolve_batch_cuda(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    cuda::std::span<const ParamLimit> param_limits,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    double minimum_snap_cells,
    cudaStream_t stream);

void circ_taylor_ascend_resolve_batch_cuda(
    cuda::std::span<const double> leaves_tree,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    cuda::std::span<const ParamLimit> param_limits,
    cuda::std::span<const cuda::std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_segments,
    double minimum_snap_cells,
    cudaStream_t stream);

void circ_taylor_transform_batch_cuda(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    bool use_conservative_tile,
    double minimum_snap_cells,
    cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::core