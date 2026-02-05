#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#include "loki/cuda_utils.cuh"
#endif // LOKI_ENABLE_CUDA

namespace loki::core {

// Old method (not used anymore)
std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve_generic(std::span<const double> pset_cur,
                           std::span<const SizeType> param_grid_count_prev,
                           std::span<const ParamLimit> param_limits,
                           SizeType ffa_level,
                           SizeType latter,
                           double tseg_brute,
                           SizeType nbins);

void ffa_taylor_resolve_freq_batch(SizeType n_freqs_cur,
                                   SizeType n_freqs_prev,
                                   const ParamLimit& lim_freq,
                                   std::span<coord::FFACoordFreq> coords,
                                   SizeType ffa_level,
                                   double tseg_brute,
                                   SizeType nbins);

void ffa_taylor_resolve_poly_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins,
    SizeType n_params);

SizeType poly_taylor_seed(std::span<const SizeType> param_grid_count_init,
                          std::span<const double> dparams_init,
                          std::span<const ParamLimit> param_limits,
                          std::span<double> seed_leaves,
                          std::pair<double, double> coord_init,
                          SizeType n_params);

std::vector<SizeType>
poly_taylor_branch_batch_generic(std::span<const double> leaves_batch,
                                 std::pair<double, double> coord_cur,
                                 std::span<double> leaves_branch_batch,
                                 SizeType nbins,
                                 double eta,
                                 std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 SizeType n_params);

SizeType poly_taylor_branch_batch(std::span<const double> leaves_tree,
                                  std::span<double> leaves_branch,
                                  std::span<SizeType> leaves_origins,
                                  std::pair<double, double> coord_cur,
                                  SizeType nbins,
                                  double eta,
                                  std::span<const ParamLimit> param_limits,
                                  SizeType branch_max,
                                  SizeType n_leaves,
                                  SizeType n_params,
                                  utils::BranchingWorkspaceView ws);

void poly_taylor_resolve_batch(std::span<const double> leaves_branch,
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
                               SizeType n_params);

void poly_taylor_transform_batch(std::span<double> leaves_tree,
                                 std::pair<double, double> coord_next,
                                 std::pair<double, double> coord_cur,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 bool use_conservative_tile);

void poly_taylor_report_batch(std::span<double> leaves_tree,
                              std::pair<double, double> coord_report,
                              SizeType n_leaves,
                              SizeType n_params);

std::vector<double>
poly_taylor_branch_generic(std::span<const double> leaf,
                           std::pair<double, double> coord_cur,
                           SizeType nbins,
                           double eta,
                           std::span<const ParamLimit> param_limits,
                           SizeType n_params);

// Generate an approximate branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_poly_taylor_approx(std::span<const SizeType> param_grid_count_init,
                               std::span<const double> dparams_init,
                               std::span<const ParamLimit> param_limits,
                               double tseg_ffa,
                               SizeType nsegments,
                               SizeType nbins,
                               double eta,
                               SizeType ref_seg,
                               IndexType isuggest         = 0,
                               bool use_conservative_tile = false);

// Generate an exact branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_poly_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        std::span<const ParamLimit> param_limits,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        bool use_conservative_tile = false);

#ifdef LOKI_ENABLE_CUDA

void ffa_taylor_resolve_freq_batch_cuda(
    cuda::std::span<const uint32_t> param_arr_count,
    cuda::std::span<const uint32_t> ncoords_offsets,
    cuda::std::span<const ParamLimit> param_limits,
    coord::FFACoordFreqDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    double tseg_brute,
    SizeType nbins,
    cudaStream_t stream);

void ffa_taylor_resolve_poly_batch_cuda(
    cuda::std::span<const uint32_t> param_arr_count,
    cuda::std::span<const uint32_t> ncoords_offsets,
    cuda::std::span<const ParamLimit> param_limits,
    coord::FFACoordDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    SizeType latter,
    double tseg_brute,
    SizeType nbins,
    SizeType n_params,
    cudaStream_t stream);

void poly_taylor_seed_cuda(
    cuda::std::span<const SizeType> param_grid_count_init,
    cuda::std::span<const double> dparams_init,
    cuda::std::span<const ParamLimit> param_limits,
    cuda::std::span<double> seed_leaves,
    std::pair<double, double> coord_init,
    SizeType n_leaves,
    SizeType n_params,
    cudaStream_t stream);

SizeType
poly_taylor_branch_batch_cuda(cuda::std::span<const double> leaves_tree,
                              cuda::std::span<double> leaves_branch,
                              cuda::std::span<uint32_t> leaves_origins,
                              std::pair<double, double> coord_cur,
                              SizeType nbins,
                              double eta,
                              cuda::std::span<const ParamLimit> param_limits,
                              SizeType branch_max,
                              SizeType n_leaves,
                              SizeType n_params,
                              utils::BranchingWorkspaceCUDAView ws,
                              cudaStream_t stream);

void poly_taylor_resolve_batch_cuda(
    cuda::std::span<const double> leaves_branch,
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
    SizeType n_params,
    cudaStream_t stream);

void poly_taylor_transform_batch_cuda(cuda::std::span<double> leaves_tree,
                                      cuda::std::span<uint32_t> indices_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      SizeType n_params,
                                      bool use_conservative_tile,
                                      cudaStream_t stream);

void poly_taylor_report_batch_cuda(cuda::std::span<double> leaves_tree,
                                   std::pair<double, double> coord_report,
                                   SizeType n_leaves,
                                   SizeType n_params,
                                   cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::core
