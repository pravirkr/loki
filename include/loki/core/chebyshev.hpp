#pragma once

#include <span>

#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::core {

void poly_taylor_to_cheby_batch(std::span<double> seed_leaves,
                                std::pair<double, double> coord_init,
                                SizeType n_leaves,
                                SizeType n_params);

SizeType poly_chebyshev_branch_batch(std::span<double> leaves_tree,
                                     std::span<double> leaves_branch,
                                     std::span<SizeType> leaves_origins,
                                     std::pair<double, double> coord_cur,
                                     std::pair<double, double> coord_prev,
                                     SizeType nbins,
                                     double eta,
                                     std::span<const ParamLimit> param_limits,
                                     SizeType branch_max,
                                     SizeType n_leaves,
                                     SizeType n_params,
                                     memory::BranchingWorkspace& branch_ws);

void poly_chebyshev_resolve_batch(std::span<const double> leaves_branch,
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

void poly_chebyshev_transform_batch(std::span<double> leaves_tree,
                                    std::span<SizeType> indices_tree,
                                    std::pair<double, double> coord_next,
                                    std::pair<double, double> coord_cur,
                                    SizeType n_leaves,
                                    SizeType n_params);

void poly_cheby_to_taylor_batch(std::span<double> leaves_tree,
                                std::pair<double, double> coord_report,
                                SizeType n_leaves,
                                SizeType n_params);

// Generate an approximate branching pattern for the pruning Chebyshev search.
std::vector<double> generate_bp_poly_chebyshev_approx(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    std::span<const ParamLimit> param_limits,
    double tseg_ffa,
    SizeType nsegments,
    SizeType nbins,
    double eta,
    SizeType ref_seg,
    IndexType isuggest  = 0,
    SizeType branch_max = 256);

// Generate an exact branching pattern for the pruning Chebyshev search.
std::vector<double>
generate_bp_poly_chebyshev(std::span<const std::vector<double>> param_arr,
                           std::span<const double> dparams,
                           std::span<const ParamLimit> param_limits,
                           double tseg_ffa,
                           SizeType nsegments,
                           SizeType nbins,
                           double eta,
                           SizeType ref_seg);

#ifdef LOKI_ENABLE_CUDA

void poly_taylor_to_cheby_batch_cuda(cuda::std::span<double> seed_leaves,
                                     std::pair<double, double> coord_init,
                                     SizeType n_leaves,
                                     SizeType n_params,
                                     cudaStream_t stream);

SizeType
poly_chebyshev_branch_batch_cuda(cuda::std::span<double> leaves_tree,
                                 cuda::std::span<double> leaves_branch,
                                 cuda::std::span<uint32_t> leaves_origins,
                                 cuda::std::span<uint8_t> validation_mask,
                                 std::pair<double, double> coord_cur,
                                 std::pair<double, double> coord_prev,
                                 SizeType nbins,
                                 double eta,
                                 cuda::std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 memory::BranchingWorkspaceCUDAView branch_ws,
                                 memory::CUBScratchArena& scratch_ws,
                                 cudaStream_t stream);

void poly_chebyshev_resolve_batch_cuda(
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
    SizeType n_params,
    cudaStream_t stream);

void poly_chebyshev_transform_batch_cuda(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    SizeType n_params,
    cudaStream_t stream);

void poly_cheby_to_taylor_batch_cuda(cuda::std::span<double> leaves_tree,
                                     std::pair<double, double> coord_report,
                                     SizeType n_leaves,
                                     SizeType n_params,
                                     cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::core