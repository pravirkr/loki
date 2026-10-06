#pragma once

/**
 * @file chebyshev_cuda.cuh
 * @brief Device implementations declared alongside core/chebyshev.hpp (CUDA
 * only).
 */

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "core/chebyshev.hpp"
#include "cuda/types_cuda.cuh"
#include "cuda/workspace_cuda.cuh"

namespace loki::core {

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

void poly_chebyshev_ascend_resolve_batch_cuda(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    cuda::std::span<const ParamLimit> param_limits,
    cuda::std::span<const cuda::std::pair<double, double>> coord_segments,
    std::pair<double, double> coord_cur,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params,
    SizeType n_segments,
    cudaStream_t stream);

void poly_cheby_to_taylor_batch_cuda(cuda::std::span<double> leaves_tree,
                                     std::pair<double, double> coord_report,
                                     SizeType n_leaves,
                                     SizeType n_params,
                                     cudaStream_t stream);

} // namespace loki::core
