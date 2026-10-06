#pragma once

/**
 * @file taylor_cuda.cuh
 * @brief Device implementations declared alongside core/taylor.hpp (CUDA only).
 */

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "lib/core/taylor.hpp"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::core {

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
                              cuda::std::span<uint8_t> validation_mask,
                              std::pair<double, double> coord_cur,
                              SizeType nbins,
                              double eta,
                              SizeType branch_max,
                              SizeType n_leaves,
                              SizeType n_params,
                              memory::BranchingWorkspaceCUDAView branch_ws,
                              memory::CUBScratchArena& scratch_ws,
                              cudaStream_t stream);

void poly_taylor_resolve_batch_cuda(
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

void poly_taylor_ascend_resolve_batch_cuda(
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

void poly_taylor_transform_batch_cuda(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
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

} // namespace loki::core
