#pragma once

/**
 * @file circular_cuda.cuh
 * @brief Device implementations declared alongside core/circular.hpp (CUDA
 * only).
 */

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "lib/core/circular.hpp"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::core {

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
                              double propagator_significance,
                              memory::BranchingWorkspaceCUDAView branch_ws,
                              memory::CUBScratchArena& scratch_ws,
                              cudaStream_t stream);

SizeType
circ_taylor_validate_batch_cuda(cuda::std::span<const double> leaves_branch,
                                cuda::std::span<uint8_t> validation_mask,
                                SizeType n_leaves,
                                double p_orb_min,
                                double x_mass_const,
                                double validation_significance,
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
    double propagator_significance,
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
    double propagator_significance,
    cudaStream_t stream);

void circ_taylor_transform_batch_cuda(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    bool use_conservative_tile,
    double propagator_significance,
    cudaStream_t stream);

} // namespace loki::core
