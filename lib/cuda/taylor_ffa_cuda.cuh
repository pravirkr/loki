#pragma once

/**
 * @file taylor_ffa_cuda.cuh
 * @brief Device implementations declared alongside core/taylor_ffa.hpp (CUDA
 * only).
 */

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "lib/core/taylor_ffa.hpp"
#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

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

} // namespace loki::core
