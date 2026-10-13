#pragma once

/**
 * @file ffa_legacy_kernels_cuda.cuh
 * @brief Legacy FFA merge launchers (ffa_legacy_kernels_cuda.cu): not used
 * by the library, kept for reference and comparison. CUDA only.
 */

#include <cuda_runtime.h>

#include "loki/common/types.hpp"

#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

/// Legacy complex merge level (__sincosf phases); superseded by
/// ffa_complex_iter_sweep_cuda.
void legacy_ffa_complex_iter_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                  ComplexTypeCUDA* __restrict__ fold_out,
                                  coord::FFACoordDPtrs coords,
                                  SizeType ncoords_cur,
                                  SizeType ncoords_prev,
                                  SizeType nsegments,
                                  SizeType nbins_f,
                                  SizeType nbins,
                                  cudaStream_t stream);

/// Legacy frequency-only complex merge level; superseded by
/// ffa_complex_iter_freq_sweep_cuda and the fused groups.
void legacy_ffa_complex_iter_freq_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
    ComplexTypeCUDA* __restrict__ fold_out,
    coord::FFACoordFreqDPtrs coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    cudaStream_t stream);

} // namespace loki::core
