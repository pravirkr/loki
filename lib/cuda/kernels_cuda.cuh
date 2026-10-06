#pragma once

/**
 * @file kernels_cuda.cuh
 * @brief Device implementations declared alongside core/kernels.hpp (CUDA
 * only).
 */

#include <cuda_runtime.h>

#include "core/kernels.hpp"
#include "cuda/coord_cuda.cuh"
#include "cuda/types_cuda.cuh"

namespace loki::core {

void brute_fold_ts_cuda(const float* __restrict__ ts_e,
                        const float* __restrict__ ts_v,
                        float* __restrict__ fold,
                        const uint32_t* __restrict__ phase_map,
                        SizeType nsegments,
                        SizeType nfreqs,
                        SizeType segment_len,
                        SizeType nbins,
                        cudaStream_t stream);

void brute_fold_ts_complex_cuda(const float* __restrict__ ts_e,
                                const float* __restrict__ ts_v,
                                ComplexTypeCUDA* __restrict__ fold,
                                const double* __restrict__ freqs,
                                SizeType nfreqs,
                                SizeType nsegments,
                                SizeType segment_len,
                                SizeType nbins_f,
                                double tsamp,
                                double t_ref,
                                cudaStream_t stream);

void ffa_iter_cuda(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   coord::FFACoordDPtrs coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   cudaStream_t stream);

void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        coord::FFACoordFreqDPtrs coords,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        SizeType nsegments,
                        SizeType nbins,
                        cudaStream_t stream);

void ffa_complex_iter_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                           ComplexTypeCUDA* __restrict__ fold_out,
                           coord::FFACoordDPtrs coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           cudaStream_t stream);

void ffa_complex_iter_freq_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                ComplexTypeCUDA* __restrict__ fold_out,
                                coord::FFACoordFreqDPtrs coords,
                                SizeType ncoords_cur,
                                SizeType ncoords_prev,
                                SizeType nsegments,
                                SizeType nbins_f,
                                SizeType nbins,
                                cudaStream_t stream);

void shift_add_linear_batch_cuda(const float* __restrict__ folds_tree,
                                 const uint32_t* __restrict__ indices_tree,
                                 const uint8_t* __restrict__ validation_mask,
                                 const float* __restrict__ folds_ffa,
                                 const uint32_t* __restrict__ indices_ffa,
                                 const float* __restrict__ phase_shift,
                                 float* __restrict__ folds_out,
                                 SizeType nbins,
                                 SizeType n_leaves,
                                 SizeType physical_start_idx,
                                 SizeType capacity,
                                 cudaStream_t stream);

void shift_add_linear_complex_batch_cuda(
    const ComplexTypeCUDA* __restrict__ folds_tree,
    const uint32_t* __restrict__ indices_tree,
    const uint8_t* __restrict__ validation_mask,
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_out,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_leaves,
    SizeType physical_start_idx,
    SizeType capacity,
    cudaStream_t stream);

void shift_add_ascend_linear_batch_cuda(
    const float* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    float* __restrict__ folds_tree,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments,
    cudaStream_t stream);

void shift_add_ascend_linear_complex_batch_cuda(
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_tree,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments,
    cudaStream_t stream);

} // namespace loki::core
