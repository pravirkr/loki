#pragma once

/**
 * @file kernels_cuda.cuh
 * @brief Device implementations declared alongside core/kernels.hpp (CUDA
 * only).
 */

#include <cstdint>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include "lib/core/kernels.hpp"
#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

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

// Fuse `k` frequency-only merge levels (k is 3, 4 or 5) for one
// power-of-two tile of input segments. Intermediate levels stay in shared
// memory. Valid only for levels that ffa_freq_fuse_check_levels_cuda
// reports clean and where ffa_freq_fuse_fits_smem(k, nbins) holds; use
// plan_ffa_freq_fuse_groups to choose k.
void ffa_iter_freq_fused_cuda(const float* __restrict__ fold_in,
                              float* __restrict__ fold_out,
                              const uint32_t* const* level_idx,
                              const float* const* level_shift,
                              const uint32_t* level_ncoords,
                              uint32_t ncoords_in,
                              uint32_t ncoords_out,
                              uint32_t nsegments_in,
                              uint32_t nbins,
                              int k,
                              cudaStream_t stream);

// Whether the k-level fused tile (dynamic plus static shared memory) fits
// the per-block shared memory of the current device.
[[nodiscard]] bool ffa_freq_fuse_fits_smem(int k, SizeType nbins);

// Per level, the number of coordinates whose parent index breaks the fusion
// contract (unsorted, a parent with more than 2 children, or out of range
// of the previous level). Level 0 is always 0. `bad_d` is device scratch of
// 3 * n_levels uint32. Synchronizes `stream`.
[[nodiscard]] std::vector<uint32_t>
ffa_freq_fuse_check_levels_cuda(const uint32_t* idx,
                                std::span<const uint32_t> ncoords_offsets,
                                std::span<const SizeType> ncoords,
                                uint32_t* bad_d,
                                cudaStream_t stream);

// The fused group sizes for levels 1..n_levels-1, in order: 3, 4 or 5 for a
// fused group, 1 for a single unfused level. They sum to n_levels - 1.
// `k_fits_smem[k]` says whether a k-level group fits shared memory.
[[nodiscard]] std::vector<int>
plan_ffa_freq_fuse_groups(std::span<const SizeType> ncoords,
                          std::span<const SizeType> nsegments,
                          std::span<const uint32_t> level_bad,
                          std::span<const bool> k_fits_smem);

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
