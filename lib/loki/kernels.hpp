#pragma once

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include "loki/plans_cuda.cuh"
#endif // LOKI_ENABLE_CUDA

namespace loki::kernels {

/**
 * @brief Brute force fold a time series of data.
 *
 *
 * @param ts_e  The time series of data to fold (size: nsamps)
 * @param ts_v  The time series of data to fold (size: nsamps)
 * @param fold  The output array (size: nsegments * nfreqs * 2 * nbins)
 * @param bucket_indices  The bucket indices (size: nfreqs * segment_len)
 * @param offsets  Prefix sum of the bucket indices (size: nfreqs * nbins + 1)
 * @param nsegments  The number of segments
 * @param nfreqs  The number of frequencies
 * @param nbins  The number of bins in the output array
 * @param nthreads  The number of threads to use
 */
void brute_fold_ts(const float* __restrict__ ts_e,
                   const float* __restrict__ ts_v,
                   float* __restrict__ fold,
                   const uint32_t* __restrict__ bucket_indices,
                   const SizeType* __restrict__ offsets,
                   SizeType nsegments,
                   SizeType nfreqs,
                   SizeType segment_len,
                   SizeType nbins,
                   int nthreads) noexcept;

void brute_fold_ts_complex(const float* __restrict__ ts_e,
                           const float* __restrict__ ts_v,
                           ComplexType* __restrict__ fold,
                           const double* __restrict__ freqs,
                           SizeType nfreqs,
                           SizeType nsegments,
                           SizeType segment_len,
                           SizeType nbins,
                           double tsamp,
                           double t_ref,
                           int nthreads) noexcept;

void ffa_iter(const float* __restrict__ fold_in,
              float* __restrict__ fold_out,
              const plans::FFACoord* __restrict__ coords,
              SizeType ncoords_cur,
              SizeType ncoords_prev,
              SizeType nsegments,
              SizeType nbins,
              int nthreads) noexcept;

void ffa_iter_freq(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const plans::FFACoordFreq* __restrict__ coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   int nthreads) noexcept;

void ffa_complex_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      const plans::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nsegments,
                      SizeType nbins_f,
                      SizeType nbins,
                      int nthreads) noexcept;

void ffa_complex_iter_freq(const ComplexType* __restrict__ fold_in,
                           ComplexType* __restrict__ fold_out,
                           const plans::FFACoordFreq* __restrict__ coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           int nthreads) noexcept;

/**
 * @brief Shift ffa folds and add it to the tree folds for each batch.
 *
 * @param folds_tree  The tree folds data (size: n_leaves * 2 * nbins)
 * @param indices_tree  Indices to access the leaf folds (size: n_leaves)
 * @param folds_ffa  The precomputed ffa folds (size: n_coords * 2 * nbins)
 * @param indices_ffa  Indices to access the ffa folds (size: n_leaves)
 * @param phase_shift  Phase shifts to apply to the ffa folds (size: n_leaves)
 * @param folds_out  The output array (size: n_leaves * 2 * nbins)
 * @param temp_buffer  Pre-allocated buffer of size 2 * nbins
 * @param nbins  The number of bins in the input/output arrays (time-domain)
 * @param n_leaves  The number of valid leaves in the tree
 */
void shift_add_linear_batch(const float* __restrict__ folds_tree,
                            const SizeType* __restrict__ indices_tree,
                            const float* __restrict__ folds_ffa,
                            const SizeType* __restrict__ indices_ffa,
                            const float* __restrict__ phase_shift,
                            float* __restrict__ folds_out,
                            float* __restrict__ temp_buffer,
                            SizeType nbins,
                            SizeType n_leaves) noexcept;

/**
 * @brief Shift complex ffa folds and add it to the complex tree folds for each
 * leaf.
 *
 * @param folds_tree  The tree folds data (size: n_leaves * 2 * nbins_f)
 * @param indices_tree  Indices to access the leaf folds (size: n_leaves)
 * @param folds_ffa  The precomputed ffa folds (size: n_coords * 2 * nbins_f)
 * @param indices_ffa  Indices to access the ffa folds (size: n_coords)
 * @param phase_shift  Phase shifts to apply to the ffa folds (size: n_leaves)
 * @param folds_out  The output array (size: n_leaves * 2 * nbins_f)
 * @param nbins_f  The number of frequency bins (FFT size)
 * @param nbins  The number of time-domain bins (original fold size)
 * @param n_leaves  The number of valid leaves in the tree
 *
 * @note Uses recurrence relation for phase calculation for optimal performance.
 */
void shift_add_linear_complex_batch(const ComplexType* __restrict__ folds_tree,
                                    const SizeType* __restrict__ indices_tree,
                                    const ComplexType* __restrict__ folds_ffa,
                                    const SizeType* __restrict__ indices_ffa,
                                    const float* __restrict__ phase_shift,
                                    ComplexType* __restrict__ folds_out,
                                    SizeType nbins_f,
                                    SizeType nbins,
                                    SizeType n_leaves) noexcept;

#ifdef LOKI_ENABLE_CUDA

void brute_fold_ts_cuda(const float* __restrict__ ts_e,
                        const float* __restrict__ ts_v,
                        float* __restrict__ fold,
                        const uint32_t* __restrict__ bucket_indices,
                        const SizeType* __restrict__ offsets,
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
                                SizeType nbins,
                                double tsamp,
                                double t_ref,
                                cudaStream_t stream);

void ffa_iter_cuda(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const plans::FFACoordDPtrs* __restrict__ coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   cudaStream_t stream);

void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        const plans::FFACoordFreqDPtrs* __restrict__ coords,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        SizeType nsegments,
                        SizeType nbins,
                        cudaStream_t stream);

void ffa_complex_iter_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                           ComplexTypeCUDA* __restrict__ fold_out,
                           const plans::FFACoordDPtrs* __restrict__ coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           cudaStream_t stream);

void ffa_complex_iter_freq_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
    ComplexTypeCUDA* __restrict__ fold_out,
    const plans::FFACoordFreqDPtrs* __restrict__ coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    cudaStream_t stream);

void shift_add_linear_batch_cuda(const float* __restrict__ folds_tree,
                                 const uint32_t* __restrict__ indices_tree,
                                 const float* __restrict__ folds_ffa,
                                 const uint32_t* __restrict__ indices_ffa,
                                 const float* __restrict__ phase_shift,
                                 float* __restrict__ folds_out,
                                 SizeType nbins,
                                 SizeType n_leaves,
                                 cudaStream_t stream);

void shift_add_linear_complex_batch_cuda(
    const ComplexTypeCUDA* __restrict__ folds_tree,
    const uint32_t* __restrict__ indices_tree,
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_out,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_leaves,
    cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::kernels