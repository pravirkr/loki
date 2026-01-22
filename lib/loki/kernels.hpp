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
              const plans::FFACoord* __restrict__ coords_cur,
              SizeType nsegments,
              SizeType nbins,
              SizeType ncoords_cur,
              SizeType ncoords_prev,
              int nthreads);

void ffa_iter_freq(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const plans::FFACoordFreq* __restrict__ coords_cur,
                   SizeType nsegments,
                   SizeType nbins,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   int nthreads);

void ffa_complex_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      const plans::FFACoord* __restrict__ coords_cur,
                      SizeType nsegments,
                      SizeType nbins,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      int nthreads);

void ffa_complex_iter_freq(const ComplexType* __restrict__ fold_in,
                           ComplexType* __restrict__ fold_out,
                           const plans::FFACoordFreq* __restrict__ coords_cur,
                           SizeType nsegments,
                           SizeType nbins,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           int nthreads);

/**
 * @brief Shift ffa data and add it to the folds data for each batch.
 *
 * @param data_folds  The folds data (size: nbatch * 2 * nbins)
 * @param idx_folds  The batch indices of the folds data (size: nbatch)
 * @param data_ffa  The ffa data (size: nbatch * 2 * nbins)
 * @param idx_ffa  The batch indices of the ffa data (size: nbatch)
 * @param shift_batch  The shifts to apply to the ffa data (size: nbatch)
 * @param out  The output array (size: nbatch * 2 * nbins)
 * @param temp_buffer  A pre-allocated buffer of size 2 * nbins
 * @param nbins  The number of bins in the input/output arrays (time-domain)
 * @param nbatch  The batch size.
 */
void shift_add_buffer_batch(const float* __restrict__ data_folds,
                            const SizeType* __restrict__ idx_folds,
                            const float* __restrict__ data_ffa,
                            const SizeType* __restrict__ idx_ffa,
                            const float* __restrict__ shift_batch,
                            float* __restrict__ out,
                            float* __restrict__ temp_buffer,
                            SizeType nbins,
                            SizeType nbatch) noexcept;

/**
 * @brief Shift complex ffa data and add it to the complex folds data for each
 * batch.
 *
 * @param data_folds  The folds data (size: nbatch * 2 * nbins_f)
 * @param idx_folds  The batch indices of the folds data (size: nbatch)
 * @param data_ffa  The ffa data (size: nbatch * 2 * nbins_f)
 * @param idx_ffa  The batch indices of the ffa data (size: nbatch)
 * @param shift_batch  The shifts to apply to the ffa data (size: nbatch)
 * @param out  The output array (size: nbatch * 2 * nbins_f)
 * @param nbins_f  The number of frequency bins (FFT size)
 * @param nbins  The number of time-domain bins (original fold size)
 * @param nbatch  The batch size.
 *
 * @note Uses recurrence relation for phase calculation for optimal performance.
 */
void shift_add_complex_recurrence_batch(
    const ComplexType* __restrict__ data_folds,
    const SizeType* __restrict__ idx_folds,
    const ComplexType* __restrict__ data_ffa,
    const SizeType* __restrict__ idx_ffa,
    const float* __restrict__ shift_batch,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins,
    SizeType nbatch) noexcept;

#ifdef LOKI_ENABLE_CUDA

void ffa_iter_cuda(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const plans::FFACoordDPtrs* __restrict__ coords_cur,
                   SizeType nsegments,
                   SizeType nbins,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   cudaStream_t stream);

void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        const plans::FFACoordFreqDPtrs* __restrict__ coords_cur,
                        SizeType nsegments,
                        SizeType nbins,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::kernels