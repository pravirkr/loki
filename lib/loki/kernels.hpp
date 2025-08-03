#pragma once

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"

namespace loki::kernels {

/**
 * @brief Shift two float arrays and add them together.
 *
 *
 * @param data_tail  The tail of the data to shift and add (size: 2 * nbins)
 * @param phase_shift_tail  The phase shift to apply to the tail.
 * @param data_head  The head of the data to shift and add (size: 2 * nbins)
 * @param phase_shift_head  The phase shift to apply to the head.
 * @param out  The output array (size: 2 * nbins)
 * @param nbins  The number of bins in the input/output arrays.
 */
void shift_add(const float* __restrict__ data_tail,
               float phase_shift_tail,
               const float* __restrict__ data_head,
               float phase_shift_head,
               float* __restrict__ out,
               SizeType nbins) noexcept;

/**
 * @brief Optimized version of shift_add, using a single pre-allocated buffer of
 * size 2 * nbins.
 */
void shift_add_buffer_binary(const float* __restrict__ data_tail,
                             float phase_shift_tail,
                             const float* __restrict__ data_head,
                             float phase_shift_head,
                             float* __restrict__ out,
                             float* __restrict__ temp_buffer,
                             SizeType nbins) noexcept;

/**
 * @brief Optimized version of shift_add_buffer, for frequency-only FFA.
 */
void shift_add_buffer_linear(const float* __restrict__ data_tail,
                             const float* __restrict__ data_head,
                             float phase_shift,
                             float* __restrict__ out,
                             float* __restrict__ temp_buffer,
                             SizeType nbins) noexcept;

/**
 * @brief Shift two complex arrays and add them together.
 *
 *
 * @param data_tail  The tail of the data to shift and add (size: 2 * nbins_f)
 * @param phase_shift_tail  The phase shift to apply to the tail.
 * @param data_head  The head of the data to shift and add (size: 2 * nbins_f)
 * @param phase_shift_head  The phase shift to apply to the head.
 * @param out  The output array (size: 2 * nbins_f)
 * @param nbins_f  The number of bins in the input/output arrays (FFT size)
 * @param nbins  The number of original bins in the input/output arrays
 * (time-domain)
 */
void shift_add_complex_binary(const ComplexType* __restrict__ data_tail,
                              float phase_shift_tail,
                              const ComplexType* __restrict__ data_head,
                              float phase_shift_head,
                              ComplexType* __restrict__ out,
                              SizeType nbins_f,
                              SizeType nbins) noexcept;

/**
 * @brief Optimized version of shift_add_complex using a recurrence relation for
 * the phase. Idea here is to replace the two expensive sin/cos calls with one
 * cheaper complex multiply. Processing in blocks to remove the loop-carried
 * dependency. It uses xsimd types to guarantee vectorization and operates on
 * SIMD-sized chunks of data at a time.
 *
 * @note This is the only version that vectorizes efficiently across
 * architectures. The other versions are not vectorized.
 */
void shift_add_complex_recurrence_binary(
    const ComplexType* __restrict__ data_tail,
    float phase_shift_tail,
    const ComplexType* __restrict__ data_head,
    float phase_shift_head,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins) noexcept;

/**
 * @brief Shift a complex array and add it to another complex array.
 *
 *
 * @param data_tail  The tail of the data to add (size: 2 * nbins_f)
 * @param data_head  The head of the data toshift to add (size: 2 * nbins_f)
 * @param phase_shift  The phase shift to apply to the head.
 * @param out  The output array (size: 2 * nbins_f)
 * @param nbins_f  The number of bins in the input/output arrays (FFT size)
 * @param nbins  The number of original bins in the input/output arrays
 * (time-domain)
 */
void shift_add_complex_recurrence_linear(
    const ComplexType* __restrict__ data_tail,
    const ComplexType* __restrict__ data_head,
    float phase_shift,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins) noexcept;

/**
 * @brief Brute force fold a segment of data.
 *
 *
 * @param ts_e_seg  The segment of data to fold (size: segment_len)
 * @param ts_v_seg  The segment of data to fold (size: segment_len)
 * @param fold_seg  The output array (size: nfreqs * 2 * nbins)
 * @param bucket_indices  The bucket indices (size: nfreqs * segment_len)
 * @param offsets  Prefix sum of the bucket indices (size: nfreqs * nbins + 1)
 * @param nfreqs  The number of frequencies
 * @param nbins  The number of bins in the output array
 */
void brute_fold_segment(const float* __restrict__ ts_e_seg,
                        const float* __restrict__ ts_v_seg,
                        float* __restrict__ fold_seg,
                        const uint32_t* __restrict__ bucket_indices,
                        const SizeType* __restrict__ offsets,
                        SizeType nfreqs,
                        SizeType nbins) noexcept;

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

void brute_fold_segment_complex(const float* __restrict__ ts_e_seg,
                                const float* __restrict__ ts_v_seg,
                                ComplexType* __restrict__ fold_seg,
                                const double* __restrict__ freqs,
                                SizeType nfreqs,
                                SizeType nbins_f,
                                SizeType segment_len,
                                AlignedFloatVec& proper_time,
                                AlignedFloatVec& delta_phasors_r,
                                AlignedFloatVec& delta_phasors_i,
                                AlignedFloatVec& current_phasors_r,
                                AlignedFloatVec& current_phasors_i) noexcept;

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

void ffa_iter_segment(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      const plans::FFACoord* __restrict__ coords_cur,
                      SizeType nsegments,
                      SizeType nbins,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      int nthreads);

void ffa_iter_standard(const float* __restrict__ fold_in,
                       float* __restrict__ fold_out,
                       const plans::FFACoord* __restrict__ coords_cur,
                       SizeType nsegments,
                       SizeType nbins,
                       SizeType ncoords_cur,
                       SizeType ncoords_prev,
                       int nthreads);

void ffa_iter_segment_freq(const float* __restrict__ fold_in,
                           float* __restrict__ fold_out,
                           const plans::FFACoordFreq* __restrict__ coords_cur,
                           SizeType nsegments,
                           SizeType nbins,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           int nthreads);

void ffa_iter_standard_freq(const float* __restrict__ fold_in,
                            float* __restrict__ fold_out,
                            const plans::FFACoordFreq* __restrict__ coords_cur,
                            SizeType nsegments,
                            SizeType nbins,
                            SizeType ncoords_cur,
                            SizeType ncoords_prev,
                            int nthreads);

void ffa_complex_iter_segment(const ComplexType* __restrict__ fold_in,
                              ComplexType* __restrict__ fold_out,
                              const plans::FFACoord* __restrict__ coords_cur,
                              SizeType nsegments,
                              SizeType nbins_f,
                              SizeType nbins,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev,
                              int nthreads);

void ffa_complex_iter_standard(const ComplexType* __restrict__ fold_in,
                               ComplexType* __restrict__ fold_out,
                               const plans::FFACoord* __restrict__ coords_cur,
                               SizeType nsegments,
                               SizeType nbins_f,
                               SizeType nbins,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev,
                               int nthreads);

void ffa_complex_iter_segment_freq(
    const ComplexType* __restrict__ fold_in,
    ComplexType* __restrict__ fold_out,
    const plans::FFACoordFreq* __restrict__ coords_cur,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    int nthreads);

void ffa_complex_iter_standard_freq(
    const ComplexType* __restrict__ fold_in,
    ComplexType* __restrict__ fold_out,
    const plans::FFACoordFreq* __restrict__ coords_cur,
    SizeType nsegments,
    SizeType nbins_f,
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

} // namespace loki::kernels