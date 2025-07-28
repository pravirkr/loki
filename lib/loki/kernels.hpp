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
void shift_add_buffer(const float* __restrict__ data_tail,
                      float phase_shift_tail,
                      const float* __restrict__ data_head,
                      float phase_shift_head,
                      float* __restrict__ out,
                      float* __restrict__ temp_buffer,
                      SizeType nbins) noexcept;

/**
 * @brief Shift two complex arrays and add them together.
 *
 * @note Using pre-computed phases and tweaks to auto-vectorize efficiently with
 * GCC.
 *
 * @param data_tail  The tail of the data to shift and add (size: 2 * nbins_f)
 * @param phase_shift_tail  The phase shift to apply to the tail.
 * @param data_head  The head of the data to shift and add (size: 2 * nbins_f)
 * @param phase_shift_head  The phase shift to apply to the head.
 * @param out  The output array (size: 2 * nbins_f)
 * @param temp_buffer  A pre-allocated buffer of size 2 * nbins_f
 * @param nbins_f  The number of bins in the input/output arrays (FFT size)
 * @param nbins  The number of original bins in the input/output arrays
 * (time-domain)
 */
void shift_add_complex_buffer(const ComplexType* __restrict__ data_tail,
                              float phase_shift_tail,
                              const ComplexType* __restrict__ data_head,
                              float phase_shift_head,
                              ComplexType* __restrict__ out,
                              ComplexType* __restrict__ temp_buffer,
                              SizeType nbins_f,
                              SizeType nbins) noexcept;

/**
 * @brief Optimized version of shift_add_complex using a recurrence relation for
 * the phase. Idea here is to replace the two expensive sin/cos calls with one
 * cheaper complex multiply. Processing in blocks to remove the loop-carried
 * dependency.
 *
 * @note This is the only version that vectorizes efficiently across
 * architectures. The other versions are not vectorized.
 */
void shift_add_complex_recurrence(const ComplexType* __restrict__ data_tail,
                                  float phase_shift_tail,
                                  const ComplexType* __restrict__ data_head,
                                  float phase_shift_head,
                                  ComplexType* __restrict__ out,
                                  SizeType nbins_f,
                                  SizeType nbins) noexcept;

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
                            const SizeType* __restrict__ shift_batch,
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

} // namespace loki::kernels