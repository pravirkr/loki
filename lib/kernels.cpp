#include "loki/kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include <omp.h>

namespace loki::kernels {

void shift_add(const float* __restrict__ data_tail,
               float phase_shift_tail,
               const float* __restrict__ data_head,
               float phase_shift_head,
               float* __restrict__ out,
               SizeType nbins) noexcept {

    const auto shift_tail =
        static_cast<SizeType>(std::round(phase_shift_tail)) % nbins;
    const auto shift_head =
        static_cast<SizeType>(std::round(phase_shift_head)) % nbins;

    const float* __restrict__ data_tail_e = data_tail;
    const float* __restrict__ data_tail_v = data_tail + nbins;
    const float* __restrict__ data_head_e = data_head;
    const float* __restrict__ data_head_v = data_head + nbins;
    float* __restrict__ out_e             = out;
    float* __restrict__ out_v             = out + nbins;

    for (SizeType j = 0; j < nbins; ++j) {
        const auto idx_tail =
            (j < shift_tail) ? (j + nbins - shift_tail) : (j - shift_tail);
        const auto idx_head =
            (j < shift_head) ? (j + nbins - shift_head) : (j - shift_head);
        out_e[j] = data_tail_e[idx_tail] + data_head_e[idx_head];
        out_v[j] = data_tail_v[idx_tail] + data_head_v[idx_head];
    }
}

void shift_add_buffer(const float* __restrict__ data_tail,
                      float phase_shift_tail,
                      const float* __restrict__ data_head,
                      float phase_shift_head,
                      float* __restrict__ out,
                      float* __restrict__ temp_buffer,
                      SizeType nbins) noexcept {

    const auto shift_tail_raw =
        static_cast<SizeType>(std::round(phase_shift_tail));
    const SizeType s_tail = (shift_tail_raw % nbins + nbins) % nbins;
    const auto shift_head_raw =
        static_cast<SizeType>(std::round(phase_shift_head));
    const SizeType s_head     = (shift_head_raw % nbins + nbins) % nbins;
    const SizeType total_size = 2 * nbins;

    // Rotate data_tail directly into the final output buffer for both channels
    std::copy_n(data_tail, nbins - s_tail, out + s_tail);
    std::copy_n(data_tail + nbins - s_tail, s_tail, out);
    std::copy_n(data_tail + nbins, nbins - s_tail, out + nbins + s_tail);
    std::copy_n(data_tail + nbins + nbins - s_tail, s_tail, out + nbins);

    // Rotate data_head into the temporary buffer for both channels
    std::copy_n(data_head, nbins - s_head, temp_buffer + s_head);
    std::copy_n(data_head + nbins - s_head, s_head, temp_buffer);
    std::copy_n(data_head + nbins, nbins - s_head,
                temp_buffer + nbins + s_head);
    std::copy_n(data_head + nbins + nbins - s_head, s_head,
                temp_buffer + nbins);

    // --- Perform the final addition in a single loop ---
    for (SizeType j = 0; j < total_size; ++j) {
        out[j] += temp_buffer[j];
    }
}

void shift_add_complex_buffer(const ComplexType* __restrict__ data_tail,
                              float phase_shift_tail,
                              const ComplexType* __restrict__ data_head,
                              float phase_shift_head,
                              ComplexType* __restrict__ out,
                              ComplexType* __restrict__ temp_buffer,
                              SizeType nbins_f,
                              SizeType nbins) noexcept {

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;
    ComplexType* __restrict__ phases_tail_ptr   = temp_buffer;
    ComplexType* __restrict__ phases_head_ptr   = temp_buffer + nbins_f;

    // --- Pre-computation Step ---
    const auto phase_factor_tail =
        -2.0 * std::numbers::pi * phase_shift_tail / static_cast<float>(nbins);
    const auto phase_factor_head =
        -2.0 * std::numbers::pi * phase_shift_head / static_cast<float>(nbins);

    // Fill the phase arrays (int is necessary for the compiler to vectorize)
    // Fast complex exponential: exp(i * theta) = cos(theta) + i *sin(theta)
    for (int k = 0; k < static_cast<int>(nbins_f); ++k) {
        const auto k_phase_tail = static_cast<float>(k) * phase_factor_tail;
        const auto k_phase_head = static_cast<float>(k) * phase_factor_head;
        phases_tail_ptr[k]      = {static_cast<float>(std::cos(k_phase_tail)),
                                   static_cast<float>(std::sin(k_phase_tail))};
        phases_head_ptr[k]      = {static_cast<float>(std::cos(k_phase_head)),
                                   static_cast<float>(std::sin(k_phase_head))};
    }

    // There are no loop-carried dependencies. Memory access is linear.
    // (int is necessary for the compiler to vectorize)
    for (int k = 0; k < static_cast<int>(nbins_f); ++k) {
        const ComplexType phase_tail = phases_tail_ptr[k];
        const ComplexType phase_head = phases_head_ptr[k];
        out_e[k] =
            (data_tail_e[k] * phase_tail) + (data_head_e[k] * phase_head);
        out_v[k] =
            (data_tail_v[k] * phase_tail) + (data_head_v[k] * phase_head);
    }
}

void shift_add_complex_recurrence(const ComplexType* __restrict__ data_tail,
                                  float phase_shift_tail,
                                  const ComplexType* __restrict__ data_head,
                                  float phase_shift_head,
                                  ComplexType* __restrict__ out,
                                  SizeType nbins_f,
                                  SizeType nbins) noexcept {
    // Calculate the constant phase step per iteration
    const auto phase_step_tail_angle =
        -2.0 * std::numbers::pi * phase_shift_tail / static_cast<float>(nbins);
    const auto phase_step_head_angle =
        -2.0 * std::numbers::pi * phase_shift_head / static_cast<float>(nbins);

    // This is the complex number we will multiply by in each iteration
    const ComplexType delta_phase_tail = {
        static_cast<float>(std::cos(phase_step_tail_angle)),
        static_cast<float>(std::sin(phase_step_tail_angle))};
    const ComplexType delta_phase_head = {
        static_cast<float>(std::cos(phase_step_head_angle)),
        static_cast<float>(std::sin(phase_step_head_angle))};

    // Phase steps within a SIMD block: [d^0, d^1, d^2, d^3]
    std::array<ComplexType, kUnrollFactor> delta_vec_tail;
    std::array<ComplexType, kUnrollFactor> delta_vec_head;
    delta_vec_tail[0] = {1.0F, 0.0F};
    delta_vec_head[0] = {1.0F, 0.0F};
    for (SizeType i = 1; i < kUnrollFactor; ++i) {
        delta_vec_tail[i] = delta_vec_tail[i - 1] * delta_phase_tail;
        delta_vec_head[i] = delta_vec_head[i - 1] * delta_phase_head;
    }

    // Phase step between SIMD blocks: d^SIMD_WIDTH
    const ComplexType delta_block_tail =
        delta_vec_tail.back() * delta_phase_tail;
    const ComplexType delta_block_head =
        delta_vec_head.back() * delta_phase_head;

    // Initial phase for k=0 is exp(i*0) = 1 + 0i
    ComplexType current_block_start_phase_tail = {1.0F, 0.0F};
    ComplexType current_block_start_phase_head = {1.0F, 0.0F};

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    const SizeType main_loop = nbins_f - (nbins_f % kUnrollFactor);

    // Vectorized main part
    for (SizeType k = 0; k < main_loop; k += kUnrollFactor) {
        UNROLL_VECTORIZE
        for (SizeType j = 0; j < kUnrollFactor; ++j) {
            const ComplexType phase_tail =
                current_block_start_phase_tail * delta_vec_tail[j];
            const ComplexType phase_head =
                current_block_start_phase_head * delta_vec_head[j];
            out_e[k + j] = (data_tail_e[k + j] * phase_tail) +
                           (data_head_e[k + j] * phase_head);
            out_v[k + j] = (data_tail_v[k + j] * phase_tail) +
                           (data_head_v[k + j] * phase_head);
        }
        current_block_start_phase_tail *= delta_block_tail;
        current_block_start_phase_head *= delta_block_head;
    }

    // Scalar remainder part for nbins_f not divisible by kUnrollFactor
    if (main_loop < nbins_f) {
        ComplexType current_phase_tail = current_block_start_phase_tail;
        ComplexType current_phase_head = current_block_start_phase_head;
        for (SizeType k = main_loop; k < nbins_f; ++k) {
            out_e[k] = (data_tail_e[k] * current_phase_tail) +
                       (data_head_e[k] * current_phase_head);
            out_v[k] = (data_tail_v[k] * current_phase_tail) +
                       (data_head_v[k] * current_phase_head);
            current_phase_tail *= delta_phase_tail;
            current_phase_head *= delta_phase_head;
        }
    }
}

void shift_add_buffer_batch(const float* __restrict__ data_folds,
                            const SizeType* __restrict__ idx_folds,
                            const float* __restrict__ data_ffa,
                            const SizeType* __restrict__ idx_ffa,
                            const SizeType* __restrict__ shift_batch,
                            float* __restrict__ out,
                            float* __restrict__ temp_buffer,
                            SizeType nbins,
                            SizeType nbatch) noexcept {
    const auto total_size = 2 * nbins;
    for (SizeType irow = 0; irow < nbatch; ++irow) {
        const SizeType shift_raw = shift_batch[irow];
        const SizeType shift     = (shift_raw % nbins + nbins) % nbins;
        const SizeType ffa_idx   = idx_ffa[irow];
        const SizeType fold_idx  = idx_folds[irow];

        // Get restrict pointers for current batch item
        const float* __restrict__ segment_ev =
            data_ffa + (ffa_idx * total_size);
        const float* __restrict__ fold_ev =
            data_folds + (fold_idx * total_size);
        float* __restrict__ out_ev = out + (irow * total_size);

        // Handle zero shift case
        if (shift == 0) {
            for (SizeType j = 0; j < total_size; ++j) {
                out_ev[j] = fold_ev[j] + segment_ev[j];
            }
        } else {
            // Optimized circular shift: rotate segment into temp buffer
            const SizeType copy1_size = nbins - shift;
            const SizeType copy2_size = shift;

            const float* __restrict__ segment_e = segment_ev;
            const float* __restrict__ segment_v = segment_ev + nbins;

            // E channel: RIGHT shift by 'shift' positions
            // Copy last (nbins - shift) elements to beginning
            std::copy_n(segment_e + copy1_size, copy2_size, temp_buffer);
            // Copy first shift elements to end
            std::copy_n(segment_e, copy1_size, temp_buffer + copy2_size);

            // V channel: RIGHT shift by 'shift' positions
            std::copy_n(segment_v + copy1_size, copy2_size,
                        temp_buffer + nbins);
            std::copy_n(segment_v, copy1_size,
                        temp_buffer + nbins + copy2_size);
            for (SizeType j = 0; j < total_size; ++j) {
                out_ev[j] = fold_ev[j] + temp_buffer[j];
            }
        }
    }
}

void shift_add_complex_recurrence_batch(
    const ComplexType* __restrict__ data_folds,
    const SizeType* __restrict__ idx_folds,
    const ComplexType* __restrict__ data_ffa,
    const SizeType* __restrict__ idx_ffa,
    const float* __restrict__ shift_batch,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins,
    SizeType nbatch) noexcept {

    for (SizeType irow = 0; irow < nbatch; ++irow) {
        const auto shift    = shift_batch[irow];
        const auto ffa_idx  = idx_ffa[irow];
        const auto fold_idx = idx_folds[irow];

        // Calculate phase step per frequency bin
        const auto phase_step_angle =
            -2.0 * std::numbers::pi * shift / static_cast<float>(nbins);

        // Complex phase step for recurrence relation
        const ComplexType delta_phase = {
            static_cast<float>(std::cos(phase_step_angle)),
            static_cast<float>(std::sin(phase_step_angle))};

        // Phase steps within a SIMD block: [d^0, d^1, d^2, d^3]
        std::array<ComplexType, kUnrollFactor> delta_vec;
        delta_vec[0] = {1.0F, 0.0F};
        for (SizeType i = 1; i < kUnrollFactor; ++i) {
            delta_vec[i] = delta_vec[i - 1] * delta_phase;
        }

        // Phase step between SIMD blocks
        const ComplexType delta_block = delta_vec.back() * delta_phase;

        // Get restrict pointers for current batch item
        const ComplexType* __restrict__ segment_e =
            data_ffa + ((ffa_idx * 2 + 0) * nbins_f);
        const ComplexType* __restrict__ segment_v =
            data_ffa + ((ffa_idx * 2 + 1) * nbins_f);
        const ComplexType* __restrict__ fold_e =
            data_folds + ((fold_idx * 2 + 0) * nbins_f);
        const ComplexType* __restrict__ fold_v =
            data_folds + ((fold_idx * 2 + 1) * nbins_f);
        ComplexType* __restrict__ out_e = out + ((irow * 2 + 0) * nbins_f);
        ComplexType* __restrict__ out_v = out + ((irow * 2 + 1) * nbins_f);

        // Initial phase for k=0 is exp(i*0) = 1 + 0i
        ComplexType current_block_start_phase = {1.0F, 0.0F};

        const SizeType main_loop = nbins_f - (nbins_f % kUnrollFactor);

        // Vectorized main part
        for (SizeType k = 0; k < main_loop; k += kUnrollFactor) {
            UNROLL_VECTORIZE
            for (SizeType j = 0; j < kUnrollFactor; ++j) {
                const ComplexType phase =
                    current_block_start_phase * delta_vec[j];
                out_e[k + j] = fold_e[k + j] + (segment_e[k + j] * phase);
                out_v[k + j] = fold_v[k + j] + (segment_v[k + j] * phase);
            }
            current_block_start_phase *= delta_block;
        }

        // Scalar remainder part for nbins_f not divisible by kUnrollFactor
        if (main_loop < nbins_f) {
            ComplexType current_phase = current_block_start_phase;
            for (SizeType k = main_loop; k < nbins_f; ++k) {
                out_e[k] = fold_e[k] + (segment_e[k] * current_phase);
                out_v[k] = fold_v[k] + (segment_v[k] * current_phase);
                current_phase *= delta_phase;
            }
        }
    }
}

void ffa_iter_segment(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      const plans::FFACoord* __restrict__ coords_cur,
                      SizeType nsegments,
                      SizeType nbins,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      int nthreads) {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, ncoords_cur,       \
               ncoords_prev, fold_stride, seg_prev_stride, seg_out_stride)
    {
        // Each thread allocates its own buffer once
        std::vector<float> temp_buffer(2 * nbins);
        auto* __restrict__ temp_buffer_ptr = temp_buffer.data();

#pragma omp for
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            // Process coordinates in blocks within each segment
            for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
                 icoord_block += kBlockSize) {
                const auto block_end =
                    std::min(icoord_block + kBlockSize, ncoords_cur);
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto* __restrict__ coord_cur = &coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_tail) *
                         fold_stride);
                    const auto head_offset =
                        ((iseg * 2 + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_head) *
                         fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_buffer(fold_tail, coord_cur->shift_tail,
                                     fold_head, coord_cur->shift_head, fold_sum,
                                     temp_buffer_ptr, nbins);
                }
            }
        }
    }
}

void ffa_iter_standard(const float* __restrict__ fold_in,
                       float* __restrict__ fold_out,
                       const plans::FFACoord* __restrict__ coords_cur,
                       SizeType nsegments,
                       SizeType nbins,
                       SizeType ncoords_cur,
                       SizeType ncoords_prev,
                       int nthreads) {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, ncoords_cur,       \
               ncoords_prev, fold_stride, seg_prev_stride, seg_out_stride)
    {
        std::vector<float> temp_buffer(2 * nbins);
        auto* __restrict__ temp_buffer_ptr = temp_buffer.data();

#pragma omp for
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            const auto block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto* __restrict__ coord_cur = &coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_tail) *
                         fold_stride);
                    const auto head_offset =
                        ((iseg * 2 + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_head) *
                         fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_buffer(fold_tail, coord_cur->shift_tail,
                                     fold_head, coord_cur->shift_head, fold_sum,
                                     temp_buffer_ptr, nbins);
                }
            }
        }
    }
}

void ffa_complex_iter_segment(const ComplexType* __restrict__ fold_in,
                              ComplexType* __restrict__ fold_out,
                              const plans::FFACoord* __restrict__ coords_cur,
                              SizeType nsegments,
                              SizeType nbins_f,
                              SizeType nbins,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev,
                              int nthreads) {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none) shared(           \
        fold_in, fold_out, coords_cur, nsegments, nbins, nbins_f, ncoords_cur, \
            ncoords_prev, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
        // Process coordinates in blocks within each segment
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
                const auto* __restrict__ coord_cur = &coords_cur[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_tail) * fold_stride);
                const auto head_offset =
                    ((iseg * 2 + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_head) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                kernels::shift_add_complex_recurrence(
                    fold_tail, coord_cur->shift_tail, fold_head,
                    coord_cur->shift_head, fold_sum, nbins_f, nbins);
            }
        }
    }
}

void ffa_complex_iter_standard(const ComplexType* __restrict__ fold_in,
                               ComplexType* __restrict__ fold_out,
                               const plans::FFACoord* __restrict__ coords_cur,
                               SizeType nsegments,
                               SizeType nbins_f,
                               SizeType nbins,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev,
                               int nthreads) {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none) shared(           \
        fold_in, fold_out, coords_cur, nsegments, nbins, nbins_f, ncoords_cur, \
            ncoords_prev, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
         icoord_block += kBlockSize) {
        SizeType block_end = std::min(icoord_block + kBlockSize, ncoords_cur);
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
                const auto* __restrict__ coord_cur = &coords_cur[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_tail) * fold_stride);
                const auto head_offset =
                    ((iseg * 2 + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_head) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                kernels::shift_add_complex_recurrence(
                    fold_tail, coord_cur->shift_tail, fold_head,
                    coord_cur->shift_head, fold_sum, nbins_f, nbins);
            }
        }
    }
}

} // namespace loki::kernels