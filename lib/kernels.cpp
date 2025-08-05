#include "loki/kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>

#include <omp.h>
#include <xsimd/xsimd.hpp>

#include "loki/common/types.hpp"

namespace loki::kernels {

void shift_add(const float* __restrict__ data_tail,
               float phase_shift_tail,
               const float* __restrict__ data_head,
               float phase_shift_head,
               float* __restrict__ out,
               SizeType nbins) noexcept {

    const auto shift_tail =
        static_cast<SizeType>(std::nearbyint(phase_shift_tail)) % nbins;
    const auto shift_head =
        static_cast<SizeType>(std::nearbyint(phase_shift_head)) % nbins;

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

void shift_add_buffer_binary(const float* __restrict__ data_tail,
                             float phase_shift_tail,
                             const float* __restrict__ data_head,
                             float phase_shift_head,
                             float* __restrict__ out,
                             float* __restrict__ temp_buffer,
                             SizeType nbins) noexcept {

    const auto shift_tail =
        static_cast<SizeType>(std::nearbyint(phase_shift_tail)) % nbins;
    const auto shift_head =
        static_cast<SizeType>(std::nearbyint(phase_shift_head)) % nbins;
    const SizeType total_size = 2 * nbins;

    // Circular shift data_tail into out
    const auto shift_tail_size = nbins - shift_tail;
    std::memcpy(out + shift_tail, data_tail, sizeof(float) * shift_tail_size);
    std::memcpy(out, data_tail + shift_tail_size, sizeof(float) * shift_tail);
    std::memcpy(out + nbins + shift_tail, data_tail + nbins,
                sizeof(float) * shift_tail_size);
    std::memcpy(out + nbins, data_tail + nbins + shift_tail_size,
                sizeof(float) * shift_tail);

    // Circular shift data_head into temp_buffer
    const auto shift_head_size = nbins - shift_head;
    std::memcpy(temp_buffer + shift_head, data_head,
                sizeof(float) * shift_head_size);
    std::memcpy(temp_buffer, data_head + shift_head_size,
                sizeof(float) * shift_head);
    std::memcpy(temp_buffer + nbins + shift_head, data_head + nbins,
                sizeof(float) * shift_head_size);
    std::memcpy(temp_buffer + nbins, data_head + nbins + shift_head_size,
                sizeof(float) * shift_head);

    // Perform the final addition in a single loop
    for (SizeType j = 0; j < total_size; ++j) {
        out[j] += temp_buffer[j];
    }
}

void shift_add_buffer_linear(const float* __restrict__ data_tail,
                             const float* __restrict__ data_head,
                             float phase_shift,
                             float* __restrict__ out,
                             float* __restrict__ temp_buffer,
                             SizeType nbins) noexcept {
    const auto shift = static_cast<SizeType>(std::nearbyint(phase_shift)) % nbins;
    const SizeType total_size = 2 * nbins;
    // Optimized circular shift: rotate data_head into temp buffer
    // Right shift by 'shift' positions
    const auto shift_size = nbins - shift;

    // Copy last shift_size elements to beginning
    std::memcpy(temp_buffer + shift, data_head, sizeof(float) * shift_size);
    // Copy first shift elements to end
    std::memcpy(temp_buffer, data_head + shift_size, sizeof(float) * shift);
    std::memcpy(temp_buffer + nbins + shift, data_head + nbins,
                sizeof(float) * shift_size);
    std::memcpy(temp_buffer + nbins, data_head + nbins + shift_size,
                sizeof(float) * shift);

    // Perform the final addition in a single loop
    for (SizeType j = 0; j < total_size; ++j) {
        out[j] = data_tail[j] + temp_buffer[j];
    }
}

void shift_add_complex_binary(const ComplexType* __restrict__ data_tail,
                              float phase_shift_tail,
                              const ComplexType* __restrict__ data_head,
                              float phase_shift_head,
                              ComplexType* __restrict__ out,
                              SizeType nbins_f,
                              SizeType nbins) noexcept {

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    // Precompute phase factor constants
    const auto phase_factor_tail =
        -2.0 * std::numbers::pi * phase_shift_tail / static_cast<float>(nbins);
    const auto phase_factor_head =
        -2.0 * std::numbers::pi * phase_shift_head / static_cast<float>(nbins);

    // Fast complex exponential: exp(i * theta) = cos(theta) + i *sin(theta)
    // Expensive sin/cos calls in the loop
    for (SizeType k = 0; k < nbins_f; ++k) {
        const auto k_phase_tail = static_cast<float>(k) * phase_factor_tail;
        const auto k_phase_head = static_cast<float>(k) * phase_factor_head;
        const ComplexType phase_tail = {
            static_cast<float>(std::cos(k_phase_tail)),
            static_cast<float>(std::sin(k_phase_tail))};
        const ComplexType phase_head = {
            static_cast<float>(std::cos(k_phase_head)),
            static_cast<float>(std::sin(k_phase_head))};
        out_e[k] =
            (data_tail_e[k] * phase_tail) + (data_head_e[k] * phase_head);
        out_v[k] =
            (data_tail_v[k] * phase_tail) + (data_head_v[k] * phase_head);
    }
}

void shift_add_complex_recurrence_binary(
    const ComplexType* __restrict__ data_tail,
    float phase_shift_tail,
    const ComplexType* __restrict__ data_head,
    float phase_shift_head,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins) noexcept {
    using BatchType                  = xsimd::batch<ComplexType>;
    static constexpr auto kBatchSize = BatchType::size;

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
    std::array<ComplexType, kBatchSize> delta_vec_tail_std;
    std::array<ComplexType, kBatchSize> delta_vec_head_std;
    delta_vec_tail_std[0] = {1.0F, 0.0F};
    delta_vec_head_std[0] = {1.0F, 0.0F};
    for (size_t i = 1; i < kBatchSize; ++i) {
        delta_vec_tail_std[i] = delta_vec_tail_std[i - 1] * delta_phase_tail;
        delta_vec_head_std[i] = delta_vec_head_std[i - 1] * delta_phase_head;
    }

    // Load the phase steps into SIMD registers
    const auto delta_vec_tail =
        xsimd::load_unaligned(delta_vec_tail_std.data());
    const auto delta_vec_head =
        xsimd::load_unaligned(delta_vec_head_std.data());

    // Phase step between SIMD blocks: d^SIMD_WIDTH
    const ComplexType delta_block_tail =
        delta_vec_tail_std.back() * delta_phase_tail;
    const ComplexType delta_block_head =
        delta_vec_head_std.back() * delta_phase_head;

    // Initial phase for k=0 is exp(i*0) = 1 + 0i
    ComplexType current_block_start_phase_tail = {1.0F, 0.0F};
    ComplexType current_block_start_phase_head = {1.0F, 0.0F};

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    auto compute_fused_op = [&](SizeType k, const BatchType& phase_tail,
                                const BatchType& phase_head) {
        const auto tail_e_data = xsimd::load_unaligned(&data_tail_e[k]);
        const auto head_e_data = xsimd::load_unaligned(&data_head_e[k]);
        xsimd::fma(head_e_data, phase_head, tail_e_data * phase_tail)
            .store_unaligned(&out_e[k]);
        const auto tail_v_data = xsimd::load_unaligned(&data_tail_v[k]);
        const auto head_v_data = xsimd::load_unaligned(&data_head_v[k]);
        xsimd::fma(head_v_data, phase_head, tail_v_data * phase_tail)
            .store_unaligned(&out_v[k]);
    };

    // First process two batches at a time to maximize throughput
    SizeType k = 0;
    for (; k + 2 * kBatchSize <= nbins_f; k += 2 * kBatchSize) {
        const BatchType phase0_tail =
            xsimd::broadcast(current_block_start_phase_tail) * delta_vec_tail;
        const BatchType phase0_head =
            xsimd::broadcast(current_block_start_phase_head) * delta_vec_head;
        compute_fused_op(k, phase0_tail, phase0_head);
        current_block_start_phase_tail *= delta_block_tail;
        current_block_start_phase_head *= delta_block_head;
        const BatchType phase1_tail =
            xsimd::broadcast(current_block_start_phase_tail) * delta_vec_tail;
        const BatchType phase1_head =
            xsimd::broadcast(current_block_start_phase_head) * delta_vec_head;
        compute_fused_op(k + kBatchSize, phase1_tail, phase1_head);
        current_block_start_phase_tail *= delta_block_tail;
        current_block_start_phase_head *= delta_block_head;
    }

    // Process the remaining batches
    if (k + kBatchSize <= nbins_f) {
        const BatchType phase_tail =
            xsimd::broadcast(current_block_start_phase_tail) * delta_vec_tail;
        const BatchType phase_head =
            xsimd::broadcast(current_block_start_phase_head) * delta_vec_head;
        compute_fused_op(k, phase_tail, phase_head);
        k += kBatchSize;
        current_block_start_phase_tail *= delta_block_tail;
        current_block_start_phase_head *= delta_block_head;
    }

    // Scalar remainder part
    if (k < nbins_f) {
        ComplexType current_phase_tail = current_block_start_phase_tail;
        ComplexType current_phase_head = current_block_start_phase_head;
        for (; k < nbins_f; ++k) {
            out_e[k] = data_head_e[k] * current_phase_head +
                       data_tail_e[k] * current_phase_tail;
            out_v[k] = data_head_v[k] * current_phase_head +
                       data_tail_v[k] * current_phase_tail;
            current_phase_tail *= delta_phase_tail;
            current_phase_head *= delta_phase_head;
        }
    }
}

// On the fly phase computation is faster than precomputed phase steps on Intel
// CPUs.
void shift_add_complex_recurrence_linear(
    const ComplexType* __restrict__ data_tail,
    const ComplexType* __restrict__ data_head,
    float phase_shift,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins) noexcept {
    using BatchType                  = xsimd::batch<ComplexType>;
    static constexpr auto kBatchSize = BatchType::size;

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    // Calculate the constant phase step per iteration
    const auto phase_step_angle =
        -2.0 * std::numbers::pi * phase_shift / static_cast<float>(nbins);

    // This is the complex number we will multiply by in each iteration
    const ComplexType delta_phase = {
        static_cast<float>(std::cos(phase_step_angle)),
        static_cast<float>(std::sin(phase_step_angle))};

    // Phase steps within a SIMD block: [d^0, d^1, d^2, d^3]
    std::array<ComplexType, kBatchSize> delta_vec_std;
    delta_vec_std[0] = {1.0F, 0.0F};
    for (size_t i = 1; i < kBatchSize; ++i) {
        delta_vec_std[i] = delta_vec_std[i - 1] * delta_phase;
    }

    // Load the phase steps into SIMD registers
    const auto delta_vec = xsimd::load_unaligned(delta_vec_std.data());

    // Phase step between SIMD blocks: d^SIMD_WIDTH
    const ComplexType delta_block = delta_vec_std.back() * delta_phase;

    // Initial phase for k=0 is exp(i*0) = 1 + 0i
    ComplexType current_block_start_phase = {1.0F, 0.0F};

    auto compute_and_store = [&](SizeType k, const BatchType& phase) {
        const auto tail_e_data = xsimd::load_unaligned(&data_tail_e[k]);
        const auto head_e_data = xsimd::load_unaligned(&data_head_e[k]);
        (tail_e_data + head_e_data * phase).store_unaligned(&out_e[k]);
        const auto tail_v_data = xsimd::load_unaligned(&data_tail_v[k]);
        const auto head_v_data = xsimd::load_unaligned(&data_head_v[k]);
        (tail_v_data + head_v_data * phase).store_unaligned(&out_v[k]);
    };

    // First process two batches at a time to maximize throughput
    SizeType k = 0;
    for (; k + 2 * kBatchSize <= nbins_f; k += 2 * kBatchSize) {
        const BatchType phase0 =
            xsimd::broadcast(current_block_start_phase) * delta_vec;
        compute_and_store(k, phase0);
        current_block_start_phase *= delta_block;
        const BatchType phase1 =
            xsimd::broadcast(current_block_start_phase) * delta_vec;
        compute_and_store(k + kBatchSize, phase1);
        current_block_start_phase *= delta_block;
    }

    // Process the remaining batches
    if (k + kBatchSize <= nbins_f) {
        const BatchType phase =
            xsimd::broadcast(current_block_start_phase) * delta_vec;
        compute_and_store(k, phase);
        k += kBatchSize;
        current_block_start_phase *= delta_block;
    }

    // Scalar remainder part
    if (k < nbins_f) {
        ComplexType current_phase = current_block_start_phase;
        for (; k < nbins_f; ++k) {
            out_e[k] = data_tail_e[k] + data_head_e[k] * current_phase;
            out_v[k] = data_tail_v[k] + data_head_v[k] * current_phase;
            current_phase *= delta_phase;
        }
    }
}

void brute_fold_segment(const float* __restrict__ ts_e_seg,
                        const float* __restrict__ ts_v_seg,
                        float* __restrict__ fold_seg,
                        const uint32_t* __restrict__ bucket_indices,
                        const SizeType* __restrict__ offsets,
                        SizeType nfreqs,
                        SizeType nbins) noexcept {
    for (SizeType ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const auto freq_offset_out      = ifreq * 2 * nbins;
        float* __restrict__ fold_e_base = fold_seg + freq_offset_out;
        float* __restrict__ fold_v_base = fold_e_base + nbins;

        for (SizeType iphase = 0; iphase < nbins; ++iphase) {
            const auto bucket_idx    = (ifreq * nbins) + iphase;
            const auto buck_start    = offsets[bucket_idx];
            const auto buck_end      = offsets[bucket_idx + 1];
            const SizeType buck_size = buck_end - buck_start;
            if (buck_size == 0) {
                continue;
            }
            const uint32_t* __restrict__ indices = bucket_indices + buck_start;

            float sum_e = 0.0F, sum_v = 0.0F;
            const SizeType main_loop = buck_size - (buck_size % kUnrollFactor);

            for (SizeType i = 0; i < main_loop; i += kUnrollFactor) {
                UNROLL_VECTORIZE
                for (SizeType j = 0; j < kUnrollFactor; ++j) {
                    const auto idx = indices[i + j];
                    sum_e += ts_e_seg[idx];
                    sum_v += ts_v_seg[idx];
                }
            }

            // Handle remainder
            for (SizeType i = main_loop; i < buck_size; ++i) {
                const auto idx = indices[i];
                sum_e += ts_e_seg[idx];
                sum_v += ts_v_seg[idx];
            }

            fold_e_base[iphase] = sum_e;
            fold_v_base[iphase] = sum_v;
        }
    }
}

void brute_fold_ts(const float* __restrict__ ts_e,
                   const float* __restrict__ ts_v,
                   float* __restrict__ fold,
                   const uint32_t* __restrict__ bucket_indices,
                   const SizeType* __restrict__ offsets,
                   SizeType nsegments,
                   SizeType nfreqs,
                   SizeType segment_len,
                   SizeType nbins,
                   int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(ts_e, ts_v, fold, bucket_indices, offsets, nsegments, nfreqs,       \
               segment_len, nbins)
    for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
        const auto start_idx              = iseg * segment_len;
        const auto* __restrict__ ts_e_seg = ts_e + start_idx;
        const auto* __restrict__ ts_v_seg = ts_v + start_idx;
        auto* __restrict__ fold_seg       = fold + (iseg * nfreqs * 2 * nbins);
        kernels::brute_fold_segment(ts_e_seg, ts_v_seg, fold_seg,
                                    bucket_indices, offsets, nfreqs, nbins);
    }
}

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
                                AlignedFloatVec& current_phasors_i) noexcept {
    using BatchType           = xsimd::batch<float>;
    constexpr auto kBatchSize = BatchType::size;
    // DC component: sum over the segment
    BatchType sum_e_vec(0.0F);
    BatchType sum_v_vec(0.0F);
    SizeType i = 0;
    // Main SIMD loop: process 2 * kBatchSize at a time
    for (; i + 2 * kBatchSize <= segment_len; i += 2 * kBatchSize) {
        sum_e_vec += BatchType::load_unaligned(&ts_e_seg[i]);
        sum_v_vec += BatchType::load_unaligned(&ts_v_seg[i]);
        sum_e_vec += BatchType::load_unaligned(&ts_e_seg[i + kBatchSize]);
        sum_v_vec += BatchType::load_unaligned(&ts_v_seg[i + kBatchSize]);
    }
    // One more batch if at least kBatchSize left
    if (i + kBatchSize <= segment_len) {
        sum_e_vec += BatchType::load_unaligned(&ts_e_seg[i]);
        sum_v_vec += BatchType::load_unaligned(&ts_v_seg[i]);
        i += kBatchSize;
    }
    float sum_e = xsimd::reduce_add(sum_e_vec);
    float sum_v = xsimd::reduce_add(sum_v_vec);
    // Scalar tail
    for (; i < segment_len; ++i) {
        sum_e += ts_e_seg[i];
        sum_v += ts_v_seg[i];
    }

    for (SizeType ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const auto freq            = static_cast<float>(freqs[ifreq]);
        const auto freq_offset_out = ifreq * 2 * nbins_f;
        ComplexType* __restrict__ fold_e_base = fold_seg + freq_offset_out;
        ComplexType* __restrict__ fold_v_base = fold_e_base + nbins_f;

        // DC component: Assign pre-computed sums
        fold_e_base[0] = ComplexType(sum_e, 0.0F);
        fold_v_base[0] = ComplexType(sum_v, 0.0F);

        // --- Helper lambda for the phasors ---
        auto compute_base_phasors =
            [&](SizeType k, AlignedFloatVec& delta_phasors_r,
                AlignedFloatVec& delta_phasors_i, float phase_factor) {
                const auto phase =
                    phase_factor * BatchType::load_aligned(&proper_time[k]);
                const auto [sin_phase, cos_phase] = xsimd::sincos(phase);
                cos_phase.store_aligned(&delta_phasors_r[k]);
                sin_phase.store_aligned(&delta_phasors_i[k]);
            };

        // --- Helper lambda for the core computation ---
        auto compute_dft_op = [&](SizeType k, BatchType& acc_e_r,
                                  BatchType& acc_e_i, BatchType& acc_v_r,
                                  BatchType& acc_v_i,
                                  AlignedFloatVec& current_phasors_r,
                                  AlignedFloatVec& current_phasors_i) {
            const auto samples_e = BatchType::load_unaligned(&ts_e_seg[k]);
            const auto samples_v = BatchType::load_unaligned(&ts_v_seg[k]);
            const auto phasor_r =
                BatchType::load_aligned(&current_phasors_r[k]);
            const auto phasor_i =
                BatchType::load_aligned(&current_phasors_i[k]);

            acc_e_r = xsimd::fma(samples_e, phasor_r, acc_e_r);
            acc_e_i = xsimd::fma(samples_e, phasor_i, acc_e_i);
            acc_v_r = xsimd::fma(samples_v, phasor_r, acc_v_r);
            acc_v_i = xsimd::fma(samples_v, phasor_i, acc_v_i);

            // update current phasor ← current * base
            const auto delta_r = BatchType::load_aligned(&delta_phasors_r[k]);
            const auto delta_i = BatchType::load_aligned(&delta_phasors_i[k]);
            const auto new_r =
                xsimd::fms(phasor_r, delta_r, phasor_i * delta_i);
            const auto new_i =
                xsimd::fma(phasor_r, delta_i, phasor_i * delta_r);
            new_r.store_aligned(&current_phasors_r[k]);
            new_i.store_aligned(&current_phasors_i[k]);
        };

        // Compute base phasors
        const auto phase_factor =
            -2.0F * static_cast<float>(std::numbers::pi) * freq;
        SizeType j = 0;
        for (; j + 2 * kBatchSize <= segment_len; j += 2 * kBatchSize) {
            compute_base_phasors(j, delta_phasors_r, delta_phasors_i,
                                 phase_factor);
            compute_base_phasors(j + kBatchSize, delta_phasors_r,
                                 delta_phasors_i, phase_factor);
        }
        if (j + kBatchSize <= segment_len) {
            compute_base_phasors(j, delta_phasors_r, delta_phasors_i,
                                 phase_factor);
            j += kBatchSize;
        }
        for (; j < segment_len; ++j) {
            const auto phase =
                static_cast<float>(phase_factor) * proper_time[j];
            delta_phasors_r[j] = std::cos(phase);
            delta_phasors_i[j] = std::sin(phase);
        }
        for (SizeType i = 0; i < segment_len; ++i) {
            current_phasors_r[i] = delta_phasors_r[i];
            current_phasors_i[i] = delta_phasors_i[i];
        }

        // AC components with SIMD
        for (SizeType m = 1; m < nbins_f; ++m) {
            BatchType acc_e_r(0.0F);
            BatchType acc_e_i(0.0F);
            BatchType acc_v_r(0.0F);
            BatchType acc_v_i(0.0F);

            SizeType k = 0;
            for (; k + 2 * kBatchSize <= segment_len; k += 2 * kBatchSize) {
                compute_dft_op(k, acc_e_r, acc_e_i, acc_v_r, acc_v_i,
                               current_phasors_r, current_phasors_i);
                compute_dft_op(k + kBatchSize, acc_e_r, acc_e_i, acc_v_r,
                               acc_v_i, current_phasors_r, current_phasors_i);
            }
            if (k + kBatchSize <= segment_len) {
                compute_dft_op(k, acc_e_r, acc_e_i, acc_v_r, acc_v_i,
                               current_phasors_r, current_phasors_i);
                k += kBatchSize;
            }

            float final_e_r = xsimd::reduce_add(acc_e_r);
            float final_e_i = xsimd::reduce_add(acc_e_i);
            float final_v_r = xsimd::reduce_add(acc_v_r);
            float final_v_i = xsimd::reduce_add(acc_v_i);

            for (; k < segment_len; ++k) {
                final_e_r += ts_e_seg[k] * current_phasors_r[k];
                final_e_i += ts_e_seg[k] * current_phasors_i[k];
                final_v_r += ts_v_seg[k] * current_phasors_r[k];
                final_v_i += ts_v_seg[k] * current_phasors_i[k];

                // update current phasor ← current * base
                const float old_r = current_phasors_r[k];
                const float old_i = current_phasors_i[k];
                current_phasors_r[k] =
                    old_r * delta_phasors_r[k] - old_i * delta_phasors_i[k];
                current_phasors_i[k] =
                    old_r * delta_phasors_i[k] + old_i * delta_phasors_r[k];
            }

            fold_e_base[m] = ComplexType(final_e_r, final_e_i);
            fold_v_base[m] = ComplexType(final_v_r, final_v_i);
        }
    }
}

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
                           int nthreads) noexcept {
    nthreads           = std::clamp(nthreads, 1, omp_get_max_threads());
    const auto nbins_f = (nbins / 2) + 1;

    // Precompute proper_time vector
    AlignedFloatVec proper_time(segment_len);
    for (SizeType i = 0; i < segment_len; ++i) {
        proper_time[i] =
            static_cast<float>((static_cast<double>(i) * tsamp) - t_ref);
    }

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold, freqs, nfreqs, nsegments, segment_len, nbins,     \
               tsamp, t_ref, nbins_f, proper_time)
    {
        AlignedFloatVec delta_phasors_r(segment_len);
        AlignedFloatVec delta_phasors_i(segment_len);
        AlignedFloatVec current_phasors_r(segment_len);
        AlignedFloatVec current_phasors_i(segment_len);

#pragma omp for
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            const SizeType start_idx           = iseg * segment_len;
            const float* __restrict__ ts_e_seg = ts_e + start_idx;
            const float* __restrict__ ts_v_seg = ts_v + start_idx;
            ComplexType* __restrict__ fold_seg =
                fold + (iseg * nfreqs * 2 * nbins_f);
            brute_fold_segment_complex(
                ts_e_seg, ts_v_seg, fold_seg, freqs, nfreqs, nbins_f,
                segment_len, proper_time, delta_phasors_r, delta_phasors_i,
                current_phasors_r, current_phasors_i);
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

                    shift_add_buffer_binary(fold_tail, coord_cur->shift_tail,
                                            fold_head, coord_cur->shift_head,
                                            fold_sum, temp_buffer_ptr, nbins);
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

                    shift_add_buffer_binary(fold_tail, coord_cur->shift_tail,
                                            fold_head, coord_cur->shift_head,
                                            fold_sum, temp_buffer_ptr, nbins);
                }
            }
        }
    }
}

void ffa_iter_segment_freq(const float* __restrict__ fold_in,
                           float* __restrict__ fold_out,
                           const plans::FFACoordFreq* __restrict__ coords_cur,
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
                        (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                    const auto head_offset =
                        ((iseg * 2 + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_buffer_linear(fold_tail, fold_head,
                                            coord_cur->shift, fold_sum,
                                            temp_buffer_ptr, nbins);
                }
            }
        }
    }
}

void ffa_iter_standard_freq(const float* __restrict__ fold_in,
                            float* __restrict__ fold_out,
                            const plans::FFACoordFreq* __restrict__ coords_cur,
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
                        (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                    const auto head_offset =
                        ((iseg * 2 + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_buffer_linear(fold_tail, fold_head,
                                            coord_cur->shift, fold_sum,
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

                kernels::shift_add_complex_recurrence_binary(
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

                kernels::shift_add_complex_recurrence_binary(
                    fold_tail, coord_cur->shift_tail, fold_head,
                    coord_cur->shift_head, fold_sum, nbins_f, nbins);
            }
        }
    }
}

void ffa_complex_iter_segment_freq(
    const ComplexType* __restrict__ fold_in,
    ComplexType* __restrict__ fold_out,
    const plans::FFACoordFreq* __restrict__ coords_cur,
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
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto head_offset =
                    ((iseg * 2 + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                kernels::shift_add_complex_recurrence_linear(
                    fold_tail, fold_head, coord_cur->shift, fold_sum, nbins_f,
                    nbins);
            }
        }
    }
}

void ffa_complex_iter_standard_freq(
    const ComplexType* __restrict__ fold_in,
    ComplexType* __restrict__ fold_out,
    const plans::FFACoordFreq* __restrict__ coords_cur,
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
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto head_offset =
                    ((iseg * 2 + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                kernels::shift_add_complex_recurrence_linear(
                    fold_tail, fold_head, coord_cur->shift, fold_sum, nbins_f,
                    nbins);
            }
        }
    }
}

void shift_add_buffer_batch(const float* __restrict__ data_folds,
                            const SizeType* __restrict__ idx_folds,
                            const float* __restrict__ data_ffa,
                            const SizeType* __restrict__ idx_ffa,
                            const float* __restrict__ shift_batch,
                            float* __restrict__ out,
                            float* __restrict__ temp_buffer,
                            SizeType nbins,
                            SizeType nbatch) noexcept {
    const auto total_size = 2 * nbins;
    for (SizeType irow = 0; irow < nbatch; ++irow) {
        // Get restrict pointers for current batch item
        const float* __restrict__ data_tail =
            data_folds + (idx_folds[irow] * total_size);
        const float* __restrict__ data_head =
            data_ffa + (idx_ffa[irow] * total_size);
        float* __restrict__ data_out = out + (irow * total_size);
        shift_add_buffer_linear(data_tail, data_head, shift_batch[irow],
                                data_out, temp_buffer, nbins);
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
    const auto total_size = 2 * nbins_f;
    for (SizeType irow = 0; irow < nbatch; ++irow) {
        // Get restrict pointers for current batch item
        const auto* __restrict__ data_tail =
            data_folds + (idx_folds[irow] * total_size);
        const auto* __restrict__ data_head =
            data_ffa + (idx_ffa[irow] * total_size);
        auto* __restrict__ data_out = out + (irow * total_size);
        shift_add_complex_recurrence_linear(
            data_tail, data_head, shift_batch[irow], data_out, nbins_f, nbins);
    }
}

} // namespace loki::kernels