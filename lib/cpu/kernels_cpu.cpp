#include "lib/core/kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <utility>
#include <vector>

#include <omp.h>
#include <xsimd/xsimd.hpp>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

#include "lib/cpu/brute_fold_intrinsics.hpp"

namespace loki::core {

namespace {

// Defined here so it does not interfer with nvcc
using AlignedFloatVec = std::vector<float, xsimd::aligned_allocator<float>>;

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

[[maybe_unused]] void shift_add_binary(const float* __restrict__ data_tail,
                                       float phase_shift_tail,
                                       const float* __restrict__ data_head,
                                       float phase_shift_head,
                                       float* __restrict__ out,
                                       SizeType nbins) noexcept {

    auto shift_tail = static_cast<SizeType>(phase_shift_tail + 0.5F);
    if (shift_tail == nbins) {
        shift_tail = 0;
    }
    auto shift_head = static_cast<SizeType>(phase_shift_head + 0.5F);
    if (shift_head == nbins) {
        shift_head = 0;
    }
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

/**
 * @brief Optimized version of shift_add_binary, using a single pre-allocated
 * buffer of size 2 * nbins.
 */
void shift_add_binary_with_buffer(const float* __restrict__ data_tail,
                                  float phase_shift_tail,
                                  const float* __restrict__ data_head,
                                  float phase_shift_head,
                                  float* __restrict__ out,
                                  float* __restrict__ temp_buffer,
                                  SizeType nbins) noexcept {
    auto shift_tail = static_cast<SizeType>(phase_shift_tail + 0.5F);
    if (shift_tail == nbins) {
        shift_tail = 0;
    }
    auto shift_head = static_cast<SizeType>(phase_shift_head + 0.5F);
    if (shift_head == nbins) {
        shift_head = 0;
    }
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

/**
 * @brief Right-rotate `head` by `shift` bins and add it to `tail`.
 *
 * out[j] = tail[j] + head[(j - shift) mod nbins], for one channel. Two
 * contiguous loops (the wrapped prefix, then the in-order suffix) so the
 * add stream-vectorizes without a temporary buffer.
 */
inline void shift_add_channel(const float* __restrict__ tail,
                              const float* __restrict__ head,
                              float* __restrict__ out,
                              SizeType shift,
                              SizeType nbins) noexcept {
    const SizeType wrapped = nbins - shift;
#pragma omp simd
    for (SizeType j = 0; j < shift; ++j) {
        out[j] = tail[j] + head[wrapped + j];
    }
#pragma omp simd
    for (SizeType j = shift; j < nbins; ++j) {
        out[j] = tail[j] + head[j - shift];
    }
}

inline void shift_add_channel_inplace(const float* __restrict__ head,
                                      float* __restrict__ out,
                                      SizeType shift,
                                      SizeType nbins) noexcept {
    const SizeType wrapped = nbins - shift;
#pragma omp simd
    for (SizeType j = 0; j < shift; ++j) {
        out[j] += head[wrapped + j];
    }
#pragma omp simd
    for (SizeType j = shift; j < nbins; ++j) {
        out[j] += head[j - shift];
    }
}

[[nodiscard]] inline SizeType phase_shift_bins(float phase_shift,
                                               SizeType nbins) noexcept {
    auto shift = static_cast<SizeType>(phase_shift + 0.5F);
    if (shift == nbins) {
        shift = 0;
    }
    return shift;
}

/**
 * @brief Add a rotated head profile (energy and variance) onto the tail.
 *
 * Equivalent to the previous rotate-into-temp-buffer implementation. Both
 * channels are independent contiguous adds.
 */
void shift_add_linear_with_buffer(const float* __restrict__ data_tail,
                                  const float* __restrict__ data_head,
                                  float phase_shift,
                                  float* __restrict__ out,
                                  SizeType nbins) noexcept {
    const SizeType shift = phase_shift_bins(phase_shift, nbins);
    shift_add_channel(data_tail, data_head, out, shift, nbins);
    shift_add_channel(data_tail + nbins, data_head + nbins, out + nbins, shift,
                      nbins);
}

void shift_add_linear_with_buffer_inplace(const float* __restrict__ data_head,
                                          float phase_shift,
                                          float* __restrict__ out,
                                          SizeType nbins) noexcept {
    const SizeType shift = phase_shift_bins(phase_shift, nbins);
    shift_add_channel_inplace(data_head, out, shift, nbins);
    shift_add_channel_inplace(data_head + nbins, out + nbins, shift, nbins);
}

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
[[maybe_unused]] void
shift_add_complex_binary(const ComplexType* __restrict__ data_tail,
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

    auto const compute_fused_op = [&](SizeType k, const BatchType& phase_tail,
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
    for (; k + (2 * kBatchSize) <= nbins_f; k += 2 * kBatchSize) {
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
    SizeType nbins) noexcept {
    // On the fly phase computation is faster than precomputed phase steps on
    // Intel CPUs.
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

    auto const compute_and_store = [&](SizeType k, const BatchType& phase) {
        const auto tail_e_data = xsimd::load_unaligned(&data_tail_e[k]);
        const auto head_e_data = xsimd::load_unaligned(&data_head_e[k]);
        (tail_e_data + head_e_data * phase).store_unaligned(&out_e[k]);
        const auto tail_v_data = xsimd::load_unaligned(&data_tail_v[k]);
        const auto head_v_data = xsimd::load_unaligned(&data_head_v[k]);
        (tail_v_data + head_v_data * phase).store_unaligned(&out_v[k]);
    };

    // First process two batches at a time to maximize throughput
    SizeType k = 0;
    for (; k + (2 * kBatchSize) <= nbins_f; k += 2 * kBatchSize) {
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

void shift_add_complex_recurrence_linear_inplace(
    const ComplexType* __restrict__ data_head,
    float phase_shift,
    ComplexType* __restrict__ out,
    SizeType nbins_f,
    SizeType nbins) noexcept {
    // On the fly phase computation is faster than precomputed phase steps on
    // Intel CPUs.
    using BatchType                  = xsimd::batch<ComplexType>;
    static constexpr auto kBatchSize = BatchType::size;

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

    auto const accumulate_inplace = [&](SizeType k, const BatchType& phase) {
        const auto out_e_data  = xsimd::load_unaligned(&out_e[k]);
        const auto head_e_data = xsimd::load_unaligned(&data_head_e[k]);
        (out_e_data + head_e_data * phase).store_unaligned(&out_e[k]);
        const auto out_v_data  = xsimd::load_unaligned(&out_v[k]);
        const auto head_v_data = xsimd::load_unaligned(&data_head_v[k]);
        (out_v_data + head_v_data * phase).store_unaligned(&out_v[k]);
    };

    // First process two batches at a time to maximize throughput
    SizeType k = 0;
    for (; k + (2 * kBatchSize) <= nbins_f; k += 2 * kBatchSize) {
        const BatchType phase0 =
            xsimd::broadcast(current_block_start_phase) * delta_vec;
        accumulate_inplace(k, phase0);
        current_block_start_phase *= delta_block;
        const BatchType phase1 =
            xsimd::broadcast(current_block_start_phase) * delta_vec;
        accumulate_inplace(k + kBatchSize, phase1);
        current_block_start_phase *= delta_block;
    }

    // Process the remaining batches
    if (k + kBatchSize <= nbins_f) {
        const BatchType phase =
            xsimd::broadcast(current_block_start_phase) * delta_vec;
        accumulate_inplace(k, phase);
        k += kBatchSize;
        current_block_start_phase *= delta_block;
    }

    // Scalar remainder part
    if (k < nbins_f) {
        ComplexType current_phase = current_block_start_phase;
        for (; k < nbins_f; ++k) {
            out_e[k] += data_head_e[k] * current_phase;
            out_v[k] += data_head_v[k] * current_phase;
            current_phase *= delta_phase;
        }
    }
}

// One-shot m=1 phasor table. Double phase, scalar libm; OpenMP over
// frequencies.
void precompute_base_phasors_scalar(float* __restrict__ delta_r,
                                    float* __restrict__ delta_i,
                                    const double* __restrict__ freqs,
                                    SizeType nfreqs,
                                    SizeType segment_len,
                                    double tsamp,
                                    double t_ref,
                                    int nthreads) {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(delta_r, delta_i, freqs, nfreqs, segment_len, tsamp, t_ref)
    for (SizeType ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const double phase_factor = -2.0 * std::numbers::pi * freqs[ifreq];
        const auto off            = ifreq * segment_len;
        for (SizeType i = 0; i < segment_len; ++i) {
            const double proper_time = (static_cast<double>(i) * tsamp) - t_ref;
            const double phase       = phase_factor * proper_time;
            delta_r[off + i]         = static_cast<float>(std::cos(phase));
            delta_i[off + i]         = static_cast<float>(std::sin(phase));
        }
    }
}

void brute_fold_segment_complex_xsimd(
    const float* __restrict__ ts_e_seg,
    const float* __restrict__ ts_v_seg,
    ComplexType* __restrict__ fold_seg,
    SizeType nfreqs,
    SizeType nbins_f,
    SizeType segment_len,
    const float* __restrict__ delta_phasors_r,
    const float* __restrict__ delta_phasors_i,
    float* __restrict__ current_phasors_r,
    float* __restrict__ current_phasors_i) noexcept {
    using BatchType           = xsimd::batch<float>;
    constexpr auto kBatchSize = BatchType::size;
    // DC component: sum over the segment
    BatchType sum_e_vec(0.0F);
    BatchType sum_v_vec(0.0F);
    SizeType i = 0;
    // Main SIMD loop: process 2 * kBatchSize at a time
    for (; i + (2 * kBatchSize) <= segment_len; i += 2 * kBatchSize) {
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
        const auto freq_offset_out            = ifreq * 2 * nbins_f;
        const auto phasor_offset              = ifreq * segment_len;
        ComplexType* __restrict__ fold_e_base = fold_seg + freq_offset_out;
        ComplexType* __restrict__ fold_v_base = fold_e_base + nbins_f;

        // DC component: Assign pre-computed sums
        fold_e_base[0] = ComplexType(sum_e, 0.0F);
        fold_v_base[0] = ComplexType(sum_v, 0.0F);

        // --- Helper lambda for the core computation ---
        auto const compute_dft_op = [&](SizeType k, BatchType& acc_e_r,
                                        BatchType& acc_e_i, BatchType& acc_v_r,
                                        BatchType& acc_v_i) {
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
            const auto delta_r =
                BatchType::load_unaligned(&delta_phasors_r[phasor_offset + k]);
            const auto delta_i =
                BatchType::load_unaligned(&delta_phasors_i[phasor_offset + k]);
            const auto new_r =
                xsimd::fms(phasor_r, delta_r, phasor_i * delta_i);
            const auto new_i =
                xsimd::fma(phasor_r, delta_i, phasor_i * delta_r);
            new_r.store_aligned(&current_phasors_r[k]);
            new_i.store_aligned(&current_phasors_i[k]);
        };
        const float* __restrict__ base_r_start =
            delta_phasors_r + static_cast<std::ptrdiff_t>(phasor_offset);
        const float* __restrict__ base_i_start =
            delta_phasors_i + static_cast<std::ptrdiff_t>(phasor_offset);
        std::copy(base_r_start,
                  base_r_start + static_cast<std::ptrdiff_t>(segment_len),
                  current_phasors_r);
        std::copy(base_i_start,
                  base_i_start + static_cast<std::ptrdiff_t>(segment_len),
                  current_phasors_i);

        // AC components with SIMD
        for (SizeType m = 1; m < nbins_f; ++m) {
            BatchType acc_e_r(0.0F);
            BatchType acc_e_i(0.0F);
            BatchType acc_v_r(0.0F);
            BatchType acc_v_i(0.0F);

            SizeType k = 0;
            for (; k + (2 * kBatchSize) <= segment_len; k += 2 * kBatchSize) {
                compute_dft_op(k, acc_e_r, acc_e_i, acc_v_r, acc_v_i);
                compute_dft_op(k + kBatchSize, acc_e_r, acc_e_i, acc_v_r,
                               acc_v_i);
            }
            if (k + kBatchSize <= segment_len) {
                compute_dft_op(k, acc_e_r, acc_e_i, acc_v_r, acc_v_i);
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
                    (old_r * base_r_start[k]) - (old_i * base_i_start[k]);
                current_phasors_i[k] =
                    (old_r * base_i_start[k]) + (old_i * base_r_start[k]);
            }

            fold_e_base[m] = ComplexType(final_e_r, final_e_i);
            fold_v_base[m] = ComplexType(final_v_r, final_v_i);
        }
    }
}

/**
 * @brief Merge one pair of adjacent segments of a frequency-only FFA level.
 *
 * @param fold_tail_seg  Level-(l-1) segment 2*s (ncoords_prev profiles).
 * @param fold_head_seg  Level-(l-1) segment 2*s+1 (ncoords_prev profiles).
 * @param fold_out_seg   Level-l segment s (ncoords_cur profiles).
 */
inline void
ffa_merge_segment_freq(const float* __restrict__ fold_tail_seg,
                       const float* __restrict__ fold_head_seg,
                       float* __restrict__ fold_out_seg,
                       const coord::FFACoordFreq* __restrict__ coords,
                       SizeType ncoords_cur,
                       SizeType nbins) noexcept {
    constexpr SizeType kBlockSize = 32;
    const SizeType fold_stride    = 2 * nbins;
    for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
         icoord_block += kBlockSize) {
        const auto block_end = std::min(icoord_block + kBlockSize, ncoords_cur);
        for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
            const auto* __restrict__ coord_cur = &coords[icoord];
            const auto prev_offset =
                static_cast<SizeType>(coord_cur->idx) * fold_stride;
            shift_add_linear_with_buffer(
                fold_tail_seg + prev_offset, fold_head_seg + prev_offset,
                coord_cur->shift, fold_out_seg + (icoord * fold_stride), nbins);
        }
    }
}

void ffa_iter_segment_freq(const float* __restrict__ fold_in,
                           float* __restrict__ fold_out,
                           const coord::FFACoordFreq* __restrict__ coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins,
                           int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    // Process one segment at a time to keep data in cache
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, nsegments, nbins,           \
               seg_prev_stride, seg_out_stride)
    {
#pragma omp for schedule(static)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            ffa_merge_segment_freq(
                fold_in + ((iseg * 2) * seg_prev_stride),
                fold_in + (((iseg * 2) + 1) * seg_prev_stride),
                fold_out + (iseg * seg_out_stride), coords, ncoords_cur, nbins);
        }
    }
}

void ffa_iter_standard_freq(const float* __restrict__ fold_in,
                            float* __restrict__ fold_out,
                            const coord::FFACoordFreq* __restrict__ coords,
                            SizeType ncoords_cur,
                            SizeType ncoords_prev,
                            SizeType nsegments,
                            SizeType nbins,
                            int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, nsegments, nbins,           \
               fold_stride, seg_prev_stride, seg_out_stride)
    {
#pragma omp for schedule(static)
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            const auto block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto* __restrict__ coord_cur = &coords[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                    const auto head_offset =
                        (((iseg * 2) + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_linear_with_buffer(fold_tail, fold_head,
                                                 coord_cur->shift, fold_sum,
                                                 nbins);
                }
            }
        }
    }
}

void ffa_iter_segment(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      const coord::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nsegments,
                      SizeType nbins,
                      int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins, fold_stride, seg_prev_stride, seg_out_stride)
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
                    const auto* __restrict__ coord_cur = &coords[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_tail) *
                         fold_stride);
                    const auto head_offset =
                        (((iseg * 2) + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_head) *
                         fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_binary_with_buffer(
                        fold_tail, coord_cur->shift_tail, fold_head,
                        coord_cur->shift_head, fold_sum, temp_buffer_ptr,
                        nbins);
                }
            }
        }
    }
}

void ffa_iter_standard(const float* __restrict__ fold_in,
                       float* __restrict__ fold_out,
                       const coord::FFACoord* __restrict__ coords,
                       SizeType ncoords_cur,
                       SizeType ncoords_prev,
                       SizeType nsegments,
                       SizeType nbins,
                       int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins, fold_stride, seg_prev_stride, seg_out_stride)
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
                    const auto* __restrict__ coord_cur = &coords[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_tail) *
                         fold_stride);
                    const auto head_offset =
                        (((iseg * 2) + 1) * seg_prev_stride) +
                        (static_cast<SizeType>(coord_cur->i_head) *
                         fold_stride);
                    const auto out_offset =
                        (iseg * seg_out_stride) + (icoord * fold_stride);

                    const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                    const auto* __restrict__ fold_head = &fold_in[head_offset];
                    auto* __restrict__ fold_sum        = &fold_out[out_offset];

                    shift_add_binary_with_buffer(
                        fold_tail, coord_cur->shift_tail, fold_head,
                        coord_cur->shift_head, fold_sum, temp_buffer_ptr,
                        nbins);
                }
            }
        }
    }
}

void ffa_complex_iter_segment(const ComplexType* __restrict__ fold_in,
                              ComplexType* __restrict__ fold_out,
                              const coord::FFACoord* __restrict__ coords,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev,
                              SizeType nsegments,
                              SizeType nbins_f,
                              SizeType nbins,
                              int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
        // Process coordinates in blocks within each segment
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType const block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
                const auto* __restrict__ coord_cur = &coords[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_tail) * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_head) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                shift_add_complex_recurrence_binary(
                    fold_tail, coord_cur->shift_tail, fold_head,
                    coord_cur->shift_head, fold_sum, nbins_f, nbins);
            }
        }
    }
}

void ffa_complex_iter_standard(const ComplexType* __restrict__ fold_in,
                               ComplexType* __restrict__ fold_out,
                               const coord::FFACoord* __restrict__ coords,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev,
                               SizeType nsegments,
                               SizeType nbins_f,
                               SizeType nbins,
                               int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
         icoord_block += kBlockSize) {
        SizeType const block_end =
            std::min(icoord_block + kBlockSize, ncoords_cur);
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
                const auto* __restrict__ coord_cur = &coords[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_tail) * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_head) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                shift_add_complex_recurrence_binary(
                    fold_tail, coord_cur->shift_tail, fold_head,
                    coord_cur->shift_head, fold_sum, nbins_f, nbins);
            }
        }
    }
}

void ffa_complex_iter_segment_freq(
    const ComplexType* __restrict__ fold_in,
    ComplexType* __restrict__ fold_out,
    const coord::FFACoordFreq* __restrict__ coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
        // Process coordinates in blocks within each segment
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType const block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
                const auto* __restrict__ coord_cur = &coords[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                shift_add_complex_recurrence_linear(fold_tail, fold_head,
                                                    coord_cur->shift, fold_sum,
                                                    nbins_f, nbins);
            }
        }
    }
}

void ffa_complex_iter_standard_freq(
    const ComplexType* __restrict__ fold_in,
    ComplexType* __restrict__ fold_out,
    const coord::FFACoordFreq* __restrict__ coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
         icoord_block += kBlockSize) {
        SizeType const block_end =
            std::min(icoord_block + kBlockSize, ncoords_cur);
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType icoord = icoord_block; icoord < block_end; ++icoord) {
                const auto* __restrict__ coord_cur = &coords[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->idx) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);

                const auto* __restrict__ fold_tail = &fold_in[tail_offset];
                const auto* __restrict__ fold_head = &fold_in[head_offset];
                auto* __restrict__ fold_sum        = &fold_out[out_offset];

                shift_add_complex_recurrence_linear(fold_tail, fold_head,
                                                    coord_cur->shift, fold_sum,
                                                    nbins_f, nbins);
            }
        }
    }
}

void fill_segment_prefix(const float* __restrict__ src,
                         double* __restrict__ dst,
                         SizeType n) noexcept {
    double acc = 0.0;
    dst[0]     = 0.0;
    for (SizeType i = 0; i < n; ++i) {
        acc += static_cast<double>(src[i]);
        dst[i + 1] = acc;
    }
}

/// Fold coordinates `[coord_begin, coord_end)` from prefix sums into a packed
/// `[ncoords, 2, nbins]` buffer. Each bin is the float-rounded double sum of
/// its runs, accumulated when a bin is visited more than once.
void brute_fold_runs_range(const double* __restrict__ prefix_e,
                           const double* __restrict__ prefix_v,
                           float* __restrict__ fold_packed,
                           const PhaseRun* __restrict__ runs,
                           const SizeType* __restrict__ run_offsets,
                           SizeType coord_begin,
                           SizeType coord_end,
                           SizeType nbins) noexcept {
    const SizeType ncoords     = coord_end - coord_begin;
    const SizeType fold_stride = 2 * nbins;
    std::fill(fold_packed, fold_packed + (ncoords * fold_stride), 0.0F);
    for (SizeType local = 0; local < ncoords; ++local) {
        const SizeType ifreq       = coord_begin + local;
        float* __restrict__ fold_e = fold_packed + (local * fold_stride);
        float* __restrict__ fold_v = fold_e + nbins;
        const PhaseRun* run        = runs + run_offsets[ifreq];
        const PhaseRun* run_end    = runs + run_offsets[ifreq + 1];
        uint32_t start             = 0;
        for (; run != run_end; ++run) {
            const auto end = run->end;
            fold_e[run->bin] +=
                static_cast<float>(prefix_e[end] - prefix_e[start]);
            fold_v[run->bin] +=
                static_cast<float>(prefix_v[end] - prefix_v[start]);
            start = end;
        }
    }
}

#if defined(__AVX2__)
/// 4-segment batched brute fold using AVX2 SIMD on pre-interleaved prefixes.
/// Folds 4 segments simultaneously into fold_buf, then applies an SSE 4x4
/// matrix transpose into standard [iseg, ncoords, 2, nbins] layout.
void brute_fold_runs_range_4seg(const double* __restrict__ p4_e,
                                const double* __restrict__ p4_v,
                                float* __restrict__ cur_base,
                                float* __restrict__ fold_buf,
                                SizeType seg_stride,
                                const PhaseRun* __restrict__ runs,
                                const SizeType* __restrict__ run_offsets,
                                SizeType coord_begin,
                                SizeType coord_end,
                                SizeType nbins) noexcept {
    const SizeType ncoords       = coord_end - coord_begin;
    const SizeType total_bins    = 2 * nbins;
    const SizeType fold_stride_4 = total_bins * 4;

    float* dst0 = cur_base;
    float* dst1 = cur_base + seg_stride;
    float* dst2 = cur_base + 2 * seg_stride;
    float* dst3 = cur_base + 3 * seg_stride;

    for (SizeType local = 0; local < ncoords; ++local) {
        std::fill(fold_buf, fold_buf + fold_stride_4, 0.0F);

        float* __restrict__ fe = fold_buf;
        float* __restrict__ fv = fe + (nbins * 4);

        const SizeType ifreq    = coord_begin + local;
        const PhaseRun* run     = runs + run_offsets[ifreq];
        const PhaseRun* run_end = runs + run_offsets[ifreq + 1];

        __m256d prev_e = _mm256_loadu_pd(p4_e);
        __m256d prev_v = _mm256_loadu_pd(p4_v);

        for (; run + 2 <= run_end; run += 2) {
            const auto end0 = run[0].end;
            const auto bin0 = run[0].bin;
            const auto end1 = run[1].end;
            const auto bin1 = run[1].bin;

            __m256d cur_e0 = _mm256_loadu_pd(p4_e + end0 * 4);
            __m256d cur_v0 = _mm256_loadu_pd(p4_v + end0 * 4);
            __m256d cur_e1 = _mm256_loadu_pd(p4_e + end1 * 4);
            __m256d cur_v1 = _mm256_loadu_pd(p4_v + end1 * 4);

            __m256d diff_e0 = _mm256_sub_pd(cur_e0, prev_e);
            __m256d diff_v0 = _mm256_sub_pd(cur_v0, prev_v);
            __m256d diff_e1 = _mm256_sub_pd(cur_e1, cur_e0);
            __m256d diff_v1 = _mm256_sub_pd(cur_v1, cur_v0);

            prev_e = cur_e1;
            prev_v = cur_v1;

            __m128 de0 = _mm256_cvtpd_ps(diff_e0);
            __m128 dv0 = _mm256_cvtpd_ps(diff_v0);
            __m128 de1 = _mm256_cvtpd_ps(diff_e1);
            __m128 dv1 = _mm256_cvtpd_ps(diff_v1);

            __m128 old_e0 = _mm_loadu_ps(fe + bin0 * 4);
            __m128 old_v0 = _mm_loadu_ps(fv + bin0 * 4);
            __m128 old_e1 = _mm_loadu_ps(fe + bin1 * 4);
            __m128 old_v1 = _mm_loadu_ps(fv + bin1 * 4);

            _mm_storeu_ps(fe + bin0 * 4, _mm_add_ps(old_e0, de0));
            _mm_storeu_ps(fv + bin0 * 4, _mm_add_ps(old_v0, dv0));
            _mm_storeu_ps(fe + bin1 * 4, _mm_add_ps(old_e1, de1));
            _mm_storeu_ps(fv + bin1 * 4, _mm_add_ps(old_v1, dv1));
        }

        for (; run != run_end; ++run) {
            const auto end = run->end;
            const auto bin = run->bin;

            __m256d cur_e = _mm256_loadu_pd(p4_e + end * 4);
            __m256d cur_v = _mm256_loadu_pd(p4_v + end * 4);

            __m256d diff_e_d = _mm256_sub_pd(cur_e, prev_e);
            __m256d diff_v_d = _mm256_sub_pd(cur_v, prev_v);

            prev_e = cur_e;
            prev_v = cur_v;

            __m128 diff_e = _mm256_cvtpd_ps(diff_e_d);
            __m128 diff_v = _mm256_cvtpd_ps(diff_v_d);

            __m128 old_e = _mm_loadu_ps(fe + bin * 4);
            __m128 old_v = _mm_loadu_ps(fv + bin * 4);

            _mm_storeu_ps(fe + bin * 4, _mm_add_ps(old_e, diff_e));
            _mm_storeu_ps(fv + bin * 4, _mm_add_ps(old_v, diff_v));
        }

        const SizeType off = local * total_bins;
        float* d0          = dst0 + off;
        float* d1          = dst1 + off;
        float* d2          = dst2 + off;
        float* d3          = dst3 + off;

        SizeType b = 0;
        for (; b + 4 <= total_bins; b += 4) {
            __m128 r0 = _mm_loadu_ps(fold_buf + (b + 0) * 4);
            __m128 r1 = _mm_loadu_ps(fold_buf + (b + 1) * 4);
            __m128 r2 = _mm_loadu_ps(fold_buf + (b + 2) * 4);
            __m128 r3 = _mm_loadu_ps(fold_buf + (b + 3) * 4);

            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

            _mm_storeu_ps(d0 + b, r0);
            _mm_storeu_ps(d1 + b, r1);
            _mm_storeu_ps(d2 + b, r2);
            _mm_storeu_ps(d3 + b, r3);
        }
        for (; b < total_bins; ++b) {
            d0[b] = fold_buf[b * 4 + 0];
            d1[b] = fold_buf[b * 4 + 1];
            d2[b] = fold_buf[b * 4 + 2];
            d3[b] = fold_buf[b * 4 + 3];
        }
    }
}
#endif

[[nodiscard]] bool coords_idx_monotone(const coord::FFACoordFreq* coords,
                                       SizeType n) noexcept {
    for (SizeType i = 1; i < n; ++i) {
        if (coords[i].idx < coords[i - 1].idx) {
            return false;
        }
    }
    return true;
}

/// Widest `idx` span of any `window` consecutive coordinates. Monotone idx.
[[nodiscard]] SizeType max_idx_span(const coord::FFACoordFreq* coords,
                                    SizeType n,
                                    SizeType window) noexcept {
    if (n == 0 || window == 0) {
        return 0;
    }
    window        = std::min(window, n);
    SizeType best = 0;
    for (SizeType i = 0; i + window <= n; ++i) {
        const SizeType span =
            static_cast<SizeType>(coords[i + window - 1].idx) -
            static_cast<SizeType>(coords[i].idx) + 1;
        best = std::max(best, span);
    }
    return best;
}

struct CoordRange {
    SizeType lo{0};
    SizeType hi{0};
};

void tile_input_ranges(const coord::FFACoordFreq* const* coords_levels,
                       SizeType k_levels,
                       SizeType coord_begin,
                       SizeType coord_end,
                       CoordRange* ranges) noexcept {
    ranges[k_levels] = {.lo = coord_begin, .hi = coord_end};
    for (SizeType level = k_levels; level >= 1; --level) {
        const auto* coords   = coords_levels[level];
        const SizeType lo    = ranges[level].lo;
        const SizeType hi    = ranges[level].hi;
        ranges[level - 1].lo = static_cast<SizeType>(coords[lo].idx);
        ranges[level - 1].hi = static_cast<SizeType>(coords[hi - 1].idx) + 1;
    }
}

} // namespace

void brute_fold_ts(const float* __restrict__ ts_e,
                   const float* __restrict__ ts_v,
                   float* __restrict__ fold,
                   const PhaseRun* __restrict__ runs,
                   const SizeType* __restrict__ run_offsets,
                   SizeType nsegments,
                   SizeType nfreqs,
                   SizeType segment_len,
                   SizeType nbins,
                   int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold, runs, run_offsets, nsegments, nfreqs,             \
               segment_len, nbins)
    {
        std::vector<double> prefix_e(segment_len + 1);
        std::vector<double> prefix_v(segment_len + 1);
#pragma omp for schedule(static)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            const auto start_idx              = iseg * segment_len;
            const auto* __restrict__ ts_e_seg = ts_e + start_idx;
            const auto* __restrict__ ts_v_seg = ts_v + start_idx;
            fill_segment_prefix(ts_e_seg, prefix_e.data(), segment_len);
            fill_segment_prefix(ts_v_seg, prefix_v.data(), segment_len);
            auto* __restrict__ fold_seg = fold + (iseg * nfreqs * 2 * nbins);
            brute_fold_runs_range(prefix_e.data(), prefix_v.data(), fold_seg,
                                  runs, run_offsets, 0, nfreqs, nbins);
        }
    }
}

void brute_fold_ffa_fused_freq(const float* __restrict__ ts_e,
                               const float* __restrict__ ts_v,
                               float* __restrict__ fold_out,
                               const PhaseRun* __restrict__ runs,
                               const SizeType* __restrict__ run_offsets,
                               const coord::FFACoordFreq* const* coords_levels,
                               const SizeType* ncoords,
                               SizeType nsegments,
                               SizeType nfreqs,
                               SizeType segment_len,
                               SizeType nbins,
                               SizeType nlevels,
                               int nthreads) noexcept {
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
    const SizeType tile_segments = SizeType{1} << nlevels;
    const SizeType ntiles        = nsegments >> nlevels;
    const SizeType fold_stride   = 2 * nbins;
    // Largest per-level working set of one tile (floats). Profiles at level j
    // number (tile_segments >> j) * ncoords[j].
    SizeType tile_floats = 0;
    for (SizeType j = 0; j <= nlevels; ++j) {
        tile_floats = std::max(tile_floats,
                               (tile_segments >> j) * ncoords[j] * fold_stride);
    }
    const SizeType out_tile_stride = ncoords[nlevels] * fold_stride;
    const SizeType prefix_stride   = segment_len + 1;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold_out, runs, run_offsets, coords_levels, ncoords,    \
               nfreqs, segment_len, nbins, nlevels, tile_segments, ntiles,     \
               fold_stride, tile_floats, out_tile_stride, prefix_stride)
    {
        std::vector<float> scratch_a(tile_floats);
        std::vector<float> scratch_b(tile_floats);
        std::vector<double> prefix_e(tile_segments * prefix_stride);
        std::vector<double> prefix_v(tile_segments * prefix_stride);

#pragma omp for schedule(static)
        for (SizeType itile = 0; itile < ntiles; ++itile) {
            float* cur  = scratch_a.data();
            float* next = scratch_b.data();
            for (SizeType iseg = 0; iseg < tile_segments; ++iseg) {
                const auto start_idx =
                    ((itile * tile_segments) + iseg) * segment_len;
                fill_segment_prefix(ts_e + start_idx,
                                    prefix_e.data() + (iseg * prefix_stride),
                                    segment_len);
                fill_segment_prefix(ts_v + start_idx,
                                    prefix_v.data() + (iseg * prefix_stride),
                                    segment_len);
                brute_fold_runs_range(prefix_e.data() + (iseg * prefix_stride),
                                      prefix_v.data() + (iseg * prefix_stride),
                                      cur + (iseg * nfreqs * fold_stride), runs,
                                      run_offsets, 0, nfreqs, nbins);
            }
            for (SizeType j = 1; j <= nlevels; ++j) {
                const SizeType nseg_out      = tile_segments >> j;
                const SizeType seg_prev_strd = ncoords[j - 1] * fold_stride;
                const SizeType seg_out_strd  = ncoords[j] * fold_stride;
                float* out_base = (j == nlevels)
                                      ? fold_out + (itile * out_tile_stride)
                                      : next;
                for (SizeType iseg = 0; iseg < nseg_out; ++iseg) {
                    ffa_merge_segment_freq(
                        cur + ((iseg * 2) * seg_prev_strd),
                        cur + (((iseg * 2) + 1) * seg_prev_strd),
                        out_base + (iseg * seg_out_strd), coords_levels[j],
                        ncoords[j], nbins);
                }
                std::swap(cur, next);
            }
        }
    }
}

// NOLINTNEXTLINE(bugprone-exception-escape): allocation failure terminates (noexcept kernel)
void brute_fold_ts_complex_xsimd(const float* __restrict__ ts_e,
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

    AlignedFloatVec delta_phasors_r(nfreqs * segment_len);
    AlignedFloatVec delta_phasors_i(nfreqs * segment_len);
    float* __restrict__ delta_phasors_r_ptr = delta_phasors_r.data();
    float* __restrict__ delta_phasors_i_ptr = delta_phasors_i.data();

    // Never xsimd this; it results in numerical error with fast_math on avx2.
    precompute_base_phasors_scalar(delta_phasors_r_ptr, delta_phasors_i_ptr,
                                   freqs, nfreqs, segment_len, tsamp, t_ref,
                                   nthreads);

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold, nfreqs, nsegments, segment_len, nbins_f,          \
               delta_phasors_r_ptr, delta_phasors_i_ptr)
    {
        AlignedFloatVec current_phasors_r(segment_len);
        AlignedFloatVec current_phasors_i(segment_len);
        float* __restrict__ current_phasors_r_ptr = current_phasors_r.data();
        float* __restrict__ current_phasors_i_ptr = current_phasors_i.data();

#pragma omp for
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            const SizeType start_idx           = iseg * segment_len;
            const float* __restrict__ ts_e_seg = ts_e + start_idx;
            const float* __restrict__ ts_v_seg = ts_v + start_idx;
            ComplexType* __restrict__ fold_seg =
                fold + (iseg * nfreqs * 2 * nbins_f);
            brute_fold_segment_complex_xsimd(
                ts_e_seg, ts_v_seg, fold_seg, nfreqs, nbins_f, segment_len,
                delta_phasors_r_ptr, delta_phasors_i_ptr, current_phasors_r_ptr,
                current_phasors_i_ptr);
        }
    }
}

// NOLINTNEXTLINE(bugprone-exception-escape): allocation failure terminates (noexcept kernel)
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
    nthreads = std::clamp(nthreads, 1, omp_get_max_threads());
#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__)) ||         \
    defined(__aarch64__) || defined(__ARM_NEON)

    const auto nbins_f = (nbins / 2) + 1;

    AlignedFloatVec delta_phasors_r(nfreqs * segment_len);
    AlignedFloatVec delta_phasors_i(nfreqs * segment_len);
    float* __restrict__ delta_phasors_r_ptr = delta_phasors_r.data();
    float* __restrict__ delta_phasors_i_ptr = delta_phasors_i.data();

    // Never xsimd this; it results in numerical error with fast_math on avx2.
    precompute_base_phasors_scalar(delta_phasors_r_ptr, delta_phasors_i_ptr,
                                   freqs, nfreqs, segment_len, tsamp, t_ref,
                                   nthreads);

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold, nfreqs, nsegments, segment_len, nbins_f,          \
               delta_phasors_r_ptr, delta_phasors_i_ptr)
    {
        AlignedFloatVec current_phasors_r(segment_len);
        AlignedFloatVec current_phasors_i(segment_len);
        float* __restrict__ current_phasors_r_ptr = current_phasors_r.data();
        float* __restrict__ current_phasors_i_ptr = current_phasors_i.data();

#pragma omp for schedule(static)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            const SizeType start_idx           = iseg * segment_len;
            const float* __restrict__ ts_e_seg = ts_e + start_idx;
            const float* __restrict__ ts_v_seg = ts_v + start_idx;
            ComplexType* __restrict__ fold_seg =
                fold + (iseg * nfreqs * 2 * nbins_f);
            brute_fold_intrinsics::fold_one_segment(
                ts_e_seg, ts_v_seg, fold_seg, current_phasors_r_ptr,
                current_phasors_i_ptr, delta_phasors_r_ptr, delta_phasors_i_ptr,
                nfreqs, segment_len, nbins_f);
        }
    }
#else
    brute_fold_ts_complex_xsimd(ts_e, ts_v, fold, freqs, nfreqs, nsegments,
                                segment_len, nbins, tsamp, t_ref, nthreads);
#endif
}

void ffa_iter(const float* __restrict__ fold_in,
              float* __restrict__ fold_out,
              const coord::FFACoord* __restrict__ coords,
              SizeType ncoords_cur,
              SizeType ncoords_prev,
              SizeType nsegments,
              SizeType nbins,
              int nthreads) noexcept {
    // Heuristic: prefer segment-major when segments dominate work
    const bool segment_major =
        (nsegments >= 128) && (nsegments > ncoords_cur / 64);
    if (segment_major) {
        ffa_iter_segment(fold_in, fold_out, coords, ncoords_cur, ncoords_prev,
                         nsegments, nbins, nthreads);
    } else {
        ffa_iter_standard(fold_in, fold_out, coords, ncoords_cur, ncoords_prev,
                          nsegments, nbins, nthreads);
    }
}

void ffa_iter_freq(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const coord::FFACoordFreq* __restrict__ coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   int nthreads) noexcept {
    const bool segment_major =
        (nsegments >= 128) && (nsegments > ncoords_cur / 64);
    if (segment_major) {
        ffa_iter_segment_freq(fold_in, fold_out, coords, ncoords_cur,
                              ncoords_prev, nsegments, nbins, nthreads);
    } else {
        ffa_iter_standard_freq(fold_in, fold_out, coords, ncoords_cur,
                               ncoords_prev, nsegments, nbins, nthreads);
    }
}

void ffa_complex_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      const coord::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nsegments,
                      SizeType nbins_f,
                      SizeType nbins,
                      int nthreads) noexcept {

    const bool segment_major =
        (nsegments >= 128) && (nsegments > ncoords_cur / 64);
    if (segment_major) {
        ffa_complex_iter_segment(fold_in, fold_out, coords, ncoords_cur,
                                 ncoords_prev, nsegments, nbins_f, nbins,
                                 nthreads);
    } else {
        ffa_complex_iter_standard(fold_in, fold_out, coords, ncoords_cur,
                                  ncoords_prev, nsegments, nbins_f, nbins,
                                  nthreads);
    }
}

void ffa_complex_iter_freq(const ComplexType* __restrict__ fold_in,
                           ComplexType* __restrict__ fold_out,
                           const coord::FFACoordFreq* __restrict__ coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           int nthreads) noexcept {
    const bool segment_major =
        (nsegments >= 128) && (nsegments > ncoords_cur / 64);
    if (segment_major) {
        ffa_complex_iter_segment_freq(fold_in, fold_out, coords, ncoords_cur,
                                      ncoords_prev, nsegments, nbins_f, nbins,
                                      nthreads);
    } else {
        ffa_complex_iter_standard_freq(fold_in, fold_out, coords, ncoords_cur,
                                       ncoords_prev, nsegments, nbins_f, nbins,
                                       nthreads);
    }
}

void shift_add_linear_batch(const float* __restrict__ folds_tree,
                            const SizeType* __restrict__ indices_tree,
                            const float* __restrict__ folds_ffa,
                            const SizeType* __restrict__ indices_ffa,
                            const float* __restrict__ phase_shift,
                            float* __restrict__ folds_out,
                            float const* __restrict__ temp_buffer,
                            SizeType nbins,
                            SizeType n_leaves,
                            SizeType physical_start_idx,
                            SizeType capacity) noexcept {
    (void)temp_buffer;
    const auto total_size = 2 * nbins;
    for (SizeType ileaf = 0; ileaf < n_leaves; ++ileaf) {
        const uint32_t tree_idx_logical =
            indices_tree[ileaf] + physical_start_idx;
        const uint32_t tree_idx_physical = tree_idx_logical < capacity
                                               ? tree_idx_logical
                                               : tree_idx_logical - capacity;
        // Get restrict pointers for current batch item
        const float* __restrict__ data_tree =
            folds_tree + (tree_idx_physical * total_size);
        const float* __restrict__ data_ffa =
            folds_ffa + (indices_ffa[ileaf] * total_size);
        float* __restrict__ data_out = folds_out + (ileaf * total_size);
        shift_add_linear_with_buffer(data_tree, data_ffa, phase_shift[ileaf],
                                     data_out, nbins);
    }
}

void shift_add_linear_complex_batch(const ComplexType* __restrict__ folds_tree,
                                    const SizeType* __restrict__ indices_tree,
                                    const ComplexType* __restrict__ folds_ffa,
                                    const SizeType* __restrict__ indices_ffa,
                                    const float* __restrict__ phase_shift,
                                    ComplexType* __restrict__ folds_out,
                                    SizeType nbins_f,
                                    SizeType nbins,
                                    SizeType n_leaves,
                                    SizeType physical_start_idx,
                                    SizeType capacity) noexcept {
    const auto total_size = 2 * nbins_f;
    for (SizeType ileaf = 0; ileaf < n_leaves; ++ileaf) {
        const uint32_t tree_idx_logical =
            indices_tree[ileaf] + physical_start_idx;
        const uint32_t tree_idx_physical = tree_idx_logical < capacity
                                               ? tree_idx_logical
                                               : tree_idx_logical - capacity;
        // Get restrict pointers for current batch item
        const ComplexType* __restrict__ data_tree =
            folds_tree + (tree_idx_physical * total_size);
        const ComplexType* __restrict__ data_ffa =
            folds_ffa + (indices_ffa[ileaf] * total_size);
        ComplexType* __restrict__ data_out = folds_out + (ileaf * total_size);
        shift_add_complex_recurrence_linear(
            data_tree, data_ffa, phase_shift[ileaf], data_out, nbins_f, nbins);
    }
}

// Overwrite folds_tree
// indices_ffa and phase_shift are stored with n_segments as first dimension
// Second dimension is n_leaves, but batched, so call to this function
// should be in the same loop as the one that computes the phase shifts
void shift_add_ascend_linear_batch(const float* __restrict__ folds_ffa,
                                   const SizeType* __restrict__ indices_segment,
                                   const SizeType* __restrict__ indices_ffa,
                                   const float* __restrict__ phase_shift,
                                   float* __restrict__ folds_tree,
                                   float const* __restrict__ temp_buffer,
                                   SizeType nbins,
                                   SizeType n_coords_init,
                                   SizeType n_leaves,
                                   SizeType n_segments) noexcept {
    (void)temp_buffer;
    const auto total_size = 2 * nbins;
    for (SizeType ileaf = 0; ileaf < n_leaves; ++ileaf) {
        float* __restrict__ data_tree = folds_tree + (ileaf * total_size);
        // Fill it with zero to avoid accumulation of garbage values
        std::fill(data_tree, data_tree + total_size, 0.0F);
        for (SizeType iseg = 0; iseg < n_segments; ++iseg) {
            const auto segment_idx     = indices_segment[iseg];
            const auto phase_shift_seg = phase_shift[(iseg * n_leaves) + ileaf];
            const auto ffa_idx_seg     = indices_ffa[(iseg * n_leaves) + ileaf];
            const float* __restrict__ data_ffa =
                folds_ffa + (segment_idx * n_coords_init * total_size) +
                (ffa_idx_seg * total_size);
            shift_add_linear_with_buffer_inplace(data_ffa, phase_shift_seg,
                                                 data_tree, nbins);
        }
    }
}

void shift_add_ascend_linear_complex_batch(
    const ComplexType* __restrict__ folds_ffa,
    const SizeType* __restrict__ indices_segment,
    const SizeType* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexType* __restrict__ folds_tree,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments) noexcept {
    const auto total_size = 2 * nbins_f;
    for (SizeType ileaf = 0; ileaf < n_leaves; ++ileaf) {
        ComplexType* __restrict__ data_tree = folds_tree + (ileaf * total_size);
        // Fill it with zero to avoid accumulation of garbage values
        std::fill(data_tree, data_tree + total_size, ComplexType{0.0F, 0.0F});
        for (SizeType iseg = 0; iseg < n_segments; ++iseg) {
            const auto segment_idx     = indices_segment[iseg];
            const auto phase_shift_seg = phase_shift[(iseg * n_leaves) + ileaf];
            const auto ffa_idx_seg     = indices_ffa[(iseg * n_leaves) + ileaf];
            const ComplexType* __restrict__ data_ffa =
                folds_ffa + (segment_idx * n_coords_init * total_size) +
                (ffa_idx_seg * total_size);
            shift_add_complex_recurrence_linear_inplace(
                data_ffa, phase_shift_seg, data_tree, nbins_f, nbins);
        }
    }
}

namespace {

using ConeClock = std::chrono::steady_clock;

[[nodiscard]] double cone_seconds(ConeClock::time_point start) noexcept {
    return std::chrono::duration<double>(ConeClock::now() - start).count();
}

void cone_execute_tile(float* scratch_a,
                       float* scratch_b,
                       const float* level_in,
                       const double* prefix_e,
                       const double* prefix_v,
                       const PhaseRun* runs,
                       const SizeType* run_offsets,
                       SizeType prefix_stride,
                       float* level_out,
                       const coord::FFACoordFreq* const* coords_levels,
                       const SizeType* ncoords,
                       SizeType group,
                       SizeType coord_begin,
                       SizeType coord_end,
                       SizeType nbins,
                       SizeType k_levels,
                       ConeScoreFn score_fn,
                       void* score_ctx,
                       bool time_it,
                       double* brute_s,
                       double* merge_s,
                       double* score_s,
                       [[maybe_unused]] const double* p4_e = nullptr,
                       [[maybe_unused]] const double* p4_v = nullptr) {
    const SizeType fold_stride = 2 * nbins;
    std::array<CoordRange, 17> ranges{};
    tile_input_ranges(coords_levels, k_levels, coord_begin, coord_end,
                      ranges.data());

    float* cur            = scratch_a;
    float* next           = scratch_b;
    const SizeType nseg0  = SizeType{1} << k_levels;
    const SizeType width0 = ranges[0].hi - ranges[0].lo;
    const auto level_start =
        time_it ? ConeClock::now() : ConeClock::time_point{};

    if (level_in == nullptr) {
        SizeType iseg = 0;
#if defined(__AVX2__)
        if (p4_e != nullptr && p4_v != nullptr && scratch_b != nullptr) {
            for (; iseg + 4 <= nseg0; iseg += 4) {
                brute_fold_runs_range_4seg(p4_e + (iseg * prefix_stride),
                                           p4_v + (iseg * prefix_stride),
                                           cur + (iseg * width0 * fold_stride),
                                           scratch_b, width0 * fold_stride,
                                           runs, run_offsets, ranges[0].lo,
                                           ranges[0].hi, nbins);
            }
        }
#endif
        for (; iseg < nseg0; ++iseg) {
            brute_fold_runs_range(prefix_e + (iseg * prefix_stride),
                                  prefix_v + (iseg * prefix_stride),
                                  cur + (iseg * width0 * fold_stride), runs,
                                  run_offsets, ranges[0].lo, ranges[0].hi,
                                  nbins);
        }
        if (time_it) {
            *brute_s += cone_seconds(level_start);
        }
    } else {
        const SizeType in_seg_stride = ncoords[0] * fold_stride;
        const SizeType nbytes        = width0 * fold_stride * sizeof(float);
        for (SizeType iseg = 0; iseg < nseg0; ++iseg) {
            const SizeType global_seg = (group * nseg0) + iseg;
            const float* src = level_in + (global_seg * in_seg_stride) +
                               (ranges[0].lo * fold_stride);
            std::memcpy(cur + (iseg * width0 * fold_stride), src, nbytes);
        }
        if (time_it) {
            *merge_s += cone_seconds(level_start);
        }
    }

    auto const merge_start =
        time_it ? ConeClock::now() : ConeClock::time_point{};
    for (SizeType level = 1; level <= k_levels; ++level) {
        const SizeType nseg_in    = SizeType{1} << (k_levels - (level - 1));
        const SizeType nseg_out   = nseg_in / 2;
        const SizeType width_prev = ranges[level - 1].hi - ranges[level - 1].lo;
        const SizeType width_cur  = ranges[level].hi - ranges[level].lo;
        const SizeType seg_prev   = width_prev * fold_stride;
        const SizeType seg_out    = width_cur * fold_stride;
        const auto* coords        = coords_levels[level];
        const SizeType base       = ranges[level].lo;
        const SizeType prev_base  = ranges[level - 1].lo;
        for (SizeType iseg = 0; iseg < nseg_out; ++iseg) {
            const float* tail = cur + ((iseg * 2) * seg_prev);
            const float* head = cur + (((iseg * 2) + 1) * seg_prev);
            float* out        = next + (iseg * seg_out);
            for (SizeType local = 0; local < width_cur; ++local) {
                const SizeType global = base + local;
                const SizeType prev =
                    static_cast<SizeType>(coords[global].idx) - prev_base;
                shift_add_linear_with_buffer(
                    tail + (prev * fold_stride), head + (prev * fold_stride),
                    coords[global].shift, out + (local * fold_stride), nbins);
            }
        }
        std::swap(cur, next);
    }
    if (time_it) {
        *merge_s += cone_seconds(merge_start);
    }

    const SizeType width_k = ranges[k_levels].hi - ranges[k_levels].lo;
    if (level_out != nullptr) {
        const SizeType out_seg_stride = ncoords[k_levels] * fold_stride;
        float* dst                    = level_out + (group * out_seg_stride) +
                                        (ranges[k_levels].lo * fold_stride);
        std::memcpy(dst, cur, width_k * fold_stride * sizeof(float));
    }
    if (score_fn != nullptr) {
        const auto score_start =
            time_it ? ConeClock::now() : ConeClock::time_point{};
        score_fn(cur, ranges[k_levels].lo, width_k, nbins, score_ctx);
        if (time_it) {
            *score_s += cone_seconds(score_start);
        }
    }
}

} // namespace

SizeType
cone_band_working_floats(const coord::FFACoordFreq* const* coords_levels,
                         const SizeType* ncoords,
                         SizeType k_levels,
                         SizeType tile_coords,
                         SizeType nbins) noexcept {
    if (k_levels == 0 || k_levels > 16 || tile_coords == 0 || nbins == 0 ||
        ncoords == nullptr || coords_levels == nullptr) {
        return 0;
    }
    const SizeType top = std::min(tile_coords, ncoords[k_levels]);
    if (top == 0) {
        return 0;
    }
    std::array<SizeType, 17> widths{};
    widths[k_levels] = top;
    for (SizeType level = k_levels; level >= 1; --level) {
        const auto* coords = coords_levels[level];
        const SizeType n   = ncoords[level];
        if (coords == nullptr || n == 0 || !coords_idx_monotone(coords, n)) {
            return 0;
        }
        widths[level - 1] = max_idx_span(coords, n, widths[level]);
        if (widths[level - 1] == 0) {
            return 0;
        }
    }
    SizeType floats = 0;
    for (SizeType level = 0; level <= k_levels; ++level) {
        const SizeType nseg = SizeType{1} << (k_levels - level);
        floats = std::max(floats, nseg * widths[level] * 2 * nbins);
    }
    return floats;
}

void ffa_cone_band_freq(const float* level_in,
                        const float* ts_e,
                        const float* ts_v,
                        const PhaseRun* runs,
                        const SizeType* run_offsets,
                        SizeType segment_len,
                        float* level_out,
                        const coord::FFACoordFreq* const* coords_levels,
                        const SizeType* ncoords,
                        SizeType nsegments_in,
                        SizeType nbins,
                        SizeType k_levels,
                        SizeType tile_coords,
                        ConeScoreFn score_fn,
                        void* score_ctx,
                        ConeBandThreadSeconds* thread_seconds,
                        int nthreads) {
    if (k_levels == 0 || k_levels > 16 || tile_coords == 0 ||
        nsegments_in == 0 || ncoords == nullptr) {
        return;
    }
    const SizeType nseg_group = SizeType{1} << k_levels;
    if ((nsegments_in % nseg_group) != 0) {
        return;
    }
    nthreads                   = std::clamp(nthreads, 1, omp_get_max_threads());
    const SizeType nseg_out    = nsegments_in >> k_levels;
    const SizeType ncoords_top = ncoords[k_levels];
    const SizeType scratch_floats = cone_band_working_floats(
        coords_levels, ncoords, k_levels, tile_coords, nbins);
    if (scratch_floats == 0 || ncoords_top == 0 || nseg_out == 0) {
        return;
    }
    const bool bottom            = level_in == nullptr;
    const bool parallel_groups   = std::cmp_greater_equal(nseg_out, nthreads);
    const bool time_it           = thread_seconds != nullptr;
    const SizeType prefix_stride = segment_len + 1;

    std::vector<double> shared_prefix_e;
    std::vector<double> shared_prefix_v;
    // NOLINTBEGIN(misc-const-correctness): resized and written under AVX2
    std::vector<double> shared_p4_e;
    std::vector<double> shared_p4_v;
    // NOLINTEND(misc-const-correctness)
    if (bottom && !parallel_groups) {
        const auto prefix_start = ConeClock::now();
        shared_prefix_e.resize(nsegments_in * prefix_stride);
        shared_prefix_v.resize(nsegments_in * prefix_stride);
#if defined(__AVX2__)
        shared_p4_e.resize(nsegments_in * prefix_stride);
        shared_p4_v.resize(nsegments_in * prefix_stride);
#endif
#pragma omp parallel for schedule(static) num_threads(nthreads) default(none)  \
    shared(ts_e, ts_v, shared_prefix_e, shared_prefix_v, nsegments_in,         \
               segment_len, prefix_stride)
        for (SizeType iseg = 0; iseg < nsegments_in; ++iseg) {
            fill_segment_prefix(ts_e + (iseg * segment_len),
                                shared_prefix_e.data() + (iseg * prefix_stride),
                                segment_len);
            fill_segment_prefix(ts_v + (iseg * segment_len),
                                shared_prefix_v.data() + (iseg * prefix_stride),
                                segment_len);
        }
#if defined(__AVX2__)
        const SizeType n_prefix = segment_len + 1;
        const SizeType nbatches = nsegments_in / 4;
#pragma omp parallel for schedule(static) num_threads(nthreads) default(none)  \
    shared(shared_prefix_e, shared_prefix_v, shared_p4_e, shared_p4_v,         \
               nbatches, prefix_stride, n_prefix)
        for (SizeType ibatch = 0; ibatch < nbatches; ++ibatch) {
            const SizeType iseg = ibatch * 4;
            const double* pe0 =
                shared_prefix_e.data() + (iseg + 0) * prefix_stride;
            const double* pe1 =
                shared_prefix_e.data() + (iseg + 1) * prefix_stride;
            const double* pe2 =
                shared_prefix_e.data() + (iseg + 2) * prefix_stride;
            const double* pe3 =
                shared_prefix_e.data() + (iseg + 3) * prefix_stride;
            double* dst_e = shared_p4_e.data() + (iseg * prefix_stride);

            const double* pv0 =
                shared_prefix_v.data() + (iseg + 0) * prefix_stride;
            const double* pv1 =
                shared_prefix_v.data() + (iseg + 1) * prefix_stride;
            const double* pv2 =
                shared_prefix_v.data() + (iseg + 2) * prefix_stride;
            const double* pv3 =
                shared_prefix_v.data() + (iseg + 3) * prefix_stride;
            double* dst_v = shared_p4_v.data() + (iseg * prefix_stride);

            for (SizeType i = 0; i < n_prefix; ++i) {
                dst_e[i * 4 + 0] = pe0[i];
                dst_e[i * 4 + 1] = pe1[i];
                dst_e[i * 4 + 2] = pe2[i];
                dst_e[i * 4 + 3] = pe3[i];

                dst_v[i * 4 + 0] = pv0[i];
                dst_v[i * 4 + 1] = pv1[i];
                dst_v[i * 4 + 2] = pv2[i];
                dst_v[i * 4 + 3] = pv3[i];
            }
        }
#endif
        if (time_it) {
            thread_seconds->prefix_wall += cone_seconds(prefix_start);
        }
    }

#pragma omp parallel num_threads(nthreads) default(none) shared(               \
        level_in, ts_e, ts_v, runs, run_offsets, level_out, coords_levels,     \
            ncoords, nbins, k_levels, tile_coords, score_fn, score_ctx,        \
            thread_seconds, nseg_group, nseg_out, ncoords_top, scratch_floats, \
            bottom, parallel_groups, time_it, prefix_stride, segment_len,      \
            shared_prefix_e, shared_prefix_v, shared_p4_e, shared_p4_v)
    {
        double brute_local = 0.0;
        double merge_local = 0.0;
        double score_local = 0.0;
        std::vector<float> scratch_a(scratch_floats);
        std::vector<float> scratch_b(scratch_floats);
        std::vector<double> prefix_e;
        std::vector<double> prefix_v;
        std::vector<double> p4_e;
        std::vector<double> p4_v;
        if (bottom && parallel_groups) {
            prefix_e.resize(nseg_group * prefix_stride);
            prefix_v.resize(nseg_group * prefix_stride);
#if defined(__AVX2__)
            p4_e.resize(nseg_group * prefix_stride);
            p4_v.resize(nseg_group * prefix_stride);
#endif
        }

        if (parallel_groups) {
#pragma omp for schedule(static)
            for (SizeType group = 0; group < nseg_out; ++group) {
                if (bottom) {
                    const auto prefix_start =
                        time_it ? ConeClock::now() : ConeClock::time_point{};
                    const SizeType seg0 = group * nseg_group;
                    for (SizeType iseg = 0; iseg < nseg_group; ++iseg) {
                        const SizeType global_seg = seg0 + iseg;
                        fill_segment_prefix(ts_e + (global_seg * segment_len),
                                            prefix_e.data() +
                                                (iseg * prefix_stride),
                                            segment_len);
                        fill_segment_prefix(ts_v + (global_seg * segment_len),
                                            prefix_v.data() +
                                                (iseg * prefix_stride),
                                            segment_len);
                    }
#if defined(__AVX2__)
                    const SizeType n_prefix = segment_len + 1;
                    for (SizeType iseg = 0; iseg + 4 <= nseg_group; iseg += 4) {
                        const double* pe0 =
                            prefix_e.data() + (iseg + 0) * prefix_stride;
                        const double* pe1 =
                            prefix_e.data() + (iseg + 1) * prefix_stride;
                        const double* pe2 =
                            prefix_e.data() + (iseg + 2) * prefix_stride;
                        const double* pe3 =
                            prefix_e.data() + (iseg + 3) * prefix_stride;
                        double* dst_e = p4_e.data() + (iseg * prefix_stride);

                        const double* pv0 =
                            prefix_v.data() + (iseg + 0) * prefix_stride;
                        const double* pv1 =
                            prefix_v.data() + (iseg + 1) * prefix_stride;
                        const double* pv2 =
                            prefix_v.data() + (iseg + 2) * prefix_stride;
                        const double* pv3 =
                            prefix_v.data() + (iseg + 3) * prefix_stride;
                        double* dst_v = p4_v.data() + (iseg * prefix_stride);

                        for (SizeType i = 0; i < n_prefix; ++i) {
                            dst_e[i * 4 + 0] = pe0[i];
                            dst_e[i * 4 + 1] = pe1[i];
                            dst_e[i * 4 + 2] = pe2[i];
                            dst_e[i * 4 + 3] = pe3[i];

                            dst_v[i * 4 + 0] = pv0[i];
                            dst_v[i * 4 + 1] = pv1[i];
                            dst_v[i * 4 + 2] = pv2[i];
                            dst_v[i * 4 + 3] = pv3[i];
                        }
                    }
#endif
                    if (time_it) {
                        brute_local += cone_seconds(prefix_start);
                    }
                }
                for (SizeType coord0 = 0; coord0 < ncoords_top;
                     coord0 += tile_coords) {
                    const SizeType coord1 =
                        std::min(coord0 + tile_coords, ncoords_top);
                    cone_execute_tile(
                        scratch_a.data(), scratch_b.data(), level_in,
                        bottom ? prefix_e.data() : nullptr,
                        bottom ? prefix_v.data() : nullptr, runs, run_offsets,
                        prefix_stride, level_out, coords_levels, ncoords, group,
                        coord0, coord1, nbins, k_levels, score_fn, score_ctx,
                        time_it, &brute_local, &merge_local, &score_local,
                        bottom ? p4_e.data() : nullptr,
                        bottom ? p4_v.data() : nullptr);
                }
            }
        } else {
#pragma omp for schedule(static)
            for (SizeType coord0 = 0; coord0 < ncoords_top;
                 coord0 += tile_coords) {
                const SizeType coord1 =
                    std::min(coord0 + tile_coords, ncoords_top);
                for (SizeType group = 0; group < nseg_out; ++group) {
                    const double* pe  = nullptr;
                    const double* pv  = nullptr;
                    const double* p4e = nullptr;
                    const double* p4v = nullptr;
                    if (bottom) {
                        const SizeType off = group * nseg_group * prefix_stride;
                        pe                 = shared_prefix_e.data() + off;
                        pv                 = shared_prefix_v.data() + off;
#if defined(__AVX2__)
                        p4e = shared_p4_e.data() + off;
                        p4v = shared_p4_v.data() + off;
#endif
                    }
                    cone_execute_tile(
                        scratch_a.data(), scratch_b.data(), level_in, pe, pv,
                        runs, run_offsets, prefix_stride, level_out,
                        coords_levels, ncoords, group, coord0, coord1, nbins,
                        k_levels, score_fn, score_ctx, time_it, &brute_local,
                        &merge_local, &score_local, p4e, p4v);
                }
            }
        }

        if (time_it) {
#pragma omp critical
            {
                thread_seconds->brute += brute_local;
                thread_seconds->merge += merge_local;
                thread_seconds->score += score_local;
            }
        }
    }
}

} // namespace loki::core
