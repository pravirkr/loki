#include "lib/core/kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <xsimd/xsimd.hpp>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

#include "lib/algorithms/planner_memory.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"

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

/// out[j] = tail[j] + head[j] for j in [0, len).
inline void add_run(const float* __restrict__ tail,
                    const float* __restrict__ head,
                    float* __restrict__ out,
                    SizeType len) noexcept {
#pragma omp simd
    for (SizeType j = 0; j < len; ++j) {
        out[j] = tail[j] + head[j];
    }
}

/**
 * @brief Rotate `tail` right by `shift_tail` and `head` right by
 * `shift_head`, and add them, for one channel.
 *
 * out[j] = tail[(j - shift_tail) mod nbins] + head[(j - shift_head) mod
 * nbins]. The two shifts split [0, nbins) into at most three runs in which
 * both sources are contiguous, so each run is one vectorised add with no
 * temporary buffer.
 */
inline void shift_add_binary_channel(const float* __restrict__ tail,
                                     SizeType shift_tail,
                                     const float* __restrict__ head,
                                     SizeType shift_head,
                                     float* __restrict__ out,
                                     SizeType nbins) noexcept {
    const SizeType lo = std::min(shift_tail, shift_head);
    const SizeType hi = std::max(shift_tail, shift_head);
    // [0, lo): both sources wrapped.
    add_run(tail + (nbins - shift_tail), head + (nbins - shift_head), out, lo);
    // [lo, hi): the smaller shift is in order, the larger one still wraps.
    if (shift_tail <= shift_head) {
        add_run(tail + (lo - shift_tail), head + (lo + nbins - shift_head),
                out + lo, hi - lo);
    } else {
        add_run(tail + (lo + nbins - shift_tail), head + (lo - shift_head),
                out + lo, hi - lo);
    }
    // [hi, nbins): both sources in order.
    add_run(tail + (hi - shift_tail), head + (hi - shift_head), out + hi,
            nbins - hi);
}

/**
 * @brief Rotate and add two (e, v) profiles: out = rot(tail) + rot(head).
 *
 * Same sums as rotating each profile into a buffer and adding (float addition
 * is commutative), in a single pass.
 */
void shift_add_binary_direct(const float* __restrict__ data_tail,
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
    shift_add_binary_channel(data_tail, shift_tail, data_head, shift_head, out,
                             nbins);
    shift_add_binary_channel(data_tail + nbins, shift_tail, data_head + nbins,
                             shift_head, out + nbins, nbins);
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

using MergeBatch = xsimd::batch<ComplexType>;
/// Harmonics per SIMD block of the complex merges.
constexpr SizeType kMergeWidth = MergeBatch::size;

/**
 * @brief Phase table of one complex merge, for the phase step
 * d = exp(-2 pi i shift / nbins) between harmonics.
 *
 * The merges build the phase of harmonic k as (d^W)^b d^j, k = W b + j, W the
 * SIMD width: the in-block phases d^0 .. d^(W-1) (split re / im, one SIMD
 * load each) and the block step d^W are computed in double from d and rounded
 * to float once; the blocks then advance by one float product each. Built
 * from a float d by float products, d^W carried about W roundings into every
 * block: the phase drifted to ~2e-5 at k = 512, ~1e-6 now (B10). The tables
 * are filled for a block of coordinates before their merges, so this work
 * stays off the merges' dependency chains.
 */
struct alignas(64) PhaseStep {
    std::array<float, kMergeWidth> re;
    std::array<float, kMergeWidth> im;
    ComplexType block; // d^W
    ComplexType step;  // d, for the scalar remainder
};

static_assert(sizeof(PhaseStep) <= algorithms::detail::kMaxPhaseStepBytes,
              "the memory model's phase-step size is too small");

/// Sets `out` for the phase step d = (d_re, d_im).
[[gnu::always_inline]] inline void
set_phase_step(PhaseStep& out, double d_re, double d_im) noexcept {
    static_assert(std::has_single_bit(kMergeWidth),
                  "kMergeWidth must be a power of two");
    // (re, im) *= (b_re, b_im), explicit FMA order.
    const auto cmul = [](double& re, double& im, double b_re, double b_im) {
        const double r = std::fma(re, b_re, -(im * b_im));
        im             = std::fma(re, b_im, im * b_re);
        re             = r;
    };
    out.step = ComplexType(static_cast<float>(d_re), static_cast<float>(d_im));
#if defined(__AVX512F__)
    static_assert(kMergeWidth == 16, "AVX-512 merges use 16 complex lanes");
    // d^(2^i) for i < 5.
    std::array<double, 5> p_re{d_re};
    std::array<double, 5> p_im{d_im};
    for (SizeType i = 1; i < 5; ++i) {
        p_re[i] = p_re[i - 1];
        p_im[i] = p_im[i - 1];
        cmul(p_re[i], p_im[i], p_re[i - 1], p_im[i - 1]);
    }
    const auto vmul = [](__m512d& re, __m512d& im, __m512d b_re, __m512d b_im) {
        const __m512d r = _mm512_fmsub_pd(re, b_re, _mm512_mul_pd(im, b_im));
        im              = _mm512_fmadd_pd(re, b_im, _mm512_mul_pd(im, b_re));
        re              = r;
    };
    // Lane j < 8: the product of d^(2^i) over the set bits i of j.
    const __m512d one  = _mm512_set1_pd(1.0);
    const __m512d zero = _mm512_setzero_pd();
    __m512d re = _mm512_mask_blend_pd(0xAA, one, _mm512_set1_pd(p_re[0]));
    __m512d im = _mm512_mask_blend_pd(0xAA, zero, _mm512_set1_pd(p_im[0]));
    vmul(re, im, _mm512_mask_blend_pd(0xCC, one, _mm512_set1_pd(p_re[1])),
         _mm512_mask_blend_pd(0xCC, zero, _mm512_set1_pd(p_im[1])));
    vmul(re, im, _mm512_mask_blend_pd(0xF0, one, _mm512_set1_pd(p_re[2])),
         _mm512_mask_blend_pd(0xF0, zero, _mm512_set1_pd(p_im[2])));
    // Lanes 8 .. 15: times d^8.
    __m512d hre = re;
    __m512d him = im;
    vmul(hre, him, _mm512_set1_pd(p_re[3]), _mm512_set1_pd(p_im[3]));
    _mm512_store_ps(out.re.data(), _mm512_insertf32x8(_mm512_castps256_ps512(
                                                          _mm512_cvtpd_ps(re)),
                                                      _mm512_cvtpd_ps(hre), 1));
    _mm512_store_ps(out.im.data(), _mm512_insertf32x8(_mm512_castps256_ps512(
                                                          _mm512_cvtpd_ps(im)),
                                                      _mm512_cvtpd_ps(him), 1));
    out.block =
        ComplexType(static_cast<float>(p_re[4]), static_cast<float>(p_im[4]));
#else
    // d^(i + h) = d^i d^h for h = 1, 2, 4, ...: a chain of log2(W) products.
    std::array<double, kMergeWidth> re{};
    std::array<double, kMergeWidth> im{};
    re[0]       = 1.0;
    double h_re = d_re;
    double h_im = d_im;
    for (SizeType h = 1; h < kMergeWidth; h *= 2) {
        for (SizeType i = 0; i < h; ++i) {
            re[i + h] = re[i];
            im[i + h] = im[i];
            cmul(re[i + h], im[i + h], h_re, h_im);
        }
        const double sq_re = h_re;
        const double sq_im = h_im;
        cmul(h_re, h_im, sq_re, sq_im);
    }
    for (SizeType i = 0; i < kMergeWidth; ++i) {
        out.re[i] = static_cast<float>(re[i]);
        out.im[i] = static_cast<float>(im[i]);
    }
    out.block = ComplexType(static_cast<float>(h_re), static_cast<float>(h_im));
#endif
}

/// The in-block phases of a phase table, as a SIMD batch.
[[gnu::always_inline]] inline MergeBatch
in_block_phases(const PhaseStep& p) noexcept {
    return {xsimd::batch<float>::load_aligned(p.re.data()),
            xsimd::batch<float>::load_aligned(p.im.data())};
}

/**
 * @brief Phase steps exp(-2 pi i shift_of(i) / nbins), i in [0, n).
 *
 * Scalar double cos/sin per merge call dominated the complex merges (about
 * 500 ns of a 590 ns merge at nbins 64). A separate loop over a block of
 * shifts vectorises the trigonometry, and the steps are reused across
 * segments.
 */
// Coordinates per phase-step block in the complex merges. The fused final
// level uses the same blocks, so the vectorised trigonometry sees the same
// inputs in the same lanes.
constexpr SizeType kMergeStepBlock = 32;

template <typename ShiftOf>
void fill_merge_phase_steps(ShiftOf shift_of,
                            SizeType n,
                            SizeType nbins,
                            PhaseStep* __restrict__ steps) noexcept {
    constexpr SizeType kChunk = 64;
    std::array<double, kChunk> angle{};
    std::array<double, kChunk> re{};
    std::array<double, kChunk> im{};
    for (SizeType i0 = 0; i0 < n; i0 += kChunk) {
        const SizeType m = std::min(kChunk, n - i0);
        for (SizeType i = 0; i < m; ++i) {
            angle[i] = -2.0 * std::numbers::pi * shift_of(i0 + i) /
                       static_cast<float>(nbins);
        }
#pragma omp simd
        for (SizeType i = 0; i < m; ++i) {
            re[i] = std::cos(angle[i]);
            im[i] = std::sin(angle[i]);
        }
        for (SizeType i = 0; i < m; ++i) {
            set_phase_step(steps[i0 + i], re[i], im[i]);
        }
    }
}

/// The caller's phase-step table (FFAWorkspaceCPU::phase_steps) as steps.
/// Called before a parallel region (it checks).
[[nodiscard]] std::span<PhaseStep> as_phase_steps(std::span<std::byte> table,
                                                  SizeType min_steps) {
    error_check::check(reinterpret_cast<std::uintptr_t>(table.data()) %
                               alignof(PhaseStep) ==
                           0,
                       "phase-step table must be 64-byte aligned");
    const SizeType n = table.size() / sizeof(PhaseStep);
    error_check::check_greater_equal(
        n, min_steps, "phase-step table too small for one step block");
    return {reinterpret_cast<PhaseStep*>(table.data()), n};
}

/// Leaves per phase-step block of the EP shift-adds (one fused scoring tile).
constexpr SizeType kEPStepBlock = 16;

/**
 * @brief Phase steps of shifts[i] for i in [0, m), m <= kEPStepBlock.
 *
 * The trigonometry always runs over the full block (unused lanes padded with
 * zero shifts), so it vectorises and each step is independent of m and of
 * the leaf's position in the block: the fused and unfused EP paths, which
 * cut the leaves differently, produce identical steps.
 */
void ep_phase_steps(const float* __restrict__ shifts,
                    SizeType m,
                    SizeType nbins,
                    PhaseStep* __restrict__ steps) noexcept {
    std::array<double, kEPStepBlock> angle{};
    std::array<double, kEPStepBlock> re{};
    std::array<double, kEPStepBlock> im{};
    for (SizeType i = 0; i < m; ++i) {
        angle[i] =
            -2.0 * std::numbers::pi * shifts[i] / static_cast<float>(nbins);
    }
#pragma omp simd
    for (SizeType i = 0; i < kEPStepBlock; ++i) {
        re[i] = std::cos(angle[i]);
        im[i] = std::sin(angle[i]);
    }
    for (SizeType i = 0; i < m; ++i) {
        set_phase_step(steps[i], re[i], im[i]);
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
void shift_add_complex_recurrence_binary_steps(
    const ComplexType* __restrict__ data_tail,
    const PhaseStep& delta_tail,
    const ComplexType* __restrict__ data_head,
    const PhaseStep& delta_head,
    ComplexType* __restrict__ out,
    SizeType nbins_f) noexcept {
    using BatchType                  = xsimd::batch<ComplexType>;
    static constexpr auto kBatchSize = BatchType::size;

    // Phase steps within a SIMD block, [d^0, ..., d^(W-1)], and between
    // blocks, d^W.
    const BatchType delta_vec_tail     = in_block_phases(delta_tail);
    const BatchType delta_vec_head     = in_block_phases(delta_head);
    const ComplexType delta_block_tail = delta_tail.block;
    const ComplexType delta_block_head = delta_head.block;
    const ComplexType delta_phase_tail = delta_tail.step;
    const ComplexType delta_phase_head = delta_head.step;

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
void shift_add_complex_recurrence_linear_steps(
    const ComplexType* __restrict__ data_tail,
    const ComplexType* __restrict__ data_head,
    const PhaseStep& delta,
    ComplexType* __restrict__ out,
    SizeType nbins_f) noexcept {
    using BatchType                  = xsimd::batch<ComplexType>;
    static constexpr auto kBatchSize = BatchType::size;

    const ComplexType* __restrict__ data_tail_e = data_tail;
    const ComplexType* __restrict__ data_tail_v = data_tail + nbins_f;
    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    // Phase steps within a SIMD block, [d^0, ..., d^(W-1)], and between
    // blocks, d^W.
    const BatchType delta_vec     = in_block_phases(delta);
    const ComplexType delta_block = delta.block;
    const ComplexType delta_phase = delta.step;

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

/// shift_add_complex_recurrence_linear_steps accumulating into `out`:
/// out += head * d^k.
void shift_add_complex_recurrence_linear_inplace_steps(
    const ComplexType* __restrict__ data_head,
    const PhaseStep& delta,
    ComplexType* __restrict__ out,
    SizeType nbins_f) noexcept {
    using BatchType                  = xsimd::batch<ComplexType>;
    static constexpr auto kBatchSize = BatchType::size;

    const ComplexType* __restrict__ data_head_e = data_head;
    const ComplexType* __restrict__ data_head_v = data_head + nbins_f;
    ComplexType* __restrict__ out_e             = out;
    ComplexType* __restrict__ out_v             = out + nbins_f;

    // Phase steps within a SIMD block, [d^0, ..., d^(W-1)], and between
    // blocks, d^W.
    const BatchType delta_vec     = in_block_phases(delta);
    const ComplexType delta_block = delta.block;
    const ComplexType delta_phase = delta.step;

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
                           int nthreads) {
    nthreads = std::max(nthreads, 1);
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
                            int nthreads) {
    nthreads                       = std::max(nthreads, 1);
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
                      int nthreads) {
    nthreads = std::max(nthreads, 1);
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins, fold_stride, seg_prev_stride, seg_out_stride)
    {
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

                    shift_add_binary_direct(fold_tail, coord_cur->shift_tail,
                                            fold_head, coord_cur->shift_head,
                                            fold_sum, nbins);
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
                       int nthreads) {
    nthreads                       = std::max(nthreads, 1);
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins, fold_stride, seg_prev_stride, seg_out_stride)
    {
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

                    shift_add_binary_direct(fold_tail, coord_cur->shift_tail,
                                            fold_head, coord_cur->shift_head,
                                            fold_sum, nbins);
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
                              std::span<PhaseStep> table,
                              int nthreads) {
    nthreads = std::max(nthreads, 1);
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;
    // Few coordinates and many segments: the phase steps of a batch of
    // coordinates (whole step blocks, tail and head halves of the table),
    // reused by every segment.
    const SizeType batch = (table.size() / 2 / kBlockSize) * kBlockSize;
    PhaseStep* __restrict__ steps_tail = table.data();
    PhaseStep* __restrict__ steps_head = table.data() + batch;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride,   \
               steps_tail, steps_head, batch)
    for (SizeType c0 = 0; c0 < ncoords_cur; c0 += batch) {
        const SizeType c1 = std::min(c0 + batch, ncoords_cur);
#pragma omp for schedule(static)
        for (SizeType icoord_block = c0; icoord_block < c1;
             icoord_block += kBlockSize) {
            const SizeType nblock =
                std::min(icoord_block + kBlockSize, c1) - icoord_block;
            const auto* block = coords + icoord_block;
            fill_merge_phase_steps(
                [block](SizeType i) { return block[i].shift_tail; }, nblock,
                nbins, steps_tail + (icoord_block - c0));
            fill_merge_phase_steps(
                [block](SizeType i) { return block[i].shift_head; }, nblock,
                nbins, steps_head + (icoord_block - c0));
        }
#pragma omp for schedule(static)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType icoord = c0; icoord < c1; ++icoord) {
                const auto* __restrict__ coord_cur = &coords[icoord];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_tail) * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_head) * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);
                shift_add_complex_recurrence_binary_steps(
                    &fold_in[tail_offset], steps_tail[icoord - c0],
                    &fold_in[head_offset], steps_head[icoord - c0],
                    &fold_out[out_offset], nbins_f);
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
                               int nthreads) {
    nthreads                       = std::max(nthreads, 1);
    constexpr SizeType kBlockSize  = kMergeStepBlock;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
         icoord_block += kBlockSize) {
        const SizeType nblock =
            std::min(icoord_block + kBlockSize, ncoords_cur) - icoord_block;
        const auto* block = coords + icoord_block;
        // Phase steps of this block, reused by every segment.
        std::array<PhaseStep, kBlockSize> steps_tail;
        std::array<PhaseStep, kBlockSize> steps_head;
        fill_merge_phase_steps(
            [block](SizeType i) { return block[i].shift_tail; }, nblock, nbins,
            steps_tail.data());
        fill_merge_phase_steps(
            [block](SizeType i) { return block[i].shift_head; }, nblock, nbins,
            steps_head.data());
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType ib = 0; ib < nblock; ++ib) {
                const auto* __restrict__ coord_cur = &block[ib];
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_tail) * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) +
                    (static_cast<SizeType>(coord_cur->i_head) * fold_stride);
                const auto out_offset = (iseg * seg_out_stride) +
                                        ((icoord_block + ib) * fold_stride);
                shift_add_complex_recurrence_binary_steps(
                    &fold_in[tail_offset], steps_tail[ib],
                    &fold_in[head_offset], steps_head[ib],
                    &fold_out[out_offset], nbins_f);
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
    std::span<PhaseStep> table,
    int nthreads) {
    nthreads = std::max(nthreads, 1);
    // Process one segment at a time to keep data in cache
    constexpr SizeType kBlockSize  = 32;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;
    // Few coordinates and many segments: the phase steps of a batch of
    // coordinates (whole step blocks), reused by every segment.
    const SizeType batch          = (table.size() / kBlockSize) * kBlockSize;
    PhaseStep* __restrict__ steps = table.data();

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride,   \
               steps, batch)
    for (SizeType c0 = 0; c0 < ncoords_cur; c0 += batch) {
        const SizeType c1 = std::min(c0 + batch, ncoords_cur);
#pragma omp for schedule(static)
        for (SizeType icoord_block = c0; icoord_block < c1;
             icoord_block += kBlockSize) {
            const SizeType nblock =
                std::min(icoord_block + kBlockSize, c1) - icoord_block;
            const auto* block = coords + icoord_block;
            fill_merge_phase_steps(
                [block](SizeType i) { return block[i].shift; }, nblock, nbins,
                steps + (icoord_block - c0));
        }
#pragma omp for schedule(static)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType icoord = c0; icoord < c1; ++icoord) {
                const auto idx = static_cast<SizeType>(coords[icoord].idx);
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) + (idx * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) + (idx * fold_stride);
                const auto out_offset =
                    (iseg * seg_out_stride) + (icoord * fold_stride);
                shift_add_complex_recurrence_linear_steps(
                    &fold_in[tail_offset], &fold_in[head_offset],
                    steps[icoord - c0], &fold_out[out_offset], nbins_f);
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
    int nthreads) {
    nthreads                       = std::max(nthreads, 1);
    constexpr SizeType kBlockSize  = kMergeStepBlock;
    const SizeType fold_stride     = 2 * nbins_f;
    const SizeType seg_prev_stride = ncoords_prev * fold_stride;
    const SizeType seg_out_stride  = ncoords_cur * fold_stride;

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,    \
               nbins_f, nbins, fold_stride, seg_prev_stride, seg_out_stride)
    for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
         icoord_block += kBlockSize) {
        const SizeType nblock =
            std::min(icoord_block + kBlockSize, ncoords_cur) - icoord_block;
        const auto* block = coords + icoord_block;
        // Phase steps of this block, reused by every segment.
        std::array<PhaseStep, kBlockSize> steps;
        fill_merge_phase_steps([block](SizeType i) { return block[i].shift; },
                               nblock, nbins, steps.data());
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            for (SizeType ib = 0; ib < nblock; ++ib) {
                const auto idx = static_cast<SizeType>(block[ib].idx);
                const auto tail_offset =
                    ((iseg * 2) * seg_prev_stride) + (idx * fold_stride);
                const auto head_offset =
                    (((iseg * 2) + 1) * seg_prev_stride) + (idx * fold_stride);
                const auto out_offset = (iseg * seg_out_stride) +
                                        ((icoord_block + ib) * fold_stride);
                shift_add_complex_recurrence_linear_steps(
                    &fold_in[tail_offset], &fold_in[head_offset], steps[ib],
                    &fold_out[out_offset], nbins_f);
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

#if defined(__AVX2__)
/// Prefix sums of 4 consecutive segments, interleaved: dst[4 i + s] is the
/// sum of the first i samples of segment s (the same sequential double sums
/// as fill_segment_prefix).
void fill_segment_prefix_4seg(const float* __restrict__ src,
                              SizeType segment_len,
                              double* __restrict__ dst) noexcept {
    std::array<double, 4> acc{};
    for (SizeType s = 0; s < 4; ++s) {
        dst[s] = 0.0;
    }
    for (SizeType i = 0; i < segment_len; ++i) {
        for (SizeType s = 0; s < 4; ++s) {
            acc[s] += static_cast<double>(src[(s * segment_len) + i]);
            dst[((i + 1) * 4) + s] = acc[s];
        }
    }
}
#endif

void brute_fold_ts(const float* __restrict__ ts_e,
                   const float* __restrict__ ts_v,
                   float* __restrict__ fold,
                   const PhaseRun* __restrict__ runs,
                   const SizeType* __restrict__ run_offsets,
                   SizeType nsegments,
                   SizeType nfreqs,
                   SizeType segment_len,
                   SizeType nbins,
                   int nthreads) {
    nthreads                  = std::max(nthreads, 1);
    const SizeType seg_stride = nfreqs * 2 * nbins;
    // Groups of 4 segments go through the SIMD kernel (bit-exact with the
    // scalar one); the rest one by one.
#if defined(__AVX2__)
    const SizeType ngroups = nsegments / 4;
#else
    const SizeType ngroups = 0;
#endif
#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold, runs, run_offsets, nsegments, nfreqs,             \
               segment_len, nbins, seg_stride, ngroups)
    {
#if defined(__AVX2__)
        std::vector<double> p4_e(4 * (segment_len + 1));
        std::vector<double> p4_v(4 * (segment_len + 1));
        std::vector<float> fold_buf(4 * 2 * nbins);
#pragma omp for schedule(static)
        for (SizeType igroup = 0; igroup < ngroups; ++igroup) {
            const SizeType iseg = igroup * 4;
            fill_segment_prefix_4seg(ts_e + (iseg * segment_len), segment_len,
                                     p4_e.data());
            fill_segment_prefix_4seg(ts_v + (iseg * segment_len), segment_len,
                                     p4_v.data());
            brute_fold_runs_range_4seg(p4_e.data(), p4_v.data(),
                                       fold + (iseg * seg_stride),
                                       fold_buf.data(), seg_stride, runs,
                                       run_offsets, 0, nfreqs, nbins);
        }
#endif
        std::vector<double> prefix_e(segment_len + 1);
        std::vector<double> prefix_v(segment_len + 1);
#pragma omp for schedule(static)
        for (SizeType iseg = ngroups * 4; iseg < nsegments; ++iseg) {
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
                               int nthreads) {
    nthreads                     = std::max(nthreads, 1);
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

namespace {

/// Phasor table of one frequency: row j < nrows holds exp(i m phase_j),
/// m < nbins_f, as (re, im) pairs padded with zeros to row_len floats;
/// phase_j = -2 pi f (j tsamp - t_ref). Double, rounded once.
void fill_brute_phasors(double freq,
                        SizeType nrows,
                        SizeType nbins_f,
                        double tsamp,
                        double t_ref,
                        SizeType row_len,
                        float* __restrict__ table) noexcept {
    const double phase_factor = -2.0 * std::numbers::pi * freq;
    for (SizeType j = 0; j < nrows; ++j) {
        const double phase =
            phase_factor * ((static_cast<double>(j) * tsamp) - t_ref);
        float* row = table + (j * row_len);
        // exp(i m phase) by a double recurrence over m: its error (~m 1e-16)
        // stays far below the float rounding.
        const double w_re = std::cos(phase);
        const double w_im = std::sin(phase);
        double p_re       = 1.0;
        double p_im       = 0.0;
        for (SizeType m = 0; m < nbins_f; ++m) {
            row[2 * m]       = static_cast<float>(p_re);
            row[(2 * m) + 1] = static_cast<float>(p_im);
            const double r   = std::fma(p_re, w_re, -(p_im * w_im));
            p_im             = std::fma(p_re, w_im, p_im * w_re);
            p_re             = r;
        }
        for (SizeType c = 2 * nbins_f; c < row_len; ++c)
            row[c] = 0.0F;
    }
}

/// out[s][c0 + c] (+)= sum_j x[s][j] table[j][c0 + c], j < len, c < NV W,
/// for NS rows; `accumulate` adds to out instead of overwriting it.
template <SizeType NS, SizeType NV>
[[gnu::always_inline]] inline void
brute_gemm_block(const std::array<const float*, NS>& x,
                 const float* __restrict__ table,
                 SizeType segment_len,
                 SizeType row_len,
                 SizeType c0,
                 const std::array<float*, NS>& out,
                 bool accumulate) noexcept {
    using B               = xsimd::batch<float>;
    constexpr SizeType kW = B::size;
    // Two accumulators (even and odd samples): shorter float sums.
    std::array<std::array<B, NV>, NS> acc0;
    std::array<std::array<B, NV>, NS> acc1;
    for (SizeType s = 0; s < NS; ++s) {
        for (SizeType v = 0; v < NV; ++v) {
            acc0[s][v] =
                accumulate ? B::load_aligned(out[s] + c0 + (v * kW)) : B(0.0F);
            acc1[s][v] = B(0.0F);
        }
    }
    SizeType j = 0;
    for (; j + 2 <= segment_len; j += 2) {
        const float* t0 = table + (j * row_len) + c0;
        const float* t1 = t0 + row_len;
        std::array<B, NS> x0;
        std::array<B, NS> x1;
        for (SizeType s = 0; s < NS; ++s) {
            x0[s] = B(x[s][j]);
            x1[s] = B(x[s][j + 1]);
        }
        for (SizeType v = 0; v < NV; ++v) {
            const B a = B::load_aligned(t0 + (v * kW));
            const B b = B::load_aligned(t1 + (v * kW));
            for (SizeType s = 0; s < NS; ++s) {
                acc0[s][v] = xsimd::fma(x0[s], a, acc0[s][v]);
                acc1[s][v] = xsimd::fma(x1[s], b, acc1[s][v]);
            }
        }
    }
    for (; j < segment_len; ++j) {
        const float* t0 = table + (j * row_len) + c0;
        for (SizeType v = 0; v < NV; ++v) {
            const B a = B::load_aligned(t0 + (v * kW));
            for (SizeType s = 0; s < NS; ++s)
                acc0[s][v] = xsimd::fma(B(x[s][j]), a, acc0[s][v]);
        }
    }
    for (SizeType s = 0; s < NS; ++s)
        for (SizeType v = 0; v < NV; ++v)
            (acc0[s][v] + acc1[s][v]).store_aligned(out[s] + c0 + (v * kW));
}

/// Samples per sub-block of the direct Fourier brute fold (one table).
constexpr SizeType kBruteSub = 64;

template <SizeType NS>
void brute_gemm_rows(const std::array<const float*, NS>& x,
                     const float* table,
                     SizeType segment_len,
                     SizeType row_len,
                     const std::array<float*, NS>& out,
                     bool accumulate) noexcept {
    using B               = xsimd::batch<float>;
    constexpr SizeType kW = B::size;
#if defined(__AVX512F__)
    constexpr SizeType kMaxNV = NS >= 4 ? 3 : 5;
#else
    constexpr SizeType kMaxNV = NS >= 4 ? 1 : 2;
#endif
    SizeType c0 = 0;
    for (; c0 + (kMaxNV * kW) <= row_len; c0 += kMaxNV * kW)
        brute_gemm_block<NS, kMaxNV>(x, table, segment_len, row_len, c0, out,
                                     accumulate);
    switch ((row_len - c0) / kW) {
    case 0:
        break;
    case 1:
        brute_gemm_block<NS, 1>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    case 2:
        brute_gemm_block<NS, 2>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    case 3:
        brute_gemm_block<NS, 3>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    case 4:
        brute_gemm_block<NS, 4>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    case 5:
        brute_gemm_block<NS, 5>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    case 6:
        brute_gemm_block<NS, 6>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    default:
        brute_gemm_block<NS, 7>(x, table, segment_len, row_len, c0, out,
                                accumulate);
        break;
    }
}

} // namespace

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
                           int nthreads) {
    // fold[s][m] = sum_j x_s[j] exp(i m phase_j) as GEMMs against phasor
    // tables shared by every segment (each is folded from its own start).
    // The phase is linear in j, so j = B q + r splits it: sub-block q of B
    // samples is a GEMM against one B-row table T[r] = exp(i m phase'_r)
    // (phase' without t_ref: small, in L1), rotated by G[q] = exp(i m
    // phase_(B q)) and summed. A segment of at most B samples is one
    // sub-block with the full phase in T. Pairs of segments (e and v rows)
    // share the table loads.
    nthreads                = std::max(nthreads, 1);
    constexpr SizeType kW   = xsimd::batch<float>::size;
    const SizeType nbins_f  = (nbins / 2) + 1;
    const SizeType row_len  = (((2 * nbins_f) + kW - 1) / kW) * kW;
    const SizeType sub      = std::min(segment_len, kBruteSub);
    const SizeType nsub     = (segment_len + sub - 1) / sub;
    const bool split        = nsub > 1;
    const SizeType t_stride = sub * row_len;
    const SizeType g_stride = nsub * 2 * nbins_f;
    AlignedFloatVec t_tables(nfreqs * t_stride);
    AlignedFloatVec g_tables(split ? nfreqs * g_stride : 0);
    for (SizeType f = 0; f < nfreqs; ++f) {
        fill_brute_phasors(freqs[f], sub, nbins_f, tsamp, split ? 0.0 : t_ref,
                           row_len, t_tables.data() + (f * t_stride));
        if (split) {
            fill_brute_phasors(freqs[f], nsub, nbins_f,
                               tsamp * static_cast<double>(sub), t_ref,
                               2 * nbins_f, g_tables.data() + (f * g_stride));
        }
    }
    const float* t_tab         = t_tables.data();
    const float* g_tab         = g_tables.data();
    const SizeType fold_stride = nfreqs * 2 * nbins_f; // per segment
    const SizeType npairs      = (nsegments + 1) / 2;
#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(ts_e, ts_v, fold, nfreqs, nsegments, segment_len, nbins_f, row_len, \
               sub, nsub, split, t_tab, g_tab, t_stride, g_stride,             \
               fold_stride, npairs)
    {
        // Sub-block results, and their rotated sums (split only).
        AlignedFloatVec y_rows(4 * row_len);
        std::vector<float> acc(split ? 4 * 2 * nbins_f : 0);
        const std::array<float*, 4> y{y_rows.data(), y_rows.data() + row_len,
                                      y_rows.data() + (2 * row_len),
                                      y_rows.data() + (3 * row_len)};
#pragma omp for schedule(static)
        for (SizeType ip = 0; ip < npairs; ++ip) {
            const SizeType s0   = 2 * ip;
            const SizeType nseg = std::min<SizeType>(2, nsegments - s0);
            const float* e0     = ts_e + (s0 * segment_len);
            const float* v0     = ts_v + (s0 * segment_len);
            for (SizeType f = 0; f < nfreqs; ++f) {
                const float* table = t_tab + (f * t_stride);
                std::fill(acc.begin(), acc.end(), 0.0F);
                for (SizeType q = 0; q < nsub; ++q) {
                    const SizeType j0  = q * sub;
                    const SizeType len = std::min(sub, segment_len - j0);
                    if (nseg == 2) {
                        brute_gemm_rows<4>({e0 + j0, v0 + j0,
                                            e0 + segment_len + j0,
                                            v0 + segment_len + j0},
                                           table, len, row_len, y, false);
                    } else {
                        brute_gemm_rows<2>({e0 + j0, v0 + j0}, table, len,
                                           row_len, {y[0], y[1]}, false);
                    }
                    if (!split) {
                        break;
                    }
                    // acc += G[q] * y, complex, per row.
                    const float* g = g_tab + (f * g_stride) + (q * 2 * nbins_f);
                    for (SizeType r = 0; r < 2 * nseg; ++r) {
                        float* a       = acc.data() + (r * 2 * nbins_f);
                        const float* v = y[r];
                        for (SizeType m = 0; m < nbins_f; ++m) {
                            const float gr = g[2 * m];
                            const float gi = g[(2 * m) + 1];
                            const float vr = v[2 * m];
                            const float vi = v[(2 * m) + 1];
                            a[2 * m] += (gr * vr) - (gi * vi);
                            a[(2 * m) + 1] += (gr * vi) + (gi * vr);
                        }
                    }
                }
                for (SizeType s = 0; s < nseg; ++s) {
                    auto* dst = reinterpret_cast<float*>(
                        fold + ((s0 + s) * fold_stride) + (f * 2 * nbins_f));
                    for (SizeType r = 0; r < 2; ++r) {
                        const float* src =
                            split ? acc.data() + (((2 * s) + r) * 2 * nbins_f)
                                  : y[(2 * s) + r];
                        std::copy_n(src, 2 * nbins_f, dst + (r * 2 * nbins_f));
                    }
                }
            }
        }
    }
}

namespace {
// A tile of final coordinates: `nrows` rows of the coordinate grid (all
// parameters but frequency) times `ncols` frequencies from `col_begin`.
struct FinalTile {
    SizeType row_begin;
    SizeType nrows;
    SizeType col_begin;
    SizeType ncols;
};

// Tiles of the final level ordered for parent reuse. Neighbouring accel
// (jerk) rows read the same parent rows of the level before (its grid is
// coarser), and a parent row of a wide chunk is far larger than L2. Rows with
// the same parent rows form a group, and a group's tiles walk the frequency
// axis, so a tile's parents stay in cache across its rows.
template <typename Coord>
std::vector<FinalTile> plan_final_tiles(const Coord* coords,
                                        SizeType ncoords_cur,
                                        SizeType row_len,
                                        SizeType row_len_prev,
                                        SizeType tile_cols) {
    std::vector<FinalTile> tiles;
    if (row_len == 0 || row_len_prev == 0 || ncoords_cur % row_len != 0) {
        row_len      = ncoords_cur;
        row_len_prev = 0;
    }
    const SizeType nrows   = ncoords_cur / row_len;
    const auto parent_rows = [&](SizeType row) {
        if constexpr (std::is_same_v<Coord, coord::FFACoord>) {
            const auto& c = coords[row * row_len];
            return std::pair<SizeType, SizeType>{c.i_tail / row_len_prev,
                                                 c.i_head / row_len_prev};
        } else {
            return std::pair<SizeType, SizeType>{row, row};
        }
    };
    for (SizeType row = 0; row < nrows;) {
        SizeType end = row + 1;
        if (row_len_prev != 0) {
            const auto key = parent_rows(row);
            while (end < nrows && parent_rows(end) == key) {
                ++end;
            }
        }
        for (SizeType col = 0; col < row_len; col += tile_cols) {
            tiles.push_back({.row_begin = row,
                             .nrows     = end - row,
                             .col_begin = col,
                             .ncols     = std::min(tile_cols, row_len - col)});
        }
        row = end;
    }
    return tiles;
}
} // namespace

void ffa_final_scored(const float* __restrict__ fold_in,
                      const coord::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nbins,
                      SizeType row_len,
                      SizeType row_len_prev,
                      SizeType tile_coords,
                      ConeScoreFn score_fn,
                      void* score_ctx,
                      int nthreads) {
    nthreads                           = std::max(nthreads, 1);
    tile_coords                        = std::max<SizeType>(tile_coords, 1);
    const SizeType fold_stride         = 2 * nbins;
    const float* __restrict__ seg_tail = fold_in;
    const float* __restrict__ seg_head = fold_in + (ncoords_prev * fold_stride);
    const std::vector<FinalTile> tiles = plan_final_tiles(
        coords, ncoords_cur, row_len, row_len_prev, tile_coords);
    const SizeType ntiles = tiles.size();
    if (row_len == 0 || ncoords_cur % row_len != 0) {
        row_len = ncoords_cur;
    }

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(seg_tail, seg_head, coords, nbins, tile_coords, score_fn,           \
               score_ctx, fold_stride, ntiles, tiles, row_len)
    {
        utils::UninitVector<float> scratch(tile_coords * fold_stride);
#pragma omp for schedule(static)
        for (SizeType itile = 0; itile < ntiles; ++itile) {
            const FinalTile& tile = tiles[itile];
            for (SizeType row = tile.row_begin;
                 row < tile.row_begin + tile.nrows; ++row) {
                const SizeType begin = (row * row_len) + tile.col_begin;
                for (SizeType i = 0; i < tile.ncols; ++i) {
                    const auto& coord = coords[begin + i];
                    shift_add_binary_direct(
                        seg_tail +
                            (static_cast<SizeType>(coord.i_tail) * fold_stride),
                        coord.shift_tail,
                        seg_head +
                            (static_cast<SizeType>(coord.i_head) * fold_stride),
                        coord.shift_head, scratch.data() + (i * fold_stride),
                        nbins);
                }
                score_fn(scratch.data(), begin, tile.ncols, nbins, score_ctx);
            }
        }
    }
}

namespace {
template <typename Coord, typename MergeOne>
void complex_final_tiles(const Coord* __restrict__ coords,
                         SizeType ncoords_cur,
                         SizeType nbins_f,
                         SizeType row_len,
                         SizeType row_len_prev,
                         SizeType tile_coords,
                         ComplexTileFn tile_fn,
                         void* tile_ctx,
                         int nthreads,
                         MergeOne merge_one) {
    nthreads                           = std::max(nthreads, 1);
    const SizeType fold_stride         = 2 * nbins_f;
    const std::vector<FinalTile> tiles = plan_final_tiles(
        coords, ncoords_cur, row_len, row_len_prev, tile_coords);
    const SizeType ntiles = tiles.size();
    if (row_len == 0 || ncoords_cur % row_len != 0) {
        row_len = ncoords_cur;
    }

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(coords, nbins_f, tile_coords, tile_fn, tile_ctx, fold_stride,       \
               ntiles, tiles, row_len, merge_one)
    {
        utils::UninitVector<ComplexType> scratch(tile_coords * fold_stride);
#pragma omp for schedule(static)
        for (SizeType itile = 0; itile < ntiles; ++itile) {
            const FinalTile& tile = tiles[itile];
            for (SizeType row = tile.row_begin;
                 row < tile.row_begin + tile.nrows; ++row) {
                const SizeType begin = (row * row_len) + tile.col_begin;
                for (SizeType b0 = 0; b0 < tile.ncols; b0 += kMergeStepBlock) {
                    const SizeType nblock =
                        std::min(SizeType{kMergeStepBlock}, tile.ncols - b0);
                    merge_one(coords + begin + b0, nblock,
                              scratch.data() + (b0 * fold_stride));
                }
                tile_fn(scratch.data(), begin, tile.ncols, tile_ctx);
            }
        }
    }
}
} // namespace

void ffa_complex_final_tiles(const ComplexType* __restrict__ fold_in,
                             const coord::FFACoord* __restrict__ coords,
                             SizeType ncoords_cur,
                             SizeType ncoords_prev,
                             SizeType nbins_f,
                             SizeType nbins,
                             SizeType row_len,
                             SizeType row_len_prev,
                             SizeType tile_coords,
                             ComplexTileFn tile_fn,
                             void* tile_ctx,
                             int nthreads) {
    const SizeType fold_stride  = 2 * nbins_f;
    const ComplexType* seg_tail = fold_in;
    const ComplexType* seg_head = fold_in + (ncoords_prev * fold_stride);
    complex_final_tiles(
        coords, ncoords_cur, nbins_f, row_len, row_len_prev, tile_coords,
        tile_fn, tile_ctx, nthreads,
        [=](const coord::FFACoord* block, SizeType nblock, ComplexType* out) {
            std::array<PhaseStep, kMergeStepBlock> steps_tail;
            std::array<PhaseStep, kMergeStepBlock> steps_head;
            fill_merge_phase_steps(
                [block](SizeType i) { return block[i].shift_tail; }, nblock,
                nbins, steps_tail.data());
            fill_merge_phase_steps(
                [block](SizeType i) { return block[i].shift_head; }, nblock,
                nbins, steps_head.data());
            for (SizeType i = 0; i < nblock; ++i) {
                shift_add_complex_recurrence_binary_steps(
                    seg_tail +
                        (static_cast<SizeType>(block[i].i_tail) * fold_stride),
                    steps_tail[i],
                    seg_head +
                        (static_cast<SizeType>(block[i].i_head) * fold_stride),
                    steps_head[i], out + (i * fold_stride), nbins_f);
            }
        });
}

namespace {
// Parent rows of a set of final coordinates: [lo, hi) of i_tail and i_head.
struct ParentSpan {
    SizeType tail_lo;
    SizeType tail_hi;
    SizeType head_lo;
    SizeType head_hi;
};

ParentSpan parent_span(const coord::FFACoord* coords,
                       SizeType row_len,
                       SizeType row_begin,
                       SizeType nrows,
                       SizeType col_begin,
                       SizeType ncols) {
    ParentSpan span{.tail_lo = std::numeric_limits<SizeType>::max(),
                    .tail_hi = 0,
                    .head_lo = std::numeric_limits<SizeType>::max(),
                    .head_hi = 0};
    for (SizeType row = row_begin; row < row_begin + nrows; ++row) {
        const coord::FFACoord* c = coords + (row * row_len) + col_begin;
        for (SizeType i = 0; i < ncols; ++i) {
            span.tail_lo = std::min<SizeType>(span.tail_lo, c[i].i_tail);
            span.tail_hi = std::max<SizeType>(span.tail_hi, c[i].i_tail + 1);
            span.head_lo = std::min<SizeType>(span.head_lo, c[i].i_head);
            span.head_hi = std::max<SizeType>(span.head_hi, c[i].i_head + 1);
        }
    }
    return span;
}

// Visit the sub-tiles of `tile` whose parents fit `cap` rows per segment:
// the whole tile, else row by row, else halves of a row's columns.
template <typename Visit>
void for_each_parent_fit(const coord::FFACoord* coords,
                         SizeType row_len,
                         const FinalTile& tile,
                         SizeType cap,
                         SizeType align,
                         Visit visit) {
    const auto fits = [&](const ParentSpan& sp) {
        const auto rows = [align](SizeType lo, SizeType hi) {
            return (((hi + align - 1) / align) * align) -
                   ((lo / align) * align);
        };
        return rows(sp.tail_lo, sp.tail_hi) <= cap &&
               rows(sp.head_lo, sp.head_hi) <= cap;
    };
    const ParentSpan whole =
        parent_span(coords, row_len, tile.row_begin, tile.nrows, tile.col_begin,
                    tile.ncols);
    if (fits(whole)) {
        visit(tile, whole);
        return;
    }
    for (SizeType row = tile.row_begin; row < tile.row_begin + tile.nrows;
         ++row) {
        // Split the row's columns until each part fits (one column always
        // does: it reads one parent per segment).
        std::vector<FinalTile> parts = {{.row_begin = row,
                                         .nrows     = 1,
                                         .col_begin = tile.col_begin,
                                         .ncols     = tile.ncols}};
        while (!parts.empty()) {
            const FinalTile part = parts.back();
            parts.pop_back();
            const ParentSpan sp = parent_span(coords, row_len, part.row_begin,
                                              1, part.col_begin, part.ncols);
            if (fits(sp) || part.ncols == 1) {
                visit(part, sp);
                continue;
            }
            const SizeType half = ((part.ncols / 2) / align) * align;
            const SizeType left = half == 0 ? part.ncols / 2 : half;
            // Right half first: the stack visits the left half first.
            parts.push_back({.row_begin = row,
                             .nrows     = 1,
                             .col_begin = part.col_begin + left,
                             .ncols     = part.ncols - left});
            parts.push_back({.row_begin = row,
                             .nrows     = 1,
                             .col_begin = part.col_begin,
                             .ncols     = left});
        }
    }
}
} // namespace

void ffa_final2_scored(const float* __restrict__ fold_in,
                       const coord::FFACoord* __restrict__ coords_prev,
                       const coord::FFACoord* __restrict__ coords,
                       SizeType ncoords_cur,
                       SizeType ncoords_prev2,
                       SizeType nbins,
                       SizeType row_len,
                       SizeType row_len_prev,
                       SizeType tile_coords,
                       ConeScoreFn score_fn,
                       void* score_ctx,
                       int nthreads) {
    nthreads                   = std::max(nthreads, 1);
    tile_coords                = std::max<SizeType>(tile_coords, 1);
    const SizeType fold_stride = 2 * nbins;
    // A level-(L-1) parent in scratch: each channel stored twice ([e, e, v,
    // v]), so a rotation of it is one contiguous run.
    const SizeType par_stride          = 4 * nbins;
    const SizeType seg2                = ncoords_prev2 * fold_stride;
    const std::vector<FinalTile> tiles = plan_final_tiles(
        coords, ncoords_cur, row_len, row_len_prev, tile_coords);
    const SizeType ntiles = tiles.size();
    if (row_len == 0 || ncoords_cur % row_len != 0) {
        row_len = ncoords_cur;
    }
    // Parents per segment held in scratch; a tile of a row group reads about
    // half its columns plus the head drift.
    const SizeType cap = tile_coords + 64;

#pragma omp parallel num_threads(nthreads) default(none) shared(               \
        fold_in, coords_prev, coords, nbins, tile_coords, score_fn, score_ctx, \
            fold_stride, par_stride, seg2, ntiles, tiles, row_len, cap)
    {
        utils::UninitVector<float> scratch(tile_coords * fold_stride);
        utils::UninitVector<float> par_tail(cap * par_stride);
        utils::UninitVector<float> par_head(cap * par_stride);
        // Level L-1 segment `seg` (0 tail, 1 head) rows [lo, hi) from level
        // L-2 (segments 2 seg, 2 seg + 1), the arithmetic of ffa_iter.
        const auto fill = [&](SizeType seg, SizeType lo, SizeType hi,
                              float* par) {
            const float* l2_tail = fold_in + ((2 * seg) * seg2);
            const float* l2_head = fold_in + ((2 * seg + 1) * seg2);
            for (SizeType idx = lo; idx < hi; ++idx) {
                const auto& c     = coords_prev[idx];
                const SizeType st = phase_shift_bins(c.shift_tail, nbins);
                const SizeType sh = phase_shift_bins(c.shift_head, nbins);
                const float* t    = l2_tail + (c.i_tail * fold_stride);
                const float* h    = l2_head + (c.i_head * fold_stride);
                float* out        = par + ((idx - lo) * par_stride);
                shift_add_binary_channel(t, st, h, sh, out, nbins);
                shift_add_binary_channel(t + nbins, st, h + nbins, sh,
                                         out + (2 * nbins), nbins);
                std::memcpy(out + nbins, out, nbins * sizeof(float));
                std::memcpy(out + (3 * nbins), out + (2 * nbins),
                            nbins * sizeof(float));
            }
        };
#pragma omp for schedule(static)
        for (SizeType itile = 0; itile < ntiles; ++itile) {
            for_each_parent_fit(
                coords, row_len, tiles[itile], cap, 1,
                [&](const FinalTile& tile, const ParentSpan& sp) {
                    fill(0, sp.tail_lo, sp.tail_hi, par_tail.data());
                    fill(1, sp.head_lo, sp.head_hi, par_head.data());
                    for (SizeType row = tile.row_begin;
                         row < tile.row_begin + tile.nrows; ++row) {
                        const SizeType begin = (row * row_len) + tile.col_begin;
                        for (SizeType i = 0; i < tile.ncols; ++i) {
                            const auto& c = coords[begin + i];
                            const SizeType st =
                                phase_shift_bins(c.shift_tail, nbins);
                            const SizeType sh =
                                phase_shift_bins(c.shift_head, nbins);
                            const float* __restrict__ t =
                                par_tail.data() +
                                ((c.i_tail - sp.tail_lo) * par_stride) +
                                (nbins - st);
                            const float* __restrict__ h =
                                par_head.data() +
                                ((c.i_head - sp.head_lo) * par_stride) +
                                (nbins - sh);
                            float* __restrict__ out =
                                scratch.data() + (i * fold_stride);
#pragma omp simd
                            for (SizeType j = 0; j < nbins; ++j) {
                                out[j] = t[j] + h[j];
                            }
#pragma omp simd
                            for (SizeType j = 0; j < nbins; ++j) {
                                out[nbins + j] =
                                    t[(2 * nbins) + j] + h[(2 * nbins) + j];
                            }
                        }
                        score_fn(scratch.data(), begin, tile.ncols, nbins,
                                 score_ctx);
                    }
                });
        }
    }
}

void ffa_complex_final2_tiles(const ComplexType* __restrict__ fold_in,
                              const coord::FFACoord* __restrict__ coords_prev,
                              const coord::FFACoord* __restrict__ coords,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev,
                              SizeType ncoords_prev2,
                              SizeType nbins_f,
                              SizeType nbins,
                              SizeType row_len,
                              SizeType row_len_prev,
                              SizeType tile_coords,
                              ComplexTileFn tile_fn,
                              void* tile_ctx,
                              int nthreads) {
    nthreads                           = std::max(nthreads, 1);
    const SizeType fold_stride         = 2 * nbins_f;
    const SizeType seg2                = ncoords_prev2 * fold_stride;
    const std::vector<FinalTile> tiles = plan_final_tiles(
        coords, ncoords_cur, row_len, row_len_prev, tile_coords);
    const SizeType ntiles = tiles.size();
    if (row_len == 0 || ncoords_cur % row_len != 0) {
        row_len = ncoords_cur;
    }
    // Level-(L-1) rows are merged in the phase-step blocks of the stored
    // level (multiples of kMergeStepBlock), so the steps are identical.
    const SizeType cap = tile_coords + (2 * kMergeStepBlock) + 64;

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(fold_in, coords_prev, coords, ncoords_prev, nbins_f, nbins,         \
               tile_coords, tile_fn, tile_ctx, fold_stride, seg2, ntiles,      \
               tiles, row_len, cap)
    {
        utils::UninitVector<ComplexType> scratch(tile_coords * fold_stride);
        utils::UninitVector<ComplexType> par_tail(cap * fold_stride);
        utils::UninitVector<ComplexType> par_head(cap * fold_stride);
        std::array<PhaseStep, kMergeStepBlock> steps_tail;
        std::array<PhaseStep, kMergeStepBlock> steps_head;
        // Level L-1 segment `seg` rows [lo, hi) (lo a multiple of
        // kMergeStepBlock).
        const auto fill = [&](SizeType seg, SizeType lo, SizeType hi,
                              ComplexType* par) {
            const ComplexType* l2_tail = fold_in + ((2 * seg) * seg2);
            const ComplexType* l2_head = fold_in + ((2 * seg + 1) * seg2);
            for (SizeType b0 = lo; b0 < hi; b0 += kMergeStepBlock) {
                const SizeType nblock =
                    std::min<SizeType>(kMergeStepBlock, ncoords_prev - b0);
                const coord::FFACoord* block = coords_prev + b0;
                fill_merge_phase_steps(
                    [block](SizeType i) { return block[i].shift_tail; }, nblock,
                    nbins, steps_tail.data());
                fill_merge_phase_steps(
                    [block](SizeType i) { return block[i].shift_head; }, nblock,
                    nbins, steps_head.data());
                for (SizeType i = 0; i < nblock; ++i) {
                    shift_add_complex_recurrence_binary_steps(
                        l2_tail + (static_cast<SizeType>(block[i].i_tail) *
                                   fold_stride),
                        steps_tail[i],
                        l2_head + (static_cast<SizeType>(block[i].i_head) *
                                   fold_stride),
                        steps_head[i], par + ((b0 + i - lo) * fold_stride),
                        nbins_f);
                }
            }
        };
#pragma omp for schedule(static)
        for (SizeType itile = 0; itile < ntiles; ++itile) {
            for_each_parent_fit(
                coords, row_len, tiles[itile], cap, kMergeStepBlock,
                [&](const FinalTile& tile, const ParentSpan& sp) {
                    const SizeType tail_lo =
                        (sp.tail_lo / kMergeStepBlock) * kMergeStepBlock;
                    const SizeType head_lo =
                        (sp.head_lo / kMergeStepBlock) * kMergeStepBlock;
                    fill(0, tail_lo,
                         std::min(((sp.tail_hi + kMergeStepBlock - 1) /
                                   kMergeStepBlock) *
                                      kMergeStepBlock,
                                  ncoords_prev),
                         par_tail.data());
                    fill(1, head_lo,
                         std::min(((sp.head_hi + kMergeStepBlock - 1) /
                                   kMergeStepBlock) *
                                      kMergeStepBlock,
                                  ncoords_prev),
                         par_head.data());
                    for (SizeType row = tile.row_begin;
                         row < tile.row_begin + tile.nrows; ++row) {
                        const SizeType begin = (row * row_len) + tile.col_begin;
                        for (SizeType b0 = 0; b0 < tile.ncols;
                             b0 += kMergeStepBlock) {
                            const SizeType nblock = std::min(
                                SizeType{kMergeStepBlock}, tile.ncols - b0);
                            const coord::FFACoord* block = coords + begin + b0;
                            fill_merge_phase_steps(
                                [block](SizeType i) {
                                    return block[i].shift_tail;
                                },
                                nblock, nbins, steps_tail.data());
                            fill_merge_phase_steps(
                                [block](SizeType i) {
                                    return block[i].shift_head;
                                },
                                nblock, nbins, steps_head.data());
                            for (SizeType i = 0; i < nblock; ++i) {
                                shift_add_complex_recurrence_binary_steps(
                                    par_tail.data() +
                                        ((block[i].i_tail - tail_lo) *
                                         fold_stride),
                                    steps_tail[i],
                                    par_head.data() +
                                        ((block[i].i_head - head_lo) *
                                         fold_stride),
                                    steps_head[i],
                                    scratch.data() + ((b0 + i) * fold_stride),
                                    nbins_f);
                            }
                        }
                        tile_fn(scratch.data(), begin, tile.ncols, tile_ctx);
                    }
                });
        }
    }
}

void ffa_complex_final_tiles_freq(
    const ComplexType* __restrict__ fold_in,
    const coord::FFACoordFreq* __restrict__ coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nbins_f,
    SizeType nbins,
    SizeType tile_coords,
    ComplexTileFn tile_fn,
    void* tile_ctx,
    int nthreads) {
    const SizeType fold_stride  = 2 * nbins_f;
    const ComplexType* seg_tail = fold_in;
    const ComplexType* seg_head = fold_in + (ncoords_prev * fold_stride);
    complex_final_tiles(
        coords, ncoords_cur, nbins_f, /*row_len=*/ncoords_cur,
        /*row_len_prev=*/0, tile_coords, tile_fn, tile_ctx, nthreads,
        [=](const coord::FFACoordFreq* block, SizeType nblock,
            ComplexType* out) {
            std::array<PhaseStep, kMergeStepBlock> steps;
            fill_merge_phase_steps(
                [block](SizeType i) { return block[i].shift; }, nblock, nbins,
                steps.data());
            for (SizeType i = 0; i < nblock; ++i) {
                const auto row =
                    static_cast<SizeType>(block[i].idx) * fold_stride;
                shift_add_complex_recurrence_linear_steps(
                    seg_tail + row, seg_head + row, steps[i],
                    out + (i * fold_stride), nbins_f);
            }
        });
}

void ffa_iter(const float* __restrict__ fold_in,
              float* __restrict__ fold_out,
              const coord::FFACoord* __restrict__ coords,
              SizeType ncoords_cur,
              SizeType ncoords_prev,
              SizeType nsegments,
              SizeType nbins,
              int nthreads) {
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
                   int nthreads) {
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
                      std::span<std::byte> step_table,
                      int nthreads) {

    const bool segment_major =
        (nsegments >= 128) && (nsegments > ncoords_cur / 64);
    if (segment_major) {
        ffa_complex_iter_segment(
            fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
            nbins_f, nbins, as_phase_steps(step_table, 2 * kMergeStepBlock),
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
                           std::span<std::byte> step_table,
                           int nthreads) {
    const bool segment_major =
        (nsegments >= 128) && (nsegments > ncoords_cur / 64);
    if (segment_major) {
        ffa_complex_iter_segment_freq(
            fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
            nbins_f, nbins, as_phase_steps(step_table, kMergeStepBlock),
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
                            SizeType nbins,
                            SizeType n_leaves,
                            SizeType physical_start_idx,
                            SizeType capacity) noexcept {
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
    std::array<PhaseStep, kEPStepBlock> steps{};
    for (SizeType l0 = 0; l0 < n_leaves; l0 += kEPStepBlock) {
        const SizeType m = std::min(kEPStepBlock, n_leaves - l0);
        ep_phase_steps(phase_shift + l0, m, nbins, steps.data());
        for (SizeType i = 0; i < m; ++i) {
            const SizeType ileaf = l0 + i;
            const uint32_t tree_idx_logical =
                indices_tree[ileaf] + physical_start_idx;
            const uint32_t tree_idx_physical =
                tree_idx_logical < capacity ? tree_idx_logical
                                            : tree_idx_logical - capacity;
            const ComplexType* __restrict__ data_tree =
                folds_tree + (tree_idx_physical * total_size);
            const ComplexType* __restrict__ data_ffa =
                folds_ffa + (indices_ffa[ileaf] * total_size);
            ComplexType* __restrict__ data_out =
                folds_out + (ileaf * total_size);
            shift_add_complex_recurrence_linear_steps(
                data_tree, data_ffa, steps[i], data_out, nbins_f);
        }
    }
}

// Overwrite folds_tree
// indices_ffa and phase_shift are stored with n_segments as first dimension
// Second dimension is n_leaves, but batched, so call to this function
// should be in the same loop as the one that computes the phase shifts
void shift_add_ascend_linear_batch(const float* __restrict__ folds_ffa,
                                   const uint32_t* __restrict__ indices_segment,
                                   const SizeType* __restrict__ indices_ffa,
                                   const float* __restrict__ phase_shift,
                                   float* __restrict__ folds_tree,
                                   SizeType nbins,
                                   SizeType n_coords_init,
                                   SizeType n_leaves,
                                   SizeType n_segments) noexcept {
    const auto total_size = 2 * nbins;
    for (SizeType ileaf = 0; ileaf < n_leaves; ++ileaf) {
        float* __restrict__ data_tree = folds_tree + (ileaf * total_size);
        // Fill it with zero to avoid accumulation of garbage values
        std::fill(data_tree, data_tree + total_size, 0.0F);
        for (SizeType iseg = 0; iseg < n_segments; ++iseg) {
            const SizeType segment_idx = indices_segment[iseg];
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
    const uint32_t* __restrict__ indices_segment,
    const SizeType* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexType* __restrict__ folds_tree,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments) noexcept {
    const auto total_size = 2 * nbins_f;
    std::array<PhaseStep, kEPStepBlock> steps{};
    // Blocks of leaves, segments in order within a block: each leaf still
    // accumulates its segments in segment order.
    for (SizeType l0 = 0; l0 < n_leaves; l0 += kEPStepBlock) {
        const SizeType m = std::min(kEPStepBlock, n_leaves - l0);
        // Fill with zero to avoid accumulation of garbage values
        std::fill(folds_tree + (l0 * total_size),
                  folds_tree + ((l0 + m) * total_size),
                  ComplexType{0.0F, 0.0F});
        for (SizeType iseg = 0; iseg < n_segments; ++iseg) {
            const SizeType segment_idx = indices_segment[iseg];
            const SizeType row         = (iseg * n_leaves) + l0;
            ep_phase_steps(phase_shift + row, m, nbins, steps.data());
            for (SizeType i = 0; i < m; ++i) {
                const ComplexType* __restrict__ data_ffa =
                    folds_ffa + (segment_idx * n_coords_init * total_size) +
                    (indices_ffa[row + i] * total_size);
                shift_add_complex_recurrence_linear_inplace_steps(
                    data_ffa, steps[i], folds_tree + ((l0 + i) * total_size),
                    nbins_f);
            }
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

namespace {

/// Profiles per band level that one tile of `tile_coords` top coordinates
/// reads at most (widths[k] is the tile). False when a level is not
/// monotone in idx (the band must not run).
[[nodiscard]] bool
cone_band_widths(const coord::FFACoordFreq* const* coords_levels,
                 const SizeType* ncoords,
                 SizeType k_levels,
                 SizeType tile_coords,
                 std::array<SizeType, 17>& widths) noexcept {
    if (k_levels == 0 || k_levels > 16 || tile_coords == 0 ||
        ncoords == nullptr || coords_levels == nullptr) {
        return false;
    }
    const SizeType top = std::min(tile_coords, ncoords[k_levels]);
    if (top == 0) {
        return false;
    }
    widths           = {};
    widths[k_levels] = top;
    for (SizeType level = k_levels; level >= 1; --level) {
        const auto* coords = coords_levels[level];
        const SizeType n   = ncoords[level];
        if (coords == nullptr || n == 0 || !coords_idx_monotone(coords, n)) {
            return false;
        }
        widths[level - 1] = max_idx_span(coords, n, widths[level]);
        if (widths[level - 1] == 0) {
            return false;
        }
    }
    return true;
}

/**
 * @brief A batch of whole tiles of a cone band, top coordinates [c0, c1):
 * the block-aligned coordinate range [lo, hi) of every band level whose
 * phase steps the batch uses, and where each starts in the step table.
 */
struct ConeBatch {
    SizeType c0{0};
    SizeType c1{0};
    std::array<SizeType, 17> lo{};
    std::array<SizeType, 17> hi{};
    std::array<SizeType, 17> off{};
    SizeType nsteps{0};
};

ConeBatch cone_batch_steps(const coord::FFACoordFreq* const* coords_levels,
                           const SizeType* ncoords,
                           SizeType k_levels,
                           SizeType c0,
                           SizeType c1) noexcept {
    ConeBatch b{.c0 = c0, .c1 = c1};
    std::array<CoordRange, 17> ranges{};
    tile_input_ranges(coords_levels, k_levels, c0, c1, ranges.data());
    for (SizeType level = 1; level <= k_levels; ++level) {
        b.lo[level] = (ranges[level].lo / kMergeStepBlock) * kMergeStepBlock;
        b.hi[level] = std::min(
            ((ranges[level].hi + kMergeStepBlock - 1) / kMergeStepBlock) *
                kMergeStepBlock,
            ncoords[level]);
        b.off[level] = b.nsteps;
        b.nsteps += b.hi[level] - b.lo[level];
    }
    return b;
}

/**
 * @brief Split the top coordinates of a cone band into batches of whole
 * tiles: at most `max_tiles` tiles each, and (when `table_steps` is not 0)
 * no more phase steps than the table holds. Batch edges are multiples of
 * tile_coords, which is a multiple of kMergeStepBlock for complex bands,
 * so the step blocks are those of the unbatched merge.
 */
std::vector<ConeBatch>
plan_cone_batches(const coord::FFACoordFreq* const* coords_levels,
                  const SizeType* ncoords,
                  SizeType k_levels,
                  SizeType tile_coords,
                  SizeType max_tiles,
                  SizeType table_steps) {
    const SizeType ncoords_top = ncoords[k_levels];
    std::vector<ConeBatch> batches;
    for (SizeType c0 = 0; c0 < ncoords_top;) {
        const SizeType tiles_left =
            (ncoords_top - c0 + tile_coords - 1) / tile_coords;
        const auto batch_of = [&](SizeType ntiles) {
            return cone_batch_steps(
                coords_levels, ncoords, k_levels, c0,
                std::min(c0 + (ntiles * tile_coords), ncoords_top));
        };
        SizeType ntiles = std::min(tiles_left, max_tiles);
        if (table_steps != 0 && batch_of(ntiles).nsteps > table_steps) {
            // The steps grow with the batch: bisect the largest that fits.
            SizeType fit = 0;
            SizeType bad = ntiles;
            while (bad - fit > 1) {
                const SizeType mid = fit + ((bad - fit) / 2);
                if (batch_of(mid).nsteps <= table_steps) {
                    fit = mid;
                } else {
                    bad = mid;
                }
            }
            error_check::check_greater(
                fit, SizeType{0},
                "ffa_complex_cone_band_freq: the phase steps of one tile do "
                "not fit the step table");
            ntiles = fit;
        }
        batches.push_back(batch_of(ntiles));
        c0 = batches.back().c1;
    }
    return batches;
}

/// Tiles per batch: all of them, or at most batch_coords top coordinates
/// (at least one tile).
[[nodiscard]] SizeType cone_max_tiles(SizeType ncoords_top,
                                      SizeType tile_coords,
                                      SizeType batch_coords) noexcept {
    const SizeType all = (ncoords_top + tile_coords - 1) / tile_coords;
    if (batch_coords == 0) {
        return all;
    }
    return std::clamp<SizeType>(batch_coords / tile_coords, 1, all);
}

} // namespace

SizeType
cone_band_working_floats(const coord::FFACoordFreq* const* coords_levels,
                         const SizeType* ncoords,
                         SizeType k_levels,
                         SizeType tile_coords,
                         SizeType nbins) noexcept {
    std::array<SizeType, 17> widths{};
    if (nbins == 0 || !cone_band_widths(coords_levels, ncoords, k_levels,
                                        tile_coords, widths)) {
        return 0;
    }
    SizeType floats = 0;
    for (SizeType level = 0; level <= k_levels; ++level) {
        const SizeType nseg = SizeType{1} << (k_levels - level);
        floats = std::max(floats, nseg * widths[level] * 2 * nbins);
    }
    return floats;
}

SizeType phase_step_bytes() noexcept { return sizeof(PhaseStep); }

SizeType
cone_band_tile_step_bytes(const coord::FFACoordFreq* const* coords_levels,
                          const SizeType* ncoords,
                          SizeType k_levels,
                          SizeType tile_coords) noexcept {
    std::array<SizeType, 17> widths{};
    if (!cone_band_widths(coords_levels, ncoords, k_levels, tile_coords,
                          widths)) {
        return 0;
    }
    // A range of w coordinates rounded out to step blocks.
    SizeType steps = 0;
    for (SizeType level = 1; level <= k_levels; ++level) {
        steps +=
            std::min(widths[level] + (2 * kMergeStepBlock), ncoords[level]);
    }
    return steps * sizeof(PhaseStep);
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
                        SizeType batch_coords,
                        ConeBatchFn batch_fn,
                        void* batch_ctx,
                        int nthreads) {
    if (k_levels == 0 || k_levels > 16 || tile_coords == 0 ||
        nsegments_in == 0 || ncoords == nullptr) {
        return;
    }
    const SizeType nseg_group = SizeType{1} << k_levels;
    if ((nsegments_in % nseg_group) != 0) {
        return;
    }
    nthreads                      = std::max(nthreads, 1);
    const SizeType nseg_out       = nsegments_in >> k_levels;
    const SizeType ncoords_top    = ncoords[k_levels];
    const SizeType scratch_floats = cone_band_working_floats(
        coords_levels, ncoords, k_levels, tile_coords, nbins);
    if (scratch_floats == 0 || ncoords_top == 0 || nseg_out == 0) {
        return;
    }
    const bool bottom            = level_in == nullptr;
    const bool parallel_groups   = std::cmp_greater_equal(nseg_out, nthreads);
    const bool time_it           = thread_seconds != nullptr;
    const SizeType prefix_stride = segment_len + 1;
    // Batches of whole tiles; batch_fn runs after each.
    const SizeType batch_tiles =
        cone_max_tiles(ncoords_top, tile_coords, batch_coords);
    const SizeType batch_span = batch_tiles * tile_coords;

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

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(level_in, ts_e, ts_v, runs, run_offsets, level_out, coords_levels,  \
               ncoords, nbins, k_levels, tile_coords, score_fn, score_ctx,     \
               thread_seconds, nseg_group, nseg_out, ncoords_top,              \
               scratch_floats, bottom, parallel_groups, time_it,               \
               prefix_stride, segment_len, shared_prefix_e, shared_prefix_v,   \
               shared_p4_e, shared_p4_v, batch_span, batch_fn, batch_ctx)
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

        for (SizeType batch0 = 0; batch0 < ncoords_top; batch0 += batch_span) {
            const SizeType batch1 = std::min(batch0 + batch_span, ncoords_top);
            if (parallel_groups) {
#pragma omp for schedule(static)
                for (SizeType group = 0; group < nseg_out; ++group) {
                    if (bottom) {
                        const auto prefix_start = time_it
                                                      ? ConeClock::now()
                                                      : ConeClock::time_point{};
                        const SizeType seg0     = group * nseg_group;
                        for (SizeType iseg = 0; iseg < nseg_group; ++iseg) {
                            const SizeType global_seg = seg0 + iseg;
                            fill_segment_prefix(
                                ts_e + (global_seg * segment_len),
                                prefix_e.data() + (iseg * prefix_stride),
                                segment_len);
                            fill_segment_prefix(
                                ts_v + (global_seg * segment_len),
                                prefix_v.data() + (iseg * prefix_stride),
                                segment_len);
                        }
#if defined(__AVX2__)
                        const SizeType n_prefix = segment_len + 1;
                        for (SizeType iseg = 0; iseg + 4 <= nseg_group;
                             iseg += 4) {
                            const double* pe0 =
                                prefix_e.data() + (iseg + 0) * prefix_stride;
                            const double* pe1 =
                                prefix_e.data() + (iseg + 1) * prefix_stride;
                            const double* pe2 =
                                prefix_e.data() + (iseg + 2) * prefix_stride;
                            const double* pe3 =
                                prefix_e.data() + (iseg + 3) * prefix_stride;
                            double* dst_e =
                                p4_e.data() + (iseg * prefix_stride);

                            const double* pv0 =
                                prefix_v.data() + (iseg + 0) * prefix_stride;
                            const double* pv1 =
                                prefix_v.data() + (iseg + 1) * prefix_stride;
                            const double* pv2 =
                                prefix_v.data() + (iseg + 2) * prefix_stride;
                            const double* pv3 =
                                prefix_v.data() + (iseg + 3) * prefix_stride;
                            double* dst_v =
                                p4_v.data() + (iseg * prefix_stride);

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
                    for (SizeType coord0 = batch0; coord0 < batch1;
                         coord0 += tile_coords) {
                        const SizeType coord1 =
                            std::min(coord0 + tile_coords, ncoords_top);
                        cone_execute_tile(
                            scratch_a.data(), scratch_b.data(), level_in,
                            bottom ? prefix_e.data() : nullptr,
                            bottom ? prefix_v.data() : nullptr, runs,
                            run_offsets, prefix_stride, level_out,
                            coords_levels, ncoords, group, coord0, coord1,
                            nbins, k_levels, score_fn, score_ctx, time_it,
                            &brute_local, &merge_local, &score_local,
                            bottom ? p4_e.data() : nullptr,
                            bottom ? p4_v.data() : nullptr);
                    }
                }
            } else {
#pragma omp for schedule(static)
                for (SizeType coord0 = batch0; coord0 < batch1;
                     coord0 += tile_coords) {
                    const SizeType coord1 =
                        std::min(coord0 + tile_coords, ncoords_top);
                    for (SizeType group = 0; group < nseg_out; ++group) {
                        const double* pe  = nullptr;
                        const double* pv  = nullptr;
                        const double* p4e = nullptr;
                        const double* p4v = nullptr;
                        if (bottom) {
                            const SizeType off =
                                group * nseg_group * prefix_stride;
                            pe = shared_prefix_e.data() + off;
                            pv = shared_prefix_v.data() + off;
#if defined(__AVX2__)
                            p4e = shared_p4_e.data() + off;
                            p4v = shared_p4_v.data() + off;
#endif
                        }
                        cone_execute_tile(
                            scratch_a.data(), scratch_b.data(), level_in, pe,
                            pv, runs, run_offsets, prefix_stride, level_out,
                            coords_levels, ncoords, group, coord0, coord1,
                            nbins, k_levels, score_fn, score_ctx, time_it,
                            &brute_local, &merge_local, &score_local, p4e, p4v);
                    }
                }
            }
            if (batch_fn != nullptr) {
#pragma omp single
                batch_fn(batch_ctx);
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

void ffa_complex_cone_band_freq(const ComplexType* level_in,
                                ComplexType* level_out,
                                const coord::FFACoordFreq* const* coords_levels,
                                const SizeType* ncoords,
                                SizeType nsegments_in,
                                SizeType nbins_f,
                                SizeType nbins,
                                SizeType k_levels,
                                SizeType tile_coords,
                                ComplexTileFn tile_fn,
                                void* tile_ctx,
                                std::span<std::byte> step_table,
                                SizeType batch_coords,
                                ConeBatchFn batch_fn,
                                void* batch_ctx,
                                int nthreads) {
    if (level_in == nullptr || k_levels == 0 || k_levels > 16 ||
        tile_coords == 0 || nsegments_in == 0 || ncoords == nullptr) {
        return;
    }
    const SizeType nseg_group = SizeType{1} << k_levels;
    if ((nsegments_in % nseg_group) != 0) {
        return;
    }
    nthreads                     = std::max(nthreads, 1);
    const SizeType nseg_out      = nsegments_in >> k_levels;
    const SizeType ncoords_top   = ncoords[k_levels];
    const SizeType fold_stride   = 2 * nbins_f;
    const SizeType scratch_elems = cone_band_working_floats(
        coords_levels, ncoords, k_levels, tile_coords, nbins_f);
    if (scratch_elems == 0 || ncoords_top == 0 || nseg_out == 0) {
        return;
    }
    // The phase steps of each level of the band, in the step blocks of the
    // stored merge (ffa_complex_iter_freq), reused by every segment group
    // and tile. They are computed for one batch of top coordinates at a
    // time, into the caller's fixed table.
    const std::span<PhaseStep> table =
        as_phase_steps(step_table, kMergeStepBlock);
    const std::vector<ConeBatch> batches = plan_cone_batches(
        coords_levels, ncoords, k_levels, tile_coords,
        cone_max_tiles(ncoords_top, tile_coords, batch_coords), table.size());
    PhaseStep* const steps     = table.data();
    const bool parallel_groups = std::cmp_greater_equal(nseg_out, nthreads);

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(level_in, level_out, coords_levels, ncoords, nbins_f, nbins,        \
               k_levels, tile_coords, tile_fn, tile_ctx, nseg_group, nseg_out, \
               ncoords_top, fold_stride, scratch_elems, steps, batches,        \
               parallel_groups, batch_fn, batch_ctx)
    {
        utils::UninitVector<ComplexType> scratch_a(scratch_elems);
        utils::UninitVector<ComplexType> scratch_b(scratch_elems);
        // The 2^k input segments of `group`, merged k levels for the top
        // coordinates [coord0, coord1) of batch `b` in scratch.
        const auto run_tile = [&](const ConeBatch& b, SizeType group,
                                  SizeType coord0, SizeType coord1) {
            std::array<CoordRange, 17> ranges{};
            tile_input_ranges(coords_levels, k_levels, coord0, coord1,
                              ranges.data());
            ComplexType* cur      = scratch_a.data();
            ComplexType* next     = scratch_b.data();
            const SizeType width0 = ranges[0].hi - ranges[0].lo;
            const SizeType in_seg = ncoords[0] * fold_stride;
            for (SizeType iseg = 0; iseg < nseg_group; ++iseg) {
                const SizeType global_seg = (group * nseg_group) + iseg;
                std::memcpy(
                    static_cast<void*>(cur + (iseg * width0 * fold_stride)),
                    level_in + (global_seg * in_seg) +
                        (ranges[0].lo * fold_stride),
                    width0 * fold_stride * sizeof(ComplexType));
            }
            for (SizeType level = 1; level <= k_levels; ++level) {
                const SizeType nseg_lvl = SizeType{1} << (k_levels - level);
                const SizeType width_prv =
                    ranges[level - 1].hi - ranges[level - 1].lo;
                const SizeType width_cur = ranges[level].hi - ranges[level].lo;
                const SizeType seg_prev  = width_prv * fold_stride;
                const SizeType seg_cur   = width_cur * fold_stride;
                const auto* coords       = coords_levels[level];
                const SizeType base      = ranges[level].lo;
                const SizeType prev_base = ranges[level - 1].lo;
                const PhaseStep* level_steps = steps + b.off[level];
                const SizeType steps_lo      = b.lo[level];
                for (SizeType iseg = 0; iseg < nseg_lvl; ++iseg) {
                    const ComplexType* tail = cur + ((iseg * 2) * seg_prev);
                    const ComplexType* head =
                        cur + (((iseg * 2) + 1) * seg_prev);
                    ComplexType* out = next + (iseg * seg_cur);
                    for (SizeType local = 0; local < width_cur; ++local) {
                        const SizeType global = base + local;
                        const SizeType prev =
                            static_cast<SizeType>(coords[global].idx) -
                            prev_base;
                        shift_add_complex_recurrence_linear_steps(
                            tail + (prev * fold_stride),
                            head + (prev * fold_stride),
                            level_steps[global - steps_lo],
                            out + (local * fold_stride), nbins_f);
                    }
                }
                std::swap(cur, next);
            }
            const SizeType width_k = ranges[k_levels].hi - ranges[k_levels].lo;
            if (level_out != nullptr) {
                std::memcpy(
                    static_cast<void*>(level_out +
                                       (group * ncoords_top * fold_stride) +
                                       (ranges[k_levels].lo * fold_stride)),
                    cur, width_k * fold_stride * sizeof(ComplexType));
            }
            if (tile_fn != nullptr) {
                tile_fn(cur, ranges[k_levels].lo, width_k, tile_ctx);
            }
        };

        for (const ConeBatch& b : batches) {
            // Steps of every level of the batch (disjoint table ranges).
            for (SizeType level = 1; level <= k_levels; ++level) {
                const coord::FFACoordFreq* level_coords = coords_levels[level];
                const SizeType n                        = ncoords[level];
                PhaseStep* level_steps                  = steps + b.off[level];
#pragma omp for schedule(static) nowait
                for (SizeType b0 = b.lo[level]; b0 < b.hi[level];
                     b0 += kMergeStepBlock) {
                    const coord::FFACoordFreq* block = level_coords + b0;
                    fill_merge_phase_steps(
                        [block](SizeType i) { return block[i].shift; },
                        std::min(n - b0, SizeType{kMergeStepBlock}), nbins,
                        level_steps + (b0 - b.lo[level]));
                }
            }
#pragma omp barrier
            if (parallel_groups) {
#pragma omp for schedule(static)
                for (SizeType group = 0; group < nseg_out; ++group) {
                    for (SizeType coord0 = b.c0; coord0 < b.c1;
                         coord0 += tile_coords) {
                        run_tile(b, group, coord0,
                                 std::min(coord0 + tile_coords, b.c1));
                    }
                }
            } else {
#pragma omp for schedule(static)
                for (SizeType coord0 = b.c0; coord0 < b.c1;
                     coord0 += tile_coords) {
                    for (SizeType group = 0; group < nseg_out; ++group) {
                        run_tile(b, group, coord0,
                                 std::min(coord0 + tile_coords, b.c1));
                    }
                }
            }
            if (batch_fn != nullptr) {
#pragma omp single
                batch_fn(batch_ctx);
            }
        }
    }
}

} // namespace loki::core
