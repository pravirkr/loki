#pragma once

#include <array>

#include <xsimd/xsimd.hpp>
#ifdef __x86_64__
#include <immintrin.h> // For AVX-512 intrinsics
#elif defined(__aarch64__)
#include <arm_neon.h> // For NEON intrinsics
#endif

#include "loki/common/types.hpp"

namespace loki::simd_utils {

// Block transposes. Rows come as pointers (they need not be evenly spaced,
// and a caller may point unused rows at a dummy), and each primitive reads
// and writes exactly one block, so callers handle the tail columns.

#if defined(__AVX2__)
/// dst[c][r] = src[r][c] for an 8 x 8 block of floats.
[[gnu::always_inline]] inline void transpose8x8(const float* const* src,
                                                float* const* dst) noexcept {
    const __m256 t0 =
        _mm256_unpacklo_ps(_mm256_loadu_ps(src[0]), _mm256_loadu_ps(src[1]));
    const __m256 t1 =
        _mm256_unpackhi_ps(_mm256_loadu_ps(src[0]), _mm256_loadu_ps(src[1]));
    const __m256 t2 =
        _mm256_unpacklo_ps(_mm256_loadu_ps(src[2]), _mm256_loadu_ps(src[3]));
    const __m256 t3 =
        _mm256_unpackhi_ps(_mm256_loadu_ps(src[2]), _mm256_loadu_ps(src[3]));
    const __m256 t4 =
        _mm256_unpacklo_ps(_mm256_loadu_ps(src[4]), _mm256_loadu_ps(src[5]));
    const __m256 t5 =
        _mm256_unpackhi_ps(_mm256_loadu_ps(src[4]), _mm256_loadu_ps(src[5]));
    const __m256 t6 =
        _mm256_unpacklo_ps(_mm256_loadu_ps(src[6]), _mm256_loadu_ps(src[7]));
    const __m256 t7 =
        _mm256_unpackhi_ps(_mm256_loadu_ps(src[6]), _mm256_loadu_ps(src[7]));
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_storeu_ps(dst[0], _mm256_permute2f128_ps(u0, u4, 0x20));
    _mm256_storeu_ps(dst[1], _mm256_permute2f128_ps(u1, u5, 0x20));
    _mm256_storeu_ps(dst[2], _mm256_permute2f128_ps(u2, u6, 0x20));
    _mm256_storeu_ps(dst[3], _mm256_permute2f128_ps(u3, u7, 0x20));
    _mm256_storeu_ps(dst[4], _mm256_permute2f128_ps(u0, u4, 0x31));
    _mm256_storeu_ps(dst[5], _mm256_permute2f128_ps(u1, u5, 0x31));
    _mm256_storeu_ps(dst[6], _mm256_permute2f128_ps(u2, u6, 0x31));
    _mm256_storeu_ps(dst[7], _mm256_permute2f128_ps(u3, u7, 0x31));
}
#endif

#if defined(__AVX512F__)
/// dst[c * dst_stride + r] = src[r][c] for a 16 x 16 block of floats.
[[gnu::always_inline]] inline void
transpose16x16(const float* const* src,
               float* __restrict__ dst,
               SizeType dst_stride) noexcept {
    // NOLINTBEGIN(modernize-avoid-c-arrays): std::array drops the vector alignment attribute
    __m512 r[16];
    __m512 t[16];
    // NOLINTEND(modernize-avoid-c-arrays)
    for (SizeType i = 0; i < 16; ++i) {
        r[i] = _mm512_loadu_ps(src[i]);
    }
    // 32-bit interleave within 128-bit lanes, then 64-bit.
    for (SizeType i = 0; i < 16; i += 2) {
        t[i]     = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }
    for (SizeType i = 0; i < 16; i += 4) {
        r[i] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i]),
                                                   _mm512_castps_pd(t[i + 2])));
        r[i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(
            _mm512_castps_pd(t[i]), _mm512_castps_pd(t[i + 2])));
        r[i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(
            _mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
        r[i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(
            _mm512_castps_pd(t[i + 1]), _mm512_castps_pd(t[i + 3])));
    }
    // r[4 g + j] holds column (4 l + j) of rows 4 g .. 4 g + 3 in 128-bit
    // lane l. Gather the four row groups of each column: 128-bit shuffles.
    for (SizeType j = 0; j < 4; ++j) {
        const __m512 a = _mm512_shuffle_f32x4(r[j], r[4 + j], 0x44);
        const __m512 b = _mm512_shuffle_f32x4(r[j], r[4 + j], 0xEE);
        const __m512 c = _mm512_shuffle_f32x4(r[8 + j], r[12 + j], 0x44);
        const __m512 d = _mm512_shuffle_f32x4(r[8 + j], r[12 + j], 0xEE);
        // a = [g0 l0, g0 l1, g1 l0, g1 l1], c likewise for g2, g3.
        _mm512_storeu_ps(dst + ((0 + j) * dst_stride),
                         _mm512_shuffle_f32x4(a, c, 0x88));
        _mm512_storeu_ps(dst + ((4 + j) * dst_stride),
                         _mm512_shuffle_f32x4(a, c, 0xDD));
        _mm512_storeu_ps(dst + ((8 + j) * dst_stride),
                         _mm512_shuffle_f32x4(b, d, 0x88));
        _mm512_storeu_ps(dst + ((12 + j) * dst_stride),
                         _mm512_shuffle_f32x4(b, d, 0xDD));
    }
}
#endif

namespace detail {

template <typename BatchType>
inline void transpose_xsimd(const float* __restrict__ src,
                            float* __restrict__ dst,
                            SizeType nbins,
                            SizeType row_stride) noexcept {
    constexpr SizeType kBatchSize = BatchType::size;
    const SizeType nbins_rounded =
        ((nbins + kBatchSize - 1) / kBatchSize) * kBatchSize;
    std::array<BatchType, kBatchSize> rows{};
    for (SizeType jb = 0; jb < nbins_rounded; jb += kBatchSize) {
        // Load rows from temp_fn
        for (SizeType r = 0; r < kBatchSize; ++r) {
            rows[r] = BatchType::load_unaligned(&src[(r * row_stride) + jb]);
        }

        // In-place transpose (end must be begin + batch size; see
        // lane_fft_cpu.cpp).
        xsimd::transpose(rows.data(), rows.data() + kBatchSize);

        // Store transposed result
        for (SizeType r = 0; r < kBatchSize; ++r) {
            rows[r].store_unaligned(&dst[(jb + r) * kBatchSize]);
        }
    }
}

#if defined(__AVX2__)
inline void transpose_avx2(const float* __restrict__ src,
                           float* __restrict__ dst,
                           SizeType nbins,
                           SizeType row_stride) noexcept {
    const SizeType nbins_rounded = ((nbins + 7) / 8) * 8;
    std::array<const float*, 8> in{};
    std::array<float*, 8> out{};
    for (SizeType jb = 0; jb < nbins_rounded; jb += 8) {
        for (SizeType r = 0; r < 8; ++r) {
            in[r]  = src + (r * row_stride) + jb;
            out[r] = dst + ((jb + r) * 8);
        }
        transpose8x8(in.data(), out.data());
    }
}
#endif

#if defined(__AVX512F__)
inline void transpose_avx512(const float* __restrict__ src,
                             float* __restrict__ dst,
                             SizeType nbins,
                             SizeType row_stride) noexcept {
    const SizeType nbins_rounded = ((nbins + 15) / 16) * 16;
    std::array<const float*, 16> in{};
    for (SizeType jb = 0; jb < nbins_rounded; jb += 16) {
        for (SizeType r = 0; r < 16; ++r) {
            in[r] = src + (r * row_stride) + jb;
        }
        transpose16x16(in.data(), dst + (jb * 16), 16);
    }
}
#endif

#if defined(__ARM_NEON)
inline void transpose_neon(const float* __restrict__ src,
                           float* __restrict__ dst,
                           SizeType nbins,
                           SizeType row_stride) noexcept {
    const SizeType nbins_rounded = ((nbins + 3) / 4) * 4;
    for (SizeType jb = 0; jb < nbins_rounded; jb += 4) {
        float32x4_t r0 = vld1q_f32(&src[(0 * row_stride) + jb]);
        float32x4_t r1 = vld1q_f32(&src[(1 * row_stride) + jb]);
        float32x4_t r2 = vld1q_f32(&src[(2 * row_stride) + jb]);
        float32x4_t r3 = vld1q_f32(&src[(3 * row_stride) + jb]);

        const float32x4x2_t t01 = vtrnq_f32(r0, r1);
        const float32x4x2_t t23 = vtrnq_f32(r2, r3);

        r0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
        r1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
        r2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
        r3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

        vst1q_f32(&dst[(jb + 0) * 4], r0);
        vst1q_f32(&dst[(jb + 1) * 4], r1);
        vst1q_f32(&dst[(jb + 2) * 4], r2);
        vst1q_f32(&dst[(jb + 3) * 4], r3);
    }
}
#endif

} // namespace detail

template <typename BatchType>
inline void transpose(const float* __restrict__ src,
                      float* __restrict__ dst,
                      SizeType nbins,
                      SizeType row_stride) noexcept {

#if defined(__AVX512F__)
    if constexpr (BatchType::size == 16) {
        detail::transpose_avx512(src, dst, nbins, row_stride);
        return;
    }
#endif

#if defined(__AVX2__)
    if constexpr (BatchType::size == 8) {
        detail::transpose_avx2(src, dst, nbins, row_stride);
        return;
    }
#endif

#if defined(__ARM_NEON)
    if constexpr (BatchType::size == 4) {
        detail::transpose_neon(src, dst, nbins, row_stride);
        return;
    }
#endif

    detail::transpose_xsimd<BatchType>(src, dst, nbins, row_stride);
}

} // namespace loki::simd_utils
