#pragma once

#include <xsimd/xsimd.hpp>
#ifdef __x86_64__
#include <immintrin.h> // For AVX-512 intrinsics
#elif defined(__aarch64__)
#include <arm_neon.h> // For NEON intrinsics
#endif

#include "loki/common/types.hpp"

namespace loki::simd_utils {

namespace detail {

template <typename BatchType>
inline void transpose_xsimd(const float* __restrict__ src,
                            float* __restrict__ dst,
                            SizeType nbins) noexcept {
    constexpr SizeType kBatchSize = BatchType::size;
    const SizeType nbins_rounded =
        ((nbins + kBatchSize - 1) / kBatchSize) * kBatchSize;
    std::array<BatchType, kBatchSize> rows{};
    for (SizeType jb = 0; jb < nbins_rounded; jb += kBatchSize) {
        // Load rows from temp_fn
        for (SizeType r = 0; r < kBatchSize; ++r) {
            rows[r] = BatchType::load_unaligned(&src[(r * nbins) + jb]);
        }

        // In-place transpose
        xsimd::transpose(rows.data(), rows.data());

        // Store transposed result
        for (SizeType r = 0; r < kBatchSize; ++r) {
            rows[r].store_unaligned(&dst[(jb + r) * kBatchSize]);
        }
    }
}

#if defined(__AVX2__)
inline void transpose_avx2(const float* __restrict__ src,
                           float* __restrict__ dst,
                           SizeType nbins) noexcept {
    const SizeType nbins_rounded = ((nbins + 7) / 8) * 8;
    for (SizeType jb = 0; jb < nbins_rounded; jb += 8) {
        __m256 r0 = _mm256_loadu_ps(&src[0 * nbins + jb]);
        __m256 r1 = _mm256_loadu_ps(&src[1 * nbins + jb]);
        __m256 r2 = _mm256_loadu_ps(&src[2 * nbins + jb]);
        __m256 r3 = _mm256_loadu_ps(&src[3 * nbins + jb]);
        __m256 r4 = _mm256_loadu_ps(&src[4 * nbins + jb]);
        __m256 r5 = _mm256_loadu_ps(&src[5 * nbins + jb]);
        __m256 r6 = _mm256_loadu_ps(&src[6 * nbins + jb]);
        __m256 r7 = _mm256_loadu_ps(&src[7 * nbins + jb]);

        __m256 tmp0 = _mm256_unpacklo_ps(r0, r1);
        __m256 tmp1 = _mm256_unpackhi_ps(r0, r1);
        __m256 tmp2 = _mm256_unpacklo_ps(r2, r3);
        __m256 tmp3 = _mm256_unpackhi_ps(r2, r3);
        __m256 tmp4 = _mm256_unpacklo_ps(r4, r5);
        __m256 tmp5 = _mm256_unpackhi_ps(r4, r5);
        __m256 tmp6 = _mm256_unpacklo_ps(r6, r7);
        __m256 tmp7 = _mm256_unpackhi_ps(r6, r7);

        __m256 tmp8 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tmp9 = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tmpa = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tmpb = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tmpc = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tmpd = _mm256_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tmpe = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tmpf = _mm256_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(3, 2, 3, 2));

        r0 = _mm256_permute2f128_ps(tmp8, tmpc, 0x20);
        r1 = _mm256_permute2f128_ps(tmp9, tmpd, 0x20);
        r2 = _mm256_permute2f128_ps(tmpa, tmpe, 0x20);
        r3 = _mm256_permute2f128_ps(tmpb, tmpf, 0x20);
        r4 = _mm256_permute2f128_ps(tmp8, tmpc, 0x31);
        r5 = _mm256_permute2f128_ps(tmp9, tmpd, 0x31);
        r6 = _mm256_permute2f128_ps(tmpa, tmpe, 0x31);
        r7 = _mm256_permute2f128_ps(tmpb, tmpf, 0x31);

        _mm256_store_ps(&dst[(jb + 0) * 8], r0);
        _mm256_store_ps(&dst[(jb + 1) * 8], r1);
        _mm256_store_ps(&dst[(jb + 2) * 8], r2);
        _mm256_store_ps(&dst[(jb + 3) * 8], r3);
        _mm256_store_ps(&dst[(jb + 4) * 8], r4);
        _mm256_store_ps(&dst[(jb + 5) * 8], r5);
        _mm256_store_ps(&dst[(jb + 6) * 8], r6);
        _mm256_store_ps(&dst[(jb + 7) * 8], r7);
    }
}
#endif

#if defined(__AVX512F__)
inline void transpose_avx512(const float* __restrict__ src,
                             float* __restrict__ dst,
                             SizeType nbins) noexcept {
    const SizeType nbins_rounded = ((nbins + 15) / 16) * 16;
    for (SizeType jb = 0; jb < nbins_rounded; jb += 16) {
        __m512 r0 = _mm512_loadu_ps(&src[0 * nbins + jb]);
        __m512 r1 = _mm512_loadu_ps(&src[1 * nbins + jb]);
        __m512 r2 = _mm512_loadu_ps(&src[2 * nbins + jb]);
        __m512 r3 = _mm512_loadu_ps(&src[3 * nbins + jb]);
        __m512 r4 = _mm512_loadu_ps(&src[4 * nbins + jb]);
        __m512 r5 = _mm512_loadu_ps(&src[5 * nbins + jb]);
        __m512 r6 = _mm512_loadu_ps(&src[6 * nbins + jb]);
        __m512 r7 = _mm512_loadu_ps(&src[7 * nbins + jb]);
        __m512 r8 = _mm512_loadu_ps(&src[8 * nbins + jb]);
        __m512 r9 = _mm512_loadu_ps(&src[9 * nbins + jb]);
        __m512 ra = _mm512_loadu_ps(&src[10 * nbins + jb]);
        __m512 rb = _mm512_loadu_ps(&src[11 * nbins + jb]);
        __m512 rc = _mm512_loadu_ps(&src[12 * nbins + jb]);
        __m512 rd = _mm512_loadu_ps(&src[13 * nbins + jb]);
        __m512 re = _mm512_loadu_ps(&src[14 * nbins + jb]);
        __m512 rf = _mm512_loadu_ps(&src[15 * nbins + jb]);

        __m512 tmp0 = _mm512_unpacklo_ps(r0, r1);
        __m512 tmp1 = _mm512_unpackhi_ps(r0, r1);
        __m512 tmp2 = _mm512_unpacklo_ps(r2, r3);
        __m512 tmp3 = _mm512_unpackhi_ps(r2, r3);
        __m512 tmp4 = _mm512_unpacklo_ps(r4, r5);
        __m512 tmp5 = _mm512_unpackhi_ps(r4, r5);
        __m512 tmp6 = _mm512_unpacklo_ps(r6, r7);
        __m512 tmp7 = _mm512_unpackhi_ps(r6, r7);
        __m512 tmp8 = _mm512_unpacklo_ps(r8, r9);
        __m512 tmp9 = _mm512_unpackhi_ps(r8, r9);
        __m512 tmpa = _mm512_unpacklo_ps(ra, rb);
        __m512 tmpb = _mm512_unpackhi_ps(ra, rb);
        __m512 tmpc = _mm512_unpacklo_ps(rc, rd);
        __m512 tmpd = _mm512_unpackhi_ps(rc, rd);
        __m512 tmpe = _mm512_unpacklo_ps(re, rf);
        __m512 tmpf = _mm512_unpackhi_ps(re, rf);

        __m512 tmpg = _mm512_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmph = _mm512_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmpi = _mm512_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpj = _mm512_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmpk = _mm512_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpl = _mm512_shuffle_ps(tmp4, tmp6, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmpm = _mm512_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpn = _mm512_shuffle_ps(tmp5, tmp7, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmpo = _mm512_shuffle_ps(tmp8, tmpa, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpp = _mm512_shuffle_ps(tmp8, tmpa, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmpq = _mm512_shuffle_ps(tmp9, tmpb, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpr = _mm512_shuffle_ps(tmp9, tmpb, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmps = _mm512_shuffle_ps(tmpc, tmpe, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpt = _mm512_shuffle_ps(tmpc, tmpe, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 tmpu = _mm512_shuffle_ps(tmpd, tmpf, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 tmpv = _mm512_shuffle_ps(tmpd, tmpf, _MM_SHUFFLE(3, 2, 3, 2));

        tmp0 = _mm512_shuffle_f32x4(tmpg, tmpk, _MM_SHUFFLE(2, 0, 2, 0));
        tmp1 = _mm512_shuffle_f32x4(tmpo, tmps, _MM_SHUFFLE(2, 0, 2, 0));
        tmp2 = _mm512_shuffle_f32x4(tmph, tmpl, _MM_SHUFFLE(2, 0, 2, 0));
        tmp3 = _mm512_shuffle_f32x4(tmpp, tmpt, _MM_SHUFFLE(2, 0, 2, 0));
        tmp4 = _mm512_shuffle_f32x4(tmpi, tmpm, _MM_SHUFFLE(2, 0, 2, 0));
        tmp5 = _mm512_shuffle_f32x4(tmpq, tmpu, _MM_SHUFFLE(2, 0, 2, 0));
        tmp6 = _mm512_shuffle_f32x4(tmpj, tmpn, _MM_SHUFFLE(2, 0, 2, 0));
        tmp7 = _mm512_shuffle_f32x4(tmpr, tmpv, _MM_SHUFFLE(2, 0, 2, 0));
        tmp8 = _mm512_shuffle_f32x4(tmpg, tmpk, _MM_SHUFFLE(3, 1, 3, 1));
        tmp9 = _mm512_shuffle_f32x4(tmpo, tmps, _MM_SHUFFLE(3, 1, 3, 1));
        tmpa = _mm512_shuffle_f32x4(tmph, tmpl, _MM_SHUFFLE(3, 1, 3, 1));
        tmpb = _mm512_shuffle_f32x4(tmpp, tmpt, _MM_SHUFFLE(3, 1, 3, 1));
        tmpc = _mm512_shuffle_f32x4(tmpi, tmpm, _MM_SHUFFLE(3, 1, 3, 1));
        tmpd = _mm512_shuffle_f32x4(tmpq, tmpu, _MM_SHUFFLE(3, 1, 3, 1));
        tmpe = _mm512_shuffle_f32x4(tmpj, tmpn, _MM_SHUFFLE(3, 1, 3, 1));
        tmpf = _mm512_shuffle_f32x4(tmpr, tmpv, _MM_SHUFFLE(3, 1, 3, 1));

        r0 = _mm512_shuffle_f32x4(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0));
        r1 = _mm512_shuffle_f32x4(tmp2, tmp3, _MM_SHUFFLE(2, 0, 2, 0));
        r2 = _mm512_shuffle_f32x4(tmp4, tmp5, _MM_SHUFFLE(2, 0, 2, 0));
        r3 = _mm512_shuffle_f32x4(tmp6, tmp7, _MM_SHUFFLE(2, 0, 2, 0));
        r4 = _mm512_shuffle_f32x4(tmp8, tmp9, _MM_SHUFFLE(2, 0, 2, 0));
        r5 = _mm512_shuffle_f32x4(tmpa, tmpb, _MM_SHUFFLE(2, 0, 2, 0));
        r6 = _mm512_shuffle_f32x4(tmpc, tmpd, _MM_SHUFFLE(2, 0, 2, 0));
        r7 = _mm512_shuffle_f32x4(tmpe, tmpf, _MM_SHUFFLE(2, 0, 2, 0));
        r8 = _mm512_shuffle_f32x4(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1));
        r9 = _mm512_shuffle_f32x4(tmp2, tmp3, _MM_SHUFFLE(3, 1, 3, 1));
        ra = _mm512_shuffle_f32x4(tmp4, tmp5, _MM_SHUFFLE(3, 1, 3, 1));
        rb = _mm512_shuffle_f32x4(tmp6, tmp7, _MM_SHUFFLE(3, 1, 3, 1));
        rc = _mm512_shuffle_f32x4(tmp8, tmp9, _MM_SHUFFLE(3, 1, 3, 1));
        rd = _mm512_shuffle_f32x4(tmpa, tmpb, _MM_SHUFFLE(3, 1, 3, 1));
        re = _mm512_shuffle_f32x4(tmpc, tmpd, _MM_SHUFFLE(3, 1, 3, 1));
        rf = _mm512_shuffle_f32x4(tmpe, tmpf, _MM_SHUFFLE(3, 1, 3, 1));

        _mm512_store_ps(&dst[(jb + 0) * 16], r0);
        _mm512_store_ps(&dst[(jb + 1) * 16], r1);
        _mm512_store_ps(&dst[(jb + 2) * 16], r2);
        _mm512_store_ps(&dst[(jb + 3) * 16], r3);
        _mm512_store_ps(&dst[(jb + 4) * 16], r4);
        _mm512_store_ps(&dst[(jb + 5) * 16], r5);
        _mm512_store_ps(&dst[(jb + 6) * 16], r6);
        _mm512_store_ps(&dst[(jb + 7) * 16], r7);
        _mm512_store_ps(&dst[(jb + 8) * 16], r8);
        _mm512_store_ps(&dst[(jb + 9) * 16], r9);
        _mm512_store_ps(&dst[(jb + 10) * 16], ra);
        _mm512_store_ps(&dst[(jb + 11) * 16], rb);
        _mm512_store_ps(&dst[(jb + 12) * 16], rc);
        _mm512_store_ps(&dst[(jb + 13) * 16], rd);
        _mm512_store_ps(&dst[(jb + 14) * 16], re);
        _mm512_store_ps(&dst[(jb + 15) * 16], rf);
    }
}
#endif

#if defined(__ARM_NEON)
inline void transpose_neon(const float* __restrict__ src,
                           float* __restrict__ dst,
                           SizeType nbins) noexcept {
    const SizeType nbins_rounded = ((nbins + 3) / 4) * 4;
    for (SizeType jb = 0; jb < nbins_rounded; jb += 4) {
        float32x4_t r0 = vld1q_f32(&src[(0 * nbins) + jb]);
        float32x4_t r1 = vld1q_f32(&src[(1 * nbins) + jb]);
        float32x4_t r2 = vld1q_f32(&src[(2 * nbins) + jb]);
        float32x4_t r3 = vld1q_f32(&src[(3 * nbins) + jb]);

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
                      SizeType nbins) noexcept {

#if defined(__AVX512F__)
    if constexpr (BatchType::size == 16) {
        detail::transpose_avx512(src, dst, nbins);
        return;
    }
#endif

#if defined(__AVX2__)
    if constexpr (BatchType::size == 8) {
        detail::transpose_avx2(src, dst, nbins);
        return;
    }
#endif

#if defined(__ARM_NEON)
    if constexpr (BatchType::size == 4) {
        detail::transpose_neon(src, dst, nbins);
        return;
    }
#endif

    detail::transpose_xsimd<BatchType>(src, dst, nbins);
}

} // namespace loki::simd_utils