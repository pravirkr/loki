#pragma once

#include <cstddef>

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "loki/common/types.hpp"

namespace loki::brute_fold_intrinsics {

#if defined(__GNUC__) || defined(__clang__)
#define BRUTE_FOLD_INLINE inline __attribute__((always_inline))
#else
#define BRUTE_FOLD_INLINE inline
#endif

// Compile-time ISA: AVX-512, then AVX2+FMA, then NEON. No native kernel →
// brute_fold_segment_xsimd is the fallback.
#if defined(__AVX512F__)
inline constexpr bool kHasNativeKernel = true;
inline const char* isa_name() noexcept { return "avx512f"; }

using Vec                            = __m512;
inline constexpr SizeType kBatchSize = 16;

BRUTE_FOLD_INLINE Vec vzero() noexcept { return _mm512_setzero_ps(); }
BRUTE_FOLD_INLINE Vec loadu(const float* p) noexcept {
    return _mm512_loadu_ps(p);
}
BRUTE_FOLD_INLINE Vec loada(const float* p) noexcept {
    return _mm512_load_ps(p);
}
BRUTE_FOLD_INLINE void storea(float* p, Vec v) noexcept {
    _mm512_store_ps(p, v);
}
BRUTE_FOLD_INLINE Vec vadd(Vec a, Vec b) noexcept {
    return _mm512_add_ps(a, b);
}
BRUTE_FOLD_INLINE Vec vmul(Vec a, Vec b) noexcept {
    return _mm512_mul_ps(a, b);
}
BRUTE_FOLD_INLINE Vec vfma(Vec a, Vec b, Vec c) noexcept {
    return _mm512_fmadd_ps(a, b, c);
}
BRUTE_FOLD_INLINE Vec vfms(Vec a, Vec b, Vec c) noexcept {
    return _mm512_fmsub_ps(a, b, c);
}
BRUTE_FOLD_INLINE Vec complex_mul_r(Vec ar, Vec ai, Vec br, Vec bi) noexcept {
    return _mm512_fmsub_ps(ar, br, _mm512_mul_ps(ai, bi));
}
BRUTE_FOLD_INLINE Vec complex_mul_i(Vec ar, Vec ai, Vec br, Vec bi) noexcept {
    return _mm512_fmadd_ps(ar, bi, _mm512_mul_ps(ai, br));
}
BRUTE_FOLD_INLINE float vreduce_add(Vec v) noexcept {
    return _mm512_reduce_add_ps(v);
}

#elif defined(__AVX2__) && defined(__FMA__)
inline constexpr bool kHasNativeKernel = true;
inline const char* isa_name() noexcept { return "avx2+fma"; }

using Vec                            = __m256;
inline constexpr SizeType kBatchSize = 8;

BRUTE_FOLD_INLINE Vec vzero() noexcept { return _mm256_setzero_ps(); }
BRUTE_FOLD_INLINE Vec loadu(const float* p) noexcept {
    return _mm256_loadu_ps(p);
}
BRUTE_FOLD_INLINE Vec loada(const float* p) noexcept {
    return _mm256_load_ps(p);
}
BRUTE_FOLD_INLINE void storea(float* p, Vec v) noexcept {
    _mm256_store_ps(p, v);
}
BRUTE_FOLD_INLINE Vec vadd(Vec a, Vec b) noexcept {
    return _mm256_add_ps(a, b);
}
BRUTE_FOLD_INLINE Vec vmul(Vec a, Vec b) noexcept {
    return _mm256_mul_ps(a, b);
}
BRUTE_FOLD_INLINE Vec vfma(Vec a, Vec b, Vec c) noexcept {
    return _mm256_fmadd_ps(a, b, c);
}
BRUTE_FOLD_INLINE Vec vfms(Vec a, Vec b, Vec c) noexcept {
    return _mm256_fmsub_ps(a, b, c);
}
BRUTE_FOLD_INLINE Vec complex_mul_r(Vec ar, Vec ai, Vec br, Vec bi) noexcept {
    return _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi));
}
BRUTE_FOLD_INLINE Vec complex_mul_i(Vec ar, Vec ai, Vec br, Vec bi) noexcept {
    return _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));
}
BRUTE_FOLD_INLINE float vreduce_add(Vec v) noexcept {
    const __m128 lo   = _mm256_castps256_ps128(v);
    const __m128 hi   = _mm256_extractf128_ps(v, 1);
    const __m128 sum  = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(sum);
    const __m128 sums = _mm_add_ps(sum, shuf);
    const __m128 last = _mm_add_ss(sums, _mm_movehl_ps(shuf, sums));
    return _mm_cvtss_f32(last);
}

#elif defined(__aarch64__) || defined(__ARM_NEON)
inline constexpr bool kHasNativeKernel = true;
inline const char* isa_name() noexcept { return "neon"; }

using Vec                            = float32x4_t;
inline constexpr SizeType kBatchSize = 4;

BRUTE_FOLD_INLINE Vec vzero() noexcept { return vdupq_n_f32(0.0F); }
BRUTE_FOLD_INLINE Vec loadu(const float* p) noexcept { return vld1q_f32(p); }
BRUTE_FOLD_INLINE Vec loada(const float* p) noexcept { return vld1q_f32(p); }
BRUTE_FOLD_INLINE void storea(float* p, Vec v) noexcept { vst1q_f32(p, v); }
BRUTE_FOLD_INLINE Vec vadd(Vec a, Vec b) noexcept { return vaddq_f32(a, b); }
BRUTE_FOLD_INLINE Vec vmul(Vec a, Vec b) noexcept { return vmulq_f32(a, b); }
BRUTE_FOLD_INLINE Vec vfma(Vec a, Vec b, Vec c) noexcept {
    return vfmaq_f32(c, a, b);
}
BRUTE_FOLD_INLINE Vec vfms(Vec a, Vec b, Vec c) noexcept {
    return vfmsq_f32(c, a, b);
}
BRUTE_FOLD_INLINE Vec complex_mul_r(Vec ar, Vec ai, Vec br, Vec bi) noexcept {
    return vfmsq_f32(vmulq_f32(ar, br), ai, bi);
}
BRUTE_FOLD_INLINE Vec complex_mul_i(Vec ar, Vec ai, Vec br, Vec bi) noexcept {
    return vfmaq_f32(vmulq_f32(ar, bi), ai, br);
}
BRUTE_FOLD_INLINE float vreduce_add(Vec v) noexcept { return vaddvq_f32(v); }
#else
inline constexpr bool kHasNativeKernel = false;
inline const char* isa_name() noexcept { return "none (xsimd fallback)"; }
#endif

#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__)) ||         \
    defined(__aarch64__) || defined(__ARM_NEON)

BRUTE_FOLD_INLINE void dc_sum(const float* ts_e,
                              const float* ts_v,
                              SizeType segment_len,
                              float& sum_e,
                              float& sum_v) noexcept {
    Vec sum_e0 = vzero();
    Vec sum_e1 = vzero();
    Vec sum_v0 = vzero();
    Vec sum_v1 = vzero();
    SizeType i = 0;
    for (; i + (2 * kBatchSize) <= segment_len; i += 2 * kBatchSize) {
        sum_e0 = vadd(sum_e0, loadu(&ts_e[i]));
        sum_v0 = vadd(sum_v0, loadu(&ts_v[i]));
        sum_e1 = vadd(sum_e1, loadu(&ts_e[i + kBatchSize]));
        sum_v1 = vadd(sum_v1, loadu(&ts_v[i + kBatchSize]));
    }
    if (i + kBatchSize <= segment_len) {
        sum_e0 = vadd(sum_e0, loadu(&ts_e[i]));
        sum_v0 = vadd(sum_v0, loadu(&ts_v[i]));
        i += kBatchSize;
    }
    sum_e = vreduce_add(vadd(sum_e0, sum_e1));
    sum_v = vreduce_add(vadd(sum_v0, sum_v1));
    for (; i < segment_len; ++i) {
        sum_e += ts_e[i];
        sum_v += ts_v[i];
    }
}

inline void fold_one_segment(const float* __restrict__ ts_e_seg,
                             const float* __restrict__ ts_v_seg,
                             ComplexType* __restrict__ fold_seg,
                             float* __restrict__ current_r,
                             float* __restrict__ current_i,
                             const float* __restrict__ delta_r,
                             const float* __restrict__ delta_i,
                             SizeType nfreqs,
                             SizeType segment_len,
                             SizeType nbins_f) noexcept {
    float sum_e = 0.0F;
    float sum_v = 0.0F;
    dc_sum(ts_e_seg, ts_v_seg, segment_len, sum_e, sum_v);

    for (SizeType ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const auto freq_offset_out            = ifreq * 2 * nbins_f;
        const auto phasor_offset              = ifreq * segment_len;
        ComplexType* __restrict__ fold_e_base = fold_seg + freq_offset_out;
        ComplexType* __restrict__ fold_v_base = fold_e_base + nbins_f;

        fold_e_base[0] = ComplexType(sum_e, 0.0F);
        fold_v_base[0] = ComplexType(sum_v, 0.0F);

        const float* __restrict__ base_r = delta_r + phasor_offset;
        const float* __restrict__ base_i = delta_i + phasor_offset;

        SizeType m = 1;

#if defined(__AVX512F__) || defined(__aarch64__) || defined(__ARM_NEON)
        // 4-way harmonic unrolling.
        // First group (m=1..4) reads base_r/base_i directly, avoiding 100% of
        // initial std::copy overhead.
        for (; m + 4 <= nbins_f; m += 4) {
            const bool from_base = (m == 1);
            Vec acc_e_r0 = vzero(), acc_e_i0 = vzero();
            Vec acc_v_r0 = vzero(), acc_v_i0 = vzero();
            Vec acc_e_r1 = vzero(), acc_e_i1 = vzero();
            Vec acc_v_r1 = vzero(), acc_v_i1 = vzero();
            Vec acc_e_r2 = vzero(), acc_e_i2 = vzero();
            Vec acc_v_r2 = vzero(), acc_v_i2 = vzero();
            Vec acc_e_r3 = vzero(), acc_e_i3 = vzero();
            Vec acc_v_r3 = vzero(), acc_v_i3 = vzero();

            SizeType k = 0;
            for (; k + (2 * kBatchSize) <= segment_len; k += 2 * kBatchSize) {
                // Sub-step 0: at k
                {
                    const Vec e   = loadu(&ts_e_seg[k]);
                    const Vec v   = loadu(&ts_v_seg[k]);
                    const Vec br  = loada(&base_r[k]);
                    const Vec bi  = loada(&base_i[k]);
                    const Vec cr0 = from_base ? br : loada(&current_r[k]);
                    const Vec ci0 = from_base ? bi : loada(&current_i[k]);

                    acc_e_r0 = vfma(e, cr0, acc_e_r0);
                    acc_e_i0 = vfma(e, ci0, acc_e_i0);
                    acc_v_r0 = vfma(v, cr0, acc_v_r0);
                    acc_v_i0 = vfma(v, ci0, acc_v_i0);

                    const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                    const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);

                    acc_e_r1 = vfma(e, cr1, acc_e_r1);
                    acc_e_i1 = vfma(e, ci1, acc_e_i1);
                    acc_v_r1 = vfma(v, cr1, acc_v_r1);
                    acc_v_i1 = vfma(v, ci1, acc_v_i1);

                    const Vec cr2 = complex_mul_r(cr1, ci1, br, bi);
                    const Vec ci2 = complex_mul_i(cr1, ci1, br, bi);

                    acc_e_r2 = vfma(e, cr2, acc_e_r2);
                    acc_e_i2 = vfma(e, ci2, acc_e_i2);
                    acc_v_r2 = vfma(v, cr2, acc_v_r2);
                    acc_v_i2 = vfma(v, ci2, acc_v_i2);

                    const Vec cr3 = complex_mul_r(cr2, ci2, br, bi);
                    const Vec ci3 = complex_mul_i(cr2, ci2, br, bi);

                    acc_e_r3 = vfma(e, cr3, acc_e_r3);
                    acc_e_i3 = vfma(e, ci3, acc_e_i3);
                    acc_v_r3 = vfma(v, cr3, acc_v_r3);
                    acc_v_i3 = vfma(v, ci3, acc_v_i3);

                    const Vec cr4 = complex_mul_r(cr3, ci3, br, bi);
                    const Vec ci4 = complex_mul_i(cr3, ci3, br, bi);
                    storea(&current_r[k], cr4);
                    storea(&current_i[k], ci4);
                }

                // Sub-step 1: at k + kBatchSize
                {
                    const SizeType k1 = k + kBatchSize;
                    const Vec e       = loadu(&ts_e_seg[k1]);
                    const Vec v       = loadu(&ts_v_seg[k1]);
                    const Vec br      = loada(&base_r[k1]);
                    const Vec bi      = loada(&base_i[k1]);
                    const Vec cr0     = from_base ? br : loada(&current_r[k1]);
                    const Vec ci0     = from_base ? bi : loada(&current_i[k1]);

                    acc_e_r0 = vfma(e, cr0, acc_e_r0);
                    acc_e_i0 = vfma(e, ci0, acc_e_i0);
                    acc_v_r0 = vfma(v, cr0, acc_v_r0);
                    acc_v_i0 = vfma(v, ci0, acc_v_i0);

                    const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                    const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);

                    acc_e_r1 = vfma(e, cr1, acc_e_r1);
                    acc_e_i1 = vfma(e, ci1, acc_e_i1);
                    acc_v_r1 = vfma(v, cr1, acc_v_r1);
                    acc_v_i1 = vfma(v, ci1, acc_v_i1);

                    const Vec cr2 = complex_mul_r(cr1, ci1, br, bi);
                    const Vec ci2 = complex_mul_i(cr1, ci1, br, bi);

                    acc_e_r2 = vfma(e, cr2, acc_e_r2);
                    acc_e_i2 = vfma(e, ci2, acc_e_i2);
                    acc_v_r2 = vfma(v, cr2, acc_v_r2);
                    acc_v_i2 = vfma(v, ci2, acc_v_i2);

                    const Vec cr3 = complex_mul_r(cr2, ci2, br, bi);
                    const Vec ci3 = complex_mul_i(cr2, ci2, br, bi);

                    acc_e_r3 = vfma(e, cr3, acc_e_r3);
                    acc_e_i3 = vfma(e, ci3, acc_e_i3);
                    acc_v_r3 = vfma(v, cr3, acc_v_r3);
                    acc_v_i3 = vfma(v, ci3, acc_v_i3);

                    const Vec cr4 = complex_mul_r(cr3, ci3, br, bi);
                    const Vec ci4 = complex_mul_i(cr3, ci3, br, bi);
                    storea(&current_r[k1], cr4);
                    storea(&current_i[k1], ci4);
                }
            }
            if (k + kBatchSize <= segment_len) {
                const Vec e   = loadu(&ts_e_seg[k]);
                const Vec v   = loadu(&ts_v_seg[k]);
                const Vec br  = loada(&base_r[k]);
                const Vec bi  = loada(&base_i[k]);
                const Vec cr0 = from_base ? br : loada(&current_r[k]);
                const Vec ci0 = from_base ? bi : loada(&current_i[k]);

                acc_e_r0 = vfma(e, cr0, acc_e_r0);
                acc_e_i0 = vfma(e, ci0, acc_e_i0);
                acc_v_r0 = vfma(v, cr0, acc_v_r0);
                acc_v_i0 = vfma(v, ci0, acc_v_i0);

                const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);

                acc_e_r1 = vfma(e, cr1, acc_e_r1);
                acc_e_i1 = vfma(e, ci1, acc_e_i1);
                acc_v_r1 = vfma(v, cr1, acc_v_r1);
                acc_v_i1 = vfma(v, ci1, acc_v_i1);

                const Vec cr2 = complex_mul_r(cr1, ci1, br, bi);
                const Vec ci2 = complex_mul_i(cr1, ci1, br, bi);

                acc_e_r2 = vfma(e, cr2, acc_e_r2);
                acc_e_i2 = vfma(e, ci2, acc_e_i2);
                acc_v_r2 = vfma(v, cr2, acc_v_r2);
                acc_v_i2 = vfma(v, ci2, acc_v_i2);

                const Vec cr3 = complex_mul_r(cr2, ci2, br, bi);
                const Vec ci3 = complex_mul_i(cr2, ci2, br, bi);

                acc_e_r3 = vfma(e, cr3, acc_e_r3);
                acc_e_i3 = vfma(e, ci3, acc_e_i3);
                acc_v_r3 = vfma(v, cr3, acc_v_r3);
                acc_v_i3 = vfma(v, ci3, acc_v_i3);

                const Vec cr4 = complex_mul_r(cr3, ci3, br, bi);
                const Vec ci4 = complex_mul_i(cr3, ci3, br, bi);
                storea(&current_r[k], cr4);
                storea(&current_i[k], ci4);
                k += kBatchSize;
            }

            float final_e_r0 = vreduce_add(acc_e_r0);
            float final_e_i0 = vreduce_add(acc_e_i0);
            float final_v_r0 = vreduce_add(acc_v_r0);
            float final_v_i0 = vreduce_add(acc_v_i0);

            float final_e_r1 = vreduce_add(acc_e_r1);
            float final_e_i1 = vreduce_add(acc_e_i1);
            float final_v_r1 = vreduce_add(acc_v_r1);
            float final_v_i1 = vreduce_add(acc_v_i1);

            float final_e_r2 = vreduce_add(acc_e_r2);
            float final_e_i2 = vreduce_add(acc_e_i2);
            float final_v_r2 = vreduce_add(acc_v_r2);
            float final_v_i2 = vreduce_add(acc_v_i2);

            float final_e_r3 = vreduce_add(acc_e_r3);
            float final_e_i3 = vreduce_add(acc_e_i3);
            float final_v_r3 = vreduce_add(acc_v_r3);
            float final_v_i3 = vreduce_add(acc_v_i3);

            for (; k < segment_len; ++k) {
                const float br = base_r[k];
                const float bi = base_i[k];
                float cur_r    = from_base ? br : current_r[k];
                float cur_i    = from_base ? bi : current_i[k];

                final_e_r0 += ts_e_seg[k] * cur_r;
                final_e_i0 += ts_e_seg[k] * cur_i;
                final_v_r0 += ts_v_seg[k] * cur_r;
                final_v_i0 += ts_v_seg[k] * cur_i;

                float nr = (cur_r * br) - (cur_i * bi);
                float ni = (cur_r * bi) + (cur_i * br);
                cur_r    = nr;
                cur_i    = ni;
                final_e_r1 += ts_e_seg[k] * cur_r;
                final_e_i1 += ts_e_seg[k] * cur_i;
                final_v_r1 += ts_v_seg[k] * cur_r;
                final_v_i1 += ts_v_seg[k] * cur_i;

                nr    = (cur_r * br) - (cur_i * bi);
                ni    = (cur_r * bi) + (cur_i * br);
                cur_r = nr;
                cur_i = ni;
                final_e_r2 += ts_e_seg[k] * cur_r;
                final_e_i2 += ts_e_seg[k] * cur_i;
                final_v_r2 += ts_v_seg[k] * cur_r;
                final_v_i2 += ts_v_seg[k] * cur_i;

                nr    = (cur_r * br) - (cur_i * bi);
                ni    = (cur_r * bi) + (cur_i * br);
                cur_r = nr;
                cur_i = ni;
                final_e_r3 += ts_e_seg[k] * cur_r;
                final_e_i3 += ts_e_seg[k] * cur_i;
                final_v_r3 += ts_v_seg[k] * cur_r;
                final_v_i3 += ts_v_seg[k] * cur_i;

                nr           = (cur_r * br) - (cur_i * bi);
                ni           = (cur_r * bi) + (cur_i * br);
                current_r[k] = nr;
                current_i[k] = ni;
            }

            fold_e_base[m + 0] = ComplexType(final_e_r0, final_e_i0);
            fold_v_base[m + 0] = ComplexType(final_v_r0, final_v_i0);
            fold_e_base[m + 1] = ComplexType(final_e_r1, final_e_i1);
            fold_v_base[m + 1] = ComplexType(final_v_r1, final_v_i1);
            fold_e_base[m + 2] = ComplexType(final_e_r2, final_e_i2);
            fold_v_base[m + 2] = ComplexType(final_v_r2, final_v_i2);
            fold_e_base[m + 3] = ComplexType(final_e_r3, final_e_i3);
            fold_v_base[m + 3] = ComplexType(final_v_r3, final_v_i3);
        }
#endif

        // 2-way harmonic unrolled kernel: optimal for 16-register architectures
        // (AVX2) and handles remaining even harmonics on 32-register
        // architectures.
        for (; m + 2 <= nbins_f; m += 2) {
            const bool from_base = (m == 1);
            Vec acc_e_r0 = vzero(), acc_e_i0 = vzero();
            Vec acc_v_r0 = vzero(), acc_v_i0 = vzero();
            Vec acc_e_r1 = vzero(), acc_e_i1 = vzero();
            Vec acc_v_r1 = vzero(), acc_v_i1 = vzero();

            SizeType k = 0;
            for (; k + (2 * kBatchSize) <= segment_len; k += 2 * kBatchSize) {
                // Sub-step 0: at k
                {
                    const Vec e   = loadu(&ts_e_seg[k]);
                    const Vec v   = loadu(&ts_v_seg[k]);
                    const Vec br  = loada(&base_r[k]);
                    const Vec bi  = loada(&base_i[k]);
                    const Vec cr0 = from_base ? br : loada(&current_r[k]);
                    const Vec ci0 = from_base ? bi : loada(&current_i[k]);

                    acc_e_r0 = vfma(e, cr0, acc_e_r0);
                    acc_e_i0 = vfma(e, ci0, acc_e_i0);
                    acc_v_r0 = vfma(v, cr0, acc_v_r0);
                    acc_v_i0 = vfma(v, ci0, acc_v_i0);

                    const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                    const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);

                    acc_e_r1 = vfma(e, cr1, acc_e_r1);
                    acc_e_i1 = vfma(e, ci1, acc_e_i1);
                    acc_v_r1 = vfma(v, cr1, acc_v_r1);
                    acc_v_i1 = vfma(v, ci1, acc_v_i1);

                    const Vec cr2 = complex_mul_r(cr1, ci1, br, bi);
                    const Vec ci2 = complex_mul_i(cr1, ci1, br, bi);
                    storea(&current_r[k], cr2);
                    storea(&current_i[k], ci2);
                }

                // Sub-step 1: at k + kBatchSize
                {
                    const SizeType k1 = k + kBatchSize;
                    const Vec e       = loadu(&ts_e_seg[k1]);
                    const Vec v       = loadu(&ts_v_seg[k1]);
                    const Vec br      = loada(&base_r[k1]);
                    const Vec bi      = loada(&base_i[k1]);
                    const Vec cr0     = from_base ? br : loada(&current_r[k1]);
                    const Vec ci0     = from_base ? bi : loada(&current_i[k1]);

                    acc_e_r0 = vfma(e, cr0, acc_e_r0);
                    acc_e_i0 = vfma(e, ci0, acc_e_i0);
                    acc_v_r0 = vfma(v, cr0, acc_v_r0);
                    acc_v_i0 = vfma(v, ci0, acc_v_i0);

                    const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                    const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);

                    acc_e_r1 = vfma(e, cr1, acc_e_r1);
                    acc_e_i1 = vfma(e, ci1, acc_e_i1);
                    acc_v_r1 = vfma(v, cr1, acc_v_r1);
                    acc_v_i1 = vfma(v, ci1, acc_v_i1);

                    const Vec cr2 = complex_mul_r(cr1, ci1, br, bi);
                    const Vec ci2 = complex_mul_i(cr1, ci1, br, bi);
                    storea(&current_r[k1], cr2);
                    storea(&current_i[k1], ci2);
                }
            }
            if (k + kBatchSize <= segment_len) {
                const Vec e   = loadu(&ts_e_seg[k]);
                const Vec v   = loadu(&ts_v_seg[k]);
                const Vec br  = loada(&base_r[k]);
                const Vec bi  = loada(&base_i[k]);
                const Vec cr0 = from_base ? br : loada(&current_r[k]);
                const Vec ci0 = from_base ? bi : loada(&current_i[k]);

                acc_e_r0 = vfma(e, cr0, acc_e_r0);
                acc_e_i0 = vfma(e, ci0, acc_e_i0);
                acc_v_r0 = vfma(v, cr0, acc_v_r0);
                acc_v_i0 = vfma(v, ci0, acc_v_i0);

                const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);

                acc_e_r1 = vfma(e, cr1, acc_e_r1);
                acc_e_i1 = vfma(e, ci1, acc_e_i1);
                acc_v_r1 = vfma(v, cr1, acc_v_r1);
                acc_v_i1 = vfma(v, ci1, acc_v_i1);

                const Vec cr2 = complex_mul_r(cr1, ci1, br, bi);
                const Vec ci2 = complex_mul_i(cr1, ci1, br, bi);
                storea(&current_r[k], cr2);
                storea(&current_i[k], ci2);
                k += kBatchSize;
            }

            float final_e_r0 = vreduce_add(acc_e_r0);
            float final_e_i0 = vreduce_add(acc_e_i0);
            float final_v_r0 = vreduce_add(acc_v_r0);
            float final_v_i0 = vreduce_add(acc_v_i0);

            float final_e_r1 = vreduce_add(acc_e_r1);
            float final_e_i1 = vreduce_add(acc_e_i1);
            float final_v_r1 = vreduce_add(acc_v_r1);
            float final_v_i1 = vreduce_add(acc_v_i1);

            for (; k < segment_len; ++k) {
                const float br = base_r[k];
                const float bi = base_i[k];
                float cur_r    = from_base ? br : current_r[k];
                float cur_i    = from_base ? bi : current_i[k];

                final_e_r0 += ts_e_seg[k] * cur_r;
                final_e_i0 += ts_e_seg[k] * cur_i;
                final_v_r0 += ts_v_seg[k] * cur_r;
                final_v_i0 += ts_v_seg[k] * cur_i;

                float nr = (cur_r * br) - (cur_i * bi);
                float ni = (cur_r * bi) + (cur_i * br);
                cur_r    = nr;
                cur_i    = ni;
                final_e_r1 += ts_e_seg[k] * cur_r;
                final_e_i1 += ts_e_seg[k] * cur_i;
                final_v_r1 += ts_v_seg[k] * cur_r;
                final_v_i1 += ts_v_seg[k] * cur_i;

                nr           = (cur_r * br) - (cur_i * bi);
                ni           = (cur_r * bi) + (cur_i * br);
                current_r[k] = nr;
                current_i[k] = ni;
            }

            fold_e_base[m + 0] = ComplexType(final_e_r0, final_e_i0);
            fold_v_base[m + 0] = ComplexType(final_v_r0, final_v_i0);
            fold_e_base[m + 1] = ComplexType(final_e_r1, final_e_i1);
            fold_v_base[m + 1] = ComplexType(final_v_r1, final_v_i1);
        }

        // 1-way harmonic cleanup for odd remainder
        for (; m < nbins_f; ++m) {
            const bool from_base = (m == 1);
            Vec acc_e_r0 = vzero(), acc_e_i0 = vzero();
            Vec acc_v_r0 = vzero(), acc_v_i0 = vzero();

            SizeType k = 0;
            for (; k + (2 * kBatchSize) <= segment_len; k += 2 * kBatchSize) {
                // Sub-step 0: at k
                {
                    const Vec e   = loadu(&ts_e_seg[k]);
                    const Vec v   = loadu(&ts_v_seg[k]);
                    const Vec br  = loada(&base_r[k]);
                    const Vec bi  = loada(&base_i[k]);
                    const Vec cr0 = from_base ? br : loada(&current_r[k]);
                    const Vec ci0 = from_base ? bi : loada(&current_i[k]);

                    acc_e_r0 = vfma(e, cr0, acc_e_r0);
                    acc_e_i0 = vfma(e, ci0, acc_e_i0);
                    acc_v_r0 = vfma(v, cr0, acc_v_r0);
                    acc_v_i0 = vfma(v, ci0, acc_v_i0);

                    const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                    const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);
                    storea(&current_r[k], cr1);
                    storea(&current_i[k], ci1);
                }

                // Sub-step 1: at k + kBatchSize
                {
                    const SizeType k1 = k + kBatchSize;
                    const Vec e       = loadu(&ts_e_seg[k1]);
                    const Vec v       = loadu(&ts_v_seg[k1]);
                    const Vec br      = loada(&base_r[k1]);
                    const Vec bi      = loada(&base_i[k1]);
                    const Vec cr0     = from_base ? br : loada(&current_r[k1]);
                    const Vec ci0     = from_base ? bi : loada(&current_i[k1]);

                    acc_e_r0 = vfma(e, cr0, acc_e_r0);
                    acc_e_i0 = vfma(e, ci0, acc_e_i0);
                    acc_v_r0 = vfma(v, cr0, acc_v_r0);
                    acc_v_i0 = vfma(v, ci0, acc_v_i0);

                    const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                    const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);
                    storea(&current_r[k1], cr1);
                    storea(&current_i[k1], ci1);
                }
            }
            if (k + kBatchSize <= segment_len) {
                const Vec e   = loadu(&ts_e_seg[k]);
                const Vec v   = loadu(&ts_v_seg[k]);
                const Vec br  = loada(&base_r[k]);
                const Vec bi  = loada(&base_i[k]);
                const Vec cr0 = from_base ? br : loada(&current_r[k]);
                const Vec ci0 = from_base ? bi : loada(&current_i[k]);

                acc_e_r0 = vfma(e, cr0, acc_e_r0);
                acc_e_i0 = vfma(e, ci0, acc_e_i0);
                acc_v_r0 = vfma(v, cr0, acc_v_r0);
                acc_v_i0 = vfma(v, ci0, acc_v_i0);

                const Vec cr1 = complex_mul_r(cr0, ci0, br, bi);
                const Vec ci1 = complex_mul_i(cr0, ci0, br, bi);
                storea(&current_r[k], cr1);
                storea(&current_i[k], ci1);
                k += kBatchSize;
            }

            float final_e_r0 = vreduce_add(acc_e_r0);
            float final_e_i0 = vreduce_add(acc_e_i0);
            float final_v_r0 = vreduce_add(acc_v_r0);
            float final_v_i0 = vreduce_add(acc_v_i0);

            for (; k < segment_len; ++k) {
                const float br = base_r[k];
                const float bi = base_i[k];
                float cur_r    = from_base ? br : current_r[k];
                float cur_i    = from_base ? bi : current_i[k];

                final_e_r0 += ts_e_seg[k] * cur_r;
                final_e_i0 += ts_e_seg[k] * cur_i;
                final_v_r0 += ts_v_seg[k] * cur_r;
                final_v_i0 += ts_v_seg[k] * cur_i;

                const float nr = (cur_r * br) - (cur_i * bi);
                const float ni = (cur_r * bi) + (cur_i * br);
                current_r[k]   = nr;
                current_i[k]   = ni;
            }

            fold_e_base[m] = ComplexType(final_e_r0, final_e_i0);
            fold_v_base[m] = ComplexType(final_v_r0, final_v_i0);
        }
    }
}

#endif

#undef BRUTE_FOLD_INLINE

} // namespace loki::brute_fold_intrinsics