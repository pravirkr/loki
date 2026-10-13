#pragma once

/**
 * @file boxcar_kernels.hpp
 * @brief CPU boxcar S/N kernels shared by lib/detection/score.cpp and
 * lib/cpu/score_cpu.cpp. Internal.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"

#include "lib/detail/utils.hpp"

namespace loki::detection::detail {

/// Branchless e/sqrt(v) folded into the circular prefix sum. The first
/// `nbins` entries match `circular_prefix_sum` of the normalised profile, so
/// `diff_max` sees the same values as the two-pass form.
inline void ev_circular_prefix(const float* __restrict__ ts_e,
                               const float* __restrict__ ts_v,
                               float* __restrict__ psum,
                               SizeType nbins,
                               SizeType nsum) noexcept {
    float run = 0.0F;
    for (SizeType j = 0; j < nbins; ++j) {
        const float variance = ts_v[j];
        const float sample =
            (variance > 0.0F) ? (ts_e[j] / std::sqrt(variance)) : 0.0F;
        run += sample;
        psum[j] = run;
    }
    if (nsum <= nbins) {
        return;
    }
    const float last_sum          = psum[nbins - 1];
    const SizeType first_wrap_end = std::min(nsum, 2 * nbins);
    for (SizeType i = nbins; i < first_wrap_end; ++i) {
        psum[i] = psum[i - nbins] + last_sum;
    }
    if (nsum > 2 * nbins) {
        for (SizeType i = 2 * nbins; i < nsum; ++i) {
            const auto wrap_count   = i / nbins;
            const auto pos_in_cycle = i % nbins;
            psum[i] = psum[pos_in_cycle] +
                      (static_cast<float>(wrap_count) * last_sum);
        }
    }
}

template <bool Is3D, bool FindMax>
void snr_boxcar_impl(const float* __restrict__ folds,
                     SizeType nprofiles,
                     SizeType nbins,
                     const SizeType* __restrict__ widths,
                     SizeType nwidths,
                     float* __restrict__ scores,
                     float stdnoise = 1.0F, // stdnoise is only used for 2D
                     int nthreads   = 1) {
    nthreads            = std::max(nthreads, 1);
    const SizeType wmax = *std::ranges::max_element(widths, widths + nwidths);

    // Precompute template parameters (h, b) for all widths
    std::vector<float> h_vals(nwidths);
    std::vector<float> b_vals(nwidths);
    for (SizeType iw = 0; iw < nwidths; ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
    const float inv_stdnoise = Is3D ? 1.0F : (1.0F / stdnoise);

#pragma omp parallel num_threads(nthreads) default(none)                       \
    shared(folds, widths, scores, nbins, nprofiles, nwidths, wmax, h_vals,     \
               b_vals, inv_stdnoise)
    {
        // Thread-local buffers
        std::vector<float> psum(nbins + wmax, 0.0F);

#pragma omp for
        for (SizeType i = 0; i < nprofiles; ++i) {
            if constexpr (Is3D) {
                const SizeType base_idx            = i * 2 * nbins;
                const float* __restrict__ ts_e_ptr = folds + base_idx;
                const float* __restrict__ ts_v_ptr = folds + base_idx + nbins;
                ev_circular_prefix(ts_e_ptr, ts_v_ptr, psum.data(), nbins,
                                   nbins + wmax);
            } else {
                const float* fold_ptr = folds + (i * nbins);
                utils::circular_prefix_sum(fold_ptr, psum.data(), nbins,
                                           nbins + wmax);
            }
            const float sum                    = psum[nbins - 1];
            const float* __restrict__ psum_ptr = psum.data();

            // Compute SNR for each width, find maximum
            // NOLINTNEXTLINE(misc-const-correctness): written when FindMax
            float max_snr = std::numeric_limits<float>::lowest();
            for (SizeType iw = 0; iw < nwidths; ++iw) {
                const auto dmax =
                    utils::diff_max(psum_ptr + widths[iw], psum_ptr, nbins);
                const float snr_base =
                    ((h_vals[iw] + b_vals[iw]) * dmax) - (b_vals[iw] * sum);
                const float snr = snr_base * inv_stdnoise;
                if constexpr (FindMax) {
                    max_snr = std::max(max_snr, snr);
                } else {
                    scores[(i * nwidths) + iw] = snr;
                }
            }
            if constexpr (FindMax) {
                scores[i] = max_snr;
            }
        }
    }
}

/// Boxcar template constants for every width: snr = (h + b) * dmax - b * sum.
/// The single definition keeps every 3D scorer on the same constants.
inline void boxcar_template_consts(SizeType nbins,
                                   const SizeType* __restrict__ widths,
                                   SizeType nwidths,
                                   float* __restrict__ h_vals,
                                   float* __restrict__ b_vals) noexcept {
    for (SizeType iw = 0; iw < nwidths; ++iw) {
        const auto w = widths[iw];
        h_vals[iw]   = std::sqrt(static_cast<float>(nbins - w) /
                                 static_cast<float>(nbins * w));
        b_vals[iw] =
            static_cast<float>(w) * h_vals[iw] / static_cast<float>(nbins - w);
    }
}

/**
 * @brief Boxcar S/N of (e, v) profiles, several profiles per SIMD register.
 *
 * Per profile the arithmetic is the scalar scorer's (`ev_circular_prefix` +
 * `diff_max`) as GCC emits it under -ffast-math: x = e * rsqrt(v) with the
 * legacy `rsqrtps` estimate and one Newton step, fused into the running
 * prefix sum (one FMA per bin). Each SIMD lane owns one profile, so the
 * sequential scan is kept bit for bit, and its latency is hidden across 16
 * profiles in flight. Holds per-thread scratch for one (nbins, widths);
 * reuse it across calls. Without AVX2/FMA it scores profile by profile.
 */
class BoxcarScorer3D {
public:
    BoxcarScorer3D(SizeType nbins, const SizeType* widths, SizeType nwidths)
        : m_nbins(nbins),
          m_widths(widths, widths + nwidths),
          m_wmax(*std::max_element(widths, widths + nwidths)),
          m_hb(nwidths),
          m_b(nwidths),
          m_psum((nbins + m_wmax) * kGroup) {
        std::vector<float> h(nwidths);
        boxcar_template_consts(nbins, widths, nwidths, h.data(), m_b.data());
        for (SizeType iw = 0; iw < nwidths; ++iw) {
            m_hb[iw] = h[iw] + m_b[iw];
        }
        m_snr.resize(nwidths * kGroup);
        m_max.resize(nwidths * kGroup);
        m_dummy.assign(2 * nbins, 1.0F);
        std::fill_n(m_dummy.begin(), nbins, 0.0F);
    }

    [[nodiscard]] SizeType nwidths() const noexcept { return m_widths.size(); }

    /// Calls emit(profile, iw, snr) for every profile in [0, nprofiles) and
    /// width, in (profile, width) order. Profiles are [n][e, v][nbins].
    template <typename Emit>
    void
    score(const float* __restrict__ folds, SizeType nprofiles, Emit&& emit) {
        const SizeType nw = m_widths.size();
        for (SizeType p0 = 0; p0 < nprofiles; p0 += kGroup) {
            const SizeType nvalid = std::min(kGroup, nprofiles - p0);
            score_group(folds + (p0 * 2 * m_nbins), nvalid);
            for (SizeType lane = 0; lane < nvalid; ++lane) {
                for (SizeType iw = 0; iw < nw; ++iw) {
                    emit(p0 + lane, iw, m_snr[(iw * kGroup) + lane]);
                }
            }
        }
    }

    /// Lane stride of score_lanes() input.
    static constexpr SizeType kLaneStride = 16;

    /// As score(), for nvalid <= kLaneStride profiles given in lane layout:
    /// e and v are [nbins][kLaneStride], lane = profile (lane_fft output).
    /// The same arithmetic per profile, without the transposes.
    template <typename Emit>
    void score_lanes(const float* __restrict__ e,
                     const float* __restrict__ v,
                     SizeType nvalid,
                     Emit&& emit) {
        const SizeType nw = m_widths.size();
#if defined(__AVX2__) && defined(__FMA__)
        static_assert(kGroup == kLaneStride);
        scan_lanes(e, v);
        finish_group();
        for (SizeType lane = 0; lane < nvalid; ++lane) {
            for (SizeType iw = 0; iw < nw; ++iw) {
                emit(lane, iw, m_snr[(iw * kGroup) + lane]);
            }
        }
#else
        const SizeType nbins = m_nbins;
        for (SizeType lane = 0; lane < nvalid; ++lane) {
            for (SizeType j = 0; j < nbins; ++j) {
                m_row[j]         = e[(j * kLaneStride) + lane];
                m_row[nbins + j] = v[(j * kLaneStride) + lane];
            }
            score_group(m_row.data(), 1);
            for (SizeType iw = 0; iw < nw; ++iw) {
                emit(lane, iw, m_snr[iw]);
            }
        }
#endif
    }

private:
#if defined(__AVX2__) && defined(__FMA__)
    static constexpr SizeType kGroup = 16; // two 8-lane chains in flight
#else
    static constexpr SizeType kGroup = 1;
#endif
    SizeType m_nbins;
    std::vector<SizeType> m_widths;
    SizeType m_wmax;
    std::vector<float> m_hb; // h + b per width
    std::vector<float> m_b;
    std::vector<float> m_psum;  // [nbins + wmax][kGroup]
    std::vector<float> m_snr;   // [width][kGroup]
    std::vector<float> m_max;   // [width][kGroup] running maxima
    std::vector<float> m_dummy; // e = 0, v = 1: pads the last group
#if !(defined(__AVX2__) && defined(__FMA__))
    std::vector<float> m_row = std::vector<float>(2 * m_nbins);
#endif

#if defined(__AVX2__) && defined(__FMA__)
    /// Transpose an 8 x 8 block: rows[r][c0 + c] -> dst[c * stride + r].
    static void transpose8(const float* const* rows,
                           SizeType c0,
                           float* dst,
                           SizeType stride) noexcept {
        const __m256 r0 = _mm256_loadu_ps(rows[0] + c0);
        const __m256 r1 = _mm256_loadu_ps(rows[1] + c0);
        const __m256 r2 = _mm256_loadu_ps(rows[2] + c0);
        const __m256 r3 = _mm256_loadu_ps(rows[3] + c0);
        const __m256 r4 = _mm256_loadu_ps(rows[4] + c0);
        const __m256 r5 = _mm256_loadu_ps(rows[5] + c0);
        const __m256 r6 = _mm256_loadu_ps(rows[6] + c0);
        const __m256 r7 = _mm256_loadu_ps(rows[7] + c0);
        const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
        const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
        const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
        const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
        const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
        const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
        const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
        const __m256 t7 = _mm256_unpackhi_ps(r6, r7);
        const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
        const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
        const __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
        _mm256_storeu_ps(dst + (0 * stride),
                         _mm256_permute2f128_ps(u0, u4, 0x20));
        _mm256_storeu_ps(dst + (1 * stride),
                         _mm256_permute2f128_ps(u1, u5, 0x20));
        _mm256_storeu_ps(dst + (2 * stride),
                         _mm256_permute2f128_ps(u2, u6, 0x20));
        _mm256_storeu_ps(dst + (3 * stride),
                         _mm256_permute2f128_ps(u3, u7, 0x20));
        _mm256_storeu_ps(dst + (4 * stride),
                         _mm256_permute2f128_ps(u0, u4, 0x31));
        _mm256_storeu_ps(dst + (5 * stride),
                         _mm256_permute2f128_ps(u1, u5, 0x31));
        _mm256_storeu_ps(dst + (6 * stride),
                         _mm256_permute2f128_ps(u2, u6, 0x31));
        _mm256_storeu_ps(dst + (7 * stride),
                         _mm256_permute2f128_ps(u3, u7, 0x31));
    }

    /// Normalised sample e * y, y = rsqrt(v) refined: the scalar lowering.
    static __m256 inv_sigma(__m256 v) noexcept {
        const __m256 r  = _mm256_rsqrt_ps(v);
        const __m256 t1 = _mm256_mul_ps(v, r);
        const __m256 t2 = _mm256_fmadd_ps(r, t1, _mm256_set1_ps(-3.0F));
        const __m256 y =
            _mm256_mul_ps(t2, _mm256_mul_ps(r, _mm256_set1_ps(-0.5F)));
        // v <= 0 contributes nothing (the scalar code skips the add).
        return _mm256_and_ps(y,
                             _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
    }

    void score_group(const float* __restrict__ folds, SizeType nvalid) {
        const SizeType nbins = m_nbins;
        std::array<const float*, kGroup> rows_e{};
        std::array<const float*, kGroup> rows_v{};
        for (SizeType lane = 0; lane < kGroup; ++lane) {
            const float* row =
                lane < nvalid ? folds + (lane * 2 * nbins) : m_dummy.data();
            rows_e[lane] = row;
            rows_v[lane] = row + nbins;
        }
        float* __restrict__ ps = m_psum.data();
        // Transpose e into psum[bin][lane] and v into the scratch tail; the
        // scan then overwrites psum in place.
        std::array<float, 8 * kGroup> vt{};
        __m256 run0          = _mm256_setzero_ps();
        __m256 run1          = _mm256_setzero_ps();
        const SizeType nfull = nbins - (nbins % 8);
        for (SizeType c0 = 0; c0 < nbins; c0 += 8) {
            const SizeType ncol = std::min<SizeType>(8, nbins - c0);
            if (c0 < nfull) {
                transpose8(rows_e.data(), c0, ps + (c0 * kGroup), kGroup);
                transpose8(rows_e.data() + 8, c0, ps + (c0 * kGroup) + 8,
                           kGroup);
                transpose8(rows_v.data(), c0, vt.data(), kGroup);
                transpose8(rows_v.data() + 8, c0, vt.data() + 8, kGroup);
            } else {
                for (SizeType c = 0; c < ncol; ++c) {
                    for (SizeType lane = 0; lane < kGroup; ++lane) {
                        ps[((c0 + c) * kGroup) + lane] = rows_e[lane][c0 + c];
                        vt[(c * kGroup) + lane]        = rows_v[lane][c0 + c];
                    }
                }
            }
            for (SizeType c = 0; c < ncol; ++c) {
                float* row        = ps + ((c0 + c) * kGroup);
                const float* vrow = vt.data() + (c * kGroup);
                run0 = _mm256_fmadd_ps(_mm256_loadu_ps(row),
                                       inv_sigma(_mm256_loadu_ps(vrow)), run0);
                run1 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(row + 8),
                                    inv_sigma(_mm256_loadu_ps(vrow + 8)), run1);
                _mm256_storeu_ps(row, run0);
                _mm256_storeu_ps(row + 8, run1);
            }
        }
        finish_group();
    }

    /// The prefix scan of score_group() on lane-layout e and v.
    void scan_lanes(const float* __restrict__ e,
                    const float* __restrict__ v) noexcept {
        float* __restrict__ ps = m_psum.data();
        __m256 run0            = _mm256_setzero_ps();
        __m256 run1            = _mm256_setzero_ps();
        for (SizeType j = 0; j < m_nbins; ++j) {
            const SizeType at = j * kGroup;
            run0 = _mm256_fmadd_ps(_mm256_loadu_ps(e + at),
                                   inv_sigma(_mm256_loadu_ps(v + at)), run0);
            run1 =
                _mm256_fmadd_ps(_mm256_loadu_ps(e + at + 8),
                                inv_sigma(_mm256_loadu_ps(v + at + 8)), run1);
            _mm256_storeu_ps(ps + at, run0);
            _mm256_storeu_ps(ps + at + 8, run1);
        }
    }

    /// Circular extension, width maxima and S/N of the scanned group in psum.
    void finish_group() noexcept {
        const SizeType nbins   = m_nbins;
        float* __restrict__ ps = m_psum.data();
        // Circular extension, as ev_circular_prefix.
        const SizeType nsum      = nbins + m_wmax;
        const SizeType first_end = std::min(nsum, 2 * nbins);
        const float* last        = ps + ((nbins - 1) * kGroup);
        for (SizeType i = nbins; i < first_end; ++i) {
            for (SizeType lane = 0; lane < kGroup; ++lane) {
                ps[(i * kGroup) + lane] =
                    ps[((i - nbins) * kGroup) + lane] + last[lane];
            }
        }
        for (SizeType i = 2 * nbins; i < nsum; ++i) {
            const auto wraps = static_cast<float>(i / nbins);
            for (SizeType lane = 0; lane < kGroup; ++lane) {
                ps[(i * kGroup) + lane] = std::fma(
                    wraps, last[lane], ps[((i % nbins) * kGroup) + lane]);
            }
        }
        // Per width: max over start bins of psum[j + w] - psum[j] (exact in any
        // order), then snr = fma(h + b, dmax, -(b * sum)). Start bins are
        // tiled so a tile's rows stay in L1 across every width.
        constexpr SizeType kTile = 128;
        const SizeType nw        = m_widths.size();
        float* __restrict__ mx   = m_max.data();
        std::fill_n(mx, nw * kGroup, std::numeric_limits<float>::lowest());
        for (SizeType j0 = 0; j0 < nbins; j0 += kTile) {
            const SizeType j1 = std::min(nbins, j0 + kTile);
            for (SizeType iw = 0; iw < nw; ++iw) {
                const float* hi = ps + (m_widths[iw] * kGroup);
                float* m_iw     = mx + (iw * kGroup);
#if defined(__AVX512F__)
                __m512 m = _mm512_loadu_ps(m_iw);
                for (SizeType j = j0; j < j1; ++j) {
                    m = _mm512_max_ps(
                        m, _mm512_sub_ps(_mm512_loadu_ps(hi + (j * kGroup)),
                                         _mm512_loadu_ps(ps + (j * kGroup))));
                }
                _mm512_storeu_ps(m_iw, m);
#else
                __m256 m0 = _mm256_loadu_ps(m_iw);
                __m256 m1 = _mm256_loadu_ps(m_iw + 8);
                for (SizeType j = j0; j < j1; ++j) {
                    const float* lo = ps + (j * kGroup);
                    m0              = _mm256_max_ps(
                        m0, _mm256_sub_ps(_mm256_loadu_ps(hi + (j * kGroup)),
                                          _mm256_loadu_ps(lo)));
                    m1 = _mm256_max_ps(
                        m1,
                        _mm256_sub_ps(_mm256_loadu_ps(hi + (j * kGroup) + 8),
                                      _mm256_loadu_ps(lo + 8)));
                }
                _mm256_storeu_ps(m_iw, m0);
                _mm256_storeu_ps(m_iw + 8, m1);
#endif
            }
        }
        for (SizeType iw = 0; iw < nw; ++iw) {
            const __m256 hb = _mm256_set1_ps(m_hb[iw]);
            const __m256 b  = _mm256_set1_ps(m_b[iw]);
            for (SizeType half = 0; half < kGroup; half += 8) {
                const __m256 dmax = _mm256_loadu_ps(mx + (iw * kGroup) + half);
                _mm256_storeu_ps(
                    m_snr.data() + (iw * kGroup) + half,
                    _mm256_fmsub_ps(
                        hb, dmax,
                        _mm256_mul_ps(b, _mm256_loadu_ps(last + half))));
            }
        }
    }
#else
    void score_group(const float* __restrict__ folds, SizeType /*nvalid*/) {
        const SizeType nbins = m_nbins;
        ev_circular_prefix(folds, folds + nbins, m_psum.data(), nbins,
                           nbins + m_wmax);
        const float sum = m_psum[nbins - 1];
        for (SizeType iw = 0; iw < m_widths.size(); ++iw) {
            const float dmax = utils::diff_max(m_psum.data() + m_widths[iw],
                                               m_psum.data(), nbins);
            m_snr[iw]        = (m_hb[iw] * dmax) - (m_b[iw] * sum);
        }
    }
#endif
};

/// Score profiles [n][e, v][nbins] and append the (profile_base + p) *
/// nwidths + iw indices of S/N >= threshold, in index order.
inline void append_boxcar_hits(BoxcarScorer3D& scorer,
                               const float* folds,
                               SizeType profile_base,
                               SizeType nprofiles,
                               float threshold,
                               std::vector<SnrHit>& hits) {
    const SizeType nwidths = scorer.nwidths();
    scorer.score(folds, nprofiles,
                 [&hits, threshold, profile_base,
                  nwidths](SizeType p, SizeType iw, float snr) {
                     if (snr >= threshold) {
                         hits.push_back(SnrHit{
                             .score_index = static_cast<uint32_t>(
                                 ((profile_base + p) * nwidths) + iw),
                             .snr = snr,
                         });
                     }
                 });
}

} // namespace loki::detection::detail
