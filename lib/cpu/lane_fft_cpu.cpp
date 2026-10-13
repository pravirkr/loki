#include "lib/cpu/lane_fft.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <mutex>
#include <numbers>

#include <omp.h>
#include <xsimd/xsimd.hpp>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "lib/detail/error_check.hpp"

namespace loki::lane_fft {

namespace {

using Batch               = xsimd::batch<float>;
constexpr SizeType kWidth = Batch::size;
static_assert(kLanes % kWidth == 0,
              "kLanes must be a multiple of the SIMD width");

struct CBatch {
    Batch re;
    Batch im;
};

[[gnu::always_inline]] inline CBatch
load(const float* re, const float* im, SizeType idx) noexcept {
    return {Batch::load_aligned(re + idx), Batch::load_aligned(im + idx)};
}

[[gnu::always_inline]] inline void
store(float* re, float* im, SizeType idx, CBatch v) noexcept {
    v.re.store_aligned(re + idx);
    v.im.store_aligned(im + idx);
}

/// v * (wr + i wi) (Inverse) or v * (wr - i wi) (forward).
template <bool Inverse>
[[gnu::always_inline]] inline CBatch twiddle(CBatch v,
                                             const float* w) noexcept {
    const Batch wr(w[0]);
    const Batch wi(Inverse ? w[1] : -w[1]);
    return {xsimd::fms(v.re, wr, v.im * wi), xsimd::fma(v.re, wi, v.im * wr)};
}

#if defined(__AVX2__)
constexpr SizeType kBlock = 8;

/// dst[c][0..8) = src[0..8)[c] for an 8 x 8 block of floats.
[[gnu::always_inline]] inline void
transpose8(const std::array<const float*, kBlock>& src,
           const std::array<float*, kBlock>& dst) noexcept {
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
#else
constexpr SizeType kBlock = kWidth;

/// dst[c][0..kBlock) = src[0..kBlock)[c] (xsimd, any architecture).
inline void transpose8(const std::array<const float*, kBlock>& src,
                       const std::array<float*, kBlock>& dst) noexcept {
    std::array<Batch, kBlock> block;
    for (SizeType r = 0; r < kBlock; ++r) {
        block[r] = Batch::load_unaligned(src[r]);
    }
    xsimd::transpose(block.data(), block.data() + kBlock);
    for (SizeType r = 0; r < kBlock; ++r) {
        block[r].store_unaligned(dst[r]);
    }
}
#endif
static_assert(kLanes % kBlock == 0, "kLanes must be a multiple of the block");

/// dst[c][lane] = rows[lane][c] for c < ncols; dst has a kLanes stride.
void rows_to_lanes(const std::array<const float*, kLanes>& rows,
                   SizeType ncols,
                   float* __restrict__ dst) noexcept {
    const SizeType nfull = ncols - (ncols % kBlock);
    std::array<const float*, kBlock> src{};
    std::array<float*, kBlock> out{};
    for (SizeType c0 = 0; c0 < nfull; c0 += kBlock) {
        for (SizeType h = 0; h < kLanes; h += kBlock) {
            for (SizeType r = 0; r < kBlock; ++r) {
                src[r] = rows[h + r] + c0;
                out[r] = dst + ((c0 + r) * kLanes) + h;
            }
            transpose8(src, out);
        }
    }
    for (SizeType c = nfull; c < ncols; ++c) {
        for (SizeType lane = 0; lane < kLanes; ++lane) {
            dst[(c * kLanes) + lane] = rows[lane][c];
        }
    }
}

/// rows[lane][c] = src[c][lane] for lane < nrows and c < ncols.
void lanes_to_rows(const float* __restrict__ src,
                   SizeType ncols,
                   const std::array<float*, kLanes>& rows,
                   SizeType nrows) noexcept {
    const SizeType nfull = ncols - (ncols % kBlock);
    std::array<const float*, kBlock> in{};
    std::array<float*, kBlock> out{};
    std::array<float, kBlock> sink{};
    for (SizeType c0 = 0; c0 < nfull; c0 += kBlock) {
        for (SizeType h = 0; h < nrows; h += kBlock) {
            for (SizeType r = 0; r < kBlock; ++r) {
                in[r]  = src + ((c0 + r) * kLanes) + h;
                out[r] = h + r < nrows ? rows[h + r] + c0 : sink.data();
            }
            transpose8(in, out);
        }
    }
    for (SizeType c = nfull; c < ncols; ++c) {
        for (SizeType lane = 0; lane < nrows; ++lane) {
            rows[lane][c] = src[(c * kLanes) + lane];
        }
    }
}

} // namespace

namespace {

/// Radices of the direct transform, in stage order.
constexpr std::array<SizeType, 7> kRadices{4, 2, 3, 5, 7, 11, 13};

/// Whether n factors over kRadices.
bool is_direct(SizeType n) noexcept {
    for (const SizeType r : kRadices) {
        while (n % r == 0) {
            n /= r;
        }
    }
    return n == 1;
}

/// Smallest 2^a 3^b 5^c >= n.
SizeType smooth_ceil(SizeType n) noexcept {
    SizeType best = std::bit_ceil(n);
    for (SizeType p5 = 1; p5 < best; p5 *= 5) {
        for (SizeType p35 = p5; p35 < best; p35 *= 3) {
            SizeType m = p35;
            while (m < n) {
                m *= 2;
            }
            best = std::min(best, m);
        }
    }
    return best;
}

/// e^(i 2 pi num / den) in double, the angle reduced exactly first.
std::pair<double, double> unit_root(SizeType num, SizeType den) noexcept {
    const double angle = 2.0 * std::numbers::pi *
                         static_cast<double>(num % den) /
                         static_cast<double>(den);
    return {std::cos(angle), std::sin(angle)};
}

[[gnu::always_inline]] inline CBatch cadd(CBatch a, CBatch b) noexcept {
    return {a.re + b.re, a.im + b.im};
}
[[gnu::always_inline]] inline CBatch csub(CBatch a, CBatch b) noexcept {
    return {a.re - b.re, a.im - b.im};
}
/// +i v (Inverse) or -i v.
template <bool Inverse>
[[gnu::always_inline]] inline CBatch rot(CBatch v) noexcept {
    return Inverse ? CBatch{-v.im, v.re} : CBatch{v.im, -v.re};
}

/// cos / sin (2 pi j k / R), j, k = 1 .. (R - 1) / 2, at [(j - 1) H + k - 1].
template <SizeType R> struct OddRadixTable {
    static constexpr SizeType kH = (R - 1) / 2;
    std::array<float, (kH * kH) + 1> cos{};
    std::array<float, (kH * kH) + 1> sin{};
    OddRadixTable() noexcept {
        for (SizeType j = 1; j <= kH; ++j) {
            for (SizeType k = 1; k <= kH; ++k) {
                const auto [c, sn]            = unit_root(j * k, R);
                cos[((j - 1) * kH) + (k - 1)] = static_cast<float>(c);
                sin[((j - 1) * kH) + (k - 1)] = static_cast<float>(sn);
            }
        }
    }
};

/// One Stockham decimation-in-frequency stage of radix R: for every
/// sub-transform q < stride and butterfly p < len / R, the R-point DFT of
/// x[q + stride (p + j len / R)], twiddled by w^(p k), into
/// y[q + stride (R p + k)]. Twiddle is false on the last stage (w = 1).
template <bool Inverse, SizeType R, bool Twiddle>
void radix_stage(const float* __restrict__ ar,
                 const float* __restrict__ ai,
                 float* __restrict__ br,
                 float* __restrict__ bi,
                 SizeType len,
                 SizeType stride,
                 const float* __restrict__ tw) noexcept {
    constexpr SizeType kH = (R - 1) / 2;
    static const OddRadixTable<R> table;
    const auto& cosv  = table.cos;
    const auto& sinv  = table.sin;
    const SizeType n1 = len / R;
    for (SizeType p = 0; p < n1; ++p) {
        const float* wp = tw + (2 * (R - 1) * p);
        for (SizeType q = 0; q < stride; ++q) {
            const SizeType in  = (q + (stride * p)) * kLanes;
            const SizeType ins = stride * n1 * kLanes;
            const SizeType out = (q + (stride * R * p)) * kLanes;
            const SizeType os  = stride * kLanes;
            for (SizeType h = 0; h < kLanes; h += kWidth) {
                std::array<CBatch, R> x;
                for (SizeType j = 0; j < R; ++j) {
                    x[j] = load(ar, ai, in + (j * ins) + h);
                }
                std::array<CBatch, R> y;
                if constexpr (R == 2) {
                    y[0] = cadd(x[0], x[1]);
                    y[1] = csub(x[0], x[1]);
                } else if constexpr (R == 4) {
                    const CBatch apc = cadd(x[0], x[2]);
                    const CBatch amc = csub(x[0], x[2]);
                    const CBatch bpd = cadd(x[1], x[3]);
                    const CBatch jbd = rot<Inverse>(csub(x[1], x[3]));
                    y[0]             = cadd(apc, bpd);
                    y[1]             = cadd(amc, jbd);
                    y[2]             = csub(apc, bpd);
                    y[3]             = csub(amc, jbd);
                } else {
                    // Pair x[j] with x[R - j]: y[k], y[R - k] = P +- (+-i) Q.
                    std::array<CBatch, kH> sum;
                    std::array<CBatch, kH> dif;
                    y[0] = x[0];
                    for (SizeType j = 0; j < kH; ++j) {
                        sum[j] = cadd(x[j + 1], x[R - 1 - j]);
                        dif[j] = csub(x[j + 1], x[R - 1 - j]);
                        y[0]   = cadd(y[0], sum[j]);
                    }
                    for (SizeType k = 0; k < kH; ++k) {
                        CBatch pk = x[0];
                        CBatch qk{Batch(0.0F), Batch(0.0F)};
                        for (SizeType j = 0; j < kH; ++j) {
                            const Batch c(cosv[(j * kH) + k]);
                            const Batch sn(sinv[(j * kH) + k]);
                            pk = {xsimd::fma(sum[j].re, c, pk.re),
                                  xsimd::fma(sum[j].im, c, pk.im)};
                            qk = {xsimd::fma(dif[j].re, sn, qk.re),
                                  xsimd::fma(dif[j].im, sn, qk.im)};
                        }
                        const CBatch iq = rot<Inverse>(qk);
                        y[k + 1]        = cadd(pk, iq);
                        y[R - 1 - k]    = csub(pk, iq);
                    }
                }
                store(br, bi, out + h, y[0]);
                for (SizeType k = 1; k < R; ++k) {
                    store(br, bi, out + (k * os) + h,
                          Twiddle ? twiddle<Inverse>(y[k], wp + (2 * (k - 1)))
                                  : y[k]);
                }
            }
        }
    }
}

template <bool Inverse, SizeType R>
void run_radix(const float* ar,
               const float* ai,
               float* br,
               float* bi,
               SizeType len,
               SizeType stride,
               const float* tw) noexcept {
    if (len == R) {
        radix_stage<Inverse, R, false>(ar, ai, br, bi, len, stride, tw);
    } else {
        radix_stage<Inverse, R, true>(ar, ai, br, bi, len, stride, tw);
    }
}

/// (a + ib) * (c + id) per lane, c + id broadcast.
[[gnu::always_inline]] inline CBatch
cmul_scalar(CBatch v, float c, float d) noexcept {
    const Batch cr(c);
    const Batch ci(d);
    return {xsimd::fms(v.re, cr, v.im * ci), xsimd::fma(v.re, ci, v.im * cr)};
}

} // namespace

struct EVTransform::Plan {
    struct Stage {
        SizeType radix;
        SizeType len;      // sub-transform length this stage splits
        SizeType stride;   // number of interleaved sub-transforms
        SizeType twiddles; // offset in `twiddles`
    };

    SizeType n;
    SizeType fft_size; // n, or the Bluestein convolution length
    std::vector<Stage> stages;
    // Per stage, per butterfly p: w^(p k), k = 1 .. radix - 1, (re, im) of
    // the inverse transform (w = e^(2 pi i / len)); the forward one uses
    // their conjugates.
    std::vector<float> twiddles;
    // Bluestein: chirp e^(i pi k^2 / n), k < n, and the transformed
    // convolution kernels (scaled by 1 / fft_size) of the inverse and the
    // forward transform; empty for a direct transform.
    std::vector<float> chirp;
    std::vector<float> kernel_inv;
    std::vector<float> kernel_fwd;

    explicit Plan(SizeType n_in) : n(n_in) {
        if (is_direct(n)) {
            fft_size = n;
            plan_stages();
        } else {
            fft_size = smooth_ceil((2 * n) - 1);
            plan_stages();
            plan_bluestein();
        }
    }

    void plan_stages() {
        SizeType len    = fft_size;
        SizeType stride = 1;
        for (const SizeType radix : kRadices) {
            while (len % radix == 0 && len > 1) {
                stages.push_back({.radix    = radix,
                                  .len      = len,
                                  .stride   = stride,
                                  .twiddles = twiddles.size()});
                for (SizeType p = 0; p < len / radix; ++p) {
                    for (SizeType k = 1; k < radix; ++k) {
                        const auto [c, sn] = unit_root(p * k, len);
                        twiddles.push_back(static_cast<float>(c));
                        twiddles.push_back(static_cast<float>(sn));
                    }
                }
                len /= radix;
                stride *= radix;
            }
        }
    }

    void plan_bluestein() {
        const SizeType m = fft_size;
        // chirp c_k = e^(i pi k^2 / n): k^2 reduced mod 2n keeps the angle
        // exact.
        chirp.resize(2 * n);
        for (SizeType k = 0; k < n; ++k) {
            const auto [c, sn] = unit_root((k * k) % (2 * n), 2 * n);
            chirp[2 * k]       = static_cast<float>(c);
            chirp[(2 * k) + 1] = static_cast<float>(sn);
        }
        // Kernel b_j = conj(c_j) (inverse) or c_j (forward) on the circle of
        // length m, transformed by the forward m-point DFT in double and
        // scaled by 1 / m (the unnormalised inverse that follows).
        std::vector<double> root(2 * m);
        for (SizeType t = 0; t < m; ++t) {
            const auto [c, sn] = unit_root(t, m);
            root[2 * t]        = c;
            root[(2 * t) + 1]  = -sn;
        }
        for (const bool inverse : {true, false}) {
            std::vector<double> b(2 * m, 0.0);
            for (SizeType j = 0; j < n; ++j) {
                const auto [c, sn] = unit_root((j * j) % (2 * n), 2 * n);
                const double im    = inverse ? -sn : sn;
                b[2 * j]           = c;
                b[(2 * j) + 1]     = im;
                if (j > 0) {
                    b[2 * (m - j)]       = c;
                    b[(2 * (m - j)) + 1] = im;
                }
            }
            // b is zero on j in [n, m - n]: sum the two nonzero arcs only.
            const auto accumulate = [&](SizeType j, SizeType t, double& re,
                                        double& im) {
                re += (b[2 * j] * root[2 * t]) -
                      (b[(2 * j) + 1] * root[(2 * t) + 1]);
                im += (b[2 * j] * root[(2 * t) + 1]) +
                      (b[(2 * j) + 1] * root[2 * t]);
            };
            auto& kernel = inverse ? kernel_inv : kernel_fwd;
            kernel.resize(2 * m);
            for (SizeType k = 0; k < m; ++k) {
                double re  = 0.0;
                double im  = 0.0;
                SizeType t = 0; // j k mod m
                for (SizeType j = 0; j < n; ++j) {
                    accumulate(j, t, re, im);
                    t += k;
                    t -= t >= m ? m : 0;
                }
                t = ((m - n + 1) * k) % m;
                for (SizeType j = m - n + 1; j < m; ++j) {
                    accumulate(j, t, re, im);
                    t += k;
                    t -= t >= m ? m : 0;
                }
                kernel[2 * k] = static_cast<float>(re / static_cast<double>(m));
                kernel[(2 * k) + 1] =
                    static_cast<float>(im / static_cast<double>(m));
            }
        }
    }
};

std::shared_ptr<const EVTransform::Plan> EVTransform::get_plan(SizeType n) {
    // Plans are immutable and small: one per size for the process.
    static std::mutex mutex;
    static std::map<SizeType, std::shared_ptr<const Plan>> cache;
    const std::scoped_lock lock(mutex);
    auto& plan = cache[n];
    if (!plan) {
        plan = std::make_shared<const Plan>(n);
    }
    return plan;
}

bool EVTransform::supports(SizeType n) noexcept { return n >= 2; }

void EVTransform::prepare(SizeType n) {
    error_check::check(supports(n), "EVTransform: n must be >= 2");
    static_cast<void>(get_plan(n));
}

EVTransform::EVTransform(SizeType n) : m_n(n) {
    error_check::check(supports(n), "EVTransform: n must be >= 2");
    m_plan = get_plan(n);
    // Four buffers, staggered by a few cache lines so the butterflies'
    // streams do not alias in the L1 sets.
    const SizeType len     = std::max(n + 2, m_plan->fft_size) * kLanes;
    const SizeType stagger = 16;
    m_scratch.resize((4 * len) + (6 * stagger));
    m_z_re = m_scratch.data();
    m_z_im = m_z_re + len + stagger;
    m_y_re = m_z_im + len + (2 * stagger);
    m_y_im = m_y_re + len + (3 * stagger);
    m_zeros.assign(n + 2, 0.0F);
}

template <bool Inverse>
std::pair<float*, float*> EVTransform::run_stages(float* re,
                                                  float* im) noexcept {
    // Ping-pong between (re, im) and the other buffer pair.
    const bool in_z = re == m_z_re;
    float* ar       = re;
    float* ai       = im;
    float* br       = in_z ? m_y_re : m_z_re;
    float* bi       = in_z ? m_y_im : m_z_im;
    for (const Plan::Stage& st : m_plan->stages) {
        const float* tw = m_plan->twiddles.data() + st.twiddles;
        switch (st.radix) {
        case 2:
            run_radix<Inverse, 2>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        case 3:
            run_radix<Inverse, 3>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        case 4:
            run_radix<Inverse, 4>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        case 5:
            run_radix<Inverse, 5>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        case 7:
            run_radix<Inverse, 7>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        case 11:
            run_radix<Inverse, 11>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        default:
            run_radix<Inverse, 13>(ar, ai, br, bi, st.len, st.stride, tw);
            break;
        }
        std::swap(ar, br);
        std::swap(ai, bi);
    }
    return {ar, ai};
}

template <bool Inverse> void EVTransform::transform() noexcept {
    const Plan& plan = *m_plan;
    if (plan.chirp.empty()) {
        const auto [re, im] = run_stages<Inverse>(m_z_re, m_z_im);
        m_res_re            = re;
        m_res_im            = im;
        return;
    }
    // Bluestein: y_j = c_j sum_k (z_k c_k) conj(c_(j-k)), with the chirp
    // c = e^(+-i pi k^2 / n); the convolution is a forward FFT, the product
    // with the kernel, and an inverse FFT, all of length m.
    const SizeType n   = m_n;
    const SizeType m   = plan.fft_size;
    const float sign   = Inverse ? 1.0F : -1.0F;
    const float* chirp = plan.chirp.data();
    for (SizeType k = 0; k < n; ++k) {
        for (SizeType h = 0; h < kLanes; h += kWidth) {
            const SizeType at = (k * kLanes) + h;
            store(m_y_re, m_y_im, at,
                  cmul_scalar(load(m_z_re, m_z_im, at), chirp[2 * k],
                              sign * chirp[(2 * k) + 1]));
        }
    }
    std::fill(m_y_re + (n * kLanes), m_y_re + (m * kLanes), 0.0F);
    std::fill(m_y_im + (n * kLanes), m_y_im + (m * kLanes), 0.0F);
    auto [fr, fi] = run_stages<false>(m_y_re, m_y_im);
    const float* kernel =
        Inverse ? plan.kernel_inv.data() : plan.kernel_fwd.data();
    for (SizeType k = 0; k < m; ++k) {
        for (SizeType h = 0; h < kLanes; h += kWidth) {
            const SizeType at = (k * kLanes) + h;
            store(fr, fi, at,
                  cmul_scalar(load(fr, fi, at), kernel[2 * k],
                              kernel[(2 * k) + 1]));
        }
    }
    auto [rr, ri] = run_stages<true>(fr, fi);
    for (SizeType j = 0; j < n; ++j) {
        for (SizeType h = 0; h < kLanes; h += kWidth) {
            const SizeType at = (j * kLanes) + h;
            store(rr, ri, at,
                  cmul_scalar(load(rr, ri, at), chirp[2 * j],
                              sign * chirp[(2 * j) + 1]));
        }
    }
    m_res_re = rr;
    m_res_im = ri;
}

void EVTransform::inverse(const ComplexType* spectra,
                          SizeType nprofiles) noexcept {
    inverse_impl(spectra, 2 * nprofiles);
}

void EVTransform::inverse_impl(const ComplexType* spectra,
                               SizeType nrows) noexcept {
    const SizeType n   = m_n;
    const SizeType nf  = (n / 2) + 1;
    const auto* in     = reinterpret_cast<const float*>(spectra);
    const float* zeros = m_zeros.data();
    std::array<const float*, kLanes> rows_e{};
    std::array<const float*, kLanes> rows_v{};
    for (SizeType lane = 0; lane < kLanes; ++lane) {
        const SizeType row = 2 * lane;
        rows_e[lane]       = row < nrows ? in + (row * 2 * nf) : zeros;
        rows_v[lane] = row + 1 < nrows ? in + ((row + 1) * 2 * nf) : zeros;
    }
    // Spectra [bin][re, im][lane] into the y buffers.
    rows_to_lanes(rows_e, 2 * nf, m_y_re);
    rows_to_lanes(rows_v, 2 * nf, m_y_im);
    // Hermitian Z = (E + iV) / n over all n bins: for 0 < k < n/2,
    // Z[k] = (Er - Vi) + i(Ei + Vr) and Z[n-k] = (Er + Vi) + i(Vr - Ei); the
    // DC and (even n) Nyquist bins keep their real parts only.
    const Batch scale(1.0F / static_cast<float>(n));
    const float* te = m_y_re;
    const float* tv = m_y_im;
    for (SizeType h = 0; h < kLanes; h += kWidth) {
        const Batch er = Batch::load_aligned(te + h);
        const Batch vr = Batch::load_aligned(tv + h);
        store(m_z_re, m_z_im, h, CBatch{er * scale, vr * scale});
        if (n % 2 == 0) {
            const SizeType k = n / 2;
            const Batch en   = Batch::load_aligned(te + (2 * k * kLanes) + h);
            const Batch vn   = Batch::load_aligned(tv + (2 * k * kLanes) + h);
            store(m_z_re, m_z_im, (k * kLanes) + h,
                  CBatch{en * scale, vn * scale});
        }
    }
    for (SizeType k = 1; 2 * k < n; ++k) {
        for (SizeType h = 0; h < kLanes; h += kWidth) {
            const SizeType at = (2 * k * kLanes) + h;
            const Batch er    = Batch::load_aligned(te + at);
            const Batch ei    = Batch::load_aligned(te + at + kLanes);
            const Batch vr    = Batch::load_aligned(tv + at);
            const Batch vi    = Batch::load_aligned(tv + at + kLanes);
            store(m_z_re, m_z_im, (k * kLanes) + h,
                  CBatch{(er - vi) * scale, (ei + vr) * scale});
            store(m_z_re, m_z_im, ((n - k) * kLanes) + h,
                  CBatch{(er + vi) * scale, (vr - ei) * scale});
        }
    }
    transform<true>();
}

void EVTransform::inverse_rows(const ComplexType* spectra,
                               SizeType nrows,
                               float* samples) noexcept {
    inverse_impl(spectra, nrows);
    std::array<float*, kLanes> rows_e{};
    std::array<float*, kLanes> rows_v{};
    for (SizeType row = 0; row < nrows; ++row) {
        (row % 2 == 0 ? rows_e : rows_v)[row / 2] = samples + (row * m_n);
    }
    lanes_to_rows(m_res_re, m_n, rows_e, (nrows + 1) / 2);
    lanes_to_rows(m_res_im, m_n, rows_v, nrows / 2);
}

void EVTransform::forward_rows(const float* samples,
                               SizeType nrows,
                               ComplexType* spectra) noexcept {
    const SizeType n   = m_n;
    const SizeType nf  = (n / 2) + 1;
    const float* zeros = m_zeros.data();
    std::array<const float*, kLanes> rows_e{};
    std::array<const float*, kLanes> rows_v{};
    for (SizeType lane = 0; lane < kLanes; ++lane) {
        const SizeType row = 2 * lane;
        rows_e[lane]       = row < nrows ? samples + (row * n) : zeros;
        rows_v[lane] = row + 1 < nrows ? samples + ((row + 1) * n) : zeros;
    }
    // z = e + iv, then Z = FFT(z).
    rows_to_lanes(rows_e, n, m_z_re);
    rows_to_lanes(rows_v, n, m_z_im);
    transform<false>();
    // E[k] = (Z[k] + conj Z[n-k]) / 2, V[k] = (Z[k] - conj Z[n-k]) / 2i, as
    // [bin][re, im][lane] in a buffer pair the result is not in.
    const bool in_z = m_res_re == m_z_re;
    float* te       = in_z ? m_y_re : m_z_re;
    float* tv       = in_z ? m_y_im : m_z_im;
    const Batch half(0.5F);
    for (SizeType k = 0; k < nf; ++k) {
        const SizeType m = (n - k) % n;
        for (SizeType h = 0; h < kLanes; h += kWidth) {
            const CBatch zk   = load(m_res_re, m_res_im, (k * kLanes) + h);
            const CBatch zm   = load(m_res_re, m_res_im, (m * kLanes) + h);
            const SizeType at = (2 * k * kLanes) + h;
            ((zk.re + zm.re) * half).store_aligned(te + at);
            ((zk.im - zm.im) * half).store_aligned(te + at + kLanes);
            ((zk.im + zm.im) * half).store_aligned(tv + at);
            ((zm.re - zk.re) * half).store_aligned(tv + at + kLanes);
        }
    }
    auto* out = reinterpret_cast<float*>(spectra);
    std::array<float*, kLanes> out_e{};
    std::array<float*, kLanes> out_v{};
    for (SizeType row = 0; row < nrows; ++row) {
        (row % 2 == 0 ? out_e : out_v)[row / 2] = out + (row * 2 * nf);
    }
    lanes_to_rows(te, 2 * nf, out_e, (nrows + 1) / 2);
    lanes_to_rows(tv, 2 * nf, out_v, nrows / 2);
}

namespace {

/// Runs fn(transform, first_row, nrows) over groups of 2 kLanes rows;
/// contiguous blocks of groups per thread.
template <typename Fn>
void for_each_group(SizeType nrows, SizeType n, int nthreads, Fn&& fn) {
    nthreads                 = std::max(nthreads, 1);
    constexpr SizeType kRows = 2 * kLanes;
    const SizeType groups    = (nrows + kRows - 1) / kRows;
#pragma omp parallel num_threads(nthreads) if (groups > 1)
    {
        EVTransform transform(n);
#pragma omp for schedule(static)
        for (SizeType g = 0; g < groups; ++g) {
            const SizeType r0 = g * kRows;
            fn(transform, r0, std::min(kRows, nrows - r0));
        }
    }
}

} // namespace

void inverse_batch(const ComplexType* spectra,
                   float* samples,
                   SizeType nrows,
                   SizeType n,
                   int nthreads) {
    const SizeType nf = (n / 2) + 1;
    for_each_group(
        nrows, n, nthreads, [&](EVTransform& t, SizeType r0, SizeType count) {
            t.inverse_rows(spectra + (r0 * nf), count, samples + (r0 * n));
        });
}

void forward_batch(const float* samples,
                   ComplexType* spectra,
                   SizeType nrows,
                   SizeType n,
                   int nthreads) {
    const SizeType nf = (n / 2) + 1;
    for_each_group(
        nrows, n, nthreads, [&](EVTransform& t, SizeType r0, SizeType count) {
            t.forward_rows(samples + (r0 * n), count, spectra + (r0 * nf));
        });
}

} // namespace loki::lane_fft
