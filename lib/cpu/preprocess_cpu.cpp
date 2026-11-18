#include "loki/io/preprocess.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "loki/common/types.hpp"

#include "lib/cpu/preprocess_kernels.hpp"
#include "lib/detail/math.hpp"
#include "lib/detail/utils.hpp"
#include "lib/io/preprocess_engine.hpp"
#include "lib/utils/fft_impl.hpp"

namespace loki::io::detail {

namespace {

constexpr SizeType kParallelThreshold = 1U << 14U;

/// Exact median of v[0, n), same value as median_inplace but about 2x faster
/// on blocks: a sorted sample brackets the middle, a branchless pass copies
/// the bracket into @p tmp (capacity n) and only that is selected. Falls back
/// to median_inplace when the bracket misses. Reorders v only on fallback.
double median_bracketed(float* v, SizeType n, float* tmp) noexcept {
    constexpr SizeType kMinN = 64;
    if (n < kMinN) {
        return median_inplace(v, n);
    }
    const SizeType ns   = n < 1024 ? 15 : 63;
    const SizeType half = n < 1024 ? 3 : 6;
    std::array<float, 63> sample{};
    for (SizeType i = 0; i < ns; ++i) {
        sample[i] = v[((i * n) / ns) + (n / (2 * ns))];
    }
    std::sort(sample.begin(), sample.begin() + static_cast<std::ptrdiff_t>(ns));
    const SizeType c = ns / 2;
    const float lo   = sample[c - half];
    const float hi   = sample[c + half];
    SizeType below   = 0;
    SizeType m       = 0;
    for (SizeType i = 0; i < n; ++i) {
        const float a = v[i];
        below += a < lo ? 1 : 0;
        tmp[m] = a;
        m += (a >= lo && a <= hi) ? 1 : 0;
    }
    const SizeType h = n / 2;
    // Odd n needs rank h; even n needs ranks h - 1 and h.
    const SizeType first = n % 2 == 1 ? h : h - 1;
    if (first < below || h >= below + m) {
        return median_inplace(v, n);
    }
    const SizeType k = h - below;
    std::nth_element(tmp, tmp + k, tmp + m);
    const auto upper = static_cast<double>(tmp[k]);
    if (n % 2 == 1) {
        return upper;
    }
    const auto lower = static_cast<double>(*std::max_element(tmp, tmp + k));
    return 0.5 * (lower + upper);
}

void check_finite_cpu(std::span<const float> x, int nthreads) {
    const SizeType n     = x.size();
    const SizeType chunk = std::max<SizeType>(kParallelThreshold, 1);
    const SizeType nc    = (n + chunk - 1) / chunk;
    int bad              = 0;
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(n, x, chunk, nc) reduction(| : bad) if (nc > 1)
    for (SizeType c = 0; c < nc; ++c) {
        const SizeType i0 = c * chunk;
        bad |=
            utils::all_finite(x.subspan(i0, std::min(chunk, n - i0))) ? 0 : 1;
    }
    if (bad != 0) {
        throw std::invalid_argument("preprocess: input has non-finite values");
    }
}

PreprocessReport preprocess_zscore(std::span<float> x,
                                   double tsamp,
                                   std::span<float> ts_v,
                                   const PreprocessOptions& o,
                                   int nthreads) {
    PreprocessReport rep;
    rep.method = PreprocessMethod::kZScore;
    rep.nsamps = x.size();
    if (o.filter_window > 0.0) {
        const SizeType window =
            window_in_samples(o.filter_window, tsamp, x.size());
        rep.baseline_window = window;
        if (window > 1) {
            math::subtract_running_filter(
                x, window, math::FilterMethod::kMedian, o.fast_median,
                o.fast_median_min_points, nthreads);
        }
    }
    const auto res   = math::zscore(x, o.loc, o.scale, nthreads);
    rep.global_scale = res.scale.left;
    std::ranges::fill(ts_v, 1.0F);
    return rep;
}

/// 64-bit hash of the bits of @p x (FNV-1a per chunk, chunks combined in
/// order), independent of the thread count.
std::uint64_t data_seed(std::span<const float> x, int nthreads) {
    constexpr SizeType kChunk = SizeType{1} << 20U;
    const SizeType n          = x.size();
    const SizeType nchunks    = (n + kChunk - 1) / kChunk;
    std::vector<std::uint64_t> h(nchunks);
    const float* xp   = x.data();
    std::uint64_t* hp = h.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(n, nchunks, xp, hp, kChunk)
    for (SizeType c = 0; c < nchunks; ++c) {
        std::uint64_t v = 0xcbf29ce484222325ULL;
        for (SizeType i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) {
            v = (v ^ std::bit_cast<std::uint32_t>(xp[i])) * 0x100000001b3ULL;
        }
        hp[c] = v;
    }
    std::uint64_t seed = math::splitmix64(n);
    for (const auto v : h) {
        seed = math::splitmix64(seed ^ v);
    }
    return seed;
}

/// Zap step of the robust method (PreprocessOptions::zap_periodic, birdies)
/// on @p x with the sample mask @p mask; @p z is scratch.
SizeType zap_cpu(const BlockGrid& grid,
                 std::span<const float> mu,
                 std::span<const float> var,
                 double tsamp,
                 const PreprocessOptions& o,
                 std::span<float> x,
                 std::span<float> z,
                 std::span<std::uint8_t> mask,
                 std::vector<std::uint8_t>& flagged,
                 int nthreads) {
    const SizeType n = x.size();
    // The noise seed follows the input, so a rerun on the same data
    // reproduces the output bit for bit.
    const std::uint64_t seed = o.zap_seed.value_or(o.zap_mode == ZapMode::kNoise
                                                       ? data_seed(x, nthreads)
                                                       : std::uint64_t{0});
    flagged.assign((n / 2) + 1, 0U);
    const SizeType block = grid.block;
    const double clamp   = o.clip_sigma;
    whiten(x, grid, mu, var, z, nthreads);
    // Masked samples and outliers stay out of the spectrum; outliers are
    // marked 2 (still valid) so they keep their raw value below.
    const float clip = clamp > 0.0 ? static_cast<float>(clamp) : 0.0F;
    std::uint8_t* m  = mask.data();
    float* zp        = z.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(n, m, zp, clip) if (n >= kParallelThreshold)
    for (SizeType i = 0; i < n; ++i) {
        const bool outlier = clip > 0.0F && std::abs(zp[i]) > clip;
        m[i]               = (m[i] != 0U && outlier) ? std::uint8_t{2} : m[i];
        zp[i]              = m[i] == 1U ? zp[i] : 0.0F;
    }
    const SpectralZap zs{.use_threshold = o.zap_periodic,
                         .sigma         = o.zap_sigma,
                         .whiten_bins   = o.zap_whiten_bins,
                         .birdies       = o.birdies,
                         .mode          = o.zap_mode,
                         .seed          = seed,
                         .gap_passes    = kZapGapPasses};
    const SizeType n_zapped =
        zap_spectrum(z, tsamp, zs, mask, flagged, nthreads);
    float* xp         = x.data();
    const bool update = n_zapped > 0;
#pragma omp parallel num_threads(nthreads) default(none) shared(grid)          \
    firstprivate(mu, var, xp, zp, m, block,                                    \
                     update) if (n >= kParallelThreshold)
    {
        std::vector<float> mub(block);
        std::vector<float> varb(block);
#pragma omp for schedule(static)
        for (SizeType b = 0; b < grid.nblocks; ++b) {
            const SizeType i0  = grid.begin(b);
            const SizeType len = grid.end(b) - i0;
            if (update) {
                interpolate_block(grid, mu, b, mub.data());
                interpolate_block(grid, var, b, varb.data());
            }
            for (SizeType j = 0; j < len; ++j) {
                const SizeType i = i0 + j;
                if (update && m[i] == 1U) {
                    xp[i] = mub[j] + (zp[i] * std::sqrt(varb[j]));
                }
                m[i] = m[i] == 0U ? std::uint8_t{0} : std::uint8_t{1};
            }
        }
    }
    return n_zapped;
}

/// Output pass of the robust method: x -> ts_e (same buffer), ts_v.
void output_cpu(const BlockGrid& grid,
                std::span<const float> mu,
                std::span<const float> var,
                float cf,
                float clip2,
                bool mult,
                std::span<float> x,
                std::span<float> ts_v,
                std::span<const std::uint8_t> mask,
                std::span<SizeType> n_clip,
                std::span<SizeType> n_mask,
                int nthreads) {
    {
        float* xp             = x.data();
        float* vp             = ts_v.data();
        const std::uint8_t* m = mask.data();
        SizeType* ncp         = n_clip.data();
        SizeType* nmp         = n_mask.data();
        const SizeType n      = x.size();
        const SizeType block  = grid.block;
#pragma omp parallel num_threads(nthreads) default(none) shared(grid)          \
    firstprivate(mu, var, xp, vp, m, cf, clip2, ncp, nmp, mult,                \
                     block) if (n >= kParallelThreshold)
        {
            std::vector<float> mub(block);
            std::vector<float> varb(block);
#pragma omp for schedule(static)
            for (SizeType b = 0; b < grid.nblocks; ++b) {
                interpolate_block(grid, mu, b, mub.data());
                interpolate_block(grid, var, b, varb.data());
                const SizeType i0  = grid.begin(b);
                const SizeType len = grid.end(b) - i0;
                SizeType nc        = 0;
                SizeType nm        = 0;
                for (SizeType j = 0; j < len; ++j) {
                    const SizeType i  = i0 + j;
                    const float r     = xp[i] - mub[j];
                    const float g     = mult ? mub[j] : 1.0F;
                    const float wgt   = cf * g / varb[j];
                    const bool masked = m[i] == 0U;
                    const bool clipped =
                        !masked && clip2 > 0.0F && r * r > clip2 * varb[j];
                    nm += masked ? 1 : 0;
                    nc += clipped ? 1 : 0;
                    const bool keep = !masked && !clipped;
                    xp[i]           = keep ? r * wgt : 0.0F;
                    vp[i]           = keep ? cf * g * wgt : 0.0F;
                }
                ncp[b] = nc;
                nmp[b] = nm;
            }
        }
    }
}

/// Count, sum and sum of squares of the valid `z` of every block of @p g.
void flag_sums_cpu(std::span<const float> z,
                   std::span<const std::uint8_t> mask,
                   const BlockGrid& g,
                   float cl,
                   std::span<SizeType> cnt_out,
                   std::span<double> s1_out,
                   std::span<double> s2_out,
                   int nthreads) {
    const SizeType n       = z.size();
    const SizeType nb      = g.nblocks;
    const std::uint8_t* mk = mask.data();
    SizeType* pc           = cnt_out.data();
    double* p1             = s1_out.data();
    double* p2             = s2_out.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    shared(g)                                                                  \
    firstprivate(z, mk, pc, p1, p2, cl, nb) if (n >= kParallelThreshold)
    for (SizeType b = 0; b < nb; ++b) {
        const SizeType i0  = g.begin(b);
        const SizeType len = g.end(b) - i0;
        SizeType cnt       = 0;
        double s1          = 0.0;
        double s2          = 0.0;
        for (SizeType i = i0; i < i0 + len; ++i) {
            const float v = z[i];
            const bool ok = mk[i] != 0U && (cl == 0.0F || std::abs(v) <= cl);
            cnt += ok ? 1 : 0;
            s1 += ok ? static_cast<double>(v) : 0.0;
            s2 += ok ? static_cast<double>(v) * v : 0.0;
        }
        pc[b] = cnt;
        p1[b] = s1;
        p2[b] = s2;
    }
}

void mask_blocks_cpu(const BlockGrid& g,
                     std::span<const std::uint8_t> bad,
                     std::span<std::uint8_t> mask) {
    for (SizeType b = 0; b < g.nblocks; ++b) {
        if (bad[b] != 0U) {
            std::fill(mask.begin() + static_cast<std::ptrdiff_t>(g.begin(b)),
                      mask.begin() + static_cast<std::ptrdiff_t>(g.end(b)), 0U);
        }
    }
}

/// The per-sample passes of the robust method on the CPU, in place on ts_e
/// (x) and ts_v (z).
class CpuRobustStages final : public RobustStages {
public:
    CpuRobustStages(std::span<float> x, std::span<float> z, int nthreads)
        : m_x(x),
          m_z(z),
          m_mask(x.size(), 1U),
          m_nthreads(nthreads) {}

    void check_finite() override { check_finite_cpu(m_x, m_nthreads); }

    void reset_mask() override { std::ranges::fill(m_mask, 1U); }

    void block_medians(const BlockGrid& grid,
                       double min_good,
                       std::span<float> med,
                       std::span<std::uint8_t> usable,
                       std::span<float> good) override {
        detail::block_medians(m_x, m_mask, grid, min_good, med, usable, good,
                              m_nthreads);
    }

    std::vector<float> running_median(std::span<const float> v,
                                      std::span<const std::uint8_t> usable,
                                      SizeType k) override {
        return masked_running_median(v, usable, k, m_nthreads);
    }

    void block_variances(const BlockGrid& grid,
                         std::span<const float> mu,
                         double min_good,
                         std::span<float> var,
                         std::span<std::uint8_t> usable) override {
        detail::block_variances(m_x, m_mask, grid, mu, min_good, var, usable,
                                m_nthreads);
    }

    void whiten(const BlockGrid& grid,
                std::span<const float> mu,
                std::span<const float> var) override {
        detail::whiten(m_x, grid, mu, var, m_z, m_nthreads);
    }

    void flag_sums(const BlockGrid& g,
                   float clip,
                   std::span<SizeType> cnt,
                   std::span<double> s1,
                   std::span<double> s2) override {
        flag_sums_cpu(m_z, m_mask, g, clip, cnt, s1, s2, m_nthreads);
    }

    void mask_blocks(const BlockGrid& g,
                     std::span<const std::uint8_t> bad) override {
        mask_blocks_cpu(g, bad, m_mask);
    }

    SizeType zap(const BlockGrid& grid,
                 std::span<const float> mu,
                 std::span<const float> var,
                 double tsamp,
                 const PreprocessOptions& o) override {
        return zap_cpu(grid, mu, var, tsamp, o, m_x, m_z, m_mask, m_flagged,
                       m_nthreads);
    }

    std::vector<PreprocessReport::ZapBand>
    zap_check(double tsamp, const PreprocessOptions& o) override {
        return zap_bands(m_x, tsamp, m_flagged, o.zap_whiten_bins, m_nthreads);
    }

    void output(const BlockGrid& grid,
                std::span<const float> mu,
                std::span<const float> var,
                float cf,
                float clip2,
                bool mult,
                std::span<SizeType> n_clip,
                std::span<SizeType> n_mask) override {
        output_cpu(grid, mu, var, cf, clip2, mult, m_x, m_z, m_mask, n_clip,
                   n_mask, m_nthreads);
    }

    SizeType nsamps() const override { return m_x.size(); }

    SizeType longest_zero_run() override {
        return detail::longest_zero_run(m_z);
    }

private:
    std::span<float> m_x;
    std::span<float> m_z;
    std::vector<std::uint8_t> m_mask;
    std::vector<std::uint8_t> m_flagged; // zapped Fourier bins
    int m_nthreads;
};

PreprocessReport preprocess_robust(std::span<float> x,
                                   double tsamp,
                                   std::span<float> ts_v,
                                   const PreprocessOptions& o,
                                   int nthreads) {
    // ts_v doubles as the whitened-residual scratch until the output pass.
    CpuRobustStages stages(x, ts_v, nthreads);
    return run_robust(stages, tsamp, o);
}

} // namespace

void interpolate_block(const BlockGrid& grid,
                       std::span<const float> values,
                       SizeType b,
                       float* out) noexcept {
    const SizeType i0  = grid.begin(b);
    const SizeType len = grid.end(b) - i0;
    const double cb    = grid.center(b);
    const SizeType nb  = grid.nblocks;
    // Samples left of the centre interpolate with block b - 1, the rest with
    // block b + 1; outside the first/last centre the end value holds.
    const auto lerp = [&](SizeType j0, SizeType j1, SizeType lo, SizeType hi) {
        const double cl = grid.center(lo);
        const double dc = grid.center(hi) - cl;
        const auto vl   = static_cast<double>(values[lo]);
        const double dv = static_cast<double>(values[hi]) - vl;
        for (SizeType j = j0; j < j1; ++j) {
            const double t = (static_cast<double>(i0 + j) - cl) / dc;
            out[j]         = static_cast<float>(vl + (t * dv));
        }
    };
    // First sample with pos >= cb, and first with pos > cb.
    const auto j_mid = static_cast<SizeType>(std::ceil(cb)) - i0;
    const auto j_hi  = static_cast<SizeType>(std::floor(cb)) + 1 - i0;
    if (b > 0) {
        lerp(0, j_mid, b - 1, b);
    } else {
        std::fill(out, out + j_mid, values[b]);
    }
    std::fill(out + j_mid, out + j_hi, values[b]);
    if (b + 1 < nb) {
        lerp(j_hi, len, b, b + 1);
    } else {
        std::fill(out + j_hi, out + len, values[b]);
    }
}

void block_medians(std::span<const float> x,
                   std::span<const std::uint8_t> mask,
                   const BlockGrid& grid,
                   double min_good,
                   std::span<float> med,
                   std::span<std::uint8_t> usable,
                   std::span<float> good,
                   int nthreads) {
    const bool all_valid = mask.empty();
    const SizeType block = grid.block;
    nthreads             = std::max(nthreads, 1);
#pragma omp parallel num_threads(nthreads) default(none) shared(grid)          \
    firstprivate(x, mask, min_good, med, usable, good, all_valid,              \
                     block) if (grid.n >= kParallelThreshold)
    {
        std::vector<float> buf(block);
        std::vector<float> tmp(block);
#pragma omp for schedule(static)
        for (SizeType b = 0; b < grid.nblocks; ++b) {
            const SizeType i0  = grid.begin(b);
            const SizeType len = grid.end(b) - i0;
            SizeType cnt       = 0;
            for (SizeType j = 0; j < len; ++j) {
                buf[cnt] = x[i0 + j];
                cnt += (all_valid || mask[i0 + j] != 0U) ? 1 : 0;
            }
            good[b]       = static_cast<float>(cnt) / static_cast<float>(len);
            const bool ok = cnt >= min_valid(min_good, len, block) ||
                            (all_valid && cnt > 0);
            usable[b]     = ok ? 1U : 0U;
            med[b] = ok ? static_cast<float>(
                              median_bracketed(buf.data(), cnt, tmp.data()))
                        : 0.0F;
        }
    }
}

void block_variances(std::span<const float> x,
                     std::span<const std::uint8_t> mask,
                     const BlockGrid& grid,
                     std::span<const float> mu,
                     double min_good,
                     std::span<float> var,
                     std::span<std::uint8_t> usable,
                     int nthreads) {
    const SizeType block = grid.block;
    nthreads             = std::max(nthreads, 1);
#pragma omp parallel num_threads(nthreads) default(none) shared(grid)          \
    firstprivate(x, mask, mu, min_good, var, usable,                           \
                     block) if (grid.n >= kParallelThreshold)
    {
        std::vector<float> mub(block);
        std::vector<float> buf(block);
        std::vector<float> tmp(block);
#pragma omp for schedule(static)
        for (SizeType b = 0; b < grid.nblocks; ++b) {
            const SizeType i0  = grid.begin(b);
            const SizeType len = grid.end(b) - i0;
            interpolate_block(grid, mu, b, mub.data());
            SizeType cnt = 0;
            for (SizeType j = 0; j < len; ++j) {
                buf[cnt] = x[i0 + j] - mub[j];
                cnt += mask[i0 + j] != 0U ? 1 : 0;
            }
            if (cnt < min_valid(min_good, len, block)) {
                usable[b] = 0U;
                var[b]    = 0.0F;
                continue;
            }
            const double med = median_bracketed(buf.data(), cnt, tmp.data());
            double abs_sum   = 0.0;
            for (SizeType j = 0; j < cnt; ++j) {
                const double d = std::abs(static_cast<double>(buf[j]) - med);
                abs_sum += d;
                buf[j] = static_cast<float>(d);
            }
            // Small-sample consistency factor of the MAD (Croux & Rousseeuw).
            const auto nd        = static_cast<double>(cnt);
            const double small_n = nd > 9.0 ? nd / (nd - 0.8) : 1.2;
            double sigma = kMadScale * small_n *
                           median_bracketed(buf.data(), cnt, tmp.data());
            if (sigma <= 0.0) {
                // Quantised data: mean absolute deviation / sqrt(2 / pi).
                sigma = (abs_sum / static_cast<double>(cnt)) *
                        std::sqrt(std::numbers::pi / 2.0);
            }
            usable[b] = sigma > 0.0 ? 1U : 0U;
            var[b]    = static_cast<float>(sigma * sigma);
        }
    }
}

std::vector<float> masked_running_median(std::span<const float> v,
                                         std::span<const std::uint8_t> usable,
                                         SizeType k,
                                         int nthreads) {
    const auto n    = static_cast<std::ptrdiff_t>(v.size());
    const auto half = static_cast<std::ptrdiff_t>(k / 2);
    std::vector<float> out(v.size(), 0.0F);
    std::vector<std::uint8_t> have(v.size(), 0U);
    // Point reflection about the first / last usable entry keeps a linear
    // trend unbiased at the edges (plain truncation lags it by half a window).
    std::ptrdiff_t first = 0;
    while (first < n && usable[first] == 0U) {
        ++first;
    }
    std::ptrdiff_t last = n - 1;
    while (last >= 0 && usable[last] == 0U) {
        --last;
    }
    if (first == n) {
        throw std::invalid_argument("preprocess: no usable block statistics");
    }
    const float a0   = v[first];
    const float a1   = v[last];
    float* op        = out.data();
    std::uint8_t* hp = have.data();
    nthreads         = std::max(nthreads, 1);
#pragma omp parallel num_threads(nthreads) default(none)                       \
    firstprivate(v, usable, n, half, op, hp, k, first, last, a0,               \
                     a1) if (v.size() * k >= kParallelThreshold)
    {
        std::vector<float> buf(k);
#pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            SizeType cnt = 0;
            for (std::ptrdiff_t j = i - half; j <= i + half; ++j) {
                std::ptrdiff_t m = j;
                float val        = 0.0F;
                if (j < first) {
                    m   = (2 * first) - j;
                    val = m <= last ? (2.0F * a0) - v[m] : 0.0F;
                } else if (j > last) {
                    m   = (2 * last) - j;
                    val = m >= first ? (2.0F * a1) - v[m] : 0.0F;
                } else {
                    val = v[m];
                }
                const bool ok = m >= first && m <= last && usable[m] != 0U;
                buf[cnt]      = val;
                cnt += ok ? 1 : 0;
            }
            hp[i] = cnt > 0 ? 1U : 0U;
            op[i] = cnt > 0
                        ? static_cast<float>(median_inplace(buf.data(), cnt))
                        : 0.0F;
        }
    }
    fill_holes(out, have);
    return out;
}

void whiten(std::span<const float> x,
            const BlockGrid& grid,
            std::span<const float> mu,
            std::span<const float> var,
            std::span<float> z,
            int nthreads) {
    const SizeType block = grid.block;
    nthreads             = std::max(nthreads, 1);
#pragma omp parallel num_threads(nthreads) default(none) shared(grid)          \
    firstprivate(x, mu, var, z, block) if (grid.n >= kParallelThreshold)
    {
        std::vector<float> mub(block);
        std::vector<float> varb(block);
#pragma omp for schedule(static)
        for (SizeType b = 0; b < grid.nblocks; ++b) {
            interpolate_block(grid, mu, b, mub.data());
            interpolate_block(grid, var, b, varb.data());
            const SizeType i0  = grid.begin(b);
            const SizeType len = grid.end(b) - i0;
            for (SizeType j = 0; j < len; ++j) {
                z[i0 + j] = (x[i0 + j] - mub[j]) / std::sqrt(varb[j]);
            }
        }
    }
}

void flag_bad_blocks(std::span<const float> z,
                     std::span<const SizeType> scales,
                     double nsigma,
                     double min_good,
                     double clip,
                     std::span<std::uint8_t> mask,
                     int nthreads) {
    const SizeType n = z.size();
    nthreads         = std::max(nthreads, 1);
    std::ranges::fill(mask, 1U);
    const auto cl = clip > 0.0 ? static_cast<float>(clip) : 0.0F;
    for (const SizeType s : scales) {
        const BlockGrid g(n, s);
        const SizeType nb = g.nblocks;
        std::vector<SizeType> cnt(nb, 0);
        std::vector<double> s1(nb, 0.0);
        std::vector<double> s2(nb, 0.0);
        std::vector<std::uint8_t> bad(nb, 0U);
        flag_sums_cpu(z, mask, g, cl, cnt, s1, s2, nthreads);
        flag_decisions(g, cnt, s1, s2, nsigma, min_good, bad);
        mask_blocks_cpu(g, bad, mask);
    }
}

double exp_power_threshold(double sigma) {
    // P(p > t) = exp(-t) for Exp(1) power, so t = -ln Q(sigma).
    if (sigma < 30.0) {
        return -std::log(0.5 * std::erfc(sigma / std::numbers::sqrt2));
    }
    // Asymptotic Gaussian tail: ln Q ~ -s^2/2 - ln(s sqrt(2 pi)).
    return (0.5 * sigma * sigma) +
           std::log(sigma * std::sqrt(2.0 * std::numbers::pi));
}

/// make_keyed_pcg32 purpose of the noise drawn into zapped bins.
constexpr std::uint32_t kZapNoisePurpose = 1;

std::vector<std::uint8_t>
birdie_bins(std::span<const Birdie> birdies, double t_obs, SizeType nf) {
    std::vector<std::uint8_t> in(nf, 0U);
    for (const auto& bd : birdies) {
        const double lo = (bd.freq - (0.5 * bd.width)) * t_obs;
        const double hi = (bd.freq + (0.5 * bd.width)) * t_obs;
        auto k_lo       = static_cast<SizeType>(std::max(std::ceil(lo), 1.0));
        auto k_hi       = static_cast<SizeType>(std::max(std::floor(hi), 0.0));
        if (k_lo > k_hi) { // narrower than a bin: the nearest bin
            k_lo = k_hi = static_cast<SizeType>(
                std::max(std::llround(bd.freq * t_obs), 1LL));
        }
        for (SizeType k = k_lo; k <= std::min(k_hi, nf - 1); ++k) {
            in[k] = 1U;
        }
    }
    return in;
}

namespace {

/// Local mean power (median / ln 2) of every bin, from block medians of
/// the bins with @p valid != 0 (all when empty) and their running median.
std::vector<float> local_mean_power(std::span<const float> power,
                                    std::span<const std::uint8_t> valid,
                                    SizeType whiten_bins,
                                    int nthreads) {
    const SizeType nf     = power.size();
    const SizeType w      = std::min(whiten_bins, nf);
    const SizeType fblock = std::max<SizeType>(w / 16, 1);
    const BlockGrid fgrid(nf, fblock);
    std::vector<float> fmed(fgrid.nblocks);
    std::vector<std::uint8_t> fuse(fgrid.nblocks);
    std::vector<float> fgood(fgrid.nblocks);
    block_medians(power, valid, fgrid, 0.0, fmed, fuse, fgood, nthreads);
    const auto fmean = masked_running_median(
        fmed, fuse, blocks_per_window(w, fblock), nthreads);
    std::vector<float> mean(nf);
    const std::span<const float> fm = fmean;
    float* mp                       = mean.data();
#pragma omp parallel num_threads(nthreads) default(none) shared(fgrid)         \
    firstprivate(fm, mp, fblock) if (nf >= kParallelThreshold)
    {
        std::vector<float> mean_b(fblock);
#pragma omp for schedule(static)
        for (SizeType b = 0; b < fgrid.nblocks; ++b) {
            interpolate_block(fgrid, fm, b, mean_b.data());
            const SizeType k0 = fgrid.begin(b);
            for (SizeType k = k0; k < fgrid.end(b); ++k) {
                mp[k] = mean_b[k - k0] / std::numbers::ln2_v<float>;
            }
        }
    }
    return mean;
}

std::vector<float> power_spectrum(std::span<const ComplexType> spec,
                                  int nthreads) {
    const SizeType nf = spec.size();
    std::vector<float> power(nf);
    float* pw             = power.data();
    const ComplexType* sp = spec.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(nf, pw, sp) if (nf >= kParallelThreshold)
    for (SizeType k = 0; k < nf; ++k) {
        pw[k] = std::norm(sp[k]);
    }
    power[0] = 0.0F;
    return power;
}

/**
 * In-window power that zeroing the gaps leaves of a fill that is white
 * over a run of W flagged bins:
 * sum_{|q| < W} |M(q)|^2 (1 - |q| / W) / n^2, with M the spectrum of the
 * valid-sample indicator. (1 - f)^2 for gaps much shorter than 1 / W (f the
 * gap fraction), 1 - f for much longer ones. M is taken from the indicator
 * averaged over blocks, which is exact enough at the low q that matter.
 */
class GapRetention {
public:
    GapRetention(std::span<const std::uint8_t> samples, int nthreads) {
        const SizeType n     = samples.size();
        const SizeType block = std::max<SizeType>(n >> 20U, 1);
        const SizeType nd    = n / block; // the tail (< block) is left out
        std::vector<float> md(nd);
        for (SizeType j = 0; j < nd; ++j) {
            SizeType cnt = 0;
            for (SizeType i = j * block; i < (j + 1) * block; ++i) {
                cnt += samples[i] == 1U ? 1U : 0U;
            }
            md[j] = static_cast<float>(cnt) / static_cast<float>(block);
        }
        std::vector<ComplexType> spec((nd / 2) + 1);
        math::rfft_batch(md, spec, 1, nd, nthreads);
        const double norm =
            1.0 / (static_cast<double>(nd) * static_cast<double>(nd));
        m_s0.resize(spec.size() + 1, 0.0);
        m_s1.resize(spec.size() + 1, 0.0);
        for (SizeType q = 0; q < spec.size(); ++q) {
            const double p = std::norm(spec[q]) * norm;
            m_s0[q + 1]    = m_s0[q] + p;
            m_s1[q + 1]    = m_s1[q] + (static_cast<double>(q) * p);
        }
        const double nyq =
            (nd % 2 == 0) ? (std::norm(spec.back()) * norm) : 0.0;
        m_dc    = m_s0[1];
        m_total = (2.0 * m_s0.back()) - m_dc - nyq; // Parseval: valid fraction
    }

    /// Retention for a run of @p width bins.
    [[nodiscard]] double operator()(SizeType width) const {
        const SizeType qmax = m_s0.size() - 1;
        if (width >= qmax) {
            return m_total;
        }
        const auto w = static_cast<double>(width);
        // q = 0 once, 0 < |q| < W twice.
        const double s0 = m_s0[width] - m_dc;
        const double s1 = m_s1[width];
        return m_dc + (2.0 * (s0 - (s1 / w)));
    }

private:
    std::vector<double> m_s0; // prefix sums of |M(q)|^2 / n^2
    std::vector<double> m_s1; // prefix sums of q |M(q)|^2 / n^2
    double m_dc{0.0};
    double m_total{0.0};
};

/// irfft of @p spec into @p z, with the gaps (samples != 1) set to zero.
void to_series(std::span<const ComplexType> spec,
               std::span<float> z,
               std::span<const std::uint8_t> samples,
               int nthreads) {
    const SizeType n = z.size();
    math::irfft_batch(spec, z, 1, n, nthreads);
    if (samples.empty()) {
        return;
    }
    float* zp               = z.data();
    const std::uint8_t* smp = samples.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(n, zp, smp) if (n >= kParallelThreshold)
    for (SizeType i = 0; i < n; ++i) {
        zp[i] = smp[i] == 1U ? zp[i] : 0.0F;
    }
}

} // namespace

SizeType zap_spectrum(std::span<float> z,
                      double tsamp,
                      const SpectralZap& zs,
                      std::span<const std::uint8_t> samples,
                      std::span<std::uint8_t> zapped,
                      int nthreads) {
    const SizeType n  = z.size();
    const SizeType nf = (n / 2) + 1;
    nthreads          = std::max(nthreads, 1);
    if (!samples.empty() && samples.size() != n) {
        throw std::invalid_argument("zap_spectrum: samples needs n entries");
    }
    std::vector<ComplexType> spec(nf);
    math::rfft_batch(z, spec, 1, n, nthreads);
    auto power         = power_spectrum(spec, nthreads);
    const double t_obs = static_cast<double>(n) * tsamp;
    const auto window  = birdie_bins(zs.birdies, t_obs, nf);
    const bool hard    = zs.mode != ZapMode::kPhaseKept;
    // Local mean power. The hard modes leave the birdie windows out of the
    // medians, so other windows never raise the fill.
    std::vector<std::uint8_t> valid;
    if (hard) {
        valid.resize(nf);
        for (SizeType k = 0; k < nf; ++k) {
            valid[k] = window[k] == 0U ? 1U : 0U;
        }
    }
    auto mean = local_mean_power(power, valid, zs.whiten_bins, nthreads);

    // Flags: bins above the threshold, and the birdie windows (phase_kept:
    // only their bins above the local mean).
    std::vector<std::uint8_t> zap(nf, 0U);
    const double thr = exp_power_threshold(zs.sigma);
    for (SizeType k = 1; k < nf; ++k) {
        const float p = power[k];
        if (zs.use_threshold && mean[k] > 0.0F && p > thr * mean[k]) {
            zap[k] = 1U;
        }
        if (window[k] != 0U && (hard || p > mean[k])) {
            zap[k] = 1U;
        }
    }
    if (!zapped.empty()) {
        for (SizeType k = 0; k < nf; ++k) {
            zapped[k] = (zap[k] != 0U || window[k] != 0U) ? 1U : 0U;
        }
    }

    SizeType nzap = 0;
    if (!hard) {
        const float* mp        = mean.data();
        const std::uint8_t* zp = zap.data();
        ComplexType* sp        = spec.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(nf, mp, zp, sp)                                               \
    reduction(+ : nzap) if (nf >= kParallelThreshold)
        for (SizeType k = 1; k < nf; ++k) {
            if (zp[k] == 0U) {
                continue;
            }
            const double p = std::norm(sp[k]);
            if (p > mp[k]) {
                sp[k] *= static_cast<float>(std::sqrt(mp[k] / p));
                ++nzap;
            }
        }
        if (nzap > 0) {
            math::irfft_batch(spec, z, 1, n, nthreads);
        }
        return nzap;
    }

    nzap = static_cast<SizeType>(std::ranges::count(zap, std::uint8_t{1}));
    if (nzap == 0) {
        return 0;
    }
    // With gaps (masked samples, zero in z), removing the flagged bins and
    // re-zeroing the gaps puts a fraction of a line back into them; repeat
    // the removal so the residual falls geometrically.
    // Gaps leave about their fraction of a line behind; below
    // kZapMinGapFraction (a few clipped samples) one pass is enough.
    SizeType n_gap = 0;
    if (!samples.empty()) {
        n_gap = n - static_cast<SizeType>(
                        std::ranges::count(samples, std::uint8_t{1}));
    }
    const bool gaps       = static_cast<double>(n_gap) >=
                            kZapMinGapFraction * static_cast<double>(n);
    const SizeType passes = gaps ? std::max<SizeType>(zs.gap_passes, 1) : 1;
    for (SizeType pass = 0; pass + 1 < passes; ++pass) {
        for (SizeType k = 1; k < nf; ++k) {
            spec[k] = zap[k] != 0U ? ComplexType{0.0F, 0.0F} : spec[k];
        }
        to_series(spec, z, samples, nthreads);
        math::rfft_batch(z, spec, 1, n, nthreads);
    }
    if (passes > 1) {
        // The fill level of the cleaned spectrum: the gapped line's skirt
        // no longer inflates the neighbourhood of a bright window.
        power = power_spectrum(spec, nthreads);
        for (SizeType k = 0; k < nf; ++k) {
            valid[k] = zap[k] == 0U ? 1U : 0U;
        }
        mean = local_mean_power(power, valid, zs.whiten_bins, nthreads);
    }
    if (gaps) {
        // Zeroing the gaps after the fill keeps only part of its in-window
        // power (see gap_retention()); draw each flagged run that much
        // stronger.
        const GapRetention retention(samples, nthreads);
        for (SizeType k = 1; k < nf;) {
            if (zap[k] == 0U) {
                ++k;
                continue;
            }
            SizeType end = k;
            while (end < nf && zap[end] != 0U) {
                ++end;
            }
            const double r  = retention(end - k);
            const auto gain = r > 0.0 ? static_cast<float>(1.0 / r) : 1.0F;
            for (; k < end; ++k) {
                mean[k] *= gain;
            }
        }
    }
    {
        const float* mp          = mean.data();
        const std::uint8_t* zp   = zap.data();
        ComplexType* sp          = spec.data();
        const bool noise         = zs.mode == ZapMode::kNoise;
        const bool even          = n % 2 == 0;
        const std::uint64_t seed = zs.seed;
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(nf, mp, zp, sp, noise, even,                                  \
                     seed) if (nf >= kParallelThreshold)
        for (SizeType k = 1; k < nf; ++k) {
            if (zp[k] == 0U) {
                continue;
            }
            if (!noise || mp[k] <= 0.0F) {
                sp[k] = {0.0F, 0.0F};
                continue;
            }
            // One stream per bin: independent of threads and of order.
            auto rng =
                math::make_keyed_pcg32(seed, kZapNoisePurpose, 0U,
                                       static_cast<std::uint32_t>(k), 0U, 0U);
            std::array<float, 2> re_im{};
            const bool real_bin = even && k == nf - 1; // Nyquist is real
            const float sd      = std::sqrt(real_bin ? mp[k] : mp[k] / 2.0F);
            math::NormalSampler::generate(rng, re_im, 0.0F, sd);
            sp[k] = {re_im[0], real_bin ? 0.0F : re_im[1]};
        }
    }
    to_series(spec, z, n_gap > 0 ? samples : std::span<const std::uint8_t>{},
              nthreads);
    return nzap;
}

PreprocessReport preprocess_cpu(std::span<const float> raw,
                                double tsamp,
                                std::span<float> ts_e,
                                std::span<float> ts_v,
                                const PreprocessOptions& options,
                                int nthreads) {
    nthreads = std::max(nthreads, 1);
    // ts_e holds the working copy; raw may alias it.
    if (raw.data() != ts_e.data()) {
        std::ranges::copy(raw, ts_e.begin());
    }
    if (options.method == PreprocessMethod::kZScore) {
        return preprocess_zscore(ts_e, tsamp, ts_v, options, nthreads);
    }
    return preprocess_robust(ts_e, tsamp, ts_v, options, nthreads);
}

std::vector<PreprocessReport::ZapBand>
zap_bands(std::span<const float> ts,
          double tsamp,
          std::span<const std::uint8_t> flagged,
          SizeType whiten_bins,
          int nthreads) {
    const SizeType n  = ts.size();
    const SizeType nf = (n / 2) + 1;
    if (flagged.size() != nf) {
        throw std::invalid_argument("zap_bands: flagged needs n/2 + 1 bins");
    }
    nthreads = std::max(nthreads, 1);
    std::vector<ComplexType> spec(nf);
    math::rfft_batch(ts, spec, 1, n, nthreads);
    std::vector<float> power(nf);
    std::vector<std::uint8_t> valid(nf);
    for (SizeType k = 0; k < nf; ++k) {
        power[k] = std::norm(spec[k]);
        valid[k] = (k > 0 && flagged[k] == 0U) ? 1U : 0U;
    }
    spec = {};
    // Whiten by the running median of the unflagged bins (as the zap does).
    const SizeType w      = std::min(whiten_bins, nf);
    const SizeType fblock = std::max<SizeType>(w / 16, 1);
    const BlockGrid fgrid(nf, fblock);
    std::vector<float> fmed(fgrid.nblocks);
    std::vector<std::uint8_t> fuse(fgrid.nblocks);
    std::vector<float> fgood(fgrid.nblocks);
    block_medians(power, valid, fgrid, 0.0, fmed, fuse, fgood, nthreads);
    const auto fmean = masked_running_median(
        fmed, fuse, blocks_per_window(w, fblock), nthreads);

    // Bands with edges 0, 10, 20, 50, 100, 200, 500, ... Hz up to Nyquist.
    const double t_obs = static_cast<double>(n) * tsamp;
    const double f_nyq = 0.5 / tsamp;
    std::vector<double> edges{0.0};
    for (double decade = 10.0; edges.back() < f_nyq; decade *= 10.0) {
        for (const double m : {1.0, 2.0, 5.0}) {
            edges.push_back(std::min(m * decade, f_nyq));
            if (edges.back() >= f_nyq) {
                break;
            }
        }
    }
    const SizeType nband = edges.size() - 1;
    std::vector<double> sum_in(nband, 0.0);
    std::vector<double> sum_out(nband, 0.0);
    std::vector<SizeType> cnt_in(nband, 0);
    std::vector<SizeType> cnt_out(nband, 0);
    std::vector<float> mean_b(fblock);
    SizeType band = 0;
    for (SizeType b = 0; b < fgrid.nblocks; ++b) {
        interpolate_block(fgrid, fmean, b, mean_b.data());
        for (SizeType k = std::max<SizeType>(fgrid.begin(b), 1);
             k < fgrid.end(b); ++k) {
            const double f = static_cast<double>(k) / t_obs;
            while (band + 1 < nband && f >= edges[band + 1]) {
                ++band;
            }
            const double mean =
                mean_b[k - fgrid.begin(b)] / std::numbers::ln2_v<float>;
            if (mean <= 0.0) {
                continue;
            }
            const double p = power[k] / mean;
            if (flagged[k] != 0U) {
                sum_in[band] += p;
                ++cnt_in[band];
            } else {
                sum_out[band] += p;
                ++cnt_out[band];
            }
        }
    }
    std::vector<PreprocessReport::ZapBand> out;
    for (SizeType i = 0; i < nband; ++i) {
        if (cnt_in[i] == 0 || cnt_out[i] == 0 || !(sum_out[i] > 0.0)) {
            continue;
        }
        out.push_back(
            {.f_lo        = edges[i],
             .f_hi        = edges[i + 1],
             .n_zapped    = cnt_in[i],
             .power_ratio = (sum_in[i] / static_cast<double>(cnt_in[i])) /
                            (sum_out[i] / static_cast<double>(cnt_out[i]))});
    }
    return out;
}

} // namespace loki::io::detail
