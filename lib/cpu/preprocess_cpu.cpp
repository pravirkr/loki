#include "loki/io/preprocess.hpp"

#include <algorithm>
#include <array>
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
                 int nthreads) {
    const SizeType n     = x.size();
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
    const SizeType n_zapped =
        zap_spectrum(z, tsamp, o.zap_periodic, o.zap_sigma, o.zap_whiten_bins,
                     o.birdies, nthreads);
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
        return zap_cpu(grid, mu, var, tsamp, o, m_x, m_z, m_mask, m_nthreads);
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

    std::span<const float> ts_v() override { return m_z; }

private:
    std::span<float> m_x;
    std::span<float> m_z;
    std::vector<std::uint8_t> m_mask;
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
            good[b] = static_cast<float>(cnt) / static_cast<float>(len);
            const bool ok =
                cnt >= min_valid(min_good, len) || (all_valid && cnt > 0);
            usable[b] = ok ? 1U : 0U;
            med[b]    = ok ? static_cast<float>(
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
            if (cnt < min_valid(min_good, len)) {
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

SizeType zap_spectrum(std::span<float> z,
                      double tsamp,
                      bool use_threshold,
                      double zap_sigma,
                      SizeType whiten_bins,
                      std::span<const Birdie> birdies,
                      int nthreads) {
    const SizeType n  = z.size();
    const SizeType nf = (n / 2) + 1;
    nthreads          = std::max(nthreads, 1);
    std::vector<ComplexType> spec(nf);
    math::rfft_batch(z, spec, 1, n, nthreads);

    std::vector<float> power(nf);
    {
        float* pw             = power.data();
        const ComplexType* sp = spec.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(nf, pw, sp) if (nf >= kParallelThreshold)
        for (SizeType k = 0; k < nf; ++k) {
            pw[k] = std::norm(sp[k]);
        }
    }
    power[0] = 0.0F;
    // Local median power: block medians, then a running median of them.
    const SizeType w      = std::min(whiten_bins, nf);
    const SizeType fblock = std::max<SizeType>(w / 16, 1);
    const BlockGrid fgrid(nf, fblock);
    std::vector<float> fmed(fgrid.nblocks);
    std::vector<std::uint8_t> fuse(fgrid.nblocks);
    std::vector<float> fgood(fgrid.nblocks);
    block_medians(power, {}, fgrid, 0.0, fmed, fuse, fgood, nthreads);
    const auto fmean = masked_running_median(
        fmed, fuse, blocks_per_window(w, fblock), nthreads);

    // Flagged bins get the local mean power (median / ln 2).
    std::vector<std::uint8_t> zap(nf, 0U);
    const double thr   = exp_power_threshold(zap_sigma);
    const double t_obs = static_cast<double>(n) * tsamp;
    {
        const std::span<const float> fm = fmean;
        float* pw                       = power.data();
        std::uint8_t* zp                = zap.data();
#pragma omp parallel num_threads(nthreads) default(none) shared(fgrid)         \
    firstprivate(fm, pw, zp, fblock, thr,                                      \
                     use_threshold) if (nf >= kParallelThreshold)
        {
            std::vector<float> mean_b(fblock);
#pragma omp for schedule(static)
            for (SizeType b = 0; b < fgrid.nblocks; ++b) {
                interpolate_block(fgrid, fm, b, mean_b.data());
                const SizeType k0 = fgrid.begin(b);
                for (SizeType k = k0; k < fgrid.end(b); ++k) {
                    const float mean =
                        mean_b[k - k0] / std::numbers::ln2_v<float>;
                    if (use_threshold && k > 0 && mean > 0.0F &&
                        pw[k] > thr * mean) {
                        zp[k] = 1U;
                    }
                    pw[k] = mean; // reuse: local mean power
                }
            }
        }
    }
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
            const double p = std::norm(spec[k]);
            zap[k]         = p > power[k] ? 1U : zap[k];
        }
    }
    SizeType nzap = 0;
    {
        const float* pw        = power.data();
        const std::uint8_t* zp = zap.data();
        ComplexType* sp        = spec.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(nf, pw, zp, sp)                                               \
    reduction(+ : nzap) if (nf >= kParallelThreshold)
        for (SizeType k = 1; k < nf; ++k) {
            if (zp[k] == 0U) {
                continue;
            }
            const double p = std::norm(sp[k]);
            if (p > pw[k]) {
                sp[k] *= static_cast<float>(std::sqrt(pw[k] / p));
                ++nzap;
            }
        }
    }
    if (nzap > 0) {
        math::irfft_batch(spec, z, 1, n, nthreads);
    }
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

} // namespace loki::io::detail
