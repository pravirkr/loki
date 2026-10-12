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

/// Median of v[0, n) (mean of the two middle values for even n). Reorders v.
template <typename T> double median_inplace(T* v, SizeType n) noexcept {
    const SizeType h = n / 2;
    std::nth_element(v, v + h, v + n);
    const auto hi = static_cast<double>(v[h]);
    if (n % 2 == 1) {
        return hi;
    }
    const auto lo = static_cast<double>(*std::max_element(v, v + h));
    return 0.5 * (lo + hi);
}

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

/// Median of a small vector (copied).
double median_of(std::vector<double> v) {
    return v.empty() ? 0.0 : median_inplace(v.data(), v.size());
}

/// Gaussian-consistent robust spread: 1.4826 * MAD about @p med.
double robust_spread(const std::vector<double>& v, double med) {
    std::vector<double> dev(v.size());
    std::ranges::transform(v, dev.begin(),
                           [med](double a) { return std::abs(a - med); });
    return kMadScale * median_of(std::move(dev));
}

SizeType min_valid(double min_good, SizeType len) {
    const auto need =
        static_cast<SizeType>(std::ceil(min_good * static_cast<double>(len)));
    return std::max<SizeType>({need, 3, 1});
}

/// Block-level baseline and variance for the current mask.
struct BlockStats {
    std::vector<float> mu;
    std::vector<float> var;
    std::vector<float> good;
    double median_var{0.0};
};

BlockStats compute_stats(std::span<const float> x,
                         std::span<const std::uint8_t> mask,
                         const BlockGrid& grid,
                         SizeType k_mu,
                         SizeType k_var,
                         double min_good,
                         int nthreads) {
    const SizeType nb = grid.nblocks;
    BlockStats st;
    st.good.resize(nb);
    std::vector<float> med(nb);
    std::vector<std::uint8_t> usable(nb);
    block_medians(x, mask, grid, min_good, med, usable, st.good, nthreads);
    st.mu = masked_running_median(med, usable, k_mu, nthreads);
    // Median then mean: the mean removes the staircase a running median
    // leaves on a trend with superposed wander.
    st.mu = reflected_running_mean(st.mu, blocks_per_window(k_mu, 2));

    std::vector<float> var(nb);
    block_variances(x, mask, grid, st.mu, min_good, var, usable, nthreads);
    std::vector<double> used;
    used.reserve(nb);
    for (SizeType b = 0; b < nb; ++b) {
        if (usable[b] != 0U) {
            used.push_back(var[b]);
        }
    }
    if (used.empty()) {
        throw std::invalid_argument(
            "preprocess: no block has a usable variance (constant or fully "
            "masked series)");
    }
    st.median_var        = median_of(std::move(used));
    st.var               = masked_running_mean(var, usable, k_var);
    const auto floor_var = static_cast<float>(kRelVarFloor * st.median_var);
    for (auto& v : st.var) {
        v = std::max(v, floor_var);
    }
    return st;
}

void check_finite(std::span<const float> x, int nthreads) {
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

PreprocessReport preprocess_robust(std::span<float> x,
                                   double tsamp,
                                   std::span<float> ts_v,
                                   const PreprocessOptions& o,
                                   int nthreads) {
    const SizeType n = x.size();
    if (n < kMinRobustSamples) {
        throw std::invalid_argument(
            "preprocess: the robust method needs at least 16 samples");
    }
    check_finite(x, nthreads);

    // Geometry. A zero baseline window means one baseline for the series.
    const SizeType w_mu  = o.filter_window > 0.0
                               ? window_in_samples(o.filter_window, tsamp, n)
                               : n;
    const SizeType w_var = o.variance_window > 0.0
                               ? window_in_samples(o.variance_window, tsamp, n)
                               : w_mu;
    const SizeType block =
        std::clamp(w_mu / o.window_blocks, std::min(kMinStatBlock, n), n);
    const BlockGrid grid(n, block);
    const SizeType k_mu  = blocks_per_window(w_mu, block);
    const SizeType k_var = blocks_per_window(w_var, block);

    std::vector<SizeType> scales;
    for (const double s : o.block_scales) {
        const auto len = static_cast<SizeType>(std::llround(s / tsamp));
        if (len >= kMinFlagBlock &&
            n / std::max<SizeType>(len, 1) >= kMinFlagBlocks) {
            scales.push_back(len);
        }
    }
    std::ranges::sort(scales);
    const auto dup = std::ranges::unique(scales);
    scales.erase(dup.begin(), dup.end());

    // ts_v doubles as the whitened-residual scratch until the output pass.
    std::span<float> z = ts_v;
    std::vector<std::uint8_t> mask(n, 1U);
    const double clamp = o.clip_sigma;

    BlockStats st = compute_stats(x, mask, grid, k_mu, k_var,
                                  o.min_good_fraction, nthreads);
    if (!scales.empty()) {
        for (SizeType it = 0; it < o.n_iter; ++it) {
            whiten(x, grid, st.mu, st.var, z, nthreads);
            flag_bad_blocks(z, scales, o.block_sigma, o.min_good_fraction,
                            clamp, mask, nthreads);
            st = compute_stats(x, mask, grid, k_mu, k_var, o.min_good_fraction,
                               nthreads);
        }
    }

    PreprocessReport rep;
    rep.method          = PreprocessMethod::kRobust;
    rep.nsamps          = n;
    rep.baseline_window = w_mu;
    rep.variance_window = w_var;
    rep.block_size      = block;

    if (o.zap_periodic || !o.birdies.empty()) {
        whiten(x, grid, st.mu, st.var, z, nthreads);
        // Masked samples and outliers stay out of the spectrum; outliers are
        // marked 2 (still valid) so they keep their raw value below.
        const float clip = clamp > 0.0 ? static_cast<float>(clamp) : 0.0F;
        std::uint8_t* m  = mask.data();
        float* zp        = z.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    firstprivate(n, m, zp, clip) if (n >= kParallelThreshold)
        for (SizeType i = 0; i < n; ++i) {
            const bool outlier = clip > 0.0F && std::abs(zp[i]) > clip;
            m[i]  = (m[i] != 0U && outlier) ? std::uint8_t{2} : m[i];
            zp[i] = m[i] == 1U ? zp[i] : 0.0F;
        }
        rep.n_zapped = zap_spectrum(z, tsamp, o.zap_periodic, o.zap_sigma,
                                    o.zap_whiten_bins, o.birdies, nthreads);
        const std::span<const float> mu  = st.mu;
        const std::span<const float> var = st.var;
        float* xp                        = x.data();
        const bool update                = rep.n_zapped > 0;
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
        if (update) {
            st = compute_stats(x, mask, grid, k_mu, k_var, o.min_good_fraction,
                               nthreads);
        }
    }

    // Normalisation and gain.
    const bool mult = o.gain_model == GainModel::kMultiplicative;
    if (mult && *std::ranges::min_element(st.mu) <= 0.0F) {
        throw std::invalid_argument(
            "preprocess: the multiplicative gain model needs a positive "
            "baseline (is the data already mean-subtracted?)");
    }
    std::vector<double> info;
    info.reserve(grid.nblocks);
    for (SizeType b = 0; b < grid.nblocks; ++b) {
        if (st.good[b] > 0.0F) {
            const double g = mult ? st.mu[b] : 1.0;
            info.push_back(g * g / st.var[b]);
        }
    }
    if (info.empty()) {
        throw std::invalid_argument("preprocess: every sample is masked");
    }
    const double c = 1.0 / std::sqrt(median_of(std::move(info)));

    // Output pass: x -> ts_e (same buffer), ts_v.
    std::vector<SizeType> n_clip(grid.nblocks, 0);
    std::vector<SizeType> n_mask(grid.nblocks, 0);
    {
        const std::span<const float> mu  = st.mu;
        const std::span<const float> var = st.var;
        float* xp                        = x.data();
        float* vp                        = ts_v.data();
        const std::uint8_t* m            = mask.data();
        const auto cf                    = static_cast<float>(c);
        const auto clip2 = static_cast<float>(o.clip_sigma * o.clip_sigma);
        SizeType* ncp    = n_clip.data();
        SizeType* nmp    = n_mask.data();
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
    for (SizeType b = 0; b < grid.nblocks; ++b) {
        rep.n_clipped += n_clip[b];
        rep.n_masked += n_mask[b];
    }
    if (rep.n_masked + rep.n_clipped == n) {
        throw std::invalid_argument("preprocess: every sample is masked");
    }
    rep.longest_masked_run = longest_zero_run(ts_v);
    rep.global_scale       = std::sqrt(st.median_var);
    rep.norm               = c;
    rep.block_mu           = std::move(st.mu);
    rep.block_sigma.resize(grid.nblocks);
    std::ranges::transform(st.var, rep.block_sigma.begin(),
                           [](float v) { return std::sqrt(v); });
    rep.block_good_fraction = std::move(st.good);
    return rep;
}

} // namespace

BlockGrid::BlockGrid(SizeType n_samples, SizeType block_size)
    : n(n_samples),
      block(std::max<SizeType>(block_size, 1)),
      nblocks((n_samples + block - 1) / block) {}

SizeType window_in_samples(double window_sec, double tsamp, SizeType n) {
    const double bins = window_sec / tsamp;
    if (bins >= static_cast<double>(n)) {
        return n;
    }
    return bins >= 1.0 ? static_cast<SizeType>(std::llround(bins)) : 1;
}

SizeType blocks_per_window(SizeType window, SizeType block) {
    auto k = static_cast<SizeType>(
        std::llround(static_cast<double>(window) / static_cast<double>(block)));
    k = std::max<SizeType>(k, 1);
    return k % 2 == 0 ? k + 1 : k;
}

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

void fill_holes(std::span<float> v, std::span<const std::uint8_t> have) {
    const SizeType n = v.size();
    SizeType prev    = n; // last index with a value, n = none yet
    for (SizeType i = 0; i < n; ++i) {
        if (have[i] == 0U) {
            continue;
        }
        if (prev == n) {
            std::fill(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i),
                      v[i]);
        } else if (i > prev + 1) {
            const double a    = v[prev];
            const double d    = static_cast<double>(v[i]) - a;
            const auto span_d = static_cast<double>(i - prev);
            for (SizeType j = prev + 1; j < i; ++j) {
                v[j] = static_cast<float>(
                    a + (d * static_cast<double>(j - prev) / span_d));
            }
        }
        prev = i;
    }
    if (prev == n) {
        throw std::invalid_argument("preprocess: no usable block statistics");
    }
    std::fill(v.begin() + static_cast<std::ptrdiff_t>(prev) + 1, v.end(),
              v[prev]);
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

std::vector<float> masked_running_mean(std::span<const float> v,
                                       std::span<const std::uint8_t> usable,
                                       SizeType k) {
    const SizeType n    = v.size();
    const SizeType half = k / 2;
    std::vector<double> sum(n + 1, 0.0);
    std::vector<SizeType> cnt(n + 1, 0);
    for (SizeType i = 0; i < n; ++i) {
        const bool ok = usable[i] != 0U;
        sum[i + 1]    = sum[i] + (ok ? static_cast<double>(v[i]) : 0.0);
        cnt[i + 1]    = cnt[i] + (ok ? 1 : 0);
    }
    std::vector<float> out(n, 0.0F);
    std::vector<std::uint8_t> have(n, 0U);
    for (SizeType i = 0; i < n; ++i) {
        const SizeType lo = i >= half ? i - half : 0;
        const SizeType hi = std::min(n, i + half + 1);
        const SizeType c  = cnt[hi] - cnt[lo];
        if (c > 0) {
            have[i] = 1U;
            out[i]  = static_cast<float>((sum[hi] - sum[lo]) /
                                         static_cast<double>(c));
        }
    }
    fill_holes(out, have);
    return out;
}

std::vector<float> reflected_running_mean(std::span<const float> v,
                                          SizeType k) {
    const auto n    = static_cast<std::ptrdiff_t>(v.size());
    const auto half = static_cast<std::ptrdiff_t>(k / 2);
    if (half == 0 || n < 2) {
        return {v.begin(), v.end()};
    }
    // Extended series with point reflection about both ends.
    const auto at = [&](std::ptrdiff_t j) -> double {
        if (j < 0) {
            const std::ptrdiff_t m = std::min(-j, n - 1);
            return (2.0 * v[0]) - v[m];
        }
        if (j >= n) {
            const std::ptrdiff_t m =
                std::max((2 * (n - 1)) - j, std::ptrdiff_t{0});
            return (2.0 * v[n - 1]) - v[m];
        }
        return v[j];
    };
    std::vector<double> prefix(static_cast<SizeType>(n + (2 * half)) + 1, 0.0);
    for (std::ptrdiff_t j = -half; j < n + half; ++j) {
        prefix[j + half + 1] = prefix[j + half] + at(j);
    }
    std::vector<float> out(v.size());
    const auto kd = static_cast<double>((2 * half) + 1);
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        out[i] =
            static_cast<float>((prefix[i + (2 * half) + 1] - prefix[i]) / kd);
    }
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
        std::vector<double> stat_m(nb, 0.0);
        std::vector<double> stat_v(nb, 0.0);
        std::vector<std::uint8_t> bad(nb, 0U);
        double* pm             = stat_m.data();
        double* pv             = stat_v.data();
        std::uint8_t* pb       = bad.data();
        const std::uint8_t* mk = mask.data();
#pragma omp parallel for num_threads(nthreads) schedule(static) default(none)  \
    shared(g) firstprivate(z, mk, pm, pv, pb, cl, min_good,                    \
                               nb) if (n >= kParallelThreshold)
        for (SizeType b = 0; b < nb; ++b) {
            const SizeType i0  = g.begin(b);
            const SizeType len = g.end(b) - i0;
            SizeType cnt       = 0;
            double s1          = 0.0;
            double s2          = 0.0;
            for (SizeType i = i0; i < i0 + len; ++i) {
                const float v = z[i];
                const bool ok =
                    mk[i] != 0U && (cl == 0.0F || std::abs(v) <= cl);
                cnt += ok ? 1 : 0;
                s1 += ok ? static_cast<double>(v) : 0.0;
                s2 += ok ? static_cast<double>(v) * v : 0.0;
            }
            if (cnt == 0 || static_cast<double>(cnt) <
                                min_good * static_cast<double>(len)) {
                pb[b] = 1U;
                continue;
            }
            const auto c = static_cast<double>(cnt);
            pm[b]        = (s1 / c) * std::sqrt(c);
            // Wilson-Hilferty: cbrt(chi2_c / c) is close to normal, so the
            // variance statistic is not skewed by short blocks or red noise.
            const double wh = 2.0 / (9.0 * c);
            pv[b] =
                (std::cbrt(std::max(s2 / c, 0.0)) - (1.0 - wh)) / std::sqrt(wh);
        }
        std::vector<double> vm;
        std::vector<double> vv;
        vm.reserve(nb);
        vv.reserve(nb);
        for (SizeType b = 0; b < nb; ++b) {
            if (bad[b] == 0U) {
                vm.push_back(stat_m[b]);
                vv.push_back(stat_v[b]);
            }
        }
        if (!vm.empty()) {
            const double cm = median_of(vm);
            const double cv = median_of(vv);
            const double sm = std::max(robust_spread(vm, cm), 1.0);
            const double sv = std::max(robust_spread(vv, cv), 1.0);
            for (SizeType b = 0; b < nb; ++b) {
                if (bad[b] == 0U && (std::abs(stat_m[b] - cm) > nsigma * sm ||
                                     std::abs(stat_v[b] - cv) > nsigma * sv)) {
                    bad[b] = 1U;
                }
            }
        }
        for (SizeType b = 0; b < nb; ++b) {
            if (bad[b] != 0U) {
                std::fill(
                    mask.begin() + static_cast<std::ptrdiff_t>(g.begin(b)),
                    mask.begin() + static_cast<std::ptrdiff_t>(g.end(b)), 0U);
            }
        }
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

SizeType longest_zero_run(std::span<const float> v) noexcept {
    SizeType best = 0;
    SizeType run  = 0;
    for (const float a : v) {
        run  = a == 0.0F ? run + 1 : 0;
        best = std::max(best, run);
    }
    return best;
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
