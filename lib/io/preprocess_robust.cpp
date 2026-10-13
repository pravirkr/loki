#include "lib/io/preprocess_robust.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::io::detail {

namespace {

/// Block-level baseline and variance for the current mask.
struct BlockStats {
    std::vector<float> mu;
    std::vector<float> var;
    std::vector<float> good;
    double median_var{0.0};
};

BlockStats compute_stats(RobustStages& s,
                         const BlockGrid& grid,
                         SizeType k_mu,
                         SizeType k_var,
                         double min_good) {
    const SizeType nb = grid.nblocks;
    BlockStats st;
    st.good.resize(nb);
    std::vector<float> med(nb);
    std::vector<std::uint8_t> usable(nb);
    s.block_medians(grid, min_good, med, usable, st.good);
    st.mu = s.running_median(med, usable, k_mu);
    // Median then mean: the mean removes the staircase a running median
    // leaves on a trend with superposed wander.
    st.mu = reflected_running_mean(st.mu, blocks_per_window(k_mu, 2));

    std::vector<float> var(nb);
    s.block_variances(grid, st.mu, min_good, var, usable);
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

/// Multi-scale bad-block mask from the whitened residuals (see
/// flag_bad_blocks() in preprocess_kernels.hpp).
void flag_bad_blocks(RobustStages& s,
                     SizeType n,
                     std::span<const SizeType> scales,
                     double nsigma,
                     double min_good,
                     double clip) {
    s.reset_mask();
    const auto cl = clip > 0.0 ? static_cast<float>(clip) : 0.0F;
    for (const SizeType scale : scales) {
        const BlockGrid g(n, scale);
        const SizeType nb = g.nblocks;
        std::vector<SizeType> cnt(nb, 0);
        std::vector<double> s1(nb, 0.0);
        std::vector<double> s2(nb, 0.0);
        std::vector<std::uint8_t> bad(nb, 0U);
        s.flag_sums(g, cl, cnt, s1, s2);
        flag_decisions(g, cnt, s1, s2, nsigma, min_good, bad);
        s.mask_blocks(g, bad);
    }
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

SizeType min_valid(double min_good, SizeType len) {
    const auto need =
        static_cast<SizeType>(std::ceil(min_good * static_cast<double>(len)));
    return std::max<SizeType>({need, 3, 1});
}

double median_of(std::vector<double> v) {
    return v.empty() ? 0.0 : median_inplace(v.data(), v.size());
}

double robust_spread(const std::vector<double>& v, double med) {
    std::vector<double> dev(v.size());
    std::ranges::transform(v, dev.begin(),
                           [med](double a) { return std::abs(a - med); });
    return kMadScale * median_of(std::move(dev));
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

SizeType longest_zero_run(std::span<const float> v) noexcept {
    SizeType best = 0;
    SizeType run  = 0;
    for (const float a : v) {
        run  = a == 0.0F ? run + 1 : 0;
        best = std::max(best, run);
    }
    return best;
}

void flag_decisions(const BlockGrid& g,
                    std::span<const SizeType> cnt,
                    std::span<const double> s1,
                    std::span<const double> s2,
                    double nsigma,
                    double min_good,
                    std::span<std::uint8_t> bad) {
    const SizeType nb = g.nblocks;
    std::vector<double> stat_m(nb, 0.0);
    std::vector<double> stat_v(nb, 0.0);
    for (SizeType b = 0; b < nb; ++b) {
        const SizeType len = g.end(b) - g.begin(b);
        if (cnt[b] == 0 ||
            static_cast<double>(cnt[b]) < min_good * static_cast<double>(len)) {
            bad[b] = 1U;
            continue;
        }
        bad[b]       = 0U;
        const auto c = static_cast<double>(cnt[b]);
        stat_m[b]    = (s1[b] / c) * std::sqrt(c);
        // Wilson-Hilferty: cbrt(chi2_c / c) is close to normal, so the
        // variance statistic is not skewed by short blocks or red noise.
        const double wh = 2.0 / (9.0 * c);
        stat_v[b] =
            (std::cbrt(std::max(s2[b] / c, 0.0)) - (1.0 - wh)) / std::sqrt(wh);
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
    if (vm.empty()) {
        return;
    }
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

PreprocessReport
run_robust(RobustStages& s, double tsamp, const PreprocessOptions& o) {
    const SizeType n = s.ts_v().size();
    if (n < kMinRobustSamples) {
        throw std::invalid_argument(
            "preprocess: the robust method needs at least 16 samples");
    }
    s.check_finite();

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
    for (const double sc : o.block_scales) {
        const auto len = static_cast<SizeType>(std::llround(sc / tsamp));
        if (len >= kMinFlagBlock &&
            n / std::max<SizeType>(len, 1) >= kMinFlagBlocks) {
            scales.push_back(len);
        }
    }
    std::ranges::sort(scales);
    const auto dup = std::ranges::unique(scales);
    scales.erase(dup.begin(), dup.end());

    s.reset_mask();
    BlockStats st = compute_stats(s, grid, k_mu, k_var, o.min_good_fraction);
    if (!scales.empty()) {
        for (SizeType it = 0; it < o.n_iter; ++it) {
            s.whiten(grid, st.mu, st.var);
            flag_bad_blocks(s, n, scales, o.block_sigma, o.min_good_fraction,
                            o.clip_sigma);
            st = compute_stats(s, grid, k_mu, k_var, o.min_good_fraction);
        }
    }

    PreprocessReport rep;
    rep.method          = PreprocessMethod::kRobust;
    rep.nsamps          = n;
    rep.baseline_window = w_mu;
    rep.variance_window = w_var;
    rep.block_size      = block;

    if (o.zap_periodic || !o.birdies.empty()) {
        rep.n_zapped = s.zap(grid, st.mu, st.var, tsamp, o);
        if (rep.n_zapped > 0) {
            st = compute_stats(s, grid, k_mu, k_var, o.min_good_fraction);
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

    // Output pass: x -> ts_e, z -> ts_v.
    std::vector<SizeType> n_clip(grid.nblocks, 0);
    std::vector<SizeType> n_mask(grid.nblocks, 0);
    const auto cf    = static_cast<float>(c);
    const auto clip2 = static_cast<float>(o.clip_sigma * o.clip_sigma);
    s.output(grid, st.mu, st.var, cf, clip2, mult, n_clip, n_mask);
    for (SizeType b = 0; b < grid.nblocks; ++b) {
        rep.n_clipped += n_clip[b];
        rep.n_masked += n_mask[b];
    }
    if (rep.n_masked + rep.n_clipped == n) {
        throw std::invalid_argument("preprocess: every sample is masked");
    }
    rep.longest_masked_run = longest_zero_run(s.ts_v());
    rep.global_scale       = std::sqrt(st.median_var);
    rep.norm               = c;
    rep.block_mu           = std::move(st.mu);
    rep.block_sigma.resize(grid.nblocks);
    std::ranges::transform(st.var, rep.block_sigma.begin(),
                           [](float v) { return std::sqrt(v); });
    rep.block_good_fraction = std::move(st.good);
    return rep;
}

} // namespace loki::io::detail
