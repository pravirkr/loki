#include "lib/cpu/preprocess_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

#include "lib/detail/math.hpp"
#include "lib/utils/fft_impl.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using loki::SizeType;
namespace pd = loki::io::detail;

namespace {

std::vector<float> normal(SizeType n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(n);
    for (auto& v : x) {
        v = nd(rng);
    }
    return x;
}

double brute_median(std::vector<float> v) {
    std::ranges::sort(v);
    const auto n = v.size();
    return n % 2 == 1 ? v[n / 2] : 0.5 * (v[(n / 2) - 1] + v[n / 2]);
}

} // namespace

TEST_CASE("preprocess kernels: block grid and geometry", "[preprocess]") {
    const pd::BlockGrid g(100, 32);
    REQUIRE(g.nblocks == 4);
    REQUIRE(g.end(3) == 100);
    REQUIRE_THAT(g.center(0), WithinAbs(15.5, 0.0));
    REQUIRE_THAT(g.center(3), WithinAbs(97.5, 0.0));
    REQUIRE(pd::blocks_per_window(1000, 100) == 11);
    REQUIRE(pd::blocks_per_window(1000, 90) == 11);
    REQUIRE(pd::blocks_per_window(10, 100) == 1);
    REQUIRE(pd::window_in_samples(1.0, 1e-3, 1U << 20U) == 1000);
    REQUIRE(pd::window_in_samples(1e9, 1e-3, 500) == 500);
    REQUIRE(pd::window_in_samples(1e-9, 1e-3, 500) == 1);
}

TEST_CASE("preprocess kernels: interpolation is linear between centres",
          "[preprocess]") {
    const pd::BlockGrid g(40, 10);
    const std::vector<float> vals{0.0F, 10.0F, 20.0F, 40.0F};
    std::vector<float> out(10);
    pd::interpolate_block(g, vals, 0, out.data());
    REQUIRE(out[0] == 0.0F); // before the first centre: clamped
    REQUIRE_THAT(out[9], WithinAbs(4.5, 1e-5));
    pd::interpolate_block(g, vals, 2, out.data());
    REQUIRE_THAT(out[0], WithinAbs(15.5, 1e-5));
    pd::interpolate_block(g, vals, 3, out.data());
    REQUIRE(out[9] == 40.0F); // after the last centre: clamped
}

TEST_CASE("preprocess kernels: masked block medians match brute force",
          "[preprocess]") {
    const SizeType n = 1000;
    const auto x     = normal(n, 1);
    std::vector<std::uint8_t> mask(n, 1U);
    for (SizeType i = 0; i < n; i += 3) {
        mask[i] = 0U;
    }
    for (SizeType i = 100; i < 164; ++i) {
        mask[i] = 0U; // block 3 (96..127) is mostly masked
    }
    const pd::BlockGrid g(n, 32);
    std::vector<float> med(g.nblocks);
    std::vector<std::uint8_t> usable(g.nblocks);
    std::vector<float> good(g.nblocks);
    pd::block_medians(x, mask, g, 0.3, med, usable, good, 4);
    for (SizeType b = 0; b < g.nblocks; ++b) {
        std::vector<float> valid;
        for (SizeType i = g.begin(b); i < g.end(b); ++i) {
            if (mask[i] != 0U) {
                valid.push_back(x[i]);
            }
        }
        const auto len = static_cast<double>(g.end(b) - g.begin(b));
        REQUIRE_THAT(good[b],
                     WithinAbs(static_cast<double>(valid.size()) / len, 1e-6));
        const bool ok =
            static_cast<double>(valid.size()) >= 0.3 * len && valid.size() >= 3;
        REQUIRE((usable[b] != 0U) == ok);
        if (ok) {
            REQUIRE_THAT(med[b], WithinAbs(brute_median(valid), 1e-6));
        }
    }
}

TEST_CASE("preprocess kernels: masked running median and mean",
          "[preprocess]") {
    const SizeType n = 200;
    const auto v     = normal(n, 2);
    std::vector<std::uint8_t> use(n, 1U);
    for (SizeType i = 50; i < 60; ++i) {
        use[i] = 0U;
    }
    const SizeType k  = 7;
    const auto median = pd::masked_running_median(v, use, k, 3);
    const auto mean   = pd::masked_running_mean(v, use, k);
    for (SizeType i = k; i + k < n; ++i) { // interior: no reflection
        std::vector<float> w;
        double s = 0.0;
        for (SizeType j = i - (k / 2); j <= i + (k / 2); ++j) {
            if (use[j] != 0U) {
                w.push_back(v[j]);
                s += v[j];
            }
        }
        if (!w.empty()) {
            REQUIRE_THAT(median[i], WithinAbs(brute_median(w), 1e-6));
            REQUIRE_THAT(mean[i],
                         WithinAbs(s / static_cast<double>(w.size()), 1e-5));
        }
    }
    // A long gap is bridged by interpolation.
    std::vector<std::uint8_t> gap(n, 1U);
    std::fill(gap.begin() + 80, gap.begin() + 120, 0U);
    std::vector<float> ramp(n);
    for (SizeType i = 0; i < n; ++i) {
        ramp[i] = static_cast<float>(i);
    }
    const auto bridged = pd::masked_running_median(ramp, gap, 5, 1);
    REQUIRE_THAT(bridged[100], WithinAbs(100.0, 0.1));
    // Point reflection keeps a trend unbiased at the edges.
    REQUIRE_THAT(bridged[0], WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(bridged[n - 1], WithinAbs(n - 1.0, 1e-3));
    const auto smooth = pd::reflected_running_mean(ramp, 9);
    REQUIRE_THAT(smooth[0], WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(smooth[n - 1], WithinAbs(n - 1.0, 1e-3));
    REQUIRE_THROWS(pd::masked_running_median(
        ramp, std::vector<std::uint8_t>(n, 0U), 5, 1));
}

TEST_CASE("preprocess kernels: block variance is Gaussian consistent",
          "[preprocess]") {
    const SizeType n = 1U << 16U;
    auto x           = normal(n, 3);
    for (auto& v : x) {
        v = (2.0F * v) + 5.0F;
    }
    const pd::BlockGrid g(n, 64);
    const std::vector<float> mu(g.nblocks, 5.0F);
    const std::vector<std::uint8_t> mask(n, 1U);
    std::vector<float> var(g.nblocks);
    std::vector<std::uint8_t> use(g.nblocks);
    pd::block_variances(x, mask, g, mu, 0.3, var, use, 2);
    double s = 0.0;
    for (const float v : var) {
        s += v;
    }
    REQUIRE_THAT(s / static_cast<double>(g.nblocks), WithinRel(4.0, 0.05));
    REQUIRE(std::ranges::all_of(use, [](auto u) { return u == 1U; }));
}

TEST_CASE("preprocess kernels: bad-block flags", "[preprocess]") {
    const SizeType n = 1U << 16U;
    auto z           = normal(n, 4);
    for (SizeType i = 10'000; i < 10'256; ++i) {
        z[i] += 1.0F; // offset
    }
    for (SizeType i = 30'000; i < 30'256; ++i) {
        z[i] *= 3.0F; // variance
    }
    for (SizeType i = 50'000; i < 50'256; ++i) {
        z[i] *= 0.05F; // dropout
    }
    z[40'000] = 100.0F; // isolated spike: left to the per-sample clip
    const std::vector<SizeType> scales{16, 64, 256};
    std::vector<std::uint8_t> mask(n);
    pd::flag_bad_blocks(z, scales, 6.0, 0.3, 6.0, mask, 4);
    const auto masked = [&](SizeType lo, SizeType hi) {
        SizeType c = 0;
        for (SizeType i = lo; i < hi; ++i) {
            c += mask[i] == 0U ? 1 : 0;
        }
        return c;
    };
    REQUIRE(masked(10'000, 10'256) >= 200);
    REQUIRE(masked(30'000, 30'256) >= 200);
    REQUIRE(masked(50'000, 50'256) >= 200);
    REQUIRE(mask[40'000] == 1U);
    REQUIRE(masked(0, n) < 2000);
    std::vector<std::uint8_t> mask8(n);
    pd::flag_bad_blocks(z, scales, 6.0, 0.3, 6.0, mask8, 1);
    REQUIRE(mask == mask8);
}

TEST_CASE("preprocess kernels: power threshold", "[preprocess]") {
    // P(Exp(1) > t) == Q(sigma).
    for (const double s : {3.0, 5.0, 8.0}) {
        const double q = 0.5 * std::erfc(s / std::numbers::sqrt2);
        REQUIRE_THAT(pd::exp_power_threshold(s), WithinRel(-std::log(q), 1e-9));
    }
    // Continuous across the asymptotic switch.
    REQUIRE_THAT(pd::exp_power_threshold(29.999),
                 WithinRel(pd::exp_power_threshold(30.0), 1e-3));
}

TEST_CASE("preprocess kernels: spectral zap", "[preprocess]") {
    const SizeType n   = 1U << 16U;
    const double tsamp = 1e-3;
    auto z             = normal(n, 5);
    const auto clean   = z;
    const double f0    = 64.0 / (static_cast<double>(n) * tsamp) * 101.0;
    for (SizeType i = 0; i < n; ++i) {
        z[i] += static_cast<float>(std::sin(2.0 * std::numbers::pi * f0 *
                                            static_cast<double>(i) * tsamp));
    }
    SECTION("no flagged bins leaves the data untouched") {
        auto y              = clean;
        const SizeType nzap = pd::zap_spectrum(y, tsamp, true, 8.0, 101, {}, 2);
        REQUIRE(nzap == 0);
        REQUIRE(y == clean);
    }
    SECTION("a strong tone is removed, the rest is preserved") {
        auto y              = z;
        const SizeType nzap = pd::zap_spectrum(y, tsamp, true, 8.0, 101, {}, 2);
        REQUIRE(nzap >= 1);
        REQUIRE(nzap < 10);
        double err = 0.0;
        double ref = 0.0;
        for (SizeType i = 0; i < n; ++i) {
            err += (y[i] - clean[i]) * (y[i] - clean[i]);
            ref += clean[i] * clean[i];
        }
        REQUIRE(err / ref < 0.01);
    }
    SECTION("birdies zap without a threshold") {
        auto y = z;
        const std::vector<loki::io::Birdie> b{{.freq = f0, .width = 0.0}};
        const SizeType nzap = pd::zap_spectrum(y, tsamp, false, 8.0, 101, b, 2);
        REQUIRE(nzap == 1);
    }
}

TEST_CASE("preprocess: z-score method reproduces the legacy path",
          "[preprocess]") {
    const SizeType n   = 100'003;
    const double tsamp = 1e-3;
    auto raw           = normal(n, 6);
    for (SizeType i = 0; i < n; ++i) {
        raw[i] +=
            static_cast<float>(5.0 * std::sin(static_cast<double>(i) * 1e-4));
    }
    for (const bool fast : {true, false}) {
        loki::io::PreprocessOptions o;
        o.method        = loki::io::PreprocessMethod::kZScore;
        o.filter_window = 2.5;
        o.fast_median   = fast;
        std::vector<float> e(n);
        std::vector<float> v(n);
        loki::io::preprocess(raw, tsamp, e, v, o, loki::Exec::cpu(4));

        // Legacy TimeSeries::read preprocessing.
        auto ref = raw;
        loki::math::subtract_running_filter(
            ref, 2500, loki::math::FilterMethod::kMedian, fast, 101, 4);
        loki::math::zscore(ref, loki::LocMethod::kMean, loki::ScaleMethod::kIqr,
                           4);
        REQUIRE(e == ref);
        REQUIRE(std::ranges::all_of(v, [](float a) { return a == 1.0F; }));
    }
}

TEST_CASE("rfft/irfft round trip is exact for long transforms",
          "[preprocess][fft]") {
    // Regression: plans were made with in == out (nullptr) dummies, i.e.
    // in-place, but executed out of place, which broke large transforms.
    const SizeType n = 1U << 20U;
    const auto x     = normal(n, 7);
    auto in          = x;
    std::vector<loki::ComplexType> spec((n / 2) + 1);
    loki::math::rfft_batch(in, spec, 1, n, 2);
    REQUIRE(in == x);
    std::vector<float> back(n);
    loki::math::irfft_batch(spec, back, 1, n, 2);
    double err = 0.0;
    for (SizeType i = 0; i < n; ++i) {
        err = std::max(err, static_cast<double>(std::abs(back[i] - x[i])));
    }
    REQUIRE(err < 1e-4);
}
