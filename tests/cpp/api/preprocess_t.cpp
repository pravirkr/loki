#include "loki/io/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/timeseries.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using loki::SizeType;
using loki::io::GainModel;
using loki::io::PreprocessMethod;
using loki::io::PreprocessOptions;
using loki::io::PreprocessReport;

namespace {

constexpr double kTsamp = 1e-3;

std::vector<float> white(SizeType n, unsigned seed, double sigma = 1.0) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, sigma);
    std::vector<float> x(n);
    for (auto& v : x) {
        v = static_cast<float>(nd(rng));
    }
    return x;
}

struct Out {
    std::vector<float> e;
    std::vector<float> v;
    PreprocessReport rep;
};

Out run(const std::vector<float>& raw,
        const PreprocessOptions& o = {},
        int nthreads               = 4,
        double tsamp               = kTsamp) {
    Out out{std::vector<float>(raw.size()), std::vector<float>(raw.size()), {}};
    out.rep = loki::io::preprocess(raw, tsamp, out.e, out.v, o,
                                   loki::Exec::cpu(nthreads));
    return out;
}

double median(std::vector<float> v) {
    std::ranges::sort(v);
    return v[v.size() / 2];
}

/// Mean of ts_v over [lo, hi), counting only weighted samples.
double mean_weight(const Out& o, SizeType lo, SizeType hi) {
    double s   = 0.0;
    SizeType c = 0;
    for (SizeType i = lo; i < hi; ++i) {
        if (o.v[i] > 0.0F) {
            s += o.v[i];
            ++c;
        }
    }
    return c == 0 ? 0.0 : s / static_cast<double>(c);
}

SizeType count_zero(const Out& o, SizeType lo, SizeType hi) {
    SizeType c = 0;
    for (SizeType i = lo; i < hi; ++i) {
        c += o.v[i] == 0.0F ? 1 : 0;
    }
    return c;
}

/// Profile of ts_e / ts_v folded at @p period samples into @p nbins.
void fold(const Out& o,
          double period,
          SizeType nbins,
          std::vector<double>& pe,
          std::vector<double>& pv) {
    pe.assign(nbins, 0.0);
    pv.assign(nbins, 0.0);
    for (SizeType i = 0; i < o.e.size(); ++i) {
        const double ph = std::fmod(static_cast<double>(i) / period, 1.0);
        const auto b = std::min(static_cast<SizeType>(ph * nbins), nbins - 1);
        pe[b] += o.e[i];
        pv[b] += o.v[i];
    }
}

/// S/N of the on-pulse bins [0, width) against the off-pulse baseline.
double pulse_snr(const Out& o, double period, SizeType nbins, SizeType width) {
    std::vector<double> pe;
    std::vector<double> pv;
    fold(o, period, nbins, pe, pv);
    double e_on  = 0.0;
    double v_on  = 0.0;
    double e_off = 0.0;
    double v_off = 0.0;
    for (SizeType b = 0; b < nbins; ++b) {
        (b < width ? e_on : e_off) += pe[b];
        (b < width ? v_on : v_off) += pv[b];
    }
    const double r = v_on / v_off;
    return (e_on - (r * e_off)) / std::sqrt(v_on * (1.0 + r));
}

/// pulse_snr() in units of its own noise, measured by folding at wrong
/// periods. Fair across methods even when `ts_v` misstates the noise.
double
calibrated_snr(const Out& o, double period, SizeType nbins, SizeType width) {
    constexpr int kTrials = 128;
    double s2             = 0.0;
    for (int j = 1; j <= kTrials; ++j) {
        const double s =
            pulse_snr(o, period * (1.0 + (0.0137 * j)), nbins, width);
        s2 += s * s;
    }
    return pulse_snr(o, period, nbins, width) / std::sqrt(s2 / kTrials);
}

/// Amplitude of the DFT of @p x at @p freq (Hz).
double tone_amplitude(const std::vector<float>& x, double freq) {
    std::complex<double> acc{0.0, 0.0};
    const double w = 2.0 * std::numbers::pi * freq * kTsamp;
    for (SizeType i = 0; i < x.size(); ++i) {
        acc += static_cast<double>(x[i]) *
               std::polar(1.0, -w * static_cast<double>(i));
    }
    return std::abs(acc) / static_cast<double>(x.size());
}

void require_contract(const Out& o) {
    bool any = false;
    for (SizeType i = 0; i < o.e.size(); ++i) {
        REQUIRE(std::isfinite(o.e[i]));
        REQUIRE(std::isfinite(o.v[i]));
        REQUIRE(o.v[i] >= 0.0F);
        if (o.v[i] == 0.0F) {
            REQUIRE(o.e[i] == 0.0F);
        }
        any = any || o.v[i] > 0.0F;
    }
    REQUIRE(any);
}

} // namespace

TEST_CASE("preprocess: stationary white noise is untouched and N(0,1)",
          "[io][preprocess]") {
    const SizeType n = 1U << 20U;
    auto raw         = white(n, 1, 3.0);
    for (auto& v : raw) {
        v += 50.0F;
    }
    const auto o = run(raw);
    require_contract(o);
    REQUIRE(o.rep.method == PreprocessMethod::kRobust);
    REQUIRE(o.rep.n_masked < n / 200);
    REQUIRE(o.rep.n_clipped < 20);
    REQUIRE_THAT(median(o.v), WithinAbs(1.0, 0.02));
    REQUIRE_THAT(o.rep.global_scale, WithinRel(3.0, 0.03));

    // Folded bin statistics are unit normal.
    std::vector<double> pe;
    std::vector<double> pv;
    fold(o, 1234.567, 512, pe, pv);
    double s1 = 0.0;
    double s2 = 0.0;
    for (SizeType b = 0; b < pe.size(); ++b) {
        const double z = pe[b] / std::sqrt(pv[b]);
        s1 += z;
        s2 += z * z;
    }
    const auto nb   = static_cast<double>(pe.size());
    const double mu = s1 / nb;
    REQUIRE_THAT(mu, WithinAbs(0.0, 0.2));
    REQUIRE_THAT(std::sqrt((s2 / nb) - (mu * mu)), WithinAbs(1.0, 0.1));
}

TEST_CASE("preprocess: weights follow a variance step", "[io][preprocess]") {
    const SizeType n = 1U << 19U;
    auto raw         = white(n, 2);
    for (SizeType i = n / 2; i < n; ++i) {
        raw[i] *= 3.0F;
    }
    const auto o = run(raw);
    require_contract(o);
    const double w1 = mean_weight(o, 0, (n / 2) - 5000);
    const double w2 = mean_weight(o, (n / 2) + 5000, n);
    REQUIRE_THAT(w2 / w1, WithinRel(1.0 / 9.0, 0.1));
    REQUIRE(o.rep.n_masked < n / 100);
}

TEST_CASE("preprocess: red noise, drift and baseline wander are not masked",
          "[io][preprocess]") {
    const SizeType n = 1U << 20U;
    auto raw         = white(n, 3);
    // Red noise: AR(1) with a 10 s correlation time and 2 sigma rms, plus a
    // linear drift and a 30 s sinusoid of 20 sigma.
    std::mt19937_64 rng(33);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double a = std::exp(-kTsamp / 10.0);
    double red     = 0.0;
    for (SizeType i = 0; i < n; ++i) {
        red            = (a * red) + (std::sqrt(1.0 - (a * a)) * 2.0 * nd(rng));
        const double t = static_cast<double>(i) * kTsamp;
        raw[i] += static_cast<float>(
            red + (10.0 * t / (static_cast<double>(n) * kTsamp)) +
            (20.0 * std::sin(2.0 * std::numbers::pi * t / 30.0)));
    }
    const auto o = run(raw);
    require_contract(o);
    REQUIRE(o.rep.n_masked < n / 1000);
}

TEST_CASE("preprocess: RFI bursts are masked, spikes are clipped",
          "[io][preprocess]") {
    const SizeType n = 1U << 20U;
    auto raw         = white(n, 4);
    // Offset burst (0.5 s, +3 sigma) and variance explosion (0.3 s, x5).
    const SizeType b0 = 200'000;
    const SizeType b1 = b0 + 500;
    const SizeType v0 = 600'000;
    const SizeType v1 = v0 + 300;
    for (SizeType i = b0; i < b1; ++i) {
        raw[i] += 3.0F;
    }
    for (SizeType i = v0; i < v1; ++i) {
        raw[i] *= 5.0F;
    }
    // Isolated 50 sigma spikes.
    for (SizeType k = 0; k < 100; ++k) {
        raw[(k * 9973) + 1500] += 50.0F;
    }
    const auto o = run(raw);
    require_contract(o);
    REQUIRE(count_zero(o, b0, b1) >= 450);
    REQUIRE(count_zero(o, v0, v1) >= 270);
    REQUIRE(o.rep.n_clipped >= 100);
    REQUIRE(o.rep.n_masked < n / 50);
    REQUIRE(o.rep.longest_masked_run >= 300);
    for (SizeType k = 0; k < 100; ++k) {
        REQUIRE(o.v[(k * 9973) + 1500] == 0.0F);
    }
}

TEST_CASE("preprocess: masked blocks keep their neighbours unbiased",
          "[io][preprocess]") {
    const SizeType n = 1U << 19U;
    auto raw         = white(n, 5);
    for (SizeType i = 100'000; i < 101'000; ++i) {
        raw[i] += 20.0F;
    }
    const auto o = run(raw);
    // Weight next to the masked region matches the rest of the series.
    REQUIRE_THAT(mean_weight(o, 101'000, 103'000) / mean_weight(o, 0, n),
                 WithinRel(1.0, 0.1));
}

TEST_CASE("preprocess: multiplicative gain", "[io][preprocess]") {
    const SizeType n = 1U << 19U;
    auto raw         = white(n, 6);
    // Baseline 50 -> 150, noise 1% of the baseline.
    for (SizeType i = 0; i < n; ++i) {
        const double mu =
            50.0 + (100.0 * static_cast<double>(i) / static_cast<double>(n));
        raw[i] = static_cast<float>(mu + (0.01 * mu * raw[i]));
    }
    PreprocessOptions opt;
    opt.gain_model = GainModel::kMultiplicative;
    const auto om  = run(raw, opt);
    require_contract(om);
    const double head = mean_weight(om, 5000, 50'000);
    const double tail = mean_weight(om, n - 50'000, n - 5000);
    REQUIRE_THAT(tail / head, WithinRel(1.0, 0.05));
    const auto oa = run(raw);
    REQUIRE(mean_weight(oa, n - 50'000, n - 5000) /
                mean_weight(oa, 5000, 50'000) <
            0.2);

    // Mean-subtracted data has no positive baseline.
    REQUIRE_THROWS_AS(run(white(n, 7), opt), std::invalid_argument);
}

TEST_CASE("preprocess: periodic zap and birdies", "[io][preprocess]") {
    const SizeType n    = 1U << 20U;
    auto raw            = white(n, 8);
    const double f_zap  = 50.0;
    const double f_bird = 37.0;
    for (SizeType i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * kTsamp;
        raw[i] += static_cast<float>(
            (0.5 * std::sin(2.0 * std::numbers::pi * f_zap * t)) +
            (0.006 * std::sin(2.0 * std::numbers::pi * f_bird * t)));
    }
    const auto off = run(raw);
    REQUIRE(off.rep.n_zapped == 0);
    REQUIRE(tone_amplitude(off.e, f_zap) > 0.2);
    const double weak = tone_amplitude(off.e, f_bird);
    REQUIRE(weak > 0.0015);

    PreprocessOptions opt;
    opt.zap_periodic = true;
    const auto on    = run(raw, opt);
    require_contract(on);
    REQUIRE(on.rep.n_zapped >= 1);
    REQUIRE(tone_amplitude(on.e, f_zap) < 0.01);
    // The weak tone is below threshold and survives.
    REQUIRE_THAT(tone_amplitude(on.e, f_bird), WithinRel(weak, 0.25));

    opt.zap_periodic = false;
    opt.birdies      = {{.freq = f_bird, .width = 0.01}};
    const auto bird  = run(raw, opt);
    REQUIRE(bird.rep.n_zapped >= 1);
    REQUIRE(tone_amplitude(bird.e, f_bird) < 0.4 * weak);
    REQUIRE(tone_amplitude(bird.e, f_zap) > 0.2);
}

TEST_CASE("preprocess: results do not depend on the thread count",
          "[io][preprocess]") {
    const SizeType n = (1U << 19U) + 12'345;
    auto raw         = white(n, 9);
    for (SizeType i = 10'000; i < 10'400; ++i) {
        raw[i] += 4.0F;
    }
    PreprocessOptions opt;
    opt.zap_periodic = true;
    const auto a     = run(raw, opt, 1);
    const auto b     = run(raw, opt, 8);
    REQUIRE(a.e == b.e);
    REQUIRE(a.v == b.v);
    REQUIRE(a.rep.n_masked == b.rep.n_masked);
}

TEST_CASE("preprocess: robust beats z-score under non-stationary noise",
          "[io][preprocess]") {
    const SizeType n      = 1U << 20U;
    const double period   = 731.3;
    const SizeType nbins  = 64;
    const SizeType width  = 4;
    const auto add_pulsar = [&](std::vector<float>& x, double amp) {
        for (SizeType i = 0; i < n; ++i) {
            const double ph = std::fmod(static_cast<double>(i) / period, 1.0);
            x[i] += ph < static_cast<double>(width) / nbins
                        ? static_cast<float>(amp)
                        : 0.0F;
        }
    };
    PreprocessOptions zs;
    zs.method = PreprocessMethod::kZScore;

    SECTION("stationary noise: no loss") {
        for (const unsigned seed : {10U, 20U}) {
            auto raw = white(n, seed);
            add_pulsar(raw, 0.05);
            const double r = calibrated_snr(run(raw), period, nbins, width);
            const double z = calibrated_snr(run(raw, zs), period, nbins, width);
            REQUIRE(r > 0.93 * z);
        }
    }
    SECTION("variance changes and RFI: gain") {
        auto raw = white(n, 11);
        for (SizeType i = 0; i < n; ++i) {
            raw[i] *= (i / 50'000) % 2 == 0 ? 1.0F : 4.0F;
        }
        for (SizeType i = 300'000; i < 302'000; ++i) {
            raw[i] += 30.0F;
        }
        add_pulsar(raw, 0.05);
        const double r = calibrated_snr(run(raw), period, nbins, width);
        const double z = calibrated_snr(run(raw, zs), period, nbins, width);
        REQUIRE(r > 1.3 * z);
    }
}

TEST_CASE("preprocess: z-score method keeps the legacy output",
          "[io][preprocess]") {
    auto raw = white(4096, 12);
    PreprocessOptions opt;
    opt.method        = PreprocessMethod::kZScore;
    opt.filter_window = 0.0;
    opt.loc           = loki::LocMethod::kMedian;
    opt.scale         = loki::ScaleMethod::kStd;
    const auto o      = run(raw, opt);
    REQUIRE(o.rep.method == PreprocessMethod::kZScore);
    REQUIRE(std::ranges::all_of(o.v, [](float v) { return v == 1.0F; }));
    std::vector<float> s = raw;
    std::ranges::sort(s);
    const double med = 0.5 * (s[2047] + s[2048]);
    double m2        = 0.0;
    double mean      = 0.0;
    for (const float v : raw) {
        mean += v;
    }
    mean /= 4096.0;
    for (const float v : raw) {
        m2 += (v - mean) * (v - mean);
    }
    const double sd = std::sqrt(m2 / 4096.0);
    for (SizeType i = 0; i < raw.size(); i += 97) {
        REQUIRE_THAT(o.e[i], WithinAbs((raw[i] - med) / sd, 1e-4));
    }
}

TEST_CASE("preprocess: edge cases", "[io][preprocess]") {
    SECTION("series shorter than a block") {
        const auto o = run(white(20, 13));
        require_contract(o);
        REQUIRE(o.rep.block_size == 20);
    }
    SECTION("length not a multiple of the block") {
        const auto o = run(white(100'003, 14));
        require_contract(o);
        REQUIRE(o.rep.block_mu.size() ==
                (100'003 + o.rep.block_size - 1) / o.rep.block_size);
    }
    SECTION("in place") {
        const auto raw       = white(50'000, 15);
        const auto ref       = run(raw);
        std::vector<float> e = raw;
        std::vector<float> v(raw.size());
        loki::io::preprocess(e, kTsamp, e, v);
        REQUIRE(e == ref.e);
        REQUIRE(v == ref.v);
    }
    SECTION("invalid input") {
        REQUIRE_THROWS_AS(run(white(10, 16)), std::invalid_argument);
        REQUIRE_THROWS_AS(run(std::vector<float>(10'000, 3.0F)),
                          std::invalid_argument);
        auto bad = white(10'000, 17);
        // Write the NaN bits directly: -ffast-math may fold a NaN constant.
        const std::uint32_t nan_bits = 0x7FC00000U;
        std::memcpy(&bad[17], &nan_bits, sizeof(nan_bits));
        REQUIRE_THROWS_AS(run(bad), std::invalid_argument);
        std::vector<float> e(100);
        std::vector<float> v(99);
        REQUIRE_THROWS_AS(loki::io::preprocess(white(100, 18), kTsamp, e, v),
                          std::invalid_argument);
        std::vector<float> buf(200);
        REQUIRE_THROWS_AS(
            loki::io::preprocess(std::span<const float>(buf).first(100), kTsamp,
                                 std::span<float>(buf).subspan(50, 100),
                                 std::span<float>(e)),
            std::invalid_argument);
    }
    SECTION("invalid options") {
        const auto raw = white(1000, 19);
        PreprocessOptions opt;
        opt.min_good_fraction = 1.5;
        REQUIRE_THROWS_AS(run(raw, opt), std::invalid_argument);
        opt               = {};
        opt.window_blocks = 0;
        REQUIRE_THROWS_AS(run(raw, opt), std::invalid_argument);
        opt              = {};
        opt.block_scales = {-1.0};
        REQUIRE_THROWS_AS(run(raw, opt), std::invalid_argument);
    }
    SECTION("GPU backend is not implemented") {
        const auto raw = white(1000, 20);
        std::vector<float> e(raw.size());
        std::vector<float> v(raw.size());
        REQUIRE_THROWS(
            loki::io::preprocess(raw, kTsamp, e, v, {}, loki::Exec::cuda(0)));
    }
}

TEST_CASE("timeseries preprocess runs in place", "[io][preprocess]") {
    const auto raw = white(50'000, 21);
    loki::io::TimeSeries ts(raw, std::vector<float>(raw.size(), 1.0F), kTsamp);
    const auto rep = ts.preprocess({}, loki::Exec::cpu(2));
    const auto ref = run(raw);
    REQUIRE(rep.n_masked == ref.rep.n_masked);
    REQUIRE(std::ranges::equal(ts.get_ts_e(), ref.e));
    REQUIRE(std::ranges::equal(ts.get_ts_v(), ref.v));
}
