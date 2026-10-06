#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <omp.h>

#include "detail/math.hpp"
#include "math_test_utils.hpp"

using loki::SizeType;
using loki::math::FilterMethod;
using loki::math::running_filter;
using loki::math::running_filter_fast;
using loki::math::subtract_running_filter;
using loki::test::Pattern;
using loki::test::TuningGuard;

namespace {

// Values below were produced by the Python reference (np.pad "symmetric" +
// bottleneck.move_median / move_mean, window edges cut as in running_filter).
const std::vector<float> kGoldenInput = {3.0F, 1.0F, 4.0F, 1.0F, 5.0F,
                                         9.0F, 2.0F, 6.0F, 5.0F, 3.0F,
                                         5.0F, 8.0F, 9.0F, 7.0F, 9.0F};

struct GoldenCase {
    loki::SizeType window;
    std::vector<float> expected;
};

const std::vector<GoldenCase> kGoldenMedian = {
    {1,
     {3.0F, 1.0F, 4.0F, 1.0F, 5.0F, 9.0F, 2.0F, 6.0F, 5.0F, 3.0F, 5.0F, 8.0F,
      9.0F, 7.0F, 9.0F}},
    {2,
     {3.0F, 2.0F, 2.5F, 2.5F, 3.0F, 7.0F, 5.5F, 4.0F, 5.5F, 4.0F, 4.0F, 6.5F,
      8.5F, 8.0F, 8.0F}},
    {3,
     {3.0F, 3.0F, 1.0F, 4.0F, 5.0F, 5.0F, 6.0F, 5.0F, 5.0F, 5.0F, 5.0F, 8.0F,
      8.0F, 9.0F, 9.0F}},
    {4,
     {2.0F, 3.0F, 2.0F, 2.5F, 4.5F, 3.5F, 5.5F, 5.5F, 4.0F, 5.0F, 5.0F, 6.5F,
      7.5F, 8.5F, 9.0F}},
    {5,
     {3.0F, 3.0F, 3.0F, 4.0F, 4.0F, 5.0F, 5.0F, 5.0F, 5.0F, 5.0F, 5.0F, 7.0F,
      8.0F, 9.0F, 9.0F}},
    {6,
     {3.0F, 2.0F, 3.0F, 3.5F, 3.0F, 4.5F, 5.0F, 5.0F, 5.0F, 5.0F, 5.5F, 6.0F,
      7.5F, 8.5F, 8.5F}},
    {7,
     {3.0F, 3.0F, 3.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 5.0F, 5.0F, 6.0F, 7.0F,
      8.0F, 8.0F, 9.0F}},
    {20,
     {3.5F, 4.0F, 4.0F, 4.0F, 4.5F, 4.5F, 4.5F, 5.0F, 5.0F, 5.5F, 5.5F, 5.5F,
      5.5F, 6.0F, 6.0F}},
};

const std::vector<GoldenCase> kGoldenMean = {
    {1,
     {3.0F, 1.0F, 4.0F, 1.0F, 5.0F, 9.0F, 2.0F, 6.0F, 5.0F, 3.0F, 5.0F, 8.0F,
      9.0F, 7.0F, 9.0F}},
    {2,
     {3.0F, 2.0F, 2.5F, 2.5F, 3.0F, 7.0F, 5.5F, 4.0F, 5.5F, 4.0F, 4.0F, 6.5F,
      8.5F, 8.0F, 8.0F}},
    {3,
     {2.3333333333333335F, 2.6666666666666665F, 2.0F, 3.333333333333333F, 5.0F,
      5.333333333333333F, 5.666666666666666F, 4.333333333333333F,
      4.666666666666666F, 4.333333333333333F, 5.333333333333333F,
      7.333333333333333F, 8.0F, 8.333333333333332F, 8.333333333333332F}},
    {4,
     {2.0F, 2.75F, 2.25F, 2.75F, 4.75F, 4.25F, 5.5F, 5.5F, 4.0F, 4.75F, 5.25F,
      6.25F, 7.25F, 8.25F, 8.5F}},
    {5,
     {2.4F, 2.4000000000000004F, 2.8000000000000003F, 4.0F, 4.2F,
      4.6000000000000005F, 5.4F, 5.0F, 4.2F, 5.4F, 6.0F, 6.4F,
      7.6000000000000005F, 8.4F, 8.200000000000001F}},
    {6,
     {2.6666666666666665F, 2.1666666666666665F, 2.833333333333333F,
      3.833333333333333F, 3.6666666666666665F, 4.5F, 4.666666666666666F, 5.0F,
      5.0F, 4.833333333333333F, 6.0F, 6.166666666666666F, 6.833333333333333F,
      7.833333333333333F, 8.166666666666666F}},
    {7,
     {2.4285714285714284F, 2.571428571428571F, 3.714285714285714F,
      3.571428571428571F, 4.0F, 4.571428571428571F, 4.428571428571428F, 5.0F,
      5.428571428571428F, 5.428571428571428F, 6.142857142857142F,
      6.571428571428571F, 7.142857142857142F, 7.7142857142857135F,
      8.285714285714285F}},
    {20,
     {3.9F, 4.0F, 4.15F, 4.3F, 4.55F, 4.55F, 4.75F, 5.050000000000001F,
      5.300000000000001F, 5.65F, 5.75F, 5.75F, 5.95F, 6.050000000000001F,
      6.1000000000000005F}},
};

constexpr int kManyThreads = 8;

/// Symmetric reflection written as repeated single reflections (independent
/// of the closed form used by the library).
SizeType naive_reflect(long long k, SizeType n) {
    const auto len = static_cast<long long>(n);
    while (k < 0 || k >= len) {
        k = k < 0 ? -k - 1 : (2 * len) - 1 - k;
    }
    return static_cast<SizeType>(k);
}

/// Brute-force sliding window statistic, O(n w log w).
std::vector<float>
naive_filter(const std::vector<float>& x, SizeType w, FilterMethod method) {
    const SizeType n = x.size();
    const auto left  = static_cast<long long>(w / 2);
    const auto right = static_cast<long long>(w) - 1 - left;
    std::vector<float> out(n);
    std::vector<float> win(w);
    for (SizeType i = 0; i < n; ++i) {
        SizeType t = 0;
        for (long long k = static_cast<long long>(i) - left;
             k <= static_cast<long long>(i) + right; ++k) {
            win[t++] = x[naive_reflect(k, n)];
        }
        if (method == FilterMethod::kMean) {
            double acc = 0.0;
            for (float v : win) {
                acc += static_cast<double>(v);
            }
            out[i] = static_cast<float>(acc / static_cast<double>(w));
        } else {
            std::ranges::sort(win);
            const SizeType h = w / 2;
            out[i] = (w % 2 == 1) ? win[h]
                                  : static_cast<float>(
                                        0.5 * (static_cast<double>(win[h - 1]) +
                                               static_cast<double>(win[h])));
        }
    }
    return out;
}

/// Index of the first element that differs, or size() if none.
SizeType first_median_mismatch(const std::vector<float>& a,
                               const std::vector<float>& b) {
    for (SizeType i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i])) {
            return i;
        }
    }
    return a.size();
}

SizeType first_mean_mismatch(const std::vector<float>& a,
                             const std::vector<float>& b,
                             double tol) {
    for (SizeType i = 0; i < a.size(); ++i) {
        if (!(std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) <=
              tol)) {
            return i;
        }
    }
    return a.size();
}

double max_abs(const std::vector<float>& x) {
    double m = 0.0;
    for (float v : x) {
        m = std::max(m, std::abs(static_cast<double>(v)));
    }
    return m;
}

std::vector<float>
run(const std::vector<float>& x,
    SizeType w,
    FilterMethod method,
    int nthreads = 1) {
    std::vector<float> out(x.size(), -999.0F);
    running_filter(x, out, w, method, nthreads);
    return out;
}

std::vector<SizeType> window_list(SizeType n) {
    std::vector<SizeType> ws{1,  2,   3,    4, 5,     10,
                             11, 101, 1001, n, n + 1, (2 * n) + 3};
    if (n > 1) {
        ws.push_back(n - 1);
    }
    std::ranges::sort(ws);
    ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
    return ws;
}

} // namespace

TEST_CASE("running_filter matches the Python reference", "[math][filter]") {
    for (const auto& c : kGoldenMedian) {
        for (const bool heap : {false, true}) {
            const TuningGuard guard;
            TuningGuard::heap_from(heap ? 1 : 1U << 20U);
            INFO("median window " << c.window << " heap " << heap);
            const auto out = run(kGoldenInput, c.window, FilterMethod::kMedian);
            REQUIRE(out == c.expected);
        }
    }
    for (const auto& c : kGoldenMean) {
        INFO("mean window " << c.window);
        const auto out = run(kGoldenInput, c.window, FilterMethod::kMean);
        REQUIRE(first_mean_mismatch(out, c.expected, 1e-5) == out.size());
    }
}

TEST_CASE("running_filter matches a brute-force reference", "[math][filter]") {
    const bool heap = GENERATE(false, true);
    const TuningGuard guard;
    TuningGuard::heap_from(heap ? 1 : 1U << 20U);

    for (const SizeType n : {1U, 2U, 3U, 7U, 64U, 1000U}) {
        for (const auto pattern : loki::test::kAllPatterns) {
            const auto x     = loki::test::make_series(pattern, n, 1234 + n);
            const double tol = 1e-6 * std::max(1.0, max_abs(x));
            for (const SizeType w : window_list(n)) {
                INFO("heap " << heap << " pattern "
                             << loki::test::pattern_name(pattern) << " n " << n
                             << " w " << w);
                const auto med     = run(x, w, FilterMethod::kMedian);
                const auto ref_med = naive_filter(x, w, FilterMethod::kMedian);
                REQUIRE(first_median_mismatch(med, ref_med) == n);

                const auto mean     = run(x, w, FilterMethod::kMean);
                const auto ref_mean = naive_filter(x, w, FilterMethod::kMean);
                REQUIRE(first_mean_mismatch(mean, ref_mean, tol) == n);
            }
        }
    }
}

TEST_CASE("running_filter on a longer series", "[math][filter]") {
    const bool heap = GENERATE(false, true);
    const TuningGuard guard;
    TuningGuard::heap_from(heap ? 1 : 1U << 20U);
    const auto x = loki::test::make_series(Pattern::kRandom, 10007, 99);
    for (const SizeType w : {1U, 2U, 11U, 101U, 1001U}) {
        INFO("heap " << heap << " w " << w);
        REQUIRE(first_median_mismatch(
                    run(x, w, FilterMethod::kMedian),
                    naive_filter(x, w, FilterMethod::kMedian)) == x.size());
    }
}

TEST_CASE("running_filter is independent of the thread count",
          "[math][filter]") {
    // Long enough to be split into several chunks.
    const SizeType n = 200000;
    const auto x     = loki::test::make_series(Pattern::kRandom, n, 7);
    for (const SizeType w : {SizeType{5}, SizeType{101}, SizeType{2049}}) {
        for (const bool heap : {false, true}) {
            const TuningGuard guard;
            TuningGuard::heap_from(heap ? 1 : 1U << 20U);
            std::vector<float> one;
            std::vector<float> many;
            one  = run(x, w, FilterMethod::kMedian, 1);
            many = run(x, w, FilterMethod::kMedian, kManyThreads);
            INFO("w " << w << " heap " << heap);
            REQUIRE(one == many);
        }
    }
    {
        const auto ref =
            naive_filter(loki::test::make_series(Pattern::kRandom, 50000, 3),
                         101, FilterMethod::kMedian);
        const auto out =
            run(loki::test::make_series(Pattern::kRandom, 50000, 3), 101,
                FilterMethod::kMedian, kManyThreads);
        REQUIRE(out == ref);
    }
    SECTION("mean agrees within rounding") {
        const auto one = run(x, 101, FilterMethod::kMean, 1);
        const auto par = run(x, 101, FilterMethod::kMean, kManyThreads);
        REQUIRE(first_mean_mismatch(one, par, 1e-6) == n);
    }
}

TEST_CASE("subtract_running_filter equals x - running_filter",
          "[math][filter]") {
    for (const SizeType n : {SizeType{37}, SizeType{100000}}) {
        for (const SizeType w : {SizeType{1}, SizeType{4}, SizeType{11},
                                 SizeType{300}, SizeType{4001}}) {
            for (const auto method :
                 {FilterMethod::kMedian, FilterMethod::kMean}) {
                for (const bool heap : {false, true}) {
                    const TuningGuard guard;
                    TuningGuard::heap_from(heap ? 1 : 1U << 20U);
                    const auto x =
                        loki::test::make_series(Pattern::kRandom, n, 5);
                    const auto base = run(x, w, method, kManyThreads);
                    std::vector<float> expect(n);
                    for (SizeType i = 0; i < n; ++i) {
                        expect[i] = x[i] - base[i];
                    }
                    auto y = x;
                    subtract_running_filter(y, w, method, false, 101, kManyThreads);
                    INFO("n " << n << " w " << w << " heap " << heap);
                    REQUIRE(y == expect);
                }
            }
        }
    }
}

TEST_CASE("running_filter keeps long-run accuracy of the mean",
          "[math][filter]") {
    const SizeType n = 1000000;
    std::vector<float> x(n);
    loki::math::PCG32 rng(11);
    for (auto& v : x) {
        v = 1.0e5F + loki::test::uniform01(rng);
    }
    const auto out = run(x, 1000, FilterMethod::kMean);
    for (const SizeType i : {SizeType{0}, SizeType{500}, n / 2, n - 1}) {
        REQUIRE_THAT(static_cast<double>(out[i]),
                     Catch::Matchers::WithinAbs(1.0e5 + 0.5, 0.02));
    }
}

TEST_CASE("running_filter validates its arguments", "[math][filter]") {
    std::vector<float> x(10, 1.0F);
    std::vector<float> out(10);
    std::vector<float> empty;
    REQUIRE_THROWS_AS(running_filter(x, out, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(running_filter(empty, empty, 3), std::invalid_argument);
    std::vector<float> short_out(9);
    REQUIRE_THROWS_AS(running_filter(x, short_out, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(running_filter(x, x, 3), std::invalid_argument);
    const std::span<const float> head(x.data(), 9);
    const std::span<float> shifted(x.data() + 1, 9);
    REQUIRE_THROWS_AS(running_filter(head, shifted, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(running_filter_fast(x, out, 3, FilterMethod::kMean, 0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(subtract_running_filter(empty, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(subtract_running_filter(out, 0), std::invalid_argument);
}

TEST_CASE("running_filter_fast falls back to the exact filter",
          "[math][filter][fast]") {
    const auto x = loki::test::make_series(Pattern::kRandom, 5000, 21);
    for (const auto method : {FilterMethod::kMedian, FilterMethod::kMean}) {
        std::vector<float> fast(x.size());
        // ds == 1
        running_filter_fast(x, fast, 100, method, 101);
        REQUIRE(fast == run(x, 100, method));
        // fewer than min_points blocks
        running_filter_fast(x, fast, 6000, method, 101);
        REQUIRE(fast == run(x, 6000, method));
    }
}

TEST_CASE("running_filter_fast reproduces constants and ramps",
          "[math][filter][fast]") {
    const SizeType n = 40000;
    for (const SizeType w : {SizeType{1000}, SizeType{1111}, SizeType{2020}}) {
        for (const auto method : {FilterMethod::kMedian, FilterMethod::kMean}) {
            INFO("w " << w);
            std::vector<float> out(n);
            const std::vector<float> constant(n, 2.5F);
            running_filter_fast(constant, out, w, method);
            for (float v : out) {
                REQUIRE_THAT(static_cast<double>(v),
                             Catch::Matchers::WithinAbs(2.5, 1e-5));
            }
            std::vector<float> ramp(n);
            std::iota(ramp.begin(), ramp.end(), 0.0F);
            running_filter_fast(ramp, out, w, method);
            for (SizeType i = 2 * w; i < n - (2 * w); i += 7) {
                REQUIRE_THAT(
                    static_cast<double>(out[i]),
                    Catch::Matchers::WithinAbs(static_cast<double>(i), 0.05));
            }
            // Edge clamping: constant before the first block centre.
            REQUIRE(out[0] == out[1]);
            REQUIRE(out[n - 1] == out[n - 2]);
        }
    }
}

TEST_CASE("running_filter_fast follows the exact filter on smooth data",
          "[math][filter][fast]") {
    const SizeType n = 60000;
    const SizeType w = 1001;
    std::vector<float> x(n);
    loki::math::PCG32 rng(5);
    for (SizeType i = 0; i < n; ++i) {
        x[i] = std::sin(2.0F * 3.14159265F * static_cast<float>(i) / 20000.0F) +
               (0.1F * ((2.0F * loki::test::uniform01(rng)) - 1.0F));
    }
    std::vector<float> fast(n);
    running_filter_fast(x, fast, w, FilterMethod::kMedian);
    const auto exact = run(x, w, FilterMethod::kMedian);
    double worst     = 0.0;
    for (SizeType i = w; i < n - w; ++i) {
        worst =
            std::max(worst, std::abs(static_cast<double>(fast[i] - exact[i])));
    }
    REQUIRE(worst < 0.05);
}

TEST_CASE("running_filter_fast ignores narrow pulses on a slow baseline",
          "[math][filter][fast]") {
    const SizeType n = 50000;
    std::vector<float> x(n);
    std::vector<float> baseline(n);
    for (SizeType i = 0; i < n; ++i) {
        baseline[i] = 0.0002F * static_cast<float>(i);
        x[i]        = baseline[i] + ((i % 500) < 5 ? 10.0F : 0.0F);
    }
    std::vector<float> est(n);
    running_filter_fast(x, est, 2001, FilterMethod::kMedian);
    for (SizeType i = 2001; i < n - 2001; ++i) {
        REQUIRE(std::abs(est[i] - baseline[i]) < 0.1F);
    }
}

TEST_CASE("subtract_running_filter fast path equals x - running_filter_fast",
          "[math][filter][fast]") {
    const SizeType n = 100000;
    const auto x     = loki::test::make_series(Pattern::kRandom, n, 31);
    for (const SizeType w : {SizeType{1000}, SizeType{2222}, SizeType{9999}}) {
        for (const auto method : {FilterMethod::kMedian, FilterMethod::kMean}) {
            std::vector<float> base(n);
            running_filter_fast(x, base, w, method, 101, kManyThreads);
            std::vector<float> expect(n);
            for (SizeType i = 0; i < n; ++i) {
                expect[i] = x[i] - base[i];
            }
            auto y = x;
            subtract_running_filter(y, w, method, true, 101, kManyThreads);
            INFO("w " << w);
            REQUIRE(y == expect);
        }
    }
}

TEST_CASE("thread counts below 1 are treated as 1 and large counts are honoured",
          "[math][filter]") {
    const auto x = loki::test::make_series(Pattern::kRandom, 100000, 17);
    const auto ref = run(x, 51, FilterMethod::kMedian, 1);
    for (const int nthreads : {-3, 0, 64}) {
        INFO("nthreads " << nthreads);
        REQUIRE(run(x, 51, FilterMethod::kMedian, nthreads) == ref);
    }
}
