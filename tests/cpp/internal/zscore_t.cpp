#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"

#include "lib/detail/math.hpp"

#include "math_test_utils.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using loki::LocMethod;
using loki::ScaleMethod;
using loki::SizeType;
using loki::math::estimate_loc;
using loki::math::estimate_scale;
using loki::math::zscore;
using loki::test::Pattern;
using loki::test::TuningGuard;

namespace {

// Reference values below come from the Python implementation (numpy
// percentile / median / nanmedian) on these inputs.
const std::vector<float> kDataA = {
    0.001230153371579945F,
    0.2987455427646637F,
    -0.27413785457611084F,
    -0.8905918598175049F,
    -0.454670786857605F,
    -0.9916465282440186F,
    0.0601436011493206F,
    1.3402152061462402F,
    -0.49220651388168335F,
    -0.6204748749732971F,
    0.4898420572280884F,
    0.35688701272010803F,
    0.1054142490029335F,
    -0.9304680228233337F,
    -0.02925182320177555F,
    0.695303201675415F,
    -1.3442145586013794F,
    -0.45761576294898987F,
    -1.9012227058410645F,
    -1.289537787437439F,
    -1.8417350053787231F,
    -0.23509113490581512F,
    -1.267446517944336F,
    0.27126434445381165F,
    0.15675108134746552F,
    -0.18693093955516815F,
    -2.5167596340179443F,
    -0.5386928915977478F,
    -0.048500943928956985F,
    0.11330898851156235F,
    -1.5301357507705688F,
    -0.47775328159332275F,
    -0.978519082069397F,
    -0.8088372349739075F,
    1.0608986616134644F,
    -0.8075346946716309F,
    -0.03252170607447624F,
    0.8843898773193359F,
    -0.5836004614830017F,
    -0.11170195043087006F,
    8.0F,
    9.0F,
    -7.0F,
};
const std::vector<float> kDataB = {
    0.03671402484178543F,
    0.01610812172293663F,
    1.3559192419052124F,
    0.021009769290685654F,
    1.5839612483978271F,
    1.9244046211242676F,
    0.7966726422309875F,
    0.04123401269316673F,
    0.5137655138969421F,
    2.829310894012451F,
    0.665509819984436F,
    1.3133658170700073F,
    0.020341187715530396F,
    0.43793854117393494F,
    0.08202406764030457F,
    // NOLINTNEXTLINE(modernize-use-std-numbers): sample data, not a constant
    0.5643460154533386F,
    0.017155449837446213F,
    0.5450426340103149F,
    1.7253413200378418F,
    0.555385410785675F,
    0.09155640006065369F,
    0.3153581917285919F,
    0.04540262371301651F,
    1.293548822402954F,
    0.4409172534942627F,
    0.08690307289361954F,
    0.8520565032958984F,
    1.2255598306655884F,
    1.5226483345031738F,
    0.708365797996521F,
    0.520306408405304F,
    2.8123621940612793F,
    0.3152175843715668F,
    0.03034462034702301F,
    1.4093234539031982F,
    0.5724144577980042F,
    0.1871744841337204F,
    0.2237640917301178F,
    0.12514658272266388F,
    1.8805125951766968F,
    0.28002938628196716F,
};

constexpr double kGaussIqr = 1.3489795003921634;
constexpr double kGaussMad = 1.482602218505602;

double ref_quantile(std::vector<float> v, double q) {
    std::ranges::sort(v);
    const double pos  = static_cast<double>(v.size() - 1) * q;
    const auto lo     = static_cast<SizeType>(std::floor(pos));
    const SizeType hi = std::min(lo + 1, v.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    if (frac <= 0.0) {
        return static_cast<double>(v[lo]);
    }
    return std::lerp(static_cast<double>(v[lo]), static_cast<double>(v[hi]),
                     frac);
}

double ref_mean(const std::vector<float>& x) {
    double s = 0.0;
    for (const float v : x) {
        s += static_cast<double>(v);
    }
    return s / static_cast<double>(x.size());
}

double ref_std(const std::vector<float>& x) {
    const double mu = ref_mean(x);
    double s        = 0.0;
    for (const float v : x) {
        s += (static_cast<double>(v) - mu) * (static_cast<double>(v) - mu);
    }
    return std::sqrt(s / static_cast<double>(x.size()));
}

double ref_mad(const std::vector<float>& x) {
    const double med = ref_quantile(x, 0.5);
    std::vector<float> dev(x.size());
    double aad = 0.0;
    for (SizeType i = 0; i < x.size(); ++i) {
        const double d = std::abs(static_cast<double>(x[i]) - med);
        dev[i]         = static_cast<float>(d);
        aad += d;
    }
    const double mad = ref_quantile(dev, 0.5);
    if (mad > 0.0) {
        return mad * kGaussMad;
    }
    return aad / static_cast<double>(x.size()) *
           std::sqrt(std::numbers::pi / 2.0);
}

std::vector<float> denormal_series(SizeType n) {
    std::vector<float> x(n);
    loki::math::PCG32 rng(77);
    for (SizeType i = 0; i < n; ++i) {
        const float mag = static_cast<float>(rng() % 1000U) * 1.0e-41F;
        x[i]            = (rng() & 1U) != 0U ? mag : -mag;
    }
    return x;
}

} // namespace

TEST_CASE("estimators match the Python reference", "[math][zscore]") {
    SECTION("data A (outliers)") {
        REQUIRE_THAT(estimate_loc(kDataA, LocMethod::kMean),
                     WithinAbs(-0.13505596119293206, 1e-6));
        REQUIRE_THAT(estimate_loc(kDataA, LocMethod::kMedian),
                     WithinAbs(-0.27413785457611084, 1e-6));
        REQUIRE_THAT(estimate_scale(kDataA, ScaleMethod::kStd).left,
                     WithinRel(2.29120742553882, 1e-6));
        REQUIRE_THAT(estimate_scale(kDataA, ScaleMethod::kIqr).left,
                     WithinRel(0.7750747701844077, 1e-6));
        REQUIRE_THAT(estimate_scale(kDataA, ScaleMethod::kMad).left,
                     WithinRel(0.808614510259597, 1e-6));
        const auto dm = estimate_scale(kDataA, ScaleMethod::kDoubleMad);
        REQUIRE_THAT(dm.left, WithinRel(0.9435163196465212, 1e-6));
        REQUIRE_THAT(dm.right, WithinRel(0.6066332207222943, 1e-6));
    }
    SECTION("data B (skewed, even size) ") {
        REQUIRE_THAT(estimate_loc(kDataB, LocMethod::kMean),
                     WithinAbs(0.7313283669149003, 1e-6));
        REQUIRE_THAT(estimate_loc(kDataB, LocMethod::kMedian),
                     WithinAbs(0.520306408405304, 1e-6));
        REQUIRE_THAT(estimate_scale(kDataB, ScaleMethod::kStd).left,
                     WithinRel(0.7431092243180706, 1e-6));
        REQUIRE_THAT(estimate_scale(kDataB, ScaleMethod::kIqr).left,
                     WithinRel(0.8910383159958085, 1e-6));
        REQUIRE_THAT(estimate_scale(kDataB, ScaleMethod::kMad).left,
                     WithinRel(0.6497983707500163, 1e-6));
        const auto dm = estimate_scale(kDataB, ScaleMethod::kDoubleMad);
        REQUIRE_THAT(dm.left, WithinRel(0.6356657135560738, 1e-6));
        REQUIRE_THAT(dm.right, WithinRel(1.1464109184355433, 1e-6));
    }
}

TEST_CASE("order statistics agree across code paths and a sorted reference",
          "[math][zscore]") {
    const bool radix = GENERATE(false, true);
    const TuningGuard guard;
    TuningGuard::radix_from(radix ? 1 : (1U << 30U));

    std::vector<std::pair<std::string, std::vector<float>>> cases;
    for (const SizeType n : {1U, 2U, 3U, 10U, 257U, 5000U}) {
        for (const auto pattern : loki::test::kAllPatterns) {
            cases.emplace_back(loki::test::pattern_name(pattern) + "/" +
                                   std::to_string(n),
                               loki::test::make_series(pattern, n, 42 + n));
        }
        cases.emplace_back("denormal/" + std::to_string(n), denormal_series(n));
    }
    for (const auto& [name, x] : cases) {
        INFO("radix " << radix << " case " << name);
        REQUIRE(estimate_loc(x, LocMethod::kMedian) == ref_quantile(x, 0.5));
        const double iqr =
            (ref_quantile(x, 0.75) - ref_quantile(x, 0.25)) / kGaussIqr;
        REQUIRE_THAT(estimate_scale(x, ScaleMethod::kIqr).left,
                     WithinAbs(iqr, 1e-12 * std::max(1.0, std::abs(iqr))));
        const double mad = ref_mad(x);
        REQUIRE_THAT(estimate_scale(x, ScaleMethod::kMad).left,
                     WithinAbs(mad, 1e-12 * std::max(1.0, std::abs(mad))));
    }
}

TEST_CASE("large arrays use the parallel radix select and nth_element alike",
          "[math][zscore]") {
    const auto x   = loki::test::make_series(Pattern::kRandom, 300001, 8);
    double med_nth = 0.0;
    double mad_nth = 0.0;
    double iqr_nth = 0.0;
    {
        const TuningGuard guard;
        TuningGuard::radix_from(1U << 30U);
        med_nth = estimate_loc(x, LocMethod::kMedian);
        mad_nth = estimate_scale(x, ScaleMethod::kMad).left;
        iqr_nth = estimate_scale(x, ScaleMethod::kIqr).left;
    }
    {
        const TuningGuard guard;
        TuningGuard::radix_from(1);
        REQUIRE(estimate_loc(x, LocMethod::kMedian) == med_nth);
        REQUIRE(estimate_scale(x, ScaleMethod::kMad).left == mad_nth);
        REQUIRE(estimate_scale(x, ScaleMethod::kIqr).left == iqr_nth);
    }
    REQUIRE(med_nth == ref_quantile(x, 0.5));
}

TEST_CASE("mean and std match double references", "[math][zscore]") {
    for (const SizeType n : {SizeType{1}, SizeType{5}, SizeType{100000}}) {
        const auto x = loki::test::make_series(Pattern::kRandom, n, 4);
        REQUIRE_THAT(estimate_loc(x, LocMethod::kMean),
                     WithinAbs(ref_mean(x), 1e-12));
        REQUIRE_THAT(estimate_scale(x, ScaleMethod::kStd).left,
                     WithinAbs(ref_std(x), 1e-12));
    }
}

TEST_CASE("kNone leaves location and scale at 0 and 1", "[math][zscore]") {
    REQUIRE(estimate_loc(kDataA, LocMethod::kNone) == 0.0);
    REQUIRE(estimate_scale(kDataA, ScaleMethod::kNone).left == 1.0);

    auto x         = kDataA;
    const auto res = zscore(x, LocMethod::kNone, ScaleMethod::kNone);
    REQUIRE(x == kDataA);
    REQUIRE(res.loc == 0.0);
    REQUIRE(res.scale.left == 1.0);

    auto y = kDataA;
    zscore(y, LocMethod::kNone, ScaleMethod::kStd);
    const double sd = ref_std(kDataA);
    for (SizeType i = 0; i < y.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(y[i]),
                     WithinAbs(static_cast<double>(kDataA[i]) / sd, 1e-5));
    }
}

TEST_CASE("zscore applies (x - loc) / scale in place", "[math][zscore]") {
    const SizeType n = GENERATE(SizeType{43}, SizeType{70001});
    const auto src   = loki::test::make_series(Pattern::kRandom, n, 12);
    for (const auto loc : {LocMethod::kMean, LocMethod::kMedian}) {
        for (const auto scale :
             {ScaleMethod::kStd, ScaleMethod::kIqr, ScaleMethod::kMad}) {
            auto x         = src;
            const auto res = zscore(x, loc, scale);
            REQUIRE(res.loc == estimate_loc(src, loc));
            REQUIRE(res.scale.left == estimate_scale(src, scale).left);
            REQUIRE(res.scale.left == res.scale.right);
            for (SizeType i = 0; i < n; i += 3) {
                const double expect =
                    (static_cast<double>(src[i]) - res.loc) / res.scale.left;
                REQUIRE_THAT(static_cast<double>(x[i]),
                             WithinAbs(expect, 1e-5));
            }
        }
    }
}

TEST_CASE("zscore with double MAD scales each side separately",
          "[math][zscore]") {
    auto x         = kDataA;
    const auto res = zscore(x, LocMethod::kMedian, ScaleMethod::kDoubleMad);
    REQUIRE(res.scale.left != res.scale.right);
    for (SizeType i = 0; i < x.size(); ++i) {
        const double d = static_cast<double>(kDataA[i]) - res.loc;
        const double s = d < 0.0 ? res.scale.left : res.scale.right;
        REQUIRE_THAT(static_cast<double>(x[i]), WithinAbs(d / s, 1e-5));
    }
}

TEST_CASE("zscore is idempotent for mean/std", "[math][zscore]") {
    auto x = loki::test::make_series(Pattern::kRandom, 5000, 1);
    for (float& v : x) {
        v = (v * 7.0F) + 100.0F;
    }
    zscore(x, LocMethod::kMean, ScaleMethod::kStd);
    const auto res = zscore(x, LocMethod::kMean, ScaleMethod::kStd);
    REQUIRE_THAT(res.loc, WithinAbs(0.0, 1e-5));
    REQUIRE_THAT(res.scale.left, WithinAbs(1.0, 1e-5));
}

TEST_CASE("constant series is centred without NaN or inf", "[math][zscore]") {
    for (const auto scale : {
             ScaleMethod::kStd,
             ScaleMethod::kIqr,
             ScaleMethod::kMad,
             ScaleMethod::kDoubleMad,
             ScaleMethod::kNone,
         }) {
        std::vector<float> x(100, 4.25F);
        const auto res = zscore(x, LocMethod::kMedian, scale);
        REQUIRE(res.loc == 4.25);
        for (const float v : x) {
            REQUIRE(v == 0.0F);
        }
    }
}

TEST_CASE("MAD falls back to the mean absolute deviation when it is zero",
          "[math][zscore]") {
    // More than half the samples equal the median, so the raw MAD is 0.
    std::vector<float> x(60, 0.0F);
    for (int i = 1; i <= 10; ++i) {
        x.push_back(static_cast<float>(i));
    }
    const double aad    = 55.0 / 70.0;
    const double expect = aad * std::sqrt(std::numbers::pi / 2.0);
    REQUIRE_THAT(estimate_scale(x, ScaleMethod::kMad).left,
                 WithinRel(expect, 1e-9));
    const auto dm = estimate_scale(x, ScaleMethod::kDoubleMad);
    // Left side: only the zeros (all deviations 0) -> aad 0; right: all.
    REQUIRE(dm.left == 0.0);
    REQUIRE_THAT(dm.right, WithinRel(expect * 70.0 / 70.0, 1e-9));
}

TEST_CASE("estimators reject empty input", "[math][zscore]") {
    std::vector<float> x;
    REQUIRE_THROWS_AS(estimate_loc(x, LocMethod::kMean), std::invalid_argument);
    REQUIRE_THROWS_AS(estimate_loc(x, LocMethod::kNone), std::invalid_argument);
    REQUIRE_THROWS_AS(estimate_scale(x, ScaleMethod::kMad),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(zscore(x, LocMethod::kMean, ScaleMethod::kStd),
                      std::invalid_argument);
}

TEST_CASE("estimators give the same result for any thread count",
          "[math][zscore]") {
    const auto x = loki::test::make_series(Pattern::kRandom, 300001, 23);
    for (const auto scale :
         {ScaleMethod::kStd, ScaleMethod::kIqr, ScaleMethod::kMad}) {
        const auto one = estimate_scale(x, scale, 1);
        for (const int nthreads : {0, 3, 16}) {
            INFO("nthreads " << nthreads);
            const auto many = estimate_scale(x, scale, nthreads);
            if (scale == ScaleMethod::kStd) {
                REQUIRE_THAT(many.left, WithinRel(one.left, 1e-12));
            } else {
                REQUIRE(many.left == one.left);
            }
        }
    }
    auto a = x;
    auto b = x;
    zscore(a, LocMethod::kMedian, ScaleMethod::kMad, 1);
    zscore(b, LocMethod::kMedian, ScaleMethod::kMad, 16);
    REQUIRE(a == b);
}
