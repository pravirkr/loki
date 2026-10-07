#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"

#include "lib/detail/math.hpp"
#include "lib/detail/utils.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using loki::SizeType;

TEST_CASE("factorial", "[math]") {
    SECTION("Small integers") {
        REQUIRE(loki::math::factorial(0) == 1);
        REQUIRE(loki::math::factorial(1) == 1);
        REQUIRE(loki::math::factorial(5) == 120);
        REQUIRE(loki::math::factorial(10) == 3628800);
    }

    SECTION("Floating-point uses gamma") {
        REQUIRE_THAT(loki::math::factorial(4.0), WithinAbs(24.0, 1e-9));
        REQUIRE_THAT(loki::math::factorial(5.5),
                     WithinRel(287.8852778159886, 1e-9));
    }

    SECTION("Negative integer throws") {
        REQUIRE_THROWS_AS(loki::math::factorial(-5), std::invalid_argument);
    }
}

TEST_CASE("is_power_of_two", "[math]") {
    REQUIRE(loki::math::is_power_of_two(1));
    REQUIRE(loki::math::is_power_of_two(2));
    REQUIRE(loki::math::is_power_of_two(16384));
    REQUIRE_FALSE(loki::math::is_power_of_two(0));
    REQUIRE_FALSE(loki::math::is_power_of_two(3));
    REQUIRE_FALSE(loki::math::is_power_of_two(1023));
}

TEST_CASE("StatLookupTables exact reference", "[math]") {
    // Full table construction can overflow in Boost for extreme tail values;
    // smoke-test the exact reference helpers used to build the LUT.
    SECTION("norm_isf is finite at moderate minus_logsf") {
        REQUIRE(loki::utils::is_finite(
            loki::math::StatTables::exact_norm_isf(1.0F)));
    }

    SECTION("chi_sq_minus_logsf is finite for valid input") {
        REQUIRE(loki::utils::is_finite(
            loki::math::StatTables::exact_chi_sq_minus_logsf(5.0F, 4)));
    }

    SECTION("chi_sq_minus_logsf rejects invalid df") {
        REQUIRE_THROWS_AS(
            loki::math::StatTables::exact_chi_sq_minus_logsf(1.0F, 0),
            std::out_of_range);
    }
}

TEST_CASE("PCG32", "[math]") {
    SECTION("Fixed seed is deterministic") {
        loki::math::PCG32 rng_a(42U, 7U);
        loki::math::PCG32 rng_b(42U, 7U);
        for (int i = 0; i < 16; ++i) {
            REQUIRE(rng_a() == rng_b());
        }
    }

    SECTION("Different seeds diverge") {
        loki::math::PCG32 rng_a(1U);
        loki::math::PCG32 rng_b(2U);
        REQUIRE(rng_a() != rng_b());
    }
}

TEST_CASE("NormalSampler", "[math]") {
    SECTION("Produces finite samples with fixed seed") {
        loki::math::PCG32 rng(12345U);
        std::vector<float> samples(128);
        loki::math::NormalSampler::generate(rng, samples, 2.0F, 0.5F);
        REQUIRE(std::ranges::all_of(
            samples, [](float s) { return loki::utils::is_finite(s); }));
        const float mean =
            std::accumulate(samples.begin(), samples.end(), 0.0F) /
            static_cast<float>(samples.size());
        REQUIRE_THAT(mean, WithinAbs(2.0F, 0.5F));
    }

    SECTION("Draws depend only on the stream") {
        loki::math::PCG32 rng_a(7U, 3U);
        loki::math::PCG32 rng_b(7U, 3U);
        std::vector<float> a(64);
        std::vector<float> b(64);
        loki::math::NormalSampler::generate(rng_a, a, 0.0F, 1.0F);
        loki::math::NormalSampler::generate(rng_b, b, 0.0F, 1.0F);
        REQUIRE(a == b);
    }

    SECTION("Neighbouring keyed streams are independent") {
        // Keys that differ in one field by one are the riskiest case for
        // correlated streams. Fixed seed, so the bounds are deterministic.
        constexpr SizeType kStreams = 256;
        constexpr SizeType kDraws   = 4096;
        const auto draws            = [](auto make) {
            std::vector<std::vector<float>> out(kStreams,
                                                std::vector<float>(kDraws));
            for (SizeType k = 0; k < kStreams; ++k) {
                auto rng = make(static_cast<uint32_t>(k));
                loki::math::NormalSampler::generate(rng, out[k], 0.0F, 1.0F);
            }
            return out;
        };
        const auto corr = [](const std::vector<float>& a,
                             const std::vector<float>& b) {
            double sab = 0.0;
            double saa = 0.0;
            double sbb = 0.0;
            for (SizeType i = 0; i < a.size(); ++i) {
                sab += static_cast<double>(a[i]) * b[i];
                saa += static_cast<double>(a[i]) * a[i];
                sbb += static_cast<double>(b[i]) * b[i];
            }
            return sab / std::sqrt(saa * sbb);
        };
        const auto by_target = draws([](uint32_t k) {
            return loki::math::make_keyed_pcg32(42U, 5U, 3U, 7U, 0U, k);
        });
        const auto by_parent = draws([](uint32_t k) {
            return loki::math::make_keyed_pcg32(42U, 5U, 3U, k, 0U, 0U);
        });
        const auto by_branch = draws([](uint32_t k) {
            return loki::math::make_keyed_pcg32(42U, 5U, 3U, 7U, 1U, k);
        });
        // One correlation has sd 1/sqrt(4096) ~ 0.016; 0.08 is 5 sd.
        constexpr double kMaxCorr = 0.08;
        double sum                = 0.0;
        double sum2               = 0.0;
        for (SizeType k = 0; k < kStreams; ++k) {
            if (k + 1 < kStreams) {
                CHECK(std::abs(corr(by_target[k], by_target[k + 1])) <
                      kMaxCorr);
                CHECK(std::abs(corr(by_parent[k], by_parent[k + 1])) <
                      kMaxCorr);
            }
            // H0 vs H1 of the same item
            CHECK(std::abs(corr(by_target[k], by_branch[k])) < kMaxCorr);
            for (const float x : by_target[k]) {
                sum += x;
                sum2 += static_cast<double>(x) * x;
            }
        }
        // Pooled moments of 2^20 samples: sd(mean) ~ 0.001, sd(var) ~ 0.0014.
        const auto n     = static_cast<double>(kStreams * kDraws);
        const double mu  = sum / n;
        const double var = (sum2 / n) - (mu * mu);
        CHECK(std::abs(mu) < 0.005);
        CHECK(std::abs(var - 1.0) < 0.01);
    }

    SECTION("uniform_index stays within bounds") {
        loki::math::PCG32 rng(99U);
        for (int i = 0; i < 100; ++i) {
            const auto idx = loki::math::NormalSampler::uniform_index(rng, 7);
            REQUIRE(idx <= 7);
        }
    }
}
