#include <cmath>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/math.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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
        REQUIRE(std::isfinite(loki::math::StatTables::exact_norm_isf(1.0F)));
    }

    SECTION("chi_sq_minus_logsf is finite for valid input") {
        REQUIRE(std::isfinite(
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

TEST_CASE("ThreadLocalNormalRNG", "[math]") {
    SECTION("Produces finite samples with fixed seed") {
        loki::math::ThreadLocalNormalRNG rng(12345U);
        std::vector<float> samples(128);
        rng.generate(samples, 2.0F, 0.5F);
        for (float sample : samples) {
            REQUIRE(std::isfinite(sample));
        }
        const float mean =
            std::accumulate(samples.begin(), samples.end(), 0.0F) /
            static_cast<float>(samples.size());
        REQUIRE_THAT(mean, WithinAbs(2.0F, 0.5F));
    }

    SECTION("uniform_index stays within bounds") {
        loki::math::ThreadLocalNormalRNG rng(99U);
        for (int i = 0; i < 100; ++i) {
            const auto idx = rng.uniform_index(7);
            REQUIRE(idx <= 7);
        }
    }
}
