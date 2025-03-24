#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <loki/math.hpp>

// Test the factorial function
TEST_CASE("factorial", "[math]") {
    SECTION("Factorial of positive integer") {
        REQUIRE(loki::math::factorial(5) == 120);
        REQUIRE(loki::math::factorial(10) == 3628800);
    }
    SECTION("Factorial of zero") { REQUIRE(loki::math::factorial(0) == 1); }
    SECTION("Factorial of one") { REQUIRE(loki::math::factorial(1) == 1); }
    SECTION("Factorial of negative integer") {
        REQUIRE_THROWS_AS(loki::math::factorial(-5), std::invalid_argument);
    }
}
