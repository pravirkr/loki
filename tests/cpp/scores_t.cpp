#include <catch2/catch_test_macros.hpp>

#include "loki/scores.hpp"

// Test the add_scalar function
TEST_CASE("add_scalar", "[scores]") {
    SECTION("Adding scalar") {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> out(x.size());
        SECTION("Positive scalar") {
            const float scalar_p = 1.0f;
            loki::add_scalar(std::span<const float>(x), scalar_p,
                             std::span<float>(out));
            std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            REQUIRE(out == expected);
        }
        SECTION("Negative scalar") {
            const float scalar_n = -1.0f;
            loki::add_scalar(std::span<const float>(x), scalar_n,
                             std::span<float>(out));
            std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
            REQUIRE(out == expected);
        }
    }
}

// Test the diff_max function
TEST_CASE("diff_max", "[scores]") {
    SECTION("Positive/negative equal/unequal difference") {
        std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> y1 = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
        float result          = loki::diff_max(std::span<const float>(x1),
                                               std::span<const float>(y1));
        REQUIRE(result == 0.5f);

        std::vector<float> x2 = {1.0f, 4.0f, 3.0f, 2.0f, 10.0f};
        std::vector<float> y2 = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        result                = loki::diff_max(std::span<const float>(x2),
                                               std::span<const float>(y2));
        REQUIRE(result == 9.5f);

        std::vector<float> x3 = {1.0f, 4.0f, 3.0f, 2.0f, 10.0f};
        std::vector<float> y3 = {20.0f, 20.0f, 20.0f, 20.0f, 20.0f};
        result                = loki::diff_max(std::span<const float>(x3),
                                               std::span<const float>(y3));
        REQUIRE(result == -10.0f);
    }
}

// Test the circular_prefix_sum function
TEST_CASE("circular_prefix_sum", "[scores]") {
    SECTION("Positive/negative equal/unequal difference") {
        std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        SECTION("No wrap around") {
            std::vector<float> out1(x1.size());
            loki::circular_prefix_sum(std::span<const float>(x1),
                                      std::span<float>(out1));
            std::vector<float> expected1 = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
            REQUIRE(out1 == expected1);
        }
        SECTION("Wrap around") {
            std::vector<float> out1(x1.size() + 5);
            loki::circular_prefix_sum(std::span<const float>(x1),
                                      std::span<float>(out1));
            std::vector<float> expected1 = {1.0f,  3.0f,  6.0f,  10.0f, 15.0f,
                                            16.0f, 18.0f, 21.0f, 25.0f, 30.0f};
            REQUIRE(out1 == expected1);
        }
        SECTION("Length of out < length of x") {
            std::vector<float> out1(x1.size() - 2);
            loki::circular_prefix_sum(std::span<const float>(x1),
                                      std::span<float>(out1));
            std::vector<float> expected1 = {1.0f, 3.0f, 6.0f};
            REQUIRE(out1 == expected1);
        }
    }
}