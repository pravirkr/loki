#include <cstddef>

#include <catch2/catch_test_macros.hpp>

#include "loki/utils.hpp"

// Test the add_scalar function
TEST_CASE("add_scalar", "[utils]") {
    SECTION("Adding scalar") {
        std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> out(x.size());
        SECTION("Positive scalar") {
            const float scalar_p = 1.0F;
            loki::utils::add_scalar(std::span<const float>(x), scalar_p,
                                    std::span<float>(out));
            std::vector<float> expected = {2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
            REQUIRE(out == expected);
        }
        SECTION("Negative scalar") {
            const float scalar_n = -1.0F;
            loki::utils::add_scalar(std::span<const float>(x), scalar_n,
                                    std::span<float>(out));
            std::vector<float> expected = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
            REQUIRE(out == expected);
        }
    }
}

// Test the diff_max function
TEST_CASE("diff_max", "[utils]") {
    SECTION("Positive/negative equal/unequal difference") {
        std::vector<float> x1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> y1 = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F};
        float result          = loki::utils::diff_max(
            std::span<const float>(x1), std::span<const float>(y1));
        REQUIRE(result == 0.5F);

        std::vector<float> x2 = {1.0F, 4.0F, 3.0F, 2.0F, 10.0F};
        std::vector<float> y2 = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
        result                = loki::utils::diff_max(std::span<const float>(x2),
                                               std::span<const float>(y2));
        REQUIRE(result == 9.5F);

        std::vector<float> x3 = {1.0F, 4.0F, 3.0F, 2.0F, 10.0F};
        std::vector<float> y3 = {20.0F, 20.0F, 20.0F, 20.0F, 20.0F};
        result                = loki::utils::diff_max(std::span<const float>(x3),
                                               std::span<const float>(y3));
        REQUIRE(result == -10.0F);
    }
}

// Test the circular_prefix_sum function
TEST_CASE("circular_prefix_sum", "[utils]") {
    SECTION("Positive/negative equal/unequal difference") {
        std::vector<float> x1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        SECTION("No wrap around") {
            std::vector<float> out1(x1.size());
            loki::utils::circular_prefix_sum(std::span<const float>(x1),
                                             std::span<float>(out1));
            std::vector<float> expected1 = {1.0F, 3.0F, 6.0F, 10.0F, 15.0F};
            REQUIRE(out1 == expected1);
        }
        SECTION("Wrap around") {
            std::vector<float> out1(x1.size() + 5);
            loki::utils::circular_prefix_sum(std::span<const float>(x1),
                                             std::span<float>(out1));
            std::vector<float> expected1 = {1.0F,  3.0F,  6.0F,  10.0F, 15.0F,
                                            16.0F, 18.0F, 21.0F, 25.0F, 30.0F};
            REQUIRE(out1 == expected1);
        }
        SECTION("Length of out < length of x") {
            std::vector<float> out1(x1.size() - 2);
            loki::utils::circular_prefix_sum(std::span<const float>(x1),
                                             std::span<float>(out1));
            std::vector<float> expected1 = {1.0F, 3.0F, 6.0F};
            REQUIRE(out1 == expected1);
        }
    }
}

TEST_CASE("find_neighbouring_indices basic", "[find_neighbouring_indices]") {
    std::vector<size_t> indices = {0, 2, 4, 6, 8, 10};

    SECTION("Target in the middle") {
        auto result = loki::utils::find_neighbouring_indices(indices, 4, 3);
        REQUIRE(result == std::vector<size_t>{2, 4, 6});
    }

    SECTION("Target at the beginning") {
        auto result = loki::utils::find_neighbouring_indices(indices, 0, 3);
        REQUIRE(result == std::vector<size_t>{0, 2, 4});
    }

    SECTION("Target at the end") {
        auto result = loki::utils::find_neighbouring_indices(indices, 10, 3);
        REQUIRE(result == std::vector<size_t>{6, 8, 10});
    }

    SECTION("Target not in indices") {
        auto result = loki::utils::find_neighbouring_indices(indices, 5, 3);
        REQUIRE(result == std::vector<size_t>{4, 6, 8});
    }

    SECTION("Num larger than indices size") {
        auto result = loki::utils::find_neighbouring_indices(indices, 4, 10);
        REQUIRE(result == indices);
    }
}

TEST_CASE("find_neighbouring_indices edge cases",
          "[find_neighbouring_indices]") {
    SECTION("Empty indices") {
        std::vector<size_t> empty_indices;
        REQUIRE_THROWS_AS(loki::utils::find_neighbouring_indices(empty_indices, 0, 1),
                          std::invalid_argument);
    }

    SECTION("Num is zero") {
        std::vector<size_t> indices = {0, 2, 4, 6, 8};
        REQUIRE_THROWS_AS(loki::utils::find_neighbouring_indices(indices, 4, 0),
                          std::invalid_argument);
    }

    SECTION("Single element indices") {
        std::vector<size_t> single_index = {5};
        auto result = loki::utils::find_neighbouring_indices(single_index, 5, 3);
        REQUIRE(result == single_index);
    }
}
