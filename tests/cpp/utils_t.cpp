#include <catch2/catch_test_macros.hpp>

#include <loki/utils.hpp>
// Test the add_scalar function
TEST_CASE("add_scalar", "[scores]") {
  SECTION("Adding scalar") {
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    std::vector<float> out(x.size());
    SECTION("Positive scalar") {
      const float scalar_p = 1.0F;
      loki::add_scalar(std::span<const float>(x), scalar_p,
                       std::span<float>(out));
      std::vector<float> expected = {2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
      REQUIRE(out == expected);
    }
    SECTION("Negative scalar") {
      const float scalar_n = -1.0F;
      loki::add_scalar(std::span<const float>(x), scalar_n,
                       std::span<float>(out));
      std::vector<float> expected = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
      REQUIRE(out == expected);
    }
  }
}

// Test the diff_max function
TEST_CASE("diff_max", "[scores]") {
  SECTION("Positive/negative equal/unequal difference") {
    std::vector<float> x1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    std::vector<float> y1 = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F};
    float result =
        loki::diff_max(std::span<const float>(x1), std::span<const float>(y1));
    REQUIRE(result == 0.5F);

    std::vector<float> x2 = {1.0F, 4.0F, 3.0F, 2.0F, 10.0F};
    std::vector<float> y2 = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    result =
        loki::diff_max(std::span<const float>(x2), std::span<const float>(y2));
    REQUIRE(result == 9.5F);

    std::vector<float> x3 = {1.0F, 4.0F, 3.0F, 2.0F, 10.0F};
    std::vector<float> y3 = {20.0F, 20.0F, 20.0F, 20.0F, 20.0F};
    result =
        loki::diff_max(std::span<const float>(x3), std::span<const float>(y3));
    REQUIRE(result == -10.0F);
  }
}

// Test the circular_prefix_sum function
TEST_CASE("circular_prefix_sum", "[scores]") {
  SECTION("Positive/negative equal/unequal difference") {
    std::vector<float> x1 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    SECTION("No wrap around") {
      std::vector<float> out1(x1.size());
      loki::circular_prefix_sum(std::span<const float>(x1),
                                std::span<float>(out1));
      std::vector<float> expected1 = {1.0F, 3.0F, 6.0F, 10.0F, 15.0F};
      REQUIRE(out1 == expected1);
    }
    SECTION("Wrap around") {
      std::vector<float> out1(x1.size() + 5);
      loki::circular_prefix_sum(std::span<const float>(x1),
                                std::span<float>(out1));
      std::vector<float> expected1 = {1.0F,  3.0F,  6.0F,  10.0F, 15.0F,
                                      16.0F, 18.0F, 21.0F, 25.0F, 30.0F};
      REQUIRE(out1 == expected1);
    }
    SECTION("Length of out < length of x") {
      std::vector<float> out1(x1.size() - 2);
      loki::circular_prefix_sum(std::span<const float>(x1),
                                std::span<float>(out1));
      std::vector<float> expected1 = {1.0F, 3.0F, 6.0F};
      REQUIRE(out1 == expected1);
    }
  }
}