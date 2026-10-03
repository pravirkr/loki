#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "loki/utils.hpp"

TEST_CASE("score_passes_threshold rejects NaN and sub-threshold scores",
          "[score]") {
    const float snr_min = 5.0F;
    REQUIRE_FALSE(loki::utils::score_passes_threshold(
        std::numeric_limits<float>::quiet_NaN(), snr_min));
    REQUIRE_FALSE(loki::utils::score_passes_threshold(4.9F, snr_min));
    REQUIRE(loki::utils::score_passes_threshold(5.0F, snr_min));
    REQUIRE(loki::utils::score_passes_threshold(6.0F, snr_min));
}
