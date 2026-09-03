#include <catch2/catch_test_macros.hpp>

#include "loki/detection/thresholds.hpp"

namespace loki {
TEST_CASE("DynamicThresholdScheme construction rejects invalid input",
          "[thresholds]") {
    REQUIRE_THROWS_AS(
        detection::DynamicThresholdScheme(std::span<const float>{}, 0.5F),
        std::invalid_argument);
}

TEST_CASE("DynamicThresholdScheme getters", "[thresholds]") {
    const std::vector<float> branching_pattern = {0.5F, 0.5F, 0.5F};
    constexpr SizeType kNbins                  = 16;
    constexpr SizeType kNtrials                = 64;
    constexpr SizeType kNprobs                 = 4;
    constexpr SizeType kNthresholds            = 20;
    detection::DynamicThresholdScheme dyn_scheme(
        branching_pattern, 0.5F, kNbins, kNtrials, kNprobs, 0.1F, 6.0F,
        kNthresholds, 0.3F, 1.0F, 0.7F, 0, "legacy", 1);

    REQUIRE(dyn_scheme.get_branching_pattern() == branching_pattern);
    REQUIRE(dyn_scheme.get_profile().size() == kNbins);
    REQUIRE(dyn_scheme.get_thresholds().size() == kNthresholds);
    REQUIRE(dyn_scheme.get_probs().size() == kNprobs);
    REQUIRE(dyn_scheme.get_nstages() == branching_pattern.size());
    REQUIRE(dyn_scheme.get_nthresholds() == kNthresholds);
    REQUIRE(dyn_scheme.get_nprobs() == kNprobs);
    REQUIRE(dyn_scheme.get_best_path_thresholds().empty());
    REQUIRE_FALSE(dyn_scheme.get_states().empty());
}
} // namespace loki
