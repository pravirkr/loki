#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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

TEST_CASE("DynamicThresholdScheme runs back to back with different nbins",
          "[thresholds]") {
    // Per-thread scratch must follow each run's nbins, not the first run's
    const std::vector<float> branching_pattern(7, 3.0F);
    const auto* mode = GENERATE("legacy", "improved");
    for (const SizeType nbins : {32U, 64U, 32U}) {
        CAPTURE(mode, nbins);
        detection::DynamicThresholdScheme dyn_scheme(
            branching_pattern, 0.1F, nbins, 1024, 10, 0.05F, 8.0F, 100, 0.3F,
            1.0F, 0.7F, 1, mode, 4);
        dyn_scheme.run();
        REQUIRE(dyn_scheme.get_best_path_thresholds().size() ==
                branching_pattern.size());
    }
}
} // namespace loki
