#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/detection/thresholds.hpp"

using Catch::Matchers::WithinRel;

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

TEST_CASE("evaluate_scheme and determine_scheme start from the initial state",
          "[thresholds]") {
    const std::vector<float> branching_pattern = {2.0F, 3.0F, 4.0F};
    const auto nstages                         = branching_pattern.size();
    const float tol =
        static_cast<float>(nstages) * std::numeric_limits<float>::epsilon();
    constexpr float kRefDucy    = 0.1F;
    constexpr SizeType kNbins   = 32;
    constexpr SizeType kNtrials = 1024;

    // Cumulative fields follow from the per-stage ones, from State::initial()
    const auto check_cumulative =
        [&](const std::vector<detection::State>& states) {
            REQUIRE(states.size() == nstages);
            float complexity       = 1.0F;
            float complexity_cumul = 1.0F;
            float success_h1_cumul = 1.0F;
            for (SizeType i = 0; i < nstages; ++i) {
                CAPTURE(i);
                REQUIRE_FALSE(states[i].is_empty);
                complexity_cumul += complexity * branching_pattern[i];
                complexity *= branching_pattern[i] * states[i].success_h0;
                success_h1_cumul *= states[i].success_h1;
                REQUIRE_THAT(states[i].complexity, WithinRel(complexity, tol));
                REQUIRE_THAT(states[i].complexity_cumul,
                             WithinRel(complexity_cumul, tol));
                REQUIRE_THAT(states[i].success_h1_cumul,
                             WithinRel(success_h1_cumul, tol));
            }
        };

    SECTION("evaluate_scheme, every trial survives") {
        const std::vector<float> thresholds(
            nstages, std::numeric_limits<float>::lowest());
        const auto states = detection::evaluate_scheme(
            thresholds, branching_pattern, kRefDucy, kNbins, kNtrials);
        check_cumulative(states);
        float nleaves = 1.0F;
        for (SizeType i = 0; i < nstages; ++i) {
            nleaves *= branching_pattern[i];
            REQUIRE(states[i].complexity == nleaves);
            REQUIRE(states[i].success_h1_cumul == 1.0F);
        }
    }
    SECTION("evaluate_scheme, pruning path") {
        const std::vector<float> thresholds = {1.0F, 2.0F, 3.0F};
        check_cumulative(detection::evaluate_scheme(
            thresholds, branching_pattern, kRefDucy, kNbins, kNtrials));
    }
    SECTION("determine_scheme") {
        const std::vector<float> survive_probs(nstages, 0.5F);
        check_cumulative(detection::determine_scheme(
            survive_probs, branching_pattern, kRefDucy, kNbins, kNtrials));
    }
    SECTION("stages after a path dies stay empty") {
        const std::vector<float> thresholds = {
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest()};
        const auto states = detection::evaluate_scheme(
            thresholds, branching_pattern, kRefDucy, kNbins, kNtrials);
        REQUIRE_FALSE(states[1].is_empty);
        REQUIRE(states[1].success_h1_cumul == 0.0F);
        REQUIRE(states[2].is_empty);
    }
}
} // namespace loki
