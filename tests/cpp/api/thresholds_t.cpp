#include "loki/detection/thresholds.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

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
    const detection::DynamicThresholdScheme dyn_scheme(
        branching_pattern, 0.5F, kNbins, kNtrials, kNprobs, 0.1F, 6.0F,
        kNthresholds, 0.3F, 1.0F, 0.7F, 0, "legacy", /*seed=*/std::nullopt,
        /*batch_size=*/256, loki::Exec::cpu(1));

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
    // Per-thread scratch must follow each run's nbins, not the first run's.
    // Fixed seed: whether a full path survives must not vary between runs.
    const std::vector<float> branching_pattern(7, 3.0F);
    const auto* mode = GENERATE("legacy", "improved");
    for (const SizeType nbins : {32U, 64U, 32U}) {
        CAPTURE(mode, nbins);
        detection::DynamicThresholdScheme dyn_scheme(
            branching_pattern, 0.1F, nbins, 1024, 10, 0.05F, 8.0F, 100, 0.3F,
            1.0F, 0.7F, 1, mode, /*seed=*/42, /*batch_size=*/256,
            loki::Exec::cpu(4));
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
            std::numeric_limits<float>::lowest(),
        };
        const auto states = detection::evaluate_scheme(
            thresholds, branching_pattern, kRefDucy, kNbins, kNtrials);
        REQUIRE_FALSE(states[1].is_empty);
        REQUIRE(states[1].success_h1_cumul == 0.0F);
        REQUIRE(states[2].is_empty);
    }
}
namespace {

bool same_state_bits(const detection::State& a, const detection::State& b) {
    const auto f = [](float x) { return std::bit_cast<uint32_t>(x); };
    return f(a.success_h0) == f(b.success_h0) &&
           f(a.success_h1) == f(b.success_h1) &&
           f(a.complexity) == f(b.complexity) &&
           f(a.complexity_cumul) == f(b.complexity_cumul) &&
           f(a.success_h1_cumul) == f(b.success_h1_cumul) &&
           f(a.cost) == f(b.cost) && f(a.threshold) == f(b.threshold) &&
           a.is_empty == b.is_empty;
}

} // namespace

TEST_CASE("DynamicThresholdScheme evaluate does not depend on thread count",
          "[thresholds]") {
    const std::vector<float> branching(8, 2.0F);
    const auto make = [&](int nthreads) {
        return detection::DynamicThresholdScheme(
            branching, 0.1F, 32, 64, 6, 0.05F, 8.0F, 24, 0.3F, 1.0F, 1.2F, 1,
            "improved", /*seed=*/17, /*batch_size=*/256,
            loki::Exec::cpu(nthreads));
    };
    auto one         = make(1);
    const auto eight = make(8);
    one.run(4);
    const auto path = one.get_best_path_thresholds();
    REQUIRE(path.size() == branching.size());
    const auto eval_a = one.evaluate(path, 256, 3);
    const auto eval_b = eight.evaluate(path, 256, 3);
    REQUIRE(eval_a.size() == eval_b.size());
    for (SizeType i = 0; i < eval_a.size(); ++i) {
        REQUIRE(same_state_bits(eval_a[i], eval_b[i]));
    }
    REQUIRE_FALSE(eval_a.front().is_empty);
    REQUIRE_FALSE(eval_a.back().is_empty);
}

TEST_CASE("DynamicThresholdScheme run is reproducible for a fixed seed",
          "[thresholds]") {
    const std::vector<float> branching = {4.0F, 9.0F, 1.0F, 2.5F, 4.0F, 3.0F};
    const auto* mode                   = GENERATE("legacy", "improved");
    CAPTURE(mode);
    const auto run_states = [&](uint64_t seed, int nthreads) {
        detection::DynamicThresholdScheme dyn(
            branching, 0.1F, 32, 128, 6, 0.05F, 8.0F, 24, 0.3F, 1.0F, 1.2F, 1,
            mode, seed, /*batch_size=*/256, loki::Exec::cpu(nthreads));
        dyn.run(4);
        return dyn.get_states();
    };
    const auto same = [](const std::vector<detection::State>& a,
                         const std::vector<detection::State>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (SizeType i = 0; i < a.size(); ++i) {
            if (!same_state_bits(a[i], b[i])) {
                return false;
            }
        }
        return true;
    };
    const auto ref = run_states(11, 1);
    // Same seed again in the same process (the threads are reused).
    REQUIRE(same(ref, run_states(11, 1)));
    // Any thread count: every work item draws from its own stream.
    REQUIRE(same(ref, run_states(11, 4)));
    REQUIRE(same(ref, run_states(11, 8)));
    // A different seed gives different draws.
    REQUIRE_FALSE(same(ref, run_states(12, 4)));
}

} // namespace loki
