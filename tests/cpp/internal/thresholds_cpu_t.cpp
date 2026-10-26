#include <bit>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detection/thresholds_engine.hpp"

using loki::SizeType;
using loki::detection::State;
using loki::detection::detail::make_thresholds_cpu;
using loki::detection::detail::ThresholdsEngine;

namespace {

bool same_state_bits(const State& a, const State& b) {
    const auto f = [](float x) { return std::bit_cast<uint32_t>(x); };
    return f(a.success_h0) == f(b.success_h0) &&
           f(a.success_h1) == f(b.success_h1) &&
           f(a.complexity) == f(b.complexity) &&
           f(a.complexity_cumul) == f(b.complexity_cumul) &&
           f(a.success_h1_cumul) == f(b.success_h1_cumul) &&
           f(a.cost) == f(b.cost) && f(a.threshold) == f(b.threshold) &&
           a.is_empty == b.is_empty;
}

std::unique_ptr<ThresholdsEngine>
make_engine(const std::vector<float>& branching, int nthreads, bool simulate) {
    return make_thresholds_cpu(branching, 0.1F, 32, 64, 6, 0.05F, 8.0F, 24,
                               0.3F, 1.0F, 1.2F, 1, "legacy", nthreads,
                               /*seed=*/17, simulate);
}

} // namespace

TEST_CASE("Evaluation-only CPU threshold engine evaluates identically",
          "[thresholds]") {
    const std::vector<float> branching(8, 2.0F);
    auto full = make_engine(branching, 1, /*simulate=*/true);
    full->run(4);
    const auto path = full->get_best_path_thresholds(0.0F);
    REQUIRE(path.size() == branching.size());

    const auto eval_only = make_engine(branching, 3, /*simulate=*/false);
    const auto expected  = full->evaluate(path, 256, 3, false);
    const auto actual    = eval_only->evaluate(path, 256, 3, false);
    REQUIRE(expected.size() == actual.size());
    for (SizeType i = 0; i < expected.size(); ++i) {
        REQUIRE(same_state_bits(expected[i], actual[i]));
    }
    REQUIRE_FALSE(actual.back().is_empty);

    REQUIRE_THROWS_AS(eval_only->run(4), std::logic_error);
    REQUIRE_THROWS_AS(eval_only->get_states(), std::logic_error);
    REQUIRE_THROWS_AS(eval_only->get_best_path_thresholds(0.0F),
                      std::logic_error);
}
