#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detection/scheme.hpp"

using Catch::Matchers::WithinRel;
using loki::SizeType;
using loki::detection::State;
namespace detail = loki::detection::detail;

TEST_CASE("beam_guess_from_string", "[thresholds]") {
    REQUIRE(detail::beam_guess_from_string("analytic") ==
            detail::BeamGuess::kAnalytic);
    REQUIRE(detail::beam_guess_from_string("pilot") ==
            detail::BeamGuess::kPilot);
    REQUIRE(detail::beam_guess_from_string("window") ==
            detail::BeamGuess::kWindow);
    REQUIRE_THROWS_AS(detail::beam_guess_from_string("median"),
                      std::invalid_argument);
}

TEST_CASE("window_scheme takes order statistics of each stage",
          "[thresholds]") {
    // Two stages of 1000 scores: 0..999 and 1000..1999 (unsorted)
    constexpr SizeType kN = 1000;
    std::vector<float> scores(2 * kN);
    for (SizeType i = 0; i < kN; ++i) {
        scores[i]      = static_cast<float>((i * 7) % kN);
        scores[kN + i] = static_cast<float>(kN + ((i * 13) % kN));
    }
    // prob_min 0.25: upper edge Q(1 - 0.125) = rank 875; lower edge
    // Q(eps prob_min / nstages) = Q(0.02 * 0.25 / 2) = rank 2
    const auto beam = detail::window_scheme(scores, kN, 2, 0.25F, 0.02F);
    REQUIRE(beam.lo == std::vector<float>{2.0F, 1002.0F});
    REQUIRE(beam.hi == std::vector<float>{875.0F, 1875.0F});
    // Quantiles beyond the sample clamp to its extremes
    const auto wide = detail::window_scheme(scores, kN, 2, 1e-6F, 0.02F);
    REQUIRE(wide.lo == std::vector<float>{0.0F, 1000.0F});
    REQUIRE(wide.hi == std::vector<float>{999.0F, 1999.0F});
}

TEST_CASE("beam_indices and centred_beam", "[thresholds]") {
    const std::vector<float> grid = {0.1F, 0.5F, 1.0F, 1.5F, 2.0F};
    REQUIRE(detail::beam_indices(grid, 0.5F, 1.5F) ==
            std::vector<SizeType>{1, 2, 3});
    REQUIRE(detail::beam_indices(grid, 2.5F, 3.0F).empty());
    const std::vector<float> guess = {0.2F, 1.9F};
    const auto beam                = detail::centred_beam(guess, 0.5F, 2.0F);
    REQUIRE(beam.lo == std::vector<float>{0.0F, 1.4F});
    REQUIRE(beam.hi == std::vector<float>{0.7F, 2.0F});
}

namespace {

// A path of nstages cells on a 1 x 1 grid (one threshold, one bin).
std::vector<State> chain(const std::vector<float>& complexity,
                         const std::vector<float>& complexity_cumul) {
    std::vector<State> states(complexity.size());
    for (SizeType s = 0; s < states.size(); ++s) {
        auto& st                 = states[s];
        st.is_empty              = false;
        st.threshold             = 1.0F;
        st.threshold_prev        = 1.0F;
        st.success_h1_cumul      = 0.2F;
        st.success_h1_cumul_prev = 0.2F;
        st.complexity            = complexity[s];
        st.complexity_cumul      = complexity_cumul[s];
        st.cost                  = complexity_cumul[s] / 0.2F;
    }
    return states;
}

} // namespace

TEST_CASE("cost_to_go_scheme: future complexity per survivor", "[thresholds]") {
    const std::vector<float> thresholds = {1.0F};
    const std::vector<float> probs      = {0.1F};
    // C = 3, 7, 11 and n = 2, 4, 1: h = (11 - 3) / 2, (11 - 7) / 4, 0
    const auto states = chain({2.0F, 4.0F, 1.0F}, {3.0F, 7.0F, 11.0F});
    SECTION("unsmoothed") {
        const auto h =
            detail::cost_to_go_scheme(states, thresholds, probs, 3, 1, 1,
                                      /*pd_max_factor=*/8.0F,
                                      /*half_window=*/0);
        REQUIRE(h.size() == 3);
        REQUIRE_THAT(h[0], WithinRel(4.0F, 1e-5F));
        REQUIRE_THAT(h[1], WithinRel(1.0F, 1e-5F));
        REQUIRE(h[2] == 0.0F);
    }
    SECTION("smoothed in log space, the last stage left out") {
        const auto h = detail::cost_to_go_scheme(states, thresholds, probs, 3,
                                                 1, 1, 8.0F, 4);
        REQUIRE_THAT(h[0], WithinRel(2.0F, 1e-5F)); // sqrt(4 * 1)
        REQUIRE_THAT(h[1], WithinRel(2.0F, 1e-5F));
        REQUIRE(h[2] == 0.0F);
    }
    SECTION("no terminal cell below pd_max_factor * probs[0]") {
        REQUIRE(detail::cost_to_go_scheme(states, thresholds, probs, 3, 1, 1,
                                          /*pd_max_factor=*/1.0F, 0)
                    .empty());
    }
}
