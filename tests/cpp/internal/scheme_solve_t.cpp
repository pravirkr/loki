#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using loki::SizeType;
using loki::detection::State;
namespace detail = loki::detection::detail;

TEST_CASE("shared_probability_grid is anchored at min_pd and bounded",
          "[thresholds][solve]") {
    for (const float min_pd : {0.02F, 0.1F, 0.5F, 0.9F, 0.99F}) {
        CAPTURE(min_pd);
        const auto probs = detail::shared_probability_grid(min_pd, 40);
        // Increasing, inside (0, 1], ending at 1
        REQUIRE(std::ranges::is_sorted(probs));
        REQUIRE(probs.front() > 0.0F);
        REQUIRE_THAT(probs.back(), WithinAbs(1.0F, 1e-5F));
        // min_pd is a bin edge, with exactly 40 edges from it to 1 and at
        // most 16 below it
        const auto at    = std::ranges::lower_bound(probs, min_pd * 0.9999F);
        const auto below = static_cast<SizeType>(at - probs.begin());
        REQUIRE_THAT(*at, WithinAbs(min_pd, 1e-4F * min_pd));
        REQUIRE(probs.size() - below == 40);
        REQUIRE(below <= detail::kSharedMaxBelow);
        // The lowest edge is about half the target (the window's anchor)
        REQUIRE(probs.front() <= 0.5F * min_pd * 1.0001F);
        REQUIRE(probs.front() >= 0.25F * min_pd);
    }
    REQUIRE_THROWS(detail::shared_probability_grid(0.0F, 40));
    REQUIRE_THROWS(detail::shared_probability_grid(1.0F, 40));
    REQUIRE_THROWS(detail::shared_probability_grid(0.1F, 1));
}

TEST_CASE("shared_probability_grid keeps the full grid's edges",
          "[thresholds][solve]") {
    // Where no thinning is needed (min_pd 0.9: below <= 16), the edges are
    // those of the full log grid.
    const float min_pd = 0.9F;
    const auto probs   = detail::shared_probability_grid(min_pd, 40);
    const double step  = -std::log10(static_cast<double>(min_pd)) / 39.0;
    const auto below = static_cast<SizeType>(std::ceil(std::log10(2.0) / step));
    REQUIRE(below > detail::kSharedMaxBelow); // thinned: bounded
    REQUIRE(probs.size() == 40 + detail::kSharedMaxBelow);
    // Above min_pd every edge of the full grid is kept
    const auto full = detail::compute_probs(40 + below, probs.front());
    for (SizeType k = 0; k < 40; ++k) {
        REQUIRE_THAT(probs[detail::kSharedMaxBelow + k],
                     WithinRel(full[below + k], 1e-6F));
    }
}

TEST_CASE("terminal_rank and best_terminal_cell weigh the survivors",
          "[thresholds][solve]") {
    State a;
    a.is_empty         = false;
    a.complexity_cumul = 100.0F;
    a.complexity       = 50.0F;
    a.success_h1_cumul = 0.5F;
    a.cost             = a.complexity_cumul / a.success_h1_cumul;
    // w = 0: the stored cost; w = 1: (100 + 50) / 0.5
    REQUIRE(detail::terminal_rank(a, 0.0) == static_cast<double>(a.cost));
    REQUIRE_THAT(detail::terminal_rank(a, 1.0), WithinAbs(300.0, 1e-9));
    // Two terminal cells: cheap scoring with many survivors, or the reverse
    State b            = a;
    b.complexity_cumul = 140.0F;
    b.complexity       = 5.0F;
    b.cost             = b.complexity_cumul / b.success_h1_cumul;
    // 1 stage, 1 threshold, 2 prob bins
    const std::vector<State> states = {a, b};
    REQUIRE(detail::best_terminal_cell(states, 1, 1, 2, 0.1F, 0.0) == 0);
    REQUIRE(detail::best_terminal_cell(states, 1, 1, 2, 0.1F, 1.0) == 1);
    // Pd floor excludes both
    REQUIRE(detail::best_terminal_cell(states, 1, 1, 2, 0.9F, 1.0) ==
            std::numeric_limits<SizeType>::max());
}

namespace {

// n trials, M stages of N(0,1)-ish scores, stage-major
std::vector<float> scores_bank(SizeType nstages, SizeType n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(5.0F, 1.0F);
    std::vector<float> out(nstages * n);
    for (auto& x : out) {
        x = nd(gen);
    }
    return out;
}

SizeType survivors(const std::vector<float>& thr,
                   const std::vector<float>& scores,
                   SizeType n) {
    return detail::count_survivors(scores, thr, n);
}

} // namespace

TEST_CASE("relax_schedule lowers only as far as needed and never raises",
          "[thresholds][solve]") {
    constexpr SizeType kM = 6;
    constexpr SizeType kN = 4096;
    const auto scores     = scores_bank(kM, kN, 1);
    const detail::HostSignalBank bank(scores, kN);
    // A strict schedule: few survivors
    std::vector<float> thr(kM, 5.5F);
    const SizeType before = survivors(thr, scores, kN);
    const double target   = 0.2;
    REQUIRE(static_cast<double>(before) / kN < target);
    const float shift = detail::relax_schedule(thr, bank, target);
    REQUIRE(shift < 0.0F);
    const SizeType need = static_cast<SizeType>(std::ceil(target * kN));
    // At the float rounding boundary the count can be one trial short
    REQUIRE(survivors(thr, scores, kN) + 1 >= need);
    // Minimal: a slightly smaller shift (higher thresholds) misses it
    auto tighter = thr;
    for (auto& t : tighter) {
        t += 1e-3F;
    }
    REQUIRE(survivors(tighter, scores, kN) < need);
    // Already enough survivors: unchanged
    std::vector<float> loose(kM, 2.0F);
    const auto copy = loose;
    REQUIRE(detail::relax_schedule(loose, bank, 0.2) == 0.0F);
    REQUIRE(loose == copy);
}

TEST_CASE("tighten_schedule raises thresholds without losing a survivor",
          "[thresholds][solve]") {
    constexpr SizeType kM = 6;
    constexpr SizeType kN = 4096;
    const auto scores     = scores_bank(kM, kN, 2);
    const detail::HostSignalBank bank(scores, kN);
    std::vector<float> thr(kM, 3.0F);
    const SizeType before = survivors(thr, scores, kN);
    REQUIRE(detail::count_survivors(bank, thr) == before);
    const auto [raised, count] = detail::tighten_schedule(thr, bank);
    REQUIRE(count == before);
    REQUIRE(raised > 0);
    // Not one survivor lost, and every stage's threshold is as high as it can
    // be: one float step up would lose a trial (or the stage is unchanged)
    REQUIRE(survivors(thr, scores, kN) == before);
    for (SizeType s = 0; s < kM; ++s) {
        CAPTURE(s);
        auto up = thr;
        up[s]   = std::nextafter(up[s], std::numeric_limits<float>::max());
        REQUIRE(survivors(up, scores, kN) < before);
    }
    // No survivor at all: a no-op, not a threshold at the float maximum
    std::vector<float> none(kM, 100.0F);
    const auto untouched = none;
    const auto result    = detail::tighten_schedule(none, bank);
    REQUIRE(result.second == 0);
    REQUIRE(none == untouched);
}

TEST_CASE("window_scheme order statistics equal a full sort",
          "[thresholds][solve]") {
    constexpr SizeType kN = 2048;
    constexpr SizeType kM = 5;
    const auto scores     = scores_bank(kM, kN, 3);
    for (const float prob_min : {0.01F, 0.05F, 0.5F, 0.99F}) {
        CAPTURE(prob_min);
        const auto beam = detail::window_scheme(scores, kN, kM, prob_min);
        for (SizeType s = 0; s < kM; ++s) {
            std::vector<float> row(
                scores.begin() + static_cast<std::ptrdiff_t>(s * kN),
                scores.begin() + static_cast<std::ptrdiff_t>((s + 1) * kN));
            std::ranges::sort(row);
            const auto rank = [&](double q) {
                return std::min(static_cast<SizeType>(std::floor(q * kN)),
                                kN - 1);
            };
            const double q_lo = static_cast<double>(0.02F * prob_min) / kM;
            const double q_hi = 1.0 - (0.5 * static_cast<double>(prob_min));
            REQUIRE(beam.lo[s] == row[rank(q_lo)]);
            REQUIRE(beam.hi[s] == row[rank(q_hi)]);
        }
    }
}

TEST_CASE("resample_zscores carries z-scores to another stage count",
          "[thresholds][descent]") {
    const std::vector<float> z = {2.0F, 0.0F, 1.0F, 3.0F};
    // The same count: unchanged
    REQUIRE(detail::resample_zscores(z, 4) == z);
    // Linear in the fractional stage position, the ends kept
    const auto longer = detail::resample_zscores(z, 7);
    REQUIRE(longer.size() == 7);
    const std::vector<float> expected = {2.0F, 1.0F, 0.0F, 0.5F,
                                         1.0F, 2.0F, 3.0F};
    for (SizeType s = 0; s < expected.size(); ++s) {
        CAPTURE(s);
        REQUIRE_THAT(longer[s], WithinAbs(expected[s], 1e-6));
    }
    const auto shorter = detail::resample_zscores(longer, 4);
    REQUIRE(shorter.size() == 4);
    for (SizeType s = 0; s < z.size(); ++s) {
        CAPTURE(s);
        REQUIRE_THAT(shorter[s], WithinAbs(z[s], 1e-6));
    }
    const auto two = detail::resample_zscores(z, 2);
    REQUIRE(two == std::vector<float>{2.0F, 3.0F});
    // Nothing to carry over: no start
    REQUIRE(detail::resample_zscores({}, 5).empty());
    REQUIRE(detail::resample_zscores(std::vector<float>{1.0F}, 5).empty());
    REQUIRE(detail::resample_zscores(z, 1).empty());
}
