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
    // A strict schedule: few survivors
    std::vector<float> thr(kM, 5.5F);
    const SizeType before = survivors(thr, scores, kN);
    const double target   = 0.2;
    REQUIRE(static_cast<double>(before) / kN < target);
    const float shift = detail::relax_schedule(thr, scores, kN, target);
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
    REQUIRE(detail::relax_schedule(loose, scores, kN, 0.2) == 0.0F);
    REQUIRE(loose == copy);
}

TEST_CASE("tighten_schedule raises thresholds without losing a survivor",
          "[thresholds][solve]") {
    constexpr SizeType kM = 6;
    constexpr SizeType kN = 4096;
    const auto scores     = scores_bank(kM, kN, 2);
    std::vector<float> thr(kM, 3.0F);
    const SizeType before      = survivors(thr, scores, kN);
    const auto [raised, count] = detail::tighten_schedule(thr, scores, kN);
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
    const auto result    = detail::tighten_schedule(none, scores, kN);
    REQUIRE(result.second == 0);
    REQUIRE(none == untouched);
}

TEST_CASE("refinement_blocks cuts the stages into equal log-branching",
          "[thresholds][solve]") {
    // Uniform branching: four equal blocks
    const std::vector<float> flat(16, 2.0F);
    const auto cuts = detail::refinement_blocks(flat);
    REQUIRE(cuts.size() == 5);
    REQUIRE(cuts.front() == 0);
    REQUIRE(cuts.back() == 16);
    for (SizeType k = 1; k < 4; ++k) {
        // a cut lands within one stage of the exact quarter
        REQUIRE(cuts[k] + 1 >= 4 * k);
        REQUIRE(cuts[k] <= 4 * k);
    }
    // Everything in the first stage: the other cuts collapse
    std::vector<float> front(8, 1.0F);
    front[0]      = 1000.0F;
    const auto fc = detail::refinement_blocks(front);
    REQUIRE(fc.front() == 0);
    REQUIRE(fc.back() == 8);
    REQUIRE(std::ranges::is_sorted(fc));
    REQUIRE(std::ranges::adjacent_find(fc) == fc.end());
    // Fewer stages than blocks
    const auto two = detail::refinement_blocks(std::vector<float>{2.0F, 2.0F});
    REQUIRE(two.front() == 0);
    REQUIRE(two.back() == 2);
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

namespace {

using loki::detection::detail::make_thresholds_engine;

std::unique_ptr<loki::detection::detail::ThresholdsEngine>
shared_engine(const std::vector<float>& bp, SizeType nbins, float min_pd) {
    return make_thresholds_engine(bp, 0.1F, nbins, 1024, 40, 0.05F, 8.0F, 100,
                                  0.2F, 1.5F, 0.7F, 1, "shared", 5, 256,
                                  "window", min_pd, 1.0F, loki::Exec::cpu(4));
}

double cost_per_detection(const loki::detection::detail::ThresholdsEngine& eng,
                          const std::vector<float>& thresholds) {
    const auto states = eng.evaluate(thresholds, 1U << 16U, 99, false);
    const auto& last  = states.back();
    return (static_cast<double>(last.complexity_cumul) +
            static_cast<double>(last.complexity)) /
           static_cast<double>(last.success_h1_cumul);
}

} // namespace

TEST_CASE("refit_scheme re-fits a small-nbins schedule to the true nbins",
          "[thresholds][solve]") {
    // The 8-segment pattern of issue #15. A schedule searched at 64 bins is too
    // permissive at 256: the refit must deliver the target there and cost
    // clearly less than the unshifted schedule.
    const std::vector<float> bp = {1.0F,   3.0F,   3.143F, 1.0F,
                                   1.606F, 5.604F, 1.0F};
    constexpr float kMinPd      = 0.3F;
    auto proxy                  = shared_engine(bp, 64, kMinPd);
    proxy->run(10);
    const auto searched = detail::solve_scheme(
        *proxy, kMinPd, 1.0, loki::detection::RefineLevel::kNone);
    REQUIRE_FALSE(searched.thresholds.empty());
    proxy.reset();

    const auto evaluator = shared_engine(bp, 256, kMinPd);
    const detail::SolveBudget budget{.nbank = 1U << 15U, .nnoise = 1U << 13U};
    const auto refit =
        detail::refit_scheme(*evaluator, searched, kMinPd, 1.0, 5, budget);
    REQUIRE(refit.thresholds.size() == bp.size());
    REQUIRE(refit.pd_trials == budget.nbank);
    REQUIRE(refit.pd >= 0.9 * kMinPd);
    // The shift raised the thresholds
    for (SizeType s = 0; s < bp.size(); ++s) {
        REQUIRE(refit.thresholds[s] >= searched.thresholds[s] - 0.5F);
    }
    const double unshifted =
        cost_per_detection(*evaluator, searched.thresholds);
    const double refitted = cost_per_detection(*evaluator, refit.thresholds);
    CAPTURE(unshifted, refitted);
    REQUIRE(refitted < unshifted);
    // Close to a native search at the true nbins (the gate is loose at large
    // nbins)
    auto native = shared_engine(bp, 256, kMinPd);
    native->run(10);
    const auto sol = detail::solve_scheme(*native, kMinPd, 1.0,
                                          loki::detection::RefineLevel::kNone);
    const double native_cost = cost_per_detection(*native, sol.thresholds);
    CAPTURE(native_cost);
    REQUIRE(refitted < 1.5 * native_cost);
}
