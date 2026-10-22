#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

// The "window" beam (guess="window"): every threshold a path on the
// probability grid can use, every transition, and two passes ranked by cost
// to go. The issue #15 cases below are the 4-32 segment Taylor patterns
// where the analytic beam reaches only low Pd (or none at all).

namespace loki {
namespace {

using detection::DynamicThresholdScheme;
using detection::State;

bool skip_backend(std::string_view backend) {
    return backend == "cuda" && !is_available(Backend::kCUDA);
}

Exec exec_of(std::string_view backend) {
    return backend == "cuda" ? Exec::cuda() : Exec::cpu(4);
}

// FFAPlanTime(...).get_branching_pattern("taylor") for nsamps 2^23, tsamp
// 8.192e-5 s, nbins 32, eta 1, param_limits [[-10, 10], [133.333, 200]],
// bseg_ffa nsamps / nseg (bench/scripts/schemes/_branching.py).
std::vector<float> issue15_pattern(int nseg) {
    switch (nseg) {
    case 4:
        return {3.143F, 4.818F, 2.217F};
    case 8:
        return {1.0F, 3.0F, 3.143F, 1.0F, 1.606F, 5.604F, 1.0F};
    case 16:
        return {1.0F, 3.0F, 1.0F,   1.0F,   1.0F,   6.075F, 1.482F, 1.0F,
                1.0F, 1.0F, 1.921F, 1.391F, 1.123F, 1.0F,   1.0F};
    default:
        return {1.0F,   3.0F,   1.0F,   1.0F,  1.0F,   3.0F,   1.0F,   1.0F,
                1.0F,   1.0F,   1.0F,   1.23F, 1.646F, 1.317F, 1.125F, 1.0F,
                1.0F,   3.0F,   1.0F,   1.0F,  1.0F,   1.47F,  1.307F, 1.207F,
                1.152F, 1.118F, 1.004F, 1.0F,  1.0F,   1.0F,   1.0F};
    }
}

// The EP planner's probability grid: nprobs bins from min_pd to 1,
// extended below min_pd to min_pd / 2 with the same step.
std::pair<float, SizeType> anchored_grid(float min_pd, SizeType nprobs) {
    const double step = -std::log10(static_cast<double>(min_pd)) /
                        static_cast<double>(nprobs - 1);
    const auto below = static_cast<SizeType>(std::ceil(std::log10(2.0) / step));
    const double p_min = static_cast<double>(min_pd) *
                         std::pow(10.0, -static_cast<double>(below) * step);
    return {static_cast<float>(p_min), nprobs + below};
}

DynamicThresholdScheme make_issue15(std::string_view backend,
                                    std::string_view mode,
                                    std::string_view guess,
                                    int nseg,
                                    float snr,
                                    float min_pd,
                                    uint64_t seed) {
    const auto [prob_min, nprobs] = anchored_grid(min_pd, 40);
    return {issue15_pattern(nseg),
            /*ref_ducy=*/0.1F,
            /*nbins=*/32,
            /*ntrials=*/1024,
            nprobs,
            prob_min,
            snr,
            /*nthresholds=*/100,
            /*ducy_max=*/0.2F,
            /*wtsp=*/1.5F,
            /*beam_width=*/0.7F,
            /*trials_start=*/1,
            mode,
            seed,
            /*batch_size=*/256,
            guess,
            exec_of(backend)};
}

struct Rescored {
    float pd{0.0F};
    float cost{0.0F};
};

// The selected path, rescored on fresh trials; pd 0 when there is none.
Rescored select_and_rescore(DynamicThresholdScheme& scheme, float min_pd) {
    scheme.run(10);
    const auto path = scheme.get_best_path_thresholds(min_pd);
    if (path.empty()) {
        return {};
    }
    const auto ev = scheme.evaluate(path, SizeType{1} << 15U, /*seed=*/4242);
    return {ev.back().success_h1_cumul, ev.back().cost};
}

bool same_bits(const std::vector<State>& a, const std::vector<State>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    const auto f = [](float x) { return std::bit_cast<uint32_t>(x); };
    for (SizeType i = 0; i < a.size(); ++i) {
        if (a[i].is_empty != b[i].is_empty ||
            f(a[i].complexity_cumul) != f(b[i].complexity_cumul) ||
            f(a[i].success_h1_cumul) != f(b[i].success_h1_cumul) ||
            f(a[i].threshold) != f(b[i].threshold)) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("Window beam resolves issue #15 on CUDA: every target reached, "
          "cheaper than the analytic beam",
          "[thresholds][window][cuda]") {
    if (skip_backend("cuda")) {
        SKIP("needs a CUDA build");
    }
    const int nseg     = GENERATE(4, 8, 16, 32);
    const float snr    = GENERATE(6.0F, 8.1F, 10.0F);
    const float min_pd = GENERATE(0.1F, 0.5F, 0.9F);
    CAPTURE(nseg, snr, min_pd);
    auto window =
        make_issue15("cuda", "shared", "window", nseg, snr, min_pd, 7);
    const auto w = select_and_rescore(window, min_pd);
    // Delivered on fresh trials: within 10 % of min_pd or above it
    REQUIRE(w.pd >= 0.9F * min_pd);
    auto analytic =
        make_issue15("cuda", "shared", "analytic", nseg, snr, min_pd, 7);
    const auto a = select_and_rescore(analytic, min_pd);
    if (a.pd > 0.0F) {
        // At most the analytic path's cost per detection (5 % noise margin)
        CHECK(w.cost <= 1.05F * a.cost);
    }
}

TEST_CASE("Window beam resolves issue #15 on the CPU (calibrated)",
          "[thresholds][window]") {
    const int nseg     = GENERATE(4, 8);
    const float min_pd = GENERATE(0.5F, 0.9F);
    CAPTURE(nseg, min_pd);
    auto window =
        make_issue15("cpu", "improved", "window", nseg, 8.1F, min_pd, 7);
    window.run(10);
    REQUIRE_FALSE(window.get_best_path_thresholds(0.5F * min_pd).empty());
    // The CPU planner calibrates (improved mode's in-run Pd is optimistic)
    detection::CalibrationOptions opts;
    opts.seed      = 99;
    const auto cal = window.calibrate(min_pd, opts);
    REQUIRE(cal.thresholds.size() == issue15_pattern(nseg).size());
    REQUIRE(cal.pd_check >= 0.9F * min_pd);
}

TEST_CASE("Window beam: reproducible, and run() can be called again",
          "[thresholds][window]") {
    // Shared mode is CUDA-only
    const auto [backend, mode] = GENERATE(
        std::pair<std::string_view, std::string_view>{"cpu", "improved"},
        std::pair<std::string_view, std::string_view>{"cuda", "improved"},
        std::pair<std::string_view, std::string_view>{"cuda", "shared"});
    CAPTURE(backend, mode);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    auto a = make_issue15(backend, mode, "window", 8, 8.1F, 0.5F, 11);
    auto b = make_issue15(backend, mode, "window", 8, 8.1F, 0.5F, 11);
    a.run(10);
    b.run(10);
    REQUIRE(same_bits(a.get_states(), b.get_states()));
    // A second run() starts from a fresh grid: the same states again
    const auto first = a.get_states();
    a.run(10);
    REQUIRE(same_bits(a.get_states(), first));
    // thres_neigh is ignored by the window beam
    b.run(1);
    REQUIRE(same_bits(b.get_states(), first));
}

TEST_CASE("Window beam spans the signal quantiles at every stage",
          "[thresholds][window]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    constexpr float kMinPd = 0.1F;
    auto scheme =
        make_issue15(backend, "improved", "window", 16, 8.1F, kMinPd, 3);
    const auto thresholds = scheme.get_thresholds();
    const auto prob_min   = scheme.get_probs().front();
    constexpr SizeType kN = 8192;
    const auto h1         = scheme.simulate_h1_scores(kN, /*seed=*/5);
    for (SizeType s = 0; s < scheme.get_nstages(); ++s) {
        CAPTURE(s);
        const auto beam = scheme.get_current_thresholds_idx(s);
        REQUIRE(beam.size() > 2);
        const auto row        = std::span(h1).subspan(s * kN, kN);
        const auto frac_above = [&](float t) {
            return static_cast<float>(std::ranges::count_if(
                       row, [t](float x) { return x > t; })) /
                   static_cast<float>(kN);
        };
        // Top: keeps about prob_min / 2 of the signal (sampling slack)
        CHECK(frac_above(thresholds[beam.back()]) >= 0.25F * prob_min);
        // Bottom: keeps nearly all of it (lower edge eps prob_min / nstages,
        // or the grid's first threshold)
        CHECK(frac_above(thresholds[beam.front()]) >= 0.995F);
    }
}

} // namespace loki
