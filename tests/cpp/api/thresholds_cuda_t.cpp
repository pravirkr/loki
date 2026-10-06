#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/detection/thresholds.hpp"

namespace loki {
namespace {

using detection::DynamicThresholdScheme;
using detection::State;

// First 24 stages of a production-like branching pattern.
const std::vector<float> kBranching = {
    4.0F, 9.0F,        1.0F,        2.25575101F, 3.98980204F,
    3.0F, 2.80514208F, 3.20839363F, 1.0F,        1.0F,
    3.0F, 1.0F,        1.0F,        2.25575101F, 1.99490102F,
    3.0F, 1.0F,        3.0F,        1.0F,        3.0F,
    1.0F, 1.0F,        2.80514208F, 1.06946454F};

constexpr SizeType kNtrials     = 256;
constexpr SizeType kNprobs      = 12;
constexpr SizeType kNthresholds = 60;
constexpr SizeType kThresNeigh  = 6;

DynamicThresholdScheme
make_scheme(std::string_view mode,
            std::optional<uint64_t> seed,
            SizeType nbins                   = 64,
            SizeType batch_size              = 256,
            float beam_width                 = 1.5F,
            std::span<const float> branching = kBranching) {
    return DynamicThresholdScheme(
        branching, 0.1F, nbins, kNtrials, kNprobs, 0.05F, 8.0F, kNthresholds,
        0.3F, 1.2F, beam_width, 1, mode, seed, batch_size, Exec::cuda());
}

// Field-wise bitwise equality (ignores the struct's padding bytes).
bool same_bits(const State& a, const State& b) {
    const auto f = [](float x) { return std::bit_cast<uint32_t>(x); };
    return f(a.success_h0) == f(b.success_h0) &&
           f(a.success_h1) == f(b.success_h1) &&
           f(a.complexity) == f(b.complexity) &&
           f(a.complexity_cumul) == f(b.complexity_cumul) &&
           f(a.success_h1_cumul) == f(b.success_h1_cumul) &&
           f(a.nbranches) == f(b.nbranches) &&
           f(a.threshold) == f(b.threshold) && f(a.cost) == f(b.cost) &&
           f(a.threshold_prev) == f(b.threshold_prev) &&
           f(a.success_h1_cumul_prev) == f(b.success_h1_cumul_prev) &&
           a.is_empty == b.is_empty;
}

bool same_bits(const std::vector<State>& a, const std::vector<State>& b) {
    return a.size() == b.size() &&
           std::equal(
               a.begin(), a.end(), b.begin(),
               [](const State& x, const State& y) { return same_bits(x, y); });
}

SizeType count_nonempty(const std::vector<State>& states) {
    return static_cast<SizeType>(std::ranges::count_if(
        states, [](const State& s) { return !s.is_empty; }));
}

} // namespace

TEST_CASE("CUDA DynamicThresholdScheme is bit-reproducible for a fixed seed",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto* mode = GENERATE("legacy", "improved");
    CAPTURE(mode);
    auto scheme_a = make_scheme(mode, 1234);
    scheme_a.run(kThresNeigh);
    const auto states_a = scheme_a.get_states();
    REQUIRE(count_nonempty(states_a) > 0);

    // Fresh object, same seed.
    auto scheme_b = make_scheme(mode, 1234);
    scheme_b.run(kThresNeigh);
    REQUIRE(same_bits(states_a, scheme_b.get_states()));

    // Same object, second run: no state leaks from the first run.
    scheme_a.run(kThresNeigh);
    REQUIRE(same_bits(states_a, scheme_a.get_states()));

    // A different seed gives a different simulation.
    auto scheme_c = make_scheme(mode, 4321);
    scheme_c.run(kThresNeigh);
    REQUIRE_FALSE(same_bits(states_a, scheme_c.get_states()));
}

TEST_CASE("CUDA DynamicThresholdScheme states are self-consistent",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto* mode    = GENERATE("legacy", "improved");
    const SizeType nbin = GENERATE(SizeType{32}, SizeType{50}, SizeType{64});
    CAPTURE(mode, nbin);
    auto scheme = make_scheme(mode, 99, nbin);
    scheme.run(kThresNeigh);
    const auto states     = scheme.get_states();
    const auto thresholds = scheme.get_thresholds();
    const auto probs      = scheme.get_probs();
    const SizeType nprobs = probs.size();
    const SizeType nthr   = thresholds.size();
    const SizeType nst    = kBranching.size();
    REQUIRE(states.size() == nst * nthr * nprobs);

    const auto at = [&](SizeType s, SizeType t, SizeType p) -> const State& {
        return states[(s * nthr + t) * nprobs + p];
    };
    for (SizeType s = 0; s < nst; ++s) {
        SizeType nonempty = 0;
        for (SizeType t = 0; t < nthr; ++t) {
            for (SizeType p = 0; p < nprobs; ++p) {
                const State& st = at(s, t, p);
                if (st.is_empty) {
                    continue;
                }
                ++nonempty;
                CAPTURE(s, t, p);
                // Cell coordinates match the state's content.
                REQUIRE(st.threshold == thresholds[t]);
                REQUIRE(st.success_h1_cumul >= probs[p]);
                if (p + 1 < nprobs) {
                    REQUIRE(st.success_h1_cumul < probs[p + 1]);
                }
                REQUIRE(st.nbranches == kBranching[s]);
                // Survival fractions are counts out of ntrials.
                for (const float succ : {st.success_h0, st.success_h1}) {
                    REQUIRE(succ >= 0.0F);
                    REQUIRE(succ <= 1.0F);
                    const float scaled = succ * static_cast<float>(kNtrials);
                    REQUIRE(scaled == std::round(scaled));
                }
                REQUIRE(std::isfinite(st.complexity_cumul));
                REQUIRE(st.cost == st.complexity_cumul / st.success_h1_cumul);
                // The back-pointer names a non-empty parent whose cumulative
                // detection probability chains into this state.
                if (s > 0) {
                    bool found = false;
                    for (SizeType pt = 0; pt < nthr && !found; ++pt) {
                        for (SizeType pp = 0; pp < nprobs && !found; ++pp) {
                            const State& par = at(s - 1, pt, pp);
                            found = !par.is_empty &&
                                    par.threshold == st.threshold_prev &&
                                    par.success_h1_cumul ==
                                        st.success_h1_cumul_prev &&
                                    // <=: a tiny surviving complexity can
                                    // vanish in the float32 sum.
                                    par.complexity_cumul <= st.complexity_cumul;
                        }
                    }
                    REQUIRE(found);
                    REQUIRE(st.success_h1_cumul ==
                            st.success_h1_cumul_prev * st.success_h1);
                }
            }
        }
        // Every stage of this configuration is reachable.
        REQUIRE(nonempty > 0);
    }
    REQUIRE(scheme.get_best_path_thresholds().size() == nst);
}

TEST_CASE("CUDA DynamicThresholdScheme batch_size only shifts RNG streams",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    // batch_size is kept in the RNG offset bookkeeping for compatibility, so
    // results change with it, but each setting stays reproducible and valid.
    const auto* mode = GENERATE("legacy", "improved");
    CAPTURE(mode);
    for (const SizeType batch : {SizeType{16}, SizeType{1024}}) {
        auto a = make_scheme(mode, 7, 64, batch);
        auto b = make_scheme(mode, 7, 64, batch);
        a.run(kThresNeigh);
        b.run(kThresNeigh);
        REQUIRE(same_bits(a.get_states(), b.get_states()));
        REQUIRE(count_nonempty(a.get_states()) > 0);
    }
}

TEST_CASE("CUDA DynamicThresholdScheme rejects invalid input",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto make = [](std::span<const float> branching, SizeType nbins,
                         float prob_min, float ducy_max, float beam_width,
                         SizeType batch_size) {
        return DynamicThresholdScheme(branching, 0.1F, nbins, kNtrials, kNprobs,
                                      prob_min, 8.0F, kNthresholds, ducy_max,
                                      1.2F, beam_width, 1, "improved",
                                      /*seed=*/1, batch_size, Exec::cuda());
    };
    const std::vector<float> one_stage = {2.0F};
    const std::vector<float> negative  = {2.0F, -1.0F, 2.0F};
    const std::vector<float> nan_value = {
        2.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
    const std::span<const float> ok(kBranching);

    REQUIRE_THROWS_AS(
        make(std::span<const float>{}, 64, 0.05F, 0.3F, 1.5F, 256),
        std::invalid_argument);
    REQUIRE_THROWS_AS(make(one_stage, 64, 0.05F, 0.3F, 1.5F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(negative, 64, 0.05F, 0.3F, 1.5F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(nan_value, 64, 0.05F, 0.3F, 1.5F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(ok, 2048, 0.05F, 0.3F, 1.5F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(ok, 64, 0.0F, 0.3F, 1.5F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(ok, 64, 0.05F, 1.0F, 1.5F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(ok, 64, 0.05F, 0.3F, 0.0F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make(ok, 64, 0.05F, 0.3F, 1.5F, 0),
                      std::invalid_argument);
    // A beam narrower than the threshold spacing leaves some stage empty.
    REQUIRE_THROWS_AS(make(ok, 64, 0.05F, 0.3F, 1e-4F, 256),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(make_scheme("bogus", 1), std::invalid_argument);

    auto scheme = make_scheme("improved", 1);
    REQUIRE_THROWS_AS(scheme.run(0), std::invalid_argument);
}

TEST_CASE("CUDA DynamicThresholdScheme evaluate rescores a path",
          "[thresholds][cuda]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    auto scheme = make_scheme("improved", 7);
    scheme.run(kThresNeigh);
    const auto before = scheme.get_states();
    const auto path   = scheme.get_best_path_thresholds();
    REQUIRE(path.size() == kBranching.size());

    const auto first  = scheme.evaluate(path, 128, 99);
    const auto second = scheme.evaluate(path, 128, 99);
    REQUIRE(first.size() == kBranching.size());
    REQUIRE(same_bits(first, second));
    REQUIRE(same_bits(before, scheme.get_states()));
    REQUIRE_FALSE(first.back().is_empty);

    const auto other = scheme.evaluate(path, 128, 100);
    REQUIRE_FALSE(same_bits(first, other));
    REQUIRE_THROWS_AS(scheme.evaluate(std::span<const float>{}, 128, 1),
                      std::invalid_argument);
}

} // namespace loki
