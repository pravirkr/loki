#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

// DynamicThresholdScheme mode "descent": solve() designs the schedule by
// moving thresholds on fixed samples, with Pd held at the target.

namespace loki {
namespace {

using detection::DynamicThresholdScheme;
using detection::SharedOptions;

constexpr float kMinPd = 0.3F;

bool skip_backend(std::string_view backend) {
    return backend == "cuda" && !is_available(Backend::kCUDA);
}

SharedOptions options(float final_threshold = 0.0F) {
    SharedOptions shared;
    shared.min_pd          = kMinPd;
    shared.final_threshold = final_threshold;
    // Small samples keep the test fast; the defaults are larger.
    shared.descent_passes = {
        {1024, 0.4F, 0.2F}, {2048, 0.3F, 0.1F}, {2048, 0.2F, 0.05F, 4, 1}};
    shared.descent_nsignal = 8192;
    return shared;
}

DynamicThresholdScheme
make(std::string_view backend, uint64_t seed, const SharedOptions& shared) {
    const std::vector<float> branching = {2, 3, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2};
    return {branching,
            /*ref_ducy=*/0.1F,
            /*nbins=*/32,
            /*ntrials=*/512,
            /*nprobs=*/20,
            /*prob_min=*/0.05F,
            /*snr_final=*/8.0F,
            /*nthresholds=*/40,
            /*ducy_max=*/0.3F,
            /*wtsp=*/1.5F,
            /*beam_width=*/1.5F,
            /*trials_start=*/1,
            "descent",
            seed,
            /*batch_size=*/256,
            "window",
            shared,
            backend == "cuda" ? Exec::cuda() : Exec::cpu(2)};
}

TEST_CASE("descent: solve() delivers the target Pd", "[thresholds][descent]") {
    const auto backend =
        GENERATE(std::string_view{"cpu"}, std::string_view{"cuda"});
    if (skip_backend(backend)) {
        SKIP("backend not built");
    }
    auto scheme = make(backend, 7, options());
    scheme.run(); // nothing to search in this mode
    const auto sol = scheme.solve();
    REQUIRE(sol.thresholds.size() == scheme.get_nstages());
    REQUIRE(sol.delivery != detection::Delivery::kShort);
    REQUIRE(sol.delivery != detection::Delivery::kNoPath);
    REQUIRE(sol.pd >= 0.97 * kMinPd);
    REQUIRE(sol.pd <= 1.05 * kMinPd);
    REQUIRE(sol.pd_trials == (SizeType{1} << 17U));
    REQUIRE(sol.eval_states.size() == scheme.get_nstages());
    REQUIRE(sol.c > 1.0);
    REQUIRE(scheme.solution() != nullptr);

    // Deterministic per seed (and independent of host thread count on CPU)
    auto again = make(backend, 7, options());
    REQUIRE(again.solve().thresholds == sol.thresholds);
    if (backend == "cpu") {
        const std::vector<float> bp = {2, 3, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2};
        DynamicThresholdScheme one_thread(
            bp, 0.1F, 32, 512, 20, 0.05F, 8.0F, 40, 0.3F, 1.5F, 1.5F, 1,
            "descent", 7, 256, "window", options(), Exec::cpu(1));
        REQUIRE(one_thread.solve().thresholds == sol.thresholds);
    }

    // PdPolicy::kFree minimises L = C / Pd on short low-cost trees and may
    // settle at an interior Pd > min_pd with lower L than kPin.
    auto free_opts      = options();
    free_opts.pd_policy = detection::PdPolicy::kFree;
    const auto free_sol = make(backend, 7, free_opts).solve();
    REQUIRE(free_sol.pd >= 0.97 * kMinPd);
    REQUIRE(free_sol.pd <= 1.0);
    REQUIRE(free_sol.cost <= 1.05 * sol.cost);
    // A warm start at the free optimum is an interior point, not pinned back
    // to min_pd: it gives a schedule no worse than the cold solve.
    auto warm_opts          = free_opts;
    warm_opts.descent_start = free_sol.thresholds;
    const auto warm_sol     = make(backend, 7, warm_opts).solve();
    REQUIRE(warm_sol.pd >= 0.97 * kMinPd);
    REQUIRE(warm_sol.cost <= 1.01 * free_sol.cost);

    // Steep branching pattern (24 stages of 3): kPin holds Pd at the target
    // there too.
    const std::vector<float> steep(24, 3.0F);
    DynamicThresholdScheme steep_scheme(
        steep, 0.1F, 32, 512, 20, 0.05F, 9.0F, 40, 0.3F, 1.5F, 1.5F, 1,
        "descent", 7, 256, "window", options(),
        backend == "cuda" ? Exec::cuda() : Exec::cpu(2));
    const auto steep_sol = steep_scheme.solve();
    REQUIRE(steep_sol.delivery != detection::Delivery::kShort);
    REQUIRE(steep_sol.pd >= 0.97 * kMinPd);
    REQUIRE(steep_sol.pd <= 1.05 * kMinPd);
}

TEST_CASE("descent: the last threshold has a floor", "[thresholds][descent]") {
    const auto backend =
        GENERATE(std::string_view{"cpu"}, std::string_view{"cuda"});
    if (skip_backend(backend)) {
        SKIP("backend not built");
    }
    const auto free_sol = make(backend, 7, options()).solve();
    // An active floor above the free last threshold holds it at the floor.
    const float active_floor = free_sol.thresholds.back() + 0.4F;
    auto fixed               = make(backend, 7, options(active_floor));
    const auto sol           = fixed.solve();
    REQUIRE(sol.thresholds.back() >= active_floor - 1e-5F);
    REQUIRE(sol.pd >= 0.95 * kMinPd);
    REQUIRE(sol.pd <= 1.05 * kMinPd);

    // A floor below the free last threshold changes nothing.
    auto low_floor = make(backend, 7, options(1.0F));
    REQUIRE(low_floor.solve().thresholds == free_sol.thresholds);

    // A cut above what min_pd allows: no schedule, the ceiling is reported.
    auto unreachable = make(backend, 7, options(9.0F));
    const auto none  = unreachable.solve();
    REQUIRE(none.thresholds.empty());
    REQUIRE(none.delivery == detection::Delivery::kShort);
    REQUIRE(none.pd < kMinPd);
}

TEST_CASE("descent: settings that describe no descent raise at construction",
          "[thresholds][descent]") {
    const auto bad = [](auto mutate) {
        auto shared = options();
        mutate(shared);
        return shared;
    };
    const std::vector<SharedOptions> cases = {
        bad([](SharedOptions& s) { s.descent_passes = {{0, 0.3F, 0.1F}}; }),
        bad([](SharedOptions& s) { s.descent_passes = {{1024, 0.1F, 0.3F}}; }),
        bad([](SharedOptions& s) {
            s.descent_passes = {{1024, 0.3F, 0.1F, 1, 8}};
        }),
        bad([](SharedOptions& s) { s.descent_nsignal = 0; }),
        bad([](SharedOptions& s) { s.descent_start = {1.0F, 2.0F}; }),
        bad([](SharedOptions& s) { s.fap = -1.0; }),
        bad([](SharedOptions& s) { s.final_threshold = -1.0F; }),
        bad([](SharedOptions& s) { s.min_pd = 1.0F; }),
        bad([](SharedOptions& s) {
            s.descent_passes = {{1024, 0.025F, 0.0125F}};
        }),
        bad([](SharedOptions& s) {
            // A NaN from its bits: quiet_NaN() folds under -ffast-math.
            s.w_terminal = std::bit_cast<float>(0x7FC00000U);
        }),
        bad([](SharedOptions& s) { s.w_terminal = -1.0F; }),
    };
    for (const auto& shared : cases) {
        REQUIRE_THROWS_AS(make("cpu", 7, shared), std::invalid_argument);
    }
}

} // namespace
} // namespace loki
