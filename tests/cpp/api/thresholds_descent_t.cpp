#include <cstdint>
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
    // Pinned at the target: no overshoot beyond the sampling error.
    REQUIRE(sol.pd >= 0.97 * kMinPd);
    REQUIRE(sol.pd <= 1.05 * kMinPd);
    REQUIRE(sol.pd_trials == (SizeType{1} << 17U));
    REQUIRE(sol.eval_states.size() == scheme.get_nstages());
    REQUIRE(sol.c > 1.0);
    REQUIRE(scheme.solution() != nullptr);

    // Deterministic per seed
    auto again = make(backend, 7, options());
    REQUIRE(again.solve().thresholds == sol.thresholds);
}

TEST_CASE("descent: the last threshold has a floor", "[thresholds][descent]") {
    const auto backend =
        GENERATE(std::string_view{"cpu"}, std::string_view{"cuda"});
    if (skip_backend(backend)) {
        SKIP("backend not built");
    }
    auto fixed     = make(backend, 7, options(6.5F));
    const auto sol = fixed.solve();
    REQUIRE(sol.thresholds.back() >= 6.5F);
    REQUIRE(sol.pd >= 0.95 * kMinPd);

    // A floor below the free last threshold changes nothing.
    auto free_scheme = make(backend, 7, options());
    auto low_floor   = make(backend, 7, options(1.0F));
    REQUIRE(low_floor.solve().thresholds == free_scheme.solve().thresholds);

    // A cut above what min_pd allows: no schedule, the ceiling is reported.
    auto unreachable = make(backend, 7, options(9.0F));
    const auto none  = unreachable.solve();
    REQUIRE(none.thresholds.empty());
    REQUIRE(none.delivery == detection::Delivery::kShort);
    REQUIRE(none.pd < kMinPd);
}

} // namespace
} // namespace loki
