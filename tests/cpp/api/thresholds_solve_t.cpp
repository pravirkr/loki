#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

// DynamicThresholdScheme::solve() of mode "shared": the schedule to use, with
// the Pd and cost of exactly those thresholds measured on fresh samples.

namespace loki {
namespace {

using detection::DynamicThresholdScheme;
using detection::RefineLevel;
using detection::SharedOptions;

constexpr float kMinPd = 0.3F;

bool skip_backend(std::string_view backend) {
    return backend == "cuda" && !is_available(Backend::kCUDA);
}

DynamicThresholdScheme make(std::string_view backend,
                            RefineLevel refine,
                            uint64_t seed,
                            int nthreads          = 2,
                            std::string_view mode = "shared") {
    const std::vector<float> branching(8, 2.0F);
    const SharedOptions shared{
        .min_pd = kMinPd, .w_terminal = 1.0F, .refine = refine};
    return {branching,
            /*ref_ducy=*/0.1F,
            /*nbins=*/32,
            /*ntrials=*/512,
            /*nprobs=*/20,
            /*prob_min=*/0.05F,
            /*snr_final=*/8.0F,
            /*nthresholds=*/40,
            /*ducy_max=*/0.3F,
            /*wtsp=*/1.0F,
            /*beam_width=*/1.5F,
            /*trials_start=*/1,
            mode,
            seed,
            /*batch_size=*/256,
            "window",
            shared,
            backend == "cuda" ? Exec::cuda() : Exec::cpu(nthreads)};
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (SizeType i = 0; i < a.size(); ++i) {
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("solve: the Pd it reports is that of the returned thresholds",
          "[thresholds][solve]") {
    const auto* backend = GENERATE("cpu", "cuda");
    const auto refine   = GENERATE(RefineLevel::kNone, RefineLevel::kShortlist,
                                   RefineLevel::kFull);
    CAPTURE(backend, detection::refine_to_string(refine));
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    auto scheme = make(backend, refine, 7);
    REQUIRE(scheme.solution() == nullptr);
    scheme.run(10);
    const auto sol = scheme.solve();
    REQUIRE(scheme.solution() != nullptr);
    REQUIRE(sol.thresholds.size() == scheme.get_nstages());
    REQUIRE(sol.delivery != detection::Delivery::kNoPath);
    REQUIRE(sol.refine == refine);
    REQUIRE(sol.min_pd == kMinPd);
    REQUIRE(sol.w_terminal == 1.0F);
    REQUIRE(sol.pd_trials == (SizeType{1} << 17U));
    REQUIRE(sol.pd_se > 0.0);
    REQUIRE(sol.eval_states.size() == scheme.get_nstages());
    REQUIRE_FALSE(sol.search_thresholds.empty());
    // The repair aims two standard errors above the target: delivered
    REQUIRE(sol.pd >= 0.9 * kMinPd);
    // C includes the final survivors (w = 1): more than the scored nodes
    const auto& last = sol.eval_states.back();
    REQUIRE(sol.c == static_cast<double>(last.complexity_cumul) +
                         static_cast<double>(last.complexity));
    REQUIRE(std::abs(sol.cost - (sol.c / sol.pd)) <= 1e-9 * sol.cost);
    // An independent evaluation agrees on the Pd within the Monte Carlo
    // errors of the two samples
    constexpr SizeType kNcheck = 65536;
    const auto check = scheme.evaluate(sol.thresholds, kNcheck, 123456);
    const double pd  = sol.pd;
    const double se  = std::sqrt(pd * (1.0 - pd) *
                                 ((1.0 / static_cast<double>(sol.pd_trials)) +
                                  (1.0 / static_cast<double>(kNcheck))));
    REQUIRE(std::abs(static_cast<double>(check.back().success_h1_cumul) - pd) <=
            5.0 * se);
}

TEST_CASE("solve is reproducible and does not depend on the thread count",
          "[thresholds][solve]") {
    const auto* backend = GENERATE("cpu", "cuda");
    CAPTURE(backend);
    if (skip_backend(backend)) {
        SKIP("needs a CUDA build");
    }
    auto a = make(backend, RefineLevel::kFull, 11, 1);
    auto b = make(backend, RefineLevel::kFull, 11, 3);
    a.run(10);
    b.run(10);
    const auto sa = a.solve();
    const auto sb = b.solve();
    REQUIRE(same_bits(sa.thresholds, sb.thresholds));
    REQUIRE(sa.pd == sb.pd);
    REQUIRE(sa.c == sb.c);
    // A second run() and solve() reproduce the first
    a.run(10);
    REQUIRE(same_bits(a.solve().thresholds, sa.thresholds));
}

TEST_CASE("solve needs mode shared", "[thresholds][solve]") {
    auto improved = make("cpu", RefineLevel::kFull, 3, 2, "improved");
    improved.run(6);
    REQUIRE_THROWS_AS(improved.solve(), std::invalid_argument);
    REQUIRE(improved.solution() == nullptr);
}

TEST_CASE("Shared mode on the CPU needs the window beam", "[thresholds]") {
    const std::vector<float> branching(8, 2.0F);
    REQUIRE_THROWS_AS(DynamicThresholdScheme(branching, 0.1F, 32, 512, 10,
                                             0.05F, 8.0F, 40, 0.3F, 1.0F, 1.5F,
                                             1, "shared", 7, 256, Exec::cpu(2)),
                      std::invalid_argument);
}

TEST_CASE("save works in every mode, with the solution after solve()",
          "[thresholds][solve]") {
    const auto dir = std::filesystem::temp_directory_path() / "loki_solve_save";
    std::filesystem::remove_all(dir);
    for (const std::string_view mode : {"legacy", "improved", "shared"}) {
        CAPTURE(mode);
        auto scheme = make("cpu", RefineLevel::kShortlist, 5, 2, mode);
        scheme.run(6);
        const auto before = scheme.save((dir / mode).string());
        REQUIRE(std::filesystem::exists(before));
        if (mode == "shared") {
            const auto sol   = scheme.solve();
            const auto after = scheme.save((dir / "solved").string());
            REQUIRE(std::filesystem::exists(after));
            REQUIRE(std::filesystem::file_size(after) >
                    std::filesystem::file_size(before));
        }
    }
    std::filesystem::remove_all(dir);
}

} // namespace loki
