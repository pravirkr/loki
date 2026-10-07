#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

namespace loki {
namespace {

using detection::DynamicThresholdScheme;
using detection::State;

constexpr SizeType kNtrials     = 1024;
constexpr SizeType kNprobs      = 12;
constexpr SizeType kNthresholds = 60;
constexpr SizeType kThresNeigh  = 6;
constexpr int kCpuThreads       = 8;
constexpr float kSurvivalTol    = 0.10F;

const std::vector<float> kBranching = {
    4.0F, 9.0F, 1.0F, 2.25575101F, 3.98980204F, 3.0F, 2.80514208F, 3.20839363F,
};

DynamicThresholdScheme make_cpu(std::string_view mode, uint64_t seed) {
    return {kBranching, 0.1F,  64,   kNtrials,
            kNprobs,    0.05F, 8.0F, kNthresholds,
            0.3F,       1.2F,  1.5F, 1,
            mode,       seed,  256,  Exec::cpu(kCpuThreads)};
}

DynamicThresholdScheme make_cuda(std::string_view mode, uint64_t seed) {
    return {kBranching, 0.1F,         64,   kNtrials,    kNprobs, 0.05F,
            8.0F,       kNthresholds, 0.3F, 1.2F,        1.5F,    1,
            mode,       seed,         256,  Exec::cuda()};
}

using SurvivalPair = std::pair<float, float>;

std::map<SizeType, SurvivalPair>
stage0_survival_by_threshold(std::span<const State> states) {
    std::map<SizeType, SurvivalPair> out;
    for (SizeType ithr = 0; ithr < kNthresholds; ++ithr) {
        for (SizeType iprob = 0; iprob < kNprobs; ++iprob) {
            const auto idx = (ithr * kNprobs) + iprob;
            const State& s = states[idx];
            if (s.is_empty) {
                continue;
            }
            out[ithr] = {s.success_h0, s.success_h1};
        }
    }
    return out;
}

SurvivalPair
mean_over_seeds(const std::vector<std::map<SizeType, SurvivalPair>>& by_seed,
                SizeType ithr) {
    float h0   = 0.0F;
    float h1   = 0.0F;
    SizeType n = 0;
    for (const auto& m : by_seed) {
        const auto it = m.find(ithr);
        if (it == m.end()) {
            continue;
        }
        h0 += it->second.first;
        h1 += it->second.second;
        ++n;
    }
    if (n == 0) {
        return {0.0F, 0.0F};
    }
    const float inv = 1.0F / static_cast<float>(n);
    return {h0 * inv, h1 * inv};
}

} // namespace

TEST_CASE("DynamicThresholdScheme CPU and CUDA agree on stage-0 survival",
          "[thresholds][cuda][parity]") {
    if (!is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto* mode = GENERATE("legacy", "improved");
    CAPTURE(mode);

    std::vector<std::map<SizeType, SurvivalPair>> cpu_maps;
    std::vector<std::map<SizeType, SurvivalPair>> cuda_maps;
    constexpr uint64_t kSeed0  = 0;
    constexpr SizeType kNSeeds = 3;

    for (uint64_t seed = kSeed0; seed < kSeed0 + kNSeeds; ++seed) {
        auto cpu = make_cpu(mode, seed);
        cpu.run(kThresNeigh);
        cpu_maps.push_back(stage0_survival_by_threshold(cpu.get_states()));

        auto gpu = make_cuda(mode, seed);
        gpu.run(kThresNeigh);
        cuda_maps.push_back(stage0_survival_by_threshold(gpu.get_states()));
    }

    std::vector<SizeType> common;
    for (const auto& entry : cpu_maps.front()) {
        // Clang 18 with OpenMP cannot capture a structured binding.
        const SizeType ithr = entry.first;
        const bool on_cuda  = std::ranges::all_of(
            cuda_maps, [ithr](const std::map<SizeType, SurvivalPair>& m) {
                return m.contains(ithr);
            });
        if (on_cuda) {
            common.push_back(ithr);
        }
    }
    REQUIRE_FALSE(common.empty());

    for (const SizeType ithr : common) {
        const auto [cpu_h0, cpu_h1]   = mean_over_seeds(cpu_maps, ithr);
        const auto [cuda_h0, cuda_h1] = mean_over_seeds(cuda_maps, ithr);
        INFO("threshold index " << ithr);
        REQUIRE(std::abs(cpu_h0 - cuda_h0) <= kSurvivalTol);
        REQUIRE(std::abs(cpu_h1 - cuda_h1) <= kSurvivalTol);
    }
}

} // namespace loki
