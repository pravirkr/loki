#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/workspace_cuda.cuh"

using loki::Backend;
using loki::MinMaxFloat;
using loki::SizeType;

TEST_CASE("Deferred masked min/max matches the host reduction",
          "[cub_scratch][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    constexpr SizeType kBatch     = 512;
    constexpr SizeType kBranchMax = 8;
    loki::memory::CUBScratchArena arena(kBatch, kBranchMax, nullptr);

    std::mt19937 rng(5);
    std::normal_distribution<float> score_dist(3.0F, 2.0F);
    std::bernoulli_distribution keep(0.3);
    // Batch sizes as the prune loop sees them; the last one is all-masked.
    const std::vector<SizeType> sizes = {kBatch * kBranchMax, 1000, 1, 37};
    std::vector<std::vector<float>> scores(sizes.size());
    std::vector<std::vector<uint8_t>> masks(sizes.size());
    for (SizeType b = 0; b < sizes.size(); ++b) {
        scores[b].resize(sizes[b]);
        masks[b].resize(sizes[b]);
        for (SizeType i = 0; i < sizes[b]; ++i) {
            scores[b][i] = score_dist(rng);
            masks[b][i]  = (b + 1 < sizes.size()) && keep(rng) ? 1U : 0U;
        }
    }

    // Enqueue batch b, then read batch b - 1, as the prune loop does.
    std::vector<MinMaxFloat> got;
    thrust::device_vector<float> scores_d;
    thrust::device_vector<uint8_t> masks_d;
    bool pending = false;
    for (SizeType b = 0; b < sizes.size(); ++b) {
        if (pending) {
            got.push_back(arena.wait_min_max());
        }
        scores_d.assign(scores[b].begin(), scores[b].end());
        masks_d.assign(masks[b].begin(), masks[b].end());
        arena.compute_min_max_scores_async(loki::cuda_utils::as_span(scores_d),
                                           loki::cuda_utils::as_span(masks_d),
                                           sizes[b], nullptr);
        pending = true;
    }
    got.push_back(arena.wait_min_max());

    REQUIRE(got.size() == sizes.size());
    for (SizeType b = 0; b < sizes.size(); ++b) {
        CAPTURE(b);
        MinMaxFloat expected{std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::lowest()};
        for (SizeType i = 0; i < sizes[b]; ++i) {
            if (masks[b][i] != 0) {
                expected.min = std::min(expected.min, scores[b][i]);
                expected.max = std::max(expected.max, scores[b][i]);
            }
        }
        CHECK(got[b].min == expected.min);
        CHECK(got[b].max == expected.max);
    }
}
#endif // LOKI_ENABLE_CUDA
