#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/world_tree_cuda.cuh"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"

using loki::Backend;
using loki::SizeType;

TEST_CASE("CUDA WorldTree add_initial copies the leading seeds of a larger "
          "batch",
          "[world_tree][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    // A sweep sizes its seed buffers for the largest chunk of a group; a
    // smaller chunk seeds the tree from the front of them.
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    constexpr SizeType kNParams = 2;
    constexpr SizeType kNbins   = 8;
    constexpr SizeType kBatch   = 40;
    constexpr SizeType kSeeds   = 25;
    loki::memory::WorldTreeCUDA<float> tree(64, kNParams, kNbins, kBatch);
    const SizeType leaves_stride = tree.get_leaves_stride();
    const SizeType folds_stride  = tree.get_folds_stride();

    std::vector<double> leaves(kBatch * leaves_stride);
    std::vector<float> folds(kBatch * folds_stride);
    std::vector<float> scores(kBatch);
    std::iota(leaves.begin(), leaves.end(), 0.0);
    std::iota(folds.begin(), folds.end(), 0.0F);
    std::iota(scores.begin(), scores.end(), 1.0F);
    // The largest score lies past the copied seeds.
    scores.back() = 1000.0F;
    const thrust::device_vector<double> leaves_d(leaves);
    const thrust::device_vector<float> folds_d(folds);
    const thrust::device_vector<float> scores_d(scores);

    tree.add_initial(loki::cuda_utils::as_span(leaves_d),
                     loki::cuda_utils::as_span(folds_d),
                     loki::cuda_utils::as_span(scores_d), kSeeds, nullptr);
    REQUIRE(tree.get_size() == kSeeds);
    CHECK(tree.get_score_max(nullptr) == static_cast<float>(kSeeds));

    std::vector<double> leaves_out(kSeeds * leaves_stride);
    std::vector<float> folds_out(kSeeds * folds_stride);
    std::vector<float> scores_out(kSeeds);
    const auto tree_leaves = tree.get_leaves_span();
    const auto tree_folds  = tree.get_folds_span();
    const auto tree_scores = tree.get_scores_span();
    thrust::copy_n(thrust::device_pointer_cast(tree_leaves.data()),
                   leaves_out.size(), leaves_out.begin());
    thrust::copy_n(thrust::device_pointer_cast(tree_folds.data()),
                   folds_out.size(), folds_out.begin());
    thrust::copy_n(thrust::device_pointer_cast(tree_scores.data()),
                   scores_out.size(), scores_out.begin());
    CHECK(std::equal(leaves_out.begin(), leaves_out.end(), leaves.begin()));
    CHECK(std::equal(folds_out.begin(), folds_out.end(), folds.begin()));
    CHECK(std::equal(scores_out.begin(), scores_out.end(), scores.begin()));

    // A batch shorter than the count is still rejected.
    const auto short_scores =
        loki::cuda_utils::as_span(scores_d).first(kSeeds - 1);
    CHECK_THROWS_AS(tree.add_initial(loki::cuda_utils::as_span(leaves_d),
                                     loki::cuda_utils::as_span(folds_d),
                                     short_scores, kSeeds, nullptr),
                    std::runtime_error);
    const auto short_folds =
        loki::cuda_utils::as_span(folds_d).first((kSeeds - 1) * folds_stride);
    CHECK_THROWS_AS(
        tree.add_initial(loki::cuda_utils::as_span(leaves_d), short_folds,
                         loki::cuda_utils::as_span(scores_d), kSeeds, nullptr),
        std::runtime_error);
}
#endif // LOKI_ENABLE_CUDA
