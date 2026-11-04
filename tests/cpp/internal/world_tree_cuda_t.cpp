#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/world_tree_cuda.cuh"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/types_cuda.cuh"

using loki::ComplexTypeCUDA;
using loki::SizeType;
using loki::memory::WorldTreeCUDA;

namespace {

template <typename T> T make_value(SizeType i) {
    if constexpr (std::is_same_v<T, ComplexTypeCUDA>) {
        return ComplexTypeCUDA(static_cast<float>(i), -static_cast<float>(i));
    } else {
        return static_cast<T>(i);
    }
}

template <typename T> bool same_value(const T& lhs, const T& rhs) {
    if constexpr (std::is_same_v<T, ComplexTypeCUDA>) {
        return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
    } else {
        return lhs == rhs;
    }
}

template <typename T>
std::vector<T> to_host(cuda::std::span<const T> view, SizeType n) {
    std::vector<T> out(n);
    thrust::copy_n(thrust::device_pointer_cast(view.data()), n, out.begin());
    return out;
}

/// Scores of the live leaves of a tree that is not updating, read from
/// physical slot @p start on, each checked against the first value of its
/// leaf (the tests store a leaf's score there).
template <typename T>
std::vector<float> live_scores(const WorldTreeCUDA<T>& tree, SizeType start) {
    const auto capacity = tree.get_capacity();
    const auto stride   = tree.get_leaves_stride();
    const auto scores   = to_host(tree.get_scores_span(), capacity);
    const auto leaves   = to_host(tree.get_leaves_span(), capacity * stride);
    std::vector<float> out(tree.get_size());
    for (SizeType i = 0; i < out.size(); ++i) {
        const auto slot = (start + i) % capacity;
        out[i]          = scores[slot];
        REQUIRE(leaves[slot * stride] == static_cast<double>(out[i]));
    }
    return out;
}

} // namespace

TEMPLATE_TEST_CASE("WorldTreeCUDA::add_initial takes the leading seeds of a "
                   "larger batch",
                   "[world_tree][cuda]",
                   float,
                   ComplexTypeCUDA) {
    // A sweep sizes its seed buffers for the largest chunk of a group and
    // passes the whole buffer with the chunk's own leaf count.
    constexpr SizeType kCapacity = 64;
    constexpr SizeType kNParams  = 2;
    constexpr SizeType kNBins    = 8;
    constexpr SizeType kBatch    = 16;
    constexpr SizeType kNSeeds   = 40;
    constexpr SizeType kNUsed    = 25;

    WorldTreeCUDA<TestType> tree(kCapacity, kNParams, kNBins, kBatch);
    const auto leaves_stride = tree.get_leaves_stride();
    const auto folds_stride  = tree.get_folds_stride();

    std::vector<double> leaves(kNSeeds * leaves_stride);
    std::vector<TestType> folds(kNSeeds * folds_stride);
    std::vector<float> scores(kNSeeds);
    for (SizeType i = 0; i < kNSeeds; ++i) {
        for (SizeType j = 0; j < leaves_stride; ++j) {
            leaves[(i * leaves_stride) + j] =
                static_cast<double>((i * 1000) + j);
        }
        for (SizeType j = 0; j < folds_stride; ++j) {
            folds[(i * folds_stride) + j] = make_value<TestType>((i * 100) + j);
        }
        scores[i] = static_cast<float>(i) * 0.25F;
    }
    const thrust::device_vector<double> leaves_d(leaves);
    const thrust::device_vector<TestType> folds_d(folds);
    const thrust::device_vector<float> scores_d(scores);
    cudaStream_t stream = nullptr;

    tree.add_initial(loki::cuda_utils::as_span(leaves_d),
                     loki::cuda_utils::as_span(folds_d),
                     loki::cuda_utils::as_span(scores_d), kNUsed, stream);
    loki::cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                      "add_initial sync failed");
    REQUIRE(tree.get_size() == kNUsed);

    const auto scores_h = to_host(tree.get_scores_span(), kNUsed);
    const auto leaves_h =
        to_host(tree.get_leaves_span(), kNUsed * leaves_stride);
    const auto folds_h = to_host(tree.get_folds_span(), kNUsed * folds_stride);
    for (SizeType i = 0; i < kNUsed; ++i) {
        REQUIRE(scores_h[i] == scores[i]);
        for (SizeType j = 0; j < leaves_stride; ++j) {
            REQUIRE(leaves_h[(i * leaves_stride) + j] ==
                    leaves[(i * leaves_stride) + j]);
        }
        for (SizeType j = 0; j < folds_stride; ++j) {
            REQUIRE(same_value(folds_h[(i * folds_stride) + j],
                               folds[(i * folds_stride) + j]));
        }
    }
    REQUIRE(tree.get_score_max(stream) == scores[kNUsed - 1]);

    // A batch that holds fewer seeds than it is asked to copy is rejected.
    REQUIRE_THROWS(tree.add_initial(
        loki::cuda_utils::as_span(leaves_d), loki::cuda_utils::as_span(folds_d),
        loki::cuda_utils::as_span(scores_d, kNUsed - 1), kNUsed, stream));
    REQUIRE_THROWS(tree.add_initial(
        loki::cuda_utils::as_span(leaves_d),
        loki::cuda_utils::as_span(folds_d, (kNUsed - 1) * folds_stride),
        loki::cuda_utils::as_span(scores_d), kNUsed, stream));
    REQUIRE_THROWS(tree.add_initial(
        loki::cuda_utils::as_span(leaves_d, (kNUsed - 1) * leaves_stride),
        loki::cuda_utils::as_span(folds_d), loki::cuda_utils::as_span(scores_d),
        kNUsed, stream));
}

TEST_CASE(
    "WorldTreeCUDA overflow matches CPU and validate checks leaves stride",
    "[world_tree][cuda]") {
    // The median rule: an overflow keeps at most half of the pool.
    constexpr double kMedian = 0.5;
    loki::cuda_utils::CudaSetDeviceGuard const guard(0);
    constexpr SizeType kCapacity = 8;
    constexpr SizeType kNParams  = 1;
    constexpr SizeType kNBins    = 4;
    constexpr SizeType kBatch    = 6;

    WorldTreeCUDA<float> tree(kCapacity, kNParams, kNBins, kBatch);
    const auto leaves_stride = tree.get_leaves_stride();
    const auto folds_stride  = tree.get_folds_stride();

    const float exact_thresh =
        std::nextafter(35.0F, std::numeric_limits<float>::max());
    std::vector<double> batch1_leaves(6 * leaves_stride, 1.0);
    std::vector<float> batch1_folds(6 * folds_stride, 1.0F);
    std::vector<float> batch1_scores{10.0F,        20.0F, 30.0F,
                                     exact_thresh, 50.0F, 60.0F};
    std::vector<uint32_t> batch1_indices{0, 1, 2, 3, 4, 5};
    const thrust::device_vector<double> batch1_leaves_d(batch1_leaves);
    const thrust::device_vector<float> batch1_folds_d(batch1_folds);
    const thrust::device_vector<float> batch1_scores_d(batch1_scores);
    const thrust::device_vector<uint32_t> batch1_indices_d(batch1_indices);

    cudaStream_t stream = nullptr;
    tree.prepare_in_place_update();
    REQUIRE(
        tree.add_batch_scattered(loki::cuda_utils::as_span(batch1_leaves_d),
                                 loki::cuda_utils::as_span(batch1_folds_d),
                                 loki::cuda_utils::as_span(batch1_scores_d),
                                 loki::cuda_utils::as_span(batch1_indices_d),
                                 5.0F, 6, kMedian, stream) == 5.0F);
    REQUIRE(tree.get_size() == 6);

    std::vector<double> batch2_leaves(4 * leaves_stride, 2.0);
    std::vector<float> batch2_folds(4 * folds_stride, 2.0F);
    std::vector<float> batch2_scores{25.0F, 35.0F, 45.0F, 55.0F};
    std::vector<uint32_t> batch2_indices{0, 1, 2, 3};
    const thrust::device_vector<double> batch2_leaves_d(batch2_leaves);
    const thrust::device_vector<float> batch2_folds_d(batch2_folds);
    const thrust::device_vector<float> batch2_scores_d(batch2_scores);
    const thrust::device_vector<uint32_t> batch2_indices_d(batch2_indices);

    const float new_thresh = tree.add_batch_scattered(
        loki::cuda_utils::as_span(batch2_leaves_d),
        loki::cuda_utils::as_span(batch2_folds_d),
        loki::cuda_utils::as_span(batch2_scores_d),
        loki::cuda_utils::as_span(batch2_indices_d), 5.0F, 4, kMedian, stream);
    REQUIRE(new_thresh == exact_thresh);
    REQUIRE(tree.get_size() == 5);

    REQUIRE_NOTHROW(tree.validate(kCapacity, kNParams, kNBins, kBatch));
}

TEMPLATE_TEST_CASE("WorldTreeCUDA overflow keeps at most keep_fraction of the "
                   "pool",
                   "[world_tree][cuda]",
                   float,
                   ComplexTypeCUDA) {
    // The pool and table of the CPU test (world_tree_t.cpp): 2 * kBatch
    // candidates for kCapacity slots. The fraction ranks the lowest score
    // that may stay; past rank kCapacity - 1 the top-k decides. 0.5 is the
    // median of the pool.
    const auto row = GENERATE(table<double, SizeType>({
        {0.5, 40},
        {0.75, 60},
        {0.9, 63},
        {1.0, 63},
    }));

    const double keep_fraction = std::get<0>(row);
    const SizeType n_kept      = std::get<1>(row);
    CAPTURE(keep_fraction);
    // The scatter kernels read the device context, which a guard sets up.
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    constexpr SizeType kCapacity = 64;
    constexpr SizeType kNParams  = 2;
    constexpr SizeType kNBins    = 8;
    constexpr SizeType kBatch    = 40;
    constexpr SizeType kNSeeds   = 8;

    WorldTreeCUDA<TestType> tree(kCapacity, kNParams, kNBins, kBatch);
    const auto leaves_stride = tree.get_leaves_stride();
    const auto folds_stride  = tree.get_folds_stride();
    cudaStream_t stream      = nullptr;

    // Two batches of kBatch leaves: the first holds the even scores, the
    // second the odd ones, so the pool of both is 0, 1, ..., 2 * kBatch - 1.
    // Each leaf stores its score as its first value.
    std::vector<double> leaves;
    std::vector<float> scores;
    auto make_batch = [&](SizeType parity) {
        leaves.assign(kBatch * leaves_stride, 0.0);
        scores.resize(kBatch);
        for (SizeType i = 0; i < kBatch; ++i) {
            scores[i]                 = static_cast<float>((2 * i) + parity);
            leaves[i * leaves_stride] = static_cast<double>(scores[i]);
        }
    };
    const thrust::device_vector<TestType> folds_d(kBatch * folds_stride,
                                                  make_value<TestType>(1));
    std::vector<uint32_t> identity(kBatch);
    std::iota(identity.begin(), identity.end(), uint32_t{0});
    const thrust::device_vector<uint32_t> identity_d(identity);

    // Consume every parent first, so the whole ring is free for the new level,
    // which then starts at the read position.
    make_batch(0);
    thrust::device_vector<double> leaves_d(leaves);
    thrust::device_vector<float> scores_d(scores);
    tree.add_initial(loki::cuda_utils::as_span(leaves_d),
                     loki::cuda_utils::as_span(folds_d),
                     loki::cuda_utils::as_span(scores_d), kNSeeds, stream);
    tree.prepare_in_place_update();
    tree.consume_read(kNSeeds);
    const auto start = tree.get_physical_start_idx();

    // A batch that fits is stored whatever the fraction.
    float threshold = tree.add_batch_scattered(
        loki::cuda_utils::as_span(leaves_d), loki::cuda_utils::as_span(folds_d),
        loki::cuda_utils::as_span(scores_d),
        loki::cuda_utils::as_span(identity_d), 0.0F, kBatch, keep_fraction,
        stream);
    REQUIRE(threshold == 0.0F);
    REQUIRE(tree.get_size() == kBatch);
    REQUIRE(tree.overload_count() == 0);

    // The pool is 0, 1, ..., 2 * kBatch - 1, so the n_kept highest stay.
    const auto n_pool = 2 * kBatch;
    make_batch(1);
    leaves_d  = leaves;
    scores_d  = scores;
    threshold = tree.add_batch_scattered(
        loki::cuda_utils::as_span(leaves_d), loki::cuda_utils::as_span(folds_d),
        loki::cuda_utils::as_span(scores_d),
        loki::cuda_utils::as_span(identity_d), threshold, kBatch, keep_fraction,
        stream);
    loki::cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                      "add_batch_scattered sync failed");
    REQUIRE(tree.overload_count() == 1);
    REQUIRE(threshold > static_cast<float>(n_pool - n_kept - 1));
    REQUIRE(threshold <= static_cast<float>(n_pool - n_kept));
    REQUIRE(tree.get_size() == n_kept);
    tree.finalize_in_place_update();

    auto kept = live_scores(tree, start);
    std::sort(kept.begin(), kept.end());
    std::vector<float> expected(n_kept);
    std::iota(expected.begin(), expected.end(),
              static_cast<float>(n_pool - n_kept));
    REQUIRE(kept == expected);
}
#endif // LOKI_ENABLE_CUDA
