#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/world_tree_cuda.cuh"

#include <cstdint>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
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
}
#endif // LOKI_ENABLE_CUDA
