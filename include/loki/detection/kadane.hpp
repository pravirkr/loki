#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#include "loki/utils/workspace.hpp"
#endif // LOKI_ENABLE_CUDA

namespace loki::detection {

struct BoxcarKadaneCache {
    std::vector<float> biases;
    SizeType n_biases;
    std::vector<float> fold_norm_buffer;

    BoxcarKadaneCache(std::span<const float> biases, SizeType nbins);
};

// Compute the S/N of a batch of folded profiles using the Kadane algorithm
SizeType
score_and_filter_max_kadane_with_cache(std::span<const float> folds,
                                       std::span<float> scores,
                                       std::span<SizeType> indices_filtered,
                                       float threshold,
                                       SizeType nprofiles,
                                       SizeType nbins,
                                       BoxcarKadaneCache& cache);

#ifdef LOKI_ENABLE_CUDA
SizeType score_and_filter_max_cuda_kadane_d(
    cuda::std::span<const float> folds,
    cuda::std::span<const float> biases,
    cuda::std::span<float> scores,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint8_t> filtered_mask,
    float threshold,
    SizeType nprofiles,
    SizeType nbins,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::detection