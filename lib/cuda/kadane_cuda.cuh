#pragma once

#include <cstdint>
#include <cuda/std/span>
#include <cuda_runtime.h>

#include "loki/common/types.hpp"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::detection {

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

} // namespace loki::detection
