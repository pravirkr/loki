#pragma once

#include <cstdint>
#include <cuda/std/span>
#include <cuda_runtime.h>

#include "loki/common/types.hpp"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::detection {

void snr_boxcar_2d_max_cuda_d(cuda::std::span<const float> folds,
                              cuda::std::span<const uint32_t> widths,
                              cuda::std::span<float> scores,
                              SizeType nprofiles,
                              SizeType nbins,
                              float stdnoise      = 1.0F,
                              cudaStream_t stream = nullptr);

void snr_boxcar_3d_cuda_d(cuda::std::span<const float> folds,
                          cuda::std::span<const uint32_t> widths,
                          cuda::std::span<float> scores,
                          SizeType nprofiles,
                          SizeType nbins,
                          cudaStream_t stream = nullptr);

void snr_boxcar_3d_max_cuda_d(cuda::std::span<const float> folds,
                              cuda::std::span<const uint32_t> widths,
                              cuda::std::span<float> scores,
                              SizeType nprofiles,
                              SizeType nbins,
                              cudaStream_t stream = nullptr);

SizeType score_and_filter_cuda_d(cuda::std::span<const float> folds,
                                 cuda::std::span<const uint32_t> widths,
                                 cuda::std::span<float> scores,
                                 cuda::std::span<uint32_t> indices_filtered,
                                 float threshold,
                                 SizeType nprofiles,
                                 SizeType nbins,
                                 cudaStream_t stream,
                                 memory::DeviceCounter& counter);

SizeType
score_and_filter_max_cuda_d(cuda::std::span<const float> folds,
                            cuda::std::span<const uint32_t> widths,
                            cuda::std::span<float> scores,
                            cuda::std::span<const uint8_t> validation_mask,
                            cuda::std::span<uint8_t> filtered_mask,
                            float threshold,
                            SizeType nprofiles,
                            SizeType nbins,
                            memory::CUBScratchArena& scratch_ws,
                            cudaStream_t stream);

} // namespace loki::detection
