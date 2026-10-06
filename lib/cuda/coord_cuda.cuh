#pragma once

/**
 * @file coord_cuda.cuh
 * @brief Device-resident FFA coordinate tables (CUDA only).
 */

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::coord {

struct FFACoordDPtrs {
    uint32_t* __restrict__ i_tail;
    float* __restrict__ shift_tail;
    uint32_t* __restrict__ i_head;
    float* __restrict__ shift_head;
    SizeType size;

    __host__ __device__ FFACoordDPtrs offset(SizeType offset) const noexcept;
};

struct FFACoordFreqDPtrs {
    uint32_t* __restrict__ idx;
    float* __restrict__ shift;
    SizeType size;

    __host__ __device__ FFACoordFreqDPtrs
    offset(SizeType offset) const noexcept;
};

struct FFACoordD {
    thrust::device_vector<uint32_t> i_tail;
    thrust::device_vector<float> shift_tail;
    thrust::device_vector<uint32_t> i_head;
    thrust::device_vector<float> shift_head;

    FFACoordDPtrs get_raw_ptrs() noexcept;
    void resize(SizeType n_coords) noexcept;
    void copy_from_host(const std::vector<FFACoord>& coords,
                        SizeType n_coords,
                        cudaStream_t stream) noexcept;
};

struct FFACoordFreqD {
    thrust::device_vector<uint32_t> idx;
    thrust::device_vector<float> shift;

    FFACoordFreqDPtrs get_raw_ptrs() noexcept;
    void resize(SizeType n_coords) noexcept;
    void copy_from_host(const std::vector<FFACoordFreq>& coords,
                        SizeType n_coords,
                        cudaStream_t stream) noexcept;
};

} // namespace loki::coord
