#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/plans.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::plans {

struct FFACoordDPtrs {
    const uint32_t* __restrict__ i_tail;
    const float* __restrict__ shift_tail;
    const uint32_t* __restrict__ i_head;
    const float* __restrict__ shift_head;

    __host__ __device__ __forceinline__ FFACoordDPtrs
    offset(SizeType offset_value) const {
        return {.i_tail     = i_tail + offset_value,
                .shift_tail = shift_tail + offset_value,
                .i_head     = i_head + offset_value,
                .shift_head = shift_head + offset_value};
    }
};

struct FFACoordFreqDPtrs {
    const uint32_t* __restrict__ idx;
    const float* __restrict__ shift;

    __host__ __device__ __forceinline__ FFACoordFreqDPtrs
    offset(SizeType offset_value) const {
        return {.idx = idx + offset_value, .shift = shift + offset_value};
    }
};

struct FFACoordD {
    thrust::device_vector<uint32_t> i_tail;
    thrust::device_vector<float> shift_tail;
    thrust::device_vector<uint32_t> i_head;
    thrust::device_vector<float> shift_head;

    __host__ FFACoordDPtrs get_raw_ptrs() const {
        return {.i_tail     = thrust::raw_pointer_cast(i_tail.data()),
                .shift_tail = thrust::raw_pointer_cast(shift_tail.data()),
                .i_head     = thrust::raw_pointer_cast(i_head.data()),
                .shift_head = thrust::raw_pointer_cast(shift_head.data())};
    }

    __host__ void resize(SizeType n_coords) {
        i_tail.resize(n_coords);
        shift_tail.resize(n_coords);
        i_head.resize(n_coords);
        shift_head.resize(n_coords);
    }

    __host__ void copy_from_host(const std::vector<FFACoord>& coords,
                                 SizeType n_coords,
                                 cudaStream_t stream) {
        // Extract to contiguous host arrays
        std::vector<uint32_t> i_tail_h(n_coords), i_head_h(n_coords);
        std::vector<float> shift_tail_h(n_coords), shift_head_h(n_coords);

        for (SizeType i = 0; i < n_coords; ++i) {
            i_tail_h[i]     = coords[i].i_tail;
            shift_tail_h[i] = coords[i].shift_tail;
            i_head_h[i]     = coords[i].i_head;
            shift_head_h[i] = coords[i].shift_head;
        }

        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(i_tail.data()),
                            i_tail_h.data(), n_coords * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync i_tail failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(shift_tail.data()),
                            shift_tail_h.data(), n_coords * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync shift_tail failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(i_head.data()),
                            i_head_h.data(), n_coords * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync i_head failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(shift_head.data()),
                            shift_head_h.data(), n_coords * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync shift_head failed");
    }
};

struct FFACoordFreqD {
    thrust::device_vector<uint32_t> idx;
    thrust::device_vector<float> shift;

    __host__ FFACoordFreqDPtrs get_raw_ptrs() const {
        return {.idx   = thrust::raw_pointer_cast(idx.data()),
                .shift = thrust::raw_pointer_cast(shift.data())};
    }

    __host__ FFACoordFreqDPtrs get_raw_ptrs_at_offset(int offset_value) const {
        return {.idx   = thrust::raw_pointer_cast(idx.data()) + offset_value,
                .shift = thrust::raw_pointer_cast(shift.data()) + offset_value};
    }

    __host__ void resize(SizeType n_coords) {
        idx.resize(n_coords);
        shift.resize(n_coords);
    }

    __host__ void copy_from_host(const std::vector<FFACoordFreq>& coords,
                                 SizeType n_coords,
                                 cudaStream_t stream) {

        std::vector<uint32_t> idx_h(n_coords);
        std::vector<float> shift_h(n_coords);

        for (SizeType i = 0; i < n_coords; ++i) {
            idx_h[i]   = coords[i].idx;
            shift_h[i] = coords[i].shift;
        }

        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(idx.data()), idx_h.data(),
                            n_coords * sizeof(uint32_t), cudaMemcpyHostToDevice,
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync idx failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(shift.data()),
                            shift_h.data(), n_coords * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync shift failed");
    }
};

} // namespace loki::plans