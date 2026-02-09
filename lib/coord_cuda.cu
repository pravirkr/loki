#include "loki/common/coord.hpp"

#include <thrust/device_ptr.h>

#include "loki/cuda_utils.cuh"

namespace loki::coord {

__host__ __device__ FFACoordDPtrs
FFACoordDPtrs::offset(SizeType offset) const noexcept {
    return {.i_tail     = i_tail + offset,
            .shift_tail = shift_tail + offset,
            .i_head     = i_head + offset,
            .shift_head = shift_head + offset,
            .size       = size - offset};
}

__host__ __device__ FFACoordFreqDPtrs
FFACoordFreqDPtrs::offset(SizeType offset) const noexcept {
    return {
        .idx = idx + offset, .shift = shift + offset, .size = size - offset};
}

FFACoordDPtrs FFACoordD::get_raw_ptrs() noexcept {
    const SizeType n = i_tail.size();
    return {.i_tail     = thrust::raw_pointer_cast(i_tail.data()),
            .shift_tail = thrust::raw_pointer_cast(shift_tail.data()),
            .i_head     = thrust::raw_pointer_cast(i_head.data()),
            .shift_head = thrust::raw_pointer_cast(shift_head.data()),
            .size       = n};
}

void FFACoordD::resize(SizeType n_coords) noexcept {
    i_tail.resize(n_coords);
    shift_tail.resize(n_coords);
    i_head.resize(n_coords);
    shift_head.resize(n_coords);
}

void FFACoordD::copy_from_host(const std::vector<FFACoord>& coords,
                               SizeType n_coords,
                               cudaStream_t stream) noexcept {
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

FFACoordFreqDPtrs FFACoordFreqD::get_raw_ptrs() noexcept {
    const SizeType n = idx.size();
    return {.idx   = thrust::raw_pointer_cast(idx.data()),
            .shift = thrust::raw_pointer_cast(shift.data()),
            .size  = n};
}

void FFACoordFreqD::resize(SizeType n_coords) noexcept {
    idx.resize(n_coords);
    shift.resize(n_coords);
}

void FFACoordFreqD::copy_from_host(const std::vector<FFACoordFreq>& coords,
                                   SizeType n_coords,
                                   cudaStream_t stream) noexcept {
    std::vector<uint32_t> idx_h(n_coords);
    std::vector<float> shift_h(n_coords);

    for (SizeType i = 0; i < n_coords; ++i) {
        idx_h[i]   = coords[i].idx;
        shift_h[i] = coords[i].shift;
    }

    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(idx.data()), idx_h.data(),
                        n_coords * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        stream),
        "cudaMemcpyAsync idx failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(shift.data()), shift_h.data(),
                        n_coords * sizeof(float), cudaMemcpyHostToDevice,
                        stream),
        "cudaMemcpyAsync shift failed");
}
} // namespace loki::coord