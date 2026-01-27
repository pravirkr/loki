#pragma once

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::coord {

// FFA Coordinate plan for a single param coordinate in a single iteration
struct FFACoord {
    uint32_t i_tail;  // Tail coordinate index in the previous iteration
    float shift_tail; // Phase bin shift in the tail coordinate
    uint32_t i_head;  // Head coordinate index in the previous iteration
    float shift_head; // Phase bin shift in the head coordinate
};

struct FFACoordFreq {
    uint32_t idx; // Phase bin index in the previous iteration
    float shift;  // Phase bin shift
};

// A structure to hold the parameters for a single FFA search region.
struct FFARegion {
    double f_start; // Hz, inclusive (lower frequency)
    double f_end;   // Hz, inclusive (upper frequency)
    SizeType nbins; // fixed within region
    double eta;     // tolerance in bins for this region
};

// A structure to hold the stats for a single FFA search chunk.
struct FFAChunkStats {
    double nominal_f_start;
    double nominal_f_end;
    double actual_f_start;
    double actual_f_end;
    double nominal_width;
    double actual_width;
    double total_memory_gb;
    double overlap_fraction; // fraction of actual range that's overlap
};

#ifdef LOKI_ENABLE_CUDA

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

#endif // LOKI_ENABLE_CUDA

} // namespace loki::coord