#pragma once

#include <vector>

#include <thrust/device_vector.h>

#include "loki/algorithms/plans.hpp"

namespace loki::plans {

struct FFACoordDPtrs {
    const int* i_tail;
    const int* shift_tail;
    const int* i_head;
    const int* shift_head;

    __host__ __device__ void update_offsets(int offset_value) {
        i_tail += offset_value;
        shift_tail += offset_value;
        i_head += offset_value;
        shift_head += offset_value;
    }
};

struct FFACoordD {
    thrust::device_vector<int> i_tail;
    thrust::device_vector<int> shift_tail;
    thrust::device_vector<int> i_head;
    thrust::device_vector<int> shift_head;

    __host__ FFACoordDPtrs get_raw_ptrs() const {
        return {.i_tail     = thrust::raw_pointer_cast(i_tail.data()),
                .shift_tail = thrust::raw_pointer_cast(shift_tail.data()),
                .i_head     = thrust::raw_pointer_cast(i_head.data()),
                .shift_head = thrust::raw_pointer_cast(shift_head.data())};
    }
};

struct FFAPlanD {
    thrust::device_vector<int> nsegments;
    thrust::device_vector<double> freqs_arr_start;
    FFACoordD coordinates;
};

namespace {
inline void transfer_ffa_plan_to_device(const FFAPlan& plan, FFAPlanD& plan_d) {
    const auto levels = static_cast<int>(plan.fold_shapes.size());
    // Transfer shape to device
    plan_d.nsegments.resize(levels);
    for (int i = 0; i < levels; ++i) {
        plan_d.nsegments[i] = static_cast<int>(plan.fold_shapes[i][0]);
    }
    plan_d.freqs_arr_start = plan.params[0].back();

    std::vector<int> i_tail, shift_tail, i_head, shift_head;
    // Calculate total size for efficient allocation
    size_t total_size = 0;
    for (const auto& host_coords_iter : plan.coordinates) {
        total_size += host_coords_iter.size();
    }
    i_tail.resize(total_size);
    shift_tail.resize(total_size);
    i_head.resize(total_size);
    shift_head.resize(total_size);
    for (const auto& host_coords_iter : plan.coordinates) {
        for (const auto& coord : host_coords_iter) {
            i_tail.emplace_back(coord.i_tail);
            shift_tail.emplace_back(coord.shift_tail);
            i_head.emplace_back(coord.i_head);
            shift_head.emplace_back(coord.shift_head);
        }
    }
    plan_d.coordinates.i_tail     = i_tail;
    plan_d.coordinates.shift_tail = shift_tail;
    plan_d.coordinates.i_head     = i_head;
    plan_d.coordinates.shift_head = shift_head;
}
} // namespace
} // namespace loki::plans