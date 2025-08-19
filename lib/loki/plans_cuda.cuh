#pragma once

#include <vector>

#include <thrust/device_vector.h>

#include "loki/algorithms/plans.hpp"

namespace loki::plans {

struct FFACoordDPtrs {
    const int* i_tail;
    const float* shift_tail;
    const int* i_head;
    const float* shift_head;

    __host__ __device__ void update_offsets(int offset_value) {
        i_tail += offset_value;
        shift_tail += offset_value;
        i_head += offset_value;
        shift_head += offset_value;
    }
};

struct FFACoordFreqDPtrs {
    const int* idx;
    const float* shift;

    __host__ __device__ void update_offsets(int offset_value) {
        idx += offset_value;
        shift += offset_value;
    }
};

struct FFACoordD {
    thrust::device_vector<int> i_tail;
    thrust::device_vector<float> shift_tail;
    thrust::device_vector<int> i_head;
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

    __host__ void
    copy_from_host(const std::vector<std::vector<plans::FFACoord>>& coords,
                   SizeType n_coords) {
        std::vector<int> i_tail_h, i_head_h;
        std::vector<double> shift_tail_h, shift_head_h;
        i_tail.reserve(n_coords);
        shift_tail_h.reserve(n_coords);
        i_head_h.reserve(n_coords);
        shift_head_h.reserve(n_coords);
        for (const auto& host_coords_iter : coords) {
            for (const auto& coord : host_coords_iter) {
                i_tail_h.emplace_back(static_cast<int>(coord.i_tail));
                shift_tail_h.emplace_back(coord.shift_tail);
                i_head_h.emplace_back(static_cast<int>(coord.i_head));
                shift_head_h.emplace_back(coord.shift_head);
            }
        }
        // thrust copy from host to device
        thrust::copy(i_tail_h.begin(), i_tail_h.end(), i_tail.begin());
        thrust::copy(shift_tail_h.begin(), shift_tail_h.end(),
                     shift_tail.begin());
        thrust::copy(i_head_h.begin(), i_head_h.end(), i_head.begin());
        thrust::copy(shift_head_h.begin(), shift_head_h.end(),
                     shift_head.begin());
    }
};

struct FFACoordFreqD {
    thrust::device_vector<int> idx;
    thrust::device_vector<float> shift;

    __host__ FFACoordFreqDPtrs get_raw_ptrs() const {
        return {.idx   = thrust::raw_pointer_cast(idx.data()),
                .shift = thrust::raw_pointer_cast(shift.data())};
    }

    __host__ void resize(SizeType n_coords) {
        idx.resize(n_coords);
        shift.resize(n_coords);
    }

    __host__ void
    copy_from_host(const std::vector<std::vector<plans::FFACoordFreq>>& coords,
                   SizeType n_coords) {
        std::vector<int> idx_h;
        std::vector<float> shift_h;
        idx_h.reserve(n_coords);
        shift_h.reserve(n_coords);
        for (const auto& host_coords_iter : coords) {
            for (const auto& coord : host_coords_iter) {
                idx_h.emplace_back(static_cast<int>(coord.idx));
                shift_h.emplace_back(coord.shift);
            }
        }
        // thrust copy from host to device
        thrust::copy(idx_h.begin(), idx_h.end(), idx.begin());
        thrust::copy(shift_h.begin(), shift_h.end(), shift.begin());
    }
};

} // namespace loki::plans