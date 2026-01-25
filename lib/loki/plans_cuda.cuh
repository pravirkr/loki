#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/plans.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/taylor_cuda.cuh"

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
                            stream),
            "cudaMemcpyAsync idx failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(shift.data()),
                            shift_h.data(), n_coords * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync shift failed");
    }
};

struct FFAPlanD {
    thrust::device_vector<double> params_d;
    // Host-side sizes for convenience
    std::vector<std::vector<SizeType>> param_sizes;

    thrust::device_vector<SizeType> ncoords;
    thrust::device_vector<SizeType> ncoords_offsets;
    double tseg_brute;
    SizeType nbins;
    SizeType n_levels;
    SizeType n_params;

    explicit FFAPlanD(const FFAPlanBase& cpu_plan)
        : ncoords(cpu_plan.get_ncoords()),
          ncoords_offsets(cpu_plan.get_ncoords_offsets()),
          tseg_brute(cpu_plan.get_config().get_tseg_brute()),
          nbins(cpu_plan.get_config().get_nbins()),
          n_levels(cpu_plan.get_n_levels()),
          n_params(cpu_plan.get_n_params()) {
        const auto& cpu_params = cpu_plan.get_params();

        params_storage.resize(n_levels);
        param_ptrs.resize(n_levels);
        param_sizes.resize(n_levels);

        for (SizeType i = 0; i < n_levels; ++i) {
            SizeType level_n_params = cpu_params[i].size();
            params_storage[i].resize(level_n_params);
            param_sizes[i].resize(level_n_params);

            std::vector<const double*> host_ptrs(level_n_params);

            for (SizeType p = 0; p < level_n_params; ++p) {
                // 1. Copy grid data to GPU
                params_storage[i][p] = cpu_params[i][p];

                // 2. Store pointer and size
                host_ptrs[p] =
                    thrust::raw_pointer_cast(params_storage[i][p].data());
                param_sizes[i][p] = cpu_params[i][p].size();
            }

            // 3. Copy pointers to device for this level
            param_ptrs[i] = host_ptrs;
        }
    }

    void resolve_coordinates(FFACoordD& coords_d, cudaStream_t stream) {
        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto ncoords_cur = ncoords[i_level];
            const auto offset      = ncoords_offsets[i_level];

            auto ptrs = coords_d.get_raw_ptrs(); // Helper to get raw ptrs

            // Resolve Tail (latter=0)
            core::ffa_taylor_resolve_poly_cuda(
                thrust::raw_pointer_cast(param_ptrs[i_level].data()),
                param_sizes[i_level].data(),
                thrust::raw_pointer_cast(param_ptrs[i_level - 1].data()),
                param_sizes[i_level - 1].data(), ptrs.i_tail + offset,
                ptrs.shift_tail + offset, ncoords_cur, n_params, i_level, 0,
                tseg_brute, nbins, stream);

            // Resolve Head (latter=1)
            core::ffa_taylor_resolve_poly_cuda(
                thrust::raw_pointer_cast(param_ptrs[i_level].data()),
                param_sizes[i_level].data(),
                thrust::raw_pointer_cast(param_ptrs[i_level - 1].data()),
                param_sizes[i_level - 1].data(), ptrs.i_head + offset,
                ptrs.shift_head + offset, ncoords_cur, n_params, i_level, 1,
                tseg_brute, nbins, stream);
        }
    }

    void resolve_coordinates_freq(FFACoordFreqD& coords_d,
                                  cudaStream_t stream) {
        if (n_params != 1)
            return;

        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto ncoords_cur = ncoords[i_level];
            const auto offset      = ncoords_offsets[i_level];

            auto ptrs = coords_d.get_raw_ptrs();

            // Frequency resolve is simpler (only 1 param)
            core::ffa_taylor_resolve_freq_cuda(
                thrust::raw_pointer_cast(params_storage[i_level][0].data()),
                param_sizes[i_level][0],
                thrust::raw_pointer_cast(params_storage[i_level - 1][0].data()),
                param_sizes[i_level - 1][0], ptrs.idx + offset,
                ptrs.shift + offset, i_level, tseg_brute, nbins, stream);
        }
    }
};

} // namespace loki::plans