#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
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
                            stream),
            "cudaMemcpyAsync idx failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(shift.data()),
                            shift_h.data(), n_coords * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync shift failed");
    }
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct FFAPlanCUDA {
    using HostFoldType   = FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = FoldTypeTraits<FoldTypeCUDA>::DeviceType;

    std::unique_ptr<plans::FFAPlan<HostFoldType>> m_ffa_plan;
    thrust::device_vector<double> m_params_d;
    thrust::device_vector<uint32_t> m_param_counts_d;

    explicit FFAPlanCUDA(const plans::FFAPlan<HostFoldType>& ffa_plan)
        : m_ffa_plan(std::make_unique<plans::FFAPlan<HostFoldType>>(ffa_plan)) {
        m_params_d       = m_ffa_plan->get_params_flat();
        m_param_counts_d = m_ffa_plan->get_param_counts_flat();
    }

    void resolve_coordinates(FFACoordD& coords_d, cudaStream_t stream) {
        const auto& n_levels        = m_ffa_plan->get_n_levels();
        const auto& n_params        = m_ffa_plan->get_n_params();
        const auto& ncoords         = m_ffa_plan->get_ncoords();
        const auto& ncoords_offsets = m_ffa_plan->get_ncoords_offsets();
        const auto& tseg_brute      = m_ffa_plan->get_config().get_tseg_brute();
        const auto& nbins           = m_ffa_plan->get_config().get_nbins();
        const auto& [params_flat_offsets, params_flat_sizes] =
            m_ffa_plan->get_params_flat_sizes();

        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto ncoords_cur = ncoords[i_level];
            const auto offset      = ncoords_offsets[i_level];

            auto coord_ptrs = coords_d.get_raw_ptrs();
            // Tail coordinates
            core::ffa_taylor_resolve_poly_batch_cuda(
                cuda_utils::as_span(m_params_d)
                    .subspan(params_flat_offsets[i_level],
                             params_flat_sizes[i_level]),
                cuda_utils::as_span(m_param_counts_d)
                    .subspan(i_level * n_params, n_params),
                cuda_utils::as_span(m_params_d)
                    .subspan(params_flat_offsets[i_level - 1],
                             params_flat_sizes[i_level - 1]),
                cuda_utils::as_span(m_param_counts_d)
                    .subspan((i_level - 1) * n_params, n_params),
                cuda::std::span<uint32_t>(coord_ptrs.i_tail + offset,
                                          ncoords_cur),
                cuda::std::span<float>(coord_ptrs.shift_tail + offset,
                                       ncoords_cur),
                i_level, 0, tseg_brute, nbins, n_params, stream);

            // Head coordinates
            core::ffa_taylor_resolve_poly_batch_cuda(
                cuda_utils::as_span(m_params_d)
                    .subspan(params_flat_offsets[i_level],
                             params_flat_sizes[i_level]),
                cuda_utils::as_span(m_param_counts_d)
                    .subspan(i_level * n_params, n_params),
                cuda_utils::as_span(m_params_d)
                    .subspan(params_flat_offsets[i_level - 1],
                             params_flat_sizes[i_level - 1]),
                cuda_utils::as_span(m_param_counts_d)
                    .subspan((i_level - 1) * n_params, n_params),
                cuda::std::span<uint32_t>(coord_ptrs.i_head + offset,
                                          ncoords_cur),
                cuda::std::span<float>(coord_ptrs.shift_head + offset,
                                       ncoords_cur),
                i_level, 1, tseg_brute, nbins, n_params, stream);
        }
    }

    void resolve_coordinates_freq(FFACoordFreqD& coords_d,
                                  cudaStream_t stream) {
        const auto& n_levels        = m_ffa_plan->get_n_levels();
        const auto& n_params        = m_ffa_plan->get_n_params();
        const auto& ncoords         = m_ffa_plan->get_ncoords();
        const auto& ncoords_offsets = m_ffa_plan->get_ncoords_offsets();
        const auto& tseg_brute      = m_ffa_plan->get_config().get_tseg_brute();
        const auto& nbins           = m_ffa_plan->get_config().get_nbins();
        const auto& [params_flat_offsets, params_flat_sizes] =
            m_ffa_plan->get_params_flat_sizes();

        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto ncoords_cur = ncoords[i_level];
            const auto offset      = ncoords_offsets[i_level];

            auto coord_ptrs = coords_d.get_raw_ptrs();
            core::ffa_taylor_resolve_poly_batch_cuda(
                cuda_utils::as_span(m_params_d)
                    .subspan(params_flat_offsets[i_level],
                             params_flat_sizes[i_level]),
                cuda_utils::as_span(m_param_counts_d)
                    .subspan(i_level * n_params, n_params),
                cuda_utils::as_span(m_params_d)
                    .subspan(params_flat_offsets[i_level - 1],
                             params_flat_sizes[i_level - 1]),
                cuda_utils::as_span(m_param_counts_d)
                    .subspan((i_level - 1) * n_params, n_params),
                cuda::std::span<uint32_t>(coord_ptrs.idx + offset, ncoords_cur),
                cuda::std::span<float>(coord_ptrs.shift + offset, ncoords_cur),
                i_level, 0, tseg_brute, nbins, n_params, stream);
        }
    }
};

} // namespace loki::plans