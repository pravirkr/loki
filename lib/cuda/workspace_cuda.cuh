#pragma once

/**
 * @file workspace_cuda.cuh
 * @brief Concrete device workspaces and scratch arenas for CUDA FFA and EP
 * pruning. Internal.
 */

#include <cstdint>
#include <vector>

#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/world_tree_cuda.cuh"
#include "lib/utils/workspace_impl.hpp"

namespace loki::memory {

/**
 * @brief Workspace for CUDA FFA buffers (can be reused across multiple FFA
 * instances)
 *
 * @tparam FoldTypeCUDA Device fold type (float or ComplexTypeCUDA)
 */
template <SupportedFoldTypeCUDA FoldTypeCUDA> struct FFAWorkspaceCUDA {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;

    thrust::device_vector<DeviceFoldT> fold_internal_d;
    coord::FFACoordD coords_d;
    coord::FFACoordFreqD coords_freq_d;

    FFAWorkspaceCUDA() = default;
    explicit FFAWorkspaceCUDA(const plans::FFAPlan<HostFoldT>& ffa_plan);
    FFAWorkspaceCUDA(SizeType buffer_size,
                     SizeType coord_size,
                     SizeType n_levels,
                     SizeType n_params);

    ~FFAWorkspaceCUDA() = default;

    FFAWorkspaceCUDA(const FFAWorkspaceCUDA&)                = delete;
    FFAWorkspaceCUDA& operator=(const FFAWorkspaceCUDA&)     = delete;
    FFAWorkspaceCUDA(FFAWorkspaceCUDA&&) noexcept            = default;
    FFAWorkspaceCUDA& operator=(FFAWorkspaceCUDA&&) noexcept = default;

    void validate(const plans::FFAPlan<HostFoldT>& ffa_plan) const;
    void resolve_coordinates_freq(const plans::FFAPlan<HostFoldT>& ffa_plan,
                                  cudaStream_t stream);
    void resolve_coordinates(const plans::FFAPlan<HostFoldT>& ffa_plan,
                             cudaStream_t stream);

private:
    // Buffers for device resolve
    thrust::device_vector<uint32_t> m_param_counts_d;
    thrust::device_vector<uint32_t> m_ncoords_offsets_d;
    thrust::device_vector<ParamLimit> m_param_limits_d;

    void copy_plan_to_device(const plans::FFAPlan<HostFoldT>& ffa_plan,
                             cudaStream_t stream);
};

struct BranchingWorkspaceCUDAView {
    double* __restrict__ scratch_params;
    double* __restrict__ scratch_dparams;
    uint32_t* __restrict__ scratch_counts;
    uint32_t* __restrict__ leaf_branch_count;
    uint32_t* __restrict__ leaf_output_offset;
};

struct BranchingWorkspaceCUDA {
    thrust::device_vector<double> scratch_params;
    thrust::device_vector<double> scratch_dparams;
    thrust::device_vector<uint32_t> scratch_counts;
    thrust::device_vector<uint32_t> leaf_branch_count;
    thrust::device_vector<uint32_t> leaf_output_offset;

    BranchingWorkspaceCUDA() = default;
    BranchingWorkspaceCUDA(SizeType batch_size,
                           SizeType branch_max,
                           SizeType nparams);

    ~BranchingWorkspaceCUDA() = default;

    BranchingWorkspaceCUDA(const BranchingWorkspaceCUDA&)            = delete;
    BranchingWorkspaceCUDA& operator=(const BranchingWorkspaceCUDA&) = delete;
    BranchingWorkspaceCUDA(BranchingWorkspaceCUDA&&) noexcept        = default;
    BranchingWorkspaceCUDA&
    operator=(BranchingWorkspaceCUDA&&) noexcept = default;

    [[nodiscard]] BranchingWorkspaceCUDAView get_view() noexcept;
    [[nodiscard]] float get_memory_usage_gib() const noexcept;
    void
    validate(SizeType batch_size, SizeType branch_max, SizeType nparams) const;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct PruneWorkspaceCUDA {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size{};
    SizeType branch_max{};
    SizeType nparams{};
    SizeType nbins{};
    SizeType nsegments{};
    SizeType max_branched_leaves{};
    SizeType max_branched_param_idx{};
    SizeType leaves_stride{};
    SizeType folds_stride{};

    thrust::device_vector<double> branched_leaves_d;
    thrust::device_vector<FoldTypeCUDA> branched_folds_d;
    thrust::device_vector<float> branched_scores_d;
    // Scratch space for indices
    thrust::device_vector<uint32_t> branched_indices_d;
    // Scratch space for resolving parameters
    thrust::device_vector<uint32_t> branched_param_idx_d;
    thrust::device_vector<float> branched_phase_shift_d;
    // Scratch space for validation mask
    thrust::device_vector<uint8_t> validation_mask_d;
    thrust::device_vector<uint8_t> filtered_mask_d;

    PruneWorkspaceCUDA() = default;
    PruneWorkspaceCUDA(SizeType batch_size,
                       SizeType branch_max,
                       SizeType nparams,
                       SizeType nbins,
                       SizeType nsegments);
    ~PruneWorkspaceCUDA() = default;

    PruneWorkspaceCUDA(const PruneWorkspaceCUDA&)                = delete;
    PruneWorkspaceCUDA& operator=(const PruneWorkspaceCUDA&)     = delete;
    PruneWorkspaceCUDA(PruneWorkspaceCUDA&&) noexcept            = default;
    PruneWorkspaceCUDA& operator=(PruneWorkspaceCUDA&&) noexcept = default;

    [[nodiscard]] float get_memory_usage_gib() const noexcept;
    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType nsegments) const;

}; // End PruneWorkspaceCUDA definition

struct CUBScratchArena {
    void* cub_temp_storage    = nullptr;
    SizeType cub_temp_bytes   = 0;
    uint32_t* d_reduce_out    = nullptr;
    MinMaxFloat* d_minmax_out = nullptr;
    SizeType max_n_leaves     = 0;

    CUBScratchArena() = default;
    CUBScratchArena(SizeType batch_size,
                    SizeType branch_max,
                    cudaStream_t stream = nullptr);
    /// Synchronously frees all device allocations (stream-independent).
    ~CUBScratchArena();
    // Non-copyable: device memory ownership is non-shared
    CUBScratchArena(const CUBScratchArena&)            = delete;
    CUBScratchArena& operator=(const CUBScratchArena&) = delete;
    // Movable: transfers ownership, poisons source
    CUBScratchArena(CUBScratchArena&&) noexcept;
    CUBScratchArena& operator=(CUBScratchArena&&) noexcept;

    /// Returns the size of the CUB temp-storage allocation in gibibytes.
    [[nodiscard]] float get_memory_usage_gib() const noexcept;

    void convert_mask_to_indices(cuda::std::span<const uint8_t> validation_mask,
                                 cuda::std::span<uint32_t> indices,
                                 SizeType n_leaves,
                                 cudaStream_t stream);

    void compute_min_max_scores(cuda::std::span<const float> scores,
                                cuda::std::span<const uint8_t> mask,
                                MinMaxFloat* h_minmax_out,
                                SizeType n_leaves,
                                cudaStream_t stream);
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct EPWorkspaceCUDA {
    WorldTreeCUDA<FoldTypeCUDA> world_tree;
    PruneWorkspaceCUDA<FoldTypeCUDA> prune;
    BranchingWorkspaceCUDA branch;
    CUBScratchArena scratch;

    thrust::device_vector<double> seed_leaves_d;
    thrust::device_vector<float> seed_scores_d;
    // Device containers for get_segment_coords_so_far
    thrust::device_vector<uint32_t> idx_segments_d;
    thrust::device_vector<cuda::std::pair<double, double>> coord_segments_d;

    EPWorkspaceCUDA() = default;
    EPWorkspaceCUDA(SizeType batch_size,
                    SizeType branch_max,
                    SizeType max_sugg,
                    SizeType ncoords_ffa,
                    SizeType nparams,
                    SizeType nbins,
                    SizeType nsegments,
                    cudaStream_t stream = nullptr);

    ~EPWorkspaceCUDA();
    EPWorkspaceCUDA(const EPWorkspaceCUDA&)            = delete;
    EPWorkspaceCUDA& operator=(const EPWorkspaceCUDA&) = delete;
    EPWorkspaceCUDA(EPWorkspaceCUDA&&) noexcept;
    EPWorkspaceCUDA& operator=(EPWorkspaceCUDA&&) noexcept;

    [[nodiscard]] float get_memory_usage_gib() const noexcept;

    [[nodiscard]] float get_seed_memory_usage_gib() const noexcept;

    [[nodiscard]] float get_segment_coords_memory_usage_gib() const noexcept;

    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType max_sugg,
                  SizeType ncoords_ffa,
                  SizeType nparams,
                  SizeType nbins,
                  SizeType nsegments) const;
};

struct DeviceCounter {
    uint32_t* d_ptr = nullptr;
    uint32_t* h_ptr = nullptr; // pinned

    DeviceCounter();
    ~DeviceCounter();
    DeviceCounter(const DeviceCounter&)                      = delete;
    DeviceCounter& operator=(const DeviceCounter&)           = delete;
    DeviceCounter(DeviceCounter&& other) noexcept            = delete;
    DeviceCounter& operator=(DeviceCounter&& other) noexcept = delete;

    void reset(cudaStream_t stream);
    [[nodiscard]] uint32_t* data() noexcept { return d_ptr; } // NOLINT
    [[nodiscard]] const uint32_t* data() const noexcept { return d_ptr; }
    [[nodiscard]] uint32_t value_sync(cudaStream_t stream);
};

// ---------------------------------------------------------------------------
// GPU storage behind the public FFAWorkspace / EPWorkspace handles.
// ---------------------------------------------------------------------------

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class FFAWorkspaceCudaStorage final : public loki::detail::DeviceStorage {
public:
    template <typename... Args>
    explicit FFAWorkspaceCudaStorage(Args&&... args)
        : workspace(std::forward<Args>(args)...) {}

    FFAWorkspaceCUDA<FoldTypeCUDA> workspace;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class EPWorkspaceCudaStorage final : public loki::detail::DeviceStorage {
public:
    template <typename... Args>
    explicit EPWorkspaceCudaStorage(Args&&... args)
        : workspace(std::forward<Args>(args)...) {}

    [[nodiscard]] float get_memory_usage_gib() const noexcept override {
        return workspace.get_memory_usage_gib();
    }

    EPWorkspaceCUDA<FoldTypeCUDA> workspace;
};

namespace detail {

/// Device buffers behind @p ws. Throws if @p ws is not a GPU workspace.
template <SupportedFoldType FoldType>
FFAWorkspaceCUDA<CudaFoldType<FoldType>>&
cuda_workspace(FFAWorkspace<FoldType>& ws, std::string_view what) {
    auto& impl = loki::detail::HandleAccess::impl(ws);
    if (!impl.device) {
        throw std::invalid_argument(
            std::format("{}: expected a GPU FFAWorkspace", what));
    }
    return static_cast<FFAWorkspaceCudaStorage<CudaFoldType<FoldType>>&>(
               *impl.device)
        .workspace;
}

/// Device buffers behind @p ws. Throws if @p ws is not a GPU workspace.
template <SupportedFoldType FoldType>
EPWorkspaceCUDA<CudaFoldType<FoldType>>&
cuda_workspace(EPWorkspace<FoldType>& ws, std::string_view what) {
    auto& impl = loki::detail::HandleAccess::impl(ws);
    if (!impl.device) {
        throw std::invalid_argument(
            std::format("{}: expected a GPU EPWorkspace", what));
    }
    return static_cast<EPWorkspaceCudaStorage<CudaFoldType<FoldType>>&>(
               *impl.device)
        .workspace;
}

} // namespace detail

} // namespace loki::memory
