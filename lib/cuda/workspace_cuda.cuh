#pragma once

/**
 * @file workspace_cuda.cuh
 * @brief Concrete device workspaces and scratch arenas for CUDA FFA and EP
 * pruning. Internal.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/taylor_ffa_cuda.cuh"
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

    // Every FFA level writes its whole output, so it needs no fill.
    cuda_utils::DeviceVectorNoInit<DeviceFoldT> fold_internal_d;
    cuda::std::span<DeviceFoldT> shared_fold_internal;
    cuda::std::span<DeviceFoldT> internal_folds() noexcept {
        return shared_fold_internal.empty()
                   ? cuda_utils::as_span(fold_internal_d)
                   : shared_fold_internal;
    }
    cuda::std::span<const DeviceFoldT> internal_folds() const noexcept {
        return shared_fold_internal.empty()
                   ? cuda_utils::as_span(fold_internal_d)
                   : shared_fold_internal;
    }
    coord::FFACoordD coords_d;
    coord::FFACoordFreqD coords_freq_d;
    // Brute-fold tables of the frequency sweep (grown on demand): the
    // level-0 frequencies and the time-domain phase-bin map.
    cuda_utils::DeviceVectorNoInit<double> bf_freqs_d;
    cuda_utils::DeviceVectorNoInit<uint32_t> bf_phase_map_d;

    /**
     * @brief Pinned staging for the small per-chunk uploads of a pipelined
     * sweep. Copies from pageable memory can wait for earlier work on the
     * stream; from pinned memory they are truly asynchronous. The sweep
     * alternates two slots and selects one per chunk; a slot is reused only
     * after its previous chunk finished, so pending copies never see their
     * source overwritten. With no slot selected, uploads stay pageable.
     */
    void select_staging_slot(int slot);
    /// Copy @p bytes into the selected slot and return the pinned copy
    /// (nullptr when no slot is selected).
    [[nodiscard]] const void* stage(const void* src, SizeType bytes);

    FFAWorkspaceCUDA() = default;
    explicit FFAWorkspaceCUDA(const plans::FFAPlan<HostFoldT>& ffa_plan);
    FFAWorkspaceCUDA(SizeType buffer_size,
                     SizeType coord_size,
                     SizeType n_levels,
                     SizeType n_params,
                     bool allocate_internal = true);

    FFAWorkspaceCUDA(const FFAWorkspaceCUDA&)                = delete;
    FFAWorkspaceCUDA& operator=(const FFAWorkspaceCUDA&)     = delete;
    FFAWorkspaceCUDA(FFAWorkspaceCUDA&&) noexcept            = default;
    FFAWorkspaceCUDA& operator=(FFAWorkspaceCUDA&&) noexcept = default;

    void validate(const plans::FFAPlan<HostFoldT>& ffa_plan) const;
    // Same, with the ping-pong buffer size and the coordinate count the
    // caller actually needs.
    void validate(const plans::FFAPlan<HostFoldT>& ffa_plan,
                  SizeType buffer_size,
                  SizeType coord_size = SIZE_MAX) const;
    /// Bytes of the large buffers: the internal fold and the coordinates.
    [[nodiscard]] SizeType get_buffers_bytes() const noexcept;
    /// Bytes of every buffer, including the small per-level arrays.
    [[nodiscard]] SizeType get_memory_usage_bytes() const noexcept;
    void resolve_coordinates_freq(const plans::FFAPlan<HostFoldT>& ffa_plan,
                                  cudaStream_t stream);
    /// @p skip_final_accel: accel plans leave the final level unresolved
    /// (the fused sweep resolves it on the fly; see accel_resolve_level).
    void resolve_coordinates(const plans::FFAPlan<HostFoldT>& ffa_plan,
                             cudaStream_t stream,
                             bool skip_final_accel = false);
    /// Device view of one level's accel resolve tables, valid after
    /// resolve_coordinates of the same plan.
    [[nodiscard]] core::FFAAccelResolveLevel
    accel_resolve_level(const plans::FFAPlan<HostFoldT>& ffa_plan,
                        SizeType level) const;
    /// Device Bluestein table of the fused Fourier scorer for @p nbins
    /// (core::ffa_fourier_bluestein_table), built and uploaded on first use
    /// and kept for the workspace's lifetime (a sweep has few nbins).
    [[nodiscard]] const ComplexTypeCUDA* bluestein_table(SizeType nbins);
    /// Per level, the number of resolved frequency coordinates that break
    /// the fused-merge contract (see ffa_freq_fuse_check_levels_cuda). Call
    /// after resolve_coordinates_freq(). Synchronizes @p stream.
    [[nodiscard]] std::vector<uint32_t>
    check_fuse_levels_freq(const plans::FFAPlan<HostFoldT>& ffa_plan,
                           cudaStream_t stream);

private:
    // Buffers for device resolve
    thrust::device_vector<uint32_t> m_param_counts_d;
    thrust::device_vector<uint32_t> m_ncoords_offsets_d;
    thrust::device_vector<ParamLimit> m_param_limits_d;
    // Scratch for check_fuse_levels_freq (3 * n_levels, frequency-only).
    thrust::device_vector<uint32_t> m_fuse_check_d;
    // Host sources of the async resolve copies; members so they outlive
    // the copies without a stream sync.
    std::vector<uint32_t> m_param_counts_h;
    std::vector<uint32_t> m_ncoords_offsets_h;
    std::vector<ParamLimit> m_param_limits_h;
    // Accel resolve tables (grown on demand; small) and their offsets.
    thrust::device_vector<core::FFAAccelResolveRow> m_accel_rows_d;
    thrust::device_vector<double> m_accel_cols_d;
    thrust::device_vector<uint32_t> m_accel_offsets_d;
    std::vector<uint32_t> m_accel_offsets_h;
    std::map<SizeType, cuda_utils::DeviceVectorNoInit<ComplexTypeCUDA>>
        m_bluestein_d;

    struct PinnedFree {
        void operator()(std::byte* ptr) const noexcept {
            if (ptr != nullptr) {
                cudaFreeHost(ptr);
            }
        }
    };
    struct PinnedStaging {
        std::unique_ptr<std::byte, PinnedFree> ptr;
        SizeType capacity{0};
        SizeType used{0};
    };
    std::array<PinnedStaging, 2> m_staging{};
    int m_staging_slot{-1};
    // Upload @p bytes from @p src: through the selected pinned slot, or
    // directly (pageable) when none is selected.
    void
    upload(void* dst_d, const void* src, SizeType bytes, cudaStream_t stream);

    void copy_plan_to_device(const plans::FFAPlan<HostFoldT>& ffa_plan,
                             cudaStream_t stream);
    // Accel (n_params = 2) resolve through per-level row / column tables.
    void resolve_coordinates_accel(const plans::FFAPlan<HostFoldT>& ffa_plan,
                                   cudaStream_t stream,
                                   bool skip_final);
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
    [[nodiscard]] SizeType get_memory_usage_bytes() const noexcept;
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
    thrust::device_vector<uint32_t> passing_indices_d;
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

    [[nodiscard]] SizeType get_memory_usage_bytes() const noexcept;
    [[nodiscard]] float get_memory_usage_gib() const noexcept;
    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType nsegments) const;

}; // End PruneWorkspaceCUDA definition

struct CUBScratchArena {
    /// Pinned host slots for the scalars the host reads back each batch.
    enum HostSlot : std::uint8_t {
        kBranchLastOffset = 0,
        kBranchLastCount  = 1,
        kReduceOut        = 2,
        kNumHostSlots     = 4,
    };

    void* cub_temp_storage    = nullptr;
    SizeType cub_temp_bytes   = 0;
    uint32_t* d_reduce_out    = nullptr;
    MinMaxFloat* d_minmax_out = nullptr;
    SizeType max_n_leaves     = 0;
    // Pinned host memory (not device memory, so not in the memory model).
    uint32_t* h_scalars      = nullptr;
    MinMaxFloat* h_minmax    = nullptr;
    cudaEvent_t minmax_ready = nullptr;

    CUBScratchArena() = default;
    CUBScratchArena(SizeType batch_size,
                    SizeType branch_max,
                    cudaStream_t stream = nullptr);
    /// Bytes of CUB temporary storage an arena for @p max_n_leaves leaves
    /// allocates (the largest need of the four operations). Sizing only:
    /// allocates nothing. The EP memory model queries it.
    [[nodiscard]] static SizeType temp_bytes_for(SizeType max_n_leaves,
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
    [[nodiscard]] SizeType get_memory_usage_bytes() const noexcept;
    [[nodiscard]] float get_memory_usage_gib() const noexcept;

    void convert_mask_to_indices(cuda::std::span<const uint8_t> validation_mask,
                                 cuda::std::span<uint32_t> indices,
                                 SizeType n_leaves,
                                 cudaStream_t stream);

    /// Enqueues the masked min/max of @p scores and its copy to the host,
    /// without waiting. Read it with wait_min_max() before the next call.
    void compute_min_max_scores_async(cuda::std::span<const float> scores,
                                      cuda::std::span<const uint8_t> mask,
                                      SizeType n_leaves,
                                      cudaStream_t stream);
    /// Waits for the last compute_min_max_scores_async() and returns it.
    [[nodiscard]] MinMaxFloat wait_min_max() const;

private:
    void release() noexcept;
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
                    cudaStream_t stream                        = nullptr,
                    cuda::std::span<FoldTypeCUDA> shared_folds = {});

    ~EPWorkspaceCUDA();
    EPWorkspaceCUDA(const EPWorkspaceCUDA&)            = delete;
    EPWorkspaceCUDA& operator=(const EPWorkspaceCUDA&) = delete;
    EPWorkspaceCUDA(EPWorkspaceCUDA&&) noexcept;
    EPWorkspaceCUDA& operator=(EPWorkspaceCUDA&&) noexcept;

    [[nodiscard]] SizeType get_memory_usage_bytes() const noexcept;
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
