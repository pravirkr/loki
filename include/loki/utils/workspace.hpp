#pragma once

#include <vector>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/world_tree.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::memory {

// Workspace containers are structs to reduce boilerplate code

/**
 * @brief Workspace for FFA buffers (can be reused across multiple FFA
 * instances).
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> struct FFAWorkspace {
    std::vector<FoldType> fold_internal;
    std::vector<coord::FFACoord> coords;
    std::vector<coord::FFACoordFreq> coords_freq;

    FFAWorkspace() = default;
    explicit FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan);
    FFAWorkspace(SizeType buffer_size, SizeType coord_size, SizeType n_params);

    ~FFAWorkspace() = default;

    FFAWorkspace(const FFAWorkspace&)                = delete;
    FFAWorkspace& operator=(const FFAWorkspace&)     = delete;
    FFAWorkspace(FFAWorkspace&&) noexcept            = default;
    FFAWorkspace& operator=(FFAWorkspace&&) noexcept = default;

    void validate(const plans::FFAPlan<FoldType>& ffa_plan) const;
};

/**
 * @brief Scratch space for Branch function in EP algorithm.
 *
 */
struct BranchingWorkspace {
    std::vector<double> scratch_params;
    std::vector<double> scratch_dparams;
    std::vector<SizeType> scratch_counts;

    BranchingWorkspace() = default;
    BranchingWorkspace(SizeType batch_size,
                       SizeType branch_max,
                       SizeType n_params);

    ~BranchingWorkspace() = default;

    BranchingWorkspace(const BranchingWorkspace&)                = delete;
    BranchingWorkspace& operator=(const BranchingWorkspace&)     = delete;
    BranchingWorkspace(BranchingWorkspace&&) noexcept            = default;
    BranchingWorkspace& operator=(BranchingWorkspace&&) noexcept = default;

    [[nodiscard]] float get_memory_usage() const noexcept;

    void
    validate(SizeType batch_size, SizeType branch_max, SizeType nparams) const;
};

/**
 * @brief Workspace for Prune buffers (can be reused across multiple Prune
 * instances).
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> struct PruneWorkspace {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size{};
    SizeType branch_max{};
    SizeType nparams{};
    SizeType nbins{};
    SizeType max_branched_leaves{};
    SizeType leaves_stride{};
    SizeType folds_stride{};

    std::vector<double> branched_leaves;
    std::vector<FoldType> branched_folds;
    std::vector<float> branched_scores;
    // Scratch space for indices
    std::vector<SizeType> branched_indices;
    // Scratch space for resolving parameters
    std::vector<SizeType> branched_param_idx;
    std::vector<float> branched_phase_shift;

    PruneWorkspace() = default;
    PruneWorkspace(SizeType batch_size,
                   SizeType branch_max,
                   SizeType nparams,
                   SizeType nbins);

    ~PruneWorkspace() = default;

    PruneWorkspace(const PruneWorkspace&)                = delete;
    PruneWorkspace& operator=(const PruneWorkspace&)     = delete;
    PruneWorkspace(PruneWorkspace&&) noexcept            = default;
    PruneWorkspace& operator=(PruneWorkspace&&) noexcept = default;

    [[nodiscard]] float get_memory_usage() const noexcept;

    void validate(SizeType batch_size, SizeType branch_max) const;
}; // End PruneWorkspace definition

/**
 * @brief Workspace for EPMultiPass buffers (can be reused across multiple Prune
 * instances or across repeated EPMultiPass::execute calls).
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> struct EPWorkspace {
    WorldTree<FoldType> world_tree;
    PruneWorkspace<FoldType> prune;
    BranchingWorkspace branch;

    std::vector<double> seed_leaves;
    std::vector<float> seed_scores;

    EPWorkspace() = default;
    EPWorkspace(SizeType batch_size,
                SizeType branch_max,
                SizeType max_sugg,
                SizeType ncoords_ffa,
                SizeType nparams,
                SizeType nbins);

    ~EPWorkspace() = default;
    // Non-copyable, non-movable: pass by reference only
    EPWorkspace(const EPWorkspace&)                = delete;
    EPWorkspace& operator=(const EPWorkspace&)     = delete;
    EPWorkspace(EPWorkspace&&) noexcept            = default;
    EPWorkspace& operator=(EPWorkspace&&) noexcept = default;

    [[nodiscard]] float get_memory_usage() const noexcept;

    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType max_sugg,
                  SizeType ncoords_ffa,
                  SizeType nparams,
                  SizeType nbins) const;
}; // End EPWorkspace definition

#ifdef LOKI_ENABLE_CUDA

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
    [[nodiscard]] float get_memory_usage() const noexcept;
    void
    validate(SizeType batch_size, SizeType branch_max, SizeType nparams) const;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct PruneWorkspaceCUDA {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size{};
    SizeType branch_max{};
    SizeType nparams{};
    SizeType nbins{};
    SizeType max_branched_leaves{};
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

    PruneWorkspaceCUDA() = default;
    PruneWorkspaceCUDA(SizeType batch_size,
                       SizeType branch_max,
                       SizeType nparams,
                       SizeType nbins);
    ~PruneWorkspaceCUDA() = default;

    PruneWorkspaceCUDA(const PruneWorkspaceCUDA&)                = delete;
    PruneWorkspaceCUDA& operator=(const PruneWorkspaceCUDA&)     = delete;
    PruneWorkspaceCUDA(PruneWorkspaceCUDA&&) noexcept            = default;
    PruneWorkspaceCUDA& operator=(PruneWorkspaceCUDA&&) noexcept = default;

    [[nodiscard]] float get_memory_usage() const noexcept;
    void validate(SizeType batch_size, SizeType branch_max) const;

}; // End PruneWorkspaceCUDA definition

struct CUBScratchArena {
    void* cub_temp_storage  = nullptr;
    SizeType cub_temp_bytes = 0;
    uint32_t* d_reduce_out  = nullptr;
    cudaStream_t stream     = nullptr;

    CUBScratchArena() = default;
    CUBScratchArena(SizeType batch_size,
                    SizeType branch_max,
                    cudaStream_t stream = nullptr);
    ~CUBScratchArena();
    // Non-copyable: device memory ownership is non-shared
    CUBScratchArena(const CUBScratchArena&)            = delete;
    CUBScratchArena& operator=(const CUBScratchArena&) = delete;
    // Movable: transfers ownership, poisons source
    CUBScratchArena(CUBScratchArena&&) noexcept;
    CUBScratchArena& operator=(CUBScratchArena&&) noexcept;

    [[nodiscard]] float get_memory_usage() const noexcept;

    void convert_mask_to_indices(cuda::std::span<const uint8_t> validation_mask,
                                 cuda::std::span<uint32_t> indices,
                                 SizeType n_leaves);
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct EPWorkspaceCUDA {
    WorldTreeCUDA<FoldTypeCUDA> world_tree;
    PruneWorkspaceCUDA<FoldTypeCUDA> prune;
    BranchingWorkspaceCUDA branch;
    CUBScratchArena scratch;

    thrust::device_vector<double> seed_leaves_d;
    thrust::device_vector<float> seed_scores_d;

    EPWorkspaceCUDA() = default;
    EPWorkspaceCUDA(SizeType batch_size,
                    SizeType branch_max,
                    SizeType max_sugg,
                    SizeType ncoords_ffa,
                    SizeType nparams,
                    SizeType nbins,
                    cudaStream_t stream = nullptr);

    ~EPWorkspaceCUDA();
    EPWorkspaceCUDA(const EPWorkspaceCUDA&)                = delete;
    EPWorkspaceCUDA& operator=(const EPWorkspaceCUDA&)     = delete;
    EPWorkspaceCUDA(EPWorkspaceCUDA&&) noexcept;
    EPWorkspaceCUDA& operator=(EPWorkspaceCUDA&&) noexcept;

    [[nodiscard]] float get_memory_usage() const noexcept;
    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType max_sugg,
                  SizeType ncoords_ffa,
                  SizeType nparams,
                  SizeType nbins) const;
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

#endif // LOKI_ENABLE_CUDA

} // namespace loki::memory