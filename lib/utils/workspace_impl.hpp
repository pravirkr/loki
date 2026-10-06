#pragma once

/**
 * @file workspace_impl.hpp
 * @brief Concrete host workspaces for FFA and EP pruning, and the Impl of the
 * public workspace handles. Internal.
 */

#include <memory>
#include <string_view>
#include <vector>

#include "common/dispatch.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"
#include "utils/world_tree.hpp"

namespace loki::memory {

// Workspace containers are structs to reduce boilerplate code

/**
 * @brief Workspace for FFA buffers (can be reused across multiple FFA
 * instances).
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> struct FFAWorkspaceCPU {
    std::vector<FoldType> fold_internal;
    std::vector<coord::FFACoord> coords;
    std::vector<coord::FFACoordFreq> coords_freq;

    FFAWorkspaceCPU() = default;
    explicit FFAWorkspaceCPU(const plans::FFAPlan<FoldType>& ffa_plan);
    FFAWorkspaceCPU(SizeType buffer_size,
                    SizeType coord_size,
                    SizeType n_params);

    ~FFAWorkspaceCPU() = default;

    FFAWorkspaceCPU(const FFAWorkspaceCPU&)                = delete;
    FFAWorkspaceCPU& operator=(const FFAWorkspaceCPU&)     = delete;
    FFAWorkspaceCPU(FFAWorkspaceCPU&&) noexcept            = default;
    FFAWorkspaceCPU& operator=(FFAWorkspaceCPU&&) noexcept = default;

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
    std::vector<double> scratch_shifts;

    BranchingWorkspace() = default;
    BranchingWorkspace(SizeType batch_size,
                       SizeType branch_max,
                       SizeType n_params);

    ~BranchingWorkspace() = default;

    BranchingWorkspace(const BranchingWorkspace&)                = delete;
    BranchingWorkspace& operator=(const BranchingWorkspace&)     = delete;
    BranchingWorkspace(BranchingWorkspace&&) noexcept            = default;
    BranchingWorkspace& operator=(BranchingWorkspace&&) noexcept = default;

    [[nodiscard]] float get_memory_usage_gib() const noexcept;

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
    SizeType nsegments{};
    SizeType max_branched_leaves{};
    SizeType max_branched_param_idx{};
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
    // Scratch space for the parent (tree) score of each branched leaf, used by
    // the stage-consistency veto.
    std::vector<float> branched_parent_scores;

    PruneWorkspace() = default;
    PruneWorkspace(SizeType batch_size,
                   SizeType branch_max,
                   SizeType nparams,
                   SizeType nbins,
                   SizeType nsegments);

    ~PruneWorkspace() = default;

    PruneWorkspace(const PruneWorkspace&)                = delete;
    PruneWorkspace& operator=(const PruneWorkspace&)     = delete;
    PruneWorkspace(PruneWorkspace&&) noexcept            = default;
    PruneWorkspace& operator=(PruneWorkspace&&) noexcept = default;

    [[nodiscard]] float get_memory_usage_gib() const noexcept;

    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType nsegments) const;
}; // End PruneWorkspace definition

/**
 * @brief Workspace for EPMultiPass buffers (can be reused across multiple Prune
 * instances or across repeated EPMultiPass::execute calls).
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> struct EPWorkspaceCPU {
    WorldTree<FoldType> world_tree;
    PruneWorkspace<FoldType> prune;
    BranchingWorkspace branch;

    std::vector<double> seed_leaves;
    std::vector<float> seed_scores;
    // Indices of the seeds surviving the pulsar mask (size ncoords_ffa).
    std::vector<SizeType> seed_keep_indices;

    EPWorkspaceCPU() = default;
    EPWorkspaceCPU(SizeType batch_size,
                   SizeType branch_max,
                   SizeType max_sugg,
                   SizeType ncoords_ffa,
                   SizeType nparams,
                   SizeType nbins,
                   SizeType nsegments);

    ~EPWorkspaceCPU() = default;
    // Non-copyable, non-movable: pass by reference only
    EPWorkspaceCPU(const EPWorkspaceCPU&)                = delete;
    EPWorkspaceCPU& operator=(const EPWorkspaceCPU&)     = delete;
    EPWorkspaceCPU(EPWorkspaceCPU&&) noexcept            = default;
    EPWorkspaceCPU& operator=(EPWorkspaceCPU&&) noexcept = default;

    [[nodiscard]] float get_memory_usage_gib() const noexcept;

    [[nodiscard]] float get_seed_memory_usage_gib() const noexcept;

    void validate(SizeType batch_size,
                  SizeType branch_max,
                  SizeType max_sugg,
                  SizeType ncoords_ffa,
                  SizeType nparams,
                  SizeType nbins,
                  SizeType nsegments) const;
}; // End EPWorkspaceCPU definition

// ---------------------------------------------------------------------------
// Public handle storage. Exactly one of `cpu` / `device` is set, matching
// `exec.backend`. The GPU storage types are defined in cuda/workspace_cuda.cuh.
// ---------------------------------------------------------------------------

template <SupportedFoldType FoldType> class FFAWorkspace<FoldType>::Impl {
public:
    Exec exec;
    std::unique_ptr<FFAWorkspaceCPU<FoldType>> cpu;
    std::unique_ptr<loki::detail::DeviceStorage> device;
};

template <SupportedFoldType FoldType> class EPWorkspace<FoldType>::Impl {
public:
    Exec exec;
    std::unique_ptr<EPWorkspaceCPU<FoldType>> cpu;
    std::unique_ptr<loki::detail::DeviceStorage> device;
};

namespace detail {

/// Host buffers behind @p ws. Throws if @p ws is empty or not a CPU workspace.
template <SupportedFoldType FoldType>
FFAWorkspaceCPU<FoldType>& cpu_workspace(FFAWorkspace<FoldType>& ws,
                                         std::string_view what);
template <SupportedFoldType FoldType>
EPWorkspaceCPU<FoldType>& cpu_workspace(EPWorkspace<FoldType>& ws,
                                        std::string_view what);

/// GPU storage factories, defined in cuda/workspace_cuda.cu.
template <SupportedFoldType FoldType>
std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu(const plans::FFAPlan<FoldType>& ffa_plan, int device_id);
template <SupportedFoldType FoldType>
std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu(SizeType buffer_size,
                       SizeType coord_size,
                       SizeType n_levels,
                       SizeType n_params,
                       int device_id);
template <SupportedFoldType FoldType>
std::unique_ptr<loki::detail::DeviceStorage>
make_ep_workspace_gpu(SizeType batch_size,
                      SizeType branch_max,
                      SizeType max_sugg,
                      SizeType ncoords_ffa,
                      SizeType nparams,
                      SizeType nbins,
                      SizeType nsegments,
                      int device_id);

} // namespace detail

} // namespace loki::memory
