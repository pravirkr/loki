#pragma once

/**
 * @file prune_engine.hpp
 * @brief Backend engine interface for EPMultiPass. Internal.
 */

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/workspace.hpp"

#include "lib/search/candidate_processing.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::algorithms::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

/**
 * @brief Workspace slots per parent leaf.
 *
 * Single source of truth shared by EPRegionPlanner and every EPMultiPass
 * engine: the planner must size workspaces with exactly the value the engine
 * validates against. `max_children` is the worst per-frequency child count
 * from the plan of the exact band being searched; it is not monotone in band
 * width. It bounds every parent of the tree, so it is the ceiling itself:
 * a parameter that branched past it is rejected by both backends.
 *
 * @param max_children Worst leaf from BranchingForecast::max_children.
 * @return max(max_children, 1).
 */
[[nodiscard]] inline SizeType compute_branch_max(SizeType max_children) {
    return std::max(max_children, SizeType{1});
}

/**
 * @brief Workers that prune at the same time in an EPMultiPass execution.
 *
 * Each pruning run (reference segment) is one task, so at most
 * min(nthreads, number of runs) workers are busy and need a workspace. Shared
 * by EPMultiPass (workspaces, thread pool), EPFreqSweep and the planner's
 * memory model. The run count is n_runs if set, else the number of ref_segs;
 * with neither (rejected later by determine_ref_segs) it is nthreads.
 *
 * @return A value in [1, max(nthreads, 1)].
 */
[[nodiscard]] inline SizeType
compute_ep_n_workers(int nthreads,
                     std::optional<SizeType> n_runs,
                     const std::optional<std::vector<SizeType>>& ref_segs) {
    const auto max_workers = static_cast<SizeType>(std::max(nthreads, 1));
    SizeType nruns         = max_workers;
    if (n_runs) {
        nruns = *n_runs;
    } else if (ref_segs) {
        nruns = ref_segs->size();
    }
    return std::clamp<SizeType>(nruns, 1, max_workers);
}

template <SupportedFoldType FoldType> class EPMultiPassEngine {
protected:
    EPMultiPassEngine() = default;

public:
    virtual ~EPMultiPassEngine() = default;

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix) = 0;

    /// Reduce each run's final leaves before writing them. Only the CPU
    /// engine of a candidate sweep supports it.
    virtual void
    set_leaf_reduction(std::optional<search::detail::LeafReductionSpec> spec) {
        if (spec) {
            throw std::logic_error("EP engine cannot reduce final leaves");
        }
    }

    /**
     * @brief Prune each run's seeds in @p n_slices sequential frequency
     * slices over the same FFA output (see search::SeedSlice), writing one
     * merged result per run. The tree then holds one slice at a time. More
     * slices than frequency cells are clamped; 1 (the default) prunes every
     * seed at once.
     * @throws std::invalid_argument if @p n_slices is 0.
     */
    virtual void set_seed_slices(SizeType n_slices) = 0;

    EPMultiPassEngine(const EPMultiPassEngine&)            = delete;
    EPMultiPassEngine& operator=(const EPMultiPassEngine&) = delete;
    EPMultiPassEngine(EPMultiPassEngine&&)                 = delete;
    EPMultiPassEngine& operator=(EPMultiPassEngine&&)      = delete;
};

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_cpu(search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            bool show_progress,
            PruneRFIConfig rfi_config);

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_cpu(std::span<memory::EPWorkspaceCPU<FoldType>* const> workspaces,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            bool show_progress,
            PruneRFIConfig rfi_config);

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_cpu(std::span<memory::EPWorkspaceCPU<FoldType>* const> workspaces,
            memory::FFAWorkspaceCPU<FoldType>& ffa_workspace,
            std::span<FoldType> ffa_fold,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            bool show_progress,
            PruneRFIConfig rfi_config);

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_gpu(search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            int device_id);

/// Runs on the caller's GPU workspace (a GPU handle on @p device_id; the
/// facade checks). The GPU engine is single-stream: one workspace per run.
template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_gpu(memory::EPWorkspace<FoldType>& workspace,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            int device_id);

} // namespace loki::algorithms::detail
