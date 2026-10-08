#pragma once

/**
 * @file ep_memory.hpp
 * @brief Host memory model of EPFreqSweep, shared by EPRegionPlanner, the
 * sweep's runtime checks and the tests that compare it with the real
 * allocations. Internal. See docs/memory.md.
 *
 * EPFreqSweep holds three kinds of memory:
 *  - per-worker buffers: the EP workspace (world tree, prune/branch scratch,
 *    FFA seeds), the irfft scratch for Fourier folds and, when harvesting is
 *    enabled, the harvest store. They are sized once per contiguous run of
 *    chunks with the same nbins, from the maxima of that run, and freed before
 *    the next run;
 *  - the FFA workspace (internal fold buffer and coordinates) and the output
 *    fold buffer, shared by every chunk and sized from the global maxima;
 *  - the input time series (ts_e, ts_v), resident for the whole sweep whoever
 *    owns them.
 *
 * The peak is therefore, for the worst run:
 *   n_workers * per-worker bytes of the run + shared bytes + input bytes.
 */

#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms::detail {

constexpr double kBytesPerGiB = static_cast<double>(1ULL << 30U);

/// Batch size EPFreqSweep runs the pruning with.
constexpr SizeType kEPBatchSize = 1024U;

/// Version of this model. Bump it with every change to the formulas: plan
/// caches store it and are rejected when it differs.
constexpr std::uint32_t kEPMemoryModelVersion = 2U;

/// Worst-case size of the per-run harvest store (see PruneRFIConfig).
/// Normalised: all fields are zero/false when harvesting is disabled.
struct EPHarvestBound {
    bool enabled{false};
    SizeType max_harvests{0};
    bool store_folds{false};

    [[nodiscard]] static EPHarvestBound
    from(const PruneRFIConfig& rfi) noexcept {
        if (!rfi.has_harvest()) {
            return {};
        }
        return {.enabled      = true,
                .max_harvests = rfi.max_harvests,
                .store_folds  = rfi.harvest_store_folds};
    }
};

/// Sweep-wide inputs of the model.
struct EPMemoryContext {
    SizeType nparams{0};
    SizeType nsamps{0};
    /// Workers pruning at the same time, each with its own buffers.
    int n_workers{1};
    SizeType batch_size{kEPBatchSize};
    EPHarvestBound harvest;
};

/// Size of one element of the folds stored for @p FoldType.
template <SupportedFoldType FoldType> constexpr SizeType fold_bytes() noexcept {
    return std::is_same_v<FoldType, ComplexType> ? sizeof(ComplexType)
                                                 : sizeof(float);
}

/// Bytes of one EPWorkspaceCPU (world tree, prune, branch, seeds).
template <SupportedFoldType FoldType>
SizeType ep_workspace_bytes(SizeType nparams,
                            SizeType nbins,
                            SizeType nsegments,
                            SizeType ncoords_ffa,
                            SizeType max_sugg,
                            SizeType branch_max,
                            SizeType batch_size) {
    constexpr bool kIsComplex       = std::is_same_v<FoldType, ComplexType>;
    constexpr SizeType kParamStride = 2U;
    const SizeType nbins_arg        = kIsComplex ? (nbins / 2) + 1 : nbins;
    const SizeType leaves_stride    = (nparams + 2) * kParamStride;
    const SizeType max_batch_size   = batch_size * branch_max;

    // WorldTree
    const SizeType world_tree_bytes =
        (max_sugg * leaves_stride * sizeof(double)) +
        (max_sugg * 2 * nbins_arg * fold_bytes<FoldType>()) +
        (max_sugg * 2 * sizeof(float)) +
        ((max_sugg + max_batch_size) * sizeof(float)) +
        (max_batch_size * sizeof(SizeType)) + (max_sugg * sizeof(uint8_t));

    // PruneWorkspace
    const SizeType max_branched_param_idx =
        std::max(max_batch_size, nsegments * batch_size);
    const SizeType prune_bytes =
        (max_batch_size * leaves_stride * sizeof(double)) +
        (max_batch_size * 2 * nbins_arg * fold_bytes<FoldType>()) +
        (max_batch_size * sizeof(float)) + (max_batch_size * sizeof(SizeType)) +
        (max_branched_param_idx * sizeof(SizeType)) +
        (max_branched_param_idx * sizeof(float)) +
        (max_batch_size * sizeof(float));

    // BranchingWorkspace
    const SizeType branch_bytes =
        (batch_size * nparams * branch_max * sizeof(double)) +
        (batch_size * nparams * sizeof(double)) +
        (batch_size * nparams * sizeof(SizeType)) +
        (batch_size * nparams * sizeof(double));

    // Seeds
    const SizeType seed_bytes = (ncoords_ffa * leaves_stride * sizeof(double)) +
                                (ncoords_ffa * sizeof(float)) +
                                (ncoords_ffa * sizeof(SizeType));

    return world_tree_bytes + prune_bytes + branch_bytes + seed_bytes;
}

/// Bytes of the irfft scratch each pruning thread owns (Fourier folds only).
template <SupportedFoldType FoldType>
SizeType ep_irfft_scratch_bytes(SizeType nbins,
                                SizeType ncoords_ffa,
                                SizeType branch_max,
                                SizeType batch_size) {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        const SizeType max_nfft =
            std::max(2 * batch_size * branch_max, 2 * ncoords_ffa);
        return (max_nfft * ((nbins / 2) + 1) * sizeof(ComplexType)) +
               (max_nfft * nbins * sizeof(float));
    } else {
        return 0;
    }
}

/// Bytes of one run's harvest store once it reserved max_harvests records
/// (HarvestBuffer, as PruneImpl fills it). Zero when harvesting is off.
template <SupportedFoldType FoldType>
SizeType ep_harvest_bytes(SizeType nparams,
                          SizeType nbins,
                          const EPHarvestBound& harvest) noexcept {
    if (!harvest.enabled || harvest.max_harvests == 0) {
        return 0;
    }
    constexpr bool kIsComplex       = std::is_same_v<FoldType, ComplexType>;
    constexpr SizeType kParamStride = 2U;
    const SizeType nbins_arg        = kIsComplex ? (nbins / 2) + 1 : nbins;
    const SizeType leaves_stride    = (nparams + 2) * kParamStride;
    const SizeType fold_record =
        harvest.store_folds ? 2 * nbins_arg * fold_bytes<FoldType>() : 0;
    // leaf, fold, score, level, seg_idx, t_ref
    const SizeType record_bytes = (leaves_stride * sizeof(double)) +
                                  fold_record + sizeof(float) +
                                  (2 * sizeof(SizeType)) + sizeof(double);
    return harvest.max_harvests * record_bytes;
}

/// Bytes each worker needs for one chunk group.
template <SupportedFoldType FoldType>
SizeType ep_thread_bytes(const EPMemoryContext& ctx,
                         SizeType nbins,
                         SizeType nsegments,
                         SizeType ncoords_ffa,
                         SizeType max_sugg,
                         SizeType branch_max) {
    return ep_workspace_bytes<FoldType>(ctx.nparams, nbins, nsegments,
                                        ncoords_ffa, max_sugg, branch_max,
                                        ctx.batch_size) +
           ep_irfft_scratch_bytes<FoldType>(nbins, ncoords_ffa, branch_max,
                                            ctx.batch_size) +
           ep_harvest_bytes<FoldType>(ctx.nparams, nbins, ctx.harvest);
}

/// Bytes shared by all chunks: the FFA workspace (internal fold buffer and
/// coordinates) plus the output fold buffer. Both hold buffer_size folds.
template <SupportedFoldType FoldType>
SizeType
ep_shared_bytes(SizeType nparams, SizeType buffer_size, SizeType coord_size) {
    const SizeType coord_unit_bytes =
        (nparams == 1) ? sizeof(coord::FFACoordFreq) : sizeof(coord::FFACoord);
    return (2 * buffer_size * fold_bytes<FoldType>()) +
           (coord_size * coord_unit_bytes);
}

/// Bytes of the input time series (ts_e and ts_v, float32).
constexpr SizeType ep_input_bytes(SizeType nsamps) noexcept {
    return 2 * nsamps * sizeof(float);
}

/// Bytes resident for the whole sweep: shared FFA buffers plus the inputs.
template <SupportedFoldType FoldType>
SizeType ep_fixed_bytes(const EPMemoryContext& ctx,
                        SizeType buffer_size,
                        SizeType coord_size) {
    return ep_shared_bytes<FoldType>(ctx.nparams, buffer_size, coord_size) +
           ep_input_bytes(ctx.nsamps);
}

[[nodiscard]] inline double ep_total_gb(int n_workers,
                                        SizeType thread_bytes,
                                        SizeType fixed_bytes) noexcept {
    const auto total =
        (static_cast<double>(n_workers) * static_cast<double>(thread_bytes)) +
        static_cast<double>(fixed_bytes);
    return total / kBytesPerGiB;
}

/// A contiguous run of chunks with the same nbins and its maxima: one
/// allocation of the per-worker buffers in EPFreqSweep.
struct EPChunkGroup {
    SizeType begin{0}; ///< First chunk index.
    SizeType end{0};   ///< One past the last chunk index.
    SizeType nbins{0};
    SizeType max_sugg{0};
    SizeType branch_max{0};
    SizeType ncoords{0};
    SizeType nsegments{0};
};

/// Groups @p chunks into contiguous runs of equal nbins, as EPFreqSweep does.
inline std::vector<EPChunkGroup>
ep_chunk_groups(std::span<const EPChunkConfig> chunks) {
    std::vector<EPChunkGroup> groups;
    for (SizeType i = 0; i < chunks.size();) {
        EPChunkGroup g{.begin = i, .nbins = chunks[i].cfg.get_nbins()};
        for (; i < chunks.size() && chunks[i].cfg.get_nbins() == g.nbins; ++i) {
            g.max_sugg   = std::max(g.max_sugg, chunks[i].max_sugg);
            g.branch_max = std::max(g.branch_max, chunks[i].branch_max);
            g.ncoords    = std::max(g.ncoords, chunks[i].ncoords);
            g.nsegments  = std::max(g.nsegments, chunks[i].nsegments);
        }
        g.end = i;
        groups.push_back(g);
    }
    return groups;
}

/// Per-worker bytes of one chunk group.
template <SupportedFoldType FoldType>
SizeType ep_group_thread_bytes(const EPMemoryContext& ctx,
                               const EPChunkGroup& g) {
    return ep_thread_bytes<FoldType>(ctx, g.nbins, g.nsegments, g.ncoords,
                                     g.max_sugg, g.branch_max);
}

/**
 * @brief Peak memory of a sweep over @p chunks, in GiB.
 *
 * The largest chunk group's per-worker buffers plus the shared FFA buffers
 * (global maxima) and the inputs.
 */
template <SupportedFoldType FoldType>
double ep_sweep_peak_gb(std::span<const EPChunkConfig> chunks,
                        const EPMemoryContext& ctx) {
    SizeType buffer_size = 0;
    SizeType coord_size  = 0;
    for (const auto& c : chunks) {
        buffer_size = std::max(buffer_size, c.buffer_size);
        coord_size  = std::max(coord_size, c.coord_size);
    }
    SizeType peak_thread = 0;
    for (const auto& g : ep_chunk_groups(chunks)) {
        peak_thread =
            std::max(peak_thread, ep_group_thread_bytes<FoldType>(ctx, g));
    }
    return ep_total_gb(ctx.n_workers, peak_thread,
                       ep_fixed_bytes<FoldType>(ctx, buffer_size, coord_size));
}

} // namespace loki::algorithms::detail
