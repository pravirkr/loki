#pragma once

/**
 * @file ep_memory.hpp
 * @brief Host memory model of EPFreqSweep, shared by EPRegionPlanner and the
 * tests that check it against the real allocations. Internal.
 *
 * EPFreqSweep allocates two kinds of buffers:
 *  - per-thread EP workspaces (world tree, prune/branch scratch, FFA seeds and,
 *    for Fourier folds, irfft scratch). They are sized once per contiguous run
 *    of chunks with the same nbins, from the maxima of that run, and freed
 *    before the next run;
 *  - the FFA workspace (internal fold buffer and coordinates) and the output
 *    fold buffer. They are shared by every chunk and sized from the global
 *    maxima.
 *
 * The peak is therefore, for the worst run: nthreads * per-thread bytes of the
 * run + shared bytes of the whole sweep.
 */

#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms::detail {

constexpr double kBytesPerGiB = static_cast<double>(1ULL << 30U);

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

/// Bytes each thread needs for one chunk group.
template <SupportedFoldType FoldType>
SizeType ep_thread_bytes(SizeType nparams,
                         SizeType nbins,
                         SizeType nsegments,
                         SizeType ncoords_ffa,
                         SizeType max_sugg,
                         SizeType branch_max,
                         SizeType batch_size) {
    return ep_workspace_bytes<FoldType>(nparams, nbins, nsegments, ncoords_ffa,
                                        max_sugg, branch_max, batch_size) +
           ep_irfft_scratch_bytes<FoldType>(nbins, ncoords_ffa, branch_max,
                                            batch_size);
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

[[nodiscard]] inline double ep_total_gb(int nthreads,
                                        SizeType thread_bytes,
                                        SizeType shared_bytes) noexcept {
    const auto total =
        (static_cast<double>(nthreads) * static_cast<double>(thread_bytes)) +
        static_cast<double>(shared_bytes);
    return total / kBytesPerGiB;
}

/**
 * @brief Peak memory of a sweep over @p chunks, in GiB.
 *
 * Groups the chunks into contiguous runs of equal nbins, exactly as
 * EPFreqSweep does, and returns the largest group plus the shared FFA buffers.
 */
template <SupportedFoldType FoldType>
double ep_sweep_peak_gb(std::span<const EPChunkConfig> chunks,
                        int nthreads,
                        SizeType nparams,
                        SizeType batch_size) {
    SizeType buffer_size = 0;
    SizeType coord_size  = 0;
    for (const auto& c : chunks) {
        buffer_size = std::max(buffer_size, c.buffer_size);
        coord_size  = std::max(coord_size, c.coord_size);
    }
    const auto shared =
        ep_shared_bytes<FoldType>(nparams, buffer_size, coord_size);
    SizeType peak_thread = 0;
    for (SizeType i = 0; i < chunks.size();) {
        const SizeType nbins = chunks[i].cfg.get_nbins();
        SizeType max_sugg    = 0;
        SizeType branch_max  = 0;
        SizeType ncoords     = 0;
        SizeType nsegments   = 0;
        for (; i < chunks.size() && chunks[i].cfg.get_nbins() == nbins; ++i) {
            max_sugg   = std::max(max_sugg, chunks[i].max_sugg);
            branch_max = std::max(branch_max, chunks[i].branch_max);
            ncoords    = std::max(ncoords, chunks[i].ncoords);
            nsegments  = std::max(nsegments, chunks[i].nsegments);
        }
        peak_thread = std::max(
            peak_thread,
            ep_thread_bytes<FoldType>(nparams, nbins, nsegments, ncoords,
                                      max_sugg, branch_max, batch_size));
    }
    return ep_total_gb(nthreads, peak_thread, shared);
}

} // namespace loki::algorithms::detail
