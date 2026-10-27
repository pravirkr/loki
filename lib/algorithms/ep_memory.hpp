#pragma once

/**
 * @file ep_memory.hpp
 * @brief Memory model of EPFreqSweep (CPU and CUDA policies), shared by
 * EPRegionPlanner, the sweep's runtime checks and the tests that compare it
 * with the real allocations. Internal. See docs/memory.md.
 *
 * EPFreqSweep holds three kinds of memory:
 *  - per-worker buffers: the EP workspace (world tree, prune/branch scratch,
 *    FFA seeds) and, when harvesting is enabled, the harvest store. The
 *    pruning functors score tile by tile (16 leaves) and keep no batch-sized
 *    scoring scratch; their few kilobytes are left to the reserve. They are
 * sized once per contiguous run of chunks with the same nbins, from the maxima
 * of that run, and freed before the next run;
 *  - the FFA workspace (internal fold buffer and coordinates) and the output
 *    fold buffer, shared by every chunk and sized from the global maxima;
 *  - the input time series (ts_e, ts_v), resident for the whole sweep whoever
 *    owns them.
 *
 * The peak is therefore, for the worst run:
 *   n_workers * per-worker bytes of the run + shared bytes + input bytes.
 *
 * The model has two policies (EPMemoryKind). The CPU one counts host memory
 * with n_workers pruning threads. The CUDA one counts device memory of the one
 * serial worker EPFreqSweep runs on the GPU: the EP workspace (world tree,
 * prune and branching scratch, CUB temp storage, seeds, segment coordinates,
 * Fourier irfft scratch), the shared FFA workspace and fold buffer, the
 * device copy of the inputs, and the transient device scratch of the FFA
 * brute fold, which is live while the EP workspace is. There is no harvest
 * term: harvesting is rejected on CUDA.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"

#include "lib/algorithms/planner_memory.hpp"
#include "lib/detail/utils.hpp"

namespace loki::algorithms::detail {

/// Batch size the CPU pruning runs with.
constexpr SizeType kEPBatchSize = 1024U;

/// Batch size the GPU pruning runs with. A batch is one kernel round trip and
/// one host synchronisation of the world tree, so it must be large: at 1024
/// the per-batch overhead dominates a 2^25-sample run, which then takes a
/// minute per reference segment instead of seconds.
constexpr SizeType kEPBatchSizeCuda = SizeType{1} << 16U;

/// Version of this model. Bump it with every change to the formulas: plan
/// caches store it and are rejected when it differs.
///  3: CUDA policy (EPMemoryKind), transient FFA scratch of the CUDA model.
///  4: the Kadane scorer is fused too, so it no longer keeps the irfft scratch.
///  5: shared FFA/tree arena, branch lookahead and score-only overflow.
///  6: bounded canonical shift-add scratch for fused overflow recomputation.
///  7: remove branch lookahead after the SNR-10 timing regression.
///  9: no cuFFT: the CUDA reserve is 0.5 GiB for any number of FFT sizes.
/// 10: CPU pruning scores in fused tiles: no irfft scratch per worker.
/// 11: ring capacity from the margin-inflated live forecast, rounded up to a
///     multiple of kEPRingGranularity (was bit_ceil).
/// 12: final winning-width reporting, uint32 per surviving CPU leaf.
constexpr std::uint32_t kEPMemoryModelVersion = 12U;

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

/// Which allocations the model counts.
enum class EPMemoryKind : std::uint8_t {
    kCpu,  ///< Host memory, n_workers pruning threads.
    kCuda, ///< Device memory, one serial worker.
};

/// Bytes of CUB temporary storage for @p max_n_leaves leaves on @p device.
/// Defined in lib/cuda/ (GPU builds only); a CUDA policy needs it.
using CubScratchBytesFn = SizeType (*)(SizeType max_n_leaves, int device);

/// Sweep-wide inputs of the model.
struct EPMemoryContext {
    SizeType nparams{0};
    SizeType nsamps{0};
    /// Workers pruning at the same time, each with its own buffers. Always 1
    /// for the CUDA policy.
    int n_workers{1};
    SizeType batch_size{kEPBatchSize};
    EPHarvestBound harvest;
    EPMemoryKind kind{EPMemoryKind::kCpu};
    /// CUDA policy: the device the sweep runs on.
    int device{0};
    /// CUDA policy: sizes the CUB temporary storage. Required for kCuda.
    CubScratchBytesFn cub_scratch_bytes{nullptr};
    /// Sweep-only: FFA internal folds and tree folds share one allocation.
    bool share_ffa_fold{false};
    SizeType shared_fold_bytes{0};
};

/// Reserve for what the EP memory model does not count (kUnmodelledReserveGB
/// in planner_memory.hpp). The GPU transforms keep no vendor plans, so the
/// reserve no longer grows with the number of FFT sizes.
inline double ep_memory_reserve_gb(const search::PulsarSearchConfig& /*cfg*/,
                                   EPMemoryKind /*kind*/) {
    return kUnmodelledReserveGB;
}

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

// --- CUDA policy ---------------------------------------------------------
// Mirror EPWorkspaceCUDA, FFAWorkspaceCUDA and the brute fold of
// FFACudaCore, member by member (lib/cuda/workspace_cuda.{cuh,cu}).

/// World-tree capacity the CUDA workspace is built with: the world tree
/// requires a capacity above the largest batch it receives.
constexpr SizeType ep_cuda_effective_max_sugg(SizeType batch_size,
                                              SizeType branch_max,
                                              SizeType max_sugg) noexcept {
    return std::max(max_sugg, (batch_size * branch_max) + 1);
}

/// Largest world-tree capacity the backend can index.
inline constexpr SizeType ep_tree_index_limit(EPMemoryKind kind) noexcept {
    return kind == EPMemoryKind::kCuda
               ? (SizeType{1} << 30U)
               : (SizeType{1} << (std::numeric_limits<SizeType>::digits - 2));
}

/// The world tree is sized in multiples of this many leaves. Neither ring
/// needs a power of two: both index modulo the capacity.
inline constexpr SizeType kEPRingGranularity = 1024;

/// Normal quantile of the ring margin: a forecast sized with it falls short
/// of the true mean population about 1 % of the time (issue #15, section 7).
inline constexpr double kEPRingMarginZ = 2.33;

/**
 * @brief Noise population of one schedule on one branching pattern.
 *
 * Per stage: the survivors per initial seed (State::complexity) and the
 * stage's noise pass rate s_j (State::success_h0), measured on @p ntrials
 * trials per stage. @p ntrials = 0 marks an exact (synthetic) forecast.
 */
struct EPPopulationForecast {
    std::vector<double> complexity{};
    std::vector<double> pass_rate{};
    SizeType ntrials{0};
};

/**
 * @brief Per-stage factor 1 + z eps_s that covers the Monte Carlo error of a
 * forecast population.
 *
 * Population s is the seed count times a product of measured pass rates, so
 * its relative error is eps_s = sqrt(sum_{j <= s} (1 - s_j) / (N s_j)) (the
 * binomial errors of the factors, in quadrature). The evaluation keeps at
 * least one noise trial per stage, so s_j >= 1 / N; a stage at that floor
 * adds about 1 to eps_s^2 (a margin of at least 1 + z from there on, which
 * over-allocates on the safe side). A rate of 0 (a hand-made forecast) is
 * skipped. All ones for an exact forecast.
 */
inline std::vector<double>
ep_ring_margin(const EPPopulationForecast& forecast) {
    std::vector<double> margin(forecast.complexity.size(), 1.0);
    if (forecast.ntrials == 0) {
        return margin;
    }
    const auto n = static_cast<double>(forecast.ntrials);
    double var   = 0.0;
    for (SizeType s = 0; s < margin.size(); ++s) {
        const double pass =
            s < forecast.pass_rate.size() ? forecast.pass_rate[s] : 1.0;
        if (pass > 0.0 && pass < 1.0) {
            var += (1.0 - pass) / (n * pass);
        }
        margin[s] = 1.0 + (kEPRingMarginZ * std::sqrt(var));
    }
    return margin;
}

/// Forecast ring populations of a chunk with @p ncoords seeds.
struct EPRingForecast {
    /// Largest stage population L_s (with its margin).
    double peak{0.0};
    /// Largest occupancy of the ring, L_{s-1} + L_s: a stage's parents are
    /// consumed batch by batch while its survivors are written.
    double live{0.0};
    /// Largest per-seed complexity (without margin), for reports.
    double peak_complexity{1.0};
};

/**
 * @brief Ring populations: L_0 = ncoords (exact) and
 * L_s = ceil(ncoords * complexity_s * margin_s).
 *
 * @p margin is ep_ring_margin() of the forecast that sizes the chunk; the
 * planner's width probes pass the margin of their reference design.
 */
inline EPRingForecast ep_ring_forecast(SizeType ncoords,
                                       std::span<const double> complexity,
                                       std::span<const double> margin) {
    const auto seeds = static_cast<double>(ncoords);
    EPRingForecast out{.peak = seeds, .live = seeds};
    double previous = seeds;
    for (SizeType s = 0; s < complexity.size(); ++s) {
        const double c = complexity[s];
        if (!utils::is_finite(c) || c < 0.0) {
            throw std::logic_error(
                "EPRegionPlanner: invalid complexity forecast");
        }
        const double m          = s < margin.size() ? margin[s] : 1.0;
        const double population = std::ceil(seeds * c * m);
        out.peak                = std::max(out.peak, population);
        out.live                = std::max(out.live, previous + population);
        out.peak_complexity     = std::max(out.peak_complexity, c);
        previous                = population;
    }
    return out;
}

/// World-tree capacity for a forecast @p live ring: room for one branched
/// batch on top, rounded up to kEPRingGranularity leaves.
inline SizeType ep_tree_capacity(SizeType live,
                                 SizeType batch,
                                 SizeType branch_max,
                                 EPMemoryKind kind) {
    const SizeType max_capacity = ep_tree_index_limit(kind);
    if (branch_max == 0 || batch > max_capacity / branch_max ||
        batch * branch_max >= max_capacity - kEPRingGranularity ||
        live >= max_capacity - (batch * branch_max) - kEPRingGranularity) {
        throw std::overflow_error(
            "EPRegionPlanner: forecast tree exceeds index capacity");
    }
    const SizeType needed = live + (batch * branch_max) + 1;
    const SizeType units =
        (needed + kEPRingGranularity - 1) / kEPRingGranularity;
    return std::max(units, SizeType{1}) * kEPRingGranularity;
}

/// Largest nbins of the fused CUDA pruning kernels (ep_score_cuda.cuh):
/// the Fourier on-chip transforms, and time-domain rows that fit on chip.
inline constexpr SizeType kEPCudaFusedMaxNbinsFourier = 1024;
inline constexpr SizeType kEPCudaFusedMaxNbinsTime    = 1024;

/// Whether the CUDA prune functors score the seeds and the ascend (Fourier
/// folds; always the boxcar) on chip.
constexpr bool ep_cuda_fused_rows(SizeType nbins) noexcept {
    return nbins >= 2 && nbins <= kEPCudaFusedMaxNbinsFourier;
}

/// Fused scorer statistics and ordered-compaction tile states, the only
/// per-functor scratch: CUDA EP scores and transforms on chip.
inline SizeType ep_cuda_scorer_scratch_bytes(const EPMemoryContext& ctx,
                                             SizeType branch_max) {
    return (2 + ctx.batch_size * branch_max + 1) * sizeof(std::uint64_t);
}

/// Bytes of one EPWorkspaceCUDA, sized as EPFreqSweep builds it for one chunk
/// group: world tree, prune and branching workspaces, CUB scratch, seeds and
/// segment coordinates.
template <SupportedFoldType FoldType>
SizeType ep_cuda_ep_workspace_bytes(const EPMemoryContext& ctx,
                                    SizeType nbins,
                                    SizeType nsegments,
                                    SizeType ncoords_ffa,
                                    SizeType max_sugg,
                                    SizeType branch_max) {
    if (ctx.cub_scratch_bytes == nullptr) {
        throw std::logic_error(
            "EPMemoryContext: the CUDA policy needs cub_scratch_bytes");
    }
    constexpr bool kIsComplex       = std::is_same_v<FoldType, ComplexType>;
    constexpr SizeType kParamStride = 2U;
    const SizeType batch            = ctx.batch_size;
    const SizeType nparams          = ctx.nparams;
    const SizeType nbins_arg        = kIsComplex ? (nbins / 2) + 1 : nbins;
    const SizeType leaves_stride    = (nparams + 2) * kParamStride;
    const SizeType max_batch_size   = batch * branch_max;
    const SizeType capacity =
        ep_cuda_effective_max_sugg(batch, branch_max, max_sugg);

    // WorldTreeCUDA: leaves, folds, scores, scores_ep, scratch_scores,
    // two index scratch arrays, mask.
    const SizeType world_tree_bytes =
        (capacity * leaves_stride * sizeof(double)) +
        (capacity * 2 * nbins_arg * fold_bytes<FoldType>()) +
        (capacity * 2 * sizeof(float)) +
        ((capacity + max_batch_size) * sizeof(float)) +
        (capacity * 2 * sizeof(std::uint32_t)) + (capacity * sizeof(uint8_t));

    // PruneWorkspaceCUDA
    const SizeType max_branched_param_idx =
        std::max(max_batch_size, nsegments * batch);
    const SizeType prune_bytes =
        (max_batch_size * leaves_stride * sizeof(double)) +
        (batch * 2 * nbins_arg * fold_bytes<FoldType>()) +
        (max_batch_size * sizeof(float)) +
        (max_batch_size * 2 * sizeof(std::uint32_t)) +
        (max_branched_param_idx * sizeof(std::uint32_t)) +
        (max_branched_param_idx * sizeof(float)) +
        (max_batch_size * 2 * sizeof(uint8_t));

    // BranchingWorkspaceCUDA
    const SizeType branch_bytes =
        (batch * nparams * branch_max * sizeof(double)) +
        (batch * nparams * sizeof(double)) +
        (batch * nparams * sizeof(std::uint32_t)) +
        (2 * batch * sizeof(std::uint32_t));

    // CUBScratchArena: temp storage, the reduce count and the min/max pair.
    const SizeType cub_bytes =
        ctx.cub_scratch_bytes(max_batch_size, ctx.device) +
        sizeof(std::uint32_t) + (2 * sizeof(float));

    // Seeds and the segment coordinates (index + pair of doubles).
    const SizeType seed_bytes = (ncoords_ffa * leaves_stride * sizeof(double)) +
                                (ncoords_ffa * sizeof(float));
    const SizeType segment_bytes =
        nsegments * (sizeof(std::uint32_t) + (2 * sizeof(double)));

    return world_tree_bytes + prune_bytes + branch_bytes + cub_bytes +
           seed_bytes + segment_bytes;
}

/// Per-group device bytes of the CUDA policy: the EP workspace plus the
/// irfft scratch the prune functors of each chunk allocate.
template <SupportedFoldType FoldType>
SizeType ep_cuda_workspace_bytes(const EPMemoryContext& ctx,
                                 SizeType nbins,
                                 SizeType nsegments,
                                 SizeType ncoords_ffa,
                                 SizeType max_sugg,
                                 SizeType branch_max) {
    const auto capacity =
        ep_cuda_effective_max_sugg(ctx.batch_size, branch_max, max_sugg);
    const auto tree_fold_bytes =
        capacity * 2 *
        (std::is_same_v<FoldType, ComplexType> ? nbins / 2 + 1 : nbins) *
        fold_bytes<FoldType>();
    const auto overlap = ctx.share_ffa_fold
                             ? std::min(ctx.shared_fold_bytes, tree_fold_bytes)
                             : 0;
    return ep_cuda_ep_workspace_bytes<FoldType>(
               ctx, nbins, nsegments, ncoords_ffa, max_sugg, branch_max) -
           overlap + ep_cuda_scorer_scratch_bytes(ctx, branch_max);
}

/// Bytes of the FFA workspace (internal fold buffer and device coordinates)
/// plus the output fold buffer. The few small per-level arrays of the
/// workspace (counts, offsets, limits) are left to the unmodelled reserve.
template <SupportedFoldType FoldType>
SizeType ep_cuda_shared_bytes(SizeType nparams,
                              SizeType buffer_size,
                              SizeType coord_size) {
    // FFACoordFreqD: index + shift. FFACoordD: tail and head, each of both.
    const SizeType coord_unit_bytes =
        (nparams == 1) ? (sizeof(std::uint32_t) + sizeof(float))
                       : 2 * (sizeof(std::uint32_t) + sizeof(float));
    return (2 * buffer_size * fold_bytes<FoldType>()) +
           (coord_size * coord_unit_bytes);
}

/**
 * @brief Transient device scratch of one chunk's FFA, live while the EP
 * workspace is allocated (CUDA policy only).
 *
 * The brute fold keeps the frequency grid and, for time-domain folds (and the
 * lossy Fourier start of large nbins), a uint32 phase map of
 * nfreqs * segment_len entries (BruteFoldCudaEngine).
 */
template <SupportedFoldType FoldType>
SizeType ep_ffa_transient_bytes(const EPMemoryContext& ctx,
                                const plans::FFAPlan<FoldType>& plan) {
    if (ctx.kind != EPMemoryKind::kCuda) {
        return 0;
    }
    const SizeType nfreqs  = plan.get_param_counts().front().back();
    const SizeType seg_len = plan.get_segment_lens().front();
    const SizeType nbins   = plan.get_config().get_nbins();
    const bool has_phase_map =
        !std::is_same_v<FoldType, ComplexType> ||
        nbins > plan.get_config().get_nbins_min_lossy_bf();
    return (nfreqs * sizeof(double)) +
           (has_phase_map ? nfreqs * seg_len * sizeof(std::uint32_t) : 0);
}

/// Bytes each worker needs for one chunk group.
template <SupportedFoldType FoldType>
SizeType ep_thread_bytes(const EPMemoryContext& ctx,
                         SizeType nbins,
                         SizeType nsegments,
                         SizeType ncoords_ffa,
                         SizeType max_sugg,
                         SizeType branch_max) {
    if (ctx.kind == EPMemoryKind::kCuda) {
        return ep_cuda_workspace_bytes<FoldType>(
            ctx, nbins, nsegments, ncoords_ffa, max_sugg, branch_max);
    }
    return ep_workspace_bytes<FoldType>(ctx.nparams, nbins, nsegments,
                                        ncoords_ffa, max_sugg, branch_max,
                                        ctx.batch_size) +
           ep_harvest_bytes<FoldType>(ctx.nparams, nbins, ctx.harvest) +
           max_sugg * sizeof(std::uint32_t);
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
/// @p ffa_transient_bytes is the largest transient FFA scratch of any chunk
/// (CUDA policy; ignored by the CPU policy).
template <SupportedFoldType FoldType>
SizeType ep_fixed_bytes(const EPMemoryContext& ctx,
                        SizeType buffer_size,
                        SizeType coord_size,
                        SizeType ffa_transient_bytes = 0) {
    if (ctx.kind == EPMemoryKind::kCuda) {
        return ep_cuda_shared_bytes<FoldType>(ctx.nparams, buffer_size,
                                              coord_size) +
               ep_input_bytes(ctx.nsamps) + ffa_transient_bytes;
    }
    return ep_shared_bytes<FoldType>(ctx.nparams, buffer_size, coord_size) +
           ep_input_bytes(ctx.nsamps);
}

[[nodiscard]] inline double ep_total_gb(int n_workers,
                                        SizeType thread_bytes,
                                        SizeType fixed_bytes) noexcept {
    const auto total =
        (static_cast<double>(n_workers) * static_cast<double>(thread_bytes)) +
        static_cast<double>(fixed_bytes);
    return total / utils::kBytesPerGB;
}

/// A contiguous run of chunks with the same nbins and its maxima: one
/// allocation of the per-worker buffers in EPFreqSweep.
struct EPChunkGroup {
    SizeType begin{0}; ///< First chunk index.
    SizeType end{0};   ///< One past the last chunk index.
    SizeType nbins{0};
    SizeType batch_size{0};
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
        EPChunkGroup g{.begin      = i,
                       .nbins      = chunks[i].cfg.get_nbins(),
                       .batch_size = chunks[i].batch_size};
        for (; i < chunks.size() && chunks[i].cfg.get_nbins() == g.nbins &&
               chunks[i].batch_size == g.batch_size;
             ++i) {
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
    auto memory = ctx;
    if (g.batch_size != 0) {
        memory.batch_size = g.batch_size;
    }
    return ep_thread_bytes<FoldType>(memory, g.nbins, g.nsegments, g.ncoords,
                                     g.max_sugg, g.branch_max);
}

/**
 * @brief Peak memory of a sweep over @p chunks, in GB.
 *
 * The largest chunk group's per-worker buffers plus the shared FFA buffers
 * (global maxima) and the inputs.
 */
template <SupportedFoldType FoldType>
double ep_sweep_peak_gb(std::span<const EPChunkConfig> chunks,
                        const EPMemoryContext& ctx) {
    SizeType buffer_size = 0;
    SizeType coord_size  = 0;
    SizeType transient   = 0;
    for (const auto& c : chunks) {
        buffer_size = std::max(buffer_size, c.buffer_size);
        coord_size  = std::max(coord_size, c.coord_size);
        transient   = std::max(transient, c.ffa_transient_bytes);
    }
    auto memory              = ctx;
    memory.shared_fold_bytes = buffer_size * fold_bytes<FoldType>();
    SizeType peak_thread     = 0;
    for (const auto& g : ep_chunk_groups(chunks)) {
        peak_thread =
            std::max(peak_thread, ep_group_thread_bytes<FoldType>(memory, g));
    }
    return ep_total_gb(
        ctx.n_workers, peak_thread,
        ep_fixed_bytes<FoldType>(ctx, buffer_size, coord_size, transient));
}

} // namespace loki::algorithms::detail
