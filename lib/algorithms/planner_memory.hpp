#pragma once

/**
 * @file planner_memory.hpp
 * @brief Memory reserve shared by the FFA and EP region planners. Internal.
 */

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <type_traits>

#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/detail/utils.hpp"

namespace loki::algorithms::detail {

/**
 * @brief Process memory (GB) that the planners' memory models do not cover.
 *
 * The models count every large buffer exactly. What remains is hard to account
 * deterministically: heap fragmentation and allocator arenas, thread stacks,
 * small per-task objects (FFA plans, FFT tables, logging buffers) and the
 * interpreter or application around the library. A fixed amount, rather than
 * a fraction of the limit, keeps large limits usable and small limits
 * feasible. See docs/memory.md.
 */
inline constexpr double kUnmodelledReserveGB = 0.5;

/**
 * @brief Free device memory (GB) the GPU planners leave alone: the CUDA
 * runtime and context, kernel images, CUB temporaries, the small FFT tables
 * and what other processes allocate meanwhile. Applied by every GPU sweep to
 * the device's free memory before it takes the smaller of that and
 * max_process_memory_gb.
 */
inline constexpr double kDeviceReserveGB = 1.0;

/// Memory available to the modelled buffers under @p max_process_memory_gb.
[[nodiscard]] constexpr double
effective_memory_limit_gb(double max_process_memory_gb) noexcept {
    return max_process_memory_gb - kUnmodelledReserveGB;
}

/// Boxcar widths the fused GPU final-level scorers handle per launch.
inline constexpr SizeType kGpuFusedMaxWidths = 32;

/**
 * @brief Candidate slots of the GPU sweep's per-chunk score scratch.
 *
 * The fused scorer appends passing (profile, width) pairs through an atomic
 * slot counter and drops writes past this capacity; a chunk that overflows
 * is scored again in profile ranges that fit. 2^24 slots (128 MiB) covers
 * every realistic chunk in one pass.
 */
inline constexpr SizeType kGpuScoreScratchCap = SizeType{1} << 24U;

/**
 * @brief Whether the GPU sweep scores this chunk with the fused final level
 * (the final fold is never stored). Time-domain folds take any nbins;
 * Fourier folds need nbins in [2, 1024] (on-chip inverse FFT: a direct
 * C2R for powers of two from 32, Bluestein otherwise).
 */
template <SupportedFoldType FoldType>
[[nodiscard]] bool
gpu_fused_score_applies(const plans::FFAPlan<FoldType>& plan) {
    const auto& cfg = plan.get_config();
    if (plan.get_n_levels() < 2 || plan.get_nsegments().back() != 1 ||
        cfg.get_n_scoring_widths() > kGpuFusedMaxWidths) {
        return false;
    }
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        const SizeType nbins = cfg.get_nbins();
        return nbins >= 2 && nbins <= 1024;
    }
    return true;
}

/**
 * @brief How many final levels the GPU sweep merges on chip with the
 * scorer: 0 (unfused), 1, or 2 (at least three levels; the last two
 * levels are merged from the four-segment level).
 */
template <SupportedFoldType FoldType>
[[nodiscard]] int gpu_fused_score_depth(const plans::FFAPlan<FoldType>& plan) {
    if (!gpu_fused_score_applies(plan)) {
        return 0;
    }
    return plan.get_n_levels() >= 3 ? 2 : 1;
}

/**
 * @brief Ping-pong buffer size (fold elements) of the GPU sweep for a chunk:
 * the largest level the sweep stores. The last gpu_fused_score_depth levels
 * are merged and scored on chip; an unfused chunk stores every level.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
gpu_sweep_buffer_size(const plans::FFAPlan<FoldType>& plan) {
    const int depth = gpu_fused_score_depth(plan);
    if (depth == 0) {
        return plan.get_buffer_size();
    }
    const auto& shapes  = plan.get_fold_shapes();
    const SizeType keep = shapes.size() - static_cast<SizeType>(depth);
    SizeType size       = 0;
    for (SizeType i = 0; i < keep; ++i) {
        size =
            std::max(size, std::accumulate(shapes[i].begin(), shapes[i].end(),
                                           SizeType{1}, std::multiplies<>()));
    }
    return size;
}

/**
 * @brief Coordinates the GPU sweep stores for a chunk: accel searches whose
 * final level the fused scorer takes resolve that level on the fly, so it
 * is not stored.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
gpu_sweep_coord_size(const plans::FFAPlan<FoldType>& plan) {
    if (plan.get_n_params() == 2 && gpu_fused_score_depth(plan) > 0) {
        return plan.get_coord_size() - plan.get_ncoords().back();
    }
    return plan.get_coord_size();
}

/**
 * @brief Levels a CPU FFA merges straight into scored tiles (`execute_scored`,
 * used by the CPU sweep): 2 for accel (jerk) plans of three or more levels, 1
 * for two, 0 when the top level has more than one segment. Accel trees widen,
 * so the last two levels are the largest; frequency-only searches use the
 * cone path or score the final level only (1).
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_scored_fused_levels(const plans::FFAPlan<FoldType>& plan) {
    const SizeType n_levels = plan.get_fold_shapes().size();
    if (n_levels < 2 || plan.get_nsegments().back() != 1) {
        return 0;
    }
    if (plan.get_n_params() >= 2 && n_levels >= 3) {
        return 2;
    }
    return 1;
}

/**
 * @brief Fold buffer size (elements) a CPU FFA needs for scored execution
 * (`execute_scored`, used by the CPU sweep).
 *
 * Accel searches merge the last two levels tile by tile into scratch and
 * score them there (cpu_scored_fused_levels), so the ping-pong buffers hold
 * the other levels only. Frequency-only trees are flat and the cone path may
 * store the top level, so they keep the full buffer.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_scored_buffer_size(const plans::FFAPlan<FoldType>& plan) {
    const auto& shapes = plan.get_fold_shapes();
    if (plan.get_n_params() < 2 || cpu_scored_fused_levels(plan) == 0) {
        return plan.get_buffer_size();
    }
    const SizeType stored = shapes.size() - cpu_scored_fused_levels(plan);
    SizeType size         = 0;
    for (SizeType i = 0; i < stored; ++i) {
        size =
            std::max(size, std::accumulate(shapes[i].begin(), shapes[i].end(),
                                           SizeType{1}, std::multiplies<>()));
    }
    return size;
}

/**
 * @brief Bytes of final-level accel coordinates a CPU scored execution
 * resolves at a time. Accel (n_params = 2) plans resolve the final level on
 * the fly, a batch of whole rows (accel trials) at a time, instead of
 * storing it: it is the largest level, and its profiles are never stored
 * either (cpu_scored_fused_levels).
 */
inline constexpr SizeType kCpuFinalCoordBatchBytes = SizeType{32} << 20U;

/// True when a CPU scored execution resolves the final level on the fly.
template <SupportedFoldType FoldType>
[[nodiscard]] bool
cpu_final_coords_on_the_fly(const plans::FFAPlan<FoldType>& plan) {
    return plan.get_n_params() == 2 && cpu_scored_fused_levels(plan) >= 1;
}

/// Rows (accel trials) of the final level resolved per batch.
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_final_coord_batch_rows(const plans::FFAPlan<FoldType>& plan,
                           SizeType batch_bytes = kCpuFinalCoordBatchBytes) {
    const SizeType row_len = plan.get_param_counts().back().back();
    const SizeType nrows   = plan.get_ncoords().back() / row_len;
    const SizeType per_batch =
        std::max<SizeType>(1, batch_bytes / sizeof(coord::FFACoord));
    return std::clamp<SizeType>(per_batch / row_len, 1, nrows);
}

/**
 * @brief Coordinates (elements) a CPU FFA needs for scored execution: every
 * level, except that accel plans hold one batch of final-level rows in place
 * of the final level (cpu_final_coords_on_the_fly).
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_scored_coord_size(const plans::FFAPlan<FoldType>& plan) {
    if (!cpu_final_coords_on_the_fly(plan)) {
        return plan.get_coord_size();
    }
    const SizeType row_len = plan.get_param_counts().back().back();
    return plan.get_coord_size() - plan.get_ncoords().back() +
           (cpu_final_coord_batch_rows(plan) * row_len);
}

/**
 * @brief Bytes of the phase-step table of the CPU complex (Fourier) merges.
 *
 * A complex merge rotates a profile by d = exp(-2 pi i shift / nbins). The
 * steps depend only on the coordinate, so the kernels compute them once per
 * coordinate and reuse them for every segment, segment group and tile. They
 * compute them for one batch of coordinates at a time into this one table
 * (`FFAWorkspaceCPU::phase_steps`), sized here and not by the plan, so it
 * bounds the memory whatever the number of coordinates. 32 MiB holds about
 * 170 k steps (192 B each with AVX-512).
 */
inline constexpr SizeType kPhaseStepTableBytes = SizeType{32} << 20U;

/// Largest phase step of any build (AVX-512: 16 complex lanes, split re /
/// im, plus two complex values, 64-byte aligned); kernels_cpu.cpp checks.
inline constexpr SizeType kMaxPhaseStepBytes = 192;

/**
 * @brief Phase-step table bytes of a CPU FFA with this plan: complex folds
 * only. kPhaseStepTableBytes, or less when the plan's steps all fit: at
 * most two per coordinate of every merged level (tail and head), plus the
 * rounding of each level's range to step blocks.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_phase_step_table_bytes(const plans::FFAPlan<FoldType>& plan) {
    if constexpr (!std::is_same_v<FoldType, ComplexType>) {
        return 0;
    } else {
        constexpr SizeType kStepBlock = 32; // kernels_cpu.cpp kMergeStepBlock
        const auto ncoords            = plan.get_ncoords();
        SizeType steps                = 0;
        for (SizeType level = 1; level < ncoords.size(); ++level) {
            steps += (2 * ncoords[level]) + (2 * kStepBlock);
        }
        return std::min(kPhaseStepTableBytes, steps * kMaxPhaseStepBytes);
    }
}

/**
 * @brief Default per-thread scratch budget of a CPU cone band (both
 * ping-pong tiles, and for the bottom time-domain band the segment prefix
 * sums). `LOKI_CONE_SCRATCH_KB` overrides it (benchmark hook).
 */
inline constexpr SizeType kCpuConeScratchBytes = SizeType{2} << 20U;

/// The cone scratch budget in effect (the engine and the planner both use
/// it, so the memory model follows the hook).
[[nodiscard]] inline SizeType cpu_cone_scratch_budget_bytes() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): serial setup, no setenv
    const char* env = std::getenv("LOKI_CONE_SCRATCH_KB");
    if (env != nullptr && env[0] != '\0') {
        char* end = nullptr;
        // NOLINTNEXTLINE(google-runtime-int): std::strtoul return type
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && *end == '\0' && v > 0) {
            return static_cast<SizeType>(v) * 1024U;
        }
    }
    return kCpuConeScratchBytes;
}

/// Per-thread budget of each of the two scratch tiles of the fused
/// time-domain brute fold (cpu_fuse_levels).
inline constexpr SizeType kCpuFuseScratchBytes = SizeType{512} << 10U;
/// Most merge levels the fused time-domain brute fold takes.
inline constexpr SizeType kCpuFuseMaxLevels = 6;

/**
 * @brief Merge levels a time-domain frequency-only FFA fuses into the brute
 * fold on its plain (non-cone) path (BruteFold::execute_fused_freq): the
 * most, up to kCpuFuseMaxLevels, that keep a tile of 2^k segments per
 * thread and each of whose two scratch tiles fits kCpuFuseScratchBytes. 0
 * for other plans. The engine uses it (unless LOKI_FUSE_LEVELS overrides)
 * and the planner counts its prefix sums.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType cpu_fuse_levels(const plans::FFAPlan<FoldType>& plan) {
    if constexpr (!std::is_same_v<FoldType, float>) {
        return 0;
    } else {
        const SizeType levels = plan.get_n_levels();
        if (plan.get_n_params() != 1 || levels < 2) {
            return 0;
        }
        const auto ncoords   = plan.get_ncoords();
        const SizeType nbins = plan.get_config().get_nbins();
        const SizeType nseg0 = plan.get_fold_shapes_time()[0][0];
        const auto nthreads  = static_cast<SizeType>(
            std::max(plan.get_config().get_nthreads(), 1));
        const SizeType k_limit = std::min(kCpuFuseMaxLevels, levels - 1);
        SizeType best          = 0;
        for (SizeType k = 1; k <= k_limit; ++k) {
            if ((nseg0 >> k) < nthreads || (nseg0 % (SizeType{1} << k)) != 0) {
                break;
            }
            SizeType tile_floats = 0;
            for (SizeType j = 0; j <= k; ++j) {
                tile_floats = std::max(tile_floats, (SizeType{1} << (k - j)) *
                                                        ncoords[j] * 2 * nbins);
            }
            if (tile_floats * sizeof(float) > kCpuFuseScratchBytes) {
                break;
            }
            best = k;
        }
        return best;
    }
}

/**
 * @brief Profiles per tile of the CPU fused final-level scorer
 * (FFACpuEngine::run_final_batch). Time domain: about 64 KiB of profiles.
 * Fourier: a multiple of the merge step block and of the lane FFT width,
 * about 128 KiB of real profiles.
 */
[[nodiscard]] constexpr SizeType cpu_final_tile_coords(SizeType nbins,
                                                       bool fourier) noexcept {
    if (fourier) {
        return std::clamp<SizeType>(
            std::bit_floor((SizeType{32} * 1024) / (2 * nbins)), 32, 512);
    }
    return std::max<SizeType>(16, (SizeType{16} * 1024) / (2 * nbins));
}

/**
 * @brief Coordinates a CPU scored execution scores before it hands the hits
 * to the caller (`FFAEngine::execute_scored` with a sink). Hits are data
 * dependent (one per passing profile and width); handing them over batch by
 * batch bounds what is held at once.
 */
inline constexpr SizeType kCpuHitBatchCoords = SizeType{1} << 16U;

/**
 * @brief Most final-level rows (accel or jerk trials) of an accel plan that
 * read the same parent rows (a row group of the fused final level). A row's
 * parents are the trials of the level before nearest to its own, so per
 * parameter at most ceil(n_cur / (n_prev - 1)) + 1 trials share one.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_final_row_group_rows(const plans::FFAPlan<FoldType>& plan) {
    const auto& counts      = plan.get_param_counts();
    const SizeType n_params = plan.get_n_params();
    if (n_params < 2 || counts.size() < 2) {
        return 1;
    }
    const auto& cur  = counts.back();
    const auto& prev = counts[counts.size() - 2];
    SizeType rows    = 1;
    SizeType nrows   = 1;
    for (SizeType p = 0; p + 1 < n_params; ++p) {
        const SizeType step = std::max<SizeType>(prev[p], 2) - 1;
        rows *= ((cur[p] + step - 1) / step) + 1;
        nrows *= cur[p];
    }
    return std::min(rows, nrows);
}

/**
 * @brief Final-level coordinates whose hits a CPU scored execution holds at
 * once, at most. Frequency-only plans hand them over every
 * kCpuHitBatchCoords coordinates (whole tiles). Accel plans hand them over
 * after whole row groups (so tiles and hit order are those of one pass):
 * groups up to kCpuHitBatchCoords coordinates together, or one larger group.
 * @p batch_coords is a smaller batch (test hook).
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_hit_batch_coords(const plans::FFAPlan<FoldType>& plan,
                     SizeType batch_coords = kCpuHitBatchCoords) {
    const SizeType ncoords = plan.get_ncoords().back();
    if (plan.get_n_params() < 2) {
        return std::clamp<SizeType>(batch_coords, 1, ncoords);
    }
    const SizeType row_len = plan.get_param_counts().back().back();
    return std::min(
        ncoords,
        std::max(batch_coords, cpu_final_row_group_rows(plan) * row_len));
}

/**
 * @brief Bytes of hits a CPU scored execution holds at once: one hit per
 * (profile, width) of a batch at most, in per-thread vectors that at most
 * double their contents when they grow. They are freed after each batch.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_hit_buffer_bytes(const plans::FFAPlan<FoldType>& plan) {
    const SizeType nwidths = plan.get_config().get_n_scoring_widths();
    return 2 * cpu_hit_batch_coords(plan) * nwidths * sizeof(detection::SnrHit);
}

/**
 * @brief Upper bound on the heap scratch one OpenMP thread of a CPU scored
 * FFA allocates (`FFAEngine::execute_scored`, used by the CPU sweep).
 *
 * The merge paths run one after another, so the largest counts: the cone
 * bands, the fused or plain brute fold, the fused final level. The scorer
 * and the lane FFT of a thread live for the whole execution and are added.
 * Every term mirrors an allocation in `lib/cpu/kernels_cpu.cpp`,
 * `lib/cpu/ffa_cpu.cpp`, `lib/cpu/boxcar_kernels.hpp` or
 * `lib/cpu/lane_fft_cpu.cpp`; change them together.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_ffa_thread_scratch_bytes(const plans::FFAPlan<FoldType>& plan) {
    constexpr bool kFourier = std::is_same_v<FoldType, ComplexType>;
    const auto& cfg         = plan.get_config();
    const SizeType nbins    = cfg.get_nbins();
    const SizeType nbins_f  = (nbins / 2) + 1;
    const SizeType nwidths  = cfg.get_n_scoring_widths();
    const SizeType seg_len  = plan.get_segment_lens().front();
    const bool freq_only    = plan.get_n_params() == 1;
    const bool lossy        = kFourier && nbins > cfg.get_nbins_min_lossy_bf();
    constexpr SizeType kF   = sizeof(float);
    constexpr SizeType kD   = sizeof(double);
    constexpr SizeType kC   = sizeof(ComplexType);

    SizeType merge = 0;
    // Time-domain brute fold (brute_fold_ts; also the lossy Fourier base
    // fold): 4-segment prefix sums, single-segment prefix sums, fold rows.
    if (!kFourier || lossy) {
        merge = std::max(merge, (10 * (seg_len + 1) * kD) + (8 * nbins * kF));
    } else {
        // Direct Fourier brute fold (brute_fold_ts_complex): sub-block rows
        // and their rotated sums; the phasor tables are counted separately
        // (estimate_brutefold_table_bytes).
        const SizeType row_len = ((2 * nbins_f) + 15) / 16 * 16;
        merge = std::max(merge, ((4 * row_len) + (8 * nbins_f)) * kF);
    }
    if (freq_only) {
        // Cone bands. Time domain: the two tiles and the prefix sums fit the
        // budget; the 4-segment prefix copies (AVX2) double the prefix sums;
        // a band too narrow for one group per thread shares prefix sums of
        // at most 2 * budget per thread instead. Fourier: the two tiles.
        const SizeType budget = cpu_cone_scratch_budget_bytes();
        merge                 = std::max(merge, kFourier ? budget : 3 * budget);
        if (!kFourier) {
            // Fused brute fold of the plain path (execute_unified): two
            // tiles and the prefix sums of 2^k segments.
            merge =
                std::max(merge, (2 * kCpuFuseScratchBytes) +
                                    ((SizeType{2} << cpu_fuse_levels(plan)) *
                                     (seg_len + 1) * kD));
        }
    }
    // Fused final level: the tile, and for accel the level-(L-1) parents
    // (time: each channel twice; Fourier: block-aligned rows).
    const SizeType tile = cpu_final_tile_coords(nbins, kFourier);
    if (kFourier) {
        merge = std::max(
            merge,
            ((tile * 2 * nbins_f) + (2 * (tile + 128) * 2 * nbins_f)) * kC);
    } else {
        merge = std::max(
            merge, ((tile * 2 * nbins) + (2 * (tile + 64) * 4 * nbins)) * kF);
    }
    // Boxcar scorer (BoxcarScorer3D: 16 profiles in flight, wmax <= nbins)
    // and, for Fourier folds, the lane FFT (EVTransform: four buffers of 16
    // lanes of at most 4 * nbins points, and a row of zeros).
    const SizeType scorer =
        ((32 * nbins) + (32 * nwidths) + (4 * nbins) + (4 * nwidths)) * kF;
    const SizeType fft =
        kFourier ? ((4 * 16 * 4 * (nbins + 2)) + 128 + (nbins + 2)) * kF : 0;
    return merge + scorer + fft;
}

/**
 * @brief Bytes a CPU scored FFA of this plan allocates outside the fold
 * buffers, coordinates and brute-fold tables: per-thread scratch for
 * @p nthreads threads and the hits of one batch. The phase-step table is
 * allocated once per sweep (the largest cpu_phase_step_table_bytes of its
 * chunks) and not included.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_scored_scratch_bytes(const plans::FFAPlan<FoldType>& plan, int nthreads) {
    const auto n = static_cast<SizeType>(std::max(nthreads, 1));
    return (n * cpu_ffa_thread_scratch_bytes(plan)) +
           cpu_hit_buffer_bytes(plan);
}

} // namespace loki::algorithms::detail
