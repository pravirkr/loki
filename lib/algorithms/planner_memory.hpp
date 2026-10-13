#pragma once

/**
 * @file planner_memory.hpp
 * @brief Memory reserve shared by the FFA and EP region planners. Internal.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms::detail {

/**
 * @brief Process memory (GiB) that the planners' memory models do not cover.
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
 * @brief Free device memory (GiB) the GPU planners leave alone: the CUDA
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
 * @brief Fold buffer size (elements) a CPU FFA needs for scored execution
 * (`execute_scored`, used by the CPU sweep).
 *
 * Accel searches merge the final level tile by tile into scratch and score
 * it there, so the ping-pong buffers hold every level but the last (accel
 * trees widen, so the last level is the largest). Frequency-only trees are
 * flat and the cone path may store the top level, so they keep the full
 * buffer.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] SizeType
cpu_scored_buffer_size(const plans::FFAPlan<FoldType>& plan) {
    const auto& shapes = plan.get_fold_shapes();
    if (plan.get_n_params() < 2 || shapes.size() < 2 ||
        plan.get_nsegments().back() != 1) {
        return plan.get_buffer_size();
    }
    SizeType size = 0;
    for (SizeType i = 0; i + 1 < shapes.size(); ++i) {
        size =
            std::max(size, std::accumulate(shapes[i].begin(), shapes[i].end(),
                                           SizeType{1}, std::multiplies<>()));
    }
    return size;
}

} // namespace loki::algorithms::detail
