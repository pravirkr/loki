#pragma once

/**
 * @file planner_cost.hpp
 * @brief Run-time model of an FFA frequency-sweep chunk on the CPU. Internal.
 */

#include <cstddef>
#include <span>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"

#include "lib/algorithms/planner_memory.hpp"

namespace loki::algorithms::detail {

/**
 * @brief Seconds per work unit of each FFA stage, plus a fixed cost per chunk.
 *
 * A level's work unit is one output element of the (e, v) pair,
 * nsegments * ncoords * 2 * nbins, the count `LOKI_FFA_LEVEL_TIMING` logs.
 * The weights come from that timing on one core (Xeon Gold 6348H, nthreads =
 * 1, accel sweeps W3 and W4 of the CPU FFA cycle); they set the trade-off
 * between chunks, not an absolute run time, so their ratios matter more than
 * their values.
 */
struct ChunkCostModel {
    double brute_s;     ///< brute fold, per level-0 unit
    double merge_s;     ///< stored merge levels, per unit
    double prev_s;      ///< level L-1 merged in scratch (two fused levels)
    double final_s;     ///< final level merged and scored, per unit
    double per_chunk_s; ///< fixed cost of a chunk (plan, tables, engine)
};

/// CPU sweep cost model for the time or the Fourier domain.
[[nodiscard]] constexpr ChunkCostModel
cpu_chunk_cost_model(bool use_fourier) noexcept {
    if (use_fourier) {
        return {.brute_s     = 4.9e-9,
                .merge_s     = 0.81e-9,
                .prev_s      = 0.88e-9,
                .final_s     = 1.2e-9,
                .per_chunk_s = 0.03};
    }
    return {.brute_s     = 3.2e-9,
            .merge_s     = 0.68e-9,
            .prev_s      = 0.4e-9,
            .final_s     = 0.65e-9,
            .per_chunk_s = 0.03};
}

/**
 * @brief Modelled run time (seconds) of one chunk from its plan.
 *
 * Counts every level of the chunk's own grid, so it sees what a wider chunk
 * costs: an accel (jerk) grid designed at the chunk's highest frequency
 * oversamples the low end, and the trial count grows with the width.
 */
template <SupportedFoldType FoldType>
[[nodiscard]] double estimate_chunk_cost(const plans::FFAPlan<FoldType>& plan,
                                         const ChunkCostModel& model) {
    const std::span<const SizeType> nsegments = plan.get_nsegments();
    const std::span<const SizeType> ncoords   = plan.get_ncoords();
    const SizeType n_levels                   = plan.get_n_levels();
    const auto nbins = static_cast<double>(plan.get_config().get_nbins());
    const auto units = [&](SizeType level) {
        return static_cast<double>(nsegments[level]) *
               static_cast<double>(ncoords[level]) * 2.0 * nbins;
    };
    const bool fused_prev = cpu_scored_fused_levels(plan) == 2;
    double cost           = model.per_chunk_s + (model.brute_s * units(0));
    for (SizeType level = 1; level + 1 < n_levels; ++level) {
        const bool prev = fused_prev && level + 2 == n_levels;
        cost += (prev ? model.prev_s : model.merge_s) * units(level);
    }
    cost += model.final_s * units(n_levels - 1);
    return cost;
}

} // namespace loki::algorithms::detail
