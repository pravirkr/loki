#pragma once

/**
 * @file ep_chunking.hpp
 * @brief Memory-bounded subdivision of coarse FFA regions into EP chunks.
 * Internal: used by EPRegionPlanner, and by white-box tests that feed it
 * synthetic region designs.
 */

#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"

namespace loki::algorithms::detail {

/// Thresholds designed for one region's branching pattern, with the noise
/// population forecast of their own independent evaluation.
struct ChunkDesign {
    std::vector<float> thresholds{};
    /// Pd at the chunk's nbins on a fresh signal sample, and its standard
    /// error.
    double pd{1.0};
    double pd_se{0.0};
    EPPopulationForecast forecast{};
};

/// Everything the chunking needs to know about one coarse FFA region.
struct RegionDesign {
    double f_start{0.0};
    double f_end{0.0};
    SizeType nbins{0};
    double eta{0.0};
    SizeType nsegments{0};
    /// The threshold design of the region's branching pattern over the
    /// nominal band [f_start, f_end] (the band is for logs). Called once per
    /// region; every chunk runs it.
    std::function<ChunkDesign(std::span<const float> bp, double, double)>
        design{};
    /// Synthetic exact per-stage complexity used by internal tests when
    /// design is unset.
    float safe_complexity{1.0F};
};

/// Chunks of a sweep with the maxima EPRegionStats reports.
struct ChunkPlan {
    std::vector<EPChunkConfig> chunk_cfgs;
    std::vector<EPChunkStats> chunk_stats;
    SizeType max_sugg{0};
    SizeType max_ncoords{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    SizeType fold_size{0};
    /// Peak sweep memory (see ep_memory.hpp), in GiB.
    double peak_memory_gb{0.0};
    /// True if a pass with a fixed shared FFA size was needed.
    bool replanned{false};
    /// Fraction of the first pass's shared FFA size used as the fixed cap (1
    /// unless the replan at the full size was infeasible).
    double shared_cap_scale{1.0};
};

/**
 * @brief Subdivides @p designs (in order) into chunks that fit
 * @p effective_limit_gb.
 *
 * Each region is designed once over its whole band (RegionDesign::design)
 * and split from its top frequency down; every chunk runs the region
 * schedule. A chunk's population is that design's on the chunk's own
 * branching pattern: the schedule's pass rates do not depend on the
 * branching, so they carry over exactly and no probe simulates anything. The
 * widest chunk whose sweep fits is found by bisection on the width.
 *
 * Per-worker memory is accounted per contiguous run of equal nbins and the
 * FFA buffers and inputs once for the whole sweep, as EPFreqSweep allocates
 * them. The shared FFA size is only known once all chunks are planned: if it
 * grew after an earlier run was planned and that run no longer fits, the plan
 * is redone with the shared size fixed to the first pass's maximum. If that
 * is infeasible, a bounded search looks for a smaller fixed size that fits.
 *
 * @throws std::runtime_error if even a minimum-width chunk does not fit.
 */
template <SupportedFoldType FoldType>
ChunkPlan plan_chunks(const search::PulsarSearchConfig& base_cfg,
                      std::string_view poly_basis,
                      std::span<const RegionDesign> designs,
                      double max_drift,
                      double effective_limit_gb,
                      const EPMemoryContext& memory);

} // namespace loki::algorithms::detail
