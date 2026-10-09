#pragma once

/**
 * @file ep_chunking.hpp
 * @brief Memory-bounded subdivision of coarse FFA regions into EP chunks.
 * Internal: used by EPRegionPlanner, and by white-box tests that feed it
 * synthetic region designs.
 */

#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"

namespace loki::algorithms::detail {

/// Everything the chunking needs to know about one coarse FFA region. Built
/// once per region (the threshold scheme is expensive).
struct RegionDesign {
    double f_start{0.0};
    double f_end{0.0};
    SizeType nbins{0};
    double eta{0.0};
    std::vector<float> threshold_scheme;
    std::vector<float> bp_float;
    float peak_complexity{1.0F};
    /// max_sugg of a chunk is ceil(ncoords * safe_complexity), at least 1024.
    float safe_complexity{1.0F};
    SizeType nsegments{0};
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
