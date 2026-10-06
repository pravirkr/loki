#pragma once

#include <cstdint>

#include "loki/common/types.hpp"

namespace loki::coord {

// FFA Coordinate plan for a single param coordinate in a single iteration
struct FFACoord {
    uint32_t i_tail;  // Tail coordinate index in the previous iteration
    float shift_tail; // Phase bin shift in the tail coordinate
    uint32_t i_head;  // Head coordinate index in the previous iteration
    float shift_head; // Phase bin shift in the head coordinate
};

struct FFACoordFreq {
    uint32_t idx; // Phase bin index in the previous iteration
    float shift;  // Phase bin shift
};

/**
 * @brief One contiguous run of samples that land in the same phase bin.
 *
 * Runs of one frequency are ordered and cover a brute-fold segment without
 * gaps: the first run starts at sample 0 and each run ends at `end`
 * (exclusive). Phase is linear in time, so each visit to a bin (including
 * after a wrap) is one run.
 */
struct PhaseRun {
    uint32_t end; ///< Exclusive sample index within the segment.
    uint32_t bin; ///< Phase bin in `[0, nbins)`.
};

// A structure to hold the parameters for a single FFA search region.
struct FFARegion {
    double f_start; // Hz, inclusive (lower frequency)
    double f_end;   // Hz, inclusive (upper frequency)
    SizeType nbins; // fixed within region
    double eta;     // tolerance in bins for this region
};

// A structure to hold the stats for a single FFA search chunk.
struct FFAChunkStats {
    double nominal_f_start;
    double nominal_f_end;
    double actual_f_start;
    double actual_f_end;
    double nominal_width;
    double actual_width;
    double total_memory_gb;
    double overlap_fraction; // fraction of actual range that's overlap
};

} // namespace loki::coord