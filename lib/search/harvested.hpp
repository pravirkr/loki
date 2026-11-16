#pragma once

/**
 * @file harvested.hpp
 * @brief The `/harvested` table of an EP result: early-harvested sources,
 * merged over runs. Internal.
 */

#include "loki/common/types.hpp"

namespace HighFive {
class File;
} // namespace HighFive

namespace loki::search {

/**
 * @brief Writes `/harvested` from the harvest records of every run
 * (`chunks/<c>/runs/<r>/harvest`), one row per source.
 *
 * @details Records whose harvest windows (tile padding plus the Doppler
 * sweep, as the pruning masks them) overlap in frequency are one source; a
 * row holds its strongest record (parameters at that record's `t_ref`,
 * partial S/N at its harvest level) and how many records and runs saw it.
 * Rows are sorted by S/N, strongest first. Integer and fractional harmonics
 * of a source are separate rows. Replaces an existing `/harvested`.
 *
 * @return Number of sources. Without any harvest record nothing is written
 * and the result is 0.
 */
SizeType write_harvested(HighFive::File& file);

} // namespace loki::search
