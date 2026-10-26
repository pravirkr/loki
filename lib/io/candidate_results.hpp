#pragma once

#include <filesystem>

#include "loki/io/observation.hpp"
#include "loki/search/candidates.hpp"

namespace loki::io::detail {
/// Convert the internal FFA spool to the common schema. Publishes atomically.
void write_candidate_results(const std::filesystem::path& spool,
                             const std::filesystem::path& output,
                             const ObservationMetadata& observation,
                             const search::CandidateOptions& options);
} // namespace loki::io::detail
