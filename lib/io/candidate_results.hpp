#pragma once

#include <filesystem>
#include <optional>

#include "loki/io/observation.hpp"
#include "loki/search/candidates.hpp"

namespace loki::io::detail {
/// Convert the internal FFA spool to the common schema. Publishes atomically.
void write_candidate_results(const std::filesystem::path& spool,
                             const std::filesystem::path& output,
                             const ObservationMetadata& observation,
                             const search::CandidateOptions& options);

/// Read completed Taylor EP runs, then share sorting/grouping/publication.
void write_ep_candidate_results(const std::filesystem::path& spool,
                                const std::filesystem::path& output,
                                const ObservationMetadata& observation,
                                const search::CandidateOptions& options,
                                std::optional<double> snr_min = std::nullopt);
} // namespace loki::io::detail
