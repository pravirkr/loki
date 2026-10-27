#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace loki::io {

/// Metadata belonging to a single-DM timeseries, not a DM search plan.
/// Disengaged values are unknown; DM zero is a valid known value.
struct ObservationMetadata {
    std::optional<double> dm;
    std::optional<double> start_mjd;
    std::optional<bool> barycentric;
    std::string source;
    std::string input_path;
    std::uint64_t original_nsamps{};
    void validate() const;
};

} // namespace loki::io
