#pragma once

/**
 * @file provenance.hpp
 * @brief What produced a results file: the effective configuration, the
 * command line and the loki version, stored as root attributes so that two
 * results files can be compared from the files alone.
 */

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loki::search {

/// @brief How a results file was produced.
struct RunProvenance {
    /// The configuration the run used, after defaults and command-line
    /// overrides, as TOML (FFATomlConfig / EPTomlConfig::to_toml_string()).
    std::string config_effective{};
    /// The command that started the run.
    std::string command_line{};
    /// The loki version that wrote the file.
    std::string loki_version{};
    /// The raw TOML configuration text, recorded where missing (never replaces
    /// an existing attribute on root or `/search`).
    std::string config_toml{};
    /// The `loki publish` command line that published a spool into a product.
    std::string publish_command_line{};
};

/// @brief Writes @p provenance as root attributes of the HDF5 file
/// @p results_file: `config_effective`, `command_line`, `loki_version` and
/// `publish_command_line` (empty fields are skipped; an attribute already
/// present is replaced), plus `config_toml` where not yet stored on root or
/// `/search`.
/// @throws std::runtime_error if the file cannot be opened for writing.
void write_run_provenance(const std::filesystem::path& results_file,
                          const RunProvenance& provenance);

/// @brief Measured cost of a run, for batch scheduling (wall time and memory
/// a job needs). Volatile by nature: two identical runs differ here.
struct RunStats {
    /// Wall time per stage in seconds, in run order, e.g. {"load", ...},
    /// {"preprocess", ...}, {"plan", ...}, {"search", ...},
    /// {"postprocess", ...}, {"total", ...}.
    std::vector<std::pair<std::string, double>> timing_s;
    /// Peak resident set size of the process in GB (2^30 bytes), as
    /// getrusage(RUSAGE_SELF).ru_maxrss.
    std::optional<double> peak_rss_gb;
};

/// @brief Writes @p stats, plus counts derived from the file itself, as
/// attributes of the group `/run_stats` of @p results_file (replacing an
/// existing group):
/// - `timing_<stage>_s` for each stage and `peak_rss_gb`;
/// - EP files: `n_runs`, `n_runs_overloaded` and `n_overloads` (sums of the
///   runs' `level_stats.n_overloads`), and `plan_peak_memory_gb` (the largest
///   chunk `chunk_memory_gb`: device memory on CUDA);
/// - files with a `/detections` table: datasets `detections_per_nbins/nbins`
///   and `detections_per_nbins/count` (one row per profile bin count, i.e.
///   per search region).
/// @throws std::runtime_error if the file cannot be opened for writing.
void write_run_stats(const std::filesystem::path& results_file,
                     const RunStats& stats);

} // namespace loki::search
