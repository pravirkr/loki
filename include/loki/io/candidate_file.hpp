#pragma once

/**
 * @file candidate_file.hpp
 * @brief Read-only summary of an FFA or EP result file: run metadata and the
 * top candidates by S/N, for `loki inspect` and other tools without Python.
 *
 * Three layouts are understood, the same ones as the Python reader
 * (`loki.candidates.CandidateFile`) plus raw EP runs:
 * - the common candidate schema (1.x) written by FFA and EP searches;
 * - the legacy flat FFA file (detections at the root, `ffa_version`);
 * - the raw EP sweep file (`ep_sweep_version`, `chunks/<c>/runs/<r>`),
 *   written when candidate processing is off.
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace loki::io {

/// One searched parameter range, over every region or chunk of the search.
struct ParamRange {
    std::string name;
    double min{};
    double max{};
};

/// One row of the candidate table.
struct CandidateRow {
    std::uint64_t row{}; ///< Row in the table read (raw EP: leaf of the run).
    /// Native parameters in the order of `CandidateFileSummary::param_names`
    /// (frequency last, in Hz; accel in m/s^2; jerk in m/s^3).
    std::vector<double> params;
    float snr{};
    std::uint32_t width{}; ///< Boxcar width in bins; 0 when not stored.
    std::uint32_t nbins{}; ///< Folding bins; 0 when not stored.
    /// Where the row came from: "region <i>" (FFA), "chunk <c> run <r>" (EP),
    /// or empty when the file does not say.
    std::string source;
};

struct CandidateFileSummary {
    std::filesystem::path path;
    /// "common", "legacy_ffa" or "ep_runs".
    std::string layout;
    std::string algorithm;        ///< "ffa" or "ep".
    std::string schema_version;   ///< Empty for the legacy and raw layouts.
    bool complete{false};         ///< false: an interrupted or partial file.
    std::string producer_version; ///< Loki version; empty when not stored.
    std::string writer_version;   ///< ffa_version / ep_sweep_version.
    /// The execution backend; not recorded by the current writers.
    std::optional<std::string> backend;
    std::string source_name; ///< Observation source; may be empty.
    std::string input_path;  ///< Input timeseries; may be empty.
    std::uint64_t nsamps{};  ///< Samples searched.
    double tsamp{};          ///< Seconds.
    double tobs{};           ///< Seconds.
    std::optional<double> dm;
    std::optional<double> snr_min;
    std::vector<std::string> param_names;
    std::vector<ParamRange> ranges;
    /// The search configuration as written (FFA searches store it).
    std::optional<std::string> config_toml;
    /// Run provenance (root attributes, see loki/search/provenance.hpp):
    /// the configuration after defaults and command-line overrides, the
    /// command that started the run and the loki version that wrote it.
    std::optional<std::string> config_effective;
    std::optional<std::string> command_line;
    std::optional<std::string> loki_version;
    std::string table;      ///< Table read: "candidates", "/" or "chunks".
    std::uint64_t n_rows{}; ///< Rows in that table.
    std::optional<std::uint64_t> n_detections;
    /// The top rows by S/N, highest first (ties: lower row first).
    std::vector<CandidateRow> top;
    /// The folded profile of the top candidate (the `folds` row of the
    /// table, signal over the root of its variance), when the file stores
    /// one. Current searches do not.
    std::optional<std::vector<float>> top_profile;
};

/// Reads the metadata and the @p top_n best rows of @p path. Throws
/// std::invalid_argument for a file of another layout and
/// std::runtime_error when the file cannot be read.
[[nodiscard]] CandidateFileSummary
read_candidate_file_summary(const std::filesystem::path& path,
                            std::size_t top_n = 10);

} // namespace loki::io
