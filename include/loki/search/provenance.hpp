#pragma once

/**
 * @file provenance.hpp
 * @brief What produced a results file: the effective configuration, the
 * command line and the loki version, stored as root attributes so that two
 * results files can be compared from the files alone.
 */

#include <filesystem>
#include <string>

namespace loki::search {

/// @brief How a results file was produced.
struct RunProvenance {
    /// The configuration the run used, after defaults and command-line
    /// overrides, as TOML (FFATomlConfig / EPTomlConfig::to_toml_string()).
    std::string config_effective;
    /// The command that started the run.
    std::string command_line;
    /// The loki version that wrote the file.
    std::string loki_version;
};

/// @brief Writes @p provenance as root attributes of the HDF5 file
/// @p results_file: `config_effective`, `command_line` and `loki_version`.
/// Empty fields are skipped; an attribute already present is replaced. The
/// raw `config_toml` attribute of an FFA results file is left as it is.
/// @throws std::runtime_error if the file cannot be opened for writing.
void write_run_provenance(const std::filesystem::path& results_file,
                          const RunProvenance& provenance);

} // namespace loki::search
