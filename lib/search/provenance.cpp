#include "loki/search/provenance.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <hdf5.h>
#include <highfive/H5File.hpp>
#include <highfive/highfive.hpp>

#include "lib/generated/git_commit.hpp"
#include "lib/io/h5_layout.hpp"

namespace loki::search {

namespace {

/// The sum of one unsigned member (e.g. `n_overloads`) of a run's
/// `level_stats` compound table, read alone (HDF5 converts a compound by
/// member name).
std::uint64_t sum_level_stat(const HighFive::DataSet& level_stats,
                             const char* name) {
    const auto n = level_stats.getSpace().getElementCount();
    if (n == 0) {
        return 0;
    }
    const hid_t member = H5Tcreate(H5T_COMPOUND, sizeof(std::uint64_t));
    if (member < 0 || H5Tinsert(member, name, 0, H5T_NATIVE_UINT64) < 0) {
        throw std::runtime_error("write_run_stats: cannot build HDF5 type");
    }
    std::vector<std::uint64_t> values(n);
    const herr_t status = H5Dread(level_stats.getId(), member, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data());
    H5Tclose(member);
    if (status < 0) {
        throw std::runtime_error(
            fmt::format("write_run_stats: cannot read level_stats.{}", name));
    }
    std::uint64_t total = 0;
    for (const auto v : values) {
        total += v;
    }
    return total;
}

void write_ep_run_counts(HighFive::File& file, HighFive::Group& out) {
    if (!file.exist("chunks")) {
        return;
    }
    std::uint64_t n_runs       = 0;
    std::uint64_t n_overloaded = 0;
    std::uint64_t n_overloads  = 0;
    double plan_peak_memory_gb = 0.0;
    // Harvests ([ep.rfi]): written only when a run has a harvest group.
    bool any_harvest                = false;
    std::uint64_t n_harvested       = 0;
    std::uint64_t n_harvests_kept   = 0;
    std::uint64_t max_harvested_run = 0;
    // Leaves rejected by the pulsar mask and by the impulsive veto: written
    // when a chunk ran with RFI control, so a no-op is visible.
    bool any_rfi           = false;
    std::uint64_t n_masked = 0;
    std::uint64_t n_vetoed = 0;
    const auto chunks      = file.getGroup("chunks");
    for (const auto& chunk_name : chunks.listObjectNames()) {
        const auto chunk = chunks.getGroup(chunk_name);
        if (chunk.hasAttribute("chunk_memory_gb")) {
            double gb = 0.0;
            chunk.getAttribute("chunk_memory_gb").read(gb);
            plan_peak_memory_gb = std::max(plan_peak_memory_gb, gb);
        }
        const bool chunk_rfi = chunk.exist("rfi_config");
        any_rfi              = any_rfi || chunk_rfi;
        if (!chunk.exist("runs")) {
            continue;
        }
        const auto runs = chunk.getGroup("runs");
        for (const auto& run_name : runs.listObjectNames()) {
            const auto run = runs.getGroup(run_name);
            ++n_runs;
            if (run.exist("harvest")) {
                const auto harvest  = run.getGroup("harvest");
                std::uint64_t total = 0;
                if (harvest.hasAttribute("n_harvested_total")) {
                    harvest.getAttribute("n_harvested_total").read(total);
                }
                any_harvest = true;
                n_harvested += total;
                max_harvested_run = std::max(max_harvested_run, total);
                if (harvest.exist("scores")) {
                    n_harvests_kept += harvest.getDataSet("scores")
                                           .getSpace()
                                           .getElementCount();
                }
            }
            if (!run.exist("level_stats")) {
                continue;
            }
            const auto level_stats = run.getDataSet("level_stats");
            const auto overloads   = sum_level_stat(level_stats, "n_overloads");
            n_overloads += overloads;
            n_overloaded += overloads > 0 ? 1 : 0;
            if (chunk_rfi) {
                n_masked += sum_level_stat(level_stats, "n_leaves_masked");
                n_vetoed += sum_level_stat(level_stats, "n_leaves_vetoed");
            }
        }
    }
    out.createAttribute("n_runs", n_runs);
    out.createAttribute("n_runs_overloaded", n_overloaded);
    out.createAttribute("n_overloads", n_overloads);
    out.createAttribute("plan_peak_memory_gb", plan_peak_memory_gb);
    if (any_harvest) {
        out.createAttribute("n_harvested", n_harvested);
        out.createAttribute("n_harvests_recorded", n_harvests_kept);
        out.createAttribute("max_harvested_per_run", max_harvested_run);
    }
    if (file.exist("harvested")) {
        std::uint64_t n_sources = 0;
        const auto harvested    = file.getGroup("harvested");
        if (harvested.hasAttribute("n_sources")) {
            harvested.getAttribute("n_sources").read(n_sources);
        }
        out.createAttribute("n_harvested_sources", n_sources);
    }
    if (any_rfi) {
        out.createAttribute("n_leaves_masked", n_masked);
        out.createAttribute("n_leaves_vetoed", n_vetoed);
    }
}

/// FNV-1a, 64 bit: a stable hash of the effective configuration text, to
/// group runs made with the same configuration.
std::uint64_t fnv1a64(std::string_view text) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

/// Version and commit of this build, and the configuration hash and input
/// path of the run when the file records them.
void write_provenance_summary(HighFive::File& file, HighFive::Group& out) {
    out.createAttribute("loki_version", std::string(LOKI_PRODUCER_VERSION));
    out.createAttribute("loki_commit", std::string(LOKI_GIT_COMMIT));
    if (file.hasAttribute("config_effective")) {
        const auto config =
            file.getAttribute("config_effective").read<std::string>();
        out.createAttribute("config_hash",
                            fmt::format("fnv1a64:{:016x}", fnv1a64(config)));
    }
    if (file.exist("observation") &&
        file.getGroup("observation").hasAttribute("input_path")) {
        out.createAttribute("input_path", file.getGroup("observation")
                                              .getAttribute("input_path")
                                              .read<std::string>());
    }
}

/// The highest S/N the cell cap dropped in any reduced EP run.
void write_max_cell_dropped_snr(HighFive::File& file, HighFive::Group& out) {
    if (!file.exist("chunks")) {
        return;
    }
    bool any          = false;
    float most        = 0;
    const auto chunks = file.getGroup("chunks");
    for (const auto& chunk_name : chunks.listObjectNames()) {
        const auto chunk = chunks.getGroup(chunk_name);
        if (!chunk.exist("runs")) {
            continue;
        }
        const auto runs = chunk.getGroup("runs");
        for (const auto& run_name : runs.listObjectNames()) {
            const auto run = runs.getGroup(run_name);
            if (run.hasAttribute("max_cell_dropped_snr")) {
                any  = true;
                most = std::max(
                    most,
                    run.getAttribute("max_cell_dropped_snr").read<float>());
            }
        }
    }
    if (any) {
        out.createAttribute("max_cell_dropped_snr", most);
    }
}

void write_detections_per_nbins(HighFive::File& file, HighFive::Group& out) {
    if (!file.exist("detections/nbins")) {
        return;
    }
    const auto dataset = file.getDataSet("detections/nbins");
    const auto n       = dataset.getSpace().getElementCount();
    std::map<std::uint32_t, std::uint64_t> counts;
    constexpr std::size_t kBatch = std::size_t{1} << 20U;
    std::vector<std::uint32_t> rows;
    for (std::size_t start = 0; start < n; start += kBatch) {
        const auto count = std::min(kBatch, n - start);
        rows.resize(count);
        dataset.select({start}, {count}).read(rows);
        for (const auto nbins : rows) {
            ++counts[nbins];
        }
    }
    std::vector<std::uint32_t> nbins;
    std::vector<std::uint64_t> per;
    for (const auto& [key, value] : counts) {
        nbins.push_back(key);
        per.push_back(value);
    }
    auto group = out.createGroup("detections_per_nbins");
    group.createDataSet("nbins", nbins);
    group.createDataSet("count", per);
}

} // namespace

void write_run_provenance(const std::filesystem::path& results_file,
                          const RunProvenance& provenance) {
    if (!std::filesystem::exists(results_file)) {
        throw std::runtime_error(
            fmt::format("write_run_provenance: results file not found: '{}'",
                        results_file.string()));
    }
    HighFive::File file = io::detail::open_result_file(
        results_file.string(), HighFive::File::ReadWrite);
    const auto write = [&](std::string_view name, const std::string& value) {
        if (value.empty()) {
            return;
        }
        const std::string key(name);
        if (file.hasAttribute(key)) {
            file.deleteAttribute(key);
        }
        file.createAttribute(key, value);
    };
    write("config_effective", provenance.config_effective);
    write("command_line", provenance.command_line);
    write("loki_version", provenance.loki_version);
    write("publish_command_line", provenance.publish_command_line);
    if (!provenance.config_toml.empty()) {
        if (!file.hasAttribute("config_toml")) {
            file.createAttribute("config_toml", provenance.config_toml);
        }
        if (file.exist("search")) {
            auto search = file.getGroup("search");
            if (!search.hasAttribute("config_toml")) {
                search.createAttribute("config_toml", provenance.config_toml);
            }
        }
    }
    file.flush();
}

void write_run_stats(const std::filesystem::path& results_file,
                     const RunStats& stats) {
    if (!std::filesystem::exists(results_file)) {
        throw std::runtime_error(
            fmt::format("write_run_stats: results file not found: '{}'",
                        results_file.string()));
    }
    HighFive::File file = io::detail::open_result_file(
        results_file.string(), HighFive::File::ReadWrite);
    if (file.exist("run_stats")) {
        file.unlink("run_stats");
    }
    auto out = file.createGroup("run_stats");
    out.createAttribute(
        "description",
        std::string("measured cost of the run (volatile; not a search "
                    "result): wall time per stage, peak RSS, overloads"));
    for (const auto& [stage, seconds] : stats.timing_s) {
        out.createAttribute(fmt::format("timing_{}_s", stage), seconds);
    }
    if (stats.peak_rss_gb) {
        out.createAttribute("peak_rss_gb", *stats.peak_rss_gb);
    }
    write_ep_run_counts(file, out);
    write_provenance_summary(file, out);
    write_max_cell_dropped_snr(file, out);
    if (file.hasAttribute("n_cell_dropped") &&
        file.hasAttribute("n_dominated")) {
        std::uint64_t cell      = 0;
        std::uint64_t dominated = 0;
        file.getAttribute("n_cell_dropped").read(cell);
        file.getAttribute("n_dominated").read(dominated);
        out.createAttribute("leaves_dropped", cell + dominated);
    }
    write_detections_per_nbins(file, out);
    file.flush();
}

} // namespace loki::search
