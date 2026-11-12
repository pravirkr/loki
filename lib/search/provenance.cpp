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

namespace loki::search {

namespace {

/// The `n_overloads` member of a run's `level_stats` compound table, read
/// alone (HDF5 converts a compound by member name).
std::uint64_t sum_overloads(const HighFive::DataSet& level_stats) {
    const auto n = level_stats.getSpace().getElementCount();
    if (n == 0) {
        return 0;
    }
    const hid_t member = H5Tcreate(H5T_COMPOUND, sizeof(std::uint64_t));
    if (member < 0 ||
        H5Tinsert(member, "n_overloads", 0, H5T_NATIVE_UINT64) < 0) {
        throw std::runtime_error("write_run_stats: cannot build HDF5 type");
    }
    std::vector<std::uint64_t> values(n);
    const herr_t status = H5Dread(level_stats.getId(), member, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data());
    H5Tclose(member);
    if (status < 0) {
        throw std::runtime_error(
            "write_run_stats: cannot read level_stats.n_overloads");
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
    const auto chunks          = file.getGroup("chunks");
    for (const auto& chunk_name : chunks.listObjectNames()) {
        const auto chunk = chunks.getGroup(chunk_name);
        if (chunk.hasAttribute("chunk_memory_gb")) {
            double gb = 0.0;
            chunk.getAttribute("chunk_memory_gb").read(gb);
            plan_peak_memory_gb = std::max(plan_peak_memory_gb, gb);
        }
        if (!chunk.exist("runs")) {
            continue;
        }
        const auto runs = chunk.getGroup("runs");
        for (const auto& run_name : runs.listObjectNames()) {
            const auto run = runs.getGroup(run_name);
            ++n_runs;
            if (!run.exist("level_stats")) {
                continue;
            }
            const auto overloads = sum_overloads(run.getDataSet("level_stats"));
            n_overloads += overloads;
            n_overloaded += overloads > 0 ? 1 : 0;
        }
    }
    out.createAttribute("n_runs", n_runs);
    out.createAttribute("n_runs_overloaded", n_overloaded);
    out.createAttribute("n_overloads", n_overloads);
    out.createAttribute("plan_peak_memory_gb", plan_peak_memory_gb);
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
    HighFive::File file(results_file.string(), HighFive::File::ReadWrite);
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
    file.flush();
}

void write_run_stats(const std::filesystem::path& results_file,
                     const RunStats& stats) {
    if (!std::filesystem::exists(results_file)) {
        throw std::runtime_error(
            fmt::format("write_run_stats: results file not found: '{}'",
                        results_file.string()));
    }
    HighFive::File file(results_file.string(), HighFive::File::ReadWrite);
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
    write_detections_per_nbins(file, out);
    file.flush();
}

} // namespace loki::search
