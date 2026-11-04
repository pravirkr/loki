#include "loki/io/candidate_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iterator>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <hdf5.h>
#include <highfive/highfive.hpp>

namespace loki::io {
namespace {

constexpr std::size_t kBatchRows = 65536;

template <typename T, typename Node>
std::optional<T> attribute(const Node& node, const std::string& name) {
    if (!node.hasAttribute(name)) {
        return std::nullopt;
    }
    T value{};
    node.getAttribute(name).read(value);
    return value;
}

// A text attribute. Writers store text as an HDF5 string or, for a C
// string literal, as an array of chars (e.g. "ffa_version").
template <typename Node>
std::optional<std::string> text_attribute(const Node& node,
                                          const std::string& name) {
    if (!node.hasAttribute(name)) {
        return std::nullopt;
    }
    const auto attr = node.getAttribute(name);
    const auto dims = attr.getSpace().getDimensions();
    if (attr.getDataType().getClass() == HighFive::DataTypeClass::String) {
        if (dims.empty()) {
            std::string value;
            attr.read(value);
            return value;
        }
        std::vector<std::string> parts;
        attr.read(parts);
        std::string value;
        for (const auto& part : parts) {
            value += part;
        }
        return value;
    }
    if (attr.getDataType().getClass() == HighFive::DataTypeClass::Integer &&
        attr.getDataType().getSize() == 1 && dims.size() == 1) {
        std::vector<char> chars(dims[0]);
        attr.read_raw(chars.data(), HighFive::AtomicType<char>());
        std::string value(chars.begin(), chars.end());
        value.erase(std::ranges::find(value, '\0'), value.end());
        return value;
    }
    return std::nullopt;
}

template <typename T, typename Node>
T attribute_or(const Node& node, const std::string& name, T fallback) {
    return attribute<T>(node, name).value_or(std::move(fallback));
}

// A (score, row) pair; the queue keeps the lowest of the best rows on top.
struct Ranked {
    float snr;
    std::uint64_t row;
    bool operator<(const Ranked& other) const {
        // Higher S/N ranks first, then the lower row.
        return snr != other.snr ? snr > other.snr : row < other.row;
    }
};

class TopRows {
public:
    explicit TopRows(std::size_t limit) : m_limit(limit) {}

    void offer(float snr, std::uint64_t row) {
        if (m_limit == 0) {
            return;
        }
        const Ranked entry{.snr = snr, .row = row};
        if (m_heap.size() < m_limit) {
            m_heap.push(entry);
        } else if (entry < m_heap.top()) {
            m_heap.pop();
            m_heap.push(entry);
        }
    }

    /// The kept rows, best first.
    [[nodiscard]] std::vector<Ranked> take() {
        std::vector<Ranked> rows;
        rows.reserve(m_heap.size());
        while (!m_heap.empty()) {
            rows.push_back(m_heap.top());
            m_heap.pop();
        }
        std::ranges::reverse(rows);
        return rows;
    }

private:
    std::size_t m_limit;
    std::priority_queue<Ranked> m_heap;
};

void rank_column(const HighFive::DataSet& snr,
                 std::uint64_t row_offset,
                 TopRows& top) {
    const auto nrows = snr.getDimensions().at(0);
    std::vector<float> values;
    for (std::size_t start = 0; start < nrows; start += kBatchRows) {
        const auto count = std::min(kBatchRows, nrows - start);
        snr.select({start}, {count}).read(values);
        for (std::size_t i = 0; i < count; ++i) {
            top.offer(values[i], row_offset + start + i);
        }
    }
}

template <typename T>
T read_cell(const HighFive::Group& table,
            const std::string& name,
            std::uint64_t row) {
    std::vector<T> value;
    table.getDataSet(name).select({row}, {1}).read(value);
    return value.at(0);
}

std::vector<double> read_param_row(const HighFive::DataSet& params,
                                   std::uint64_t row,
                                   std::size_t nparams) {
    std::vector<double> values(nparams);
    params.select({row, 0}, {1, nparams}).read_raw(values.data());
    return values;
}

// The folded profile of one row of a table that stores `folds`
// ([rows, 2, nbins]: signal and variance).
std::optional<std::vector<float>> read_profile(const HighFive::Group& table,
                                               std::uint64_t row) {
    if (!table.exist("folds")) {
        return std::nullopt;
    }
    const auto folds = table.getDataSet("folds");
    const auto dims  = folds.getDimensions();
    if (dims.size() != 3 || dims[1] != 2 || row >= dims[0] ||
        folds.getDataType() != HighFive::create_datatype<float>()) {
        return std::nullopt;
    }
    const auto nbins = dims[2];
    std::vector<float> fold(2 * nbins);
    folds.select({row, 0, 0}, {1, 2, nbins}).read_raw(fold.data());
    std::vector<float> profile(nbins);
    for (std::size_t i = 0; i < nbins; ++i) {
        const float var = fold[nbins + i];
        profile[i]      = var > 0.0F ? fold[i] / std::sqrt(var) : 0.0F;
    }
    return profile;
}

// Union of the per-region parameter limits ([nparams, 2] each).
std::vector<ParamRange> region_ranges(const HighFive::Group& parent,
                                      const std::vector<std::string>& names) {
    std::vector<ParamRange> ranges;
    if (!parent.exist("regions")) {
        return ranges;
    }
    const auto regions = parent.getGroup("regions");
    for (const auto& name : regions.listObjectNames()) {
        const auto region = regions.getGroup(name);
        if (!region.exist("param_limits")) {
            continue;
        }
        const auto limits = region.getDataSet("param_limits");
        const auto dims   = limits.getDimensions();
        if (dims.size() != 2 || dims[0] != names.size() || dims[1] != 2) {
            continue;
        }
        std::vector<double> values(dims[0] * 2);
        limits.read_raw(values.data());
        if (ranges.empty()) {
            for (std::size_t i = 0; i < names.size(); ++i) {
                ranges.push_back({
                    .name = names[i],
                    .min  = values[2 * i],
                    .max  = values[(2 * i) + 1],
                });
            }
            continue;
        }
        for (std::size_t i = 0; i < names.size(); ++i) {
            ranges[i].min = std::min(ranges[i].min, values[2 * i]);
            ranges[i].max = std::max(ranges[i].max, values[(2 * i) + 1]);
        }
    }
    return ranges;
}

// The frequency range when no per-parameter limits are stored.
template <typename Node>
std::vector<ParamRange> frequency_range(const Node& node) {
    const auto f_min = attribute<double>(node, "f_min");
    const auto f_max = attribute<double>(node, "f_max");
    if (!f_min || !f_max) {
        return {};
    }
    return {{
        .name = "freq",
        .min  = *f_min,
        .max  = *f_max,
    }};
}

void read_common(const HighFive::File& file,
                 CandidateFileSummary& out,
                 std::size_t top_n) {
    out.layout         = "common";
    out.schema_version = text_attribute(file, "schema_version").value_or("");
    if (!out.schema_version.starts_with("1.") && out.schema_version != "1") {
        throw std::invalid_argument(fmt::format(
            "unsupported candidate schema version {}", out.schema_version));
    }
    out.algorithm = text_attribute(file, "algorithm").value_or("");
    out.producer_version =
        text_attribute(file, "producer_version").value_or("");
    out.n_detections = attribute<std::uint64_t>(file, "n_detections");
    out.tsamp        = attribute_or<double>(file, "tsamp", 0.0);
    out.tobs         = attribute_or<double>(file, "tobs", 0.0);
    out.backend      = text_attribute(file, "backend");
    if (file.exist("observation")) {
        const auto obs  = file.getGroup("observation");
        out.source_name = text_attribute(obs, "source").value_or("");
        out.input_path  = text_attribute(obs, "input_path").value_or("");
        out.nsamps =
            attribute_or<std::uint64_t>(obs, "searched_nsamps", out.nsamps);
        out.dm = attribute<double>(obs, "dm");
    }
    std::optional<HighFive::Group> search;
    if (file.exist("search")) {
        search = file.getGroup("search");
        out.writer_version =
            text_attribute(*search, out.algorithm == "ep" ? "ep_sweep_version"
                                                          : "ffa_version")
                .value_or("");
        out.config_toml = text_attribute(*search, "config_toml");
        out.snr_min     = attribute<double>(*search, "snr_min");
        out.ranges      = region_ranges(*search, out.param_names);
        if (out.ranges.empty()) {
            out.ranges = frequency_range(*search);
        }
    }
    if (file.exist("processing")) {
        // EP selects candidates by its own S/N minimum.
        if (const auto snr_min =
                attribute<double>(file.getGroup("processing"), "snr_min")) {
            out.snr_min = snr_min;
        }
    }

    out.table = "candidates";
    if (!file.exist(out.table)) {
        throw std::invalid_argument("candidate file has no candidates table");
    }
    const auto table = file.getGroup(out.table);
    const auto snr   = table.getDataSet("snr");
    out.n_rows       = snr.getDimensions().at(0);
    TopRows ranked(top_n);
    rank_column(snr, 0, ranked);

    // EP rows index the searched runs; FFA rows index the regions.
    std::vector<std::uint64_t> chunk_ids;
    std::vector<std::string> run_paths;
    if (search && search->exist("sources")) {
        const auto sources = search->getGroup("sources");
        sources.getDataSet("chunk_id").read(chunk_ids);
        sources.getDataSet("run_path").read(run_paths);
    }
    const auto params  = table.getDataSet("param_sets");
    const auto nparams = out.param_names.size();
    for (const auto& [score, row] : ranked.take()) {
        CandidateRow cand{
            .row    = row,
            .params = read_param_row(params, row, nparams),
            .snr    = score,
            .width  = read_cell<std::uint32_t>(table, "width", row),
            .nbins  = read_cell<std::uint32_t>(table, "nbins", row),
            .source = {},
        };
        if (table.exist("source_region")) {
            const auto region =
                read_cell<std::uint32_t>(table, "source_region", row);
            if (region < run_paths.size() && region < chunk_ids.size()) {
                cand.source =
                    fmt::format("chunk {} run {}", chunk_ids[region],
                                std::filesystem::path(run_paths[region])
                                    .filename()
                                    .string());
            } else if (out.algorithm != "ep") {
                cand.source = fmt::format("region {}", region);
            }
        }
        out.top.push_back(std::move(cand));
    }
    if (!out.top.empty()) {
        out.top_profile = read_profile(table, out.top.front().row);
    }
}

void read_legacy_ffa(const HighFive::File& file,
                     CandidateFileSummary& out,
                     std::size_t top_n) {
    out.layout         = "legacy_ffa";
    out.algorithm      = "ffa";
    out.writer_version = text_attribute(file, "ffa_version").value_or("");
    out.nsamps         = attribute_or<std::uint64_t>(file, "nsamps", 0);
    out.tsamp          = attribute_or<double>(file, "tsamp", 0.0);
    out.tobs           = attribute_or<double>(file, "tobs", 0.0);
    out.snr_min        = attribute<double>(file, "snr_min");
    out.config_toml    = text_attribute(file, "config_toml");
    out.backend        = text_attribute(file, "backend");
    const auto root    = file.getGroup("/");
    out.ranges         = region_ranges(root, out.param_names);
    if (out.ranges.empty()) {
        out.ranges = frequency_range(file);
    }

    out.table      = "/";
    const auto snr = file.getDataSet("snr");
    out.n_rows     = snr.getDimensions().at(0);
    TopRows ranked(top_n);
    rank_column(snr, 0, ranked);
    const auto params  = file.getDataSet("param_sets");
    const auto nparams = out.param_names.size();
    for (const auto& [score, row] : ranked.take()) {
        CandidateRow cand{
            .row    = row,
            .params = read_param_row(params, row, nparams),
            .snr    = score,
            .width  = read_cell<std::uint32_t>(root, "width", row),
            .nbins  = read_cell<std::uint32_t>(root, "nbins", row),
            .source = {},
        };
        if (file.exist("detection_id")) {
            const auto id = read_cell<std::uint64_t>(root, "detection_id", row);
            cand.source   = fmt::format("region {}", id >> 32U);
        }
        out.top.push_back(std::move(cand));
    }
    if (!out.top.empty()) {
        out.top_profile = read_profile(root, out.top.front().row);
    }
}

// Raw EP sweep: chunks/<chunk>/runs/<run> with one score per leaf and
// param_sets [leaves, nparams + 2, 2] of (value, error) pairs.
void read_ep_runs(const HighFive::File& file,
                  CandidateFileSummary& out,
                  std::size_t top_n) {
    out.layout         = "ep_runs";
    out.algorithm      = "ep";
    out.writer_version = text_attribute(file, "ep_sweep_version").value_or("");
    out.nsamps         = attribute_or<std::uint64_t>(file, "nsamps", 0);
    out.tsamp          = attribute_or<double>(file, "tsamp", 0.0);
    out.tobs           = attribute_or<double>(file, "tobs", 0.0);
    out.snr_min        = attribute<double>(file, "snr_min");
    out.config_toml    = text_attribute(file, "config_toml");
    out.backend        = text_attribute(file, "backend");
    out.ranges         = frequency_range(file);
    out.table          = "chunks";

    struct Run {
        std::string chunk;
        std::string run;
        std::uint64_t offset;
    };
    std::vector<Run> runs;
    TopRows ranked(top_n);
    if (file.exist("chunks")) {
        const auto chunks = file.getGroup("chunks");
        for (const auto& chunk_name : chunks.listObjectNames()) {
            const auto chunk = chunks.getGroup(chunk_name);
            if (!chunk.exist("runs")) {
                continue;
            }
            const auto chunk_runs = chunk.getGroup("runs");
            for (const auto& run_name : chunk_runs.listObjectNames()) {
                const auto run = chunk_runs.getGroup(run_name);
                if (!run.exist("scores")) {
                    continue;
                }
                const auto scores = run.getDataSet("scores");
                runs.push_back({
                    .chunk  = chunk_name,
                    .run    = run_name,
                    .offset = out.n_rows,
                });
                rank_column(scores, out.n_rows, ranked);
                out.n_rows += scores.getDimensions().at(0);
            }
        }
    }
    const auto nparams = out.param_names.size();
    for (const auto& [score, global] : ranked.take()) {
        const auto it =
            std::ranges::upper_bound(runs, global, std::less<>{}, &Run::offset);
        const auto& src  = *std::prev(it);
        const auto leaf  = global - src.offset;
        const auto chunk = file.getGroup("chunks").getGroup(src.chunk);
        const auto run   = chunk.getGroup("runs").getGroup(src.run);
        std::vector<double> pairs((nparams + 2) * 2);
        run.getDataSet("param_sets")
            .select({leaf, 0, 0}, {1, nparams + 2, 2})
            .read_raw(pairs.data());
        CandidateRow cand{
            .row    = leaf,
            .params = {},
            .snr    = score,
            .width  = 0,
            .nbins  = 0,
            .source = {},
        };
        for (std::size_t i = 0; i < nparams; ++i) {
            cand.params.push_back(pairs[2 * i]);
        }
        cand.source = fmt::format(
            "chunk {} run {}",
            attribute_or<std::uint64_t>(chunk, "chunk_id", 0U), src.run);
        if (run.exist("width")) {
            cand.width = read_cell<std::uint32_t>(run, "width", leaf);
        }
        cand.nbins = attribute_or<std::uint32_t>(chunk, "nbins", 0U);
        out.top.push_back(std::move(cand));
    }
}

} // namespace

CandidateFileSummary
read_candidate_file_summary(const std::filesystem::path& path,
                            std::size_t top_n) {
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument(
            fmt::format("result file not found: {}", path.string()));
    }
    if (H5Fis_hdf5(path.string().c_str()) <= 0) {
        throw std::invalid_argument(
            fmt::format("not an HDF5 file: {}", path.string()));
    }
    CandidateFileSummary out;
    out.path = path;
    try {
        const HighFive::File file(path.string(), HighFive::File::ReadOnly);
        out.complete = attribute_or<int>(file, "complete", 0) == 1;
        out.param_names =
            attribute_or<std::vector<std::string>>(file, "param_names", {});
        if (out.param_names.empty() || out.param_names.back() != "freq") {
            throw std::invalid_argument(
                "missing or unsupported param_names (frequency last)");
        }
        out.config_effective = text_attribute(file, "config_effective");
        out.command_line     = text_attribute(file, "command_line");
        out.loki_version     = text_attribute(file, "loki_version");
        if (file.hasAttribute("schema_version")) {
            read_common(file, out, top_n);
        } else if (file.hasAttribute("ffa_version")) {
            read_legacy_ffa(file, out, top_n);
        } else if (file.hasAttribute("ep_sweep_version")) {
            read_ep_runs(file, out, top_n);
        } else {
            throw std::invalid_argument(
                "not a Loki FFA or EP result file (no schema_version, "
                "ffa_version or ep_sweep_version)");
        }
    } catch (const HighFive::Exception& e) {
        throw std::runtime_error(
            fmt::format("cannot read {}: {}", path.string(), e.what()));
    }
    return out;
}

} // namespace loki::io
