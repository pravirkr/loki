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

// The row of leaf @p leaf of an EP run: the leaf itself, or the row whose
// `leaf_index` is @p leaf in a run the search reduced; nullopt when the run
// does not keep it.
std::optional<std::uint64_t> run_row(const HighFive::Group& run,
                                     std::uint64_t leaf) {
    if (!run.exist("leaf_index")) {
        if (!run.exist("scores") ||
            leaf >= run.getDataSet("scores").getDimensions().at(0)) {
            return std::nullopt;
        }
        return leaf;
    }
    std::vector<std::uint32_t> index;
    run.getDataSet("leaf_index").read(index);
    const auto it = std::ranges::find(index, leaf);
    if (it == index.end()) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(std::distance(index.begin(), it));
}

// The leaf (value, error pairs) at row @p row of an EP run.
std::vector<double> run_row_leaf(const HighFive::Group& run,
                                 std::uint64_t row,
                                 std::size_t nparams) {
    std::vector<double> pairs((nparams + 2) * 2);
    run.getDataSet("param_sets")
        .select({row, 0, 0}, {1, nparams + 2, 2})
        .read_raw(pairs.data());
    return pairs;
}

// The runs of a raw EP sweep in the order the candidate products number
// them: chunks, then runs, by name.
std::vector<std::string> ep_run_paths(const HighFive::File& file) {
    std::vector<std::string> paths;
    if (!file.exist("chunks")) {
        return paths;
    }
    const auto chunks = file.getGroup("chunks");
    auto chunk_names  = chunks.listObjectNames();
    std::ranges::sort(chunk_names);
    for (const auto& chunk_name : chunk_names) {
        const auto chunk = chunks.getGroup(chunk_name);
        if (!chunk.exist("runs")) {
            continue;
        }
        auto run_names = chunk.getGroup("runs").listObjectNames();
        std::ranges::sort(run_names);
        for (const auto& run_name : run_names) {
            paths.push_back("chunks/" + chunk_name + "/runs/" + run_name);
        }
    }
    return paths;
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

// Rows of a common candidate table, with their EP leaves.
class CommonRows {
public:
    CommonRows(const HighFive::File& file, const CandidateFileSummary& out)
        : m_file(file),
          m_table(file.getGroup(out.table)),
          m_ep(out.algorithm == "ep"),
          m_nparams(out.param_names.size()) {
        // EP rows index the searched runs; FFA rows index the regions.
        if (file.exist("search") && file.getGroup("search").exist("sources")) {
            const auto sources = file.getGroup("search").getGroup("sources");
            sources.getDataSet("chunk_id").read(m_chunk_ids);
            sources.getDataSet("run_path").read(m_run_paths);
        }
    }

    [[nodiscard]] CandidateRow read(std::uint64_t row, float score) const {
        CandidateRow cand{
            .row    = row,
            .id     = read_cell<std::uint64_t>(m_table, "id", row),
            .params = read_param_row(m_table.getDataSet("param_sets"), row,
                                     m_nparams),
            .snr    = score,
            .width  = read_cell<std::uint32_t>(m_table, "width", row),
            .nbins  = read_cell<std::uint32_t>(m_table, "nbins", row),
            .source = {},
            .leaf   = {},
        };
        if (!m_table.exist("source_region")) {
            return cand;
        }
        const auto region =
            read_cell<std::uint32_t>(m_table, "source_region", row);
        if (region < m_run_paths.size() && region < m_chunk_ids.size()) {
            cand.source = fmt::format(
                "chunk {} run {}", m_chunk_ids[region],
                std::filesystem::path(m_run_paths[region]).filename().string());
            if (m_ep && m_file.exist(m_run_paths[region])) {
                const auto run = m_file.getGroup(m_run_paths[region]);
                if (const auto run_index =
                        run_row(run, read_cell<std::uint32_t>(
                                         m_table, "source_index", row))) {
                    cand.leaf = run_row_leaf(run, *run_index, m_nparams);
                }
            }
        } else if (!m_ep) {
            cand.source = fmt::format("region {}", region);
        }
        return cand;
    }

private:
    const HighFive::File& m_file;
    HighFive::Group m_table;
    bool m_ep;
    std::size_t m_nparams;
    std::vector<std::uint64_t> m_chunk_ids;
    std::vector<std::string> m_run_paths;
};

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
        out.reference_time =
            attribute_or<double>(obs, "reference_time_seconds", out.tobs / 2.0);
    } else {
        out.reference_time = out.tobs / 2.0;
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
    const CommonRows rows(file, out);
    for (const auto& [score, row] : ranked.take()) {
        out.top.push_back(rows.read(row, score));
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
            .id     = row,
            .params = read_param_row(params, row, nparams),
            .snr    = score,
            .width  = read_cell<std::uint32_t>(root, "width", row),
            .nbins  = read_cell<std::uint32_t>(root, "nbins", row),
            .source = {},
            .leaf   = {},
        };
        if (file.exist("detection_id")) {
            const auto id = read_cell<std::uint64_t>(root, "detection_id", row);
            cand.id       = id;
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
    out.reference_time = attribute_or<double>(file, "tobs", 0.0) / 2.0;
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
    const auto nparams   = out.param_names.size();
    const auto run_order = ep_run_paths(file);
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
        const auto path    = "chunks/" + src.chunk + "/runs/" + src.run;
        const auto ordinal = std::ranges::find(run_order, path);
        const std::uint64_t identity =
            run.exist("leaf_index")
                ? read_cell<std::uint32_t>(run, "leaf_index", leaf)
                : leaf;
        CandidateRow cand{
            .row    = leaf,
            .id     = (static_cast<std::uint64_t>(
                           std::distance(run_order.begin(), ordinal))
                       << 32U) |
                      identity,
            .params = {},
            .snr    = score,
            .width  = 0,
            .nbins  = 0,
            .source = {},
            .leaf   = pairs,
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

/// The strongest @p top_n rows of `/harvested` (already sorted by S/N).
void read_harvested(const HighFive::File& file,
                    CandidateFileSummary& out,
                    std::size_t top_n) {
    if (!file.exist("harvested")) {
        return;
    }
    const auto g = file.getGroup("harvested");
    if (!g.exist("snr") || !g.exist("param_sets")) {
        return;
    }
    std::vector<float> snr;
    g.getDataSet("snr").read(snr);
    out.n_harvested_sources = snr.size();
    const auto nparams      = out.param_names.size();
    const auto dims         = g.getDataSet("param_sets").getDimensions();
    if (snr.empty() || dims.size() != 2 || dims[1] != nparams) {
        return;
    }
    const auto n = std::min(top_n, snr.size());
    std::vector<double> params(n * nparams);
    g.getDataSet("param_sets")
        .select({0, 0}, {n, nparams})
        .read_raw(params.data());
    std::vector<std::uint64_t> level(snr.size(), 0);
    std::vector<std::uint64_t> n_runs(snr.size(), 0);
    if (g.exist("level")) {
        g.getDataSet("level").read(level);
    }
    if (g.exist("n_runs")) {
        g.getDataSet("n_runs").read(n_runs);
    }
    for (std::size_t i = 0; i < n; ++i) {
        CandidateRow row{.row    = i,
                         .id     = i,
                         .params = {params.begin() + (i * nparams),
                                    params.begin() + ((i + 1) * nparams)},
                         .snr    = snr[i],
                         .width  = 0,
                         .nbins  = 0,
                         .source =
                             fmt::format("level {}, {} run{}", level[i],
                                         n_runs[i], n_runs[i] == 1 ? "" : "s"),
                         .leaf = {}};
        out.harvested.push_back(std::move(row));
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
        read_harvested(file, out, top_n);
    } catch (const HighFive::Exception& e) {
        throw std::runtime_error(
            fmt::format("cannot read {}: {}", path.string(), e.what()));
    }
    return out;
}

CandidateRow read_candidate_by_id(const std::filesystem::path& path,
                                  std::uint64_t id,
                                  CandidateFileSummary* summary) {
    auto meta = read_candidate_file_summary(path, 0);
    std::optional<CandidateRow> found;
    try {
        const HighFive::File file(path.string(), HighFive::File::ReadOnly);
        const auto nparams = meta.param_names.size();
        if (meta.layout == "common") {
            const auto table = file.getGroup(meta.table);
            const auto ids   = table.getDataSet("id");
            const auto nrows = ids.getDimensions().at(0);
            const CommonRows rows(file, meta);
            std::vector<std::uint64_t> values;
            for (std::size_t start = 0; start < nrows && !found;
                 start += kBatchRows) {
                const auto count = std::min(kBatchRows, nrows - start);
                ids.select({start}, {count}).read(values);
                const auto it = std::ranges::find(values, id);
                if (it != values.end()) {
                    const auto row =
                        start + static_cast<std::size_t>(
                                    std::distance(values.begin(), it));
                    found = rows.read(row, read_cell<float>(table, "snr", row));
                }
            }
        } else if (meta.layout == "ep_runs") {
            const auto runs    = ep_run_paths(file);
            const auto ordinal = id >> 32U;
            const auto leaf    = id & 0xFFFFFFFFULL;
            if (ordinal < runs.size()) {
                const auto run = file.getGroup(runs[ordinal]);
                if (const auto row = run_row(run, leaf)) {
                    const auto chunk_path =
                        runs[ordinal].substr(0, runs[ordinal].find("/runs/"));
                    const auto chunk = file.getGroup(chunk_path);
                    CandidateRow cand{
                        .row    = *row,
                        .id     = id,
                        .params = {},
                        .snr    = read_cell<float>(run, "scores", *row),
                        .width  = run.exist("width") ? read_cell<std::uint32_t>(
                                                           run, "width", *row)
                                                     : 0U,
                        .nbins =
                            attribute_or<std::uint32_t>(chunk, "nbins", 0U),
                        .source = fmt::format(
                            "chunk {} run {}",
                            attribute_or<std::uint64_t>(chunk, "chunk_id", 0U),
                            std::filesystem::path(runs[ordinal])
                                .filename()
                                .string()),
                        .leaf = run_row_leaf(run, *row, nparams),
                    };
                    for (std::size_t i = 0; i < nparams; ++i) {
                        cand.params.push_back(cand.leaf[2 * i]);
                    }
                    found = std::move(cand);
                }
            }
        } else {
            throw std::invalid_argument(
                "candidate identities need a common or raw EP result file");
        }
    } catch (const HighFive::Exception& e) {
        throw std::runtime_error(
            fmt::format("cannot read {}: {}", path.string(), e.what()));
    }
    if (!found) {
        throw std::invalid_argument(
            fmt::format("no candidate with id {} in {}", id, path.string()));
    }
    if (summary != nullptr) {
        *summary = std::move(meta);
    }
    return std::move(*found);
}

} // namespace loki::io
