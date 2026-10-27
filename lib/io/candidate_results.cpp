#include "lib/io/candidate_results.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <hdf5.h>
#include <highfive/highfive.hpp>

#include "loki/search/candidates.hpp"

#include "lib/detail/utils.hpp"
#include "lib/search/candidate_processing.hpp"

namespace loki::io::detail {
namespace {
using search::Candidate;
using search::CandidateCluster;

struct ScratchDirectory {
    std::filesystem::path path;
    explicit ScratchDirectory(const std::filesystem::path& parent) {
        std::filesystem::create_directories(parent);
        for (unsigned int attempt = 0; attempt < 100; ++attempt) {
            path = parent / ("loki-candidates-" +
                             std::to_string(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count()) +
                             "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(path)) {
                return;
            }
        }
        throw std::runtime_error("cannot create candidate scratch directory");
    }
    ~ScratchDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void copy_object(const HighFive::File& source,
                 const std::string& name,
                 const HighFive::Group& target,
                 const std::string& dest) {
    if (H5Ocopy(source.getId(), name.c_str(), target.getId(), dest.c_str(),
                H5P_DEFAULT, H5P_DEFAULT) < 0) {
        throw std::runtime_error("cannot copy candidate metadata object " +
                                 name);
    }
}

template <typename Source, typename Target>
void copy_attributes(const Source& source,
                     const Target& target,
                     bool skip_selection_counts = false) {
    for (const auto& name : source.listAttributeNames()) {
        if (target.hasAttribute(name) ||
            (skip_selection_counts &&
             (name == "n_clusters" || name == "n_candidates"))) {
            continue;
        }
        const auto attr  = source.getAttribute(name);
        const auto type  = attr.getDataType();
        const auto space = attr.getSpace();
        const hid_t dst = H5Acreate2(target.getId(), name.c_str(), type.getId(),
                                     space.getId(), H5P_DEFAULT, H5P_DEFAULT);
        if (dst < 0) {
            throw std::runtime_error("cannot create search attribute");
        }
        const auto bytes = type.getSize() * space.getElementCount();
        std::vector<std::byte> data(bytes);
        const herr_t rd = H5Aread(attr.getId(), type.getId(), data.data());
        const herr_t wr =
            rd < 0 ? -1 : H5Awrite(dst, type.getId(), data.data());
        if (H5Tdetect_class(type.getId(), H5T_VLEN) > 0 ||
            H5Tis_variable_str(type.getId()) > 0) {
            H5Dvlen_reclaim(type.getId(), space.getId(), H5P_DEFAULT,
                            data.data());
        }
        H5Aclose(dst);
        if (wr < 0) {
            throw std::runtime_error("cannot copy search attribute");
        }
    }
}

template <typename T>
std::vector<T> read_rows(const HighFive::File& file,
                         const std::string& name,
                         std::size_t start,
                         std::size_t count) {
    std::vector<T> values(count);
    if (count != 0) {
        file.getDataSet(name).select({start}, {count}).read(values);
    }
    return values;
}

std::vector<Candidate> read_candidates(const HighFive::File& file,
                                       std::size_t start,
                                       std::size_t count,
                                       std::size_t nparams) {
    std::vector<Candidate> records(count);
    auto scores = read_rows<float>(file, "snr", start, count);
    auto widths = read_rows<std::uint16_t>(file, "width", start, count);
    auto bins   = read_rows<std::uint16_t>(file, "nbins", start, count);
    auto ids    = read_rows<std::uint64_t>(file, "detection_id", start, count);
    std::vector<double> params(count * nparams);
    if (count != 0) {
        file.getDataSet("param_sets")
            .select({start, 0}, {count, nparams})
            .read_raw(params.data());
    }
    std::vector<double> errors;
    std::vector<std::uint8_t> error_valid;
    if (file.exist("param_errors")) {
        errors.resize(count * nparams);
        error_valid =
            read_rows<std::uint8_t>(file, "has_parameter_errors", start, count);
        if (count != 0) {
            file.getDataSet("param_errors")
                .select({start, 0}, {count, nparams})
                .read_raw(errors.data());
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        auto& c = records[i];
        if (!errors.empty()) {
            c.has_parameter_errors = error_valid[i] != 0;
            std::copy_n(errors.begin() +
                            static_cast<std::ptrdiff_t>(i * nparams),
                        nparams, c.parameter_errors.begin());
        }
        c.id            = ids[i];
        c.source_region = static_cast<std::uint32_t>(c.id >> 32U);
        c.source_index  = static_cast<std::uint32_t>(c.id);
        c.snr           = scores[i];
        c.width         = widths[i];
        c.nbins         = bins[i];
        std::copy_n(params.begin() + static_cast<std::ptrdiff_t>(i * nparams),
                    nparams, c.parameters.begin());
        c.derivatives[0] = c.parameters[nparams - 1];
        for (std::size_t k = 1; k < nparams; ++k) {
            c.derivatives[k] = -c.derivatives[0] *
                               c.parameters[nparams - 1 - k] * utils::kInvCval;
        }
        c.validate();
    }
    return records;
}

struct EPSource {
    std::string path;
    std::uint64_t chunk;
};

std::vector<EPSource> ep_sources(const HighFive::File& file,
                                 const search::CandidateOptions& options) {
    int complete = 0;
    if (!file.hasAttribute("complete")) {
        throw std::invalid_argument("EP spool has no completion marker");
    }
    file.getAttribute("complete").read(complete);
    if (complete != 1) {
        throw std::invalid_argument("EP spool is incomplete");
    }
    std::string basis;
    file.getAttribute("poly_basis").read(basis);
    if (basis != "taylor") {
        throw std::invalid_argument(
            "EP candidates currently require Taylor polynomial runs");
    }
    std::vector<EPSource> sources;
    auto chunks = file.getGroup("chunks").listObjectNames();
    std::ranges::sort(chunks);
    std::size_t bytes = 0;
    for (const auto& chunk_name : chunks) {
        const auto path  = "chunks/" + chunk_name;
        const auto chunk = file.getGroup(path);
        if (!chunk.exist("runs")) {
            throw std::invalid_argument("EP spool is missing completed runs");
        }
        std::uint64_t chunk_id = 0;
        chunk.getAttribute("chunk_id").read(chunk_id);
        auto runs = chunk.getGroup("runs").listObjectNames();
        std::ranges::sort(runs);
        for (const auto& run : runs) {
            const auto run_path = path + "/runs/" + run;
            const auto group    = file.getGroup(run_path);
            if (group.exist("harvest")) {
                throw std::invalid_argument(
                    "EP candidate publication requires RFI disabled");
            }
            bytes += 2 * (sizeof(EPSource) + run_path.size() + 1);
            if (static_cast<double>(bytes) >
                options.memory_gb * utils::kBytesPerGB / 8) {
                throw std::runtime_error(
                    "EP source index exceeds candidate memory budget");
            }
            if (sources.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "too many EP runs for detection identities");
            }
            sources.push_back({run_path, chunk_id});
        }
    }
    return sources;
}

std::vector<Candidate> read_ep_candidates(const HighFive::File& file,
                                          const EPSource& source,
                                          std::uint32_t run_id,
                                          std::size_t start,
                                          std::size_t count,
                                          std::size_t nparams,
                                          double snr_min) {
    const auto run  = file.getGroup(source.path);
    const auto dims = run.getDataSet("param_sets").getDimensions();
    if (dims.size() != 3 || dims[1] != nparams + 2 || dims[2] != 2 ||
        run.getDataSet("width").getDimensions() !=
            std::vector<std::size_t>{dims[0]} ||
        run.getDataSet("scores").getDimensions() !=
            std::vector<std::size_t>{dims[0]}) {
        throw std::invalid_argument("invalid EP final-leaf table");
    }
    std::vector<float> scores(count);
    std::vector<std::uint32_t> widths(count);
    std::vector<double> leaves(count * (nparams + 2) * 2);
    if (count != 0) {
        run.getDataSet("scores").select({start}, {count}).read(scores);
        run.getDataSet("width").select({start}, {count}).read(widths);
        run.getDataSet("param_sets")
            .select({start, 0, 0}, {count, nparams + 2, 2})
            .read_raw(leaves.data());
    }
    std::uint32_t nbins   = 0;
    const auto chunk_path = source.path.substr(0, source.path.find("/runs/"));
    file.getGroup(chunk_path).getAttribute("nbins").read(nbins);
    std::vector<Candidate> records;
    records.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!utils::is_finite(scores[i])) {
            throw std::invalid_argument("non-finite final EP score");
        }
        if (scores[i] < snr_min) {
            continue;
        }
        if (start + i > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "EP leaf index exceeds detection identity capacity");
        }
        Candidate c;
        c.id = (static_cast<std::uint64_t>(run_id) << 32U) | (start + i);
        c.source_region = run_id;
        c.source_index  = static_cast<std::uint32_t>(start + i);
        c.width         = widths[i];
        c.nbins         = nbins;
        c.snr           = scores[i];
        for (std::size_t k = 0; k < nparams; ++k) {
            c.parameters[k] = leaves[i * (nparams + 2) * 2 + 2 * k];
        }
        c.derivatives[0] = c.parameters[nparams - 1];
        for (std::size_t k = 1; k < nparams; ++k) {
            c.derivatives[k] = -c.derivatives[0] *
                               c.parameters[nparams - 1 - k] * utils::kInvCval;
        }
        c.validate();
        records.push_back(c);
    }
    return records;
}

static_assert(std::is_trivially_copyable_v<Candidate>);
void write_record(std::ostream& out, const Candidate& c) {
    out.write(reinterpret_cast<const char*>(&c), sizeof(c));
    if (!out) {
        throw std::runtime_error(
            "cannot write candidate sort run (disk full?)");
    }
}
bool read_record(std::istream& in, Candidate& c) {
    in.read(reinterpret_cast<char*>(&c), sizeof(c));
    if (in.gcount() == 0 && in.eof()) {
        return false;
    }
    if (!in) {
        throw std::runtime_error("truncated or unreadable candidate sort run");
    }
    return true;
}

// Bounded fan-in; caller stores only a logarithmic number of pending runs.
void merge_runs(std::span<const std::filesystem::path> inputs,
                const std::filesystem::path& output) {
    struct Item {
        Candidate record;
        std::size_t source;
    };
    const auto later = [](const Item& a, const Item& b) {
        return search::detail::candidate_brighter(b.record, a.record);
    };
    std::priority_queue<Item, std::vector<Item>, decltype(later)> heap(later);
    std::vector<std::ifstream> streams;
    streams.reserve(inputs.size());
    for (const auto& path : inputs) {
        streams.emplace_back(path, std::ios::binary);
        Candidate c;
        if (read_record(streams.back(), c)) {
            heap.push({c, streams.size() - 1});
        }
    }
    std::ofstream out(output, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot create candidate merge run");
    }
    while (!heap.empty()) {
        const auto item = heap.top();
        heap.pop();
        write_record(out, item.record);
        Candidate c;
        if (read_record(streams[item.source], c)) {
            heap.push({c, item.source});
        }
    }
    out.close();
    if (!out) {
        throw std::runtime_error("cannot finish candidate merge run");
    }
}

class ExternalSort {
public:
    explicit ExternalSort(std::filesystem::path directory)
        : m_directory(std::move(directory)) {}
    void add(std::vector<Candidate>& records) {
        std::ranges::sort(records, search::detail::candidate_brighter);
        auto path = next_path();
        {
            std::ofstream out(path, std::ios::binary);
            if (!out) {
                throw std::runtime_error("cannot create candidate sort run");
            }
            for (const auto& c : records) {
                write_record(out, c);
            }
            out.close();
            if (!out) {
                throw std::runtime_error("cannot finish candidate sort run");
            }
        }
        // Binary carries bound pending paths and open descriptors even when
        // tiny batches create millions of runs.
        std::size_t level = 0;
        while (true) {
            if (level == m_levels.size()) {
                m_levels.emplace_back();
            }
            if (m_levels[level].empty()) {
                m_levels[level] = path;
                break;
            }
            const auto merged = next_path();
            const std::array inputs{m_levels[level], path};
            merge_runs(inputs, merged);
            for (const auto& input : inputs) {
                std::filesystem::remove(input);
            }
            m_levels[level].clear();
            path = merged;
            ++level;
        }
    }
    std::filesystem::path finish() {
        std::filesystem::path result;
        for (auto& path : m_levels) {
            if (path.empty()) {
                continue;
            }
            if (result.empty()) {
                result = path;
            } else {
                const auto merged = next_path();
                const std::array inputs{result, path};
                merge_runs(inputs, merged);
                for (const auto& input : inputs) {
                    std::filesystem::remove(input);
                }
                result = merged;
            }
        }
        return result;
    }

private:
    std::filesystem::path next_path() {
        return m_directory / (std::to_string(m_serial++) + ".run");
    }
    std::filesystem::path m_directory;
    std::uint64_t m_serial{};
    std::vector<std::filesystem::path> m_levels;
};

class TableWriter {
public:
    TableWriter(HighFive::Group group,
                std::size_t nparams,
                std::size_t batch_size)
        : m_group(std::move(group)),
          m_nparams(nparams) {
        m_batch.reserve(batch_size);
        m_ids.reserve(batch_size);
        m_limit = batch_size;
        for (const auto* name : {"f0", "f1", "f2", "duty_cycle"}) {
            create<double>(name);
        }
        create<double>("phase_extent");
        m_group.getDataSet("phase_extent")
            .createAttribute("units", std::string("cycles"));
        create<float>("snr");
        for (const auto* name :
             {"width", "nbins", "source_region", "source_index"}) {
            create<std::uint32_t>(name);
        }
        create<std::uint64_t>("id");
        create<std::uint64_t>("cluster_id");
        create<std::uint64_t>("member_count");
        create<std::int64_t>("parent_cluster_id");
        create<std::uint32_t>("harmonic_numerator");
        create<std::uint32_t>("harmonic_denominator");
        create<std::uint8_t>("selected");
        create<std::uint8_t>("has_parameter_errors");
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking({batch_size, nparams}));
        props.add(HighFive::Shuffle());
        props.add(HighFive::Deflate(3));
        m_group.createDataSet<double>(
            "param_sets",
            HighFive::DataSpace({0, nparams},
                                {HighFive::DataSpace::UNLIMITED, nparams}),
            props);
        m_group.createDataSet<double>(
            "param_errors",
            HighFive::DataSpace({0, nparams},
                                {HighFive::DataSpace::UNLIMITED, nparams}),
            props);
        m_group.getDataSet("f0").createAttribute("units", std::string("Hz"));
        m_group.getDataSet("f1").createAttribute("units", std::string("Hz/s"));
        m_group.getDataSet("f2").createAttribute("units",
                                                 std::string("Hz/s^2"));
        m_group.getDataSet("width").createAttribute("units",
                                                    std::string("bins"));
    }
    void append(const CandidateCluster& cluster, std::uint64_t cluster_id) {
        m_batch.push_back(cluster);
        m_ids.push_back(cluster_id);
        if (m_batch.size() == m_limit) {
            flush();
        }
    }
    void flush() {
        if (m_batch.empty()) {
            return;
        }
        column<double>("f0", [](const auto& c) {
            return c.representative.derivatives[0];
        });
        column<double>("f1", [](const auto& c) {
            return c.representative.derivatives[1];
        });
        column<double>("f2", [](const auto& c) {
            return c.representative.derivatives[2];
        });
        column<double>("duty_cycle", [](const auto& c) {
            return c.representative.duty_cycle();
        });
        column<double>("phase_extent",
                       [](const auto& c) { return c.phase_extent; });
        column<float>("snr",
                      [](const auto& c) { return c.representative.snr; });
        column<std::uint32_t>(
            "width", [](const auto& c) { return c.representative.width; });
        column<std::uint32_t>(
            "nbins", [](const auto& c) { return c.representative.nbins; });
        column<std::uint32_t>("source_region", [](const auto& c) {
            return c.representative.source_region;
        });
        column<std::uint32_t>("source_index", [](const auto& c) {
            return c.representative.source_index;
        });
        column<std::uint64_t>(
            "id", [](const auto& c) { return c.representative.id; });
        column<std::uint64_t>("member_count",
                              [](const auto& c) { return c.member_count; });
        column<std::int64_t>("parent_cluster_id",
                             [](const auto& c) { return c.parent_cluster_id; });
        column<std::uint32_t>("harmonic_numerator", [](const auto& c) {
            return c.harmonic_numerator;
        });
        column<std::uint32_t>("harmonic_denominator", [](const auto& c) {
            return c.harmonic_denominator;
        });
        column<std::uint8_t>("selected",
                             [](const auto& c) { return c.selected ? 1 : 0; });
        column<std::uint8_t>("has_parameter_errors", [](const auto& c) {
            return c.representative.has_parameter_errors ? 1 : 0;
        });
        append_column("cluster_id", m_ids);
        std::vector<double> params;
        params.reserve(m_batch.size() * m_nparams);
        for (const auto& c : m_batch) {
            params.insert(params.end(), c.representative.parameters.begin(),
                          c.representative.parameters.begin() +
                              static_cast<std::ptrdiff_t>(m_nparams));
        }
        auto ds = m_group.getDataSet("param_sets");
        ds.resize({m_rows + m_batch.size(), m_nparams});
        ds.select({m_rows, 0}, {m_batch.size(), m_nparams})
            .write_raw(params.data());
        params.clear();
        for (const auto& c : m_batch) {
            params.insert(params.end(),
                          c.representative.parameter_errors.begin(),
                          c.representative.parameter_errors.begin() +
                              static_cast<std::ptrdiff_t>(m_nparams));
        }
        auto error_ds = m_group.getDataSet("param_errors");
        error_ds.resize({m_rows + m_batch.size(), m_nparams});
        error_ds.select({m_rows, 0}, {m_batch.size(), m_nparams})
            .write_raw(params.data());
        m_rows += m_batch.size();
        m_batch.clear();
        m_ids.clear();
    }

private:
    template <typename T> void create(const std::string& name) {
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking({m_limit}));
        props.add(HighFive::Shuffle());
        props.add(HighFive::Deflate(3));
        m_group.createDataSet<T>(
            name, HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
            props);
    }
    template <typename T>
    void append_column(const std::string& name, const std::vector<T>& values) {
        auto ds = m_group.getDataSet(name);
        ds.resize({m_rows + values.size()});
        ds.select({m_rows}, {values.size()}).write(values);
    }
    template <typename T, typename F>
    void column(const std::string& name, F value) {
        std::vector<T> values;
        values.reserve(m_batch.size());
        for (const auto& c : m_batch) {
            values.push_back(static_cast<T>(value(c)));
        }
        append_column(name, values);
    }
    HighFive::Group m_group;
    std::size_t m_nparams;
    std::size_t m_limit{};
    std::size_t m_rows{};
    std::vector<CandidateCluster> m_batch;
    std::vector<std::uint64_t> m_ids;
};
void publish_candidate_results(const std::filesystem::path& spool,
                               const std::filesystem::path& output,
                               const ObservationMetadata& observation,
                               const search::CandidateOptions& options,
                               bool ep,
                               double snr_min) {
    options.validate();
    observation.validate();
    ScratchDirectory scratch(options.temporary_directory.empty()
                                 ? (output.parent_path().empty()
                                        ? std::filesystem::path(".")
                                        : output.parent_path())
                                 : options.temporary_directory);
    const HighFive::File raw(spool.string(), HighFive::File::ReadOnly);
    std::vector<std::string> names;
    raw.getAttribute("param_names").read(names);
    const auto nparams = names.size();
    if (nparams == 0 || nparams > 6 || (ep && (nparams < 2 || nparams > 4))) {
        throw std::invalid_argument("unsupported candidate parameter count");
    }
    if (ep) {
        constexpr std::array<std::string_view, 4> expected{"snap", "jerk",
                                                           "accel", "freq"};
        for (std::size_t i = 0; i < nparams; ++i) {
            if (names[i] != expected[expected.size() - nparams + i]) {
                throw std::invalid_argument(
                    "unsupported EP native parameter convention");
            }
        }
    }
    double tobs = 0;
    raw.getAttribute("tobs").read(tobs);
    std::size_t nrows = 0;
    const auto sources =
        ep ? ep_sources(raw, options) : std::vector<EPSource>{};
    std::size_t source_bytes = 0;
    if (ep) {
        source_bytes = sources.capacity() * sizeof(EPSource);
        for (const auto& source : sources) {
            source_bytes += source.path.capacity() + 1;
        }
    }
    search::detail::CandidateClusterer processor(tobs, options,
                                                 2 * source_bytes);
    // At least 3/4 of half the budget remains for streams, HDF5 caches and
    // serialization scratch, including exceptionally small test budgets.
    const auto batch = std::max<std::size_t>(
        1, std::min<std::size_t>(
               65536,
               static_cast<std::size_t>(options.memory_gb * utils::kBytesPerGB /
                                        (32 * sizeof(Candidate)))));
    ExternalSort sort(scratch.path);
    if (ep) {
        for (std::size_t run_id = 0; run_id < sources.size(); ++run_id) {
            const auto count = raw.getGroup(sources[run_id].path)
                                   .getDataSet("scores")
                                   .getDimensions()[0];
            for (std::size_t start = 0; start < count; start += batch) {
                auto records = read_ep_candidates(
                    raw, sources[run_id], static_cast<std::uint32_t>(run_id),
                    start, std::min(batch, count - start), nparams, snr_min);
                nrows += records.size();
                if (!records.empty()) {
                    sort.add(records);
                }
            }
        }
    } else {
        nrows = raw.getDataSet("snr").getDimensions()[0];
        for (std::size_t start = 0; start < nrows; start += batch) {
            auto records = read_candidates(
                raw, start, std::min(batch, nrows - start), nparams);
            sort.add(records);
        }
    }
    const auto sorted = sort.finish();
    // The atomic publication file must live beside the destination, even if
    // scratch sorting is explicitly placed on another filesystem.
    ScratchDirectory publication(output.parent_path().empty()
                                     ? std::filesystem::path(".")
                                     : output.parent_path());
    const auto publish = publication.path / "result.partial";
    try {
        {
            HighFive::File result(publish.string(), HighFive::File::Excl);
            result.createAttribute("schema_version", std::string("1.0"));
            result.createAttribute("producer", std::string("loki"));
            result.createAttribute("producer_version",
                                   std::string(LOKI_PRODUCER_VERSION));
            result.createAttribute("algorithm", std::string(ep ? "ep" : "ffa"));
            result.createAttribute("complete", 0);
            result.createAttribute("param_names", names);
            result.createAttribute("parameter_model",
                                   std::string("kinematic_taylor"));
            result.createAttribute(
                "parameter_order",
                "descending kinematic derivative, frequency last");
            result.createAttribute("parameter_units",
                                   "frequency Hz; acceleration m/s^2; jerk "
                                   "m/s^3; higher orders m/s^k");
            result.createAttribute("n_detections",
                                   static_cast<std::uint64_t>(nrows));
            auto config = result.createGroup("processing");
            config.createAttribute("cluster", options.cluster);
            config.createAttribute("grouping", options.grouping);
            config.createAttribute("radius", options.radius);
            config.createAttribute("phase_radius", options.phase_radius);
            config.createAttribute("retain_detections",
                                   options.retain_detections);
            config.createAttribute("flag_harmonics", options.flag_harmonics);
            config.createAttribute("remove_harmonics",
                                   options.remove_harmonics);
            config.createAttribute("harmonic_denominator_max",
                                   options.harmonic_denominator_max);
            config.createAttribute("harmonic_phase_distance",
                                   options.harmonic_phase_distance);
            config.createAttribute("harmonic_snr_distance",
                                   options.harmonic_snr_distance);
            config.createAttribute("max_candidates", options.max_candidates);
            config.createAttribute("memory_gb", options.memory_gb);
            config.createAttribute("temporary_directory",
                                   options.temporary_directory.string());
            auto obs = result.createGroup("observation");
            if (observation.dm) {
                obs.createAttribute("dm", *observation.dm);
            }
            if (observation.start_mjd) {
                obs.createAttribute("start_mjd", *observation.start_mjd);
                obs.createAttribute("reference_mjd", *observation.start_mjd +
                                                         tobs / (2 * 86400));
            }
            if (observation.barycentric) {
                obs.createAttribute("barycentric", *observation.barycentric);
            }
            obs.createAttribute("source", observation.source);
            obs.createAttribute("input_path", observation.input_path);
            obs.createAttribute("original_nsamps", observation.original_nsamps);
            obs.createAttribute("reference_time_seconds", tobs / 2);
            obs.createAttribute("tobs", tobs);
            double tsamp = 0;
            raw.getAttribute("tsamp").read(tsamp);
            std::uint64_t nsamps = 0;
            raw.getAttribute("nsamps").read(nsamps);
            obs.createAttribute("tsamp", tsamp);
            obs.createAttribute("searched_nsamps", nsamps);
            result.createAttribute("tobs", tobs);
            result.createAttribute("tsamp", tsamp);
            auto search_group = result.createGroup("search");
            copy_attributes(raw, search_group);
            if (raw.exist("regions")) {
                copy_object(raw, "regions", search_group, "regions");
            }
            if (raw.exist("timer_stats")) {
                copy_object(raw, "timer_stats", search_group, "timer_stats");
            }
            if (ep) {
                copy_object(raw, "chunks", result.getGroup("/"), "chunks");
                copy_attributes(raw, result, true);
                config.createAttribute("snr_min", snr_min);
                config.createAttribute(
                    "width_policy",
                    std::string(
                        "one winning configured boxcar width per final leaf"));
                config.createAttribute(
                    "score_semantics",
                    std::string("final reintegrated boxcar S/N; scores_ep "
                                "remains diagnostic"));
                auto provenance = search_group.createGroup("sources");
                std::vector<std::uint64_t> chunk_ids;
                std::vector<std::string> paths;
                for (const auto& source : sources) {
                    chunk_ids.push_back(source.chunk);
                    paths.push_back(source.path);
                }
                provenance.createDataSet("chunk_id", chunk_ids);
                provenance.createDataSet("run_path", paths);
                provenance.createAttribute(
                    "identity_encoding",
                    std::string("id = (source ordinal << 32) | leaf row; "
                                "source_region indexes these sources"));
            }
            // Raw rows are written in sorted order, so membership is appended
            // sequentially without an in-memory mapping of every detection.
            auto detection_group = options.retain_detections
                                       ? result.createGroup("detections")
                                       : result.createGroup("_discarded");
            TableWriter detections(detection_group, nparams, batch);
            TableWriter clusters(result.createGroup("clusters"), nparams,
                                 batch);
            TableWriter candidates(result.createGroup("candidates"), nparams,
                                   batch);
            const bool singleton_stream =
                !options.cluster && !options.flag_harmonics;
            std::uint64_t cluster_count = 0;
            std::uint64_t selected      = 0;
            if (!sorted.empty()) {
                std::ifstream in(sorted, std::ios::binary);
                Candidate c;
                while (read_record(in, c)) {
                    std::uint64_t id = 0;
                    if (singleton_stream) {
                        id = cluster_count++;
                        CandidateCluster singleton{.representative = c};
                        singleton.selected = options.max_candidates == 0 ||
                                             selected < options.max_candidates;
                        clusters.append(singleton, id);
                        if (singleton.selected) {
                            candidates.append(singleton, id);
                            ++selected;
                        }
                    } else {
                        id = processor.append(c);
                    }
                    if (options.retain_detections) {
                        detections.append(CandidateCluster{.representative = c},
                                          id);
                    }
                }
            }
            detections.flush();
            if (!options.retain_detections) {
                result.unlink("_discarded");
            }
            if (!singleton_stream) {
                processor.finish();
                // Transitive append IDs are provisional: a later bridge can
                // join earlier groups. Remap only after the graph is complete,
                // in bounded batches, preserving raw score order and identity.
                if (options.retain_detections && options.cluster &&
                    options.grouping == "transitive") {
                    auto dataset = result.getDataSet("detections/cluster_id");
                    for (std::size_t start = 0; start < nrows; start += batch) {
                        const auto count = std::min(batch, nrows - start);
                        auto ids         = read_rows<std::uint64_t>(
                            result, "detections/cluster_id", start, count);
                        for (auto& id : ids) {
                            id = processor.cluster_id(id);
                        }
                        dataset.select({start}, {count}).write(ids);
                    }
                }
                cluster_count =
                    static_cast<std::uint64_t>(processor.clusters.size());
                for (std::size_t id = 0; id < processor.clusters.size(); ++id) {
                    const auto& c = processor.clusters[id];
                    clusters.append(c, id);
                    if (c.selected) {
                        candidates.append(c, id);
                        ++selected;
                    }
                }
            }
            clusters.flush();
            candidates.flush();
            result.createAttribute("n_clusters", cluster_count);
            result.createAttribute("n_candidates", selected);
            // Transitional aliases refer to SELECTED candidates, never raw
            // detections. Existing recovery/summary readers remain useful.
            for (const auto* name : {"snr", "param_sets", "width", "nbins"}) {
                const auto target = std::string("candidates/") + name;
                if (H5Lcreate_hard(result.getId(), target.c_str(),
                                   result.getId(), name, H5P_DEFAULT,
                                   H5P_DEFAULT) < 0) {
                    throw std::runtime_error(
                        "cannot create candidate compatibility alias");
                }
            }
            result.getAttribute("complete").write(1);
            result.flush();
        }
        std::filesystem::rename(publish, output);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(publish, error);
        throw;
    }
}
} // namespace

void write_candidate_results(const std::filesystem::path& spool,
                             const std::filesystem::path& output,
                             const ObservationMetadata& observation,
                             const search::CandidateOptions& options) {
    publish_candidate_results(spool, output, observation, options, false, 0);
}

void write_ep_candidate_results(const std::filesystem::path& spool,
                                const std::filesystem::path& output,
                                const ObservationMetadata& observation,
                                const search::CandidateOptions& options,
                                std::optional<double> snr_min) {
    if (!snr_min) {
        const HighFive::File raw(spool.string(), HighFive::File::ReadOnly);
        double saved = 7;
        raw.getAttribute("snr_min").read(saved);
        snr_min = saved;
    }
    if (!utils::is_finite(*snr_min) || *snr_min < 0) {
        throw std::invalid_argument(
            "EP candidate S/N minimum must be finite and non-negative");
    }
    publish_candidate_results(spool, output, observation, options, true,
                              *snr_min);
}
} // namespace loki::io::detail
