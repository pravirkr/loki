#include "lib/io/candidate_results.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include "loki/common/interrupt.hpp"
#include "loki/common/types.hpp"
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

/// Interrupt checkpoints and periodic progress for postprocessing, which can
/// run for minutes on a bright source. tick() is one atomic load plus, every
/// 4096 calls, a clock read; it logs at most every kInterval.
class PostProgress {
public:
    void phase(std::string name, std::uint64_t total) {
        m_phase = std::move(name);
        m_total = total;
        m_done  = 0;
    }
    void tick(std::uint64_t done_increment = 1) {
        loki::check_interrupt();
        m_done += done_increment;
        maybe_log();
    }
    /// One record rewritten by a sort merge (counted separately).
    void tick_merge() {
        loki::check_interrupt();
        ++m_merged;
        maybe_log();
    }
    /// Forces the next tick() to consider logging (after a slow step).
    void poke() { m_calls = 4095; }

private:
    void maybe_log() {
        if (++m_calls % 4096 != 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - m_last < kInterval) {
            return;
        }
        m_last = now;
        const double elapsed =
            std::chrono::duration<double>(now - m_started).count();
        const auto merged = m_merged == 0
                                ? std::string()
                                : fmt::format(", {} records merged", m_merged);
        if (m_total != 0) {
            spdlog::info("Candidate postprocessing: {} {}/{} ({:.0f}%){}, "
                         "{:.0f} s elapsed",
                         m_phase, m_done, m_total,
                         100.0 * static_cast<double>(m_done) /
                             static_cast<double>(m_total),
                         merged, elapsed);
        } else {
            spdlog::info("Candidate postprocessing: {}{}, {:.0f} s elapsed",
                         m_phase, merged, elapsed);
        }
    }
    static constexpr std::chrono::seconds kInterval{30};
    std::chrono::steady_clock::time_point m_started{
        std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point m_last{m_started};
    std::string m_phase;
    std::uint64_t m_total{};
    std::uint64_t m_done{};
    std::uint64_t m_merged{};
    std::uint64_t m_calls{};
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

// The source is a group, not a file: HDF5 1.10 cannot count the attributes
// of a file ID (H5Aget_num_attrs), so pass the root group "/" instead.
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

struct EPSource {
    std::string path;
    std::uint64_t chunk;
};

/// What the per-run reduction did to one EP run.
struct RunReduction {
    std::uint64_t final_leaves{};  ///< Rows of the run in the spool.
    std::uint64_t above_snr_min{}; ///< Rows read as trials.
    std::uint64_t cell_dropped{};
    std::uint64_t dominated{};
    float max_cell_dropped_snr{};
    std::uint64_t max_per_cell_used{};   ///< Below max_per_cell: memory bound.
    std::vector<std::uint32_t> retained; ///< Spool rows kept, ascending.
};

/// Final-leaf datasets of an EP run: one row per leaf.
constexpr std::array kLeafDatasets{"param_sets", "scores", "scores_ep",
                                   "width"};

/// Writes the leaves @p leaves (ascending) of the spool run @p src_path into
/// @p dst, reading runs of consecutive rows at once. A run the search
/// already reduced stores its leaves' indices in `leaf_index`.
void copy_leaf_rows(const HighFive::File& raw,
                    const std::string& src_path,
                    HighFive::Group& dst,
                    std::span<const std::uint32_t> leaves) {
    const auto src = raw.getGroup(src_path);
    std::vector<std::uint32_t> rows(leaves.begin(), leaves.end());
    if (src.exist("leaf_index")) {
        std::vector<std::uint32_t> spool_leaves;
        src.getDataSet("leaf_index").read(spool_leaves);
        for (auto& row : rows) {
            const auto it = std::ranges::lower_bound(spool_leaves, row);
            if (it == spool_leaves.end() || *it != row) {
                throw std::logic_error("retained leaf missing from the spool");
            }
            row = static_cast<std::uint32_t>(it - spool_leaves.begin());
        }
    }
    for (const auto* name : kLeafDatasets) {
        const auto in    = src.getDataSet(name);
        const auto type  = in.getDataType();
        auto dims        = in.getDimensions();
        std::size_t cols = 1;
        for (std::size_t d = 1; d < dims.size(); ++d) {
            cols *= dims[d];
        }
        const std::size_t row_bytes = cols * type.getSize();
        std::vector<unsigned char> buffer(rows.size() * row_bytes);
        for (std::size_t i = 0; i < rows.size();) {
            std::size_t j = i + 1;
            while (j < rows.size() && rows[j] == rows[j - 1] + 1) {
                ++j;
            }
            std::vector<std::size_t> offset(dims.size(), 0);
            std::vector<std::size_t> count(dims);
            offset[0] = rows[i];
            count[0]  = j - i;
            in.select(offset, count)
                .read_raw(buffer.data() + i * row_bytes, type);
            i = j;
        }
        dims[0]  = rows.size();
        auto out = dst.createDataSet(name, HighFive::DataSpace(dims), type);
        copy_attributes(in, out);
        if (!rows.empty()) {
            out.write_raw(buffer.data(), type);
        }
    }
    dst.createDataSet("leaf_index",
                      std::vector<std::uint32_t>(leaves.begin(), leaves.end()))
        .createAttribute("description",
                         std::string("leaf of each row among the run's final "
                                     "leaves; detections' source_index"));
}

/// Copies the spool's `chunks` tree one run at a time, with a checkpoint
/// between runs: a single H5Ocopy of a multi-GB tree cannot be interrupted.
/// The copied groups, datasets and attributes are those of one H5Ocopy.
/// With @p reductions (one per source), a run keeps only its retained
/// leaves, with their spool rows in `leaf_index` and the reduction's counts
/// as attributes; its other objects are copied whole.
void copy_ep_chunks(const HighFive::File& raw,
                    HighFive::File& result,
                    PostProgress& progress,
                    std::span<const EPSource> sources,
                    std::span<const RunReduction> reductions,
                    const search::CandidateOptions& options,
                    double snr_min) {
    std::unordered_map<std::string, std::size_t> ordinal;
    for (std::size_t i = 0; i < reductions.size(); ++i) {
        ordinal.emplace(sources[i].path, i);
    }
    const auto src_chunks  = raw.getGroup("chunks");
    auto dst_chunks        = result.createGroup("chunks");
    const auto chunk_names = src_chunks.listObjectNames();
    std::uint64_t n_runs   = 0;
    for (const auto& chunk_name : chunk_names) {
        const auto chunk = src_chunks.getGroup(chunk_name);
        if (chunk.exist("runs")) {
            n_runs += chunk.getGroup("runs").getNumberObjects();
        }
    }
    copy_attributes(src_chunks, dst_chunks);
    progress.phase("copying run diagnostics", n_runs);
    for (const auto& chunk_name : chunk_names) {
        const auto chunk_path = "chunks/" + chunk_name;
        const auto src_chunk  = raw.getGroup(chunk_path);
        auto dst_chunk        = dst_chunks.createGroup(chunk_name);
        copy_attributes(src_chunk, dst_chunk);
        for (const auto& name : src_chunk.listObjectNames()) {
            if (name != "runs" ||
                src_chunk.getObjectType(name) != HighFive::ObjectType::Group) {
                copy_object(raw, chunk_path + "/" + name, dst_chunk, name);
                continue;
            }
            const auto src_runs = src_chunk.getGroup("runs");
            auto dst_runs       = dst_chunk.createGroup("runs");
            copy_attributes(src_runs, dst_runs);
            for (const auto& run : src_runs.listObjectNames()) {
                // A run can be hundreds of MB: consider logging every run.
                progress.poke();
                progress.tick();
                const auto run_path = chunk_path + "/runs/" + run;
                const auto found    = ordinal.find(run_path);
                if (found == ordinal.end()) {
                    copy_object(raw, run_path, dst_runs, run);
                    continue;
                }
                const auto& reduction = reductions[found->second];
                const auto src_run    = raw.getGroup(run_path);
                auto dst_run          = dst_runs.createGroup(run);
                for (const auto& object : src_run.listObjectNames()) {
                    if (object != "leaf_index" &&
                        std::ranges::find(kLeafDatasets, object) ==
                            kLeafDatasets.end()) {
                        copy_object(raw, run_path + "/" + object, dst_run,
                                    object);
                    }
                }
                copy_leaf_rows(raw, run_path, dst_run, reduction.retained);
                dst_run.createAttribute("n_final_leaves",
                                        reduction.final_leaves);
                dst_run.createAttribute("n_leaves_above_snr_min",
                                        reduction.above_snr_min);
                dst_run.createAttribute("n_cell_dropped",
                                        reduction.cell_dropped);
                dst_run.createAttribute("n_dominated", reduction.dominated);
                dst_run.createAttribute(
                    "n_leaves_retained",
                    static_cast<std::uint64_t>(reduction.retained.size()));
                dst_run.createAttribute("max_cell_dropped_snr",
                                        reduction.max_cell_dropped_snr);
                dst_run.createAttribute("max_per_cell_used",
                                        reduction.max_per_cell_used);
                // The options of the reduction: republishing this product
                // needs them, as for a spool the search reduced.
                dst_run.createAttribute("reduced_max_per_cell",
                                        options.max_per_cell);
                dst_run.createAttribute("reduced_cell_bins", options.cell_bins);
                dst_run.createAttribute("reduced_cell_log_width",
                                        options.cell_log_width);
                dst_run.createAttribute("reduced_dominance_kappa",
                                        options.dominance_kappa);
                dst_run.createAttribute("reduced_radius", options.radius);
                dst_run.createAttribute("reduced_phase_radius",
                                        options.phase_radius);
                dst_run.createAttribute("reduced_snr_min", snr_min);
                dst_run.createAttribute(
                    "reduced_at",
                    src_run.hasAttribute("reduced_at")
                        ? src_run.getAttribute("reduced_at").read<std::string>()
                        : std::string("publication"));
                // A reduced spool's counts are already folded into these;
                // its other attributes are copied as they are.
                copy_attributes(src_run, dst_run);
            }
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

std::vector<EPSource> ep_sources(const HighFive::File& file,
                                 const search::CandidateOptions& options) {
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
    // A run the search already reduced keeps only some leaves; leaf_index
    // gives each row's leaf, the identity of its detection.
    std::vector<std::uint32_t> leaf_index;
    if (run.exist("leaf_index") && count != 0) {
        leaf_index.resize(count);
        run.getDataSet("leaf_index").select({start}, {count}).read(leaf_index);
    }
    const std::size_t stride = (nparams + 2) * 2;
    std::vector<Candidate> records;
    records.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!utils::is_finite(scores[i])) {
            throw std::invalid_argument("non-finite final EP score");
        }
        if (scores[i] < snr_min) {
            continue;
        }
        const std::size_t leaf = leaf_index.empty() ? start + i : leaf_index[i];
        if (leaf > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "EP leaf index exceeds detection identity capacity");
        }
        records.push_back(search::detail::ep_leaf_candidate(
            std::span<const double>(leaves).subspan(i * stride, stride),
            nparams, scores[i], widths[i], nbins, run_id,
            static_cast<std::uint32_t>(leaf)));
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
                const std::filesystem::path& output,
                PostProgress& progress) {
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
        progress.tick_merge();
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
    ExternalSort(std::filesystem::path directory, PostProgress& progress)
        : m_directory(std::move(directory)),
          m_progress(&progress) {}
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
            merge_runs(inputs, merged, *m_progress);
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
                merge_runs(inputs, merged, *m_progress);
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
    PostProgress* m_progress;
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
/// A run the search reduced keeps only some leaves: publication must use the
/// options it was reduced with, and its snr_min, to give the same product.
void check_reduced_run(const HighFive::Group& run,
                       const search::CandidateOptions& options,
                       double snr_min) {
    if (!run.hasAttribute("reduced_max_per_cell")) {
        return;
    }
    const auto u64 = [&](const char* name) {
        return run.getAttribute(name).read<std::uint64_t>();
    };
    const auto f64 = [&](const char* name) {
        return run.getAttribute(name).read<double>();
    };
    const bool same =
        u64("reduced_max_per_cell") == options.max_per_cell &&
        f64("reduced_cell_bins") == options.cell_bins &&
        f64("reduced_cell_log_width") == options.cell_log_width &&
        f64("reduced_dominance_kappa") == options.dominance_kappa &&
        f64("reduced_radius") == options.radius &&
        f64("reduced_phase_radius") == options.phase_radius &&
        f64("reduced_snr_min") == snr_min;
    if (!same) {
        throw std::invalid_argument(fmt::format(
            "the search reduced its final leaves with max_per_cell = {}, "
            "cell_bins = {}, cell_log_width = {}, dominance_kappa = {}, "
            "radius = {}, phase_radius = {} and snr_min = {}; publish with the "
            "same options, or search again with max_per_cell = 0 to keep "
            "every leaf",
            u64("reduced_max_per_cell"), f64("reduced_cell_bins"),
            f64("reduced_cell_log_width"), f64("reduced_dominance_kappa"),
            f64("reduced_radius"), f64("reduced_phase_radius"),
            f64("reduced_snr_min")));
    }
}

void publish_candidate_results(const std::filesystem::path& spool,
                               const std::filesystem::path& output,
                               const ObservationMetadata& observation,
                               const search::CandidateOptions& options,
                               bool ep,
                               double snr_min) {
    options.validate();
    observation.validate();
    const auto started = std::chrono::steady_clock::now();
    PostProgress progress;
    ScratchDirectory scratch(options.temporary_directory.empty()
                                 ? (output.parent_path().empty()
                                        ? std::filesystem::path(".")
                                        : output.parent_path())
                                 : options.temporary_directory);
    const HighFive::File raw(spool.string(), HighFive::File::ReadOnly);
    std::vector<std::string> names;
    raw.getAttribute("param_names").read(names);
    const auto nparams = names.size();
    if (nparams == 0 || nparams > kParamNames.size() || (ep && nparams < 2)) {
        throw std::invalid_argument("unsupported candidate parameter count");
    }
    // The derivative conversion below assumes the kinematic Taylor order
    // [..., jerk, accel, freq]; the Python reader checks the same suffix.
    for (std::size_t i = 0; i < nparams; ++i) {
        if (names[i] != kParamNames[kParamNames.size() - nparams + i]) {
            throw std::invalid_argument(
                "unsupported native parameter convention");
        }
    }
    // Only a finalized search spool is complete; a ".tmp" left by an
    // interrupted search would publish a truncated candidate list.
    int complete = 0;
    if (raw.hasAttribute("complete")) {
        raw.getAttribute("complete").read(complete);
    }
    if (complete != 1) {
        throw std::invalid_argument("search spool is incomplete");
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
    ExternalSort sort(scratch.path, progress);
    // With a cell cap, only the trials it keeps are sorted and grouped.
    std::optional<search::detail::CellCap> cap;
    if (options.max_per_cell != 0) {
        cap.emplace(options, tobs, 2 * source_bytes);
    }
    std::optional<search::detail::DominanceReducer> dominance;
    if (cap && options.dominance_kappa > 0) {
        dominance.emplace(tobs, options);
    }
    std::uint64_t dominated = 0;
    // Trials a search that reduced its runs dropped before the spool.
    std::uint64_t spool_cell_dropped = 0;
    std::uint64_t spool_dominated    = 0;
    // Per-run accounting of a reduced EP spool; the result keeps only the
    // retained leaves of each run.
    std::vector<RunReduction> reductions(ep && cap ? sources.size() : 0);
    // Cells belong to one source, so an EP run's kept trials can be reduced
    // and sorted as soon as it is read: the cap then holds one run at a time.
    const auto drain = [&](RunReduction* run) {
        if (!cap) {
            return;
        }
        const float max_cell_dropped = cap->max_dropped_snr();
        const auto used              = cap->max_per_cell_used();
        if (used < options.max_per_cell) {
            spdlog::warn("Candidate cells kept {} trials each instead of {} "
                         "to fit candidate memory_gb",
                         used, options.max_per_cell);
        }
        auto kept                    = cap->take();
        const std::uint64_t absorbed = dominance ? dominance->reduce(kept) : 0;
        dominated += absorbed;
        if (run != nullptr) {
            run->dominated            = absorbed;
            run->max_cell_dropped_snr = max_cell_dropped;
            run->max_per_cell_used    = used;
            run->retained.reserve(kept.size());
            for (const auto& c : kept) {
                run->retained.push_back(c.source_index);
            }
            std::ranges::sort(run->retained);
        }
        nrows += kept.size();
        for (std::size_t start = 0; start < kept.size(); start += batch) {
            std::vector<Candidate> records(
                kept.begin() + static_cast<std::ptrdiff_t>(start),
                kept.begin() + static_cast<std::ptrdiff_t>(
                                   std::min(start + batch, kept.size())));
            sort.add(records);
        }
    };
    const auto keep = [&](std::vector<Candidate>& records) {
        if (cap) {
            for (const auto& c : records) {
                cap->add(c);
            }
        } else if (!records.empty()) {
            nrows += records.size();
            sort.add(records);
        }
    };
    if (ep) {
        progress.phase("reading and sorting runs", sources.size());
        for (const auto& source : sources) {
            check_reduced_run(raw.getGroup(source.path), options, snr_min);
        }
        for (std::size_t run_id = 0; run_id < sources.size(); ++run_id) {
            progress.poke();
            progress.tick();
            const auto count = raw.getGroup(sources[run_id].path)
                                   .getDataSet("scores")
                                   .getDimensions()[0];
            for (std::size_t start = 0; start < count; start += batch) {
                loki::check_interrupt();
                auto records = read_ep_candidates(
                    raw, sources[run_id], static_cast<std::uint32_t>(run_id),
                    start, std::min(batch, count - start), nparams, snr_min);
                if (!reductions.empty()) {
                    reductions[run_id].above_snr_min += records.size();
                }
                keep(records);
            }
            if (!reductions.empty()) {
                auto& run        = reductions[run_id];
                run.final_leaves = count;
                run.cell_dropped = run.above_snr_min - cap->kept();
            }
            drain(reductions.empty() ? nullptr : &reductions[run_id]);
            // A run the search reduced: add the search's counts.
            const auto group = raw.getGroup(sources[run_id].path);
            if (!reductions.empty() && group.hasAttribute("n_final_leaves")) {
                auto& run      = reductions[run_id];
                const auto get = [&](const char* name) {
                    return group.getAttribute(name).read<std::uint64_t>();
                };
                run.final_leaves  = get("n_final_leaves");
                run.above_snr_min = get("n_leaves_above_snr_min");
                run.cell_dropped += get("n_cell_dropped");
                run.dominated += get("n_dominated");
                spool_cell_dropped += get("n_cell_dropped");
                spool_dominated += get("n_dominated");
                run.max_cell_dropped_snr = std::max(
                    run.max_cell_dropped_snr,
                    group.getAttribute("max_cell_dropped_snr").read<float>());
                if (group.hasAttribute("max_per_cell_used")) {
                    run.max_per_cell_used = std::min(run.max_per_cell_used,
                                                     get("max_per_cell_used"));
                }
            }
        }
    } else {
        const auto count = raw.getDataSet("snr").getDimensions()[0];
        progress.phase("reading and sorting detections", count);
        for (std::size_t start = 0; start < count; start += batch) {
            progress.poke();
            progress.tick(std::min(batch, count - start));
            auto records = read_candidates(
                raw, start, std::min(batch, count - start), nparams);
            keep(records);
        }
    }
    drain(nullptr);
    const std::uint64_t cell_dropped =
        (cap ? cap->dropped() : 0) + spool_cell_dropped;
    dominated += spool_dominated;
    cap.reset();
    progress.phase("merging sorted runs", 0);
    const auto sorted = sort.finish();
    // The atomic publication file must live beside the destination, even if
    // scratch sorting is explicitly placed on another filesystem.
    ScratchDirectory publication(output.parent_path().empty()
                                     ? std::filesystem::path(".")
                                     : output.parent_path());
    const auto publish          = publication.path / "result.partial";
    std::uint64_t cluster_count = 0;
    std::uint64_t selected      = 0;
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
            result.createAttribute("n_cell_dropped", cell_dropped);
            result.createAttribute("n_dominated", dominated);
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
            config.createAttribute("max_per_cell", options.max_per_cell);
            config.createAttribute("cell_bins", options.cell_bins);
            config.createAttribute("cell_log_width", options.cell_log_width);
            config.createAttribute("dominance_kappa", options.dominance_kappa);
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
            copy_attributes(raw.getGroup("/"), search_group);
            if (raw.exist("regions")) {
                copy_object(raw, "regions", search_group, "regions");
            }
            if (raw.exist("timer_stats")) {
                copy_object(raw, "timer_stats", search_group, "timer_stats");
            }
            if (ep) {
                copy_ep_chunks(raw, result, progress, sources, reductions,
                               options, snr_min);
                copy_attributes(raw.getGroup("/"), result, true);
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
            progress.phase(singleton_stream ? "writing detections"
                                            : "clustering detections",
                           nrows);
            if (!sorted.empty()) {
                std::ifstream in(sorted, std::ios::binary);
                Candidate c;
                while (read_record(in, c)) {
                    progress.tick();
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
                progress.phase("finishing clusters", 0);
                processor.finish();
                // Transitive append IDs are provisional: a later bridge can
                // join earlier groups. Remap only after the graph is complete,
                // in bounded batches, preserving raw score order and identity.
                if (options.retain_detections && options.cluster &&
                    options.grouping == "transitive") {
                    auto dataset = result.getDataSet("detections/cluster_id");
                    progress.phase("remapping cluster ids", nrows);
                    for (std::size_t start = 0; start < nrows; start += batch) {
                        const auto count = std::min(batch, nrows - start);
                        progress.poke();
                        progress.tick(count);
                        auto ids = read_rows<std::uint64_t>(
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
    // The search logs its own stages; this is the host post-processing that
    // follows it, which can dominate a GPU run (clustering a bright source).
    const auto seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    spdlog::info("Candidate postprocessing complete in {:.2f} s: {} "
                 "detections, {} clusters, {} candidates -> {}",
                 seconds, nrows, cluster_count, selected, output.string());
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
