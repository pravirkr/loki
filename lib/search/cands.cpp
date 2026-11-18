#include "lib/search/cands.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <hdf5.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/H5PropertyList.hpp>
#include <highfive/highfive.hpp>
#include <highfive/span.hpp>
#include <omp.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"
#include "lib/io/h5_layout.hpp"
#include "lib/search/slice_scratch.hpp"
#include "lib/utils/world_tree.hpp"

namespace loki::search {

namespace {
// Explicit Round-half-to-even (bankers' rounding) to match numpy.round()
// Never use std::round() or std::nearbyint() as they are not deterministic.
double round_dp(double x, int digits) noexcept {
    if (!utils::is_finite(x)) {
        return x;
    }

    const double scale = std::pow(10.0, digits);
    const double y     = x * scale;

    const double f = std::floor(y);
    const double r = y - f;

    double rounded;
    if (r > 0.5) {
        rounded = f + 1.0;
    } else if (r < 0.5) {
        rounded = f;
    } else {
        // exactly .5 → tie to even
        rounded = (std::fmod(f, 2.0) == 0.0) ? f : (f + 1.0);
    }

    return rounded / scale;
}

// Returns (ref_seg, task_id) as integers, or (-1, -1) if not matched
std::tuple<int, int> extract_ref_seg_task_id(const std::string& filename) {
    const std::regex re(R"(tmp_(\d{3})_(\d{2})_.*\.(?:txt|h5))");
    std::smatch match;
    if (std::regex_match(filename, match, re)) {
        return {std::stoi(match[1]), std::stoi(match[2])};
    }
    return {-1, -1};
}

// Create a compound type for FFATimerStats
HighFive::CompoundType create_compound_ffa_timer_stats() {
    return {
        {"brutefold", HighFive::create_datatype<float>()},
        {"ffa", HighFive::create_datatype<float>()},
        {"score", HighFive::create_datatype<float>()},
        {"io", HighFive::create_datatype<float>()},
    };
}

// Create a compound type for PruneStats
HighFive::CompoundType create_compound_prune_stats() {
    return {
        {"level", HighFive::create_datatype<SizeType>()},
        {"seg_idx", HighFive::create_datatype<SizeType>()},
        {"threshold", HighFive::create_datatype<float>()},
        {"score_min", HighFive::create_datatype<float>()},
        {"score_max", HighFive::create_datatype<float>()},
        {"n_branches", HighFive::create_datatype<SizeType>()},
        {"n_leaves", HighFive::create_datatype<SizeType>()},
        {"n_leaves_resolved", HighFive::create_datatype<SizeType>()},
        {"n_leaves_phy", HighFive::create_datatype<SizeType>()},
        {"n_leaves_surv", HighFive::create_datatype<SizeType>()},
        {"n_leaves_masked", HighFive::create_datatype<SizeType>()},
        {"n_leaves_vetoed", HighFive::create_datatype<SizeType>()},
        {"n_harvested", HighFive::create_datatype<SizeType>()},
        {"n_overloads", HighFive::create_datatype<SizeType>()},
    };
}

HighFive::CompoundType create_compound_circular_stage_counts() {
    return {
        {"level", HighFive::create_datatype<SizeType>()},
        {"n_leaves_snap", HighFive::create_datatype<SizeType>()},
        {"n_leaves_crackle", HighFive::create_datatype<SizeType>()},
    };
}

HighFive::CompoundType create_compound_prune_timer_stats() {
    return {
        {"branch", HighFive::create_datatype<float>()},
        {"validate", HighFive::create_datatype<float>()},
        {"resolve", HighFive::create_datatype<float>()},
        {"shift_add", HighFive::create_datatype<float>()},
        {"score", HighFive::create_datatype<float>()},
        {"transform", HighFive::create_datatype<float>()},
        {"threshold", HighFive::create_datatype<float>()},
        {"rfi", HighFive::create_datatype<float>()},
    };
}

} // namespace

// --- FFATimerStats ---
FFATimerStats::FFATimerStats() {
    for (const auto& name : kTimerNames) {
        m_timers[name] = 0.0F;
    }
}
float& FFATimerStats::operator[](const std::string& key) {
    return m_timers[key];
}
const float& FFATimerStats::operator[](const std::string& key) const {
    return m_timers.at(key);
}
const float& FFATimerStats::at(const std::string& key) const {
    return m_timers.at(key);
}
float& FFATimerStats::at(const std::string& key) { return m_timers.at(key); }
bool FFATimerStats::contains(const std::string& key) const {
    return m_timers.contains(key);
}
auto FFATimerStats::begin() const { return m_timers.begin(); }
auto FFATimerStats::end() const { return m_timers.end(); }
auto FFATimerStats::begin() { return m_timers.begin(); }
auto FFATimerStats::end() { return m_timers.end(); }
float FFATimerStats::total() const {
    return std::accumulate(
        m_timers.begin(), m_timers.end(), 0.0F,
        [](float sum, const auto& pair) { return sum + pair.second; });
}
void FFATimerStats::reset() {
    for (auto& [name, time] : m_timers) {
        time = 0.0F;
    }
}
FFATimerStats& FFATimerStats::operator+=(const FFATimerStats& other) {
    for (const auto& [name, time] : other.m_timers) {
        m_timers[name] += time;
    }
    return *this;
}

std::string FFATimerStats::get_concise_timer_summary() const {
    const float total_time = total();
    if (total_time == 0.0F) {
        return "Total: 0.0s";
    }

    // Copy timers to a vector to sort them by time
    std::vector<std::pair<std::string, float>> sorted_times(begin(), end());
    std::ranges::sort(sorted_times, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::string breakdown;
    int count = 0;
    for (const auto& [name, time] : sorted_times) {
        if (time > 0 && count < 4) {
            if (!breakdown.empty()) {
                breakdown += " | ";
            }
            breakdown +=
                fmt::format("{}: {:.0f}%", name, (time / total_time) * 100.0F);
            count++;
        }
    }
    return fmt::format("Total: {:.1f}s ({})", total_time, breakdown);
}

// --- FFAStatsCollection ---
void FFAStatsCollection::update_stats(const FFATimerStats& timers,
                                      float flops) {
    m_accumulated_timers += timers;
    m_accumulated_flops += flops;
}
std::string FFAStatsCollection::get_concise_timer_summary() const {
    const float total_time = m_accumulated_timers.total();
    if (total_time == 0.0F) {
        return "Total: 0.0s";
    }

    // Copy timers to a vector to sort them by time
    std::vector<std::pair<std::string, float>> sorted_times(
        m_accumulated_timers.begin(), m_accumulated_timers.end());
    std::ranges::sort(sorted_times, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::string breakdown;
    int count = 0;
    for (const auto& [name, time] : sorted_times) {
        if (time > 0 && count < 4) {
            if (!breakdown.empty()) {
                breakdown += " | ";
            }
            breakdown +=
                fmt::format("{}: {:.0f}%", name, (time / total_time) * 100.0F);
            count++;
        }
    }
    return fmt::format("Total: {:.1f}s ({})", total_time, breakdown);
}

std::vector<FFATimerStatsPacked> FFAStatsCollection::get_packed_data() const {
    std::vector<FFATimerStatsPacked> packed_timers;
    if (m_accumulated_timers.total() > 0.0F) {
        packed_timers.emplace_back(m_accumulated_timers.at("brutefold"),
                                   m_accumulated_timers.at("ffa"),
                                   m_accumulated_timers.at("score"),
                                   m_accumulated_timers.at("io"));
    }
    return packed_timers;
}

// --- PruneStats ---
double PruneStats::lb_leaves() const noexcept {
    if (n_leaves == 0) {
        return 0.0;
    }
    return round_dp(std::log2(static_cast<double>(n_leaves)), 2);
}
double PruneStats::lb_leaves_phys() const noexcept {
    if (n_leaves_phy == 0) {
        return 0.0;
    }
    return round_dp(std::log2(static_cast<double>(n_leaves_phy)), 2);
}
double PruneStats::branch_frac() const noexcept {
    if (n_branches == 0) {
        return 0.0;
    }
    return round_dp(
        static_cast<double>(n_leaves) / static_cast<double>(n_branches), 2);
}
double PruneStats::phys_frac() const noexcept {
    if (n_leaves == 0) {
        return 0.0;
    }
    return round_dp(
        static_cast<double>(n_leaves_phy) / static_cast<double>(n_leaves), 2);
}
double PruneStats::surv_frac() const noexcept {
    if (n_leaves_phy == 0) {
        return 0.0;
    }
    return round_dp(static_cast<double>(n_leaves_surv) /
                        static_cast<double>(n_leaves_phy),
                    2);
}
std::string PruneStats::get_summary() const {
    std::string summary = fmt::format(
        "Prune level: {:3d}, seg_idx: {:3d}, leaves: {:5.2f}, "
        "leaves_phys: {:5.2f}, branch_frac: {:5.2f},"
        "score thresh: {:5.2f}, max: {:5.2f}, min: {:5.2f}, "
        "P(surv): {:4.2f}",
        level, seg_idx, lb_leaves(), lb_leaves_phys(), branch_frac(), threshold,
        score_max, score_min, surv_frac());
    if (n_leaves_masked > 0 || n_leaves_vetoed > 0 || n_harvested > 0) {
        summary += fmt::format(", masked: {}, vetoed: {}, harvested: {}",
                               n_leaves_masked, n_leaves_vetoed, n_harvested);
    }
    if (n_overloads > 0) {
        summary += fmt::format(", overloads: {}", n_overloads);
    }
    summary += '\n';
    return summary;
}

// --- PruneTimerStats ---
PruneTimerStats::PruneTimerStats() {
    for (const auto& name : kTimerNames) {
        m_timers[name] = 0.0F;
    }
}
float& PruneTimerStats::operator[](const std::string& key) {
    return m_timers[key];
}
const float& PruneTimerStats::operator[](const std::string& key) const {
    return m_timers.at(key);
}
float& PruneTimerStats::at(const std::string& key) { return m_timers.at(key); }
const float& PruneTimerStats::at(const std::string& key) const {
    return m_timers.at(key);
}
bool PruneTimerStats::contains(const std::string& key) const {
    return m_timers.contains(key);
}
auto PruneTimerStats::begin() const { return m_timers.begin(); }
auto PruneTimerStats::end() const { return m_timers.end(); }
auto PruneTimerStats::begin() { return m_timers.begin(); }
auto PruneTimerStats::end() { return m_timers.end(); }
float PruneTimerStats::total() const {
    return std::accumulate(
        m_timers.begin(), m_timers.end(), 0.0F,
        [](float sum, const auto& pair) { return sum + pair.second; });
}
void PruneTimerStats::reset() {
    for (auto& [name, time] : m_timers) {
        time = 0.0F;
    }
}
PruneTimerStats& PruneTimerStats::operator+=(const PruneTimerStats& other) {
    for (const auto& [name, time] : other.m_timers) {
        m_timers[name] += time;
    }
    return *this;
}

// --- PruneStatsCollection ---
SizeType PruneStatsCollection::get_nstages() const {
    return m_stats_list.size();
}
void PruneStatsCollection::update_stats(const PruneStats& stats,
                                        const PruneTimerStats& timers) {
    m_stats_list.push_back(stats);
    m_accumulated_timers += timers;
}
void PruneStatsCollection::update_stats(const PruneStats& stats) {
    m_stats_list.push_back(stats);
}
void PruneStatsCollection::merge_slice(const PruneStatsCollection& other,
                                       bool sum_resolved) {
    for (const auto& row : other.m_stats_list) {
        const auto it =
            std::ranges::find_if(m_stats_list, [&row](const auto& s) {
                return s.level == row.level;
            });
        if (it == m_stats_list.end()) {
            m_stats_list.push_back(row);
            continue;
        }
        if (row.n_leaves_phy > 0) {
            if (it->n_leaves_phy > 0) {
                it->score_min = std::min(it->score_min, row.score_min);
                it->score_max = std::max(it->score_max, row.score_max);
            } else {
                it->score_min = row.score_min;
                it->score_max = row.score_max;
            }
        }
        it->n_branches += row.n_branches;
        it->n_leaves += row.n_leaves;
        if (sum_resolved && row.level > 0) {
            it->n_leaves_resolved += row.n_leaves_resolved;
        }
        it->n_leaves_phy += row.n_leaves_phy;
        it->n_leaves_surv += row.n_leaves_surv;
        it->n_leaves_masked += row.n_leaves_masked;
        it->n_leaves_vetoed += row.n_leaves_vetoed;
        it->n_harvested += row.n_harvested;
        it->n_overloads += row.n_overloads;
    }
    std::ranges::sort(m_stats_list, {}, &PruneStats::level);
    for (const auto& row : other.m_circular_list) {
        const auto it =
            std::ranges::find_if(m_circular_list, [&row](const auto& c) {
                return c.level == row.level;
            });
        if (it == m_circular_list.end()) {
            m_circular_list.push_back(row);
            continue;
        }
        it->n_leaves_snap += row.n_leaves_snap;
        it->n_leaves_crackle += row.n_leaves_crackle;
    }
    std::ranges::sort(m_circular_list, {}, &CircularStageCounts::level);
    m_accumulated_timers += other.m_accumulated_timers;
}
void PruneStatsCollection::update_circular(const CircularStageCounts& counts) {
    m_circular_list.push_back(counts);
}
std::optional<PruneStats>
PruneStatsCollection::get_stats(SizeType level) const {
    const auto it = std::ranges::find_if(
        m_stats_list, [level](const auto& s) { return s.level == level; });
    return it != m_stats_list.end() ? std::optional{*it} : std::nullopt;
}
std::string PruneStatsCollection::get_all_summaries() const {
    auto sorted_stats = m_stats_list;
    std::ranges::sort(sorted_stats, {}, &PruneStats::level);

    std::string result;
    for (const auto& stats : sorted_stats) {
        result += stats.get_summary();
    }
    return result;
}
std::string PruneStatsCollection::get_stats_summary() const {
    if (m_stats_list.empty()) {
        return "No stats available.";
    }
    const auto& last_stats = m_stats_list.back();
    SizeType n_masked      = 0;
    SizeType n_vetoed      = 0;
    SizeType n_harvested   = 0;
    for (const auto& stats : m_stats_list) {
        n_masked += stats.n_leaves_masked;
        n_vetoed += stats.n_leaves_vetoed;
        n_harvested += stats.n_harvested;
    }
    auto summary = fmt::format("Score: {:.2f}, Leaves: {:.2f}",
                               last_stats.score_max, last_stats.lb_leaves());
    if (n_masked > 0 || n_vetoed > 0 || n_harvested > 0) {
        summary += fmt::format(", masked: {}, vetoed: {}, harvested: {}",
                               n_masked, n_vetoed, n_harvested);
    }
    return summary;
}
std::string PruneStatsCollection::get_stats_summary_cuda(float duration) const {
    if (m_stats_list.empty()) {
        return "No stats available. Duration: 0.0s";
    }
    const auto& last_stats = m_stats_list.back();
    return fmt::format("Score: {:.2f}, Leaves: {:.2f}, Total: {:.2f}s",
                       last_stats.score_max, last_stats.lb_leaves(), duration);
}
std::string PruneStatsCollection::get_timer_summary() const {
    const float total_time = m_accumulated_timers.total();
    if (total_time == 0.0F) {
        return "Timing breakdown: 0.00s\n";
    }
    std::string summary =
        fmt::format("Timing breakdown: {:.2f}s\n", total_time);
    std::vector<std::pair<std::string, float>> sorted_timers(
        m_accumulated_timers.begin(), m_accumulated_timers.end());
    std::ranges::sort(sorted_timers, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    for (const auto& [name, time] : sorted_timers) {
        const auto percent = (time / total_time) * 100.0F;
        summary += fmt::format("  {:10s}: {:6.1f}%\n", name, percent);
    }
    return summary;
}
std::string PruneStatsCollection::get_concise_timer_summary() const {
    const float total_time = m_accumulated_timers.total();
    if (total_time == 0.0F) {
        return "Total: 0.0s";
    }

    // Copy timers to a vector to sort them by time
    std::vector<std::pair<std::string, float>> sorted_times(
        m_accumulated_timers.begin(), m_accumulated_timers.end());
    std::ranges::sort(sorted_times, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::string breakdown;
    int count = 0;
    for (const auto& [name, time] : sorted_times) {
        if (time > 0 && count < 4) {
            if (!breakdown.empty()) {
                breakdown += " | ";
            }
            breakdown +=
                fmt::format("{}: {:.0f}%", name, (time / total_time) * 100.0F);
            count++;
        }
    }
    return fmt::format("Total: {:.1f}s ({})", total_time, breakdown);
}
std::pair<std::vector<PruneStats>, std::vector<PruneTimerStatsPacked>>
PruneStatsCollection::get_packed_data() const {
    std::vector<PruneTimerStatsPacked> packed_timers;
    if (m_accumulated_timers.total() > 0.0F) {
        packed_timers.emplace_back(m_accumulated_timers.at("branch"),
                                   m_accumulated_timers.at("validate"),
                                   m_accumulated_timers.at("resolve"),
                                   m_accumulated_timers.at("shift_add"),
                                   m_accumulated_timers.at("score"),
                                   m_accumulated_timers.at("transform"),
                                   m_accumulated_timers.at("batch_add"),
                                   m_accumulated_timers.at("rfi"));
    }
    return {m_stats_list, packed_timers};
}

namespace {
// Create a 1D dataset from a span, tolerating empty input (HDF5 allows a
// zero-sized simple dataspace, but no data may be written to it).
template <typename T>
void create_1d_dataset(HighFive::Group& group,
                       const std::string& name,
                       std::span<const T> data) {
    auto ds = group.createDataSet(name, HighFive::DataSpace({data.size()}),
                                  HighFive::create_datatype<T>());
    if (!data.empty()) {
        ds.write_raw(data.data(), HighFive::create_datatype<T>());
    }
}
} // namespace

// --- HarvestBuffer ---
template <SupportedFoldType FoldType>
HarvestBuffer<FoldType>::HarvestBuffer(SizeType leaves_stride,
                                       SizeType folds_stride,
                                       bool store_folds)
    : m_leaves_stride(leaves_stride),
      m_folds_stride(folds_stride),
      m_store_folds(store_folds) {
    error_check::check_greater(leaves_stride, SizeType{0},
                               "HarvestBuffer: leaves_stride must be > 0");
    error_check::check_greater(folds_stride, SizeType{0},
                               "HarvestBuffer: folds_stride must be > 0");
}

template <SupportedFoldType FoldType>
void HarvestBuffer<FoldType>::clear() noexcept {
    m_size = 0;
    m_leaves.clear();
    m_folds.clear();
    m_scores.clear();
    m_levels.clear();
    m_seg_idx.clear();
    m_t_ref.clear();
}

template <SupportedFoldType FoldType>
void HarvestBuffer<FoldType>::reserve(SizeType n) {
    m_leaves.reserve(n * m_leaves_stride);
    if (m_store_folds) {
        m_folds.reserve(n * m_folds_stride);
    }
    m_scores.reserve(n);
    m_levels.reserve(n);
    m_seg_idx.reserve(n);
    m_t_ref.reserve(n);
}

template <SupportedFoldType FoldType>
SizeType HarvestBuffer<FoldType>::get_memory_bytes() const noexcept {
    return (m_leaves.capacity() * sizeof(double)) +
           (m_folds.capacity() * sizeof(FoldType)) +
           (m_scores.capacity() * sizeof(float)) +
           (m_levels.capacity() * sizeof(SizeType)) +
           (m_seg_idx.capacity() * sizeof(SizeType)) +
           (m_t_ref.capacity() * sizeof(double));
}

template <SupportedFoldType FoldType>
void HarvestBuffer<FoldType>::push(std::span<const double> leaf,
                                   std::span<const FoldType> fold,
                                   float score,
                                   SizeType level,
                                   SizeType seg_idx,
                                   double t_ref) {
    error_check::check_greater_equal(leaf.size(), m_leaves_stride,
                                     "HarvestBuffer::push: leaf too small");
    m_leaves.insert(m_leaves.end(), leaf.begin(),
                    leaf.begin() + static_cast<IndexType>(m_leaves_stride));
    if (m_store_folds) {
        error_check::check_greater_equal(fold.size(), m_folds_stride,
                                         "HarvestBuffer::push: fold too small");
        m_folds.insert(m_folds.end(), fold.begin(),
                       fold.begin() + static_cast<IndexType>(m_folds_stride));
    }
    m_scores.push_back(score);
    m_levels.push_back(level);
    m_seg_idx.push_back(seg_idx);
    m_t_ref.push_back(t_ref);
    ++m_size;
}

template class HarvestBuffer<float>;
template class HarvestBuffer<ComplexType>;

namespace {
constexpr std::size_t kH5ChunkRows = 65536;
constexpr int kH5DeflateLevel      = 3;

HighFive::DataSetCreateProps make_row_chunk_props() {
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({kH5ChunkRows}));
    props.add(HighFive::Shuffle());
    props.add(HighFive::Deflate(kH5DeflateLevel));
    return props;
}
} // namespace

HighFive::File FFAResultWriter::open_file() const {
    HighFive::File::AccessMode open_mode;
    if (m_mode == Mode::kWrite) {
        open_mode = HighFive::File::Overwrite;
    } else if (std::filesystem::exists(m_open_path)) {
        open_mode = HighFive::File::ReadWrite;
    } else {
        open_mode = HighFive::File::Create;
    }

    HighFive::File file =
        io::detail::open_result_file(m_open_path.string(), open_mode);
    if (!file.isValid()) {
        throw std::runtime_error("Failed to create valid HDF5 file");
    }
    return file;
}

// --- FFAResultWriter ---
FFAResultWriter::FFAResultWriter(std::filesystem::path filename, Mode mode)
    : m_final_path(std::move(filename)),
      m_mode(mode),
      m_open_path(m_mode == Mode::kWrite
                      ? std::filesystem::path(m_final_path.string() + ".tmp")
                      : m_final_path) {
    if (m_mode == Mode::kWrite && std::filesystem::exists(m_open_path)) {
        std::filesystem::remove(m_open_path);
    }
    m_file.emplace(open_file());
}

// An unfinalized write is never published: drop its partial file instead of
// leaving "<name>.tmp" beside the results.
FFAResultWriter::~FFAResultWriter() {
    if (m_finalized || m_mode != Mode::kWrite) {
        return;
    }
    try {
        m_file.reset();
    } catch (...) { // NOLINT(bugprone-empty-catch)
    }
    std::error_code error;
    std::filesystem::remove(m_open_path, error);
}

HighFive::File& FFAResultWriter::h5_file() {
    if (!m_file.has_value()) {
        throw std::runtime_error("FFAResultWriter: HDF5 file is not open");
    }
    return *m_file;
}

const HighFive::File& FFAResultWriter::h5_file() const {
    if (!m_file.has_value()) {
        throw std::runtime_error("FFAResultWriter: HDF5 file is not open");
    }
    return *m_file;
}

void FFAResultWriter::ensure_datasets(SizeType n_params) {
    if (m_datasets_initialized) {
        return;
    }
    m_n_params           = n_params;
    const auto row_props = make_row_chunk_props();

    const HighFive::DataSpace snr_space({0}, {HighFive::DataSpace::UNLIMITED});
    h5_file().createDataSet("snr", snr_space,
                            HighFive::create_datatype<float>(), row_props);

    const HighFive::DataSpace param_space(
        {0, n_params}, {HighFive::DataSpace::UNLIMITED, n_params});
    HighFive::DataSetCreateProps param_props;
    param_props.add(HighFive::Chunking({kH5ChunkRows, n_params}));
    param_props.add(HighFive::Shuffle());
    param_props.add(HighFive::Deflate(kH5DeflateLevel));
    h5_file().createDataSet("param_sets", param_space,
                            HighFive::create_datatype<double>(), param_props);

    const HighFive::DataSpace width_space({0},
                                          {HighFive::DataSpace::UNLIMITED});
    h5_file().createDataSet("width", width_space,
                            HighFive::create_datatype<std::uint16_t>(),
                            row_props);

    const HighFive::DataSpace nbins_space({0},
                                          {HighFive::DataSpace::UNLIMITED});
    h5_file().createDataSet("nbins", nbins_space,
                            HighFive::create_datatype<std::uint16_t>(),
                            row_props);

    h5_file().createDataSet("detection_id", snr_space,
                            HighFive::create_datatype<std::uint64_t>(),
                            row_props);
    m_datasets_initialized = true;
}

FFAResultMetadata::FFAResultMetadata(const search::FFASearchConfig& cfg,
                                     std::string_view config_toml)
    : param_names(cfg.get_param_names()),
      config_toml(config_toml),
      tsamp(cfg.get_tsamp()),
      nsamps(cfg.get_nsamps()),
      tobs(cfg.get_tobs()),
      f_min(cfg.get_f_min()),
      f_max(cfg.get_f_max()),
      snr_min(cfg.get_snr_min()),
      ducy_max(cfg.get_ducy_max()),
      wtsp(cfg.get_wtsp()),
      nbins_min(cfg.get_nbins()),
      nbins_max(cfg.get_nbins_max()),
      region_growth_factor(cfg.get_region_growth_factor()),
      eta(cfg.get_eta()),
      use_fourier(cfg.get_use_fourier()),
      bseg_brute(cfg.get_bseg_brute()),
      bseg_ffa(cfg.get_bseg_ffa()) {}

void FFAResultWriter::write_metadata(const FFAResultMetadata& metadata) {
    const std::scoped_lock lock(m_hdf5_mutex);
    if (m_metadata_written) {
        throw std::runtime_error("FFA metadata already written.");
    }
    ensure_datasets(metadata.param_names.size());

    h5_file().createAttribute("ffa_version", "0.1.0-cpp");
    h5_file().createAttribute("complete", 0);
    h5_file().createAttribute("param_names", metadata.param_names);
    h5_file().createAttribute("config_toml", metadata.config_toml);
    h5_file().createAttribute("tsamp", metadata.tsamp);
    h5_file().createAttribute("nsamps", metadata.nsamps);
    h5_file().createAttribute("tobs", metadata.tobs);
    h5_file().createAttribute("f_min", metadata.f_min);
    h5_file().createAttribute("f_max", metadata.f_max);
    h5_file().createAttribute("snr_min", metadata.snr_min);
    h5_file().createAttribute("ducy_max", metadata.ducy_max);
    h5_file().createAttribute("wtsp", metadata.wtsp);
    h5_file().createAttribute("nbins_min", metadata.nbins_min);
    h5_file().createAttribute("nbins_max", metadata.nbins_max);
    h5_file().createAttribute("region_growth_factor",
                              metadata.region_growth_factor);
    h5_file().createAttribute("eta", metadata.eta);
    // Duty-cycle resolution of the grid: eta / nbins turns in every region.
    h5_file().createAttribute("phase_tolerance",
                              metadata.nbins_min > 0
                                  ? metadata.eta /
                                        static_cast<double>(metadata.nbins_min)
                                  : 0.0);
    h5_file().createAttribute("use_fourier", metadata.use_fourier);
    if (metadata.bseg_brute.has_value()) {
        h5_file().createAttribute("bseg_brute", *metadata.bseg_brute);
    }
    if (metadata.bseg_ffa.has_value()) {
        h5_file().createAttribute("bseg_ffa", *metadata.bseg_ffa);
    }
    m_metadata_written = true;
}

void FFAResultWriter::write_regions(std::span<const FFASearchConfig> configs) {
    const std::scoped_lock lock(m_hdf5_mutex);
    auto regions = h5_file().createGroup("regions");
    for (std::size_t i = 0; i < configs.size(); ++i) {
        const auto& cfg = configs[i];
        auto region     = regions.createGroup(fmt::format("region_{:04d}", i));
        region.createAttribute("source_region", static_cast<std::uint64_t>(i));
        region.createAttribute("nbins", cfg.get_nbins());
        region.createAttribute("eta", cfg.get_eta());
        region.createAttribute("f_min", cfg.get_f_min());
        region.createAttribute("f_max", cfg.get_f_max());
        region.createDataSet("scoring_widths", cfg.get_scoring_widths());
        region.createDataSet("param_counts",
                             cfg.get_param_grid_count(cfg.get_tobs()));
        std::vector<double> limits;
        for (const auto& limit : cfg.get_param_limits()) {
            limits.push_back(limit.min);
            limits.push_back(limit.max);
        }
        region
            .createDataSet<double>(
                "param_limits",
                HighFive::DataSpace({cfg.get_nparams(), std::size_t{2}}))
            .write_raw(limits.data());
    }
}

void FFAResultWriter::write_results(
    std::span<const double> param_sets,
    std::span<const float> scores,
    std::span<const std::uint16_t> widths_bins,
    std::span<const std::uint16_t> nbins,
    SizeType n_param_sets,
    SizeType n_params,
    std::span<const std::uint64_t> detection_ids) {
    if (n_param_sets == 0) {
        return;
    }
    const std::scoped_lock lock(m_hdf5_mutex);
    ensure_datasets(n_params);

    if (param_sets.size() != n_param_sets * n_params) {
        throw std::invalid_argument(
            fmt::format("param_sets size does not match ({} != {} * {})",
                        param_sets.size(), n_param_sets, n_params));
    }
    if (scores.size() != n_param_sets || widths_bins.size() != n_param_sets ||
        nbins.size() != n_param_sets) {
        throw std::invalid_argument(
            "scores/width/nbins length must match n_param_sets");
    }

    const auto append_1d = [&](const std::string& name, const void* data,
                               const HighFive::DataType& dtype) {
        auto dset           = h5_file().getDataSet(name);
        const auto old_dims = dset.getSpace().getDimensions();
        const auto old_rows = old_dims.empty() ? 0UL : old_dims[0];
        dset.resize({old_rows + n_param_sets});
        dset.select({old_rows}, {n_param_sets}).write_raw(data, dtype);
    };

    std::vector<std::uint64_t> generated_ids;
    if (detection_ids.empty()) {
        const auto first = h5_file().getDataSet("snr").getDimensions()[0];
        generated_ids.resize(n_param_sets);
        std::iota(generated_ids.begin(), generated_ids.end(), first);
        detection_ids = generated_ids;
    }
    if (detection_ids.size() != n_param_sets) {
        throw std::invalid_argument("detection ID length mismatch");
    }
    append_1d("detection_id", detection_ids.data(),
              HighFive::create_datatype<std::uint64_t>());
    append_1d("snr", scores.data(), HighFive::create_datatype<float>());
    append_1d("width", widths_bins.data(),
              HighFive::create_datatype<std::uint16_t>());
    append_1d("nbins", nbins.data(),
              HighFive::create_datatype<std::uint16_t>());

    auto param_dset     = h5_file().getDataSet("param_sets");
    const auto old_dims = param_dset.getSpace().getDimensions();
    const auto old_rows = old_dims.empty() ? 0UL : old_dims[0];
    param_dset.resize({old_rows + n_param_sets, n_params});
    param_dset.select({old_rows, 0}, {n_param_sets, n_params})
        .write_raw(param_sets.data(), HighFive::create_datatype<double>());
}

void FFAResultWriter::finalize() {
    if (m_finalized) {
        return;
    }
    const std::scoped_lock lock(m_hdf5_mutex);
    if (m_mode == Mode::kWrite) {
        if (!m_metadata_written) {
            throw std::runtime_error(
                "FFAResultWriter::finalize called before write_metadata");
        }
        if (h5_file().hasAttribute("complete")) {
            h5_file().getAttribute("complete").write(1);
        } else {
            h5_file().createAttribute("complete", 1);
        }
        h5_file().flush();
        m_file.reset();
        std::filesystem::rename(m_open_path, m_final_path);
    }
    m_finalized = true;
}

void FFAResultWriter::write_ffa_stats(const FFAStatsCollection& ffa_stats,
                                      float total_runtime) {
    const std::scoped_lock lock(m_hdf5_mutex);
    h5_file().createDataSet("timer_stats", ffa_stats.get_packed_data());
    h5_file().createAttribute("flops", ffa_stats.get_flops());
    h5_file().createAttribute("total_runtime", total_runtime);
}

// --- PruneResultWriter ---
PruneResultWriter::PruneResultWriter(std::filesystem::path filename,
                                     Mode mode,
                                     std::string group_path)
    : m_filepath(std::move(filename)),
      m_mode(mode),
      m_group_path(std::move(group_path)) {}

void write_rfi_config(HighFive::Group& rfi,
                      const algorithms::PruneRFIConfig& rfi_config) {
    const auto n_windows = rfi_config.pulsar_mask.size();
    std::vector<double> windows_flat(n_windows * 4);
    for (SizeType i = 0; i < n_windows; ++i) {
        const auto& w             = rfi_config.pulsar_mask[i];
        windows_flat[(i * 4) + 0] = w.f_lo;
        windows_flat[(i * 4) + 1] = w.f_hi;
        windows_flat[(i * 4) + 2] = w.a_lo;
        windows_flat[(i * 4) + 3] = w.a_hi;
    }
    auto mask_ds =
        rfi.createDataSet("pulsar_mask", HighFive::DataSpace({n_windows, 4UL}),
                          HighFive::create_datatype<double>());
    if (n_windows > 0) {
        mask_ds.write_raw(windows_flat.data(),
                          HighFive::create_datatype<double>());
    }
    create_1d_dataset<float>(rfi, "harvest_scheme", rfi_config.harvest_scheme);
    rfi.createAttribute("n_harmonics", rfi_config.n_harmonics);
    rfi.createAttribute("fractional_harmonics",
                        static_cast<uint8_t>(rfi_config.fractional_harmonics));
    rfi.createAttribute("harvest_mask_ntiles", rfi_config.harvest_mask_ntiles);
    rfi.createAttribute("max_harvests", rfi_config.max_harvests);
    rfi.createAttribute("harvest_store_folds",
                        static_cast<uint8_t>(rfi_config.harvest_store_folds));
    rfi.createAttribute("impulsive_veto",
                        static_cast<uint8_t>(rfi_config.impulsive_veto));
    rfi.createAttribute("impulsive_kappa", rfi_config.impulsive_kappa);
    rfi.createAttribute("impulsive_min_level", rfi_config.impulsive_min_level);
    rfi.createAttribute("impulsive_min_snr", rfi_config.impulsive_min_snr);
}

void PruneResultWriter::write_metadata(
    const std::vector<std::string>& param_names,
    SizeType nsegments,
    SizeType max_sugg,
    std::span<const float> threshold_scheme,
    const algorithms::PruneRFIConfig& rfi_config) {
    const std::scoped_lock lock(m_hdf5_mutex);

    HighFive::File file = open_file();
    if (file.exist("pruning_version")) {
        throw std::runtime_error("Metadata already exists in file. Use "
                                 "append mode or new file.");
    }
    file.createAttribute("pruning_version", "1.0.0-cpp");
    file.createAttribute("param_names", param_names);
    file.createAttribute("nsegments", nsegments);
    file.createAttribute("max_sugg", max_sugg);
    file.createDataSet("threshold_scheme", threshold_scheme);

    // RFI-control configuration
    auto rfi = file.createGroup("rfi_config");
    write_rfi_config(rfi, rfi_config);
}

template <SupportedFoldType FoldType>
void PruneResultWriter::write_run_harvest(
    std::string_view run_name,
    const HarvestBuffer<FoldType>& harvest,
    SizeType n_params,
    SizeType n_harvested_total) {
    const std::scoped_lock lock(m_hdf5_mutex);

    HighFive::File file              = open_file();
    const HighFive::Group runs_group = open_runs_group(file);
    if (!runs_group.exist(std::string(run_name))) {
        throw std::runtime_error(
            fmt::format("write_run_harvest: run {} does not exist; call "
                        "write_run_results first.",
                        run_name));
    }
    HighFive::Group run_group = runs_group.getGroup(std::string(run_name));
    if (run_group.exist("harvest")) {
        throw std::runtime_error(fmt::format(
            "write_run_harvest: harvest group already exists for run {}.",
            run_name));
    }
    HighFive::Group hg = run_group.createGroup("harvest");
    hg.createAttribute("n_harvested_total", n_harvested_total);

    constexpr SizeType kParamStride = 2U;
    const auto n                    = harvest.size();
    const auto leaves_stride        = (n_params + 2) * kParamStride;
    error_check::check_equal(harvest.get_leaves_stride(), leaves_stride,
                             "write_run_harvest: leaves stride mismatch");

    // --- param_sets ---
    HighFive::DataSetCreateProps param_props;
    if (n > 0) {
        param_props.add(HighFive::Chunking({
            static_cast<hsize_t>(std::min(1024UL, n)),
            n_params + 2,
            kParamStride,
        }));
        param_props.add(HighFive::Deflate(9));
    }
    auto param_ds = hg.createDataSet(
        "param_sets", HighFive::DataSpace({n, n_params + 2, kParamStride}),
        HighFive::create_datatype<double>(), param_props);
    if (n > 0) {
        param_ds.write_raw(harvest.get_leaves().data(),
                           HighFive::create_datatype<double>());
    }
    // --- scalar per-record datasets ---
    create_1d_dataset<float>(hg, "scores", harvest.get_scores());
    create_1d_dataset<SizeType>(hg, "levels", harvest.get_levels());
    create_1d_dataset<SizeType>(hg, "seg_idx", harvest.get_seg_idx());
    create_1d_dataset<double>(hg, "t_ref", harvest.get_t_ref());

    // --- folds (optional) ---
    if (harvest.stores_folds()) {
        const auto folds_stride = harvest.get_folds_stride();
        const auto nbins_fold   = folds_stride / 2;
        HighFive::DataSetCreateProps fold_props;
        if (n > 0) {
            fold_props.add(HighFive::Chunking(
                {static_cast<hsize_t>(std::min(256UL, n)), 2UL, nbins_fold}));
            fold_props.add(HighFive::Deflate(6));
        }
        auto folds_ds =
            hg.createDataSet("folds", HighFive::DataSpace({n, 2UL, nbins_fold}),
                             HighFive::create_datatype<FoldType>(), fold_props);
        if (n > 0) {
            folds_ds.write_raw(harvest.get_folds().data(),
                               HighFive::create_datatype<FoldType>());
        }
    }
}

template void PruneResultWriter::write_run_harvest<float>(
    std::string_view, const HarvestBuffer<float>&, SizeType, SizeType);
template void PruneResultWriter::write_run_harvest<ComplexType>(
    std::string_view, const HarvestBuffer<ComplexType>&, SizeType, SizeType);

void PruneResultWriter::write_runtime(float runtime) {
    const std::scoped_lock lock(m_hdf5_mutex);
    HighFive::File file = open_file();
    file.createAttribute("final_runtime", runtime);
}

void PruneResultWriter::write_run_results(
    std::string_view run_name,
    std::span<const SizeType> snail_scheme,
    memory::CircularView<double> leaves_view,
    memory::CircularView<float> scores_view,
    memory::CircularView<float> scores_ep_view,
    double total_pruning_gflops,
    SizeType n_leaves,
    SizeType n_params,
    const PruneStatsCollection& pstats,
    std::span<const std::uint32_t> widths,
    const detail::LeafReductionSpec* reduction) {
    // With a reduction, only the retained leaves are written: compacted
    // copies replace the views, and leaf_index maps rows back to leaves.
    std::optional<detail::RunLeafReduction> reduced;
    std::vector<double> kept_leaves;
    std::vector<float> kept_scores;
    std::vector<float> kept_scores_ep;
    std::vector<std::uint32_t> kept_widths;
    if (reduction != nullptr) {
        const auto stride = (n_params + 2) * 2;
        const auto n1     = leaves_view.first.size() / stride;
        const auto leaf   = [&](std::size_t i) -> std::span<const double> {
            return i < n1
                       ? leaves_view.first.subspan(i * stride, stride)
                       : leaves_view.second.subspan((i - n1) * stride, stride);
        };
        const auto pick = [](const memory::CircularView<float>& v,
                             std::size_t i) {
            return i < v.first.size() ? v.first[i]
                                      : v.second[i - v.first.size()];
        };
        error_check::check_equal(
            leaves_view.first.size() / stride +
                leaves_view.second.size() / stride,
            n_leaves, "write_run_results: circular view size mismatch");
        reduced = detail::reduce_run_leaves(
            *reduction, n_leaves, n_params, leaf,
            [&](std::size_t i) { return pick(scores_view, i); }, widths);
        const auto n = reduced->retained.size();
        kept_leaves.reserve(n * stride);
        kept_scores.reserve(n);
        kept_scores_ep.reserve(n);
        kept_widths.reserve(n);
        for (const auto i : reduced->retained) {
            const auto row = leaf(i);
            kept_leaves.insert(kept_leaves.end(), row.begin(), row.end());
            kept_scores.push_back(pick(scores_view, i));
            kept_scores_ep.push_back(pick(scores_ep_view, i));
            kept_widths.push_back(widths[i]);
        }
        leaves_view    = {std::span<double>(kept_leaves), {}};
        scores_view    = {std::span<float>(kept_scores), {}};
        scores_ep_view = {std::span<float>(kept_scores_ep), {}};
        widths         = kept_widths;
        n_leaves       = n;
    }
    const auto leaves_stride = (n_params + 2) * 2;
    const auto n1            = leaves_view.first.size() / leaves_stride;
    const auto n2            = leaves_view.second.size() / leaves_stride;
    error_check::check_equal(n1 + n2, n_leaves,
                             "write_run_results: circular view size mismatch");
    if (!widths.empty() || n_leaves == 0) {
        error_check::check_equal(widths.size(), n_leaves,
                                 "write_run_results: width count mismatch");
    }
    const auto write_rows = [&](const RunDatasets& ds) {
        const auto write_part = [&](std::span<const double> leaves,
                                    std::span<const float> scores,
                                    std::span<const float> scores_ep,
                                    SizeType offset) {
            const auto n = scores.size();
            if (n == 0) {
                return;
            }
            ds.param_sets
                .select({offset, 0UL, 0UL}, {n, n_params + 2, SizeType{2}})
                .write_raw(leaves.data(), HighFive::create_datatype<double>());
            ds.scores.select({offset}, {n})
                .write_raw(scores.data(), HighFive::create_datatype<float>());
            ds.scores_ep.select({offset}, {n})
                .write_raw(scores_ep.data(),
                           HighFive::create_datatype<float>());
        };
        write_part(leaves_view.first, scores_view.first, scores_ep_view.first,
                   0);
        write_part(leaves_view.second, scores_view.second,
                   scores_ep_view.second, n1);
        if (ds.width && n_leaves > 0) {
            ds.width->select({0UL}, {n_leaves})
                .write_raw(widths.data(),
                           HighFive::create_datatype<std::uint32_t>());
        }
    };
    write_run_group(run_name, snail_scheme, total_pruning_gflops, n_leaves,
                    n_params, pstats, !widths.empty() || n_leaves == 0,
                    reduced ? &*reduced : nullptr, reduction, write_rows);
}

void PruneResultWriter::write_run_results(
    std::string_view run_name,
    std::span<const SizeType> snail_scheme,
    detail::SliceScratch& scratch,
    double total_pruning_gflops,
    SizeType n_params,
    const PruneStatsCollection& pstats,
    const detail::LeafReductionSpec* reduction) {
    const auto stride = (n_params + 2) * 2;
    error_check::check_equal(scratch.leaves_stride(), stride,
                             "write_run_results: scratch leaf stride");
    scratch.seal();
    const auto n_total    = scratch.rows();
    const bool has_widths = scratch.has_widths();
    // Rows are read in batches of this many: the host holds one batch of
    // the scratch at a time.
    constexpr SizeType kBatch = SizeType{1} << 16U;
    std::vector<double> leaves(kBatch * stride);
    std::vector<float> scores(kBatch);
    std::vector<float> scores_ep(kBatch);
    std::vector<std::uint32_t> widths(has_widths ? kBatch : 0);
    // Calls fn(begin, n) after reading rows [begin, begin + n) into the
    // batch buffers.
    const auto for_each_batch = [&](const auto& fn) {
        for (SizeType begin = 0; begin < n_total; begin += kBatch) {
            const auto n = std::min(kBatch, n_total - begin);
            scratch.read(begin, n, std::span(leaves).first(n * stride),
                         std::span(scores).first(n),
                         std::span(scores_ep).first(n),
                         has_widths ? std::span(widths).first(n)
                                    : std::span<std::uint32_t>{});
            fn(begin, n);
        }
    };
    if (reduction != nullptr) {
        if (!has_widths && n_total > 0) {
            throw std::invalid_argument(
                "leaf reduction needs one width per leaf");
        }
        // Pass 1: the reduction, streamed. Pass 2: the retained rows, in
        // ascending leaf order, as the in-memory path writes them.
        detail::RunLeafReducer reducer(*reduction, n_params);
        for_each_batch([&](SizeType begin, SizeType n) {
            for (SizeType k = 0; k < n; ++k) {
                reducer.add(
                    begin + k,
                    std::span<const double>(leaves).subspan(k * stride, stride),
                    scores[k], widths[k]);
            }
        });
        const auto reduced = reducer.finish(n_total);
        const auto n_kept  = reduced.retained.size();
        std::vector<double> kept_leaves;
        std::vector<float> kept_scores;
        std::vector<float> kept_scores_ep;
        std::vector<std::uint32_t> kept_widths;
        kept_leaves.reserve(n_kept * stride);
        kept_scores.reserve(n_kept);
        kept_scores_ep.reserve(n_kept);
        kept_widths.reserve(n_kept);
        auto next = reduced.retained.begin();
        for_each_batch([&](SizeType begin, SizeType n) {
            for (; next != reduced.retained.end() && *next < begin + n;
                 ++next) {
                const auto k = *next - begin;
                kept_leaves.insert(
                    kept_leaves.end(),
                    leaves.begin() + static_cast<std::ptrdiff_t>(k * stride),
                    leaves.begin() +
                        static_cast<std::ptrdiff_t>((k + 1) * stride));
                kept_scores.push_back(scores[k]);
                kept_scores_ep.push_back(scores_ep[k]);
                kept_widths.push_back(widths[k]);
            }
        });
        const auto write_rows = [&](const RunDatasets& ds) {
            if (n_kept == 0) {
                return;
            }
            ds.param_sets
                .select({0UL, 0UL, 0UL}, {n_kept, n_params + 2, SizeType{2}})
                .write_raw(kept_leaves.data(),
                           HighFive::create_datatype<double>());
            ds.scores.select({0UL}, {n_kept})
                .write_raw(kept_scores.data(),
                           HighFive::create_datatype<float>());
            ds.scores_ep.select({0UL}, {n_kept})
                .write_raw(kept_scores_ep.data(),
                           HighFive::create_datatype<float>());
            ds.width->select({0UL}, {n_kept})
                .write_raw(kept_widths.data(),
                           HighFive::create_datatype<std::uint32_t>());
        };
        write_run_group(run_name, snail_scheme, total_pruning_gflops, n_kept,
                        n_params, pstats, true, &reduced, reduction,
                        write_rows);
        return;
    }
    // Every row, copied batch by batch.
    const auto write_rows = [&](const RunDatasets& ds) {
        for_each_batch([&](SizeType begin, SizeType n) {
            ds.param_sets
                .select({begin, 0UL, 0UL}, {n, n_params + 2, SizeType{2}})
                .write_raw(leaves.data(), HighFive::create_datatype<double>());
            ds.scores.select({begin}, {n})
                .write_raw(scores.data(), HighFive::create_datatype<float>());
            ds.scores_ep.select({begin}, {n})
                .write_raw(scores_ep.data(),
                           HighFive::create_datatype<float>());
            if (ds.width) {
                ds.width->select({begin}, {n})
                    .write_raw(widths.data(),
                               HighFive::create_datatype<std::uint32_t>());
            }
        });
    };
    write_run_group(run_name, snail_scheme, total_pruning_gflops, n_total,
                    n_params, pstats, has_widths || n_total == 0, nullptr,
                    nullptr, write_rows);
}

void PruneResultWriter::write_run_group(
    std::string_view run_name,
    std::span<const SizeType> snail_scheme,
    double total_pruning_gflops,
    SizeType n_leaves,
    SizeType n_params,
    const PruneStatsCollection& pstats,
    bool write_widths,
    const detail::RunLeafReduction* reduced,
    const detail::LeafReductionSpec* reduction,
    const std::function<void(const RunDatasets&)>& write_rows) {
    const std::scoped_lock lock(m_hdf5_mutex);

    HighFive::File file        = open_file();
    HighFive::Group runs_group = open_runs_group(file);
    if (runs_group.exist(std::string(run_name))) {
        throw std::runtime_error(
            fmt::format("Run name {} already exists.", run_name));
    }
    HighFive::Group run_group = runs_group.createGroup(std::string(run_name));

    run_group.createAttribute("total_pruning_gflops", total_pruning_gflops);

    auto [level_stats, timer_stats] = pstats.get_packed_data();

    constexpr SizeType kParamStride             = 2U;
    const std::vector<SizeType> param_sets_dims = {
        n_leaves,
        n_params + 2,
        kParamStride,
    };
    const HighFive::DataSpace param_sets_space(param_sets_dims);
    // Uncompressed: deflate halves these float64 rows at best and cost
    // seconds of CPU per run, while the raw write takes a fraction of that.
    RunDatasets ds{
        .param_sets =
            run_group.createDataSet("param_sets", param_sets_space,
                                    HighFive::create_datatype<double>()),
        .scores =
            run_group.createDataSet("scores", HighFive::DataSpace({n_leaves}),
                                    HighFive::create_datatype<float>()),
        .scores_ep = run_group.createDataSet(
            "scores_ep", HighFive::DataSpace({n_leaves}),
            HighFive::create_datatype<float>()),
        .width = std::nullopt,
    };

    run_group.createDataSet(
        "snail_scheme", snail_scheme,
        io::detail::small_table_props({snail_scheme.size()}));
    if (write_widths) {
        ds.width =
            run_group.createDataSet("width", HighFive::DataSpace({n_leaves}),
                                    HighFive::create_datatype<std::uint32_t>());
    }
    write_rows(ds);
    if (!level_stats.empty()) {
        run_group.createDataSet(
            "level_stats", level_stats,
            io::detail::small_table_props({level_stats.size()}));
    }
    if (!pstats.get_circular_counts().empty()) {
        const auto& circular = pstats.get_circular_counts();
        run_group.createDataSet(
            "level_stats_circular", circular,
            io::detail::small_table_props({circular.size()}));
    }
    if (!timer_stats.empty()) {
        run_group.createDataSet("timer_stats", timer_stats);
    }
    if (reduced != nullptr) {
        run_group.createDataSet("leaf_index", reduced->retained)
            .createAttribute("description",
                             std::string("leaf of each row among the run's "
                                         "final leaves"));
        run_group.createAttribute("n_final_leaves", reduced->final_leaves);
        run_group.createAttribute("n_leaves_above_snr_min",
                                  reduced->above_snr_min);
        run_group.createAttribute("n_cell_dropped", reduced->cell_dropped);
        run_group.createAttribute("n_dominated", reduced->dominated);
        run_group.createAttribute(
            "n_leaves_retained",
            static_cast<std::uint64_t>(reduced->retained.size()));
        run_group.createAttribute("max_cell_dropped_snr",
                                  reduced->max_cell_dropped_snr);
        run_group.createAttribute("max_per_cell_used",
                                  reduced->max_per_cell_used);
        // The options the search reduced with: publication must use them.
        const auto& o = reduction->options;
        run_group.createAttribute("reduced_max_per_cell", o.max_per_cell);
        run_group.createAttribute("reduced_cell_bins", o.cell_bins);
        run_group.createAttribute("reduced_cell_log_width", o.cell_log_width);
        run_group.createAttribute("reduced_dominance_kappa", o.dominance_kappa);
        run_group.createAttribute("reduced_radius", o.radius);
        run_group.createAttribute("reduced_phase_radius", o.phase_radius);
        run_group.createAttribute("reduced_snr_min", reduction->snr_min);
        run_group.createAttribute("reduced_at", std::string("search"));
    }
}

HighFive::File PruneResultWriter::open_file() const {
    HighFive::File::AccessMode open_mode;
    if (m_mode == Mode::kWrite) {
        open_mode = HighFive::File::Overwrite;
    } else if (std::filesystem::exists(m_filepath)) {
        open_mode = HighFive::File::ReadWrite;
    } else {
        open_mode = HighFive::File::Create;
    }

    HighFive::File file =
        io::detail::open_result_file(m_filepath.string(), open_mode);
    if (!file.isValid()) {
        throw std::runtime_error("Failed to create valid HDF5 file");
    }
    return file;
}

HighFive::Group PruneResultWriter::open_runs_group(HighFive::File& file) const {
    const auto path =
        m_group_path.empty() ? std::string("runs") : m_group_path + "/runs";
    return file.exist(path) ? file.getGroup(path) : file.createGroup(path);
}

void merge_prune_result_files(const std::filesystem::path& results_dir,
                              const std::filesystem::path& log_file,
                              const std::filesystem::path& result_file) {
    if (!std::filesystem::exists(results_dir)) {
        throw std::runtime_error(fmt::format(
            "Results directory does not exist: {}", results_dir.string()));
    }

    // --- Collect and sort log files ---
    std::vector<std::filesystem::directory_entry> temp_log_files;
    for (const auto& entry : std::filesystem::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.starts_with("tmp_") && filename.ends_with("_log.txt")) {
            temp_log_files.push_back(entry);
        }
    }
    std::ranges::sort(temp_log_files, [](const auto& a, const auto& b) {
        auto [ref_a, task_a] =
            extract_ref_seg_task_id(a.path().filename().string());
        auto [ref_b, task_b] =
            extract_ref_seg_task_id(b.path().filename().string());
        return std::tie(ref_a, task_a) < std::tie(ref_b, task_b);
    });

    // --- Merge log files in order ---
    std::ofstream main_log(log_file, std::ios::app);
    if (!main_log) {
        throw std::runtime_error(
            fmt::format("Cannot open log file: {}", log_file.string()));
    }
    for (const auto& entry : temp_log_files) {
        std::ifstream temp_log(entry.path());
        if (temp_log) {
            main_log << temp_log.rdbuf();
        }
        temp_log.close();
        std::filesystem::remove(entry.path());
    }
    main_log.close();

    // --- Collect and sort HDF5 files ---
    std::vector<std::filesystem::directory_entry> temp_h5_files;
    for (const auto& entry : std::filesystem::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.starts_with("tmp_") && filename.ends_with("_results.h5")) {
            temp_h5_files.push_back(entry);
        }
    }
    std::ranges::sort(temp_h5_files, [](const auto& a, const auto& b) {
        auto [ref_a, task_a] =
            extract_ref_seg_task_id(a.path().filename().string());
        auto [ref_b, task_b] =
            extract_ref_seg_task_id(b.path().filename().string());
        return std::tie(ref_a, task_a) < std::tie(ref_b, task_b);
    });

    // --- Merge HDF5 files in order ---
    const auto open_mode = std::filesystem::exists(result_file)
                               ? HighFive::File::ReadWrite
                               : HighFive::File::Create;
    HighFive::File main_h5 =
        io::detail::open_result_file(result_file.string(), open_mode);
    const HighFive::Group main_runs_group = main_h5.exist("runs")
                                                ? main_h5.getGroup("runs")
                                                : main_h5.createGroup("runs");
    for (const auto& entry : temp_h5_files) {
        const HighFive::File temp_h5(entry.path().string(),
                                     HighFive::File::ReadOnly);
        if (temp_h5.exist("runs")) {
            const HighFive::Group temp_runs_group = temp_h5.getGroup("runs");
            for (const auto& run_name : temp_runs_group.listObjectNames()) {
                if (main_runs_group.exist(run_name)) {
                    continue;
                }
                const herr_t status =
                    H5Ocopy(temp_runs_group.getId(), run_name.c_str(),
                            main_runs_group.getId(), run_name.c_str(),
                            H5P_DEFAULT, H5P_DEFAULT);
                if (status < 0) {
                    throw std::runtime_error(
                        fmt::format("Failed to copy run '{}' from {}", run_name,
                                    entry.path().string()));
                }
            }
        }
        std::filesystem::remove(entry.path());
    }
}

// TimerStats class implementation
TimerStats::TimerStats(SizeType num_threads) : m_thread_timers(num_threads) {}

TimerStats::TimerMap& TimerStats::get_thread_local() {
    const int tid = omp_get_thread_num();
    return m_thread_timers[static_cast<SizeType>(tid)];
}

TimerStats::TimerMap TimerStats::aggregate() const {
    TimerMap result;
    for (const auto& thread_map : m_thread_timers) {
        for (const auto& [name, time] : thread_map) {
            result[name] += time;
        }
    }
    return result;
}

void TimerStats::reset() {
    for (auto& thread_map : m_thread_timers) {
        thread_map.clear();
    }
}

void TimerStats::merge(const TimerStats& other) {
    const auto other_agg = other.aggregate();
    for (const auto& [name, time] : other_agg) {
        // Accumulate into first thread's map
        m_thread_timers[0][name] += time;
    }
}

std::string TimerStats::summary(float total_time) const {
    const auto agg = aggregate();

    if (agg.empty()) {
        return "No timing data collected.";
    }
    const auto n_threads = m_thread_timers.size();

    // Compute total from aggregated timers if not provided
    if (total_time <= 0.0) {
        total_time = std::accumulate(
            agg.begin(), agg.end(), 0.0F,
            [](float sum, const auto& p) { return sum + p.second; });
    }

    // Sort by time (descending)
    std::vector<std::pair<std::string, float>> sorted_times(agg.begin(),
                                                            agg.end());
    std::ranges::sort(sorted_times, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::string breakdown;
    int count = 0;
    for (const auto& [name, time] : sorted_times) {
        if (time > 0 && count < 4) {
            if (!breakdown.empty()) {
                breakdown += " | ";
            }
            breakdown += fmt::format(
                "{}: {:.0f}%", name,
                (time / (total_time * static_cast<float>(n_threads))) * 100.0F);
            count++;
        }
    }
    return fmt::format("Total: {:.1f}s ({})", total_time, breakdown);
}
} // namespace loki::search

HIGHFIVE_REGISTER_TYPE(loki::search::FFATimerStatsPacked,
                       loki::search::create_compound_ffa_timer_stats)
HIGHFIVE_REGISTER_TYPE(loki::search::PruneStats,
                       loki::search::create_compound_prune_stats)
HIGHFIVE_REGISTER_TYPE(loki::search::PruneTimerStatsPacked,
                       loki::search::create_compound_prune_timer_stats)
HIGHFIVE_REGISTER_TYPE(loki::search::CircularStageCounts,
                       loki::search::create_compound_circular_stage_counts)
