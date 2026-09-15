#include "loki/cands.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <numeric>
#include <regex>
#include <tuple>

#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <highfive/span.hpp>
#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/utils.hpp"
#include "loki/utils/world_tree.hpp"

namespace loki::cands {

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
    std::regex re(R"(tmp_(\d{3})_(\d{2})_.*\.(?:txt|h5))");
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
                std::format("{}: {:.0f}%", name, (time / total_time) * 100.0F);
            count++;
        }
    }
    return std::format("Total: {:.1f}s ({})", total_time, breakdown);
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
                std::format("{}: {:.0f}%", name, (time / total_time) * 100.0F);
            count++;
        }
    }
    return std::format("Total: {:.1f}s ({})", total_time, breakdown);
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
    return round_dp(
        static_cast<double>(n_leaves) / static_cast<double>(n_branches), 2);
}
double PruneStats::phys_frac() const noexcept {
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
std::string PruneStats::get_summary() const noexcept {
    std::string summary = std::format(
        "Prune level: {:3d}, seg_idx: {:3d}, leaves: {:5.2f}, "
        "leaves_phys: {:5.2f}, branch_frac: {:5.2f},"
        "score thresh: {:5.2f}, max: {:5.2f}, min: {:5.2f}, "
        "P(surv): {:4.2f}",
        level, seg_idx, lb_leaves(), lb_leaves_phys(), branch_frac(), threshold,
        score_max, score_min, surv_frac());
    if (n_leaves_masked > 0 || n_leaves_vetoed > 0 || n_harvested > 0) {
        summary += std::format(
            ", masked: {}, vetoed: {}, harvested: {}",
            n_leaves_masked, n_leaves_vetoed, n_harvested);
    }
    summary += "\n";
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
std::optional<PruneStats>
PruneStatsCollection::get_stats(SizeType level) const {
    auto it = std::ranges::find_if(
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
    auto summary = std::format("Score: {:.2f}, Leaves: {:.2f}",
                               last_stats.score_max, last_stats.lb_leaves());
    if (n_masked > 0 || n_vetoed > 0 || n_harvested > 0) {
        summary += std::format(", masked: {}, vetoed: {}, harvested: {}",
                               n_masked, n_vetoed, n_harvested);
    }
    return summary;
}
std::string PruneStatsCollection::get_stats_summary_cuda(float duration) const {
    if (m_stats_list.empty()) {
        return "No stats available. Duration: 0.0s";
    }
    const auto& last_stats = m_stats_list.back();
    return std::format("Score: {:.2f}, Leaves: {:.2f}, Total: {:.2f}s",
                       last_stats.score_max, last_stats.lb_leaves(), duration);
}
std::string PruneStatsCollection::get_timer_summary() const {
    const float total_time = m_accumulated_timers.total();
    if (total_time == 0.0F) {
        return "Timing breakdown: 0.00s\n";
    }
    std::string summary =
        std::format("Timing breakdown: {:.2f}s\n", total_time);
    std::vector<std::pair<std::string, float>> sorted_timers(
        m_accumulated_timers.begin(), m_accumulated_timers.end());
    std::ranges::sort(sorted_timers, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    for (const auto& [name, time] : sorted_timers) {
        const auto percent = (time / total_time) * 100.0F;
        summary += std::format("  {:10s}: {:6.1f}%\n", name, percent);
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
                std::format("{}: {:.0f}%", name, (time / total_time) * 100.0F);
            count++;
        }
    }
    return std::format("Total: {:.1f}s ({})", total_time, breakdown);
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

// --- FFAResultWriter ---
FFAResultWriter::FFAResultWriter(std::filesystem::path filename, Mode mode)
    : m_filepath(std::move(filename)),
      m_mode(mode),
      m_datasets_initialized(false),
      m_file(open_file()) {}

void FFAResultWriter::write_metadata(
    const std::vector<std::string>& param_names,
    SizeType nbins,
    double ducy_max,
    double wtsp) {
    std::scoped_lock lock(m_hdf5_mutex);

    if (m_file.exist("ffa_version")) {
        throw std::runtime_error("FFA metadata already exists in file. Use "
                                 "append mode or new file.");
    }
    m_file.createAttribute("ffa_version", "1.0.0-cpp");
    m_file.createAttribute("param_names", param_names);
    m_file.createAttribute("nbins", nbins);
    m_file.createAttribute("ducy_max", ducy_max);
    m_file.createAttribute("wtsp", wtsp);
}

void FFAResultWriter::write_results(std::span<const double> param_sets,
                                    std::span<const float> scores,
                                    SizeType n_param_sets,
                                    SizeType n_params) {
    if (n_param_sets == 0) {
        return;
    }
    std::scoped_lock lock(m_hdf5_mutex);

    // Validate param_sets dimensions
    if (!param_sets.empty()) {
        const auto expected_size = n_param_sets * n_params;
        if (param_sets.size() != expected_size) {
            throw std::invalid_argument(std::format(
                "param_sets size does not match the expected dimension: {} != "
                "({} * {})",
                param_sets.size(), n_param_sets, n_params));
        }
    }

    if (!m_datasets_initialized) {
        // Initialize datasets for the first time
        HighFive::DataSpace snr_space({n_param_sets},
                                      {HighFive::DataSpace::UNLIMITED});
        HighFive::DataSetCreateProps snr_props;
        snr_props.add(HighFive::Chunking({std::min(1024UL, n_param_sets)}));
        snr_props.add(HighFive::Deflate(9));
        auto snr_dset = m_file.createDataSet(
            "snr", snr_space, HighFive::create_datatype<float>(), snr_props);
        snr_dset.write_raw(scores.data(), HighFive::create_datatype<float>());

        HighFive::DataSpace param_sets_space(
            {n_param_sets, n_params},
            {HighFive::DataSpace::UNLIMITED, n_params});
        HighFive::DataSetCreateProps param_props;
        if (!param_sets.empty()) {
            param_props.add(
                HighFive::Chunking({std::min(1024UL, n_param_sets), n_params}));
            param_props.add(HighFive::Deflate(9));
        }
        auto param_sets_dset = m_file.createDataSet(
            "param_sets", param_sets_space, HighFive::create_datatype<double>(),
            param_props);
        if (!param_sets.empty()) {
            // This allows writing flat data directly to multidimensional
            // datasets
            param_sets_dset.write_raw(param_sets.data(),
                                      HighFive::create_datatype<double>());
        }
        m_datasets_initialized = true;
    } else {
        // Append to existing datasets
        auto snr_dset           = m_file.getDataSet("snr");
        auto old_snr_dims       = snr_dset.getSpace().getDimensions();
        size_t old_n_param_sets = old_snr_dims[0];
        snr_dset.resize({old_n_param_sets + n_param_sets});
        snr_dset.select({old_n_param_sets}, {n_param_sets})
            .write_raw(scores.data(), HighFive::create_datatype<float>());
        auto param_sets_dset = m_file.getDataSet("param_sets");
        param_sets_dset.resize({old_n_param_sets + n_param_sets, n_params});
        param_sets_dset.select({old_n_param_sets, 0}, {n_param_sets, n_params})
            .write_raw(param_sets.data(), HighFive::create_datatype<double>());
    }
}

void FFAResultWriter::write_ffa_stats(const FFAStatsCollection& ffa_stats) {
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);
    m_file.createDataSet("timer_stats", ffa_stats.get_packed_data());
    m_file.createAttribute("flops", ffa_stats.get_flops());
}

HighFive::File FFAResultWriter::open_file() const {
    HighFive::File::AccessMode open_mode;
    if (m_mode == Mode::kWrite) {
        open_mode = HighFive::File::Overwrite;
    } else if (std::filesystem::exists(m_filepath)) {
        open_mode = HighFive::File::ReadWrite;
    } else {
        open_mode = HighFive::File::Create;
    }

    HighFive::File file(m_filepath.string(), open_mode);
    if (!file.isValid()) {
        throw std::runtime_error("Failed to create valid HDF5 file");
    }
    return file;
}

// --- PruneResultWriter ---
PruneResultWriter::PruneResultWriter(std::filesystem::path filename, Mode mode)
    : m_filepath(std::move(filename)),
      m_mode(mode) {}

void PruneResultWriter::write_metadata(
    const std::vector<std::string>& param_names,
    SizeType nsegments,
    SizeType max_sugg,
    std::span<const float> threshold_scheme,
    const algorithms::PruneRFIConfig& rfi_config) {
    std::scoped_lock lock(m_hdf5_mutex);

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
    HighFive::Group rfi  = file.createGroup("rfi_config");
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

template <SupportedFoldType FoldType>
void PruneResultWriter::write_run_harvest(
    std::string_view run_name,
    const HarvestBuffer<FoldType>& harvest,
    SizeType n_params,
    SizeType n_harvested_total) {
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);

    HighFive::File file        = open_file();
    HighFive::Group runs_group = open_runs_group(file);
    if (!runs_group.exist(std::string(run_name))) {
        throw std::runtime_error(
            std::format("write_run_harvest: run {} does not exist; call "
                        "write_run_results first.",
                        run_name));
    }
    HighFive::Group run_group = runs_group.getGroup(std::string(run_name));
    if (run_group.exist("harvest")) {
        throw std::runtime_error(std::format(
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
        param_props.add(
            HighFive::Chunking({static_cast<hsize_t>(std::min(1024UL, n)),
                                n_params + 2, kParamStride}));
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
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);
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
    const PruneStatsCollection& pstats) {
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);

    HighFive::File file        = open_file();
    HighFive::Group runs_group = open_runs_group(file);
    if (runs_group.exist(std::string(run_name))) {
        throw std::runtime_error(
            std::format("Run name {} already exists.", run_name));
    }
    HighFive::Group run_group = runs_group.createGroup(std::string(run_name));

    run_group.createAttribute("total_pruning_gflops", total_pruning_gflops);

    auto [level_stats, timer_stats] = pstats.get_packed_data();

    constexpr SizeType kParamStride             = 2U;
    const auto leaves_stride                    = (n_params + 2) * kParamStride;
    const std::vector<SizeType> param_sets_dims = {
        n_leaves,
        n_params + 2,
        kParamStride,
    };
    HighFive::DataSpace param_sets_space(param_sets_dims);
    HighFive::DataSetCreateProps props;
    if (n_leaves > 0) {
        const auto chunk_n_param_sets =
            static_cast<hsize_t>(std::min(1024UL, n_leaves));
        const std::vector<hsize_t> chunk_dims = {
            chunk_n_param_sets,
            n_params + 2,
            kParamStride,
        };
        props.add(HighFive::Chunking(chunk_dims));
        props.add(HighFive::Deflate(9));
    }
    auto param_ds =
        run_group.createDataSet("param_sets", param_sets_space,
                                HighFive::create_datatype<double>(), props);
    const auto n1 = leaves_view.first.size() / leaves_stride;
    const auto n2 = leaves_view.second.size() / leaves_stride;
    error_check::check_equal(n1 + n2, n_leaves,
                             "write_run_results: circular view size mismatch");

    // Proper 3D hyperslab writes:
    if (n1 > 0) {
        param_ds.select({0UL, 0UL, 0UL}, {n1, n_params + 2, kParamStride})
            .write_raw(leaves_view.first.data(),
                       HighFive::create_datatype<double>());
    }
    if (n2 > 0) {
        param_ds.select({n1, 0UL, 0UL}, {n2, n_params + 2, kParamStride})
            .write_raw(leaves_view.second.data(),
                       HighFive::create_datatype<double>());
    }

    // --- scores dataset ---
    auto scores_ds =
        run_group.createDataSet("scores", HighFive::DataSpace({n_leaves}),
                                HighFive::create_datatype<float>());

    auto write_chunk_score = [&](std::span<const float> chunk, size_t offset) {
        if (!chunk.empty()) {
            scores_ds.select({offset}, {chunk.size()})
                .write_raw(chunk.data(), HighFive::create_datatype<float>());
        }
    };

    auto scores_ep_ds =
        run_group.createDataSet("scores_ep", HighFive::DataSpace({n_leaves}),
                                HighFive::create_datatype<float>());

    auto write_chunk_scores_ep = [&](std::span<const float> chunk,
                                     size_t offset) {
        if (!chunk.empty()) {
            scores_ep_ds.select({offset}, {chunk.size()})
                .write_raw(chunk.data(), HighFive::create_datatype<float>());
        }
    };

    SizeType offset_s = 0;
    write_chunk_score(scores_view.first, offset_s);
    offset_s += scores_view.first.size();
    write_chunk_score(scores_view.second, offset_s);
    offset_s = 0;
    write_chunk_scores_ep(scores_ep_view.first, offset_s);
    offset_s += scores_ep_view.first.size();
    write_chunk_scores_ep(scores_ep_view.second, offset_s);

    run_group.createDataSet("snail_scheme", snail_scheme);
    if (!level_stats.empty()) {
        run_group.createDataSet("level_stats", level_stats);
    }
    if (!timer_stats.empty()) {
        run_group.createDataSet("timer_stats", timer_stats);
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

    HighFive::File file(m_filepath.string(), open_mode);
    if (!file.isValid()) {
        throw std::runtime_error("Failed to create valid HDF5 file");
    }
    return file;
}

HighFive::Group PruneResultWriter::open_runs_group(HighFive::File& file) {
    return file.exist("runs") ? file.getGroup("runs")
                              : file.createGroup("runs");
}

void merge_prune_result_files(const std::filesystem::path& results_dir,
                              const std::filesystem::path& log_file,
                              const std::filesystem::path& result_file) {
    if (!std::filesystem::exists(results_dir)) {
        throw std::runtime_error(std::format(
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
            std::format("Cannot open log file: {}", log_file.string()));
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
    auto open_mode = std::filesystem::exists(result_file)
                         ? HighFive::File::ReadWrite
                         : HighFive::File::Create;
    HighFive::File main_h5(result_file.string(), open_mode);
    HighFive::Group main_runs_group = main_h5.exist("runs")
                                          ? main_h5.getGroup("runs")
                                          : main_h5.createGroup("runs");
    for (const auto& entry : temp_h5_files) {
        HighFive::File temp_h5(entry.path().string(), HighFive::File::ReadOnly);
        if (temp_h5.exist("runs")) {
            HighFive::Group temp_runs_group = temp_h5.getGroup("runs");
            for (const auto& run_name : temp_runs_group.listObjectNames()) {
                if (main_runs_group.exist(run_name)) {
                    continue;
                }
                herr_t status =
                    H5Ocopy(temp_runs_group.getId(), run_name.c_str(),
                            main_runs_group.getId(), run_name.c_str(),
                            H5P_DEFAULT, H5P_DEFAULT);
                if (status < 0) {
                    throw std::runtime_error(
                        std::format("Failed to copy run '{}' from {}", run_name,
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
            breakdown += std::format(
                "{}: {:.0f}%", name,
                (time / (total_time * static_cast<float>(n_threads))) * 100.0F);
            count++;
        }
    }
    return std::format("Total: {:.1f}s ({})", total_time, breakdown);
}
} // namespace loki::cands

HIGHFIVE_REGISTER_TYPE(loki::cands::FFATimerStatsPacked,
                       loki::cands::create_compound_ffa_timer_stats)
HIGHFIVE_REGISTER_TYPE(loki::cands::PruneStats,
                       loki::cands::create_compound_prune_stats)
HIGHFIVE_REGISTER_TYPE(loki::cands::PruneTimerStatsPacked,
                       loki::cands::create_compound_prune_timer_stats)
