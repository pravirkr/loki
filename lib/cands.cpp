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

#include "loki/common/types.hpp"

namespace loki::cands {

namespace {
// Round-half-to-even (bankers' rounding)
double round_dp(double x, int digits) noexcept {
    if (!std::isfinite(x)) {
        return x;
    }
    const double scale = std::pow(10.0, digits);
    return std::nearbyint(x * scale) / scale;
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
    return {{"brutefold", HighFive::create_datatype<float>()},
            {"ffa", HighFive::create_datatype<float>()},
            {"score", HighFive::create_datatype<float>()},
            {"filter", HighFive::create_datatype<float>()}};
}

// Create a compound type for PruneStats
HighFive::CompoundType create_compound_prune_stats() {
    return {{{"level", HighFive::create_datatype<SizeType>()},
             {"seg_idx", HighFive::create_datatype<SizeType>()},
             {"threshold", HighFive::create_datatype<float>()},
             {"score_min", HighFive::create_datatype<float>()},
             {"score_max", HighFive::create_datatype<float>()},
             {"n_branches", HighFive::create_datatype<SizeType>()},
             {"n_leaves", HighFive::create_datatype<SizeType>()},
             {"n_leaves_phy", HighFive::create_datatype<SizeType>()},
             {"n_leaves_surv", HighFive::create_datatype<SizeType>()}}};
}

HighFive::CompoundType create_compound_prune_timer_stats() {
    return {{"branch", HighFive::create_datatype<float>()},
            {"validate", HighFive::create_datatype<float>()},
            {"resolve", HighFive::create_datatype<float>()},
            {"shift_add", HighFive::create_datatype<float>()},
            {"score", HighFive::create_datatype<float>()},
            {"transform", HighFive::create_datatype<float>()},
            {"threshold", HighFive::create_datatype<float>()}};
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
                                   m_accumulated_timers.at("filter"));
    }
    return packed_timers;
}

// --- PruneStats ---
double PruneStats::lb_leaves() const noexcept {
    return round_dp(std::log2(static_cast<double>(n_leaves_phy)), 2);
}
double PruneStats::lb_leaves_phys() const noexcept {
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
    return round_dp(static_cast<double>(n_leaves_surv) /
                        static_cast<double>(n_leaves_phy),
                    2);
}
std::string PruneStats::get_summary() const noexcept {
    return std::format("Prune level: {:3d}, seg_idx: {:3d}, leaves: {:5.2f}, "
                       "leaves_phys: {:5.2f}, branch_frac: {:5.2f},"
                       "score thresh: {:5.2f}, max: {:5.2f}, min: {:5.2f}, "
                       "P(surv): {:4.2f}\n",
                       level, seg_idx, lb_leaves(), lb_leaves_phys(),
                       branch_frac(), threshold, score_max, score_min,
                       surv_frac());
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
    return std::format("Score: {:.2f}, Leaves: {:.2f}", last_stats.score_max,
                       last_stats.lb_leaves());
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
                                   m_accumulated_timers.at("batch_add"));
    }
    return {m_stats_list, packed_timers};
}

// --- FFAResultWriter ---
FFAResultWriter::FFAResultWriter(std::filesystem::path filename, Mode mode)
    : m_filepath(std::move(filename)),
      m_mode(mode),
      m_datasets_initialized(false),
      m_file(open_file()) {}

void FFAResultWriter::write_metadata(
    const std::vector<std::string>& param_names,
    const std::vector<SizeType>& scoring_widths) {
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);

    if (m_file.exist("ffa_version")) {
        throw std::runtime_error("FFA metadata already exists in file. Use "
                                 "append mode or new file.");
    }
    m_file.createAttribute("ffa_version", "1.0.0-cpp");
    m_file.createAttribute("param_names", param_names);
    m_file.createAttribute("scoring_widths", scoring_widths);
}

void FFAResultWriter::write_results(std::span<const double> param_sets,
                                    std::span<const float> scores,
                                    SizeType n_param_sets,
                                    SizeType n_params) {
    if (n_param_sets == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);

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
    std::span<const float> threshold_scheme) {
    std::lock_guard<std::mutex> lock(m_hdf5_mutex);

    HighFive::File file = open_file();
    if (file.exist("pruning_version")) {
        throw std::runtime_error("Metadata already exists in file. Use "
                                 "append mode or new file.");
    }
    file.createAttribute("pruning_version", "1.0.0-cpp");
    file.createAttribute("param_names", param_names);
    file.createAttribute("nsegments", nsegments);
    file.createAttribute("max_sugg", max_sugg);

    HighFive::DataSetCreateProps props;
    props.add(
        HighFive::Chunking(std::vector<hsize_t>{threshold_scheme.size()}));
    props.add(HighFive::Deflate(9)); // Gzip compression level 9
    file.createDataSet("threshold_scheme", threshold_scheme, props);
}

void PruneResultWriter::write_run_results(
    std::string_view run_name,
    std::span<const SizeType> snail_scheme,
    std::span<const double> param_sets,
    std::span<const float> scores,
    SizeType n_param_sets,
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
    auto [level_stats, timer_stats] = pstats.get_packed_data();

    constexpr SizeType kParamStride = 2U;
    // Validate param_sets dimensions
    if (!param_sets.empty()) {
        const auto expected_size = n_param_sets * n_params * kParamStride;
        if (param_sets.size() != expected_size) {
            throw std::invalid_argument(std::format(
                "param_sets size does not match the expected dimension: {} != "
                "({} * {} * {})",
                param_sets.size(), n_param_sets, n_params, kParamStride));
        }
    }
    // Write only n_params rows
    const std::vector<SizeType> param_sets_dims = {n_param_sets, n_params,
                                                   kParamStride};
    HighFive::DataSpace param_sets_space(param_sets_dims);
    HighFive::DataSetCreateProps props;
    if (!param_sets.empty()) {
        const auto chunk_n_param_sets =
            static_cast<hsize_t>(std::min(1024UL, n_param_sets));
        const std::vector<hsize_t> chunk_dims = {chunk_n_param_sets, n_params,
                                                 kParamStride};
        props.add(HighFive::Chunking(chunk_dims));
        props.add(HighFive::Deflate(9));
    }
    auto param_sets_dataset =
        run_group.createDataSet("param_sets", param_sets_space,
                                HighFive::create_datatype<double>(), props);
    if (!param_sets.empty()) {
        // This allows writing flat data directly to multidimensional datasets
        param_sets_dataset.write_raw(param_sets.data(),
                                     HighFive::create_datatype<double>());
    }
    run_group.createDataSet("snail_scheme", snail_scheme);
    run_group.createDataSet("scores", scores);
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

} // namespace loki::cands

HIGHFIVE_REGISTER_TYPE(loki::cands::FFATimerStatsPacked,
                       loki::cands::create_compound_ffa_timer_stats)
HIGHFIVE_REGISTER_TYPE(loki::cands::PruneStats,
                       loki::cands::create_compound_prune_stats)
HIGHFIVE_REGISTER_TYPE(loki::cands::PruneTimerStatsPacked,
                       loki::cands::create_compound_prune_timer_stats)
