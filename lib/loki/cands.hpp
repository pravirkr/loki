#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "loki/common/types.hpp"

namespace loki::cands {

struct PruneStats {
    SizeType level{};
    SizeType seg_idx{};
    float threshold{};
    float score_min        = 0.0;
    float score_max        = 0.0;
    SizeType n_branches    = 1;
    SizeType n_leaves      = 1;
    SizeType n_leaves_phy  = 1;
    SizeType n_leaves_surv = 1;

    [[nodiscard]] float lb_leaves() const {
        return round_2dp(std::log2(static_cast<float>(n_leaves_phy)));
    }

    [[nodiscard]] float branch_frac() const {
        return round_2dp(static_cast<float>(n_leaves) /
                         static_cast<float>(n_branches));
    }

    [[nodiscard]] float phys_frac() const {
        return round_2dp(static_cast<float>(n_leaves_phy) /
                         static_cast<float>(n_leaves));
    }

    [[nodiscard]] float surv_frac() const {
        return round_2dp(static_cast<float>(n_leaves_surv) /
                         static_cast<float>(n_leaves_phy));
    }

    [[nodiscard]] std::string get_summary() const {
        return std::format("Prune level: {:3d}, seg_idx: {:3d}, lb_leaves: "
                           "{:5.2f}, branch_frac: {:5.2f},"
                           "score thresh: {:5.2f}, max: {:5.2f}, min: {:5.2f}, "
                           "P(surv): {:4.2f}\n",
                           level, seg_idx, lb_leaves(), branch_frac(),
                           threshold, score_max, score_min, surv_frac());
    }

private:
    static float round_2dp(float val) {
        return std::round(val * 100.0F) / 100.0F;
    }
};

struct TimerStatsPacked {
    float branch{};
    float validate{};
    float resolve{};
    float shift_add{};
    float score{};
    float transform{};
    float threshold{};
};

// Create a compound type for PruneStats
inline HighFive::CompoundType create_compound_prune_stats() {
    return {{"level", HighFive::create_datatype<SizeType>()},
            {"seg_idx", HighFive::create_datatype<SizeType>()},
            {"threshold", HighFive::create_datatype<float>()},
            {"score_min", HighFive::create_datatype<float>()},
            {"score_max", HighFive::create_datatype<float>()},
            {"n_branches", HighFive::create_datatype<SizeType>()},
            {"n_leaves", HighFive::create_datatype<SizeType>()},
            {"n_leaves_phy", HighFive::create_datatype<SizeType>()},
            {"n_leaves_surv", HighFive::create_datatype<SizeType>()}};
}

inline HighFive::CompoundType create_compound_timer_stats() {
    return {{"branch", HighFive::create_datatype<float>()},
            {"validate", HighFive::create_datatype<float>()},
            {"resolve", HighFive::create_datatype<float>()},
            {"shift_add", HighFive::create_datatype<float>()},
            {"score", HighFive::create_datatype<float>()},
            {"transform", HighFive::create_datatype<float>()},
            {"threshold", HighFive::create_datatype<float>()}};
}

class PruneStatsCollection {
public:
    static constexpr std::array kTimerNames = {
        "branch", "validate",  "resolve",  "shift_add",
        "score",  "transform", "threshold"};

    PruneStatsCollection() {
        for (const auto& name : kTimerNames) {
            m_timers[name] = 0.0;
        }
    }
    [[nodiscard]] size_t get_nstages() const { return m_stats_list.size(); }

    void update_stats(
        const PruneStats& stats,
        std::optional<std::span<const float>> timer_vals = std::nullopt) {
        m_stats_list.push_back(stats);
        if (timer_vals) {
            const auto& vals = *timer_vals;
            if (vals.size() != kTimerNames.size()) {
                throw std::invalid_argument(std::format(
                    "Invalid timer array length: expected {}, got {}",
                    kTimerNames.size(), vals.size()));
            }
            for (size_t i = 0; i < kTimerNames.size(); ++i) {
                m_timers[kTimerNames[i]] += vals[i];
            }
        }
    }

    [[nodiscard]] std::optional<PruneStats> get_stats(SizeType level) const {
        auto it = std::ranges::find_if(
            m_stats_list, [level](const auto& s) { return s.level == level; });
        return it != m_stats_list.end() ? std::optional{*it} : std::nullopt;
    }

    [[nodiscard]] std::string get_all_summaries() const {
        auto sorted_stats = m_stats_list;
        std::ranges::sort(sorted_stats, {}, &PruneStats::level);

        std::string result;
        for (const auto& stats : sorted_stats) {
            result += stats.get_summary();
        }
        return result;
    }

    [[nodiscard]] std::string get_stats_summary() const {
        if (m_stats_list.empty()) {
            return "No stats available.";
        }
        const auto& last_stats = m_stats_list.back();
        return std::format("Score: {:.2f}, Leaves: {:.2f}",
                           last_stats.score_max, last_stats.lb_leaves());
    }

    [[nodiscard]] std::string get_timer_summary() const {
        const float total_time = get_total_time();
        if (total_time == 0.0) {
            return "Timing breakdown: 0.00s\n";
        }
        std::string summary =
            std::format("Timing breakdown: {:.2f}s\n", total_time);
        for (const auto& name : kTimerNames) {
            if (auto it = m_timers.find(std::string{name});
                it != m_timers.end()) {
                const auto percent = (it->second / total_time) * 100.0F;
                summary += std::format("  {:10s}: {:6.1f}%\n", name, percent);
            }
        }
        return summary;
    }

    [[nodiscard]] std::string get_concise_timer_summary() const {
        const float total_time = get_total_time();
        if (total_time == 0.0) {
            return "Total: 0.0s";
        }

        // Copy timers to a vector to sort them by time
        std::vector<std::pair<std::string_view, float>> sorted_times;
        sorted_times.reserve(m_timers.size());
        for (const auto& [name, time] : m_timers) {
            sorted_times.emplace_back(name, time);
        }
        std::ranges::sort(sorted_times, [](const auto& a, const auto& b) {
            return a.second > b.second; // Sort descending
        });

        std::string breakdown;
        int count = 0;
        for (const auto& [name, time] : sorted_times) {
            if (time > 0 && count < 4) {
                if (!breakdown.empty()) {
                    breakdown += " | ";
                }
                breakdown += std::format("{}: {:.0f}%", name,
                                         (time / total_time) * 100.0F);
                count++;
            }
        }
        return std::format("Total: {:.1f}s ({})", total_time, breakdown);
    }

    [[nodiscard]] std::pair<std::vector<PruneStats>,
                            std::vector<TimerStatsPacked>>
    get_packed_data() const {
        std::vector<TimerStatsPacked> packed_timers;
        if (get_total_time() > 0.0F) {
            packed_timers.emplace_back(
                m_timers.at("branch"), m_timers.at("validate"),
                m_timers.at("resolve"), m_timers.at("shift_add"),
                m_timers.at("score"), m_timers.at("transform"),
                m_timers.at("threshold"));
        }
        return {m_stats_list, packed_timers};
    }

private:
    std::vector<PruneStats> m_stats_list;
    std::map<std::string, float> m_timers;

    [[nodiscard]] float get_total_time() const {
        return std::accumulate(
            m_timers.begin(), m_timers.end(), 0.0F,
            [](float sum, const auto& pair) { return sum + pair.second; });
    }
};

class PruneResultWriter {
public:
    enum class Mode : std::uint8_t { kWrite, kAppend };

    explicit PruneResultWriter(std::filesystem::path filename,
                               Mode mode = Mode::kWrite)
        : m_filepath(std::move(filename)),
          m_mode(mode) {

        auto open_mode = (m_mode == Mode::kWrite)
                             ? HighFive::File::Overwrite
                             : (std::filesystem::exists(m_filepath)
                                    ? HighFive::File::ReadWrite
                                    : HighFive::File::Create);

        m_file =
            std::make_unique<HighFive::File>(m_filepath.string(), open_mode);

        if (!m_file || !m_file->isValid()) {
            throw std::runtime_error("Failed to create valid HDF5 file");
        }
        if (!m_file->exist("runs")) {
            m_runs_group =
                std::make_unique<HighFive::Group>(m_file->createGroup("runs"));
        } else {
            m_runs_group =
                std::make_unique<HighFive::Group>(m_file->getGroup("runs"));
        }
    }
    ~PruneResultWriter()                                   = default;
    PruneResultWriter(const PruneResultWriter&)            = delete;
    PruneResultWriter& operator=(const PruneResultWriter&) = delete;
    PruneResultWriter(PruneResultWriter&&)                 = delete;
    PruneResultWriter& operator=(PruneResultWriter&&)      = delete;

    void write_metadata(const std::vector<std::string>& param_names,
                        SizeType nsegments,
                        SizeType max_sugg,
                        const std::vector<float>& threshold_scheme) {
        if (m_file->exist("pruning_version")) {
            throw std::runtime_error("Metadata already exists in file. Use "
                                     "append mode or new file.");
        }
        m_file->createAttribute("pruning_version", "1.0.0-cpp");
        m_file->createAttribute("param_names", param_names);
        m_file->createAttribute("nsegments", nsegments);
        m_file->createAttribute("max_sugg", max_sugg);

        HighFive::DataSetCreateProps props;
        props.add(
            HighFive::Chunking(std::vector<hsize_t>{threshold_scheme.size()}));
        props.add(HighFive::Deflate(9)); // Gzip compression level 9
        m_file->createDataSet("threshold_scheme", threshold_scheme, props);
    }

    void write_run_results(const std::string& run_name,
                           const std::vector<SizeType>& scheme,
                           const xt::xtensor<double, 3>& param_sets,
                           const std::vector<float>& scores,
                           const PruneStatsCollection& pstats) {
        if (m_runs_group->exist(run_name)) {
            throw std::runtime_error(
                std::format("Run name {} already exists.", run_name));
        }
        HighFive::Group run_group       = m_runs_group->createGroup(run_name);
        auto [level_stats, timer_stats] = pstats.get_packed_data();

        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{1024}));
        props.add(HighFive::Deflate(9));

        // Create std::vector from xtensor data with shape
        const auto& shape = param_sets.shape();
        std::vector<SizeType> tensor_shape(shape.begin(), shape.end());
        std::vector<double> tensor_data(param_sets.data(),
                                        param_sets.data() + param_sets.size());
        run_group.createDataSet("param_sets_data", tensor_data, props);

        run_group.createDataSet("scheme", scheme);
        run_group.createDataSet("param_sets_shape", tensor_shape);
        run_group.createDataSet("scores", scores);
        if (!level_stats.empty()) {
            run_group.createDataSet("level_stats", level_stats);
        }
        if (!timer_stats.empty()) {
            run_group.createDataSet("timer_stats", timer_stats);
        }
    }

private:
    std::filesystem::path m_filepath;
    Mode m_mode;
    std::unique_ptr<HighFive::File> m_file;
    std::unique_ptr<HighFive::Group> m_runs_group;
};

/**
 * @brief Merges temporary HDF5 and log files into final result files.
 *
 * This function merges temporary HDF5 files created during the multiprocessing
 * of pruning results into a final result file. It also merges log files into a
 * single log file. The temporary files are deleted after merging.
 */
inline void merge_prune_result_files(const std::filesystem::path& results_dir,
                                     const std::filesystem::path& log_file,
                                     const std::filesystem::path& result_file) {
    if (!std::filesystem::exists(results_dir)) {
        throw std::runtime_error(std::format(
            "Results directory does not exist: {}", results_dir.string()));
    }
    // Merge log files
    std::ofstream main_log(log_file, std::ios::app);
    if (!main_log) {
        throw std::runtime_error(
            std::format("Cannot open log file: {}", log_file.string()));
    }
    for (const auto& entry : std::filesystem::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.starts_with("tmp_") && filename.ends_with("_log.txt")) {
            std::ifstream temp_log(entry.path());
            if (temp_log) {
                main_log << temp_log.rdbuf();
            }
            temp_log.close();
            std::filesystem::remove(entry.path());
        }
    }
    main_log.close();

    // Merge HDF5 files
    auto open_mode = std::filesystem::exists(result_file)
                         ? HighFive::File::ReadWrite
                         : HighFive::File::Create;
    HighFive::File main_h5(result_file.string(), open_mode);
    HighFive::Group main_runs_group = main_h5.exist("runs")
                                          ? main_h5.getGroup("runs")
                                          : main_h5.createGroup("runs");
    for (const auto& entry : std::filesystem::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.starts_with("tmp_") && filename.ends_with("_results.h5")) {
            HighFive::File temp_h5(entry.path().string(),
                                   HighFive::File::ReadOnly);

            if (temp_h5.exist("runs")) {
                HighFive::Group temp_runs_group = temp_h5.getGroup("runs");
                for (const auto& run_name : temp_runs_group.listObjectNames()) {
                    if (main_runs_group.exist(run_name)) {
                        continue;
                    }
                    // Use HDF5 native copy
                    herr_t status =
                        H5Ocopy(temp_runs_group.getId(), run_name.c_str(),
                                main_runs_group.getId(), run_name.c_str(),
                                H5P_DEFAULT, H5P_DEFAULT);
                    if (status < 0) {
                        throw std::runtime_error(
                            std::format("Failed to copy run '{}' from {}",
                                        run_name, entry.path().string()));
                    }
                }
            }
            std::filesystem::remove(entry.path());
        }
    }
}
} // namespace loki::cands

HIGHFIVE_REGISTER_TYPE(loki::cands::PruneStats,
                       loki::cands::create_compound_prune_stats)
HIGHFIVE_REGISTER_TYPE(loki::cands::TimerStatsPacked,
                       loki::cands::create_compound_timer_stats)
