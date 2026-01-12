#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"

namespace loki::cands {

struct FFATimerStatsPacked {
    float brutefold{};
    float ffa{};
    float score{};
    float filter{};
};

class FFATimerStats {
public:
    FFATimerStats();
    [[nodiscard]] float& operator[](const std::string& key);
    [[nodiscard]] const float& operator[](const std::string& key) const;
    [[nodiscard]] const float& at(const std::string& key) const;
    [[nodiscard]] float& at(const std::string& key);
    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] auto begin() const;
    [[nodiscard]] auto end() const;
    [[nodiscard]] auto begin();
    [[nodiscard]] auto end();
    [[nodiscard]] float total() const;
    void reset();
    // Accumulation operator
    FFATimerStats& operator+=(const FFATimerStats& other);
    [[nodiscard]] std::string get_concise_timer_summary() const;

private:
    static constexpr std::array kTimerNames = {"brutefold", "ffa", "score",
                                               "filter"};
    std::map<std::string, float> m_timers;
};

class FFAStatsCollection {
public:
    FFAStatsCollection() = default;

    void update_stats(const FFATimerStats& timers, float flops);
    // Direct access to accumulated timers
    [[nodiscard]] const FFATimerStats& get_timers() const {
        return m_accumulated_timers;
    }
    [[nodiscard]] float get_flops() const {
        return m_accumulated_flops;
    }
    [[nodiscard]] std::string get_concise_timer_summary() const;
    [[nodiscard]] std::vector<FFATimerStatsPacked> get_packed_data() const;

private:
    FFATimerStats m_accumulated_timers;
    float m_accumulated_flops{0.0F};
};

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

    [[nodiscard]] double lb_leaves() const noexcept;
    [[nodiscard]] double lb_leaves_phys() const noexcept;
    [[nodiscard]] double branch_frac() const noexcept;
    [[nodiscard]] double phys_frac() const noexcept;
    [[nodiscard]] double surv_frac() const noexcept;
    [[nodiscard]] std::string get_summary() const noexcept;
};

struct PruneTimerStatsPacked {
    float branch{};
    float validate{};
    float resolve{};
    float shift_add{};
    float score{};
    float transform{};
    float threshold{};
};

class PruneTimerStats {
public:
    PruneTimerStats();
    [[nodiscard]] float& operator[](const std::string& key);
    [[nodiscard]] const float& operator[](const std::string& key) const;
    [[nodiscard]] const float& at(const std::string& key) const;
    [[nodiscard]] float& at(const std::string& key);
    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] auto begin() const;
    [[nodiscard]] auto end() const;
    [[nodiscard]] auto begin();
    [[nodiscard]] auto end();
    [[nodiscard]] float total() const;
    void reset();
    // Accumulation operator
    PruneTimerStats& operator+=(const PruneTimerStats& other);

private:
    static constexpr std::array kTimerNames = {
        "branch", "validate",  "resolve",   "shift_add",
        "score",  "threshold", "transform", "batch_add"};

    std::map<std::string, float> m_timers;
};

class PruneStatsCollection {
public:
    PruneStatsCollection() = default;

    void update_stats(const PruneStats& stats, const PruneTimerStats& timers);
    void update_stats(const PruneStats& stats);
    // Direct access to accumulated timers
    [[nodiscard]] const PruneTimerStats& get_timers() const {
        return m_accumulated_timers;
    }
    [[nodiscard]] SizeType get_nstages() const;
    [[nodiscard]] std::optional<PruneStats> get_stats(SizeType level) const;
    [[nodiscard]] std::string get_all_summaries() const;
    [[nodiscard]] std::string get_stats_summary() const;
    [[nodiscard]] std::string get_timer_summary() const;
    [[nodiscard]] std::string get_concise_timer_summary() const;
    [[nodiscard]] std::pair<std::vector<PruneStats>,
                            std::vector<PruneTimerStatsPacked>>
    get_packed_data() const;

private:
    std::vector<PruneStats> m_stats_list;
    PruneTimerStats m_accumulated_timers;
};

class FFAResultWriter {
public:
    enum class Mode : std::uint8_t { kWrite, kAppend };
    /**
     * @brief Construct a new FFAResultWriter object.
     *
     * @param filename Path to the HDF5 output file.
     * @param mode     kWrite will truncate the file if it exists.
     * kAppend will open an existing file.
     */
    explicit FFAResultWriter(std::filesystem::path filename,
                             Mode mode = Mode::kWrite);
    ~FFAResultWriter() = default;
    // Disable copy/move constructors and operators
    FFAResultWriter(const FFAResultWriter&)            = delete;
    FFAResultWriter& operator=(const FFAResultWriter&) = delete;
    FFAResultWriter(FFAResultWriter&&)                 = delete;
    FFAResultWriter& operator=(FFAResultWriter&&)      = delete;

    void write_metadata(const std::vector<std::string>& param_names,
                        const std::vector<SizeType>& scoring_widths);

    void write_results(std::span<const double> param_sets,
                       std::span<const float> scores,
                       SizeType n_param_sets,
                       SizeType n_params);
    void write_ffa_stats(const FFAStatsCollection& ffa_stats);

private:
    std::filesystem::path m_filepath;
    Mode m_mode;
    inline static std::mutex m_hdf5_mutex;
    bool m_datasets_initialized;

    HighFive::File m_file;
    HighFive::File open_file() const;
};

class PruneResultWriter {
public:
    enum class Mode : std::uint8_t { kWrite, kAppend };

    explicit PruneResultWriter(std::filesystem::path filename,
                               Mode mode = Mode::kWrite);
    ~PruneResultWriter()                                   = default;
    PruneResultWriter(const PruneResultWriter&)            = delete;
    PruneResultWriter& operator=(const PruneResultWriter&) = delete;
    PruneResultWriter(PruneResultWriter&&)                 = delete;
    PruneResultWriter& operator=(PruneResultWriter&&)      = delete;

    void write_metadata(const std::vector<std::string>& param_names,
                        SizeType nsegments,
                        SizeType max_sugg,
                        const std::vector<float>& threshold_scheme);

    void write_run_results(const std::string& run_name,
                           const std::vector<SizeType>& scheme,
                           const std::vector<double>& param_sets,
                           const std::vector<float>& scores,
                           SizeType n_param_sets,
                           SizeType n_params,
                           const PruneStatsCollection& pstats);

private:
    std::filesystem::path m_filepath;
    Mode m_mode;
    inline static std::mutex m_hdf5_mutex;

    HighFive::File open_file() const;
    static HighFive::Group open_runs_group(HighFive::File& file);
};

/**
 * @brief Merges temporary HDF5 and log files into final result files.
 *
 * This function merges temporary HDF5 files created during the multiprocessing
 * of pruning results into a final result file. It also merges log files into a
 * single log file. The temporary files are deleted after merging. Merging order
 * is based on ref_seg.
 */
void merge_prune_result_files(const std::filesystem::path& results_dir,
                              const std::filesystem::path& log_file,
                              const std::filesystem::path& result_file);
} // namespace loki::cands
