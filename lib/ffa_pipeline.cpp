#include "loki/pipelines/ffa_pipeline.hpp"

#include <memory>
#include <utility>

#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/plans.hpp"
#include "loki/cands.hpp"
#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

class FFAManager::BaseImpl {
public:
    BaseImpl()                           = default;
    virtual ~BaseImpl()                  = default;
    BaseImpl(const BaseImpl&)            = delete;
    BaseImpl& operator=(const BaseImpl&) = delete;
    BaseImpl(BaseImpl&&)                 = delete;
    BaseImpl& operator=(BaseImpl&&)      = delete;

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix) = 0;
};

template <typename FoldType>
class FFAManagerTypedImpl final : public FFAManager::BaseImpl {
public:
    FFAManagerTypedImpl(search::PulsarSearchConfig cfg, bool show_progress)
        : m_base_cfg(std::move(cfg)),
          m_region_planner(m_base_cfg),
          m_show_progress(show_progress) {
        const auto& planner_stats = m_region_planner.get_stats();
        // Allocate buffers once, sized for the largest chunk
        m_ffa_workspace = FFAWorkspace<FoldType>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(), m_base_cfg.get_nparams());
        m_fold.resize(planner_stats.get_max_buffer_size_time());
        m_scores.resize(planner_stats.get_max_scores_size());
        m_ffa_param_sets.resize(planner_stats.get_max_param_sets_size());

        // Log the actual memory usage for the allocated buffers
        spdlog::info("FFAManager allocated {:.2f} GB ({:.2f} GB buffers "
                     "+ {:.2f} GB coords + {:.2f} GB extra)",
                     planner_stats.get_manager_memory_usage(),
                     planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_extra_memory_usage());
        spdlog::info("FFAManager will process {} chunks",
                     m_region_planner.get_nregions());
    }

    ~FFAManagerTypedImpl() final                               = default;
    FFAManagerTypedImpl(const FFAManagerTypedImpl&)            = delete;
    FFAManagerTypedImpl& operator=(const FFAManagerTypedImpl&) = delete;
    FFAManagerTypedImpl(FFAManagerTypedImpl&&)                 = delete;
    FFAManagerTypedImpl& operator=(FFAManagerTypedImpl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        // timing::ScopeTimer timer("FFAManager::execute");
        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = cands::FFAResultWriter(
            result_file, cands::FFAResultWriter::Mode::kWrite);
        auto param_names = m_base_cfg.get_param_names();
        param_names.emplace_back("width");
        writer.write_metadata(param_names, m_base_cfg.get_scoring_widths());
        for (const auto& cfg_cur : m_region_planner.get_cfgs()) {
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.front(), freq_limits.back());
            execute_ffa_region(ts_e, ts_v, cfg_cur, writer);
        }
        spdlog::info("FFA Pipeline completed.");
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    plans::FFARegionPlanner<FoldType> m_region_planner;
    bool m_show_progress;

    algorithms::FFAWorkspace<FoldType> m_ffa_workspace;
    std::vector<float> m_fold;
    std::vector<float> m_scores;
    std::vector<double> m_ffa_param_sets; // includes width

    void execute_ffa_region(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            const search::PulsarSearchConfig& cfg,
                            cands::FFAResultWriter& result_writer) {
        // Create FFA with shared workspace
        auto the_ffa = FFA<FoldType>(cfg, m_ffa_workspace, m_show_progress);
        const plans::FFAPlan<FoldType>& ffa_plan = the_ffa.get_plan();
        const auto buffer_size_time = ffa_plan.get_buffer_size_time();
        const auto fold_size_time   = ffa_plan.get_fold_size_time();
        the_ffa.execute(ts_e, ts_v, std::span(m_fold).first(buffer_size_time));

        // Compute scores
        const auto nsegments       = ffa_plan.get_nsegments().back();
        const auto ncoords         = ffa_plan.get_ncoords().back();
        const auto n_params        = m_base_cfg.get_nparams();
        const auto& scoring_widths = m_base_cfg.get_scoring_widths();
        const SizeType n_widths    = scoring_widths.size();
        const auto n_scores        = ncoords * n_widths;
        error_check::check_equal(nsegments, 1U,
                                 "FFAManager::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        auto scores_span = std::span(m_scores).first(n_scores);
        detection::snr_boxcar_3d(std::span(m_fold).first(fold_size_time),
                                 ncoords, scoring_widths, scores_span,
                                 m_base_cfg.get_nthreads());

        // Filter scores and param sets
        const SizeType total_params = n_params + 1;
        auto param_sets_span =
            std::span(m_ffa_param_sets).first(n_scores * total_params);
        const auto& param_arr = ffa_plan.get_params().back();
        const auto snr_min    = m_base_cfg.get_snr_min();
        // Write results
        SizeType p_set_idx    = 0;
        SizeType filtered_idx = 0;
        for (const auto& p_set_view :
             utils::cartesian_product_view(param_arr)) {
            for (SizeType w_idx = 0; w_idx < n_widths; ++w_idx) {
                const SizeType score_idx = (p_set_idx * n_widths) + w_idx;
                const float score        = scores_span[score_idx];
                if (score >= snr_min) {
                    for (SizeType j = 0; j < n_params; ++j) {
                        param_sets_span[(filtered_idx * total_params) + j] =
                            p_set_view[j];
                    }
                    param_sets_span[(filtered_idx * total_params) + n_params] =
                        static_cast<double>(scoring_widths[w_idx]);
                    scores_span[filtered_idx] = score;
                    filtered_idx++;
                }
            }
            ++p_set_idx;
        }
        auto scores_span_filtered = scores_span.first(filtered_idx);
        auto param_sets_span_filtered =
            param_sets_span.first(filtered_idx * total_params);
        result_writer.write_results(param_sets_span_filtered,
                                    scores_span_filtered, filtered_idx,
                                    total_params);
    }

}; // End FFAManagerTypedImpl definition

FFAManager::FFAManager(const search::PulsarSearchConfig& cfg,
                       bool show_progress) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<FFAManagerTypedImpl<ComplexType>>(
            cfg, show_progress);
    } else {
        m_impl =
            std::make_unique<FFAManagerTypedImpl<float>>(cfg, show_progress);
    }
}
FFAManager::~FFAManager()                                      = default;
FFAManager::FFAManager(FFAManager&& other) noexcept            = default;
FFAManager& FFAManager::operator=(FFAManager&& other) noexcept = default;

void FFAManager::execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}
} // namespace loki::algorithms