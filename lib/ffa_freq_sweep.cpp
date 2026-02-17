#include "loki/pipelines/ffa_freq_sweep.hpp"

#include <memory>
#include <utility>

#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/plans.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"

namespace loki::algorithms {

class FFAFreqSweep::BaseImpl {
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

template <SupportedFoldType FoldType>
class FFAFreqSweepTypedImpl final : public FFAFreqSweep::BaseImpl {
public:
    FFAFreqSweepTypedImpl(search::PulsarSearchConfig cfg, bool show_progress)
        : m_base_cfg(std::move(cfg)),
          m_region_planner(m_base_cfg),
          m_show_progress(show_progress) {
        const auto& planner_stats = m_region_planner.get_stats();
        // Allocate buffers once, sized for the largest chunk
        m_ffa_workspace = FFAWorkspace<FoldType>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(), m_base_cfg.get_nparams());
        m_scores.resize(planner_stats.get_max_scores_size());
        m_passing_indices.resize(planner_stats.get_max_scores_size());
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_n_passing_scores_per_region.resize(m_region_planner.get_nregions());
        m_ffa_stats = std::make_unique<cands::FFAStatsCollection>();

        m_fold_time.resize(planner_stats.get_max_buffer_size_time());

        // Log the actual memory usage for the allocated buffers
        spdlog::info("FFAFreqSweep allocated {:.2f} GB ({:.2f} GB buffers "
                     "+ {:.2f} GB coords + {:.2f} GB extra)",
                     planner_stats.get_freq_sweep_memory_usage(),
                     planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_extra_memory_usage());
        spdlog::info("FFAFreqSweep will process {} chunks",
                     m_region_planner.get_nregions());
    }

    ~FFAFreqSweepTypedImpl() final                                 = default;
    FFAFreqSweepTypedImpl(const FFAFreqSweepTypedImpl&)            = delete;
    FFAFreqSweepTypedImpl& operator=(const FFAFreqSweepTypedImpl&) = delete;
    FFAFreqSweepTypedImpl(FFAFreqSweepTypedImpl&&)                 = delete;
    FFAFreqSweepTypedImpl& operator=(FFAFreqSweepTypedImpl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        timing::SimpleTimer timer;
        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = cands::FFAResultWriter(
            result_file, cands::FFAResultWriter::Mode::kWrite);
        auto param_names = m_base_cfg.get_param_names();
        param_names.emplace_back("width");
        writer.write_metadata(param_names, m_base_cfg.get_scoring_widths());

        m_total_passing_scores       = 0; // reset total passing scores
        const auto& ffa_regions_cfgs = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::PulsarSearchConfig& cfg_cur = ffa_regions_cfgs[i];
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.min, freq_limits.max);
            cands::FFATimerStats ffa_timer_stats;
            const SizeType n_passing =
                execute_ffa_region(ts_e, ts_v, cfg_cur, ffa_timer_stats);
            m_n_passing_scores_per_region[i] = n_passing;
            // Log per-chunk timing summary
            spdlog::info("FFA Chunk: timer: {}",
                         ffa_timer_stats.get_concise_timer_summary());
            // Update accumulated stats
            m_ffa_stats->update_stats(ffa_timer_stats);
        }

        // Save results
        timer.start();
        cands::FFATimerStats ffa_timer_stats_pipeline;
        const float accumulated_flops = save_results(writer);
        ffa_timer_stats_pipeline["io"] += timer.stop();
        m_ffa_stats->update_stats(ffa_timer_stats_pipeline, accumulated_flops);
        writer.write_ffa_stats(*m_ffa_stats);
        spdlog::info("FFA Freq Sweep complete.");
        spdlog::info("FFA Freq Sweep: timer: {}",
                     m_ffa_stats->get_concise_timer_summary());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    regions::FFARegionPlanner<FoldType> m_region_planner;
    bool m_show_progress;

    algorithms::FFAWorkspace<FoldType> m_ffa_workspace;
    SizeType m_total_passing_scores{};
    std::vector<float> m_scores;
    std::vector<uint32_t> m_passing_indices;
    std::vector<double> m_write_param_sets_batch; // includes width
    std::vector<SizeType> m_n_passing_scores_per_region;

    std::unique_ptr<cands::FFAStatsCollection> m_ffa_stats;
    // Persistent input/output buffers
    std::vector<float> m_fold_time;

    SizeType execute_ffa_region(std::span<const float> ts_e,
                                std::span<const float> ts_v,
                                const search::PulsarSearchConfig& cfg,
                                cands::FFATimerStats& ffa_timer_stats) {
        timing::SimpleTimer timer;
        // Create FFA with shared workspace
        timer.start();
        auto the_ffa = FFA<FoldType>(cfg, m_ffa_workspace, m_show_progress);
        const plans::FFAPlan<FoldType>& ffa_plan = the_ffa.get_plan();
        const auto buffer_size_time = ffa_plan.get_buffer_size_time();
        const auto fold_size_time   = ffa_plan.get_fold_size_time();
        the_ffa.execute(ts_e, ts_v,
                        std::span(m_fold_time).first(buffer_size_time));
        const auto brutefold_time = the_ffa.get_brute_fold_timing();
        ffa_timer_stats["brutefold"] += brutefold_time;
        ffa_timer_stats["ffa"] += timer.stop() - brutefold_time;

        // Compute scores
        timer.start();
        const auto nsegments = ffa_plan.get_nsegments().back();
        const auto ncoords   = ffa_plan.get_ncoords().back();
        const auto n_widths  = cfg.get_scoring_widths().size();
        const auto n_scores  = ncoords * n_widths;
        const auto snr_min   = cfg.get_snr_min();
        error_check::check_equal(nsegments, 1U,
                                 "FFAFreqSweep::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        // Calculate available space in buffers
        const SizeType available_space =
            m_scores.size() - m_total_passing_scores;
        error_check::check_greater_equal(
            available_space, n_scores,
            std::format(
                "Buffer overflow: {} candidates already accumulated, "
                "up to {} more could pass in this region, but only {} space "
                "available. Options: (1) Increase snr_min threshold (current: "
                "{:.2f}), (2) Increase max_passing_candidates config.",
                m_total_passing_scores, n_scores, available_space, snr_min));
        // Pass incremental spans with offset
        auto scores_span =
            std::span(m_scores).subspan(m_total_passing_scores, n_scores);
        detection::snr_boxcar_3d(std::span(m_fold_time).first(fold_size_time),
                                 cfg.get_scoring_widths(), scores_span, ncoords,
                                 cfg.get_nbins(), cfg.get_nthreads());

        // Compactify scores and passing indices
        SizeType n_passing = 0;

        const SizeType offset = m_total_passing_scores;
        for (SizeType score_idx = 0; score_idx < n_scores; ++score_idx) {
            if (m_scores[offset + score_idx] >= snr_min) {
                m_passing_indices[offset + n_passing] = score_idx;
                if (n_passing != score_idx) { // Only copy if positions differ
                    m_scores[offset + n_passing] = m_scores[offset + score_idx];
                }
                n_passing++;
            }
        }
        m_total_passing_scores += n_passing;

        ffa_timer_stats["score"] += timer.stop();
        return n_passing;
    }

    float save_results(cands::FFAResultWriter& result_writer) {
        const auto n_params         = m_base_cfg.get_nparams();
        const SizeType total_params = n_params + 1;
        const auto& scoring_widths  = m_base_cfg.get_scoring_widths();
        const SizeType n_widths     = scoring_widths.size();

        float accumulated_flops        = 0.0F;
        SizeType global_passing_offset = 0; // Track cumulative offset
        const auto& ffa_regions_cfgs   = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::PulsarSearchConfig& cfg_cur = ffa_regions_cfgs[i];
            plans::FFAPlan<FoldType> ffa_plan(cfg_cur);
            const auto& param_limits = cfg_cur.get_param_limits();
            const auto& param_counts = ffa_plan.get_param_counts().back();
            const auto& param_strides =
                ffa_plan.get_param_cart_strides().back();
            const auto n_passing = m_n_passing_scores_per_region[i];

            // Compute flops
            accumulated_flops += ffa_plan.get_gflops(/*return_in_time=*/true);
            const auto ncoords = ffa_plan.get_ncoords().back();
            const auto score_flops =
                (ncoords * 2) * (n_widths * 2 * cfg_cur.get_nbins());
            accumulated_flops += score_flops * 1e-9; // convert to GFLOPS

            // Process in batches and write incrementally
            SizeType batch_start = 0;
            while (batch_start < n_passing) {
                const SizeType batch_end =
                    std::min(batch_start + regions::kFFAFreqSweepWriteBatchSize,
                             n_passing);
                const SizeType batch_count = batch_end - batch_start;

                // Fill batch buffer
                for (SizeType i = 0; i < batch_count; ++i) {
                    // Access from global buffer with proper offset
                    const SizeType global_idx =
                        global_passing_offset + batch_start + i;
                    const SizeType score_idx = m_passing_indices[global_idx];
                    const SizeType coord_idx = score_idx / n_widths;
                    const SizeType width_idx = score_idx % n_widths;

                    // Reconstruct parameters from coord_idx using index
                    // arithmetic
                    SizeType remaining = coord_idx;
                    for (SizeType j = 0; j < n_params; ++j) {
                        const SizeType param_idx = remaining / param_strides[j];
                        remaining -= param_idx * param_strides[j];
                        m_write_param_sets_batch[(i * total_params) + j] =
                            psr_utils::get_param_val_at_idx(
                                param_limits[j], param_counts[j], param_idx);
                    }
                    m_write_param_sets_batch[(i * total_params) + n_params] =
                        static_cast<double>(scoring_widths[width_idx]);
                }

                // Write batch
                result_writer.write_results(
                    std::span(m_write_param_sets_batch)
                        .first(batch_count * total_params),
                    std::span(m_scores).subspan(
                        global_passing_offset + batch_start, batch_count),
                    batch_count, total_params);
                batch_start = batch_end;
            }
            global_passing_offset += n_passing;
        }
        return accumulated_flops;
    }
}; // End FFAFreqSweepTypedImpl definition

FFAFreqSweep::FFAFreqSweep(const search::PulsarSearchConfig& cfg,
                           bool show_progress) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<FFAFreqSweepTypedImpl<ComplexType>>(
            cfg, show_progress);
    } else {
        m_impl =
            std::make_unique<FFAFreqSweepTypedImpl<float>>(cfg, show_progress);
    }
}
FFAFreqSweep::~FFAFreqSweep()                                        = default;
FFAFreqSweep::FFAFreqSweep(FFAFreqSweep&& other) noexcept            = default;
FFAFreqSweep& FFAFreqSweep::operator=(FFAFreqSweep&& other) noexcept = default;

void FFAFreqSweep::execute(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           const std::filesystem::path& outdir,
                           std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}
} // namespace loki::algorithms