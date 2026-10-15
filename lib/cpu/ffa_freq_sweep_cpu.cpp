#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/regions.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ffa_engine.hpp"
#include "lib/detail/error_check.hpp"
#include "lib/detail/nvtx.hpp"
#include "lib/detail/timing.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/score_engine.hpp"
#include "lib/pipelines/ffa_freq_sweep_engine.hpp"
#include "lib/search/cands.hpp"
#include "lib/search/ffa_sweep_candidates.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::pipelines {

namespace {
template <SupportedFoldType FoldType>
class FFAFreqSweepCpuEngine final : public detail::FFAFreqSweepEngine {
public:
    FFAFreqSweepCpuEngine(search::FFASearchConfig cfg, bool show_progress)
        : m_base_cfg(std::move(cfg)),
          m_region_planner(m_base_cfg),
          m_region_decode(
              build_region_decode_table<FoldType>(m_region_planner.get_cfgs())),
          m_cands(m_region_planner.get_stats().get_max_candidates()),
          // A progress bar per chunk would be unreadable across many chunks.
          m_show_progress(show_progress &&
                          m_region_planner.get_nregions() == 1) {
        const auto& planner_stats = m_region_planner.get_stats();
        // Allocate buffers once, sized for the largest chunk
        m_ffa_workspace = memory::FFAWorkspaceCPU<FoldType>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(), m_base_cfg.get_nparams());
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_width_batch.resize(algorithms::kFFAFreqSweepWriteBatchSize);
        m_nbins_batch.resize(algorithms::kFFAFreqSweepWriteBatchSize);
        // Chunks score in scratch (see execute_ffa_region), so there is no
        // dense score array. The second ping-pong buffer of the merges is
        // not needed by frequency-only time-domain cone bands.
        if (m_base_cfg.get_use_fourier() || m_base_cfg.get_nparams() > 1) {
            m_fold_time.resize(planner_stats.get_max_buffer_size_time());
        }

        if (m_base_cfg.get_use_boxcar_kadane()) {
            throw std::invalid_argument(
                "use_boxcar_kadane is not supported for FFA frequency sweep "
                "(multi-width boxcar decoding is required)");
        }

        // Log the actual memory usage for the allocated buffers
        spdlog::info("FFAFreqSweep allocated {:.2f} GB ({:.2f} GB buffers "
                     "+ {:.2f} GB coords + {:.2f} GB extra)",
                     planner_stats.get_freq_sweep_memory_usage(),
                     planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_extra_memory_usage());
        spdlog::info("FFAFreqSweep will process {} chunks, keeping up to {} "
                     "candidates in RAM before flushing to disk",
                     m_region_planner.get_nregions(), m_cands.get_capacity());
    }

    ~FFAFreqSweepCpuEngine() final                                 = default;
    FFAFreqSweepCpuEngine(const FFAFreqSweepCpuEngine&)            = delete;
    FFAFreqSweepCpuEngine& operator=(const FFAFreqSweepCpuEngine&) = delete;
    FFAFreqSweepCpuEngine(FFAFreqSweepCpuEngine&&)                 = delete;
    FFAFreqSweepCpuEngine& operator=(FFAFreqSweepCpuEngine&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix,
                 std::string_view config_toml) override {
        timing::SimpleTimer timer;
        // Reset accumulated state so repeated execute() calls are independent
        m_ffa_stats = search::FFAStatsCollection();
        m_cands.clear();
        m_total_passing_scores = 0;

        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = search::FFAResultWriter(
            result_file, search::FFAResultWriter::Mode::kWrite);
        writer.write_metadata(
            search::FFAResultMetadata(m_base_cfg, config_toml));

        search::FFATimerStats ffa_timer_stats_pipeline;
        double accumulated_flops     = 0.0;
        const auto& ffa_regions_cfgs = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::FFASearchConfig& cfg_cur = ffa_regions_cfgs[i];
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.min, freq_limits.max);
            search::FFATimerStats ffa_timer_stats;
            const nvtx::Range chunk_range("ffa sweep: chunk");
            execute_ffa_region(ts_e, ts_v, cfg_cur, i, writer, ffa_timer_stats);
            accumulated_flops += m_region_decode[i].gflops;
            // Log per-chunk timing summary
            spdlog::info("FFA Chunk: timer: {}",
                         ffa_timer_stats.get_concise_timer_summary());
            // Update accumulated stats
            m_ffa_stats.update_stats(ffa_timer_stats);
        }

        // Drain whatever is still in RAM
        timer.start();
        search::flush_candidates(m_cands, m_region_decode, writer,
                                 m_write_param_sets_batch, m_width_batch,
                                 m_nbins_batch, m_base_cfg.get_nparams());
        ffa_timer_stats_pipeline["io"] += timer.stop();
        m_ffa_stats.update_stats(ffa_timer_stats_pipeline,
                                 static_cast<float>(accumulated_flops));
        writer.write_ffa_stats(m_ffa_stats);
        writer.finalize();
        spdlog::info("FFA Freq Sweep complete: {} candidates above S/N {:.2f}",
                     m_total_passing_scores, m_base_cfg.get_snr_min());
        spdlog::info("FFA Freq Sweep: timer: {}",
                     m_ffa_stats.get_concise_timer_summary());
    }

private:
    search::FFASearchConfig m_base_cfg;
    algorithms::FFARegionPlanner<FoldType> m_region_planner;
    std::vector<search::RegionDecode> m_region_decode;
    // Fixed-capacity accumulator; drained to disk whenever it fills up.
    search::CandidateBuffer m_cands;
    bool m_show_progress;

    memory::FFAWorkspaceCPU<FoldType> m_ffa_workspace;
    SizeType m_total_passing_scores{};
    std::vector<double> m_write_param_sets_batch;
    std::vector<std::uint16_t> m_width_batch;
    std::vector<std::uint16_t> m_nbins_batch;

    search::FFAStatsCollection m_ffa_stats;
    // Persistent input/output buffers
    utils::UninitVector<float> m_fold_time;

    void execute_ffa_region(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            const search::FFASearchConfig& cfg,
                            SizeType region_id,
                            search::FFAResultWriter& writer,
                            search::FFATimerStats& ffa_timer_stats) {
        timing::SimpleTimer timer;
        // Create FFA with shared workspace
        timer.start();
        // Internal pipeline: build the CPU engine directly on the shared
        // workspace (no facade, no dispatch).
        const auto ffa_engine = algorithms::detail::make_ffa_cpu<FoldType>(
            m_ffa_workspace, cfg, m_show_progress);
        auto& the_ffa                            = *ffa_engine;
        const plans::FFAPlan<FoldType>& ffa_plan = the_ffa.get_plan();
        const auto& dec                          = m_region_decode[region_id];
        const auto snr_min = static_cast<float>(cfg.get_snr_min());
        error_check::check_equal(dec.nsegments, SizeType{1},
                                 "FFAFreqSweep::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        error_check::check_equal(ffa_plan.get_ncoords().back(), dec.ncoords,
                                 "FFAFreqSweep::execute_ffa_region: decode "
                                 "table is out of sync with the FFA plan");
        // Every chunk scores in scratch: the final level is merged, (for
        // Fourier) inverted and scored tile by tile, so neither the final fold
        // nor a dense score array is stored. m_fold_time is the second
        // ping-pong buffer of the merges (unused by frequency-only time-domain
        // cone bands).
        std::vector<detection::SnrHit> hits;
        the_ffa.execute_scored(
            ts_e, ts_v, snr_min, dec.widths, hits,
            std::span<FoldType>(reinterpret_cast<FoldType*>(m_fold_time.data()),
                                (m_fold_time.size() * sizeof(float)) /
                                    sizeof(FoldType)));
        const auto brutefold_time = the_ffa.get_brute_fold_timing();
        const auto score_in_band  = the_ffa.get_last_score_timing();
        const auto wall           = timer.stop();
        const auto ffa_time =
            std::max(0.0F, wall - brutefold_time - score_in_band);
        ffa_timer_stats["brutefold"] += brutefold_time;
        ffa_timer_stats["ffa"] += ffa_time;
        spdlog::info("FFA Chunk detail: nbins={} bseg_brute={} levels={} "
                     "fuse_levels={} nfreqs0={} ncoords_top={} "
                     "brutefold_s={:.4f} brute_table_s={:.4f} ffa_s={:.4f}",
                     cfg.get_nbins(), cfg.get_bseg_brute(),
                     ffa_plan.get_n_levels(), the_ffa.get_last_fuse_levels(),
                     ffa_plan.get_ncoords().front(),
                     ffa_plan.get_ncoords().back(), brutefold_time,
                     the_ffa.get_brute_fold_init_timing(), ffa_time);
        timer.start();
        SizeType n_passing = 0;
        for (const auto& hit : hits) {
            if (m_cands.is_full()) {
                ffa_timer_stats["score"] += timer.stop();
                timer.start();
                search::flush_candidates(
                    m_cands, m_region_decode, writer, m_write_param_sets_batch,
                    m_width_batch, m_nbins_batch, m_base_cfg.get_nparams());
                ffa_timer_stats["io"] += timer.stop();
                timer.start();
            }
            m_cands.push(hit.snr, hit.score_index,
                         static_cast<uint32_t>(region_id));
            ++n_passing;
        }
        m_total_passing_scores += n_passing;
        ffa_timer_stats["score"] += timer.stop() + score_in_band;
    }

}; // End FFAFreqSweepCpuEngine definition
} // End anonymous namespace

namespace detail {
std::unique_ptr<FFAFreqSweepEngine>
make_ffa_freq_sweep_cpu(const search::FFASearchConfig& cfg,
                        bool show_progress) {
    if (cfg.get_use_fourier()) {
        return std::make_unique<FFAFreqSweepCpuEngine<ComplexType>>(
            cfg, show_progress);
    }
    return std::make_unique<FFAFreqSweepCpuEngine<float>>(cfg, show_progress);
}
} // namespace detail

} // namespace loki::pipelines
