#include "loki/pipelines/ffa_freq_sweep.hpp"

#include <memory>
#include <utility>

#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/cands.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/ffa_sweep_candidates.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

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

namespace {
template <SupportedFoldType FoldType>
class FFAFreqSweepTypedImpl final : public FFAFreqSweep::BaseImpl {
public:
    FFAFreqSweepTypedImpl(search::PulsarSearchConfig cfg, bool show_progress)
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
        m_ffa_workspace = memory::FFAWorkspace<FoldType>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(), m_base_cfg.get_nparams());
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            const auto& cfgs = m_region_planner.get_cfgs();
            std::vector<SizeType> n_reals;
            n_reals.reserve(cfgs.size());
            for (const auto& cfg : cfgs) {
                n_reals.push_back(cfg.get_nbins());
            }
            m_fft_manager.prepare_plans(n_reals);
        }
        m_scores_chunk.resize(planner_stats.get_max_scores_scratch_size());
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_fold_time.resize(planner_stats.get_max_buffer_size_time());
        validate_scratch_sizes();

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
        // Reset accumulated state so repeated execute() calls are independent
        m_ffa_stats = cands::FFAStatsCollection();
        m_cands.clear();
        m_total_passing_scores = 0;

        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = cands::FFAResultWriter(
            result_file, cands::FFAResultWriter::Mode::kWrite);
        auto param_names = m_base_cfg.get_param_names();
        param_names.emplace_back("width");
        writer.write_metadata(param_names, m_base_cfg.get_nbins(),
                              m_base_cfg.get_ducy_max(), m_base_cfg.get_wtsp());

        cands::FFATimerStats ffa_timer_stats_pipeline;
        double accumulated_flops     = 0.0;
        const auto& ffa_regions_cfgs = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::PulsarSearchConfig& cfg_cur = ffa_regions_cfgs[i];
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.min, freq_limits.max);
            cands::FFATimerStats ffa_timer_stats;
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
        flush_candidates(m_cands, m_region_decode, writer,
                         m_write_param_sets_batch, m_base_cfg.get_nparams());
        ffa_timer_stats_pipeline["io"] += timer.stop();
        m_ffa_stats.update_stats(ffa_timer_stats_pipeline,
                                 static_cast<float>(accumulated_flops));
        writer.write_ffa_stats(m_ffa_stats);
        spdlog::info("FFA Freq Sweep complete: {} candidates above S/N {:.2f}",
                     m_total_passing_scores, m_base_cfg.get_snr_min());
        spdlog::info("FFA Freq Sweep: timer: {}",
                     m_ffa_stats.get_concise_timer_summary());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    regions::FFARegionPlanner<FoldType> m_region_planner;
    std::vector<RegionDecode> m_region_decode;
    // Fixed-capacity accumulator; drained to disk whenever it fills up.
    CandidateBuffer m_cands;
    bool m_show_progress;

    memory::FFAWorkspace<FoldType> m_ffa_workspace;
    math::FFTWManager m_fft_manager;
    SizeType m_total_passing_scores{};
    // Per-chunk raw score scratch; overwritten by every chunk.
    std::vector<float> m_scores_chunk;
    std::vector<double> m_write_param_sets_batch; // includes width

    cands::FFAStatsCollection m_ffa_stats;
    // Persistent input/output buffers
    std::vector<float> m_fold_time;

    /// @brief Assert the planner-derived scratch sizes cover every chunk.
    void validate_scratch_sizes() const {
        for (SizeType i = 0; i < m_region_decode.size(); ++i) {
            error_check::check_less_equal(
                m_region_decode[i].get_n_scores(), m_scores_chunk.size(),
                std::format("FFAFreqSweep: chunk {} needs {} score slots but "
                            "the planner only sized the scratch for {}",
                            i, m_region_decode[i].get_n_scores(),
                            m_scores_chunk.size()));
        }
    }

    void execute_ffa_region(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            const search::PulsarSearchConfig& cfg,
                            SizeType region_id,
                            cands::FFAResultWriter& writer,
                            cands::FFATimerStats& ffa_timer_stats) {
        timing::SimpleTimer timer;
        // Create FFA with shared workspace
        timer.start();
        auto the_ffa =
            FFA<FoldType>(m_ffa_workspace, m_fft_manager, cfg, m_show_progress);
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
        const auto& dec     = m_region_decode[region_id];
        const auto n_scores = dec.get_n_scores();
        const auto snr_min  = static_cast<float>(cfg.get_snr_min());
        error_check::check_equal(dec.nsegments, SizeType{1},
                                 "FFAFreqSweep::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        // The decode strides must describe the fold layout we just produced.
        error_check::check_equal(ffa_plan.get_ncoords().back(), dec.ncoords,
                                 "FFAFreqSweep::execute_ffa_region: decode "
                                 "table is out of sync with the FFA plan");
        // Scratch is sized by the planner for the largest chunk; the
        // accumulator is separate, so this span never depends on how many
        // candidates have already survived.
        const auto scores_span = std::span(m_scores_chunk).first(n_scores);
        detection::snr_boxcar_3d(std::span(m_fold_time).first(fold_size_time),
                                 dec.widths, scores_span, dec.ncoords,
                                 dec.nbins, cfg.get_nthreads());

        // Move survivors into the accumulator, draining it whenever it fills.
        // The buffer is never full at the point of a push, so no input can
        // overrun it, and survivors stay in ascending score index order.
        SizeType n_passing = 0;
        for (SizeType score_idx = 0; score_idx < n_scores; ++score_idx) {
            if (scores_span[score_idx] < snr_min) {
                continue;
            }
            if (m_cands.is_full()) {
                ffa_timer_stats["score"] += timer.stop();
                timer.start();
                flush_candidates(m_cands, m_region_decode, writer,
                                 m_write_param_sets_batch,
                                 m_base_cfg.get_nparams());
                ffa_timer_stats["io"] += timer.stop();
                timer.start();
            }
            m_cands.push(scores_span[score_idx],
                         static_cast<uint32_t>(score_idx),
                         static_cast<uint32_t>(region_id));
            ++n_passing;
        }
        m_total_passing_scores += n_passing;

        ffa_timer_stats["score"] += timer.stop();
    }

}; // End FFAFreqSweepTypedImpl definition
} // End anonymous namespace

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