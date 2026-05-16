#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <type_traits>
#include <utility>

#include <BS_thread_pool.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/core/dynamic.hpp"
#include "loki/exceptions.hpp"
#include "loki/progress.hpp"
#include "loki/psr_utils.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::algorithms {

namespace {
// Iteration stats for single-threaded pruning
struct IterationStats {
    SizeType n_leaves     = 0;
    SizeType n_leaves_phy = 0;
    float score_min       = std::numeric_limits<float>::max();
    float score_max       = std::numeric_limits<float>::lowest();
    cands::PruneTimerStats batch_timers;
};

template <SupportedFoldType FoldType> class PruneImpl {
public:
    // External workspace constructor only
    PruneImpl(memory::EPWorkspace<FoldType>& workspace,
              search::PulsarSearchConfig cfg,
              std::span<const float> threshold_scheme,
              SizeType max_sugg,
              SizeType batch_size,
              SizeType branch_max,
              std::string_view poly_basis)
        : m_workspace_ptr(&workspace),
          m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_branch_max(branch_max),
          m_poly_basis(poly_basis),
          m_total_levels(m_threshold_scheme.size()) {
        error_check::check_less_equal(m_cfg.get_nparams(), 5,
                                      "Pruning not supported for nparams > 5.");
        m_prune_funcs = core::create_prune_dp_functs<FoldType>(
            m_poly_basis, m_ffa_plan.get_param_counts().back(),
            m_ffa_plan.get_dparams_actual().back(),
            m_ffa_plan.get_nsegments().back(),
            m_ffa_plan.get_tsegments().back(), m_cfg, m_batch_size,
            m_branch_max);
    }

    ~PruneImpl()                           = default;
    PruneImpl(const PruneImpl&)            = delete;
    PruneImpl& operator=(const PruneImpl&) = delete;
    PruneImpl(PruneImpl&&)                 = delete;
    PruneImpl& operator=(PruneImpl&&)      = delete;

    SizeType get_memory_usage() const noexcept {
        return get_workspace().get_memory_usage_gib();
    }

    void execute(std::span<const FoldType> ffa_fold,
                 SizeType ref_seg,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file,
                 progress::MultiprocessProgressTracker* tracker,
                 int task_id,
                 bool show_progress) {
        const std::string run_name =
            std::format("{:03d}_{:02d}", ref_seg, task_id);

        // Log detailed memory usage
        const auto& ws                 = get_workspace();
        const auto memory_workspace_gb = ws.prune.get_memory_usage_gib();
        const auto memory_tree_gb      = ws.world_tree.get_memory_usage_gib();
        spdlog::info("Pruning run {:03d}: Memory Usage: {:.2f} GB "
                     "(tree) + {:.2f} GB (workspace)",
                     ref_seg, memory_tree_gb, memory_workspace_gb);

        // Setup log and result files
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / std::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / std::format("tmp_{}_results.h5", run_name));
        std::ofstream log(actual_log_file, std::ios::app);
        log << std::format("Pruning log for ref segment: {}\n", ref_seg);
        log.close();

        const auto nsegments = m_ffa_plan.get_nsegments().back();
        std::unique_ptr<progress::ProgressGuard> progress_guard;
        std::unique_ptr<progress::ProgressTracker> bar;
        if (show_progress) {
            progress_guard = std::make_unique<progress::ProgressGuard>(true);
            bar            = std::make_unique<progress::ProgressTracker>(
                std::format("Pruning segment {:03d}", ref_seg), nsegments - 1,
                tracker, task_id);
        }

        initialize(ffa_fold, ref_seg, actual_log_file);

        for (SizeType iter = 0; iter < nsegments - 1; ++iter) {
            execute_iteration(ffa_fold, actual_log_file);
            // Check for early termination (no survivors)
            if (m_prune_complete) {
                spdlog::info(
                    "Pruning terminated early at iteration {} - no survivors",
                    iter + 1);
                break;
            }
            if (bar) {
                bar->set_score(ws.world_tree.get_score_max());
                bar->set_leaves(ws.world_tree.get_size_lb());
                bar->set_progress(iter + 1);
            }
        }

        // Ascend survivors to the middle of the data
        ascend_survivors_batched(ffa_fold);

        // Write results (after transforming to middle of the data)
        report_survivors(actual_result_file, run_name);

        // Final log entries
        std::ofstream final_log(actual_log_file, std::ios::app);
        final_log << std::format("Pruning run complete for ref segment {}\n",
                                 ref_seg);
        final_log << std::format("Time: {}\n\n", m_pstats.get_timer_summary());
        final_log.close();
        spdlog::info("Pruning run {:03d}: complete", ref_seg);
        spdlog::info("Pruning run {:03d}: stats: {}", ref_seg,
                     m_pstats.get_stats_summary());
        spdlog::info("Pruning run {:03d}: timer: {}", ref_seg,
                     m_pstats.get_concise_timer_summary());
    }

private:
    // The observer pointer that always points to the active workspace.
    memory::EPWorkspace<FoldType>* m_workspace_ptr{nullptr};

    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan<FoldType> m_ffa_plan;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    SizeType m_branch_max;
    std::string m_poly_basis;
    SizeType m_total_levels;

    bool m_prune_complete{false};
    SizeType m_prune_level{};
    psr_utils::MiddleOutScheme m_snail_scheme;
    cands::PruneStatsCollection m_pstats;
    std::unique_ptr<core::PruneDPFuncts<FoldType>> m_prune_funcs;

    [[nodiscard]] memory::EPWorkspace<FoldType>& get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::EPWorkspace<FoldType>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    void initialize(std::span<const FoldType> ffa_fold,
                    SizeType ref_seg,
                    const std::filesystem::path& log_file) {
        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        world_tree.reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        const auto tseg      = m_ffa_plan.get_tsegments().back();
        m_snail_scheme = psr_utils::MiddleOutScheme(nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;
        spdlog::info("Pruning run {:03d}: initialized", ref_seg);

        // Initialize the world tree with the first segment
        const auto fold_segment =
            m_prune_funcs->load_segment(ffa_fold, m_snail_scheme.get_ref_idx());
        const auto coord_init = m_snail_scheme.get_coord(m_prune_level);
        const auto n_leaves   = m_ffa_plan.get_ncoords().back();
        m_prune_funcs->seed(fold_segment, ws.seed_leaves, ws.seed_scores,
                            coord_init);
        world_tree.add_initial(ws.seed_leaves, fold_segment, ws.seed_scores,
                               n_leaves);

        // Initialize the prune stats
        m_pstats = cands::PruneStatsCollection();
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = m_snail_scheme.get_segment_idx(m_prune_level),
            .threshold     = 0,
            .score_min     = world_tree.get_score_min(),
            .score_max     = world_tree.get_score_max(),
            .n_branches    = world_tree.get_size(),
            .n_leaves      = world_tree.get_size(),
            .n_leaves_phy  = world_tree.get_size(),
            .n_leaves_surv = world_tree.get_size(),
        };
        m_pstats.update_stats(pstats_cur);

        // Write the initial prune stats to the log file
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary();
        log.close();
    }

    void ascend_survivors_batched(std::span<const FoldType> ffa_fold) {
        if (m_prune_complete) {
            return;
        }
        auto& ws               = get_workspace();
        auto& world_tree       = ws.world_tree;
        const auto n_survivors = world_tree.get_size();
        if (n_survivors == 0) {
            return;
        }
        auto& prune_ws       = ws.prune;
        const auto coord_mid = m_snail_scheme.get_coord(m_prune_level);
        const auto [idx_segments, coord_segments] =
            m_snail_scheme.get_segment_coords_so_far(m_prune_level);
        const auto batch_cap =
            std::max(SizeType{1}, std::min(m_batch_size, n_survivors));

        memory::CircularView<double> leaves_cv =
            world_tree.get_leaves_circular_view();
        memory::CircularView<FoldType> folds_cv =
            world_tree.get_folds_circular_view();
        memory::CircularView<float> scores_cv =
            world_tree.get_scores_circular_view();
        memory::CircularView<float> scores_ep_cv =
            world_tree.get_scores_ep_circular_view();

        const auto leaves_stride = world_tree.get_leaves_stride();
        const auto folds_stride  = world_tree.get_folds_stride();

        auto process_region = [&](std::span<const double> leaves_region,
                                  std::span<FoldType> folds_region,
                                  std::span<float> scores_region,
                                  std::span<float> scores_ep_region) {
            if (leaves_region.empty()) {
                return;
            }
            error_check::check_equal(
                leaves_region.size() % leaves_stride, SizeType{0},
                "ascend_survivors_batched: leaves segment not stride-aligned");
            const auto n_leaves_region = leaves_region.size() / leaves_stride;
            error_check::check_equal(
                folds_region.size(), n_leaves_region * folds_stride,
                "ascend_survivors_batched: folds/leaves leaf count mismatch");
            error_check::check_equal(
                scores_region.size(), n_leaves_region,
                "ascend_survivors_batched: scores/leaves leaf count mismatch");
            error_check::check_equal(
                scores_ep_region.size(), n_leaves_region,
                "ascend_survivors_batched: scores_ep/leaves leaf count "
                "mismatch");

            for (SizeType off = 0; off < n_leaves_region; off += batch_cap) {
                const auto chunk = std::min(batch_cap, n_leaves_region - off);
                m_prune_funcs->ascend(
                    ffa_fold,
                    leaves_region.subspan(off * leaves_stride,
                                          chunk * leaves_stride),
                    folds_region.subspan(off * folds_stride,
                                         chunk * folds_stride),
                    scores_region.subspan(off, chunk),
                    scores_ep_region.subspan(off, chunk), idx_segments,
                    coord_segments, coord_mid, prune_ws.branched_param_idx,
                    prune_ws.branched_phase_shift, chunk);
            }
        };

        process_region(leaves_cv.first, folds_cv.first, scores_cv.first,
                       scores_ep_cv.first);
        process_region(leaves_cv.second, folds_cv.second, scores_cv.second,
                       scores_ep_cv.second);
    }

    void report_survivors(const std::filesystem::path& actual_result_file,
                          std::string_view run_name) {
        auto& ws            = get_workspace();
        auto& world_tree    = ws.world_tree;
        const auto n_leaves = world_tree.get_size();

        memory::CircularView<double> leaves_view =
            world_tree.get_leaves_circular_view();
        memory::CircularView<float> scores_view =
            world_tree.get_scores_circular_view();
        memory::CircularView<float> scores_ep_view =
            world_tree.get_scores_ep_circular_view();

        if (n_leaves > 0) {
            // Transform the suggestion params to middle of the data
            const auto coord_mid     = m_snail_scheme.get_coord(m_prune_level);
            const auto leaves_stride = world_tree.get_leaves_stride();
            const auto n1            = leaves_view.first.size() / leaves_stride;
            const auto n2 = leaves_view.second.size() / leaves_stride;
            m_prune_funcs->report(leaves_view.first, coord_mid, n1);
            if (n2 > 0) {
                m_prune_funcs->report(leaves_view.second, coord_mid, n2);
            }
        }
        const auto total_pruning_gflops = compute_total_prune_gflops();
        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        result_writer.write_run_results(
            run_name, m_snail_scheme.get_data(), leaves_view, scores_view,
            scores_ep_view, total_pruning_gflops, n_leaves, m_cfg.get_nparams(),
            m_pstats);
    }

    [[nodiscard]] double compute_total_prune_gflops() const {
        const auto packed_stats = m_pstats.get_packed_data();
        const auto& level_stats = packed_stats.first;
        if (level_stats.empty()) {
            return 0.0;
        }

        const auto n_params = static_cast<double>(m_cfg.get_nparams());
        const auto nbins    = static_cast<double>(m_cfg.get_nbins());
        const auto nbins_f  = static_cast<double>(m_cfg.get_nbins_f());
        const auto n_widths = static_cast<double>(m_cfg.get_n_scoring_widths());
        const auto conservative_tile =
            static_cast<double>(m_cfg.get_use_conservative_tile());

        auto score_flops = [&](double n_leaves) {
            return (n_leaves * 2.0) * (n_widths * 2.0 * nbins);
        };
        auto irfft_flops = [&](double n_leaves) {
            if constexpr (std::is_same_v<FoldType, ComplexType>) {
                return (2.0 * n_leaves) * nbins * std::log2(nbins);
            } else {
                return 0.0;
            }
        };
        auto shift_add_flops = [&](double n_leaves) {
            if constexpr (std::is_same_v<FoldType, ComplexType>) {
                return n_leaves * 2.0 * nbins_f * 8.0;
            } else {
                return n_leaves * 2.0 * nbins;
            }
        };

        auto branch_flops = [&](double n_branches, double n_leaves) {
            // Dominant Taylor branch arithmetic: per-parameter step/shift work
            // over input branches plus child-center generation over outputs.
            return (n_branches * n_params * 10.0) + (n_leaves * n_params * 2.0);
        };
        auto resolve_flops = [&](double n_leaves) {
            // Polynomial propagation to acceleration/frequency plus phase and
            // nearest-grid arithmetic. The order-dependent part scales with
            // the number of Taylor parameters.
            return n_leaves * ((6.0 * n_params) + 16.0);
        };
        auto transform_flops = [&](double n_leaves) {
            // Value propagation is triangular in the Taylor order; conservative
            // tiles also propagate uncertainty with squared terms and sqrt.
            const auto value_flops = n_params * (n_params + 1.0);
            const auto error_flops =
                conservative_tile * n_params * ((2.0 * n_params) + 1.0);
            return n_leaves * (value_flops + error_flops);
        };
        auto report_flops = [&](double n_leaves) {
            // Gauge transform and error propagation for all non-frequency
            // parameters, plus final frequency/error conversion.
            return n_leaves * (((n_params - 1.0) * 12.0) + 4.0);
        };

        double total_flops = 0.0;
        for (const auto& stats : level_stats) {
            const auto n_branches    = static_cast<double>(stats.n_branches);
            const auto n_leaves      = static_cast<double>(stats.n_leaves);
            const auto n_leaves_phy  = static_cast<double>(stats.n_leaves_phy);
            const auto n_leaves_surv = static_cast<double>(stats.n_leaves_surv);

            if (stats.level == 0) {
                total_flops += irfft_flops(n_leaves_surv);
                total_flops += score_flops(n_leaves_surv);
                continue;
            }

            total_flops += branch_flops(n_branches, n_leaves);
            total_flops += resolve_flops(n_leaves_phy);
            total_flops += shift_add_flops(n_leaves_phy);
            total_flops += irfft_flops(n_leaves_phy);
            total_flops += score_flops(n_leaves_phy);
            total_flops += transform_flops(n_leaves_surv);
        }

        const auto& final_stats = level_stats.back();
        const auto n_final_survivors =
            static_cast<double>(final_stats.n_leaves_surv);
        const auto segment_coords =
            m_snail_scheme.get_segment_coords_so_far(m_prune_level);
        const auto n_segments =
            static_cast<double>(segment_coords.first.size());

        total_flops += n_segments * resolve_flops(n_final_survivors);
        total_flops += n_segments * shift_add_flops(n_final_survivors);
        total_flops += irfft_flops(n_final_survivors);
        total_flops += score_flops(n_final_survivors);
        total_flops += report_flops(n_final_survivors);

        return total_flops * 1.0e-9;
    }

    void execute_iteration(std::span<const FoldType> ffa_fold,
                           const std::filesystem::path& log_file) {
        if (m_prune_complete) {
            return;
        }
        ++m_prune_level;
        error_check::check_less_equal(
            m_prune_level, m_threshold_scheme.size(),
            "Pruning complete - exceeded threshold scheme length");

        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        // Prepare for in-place update: mark start of write region, reset size
        // for new suggestions.
        world_tree.prepare_in_place_update();

        IterationStats stats;
        const auto seg_idx_cur = m_snail_scheme.get_segment_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches = world_tree.get_size_old();

        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats);

        // Finalize: make new region active, defragment for contiguous access.
        world_tree.finalize_in_place_update();

        // Update statistics
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = seg_idx_cur,
            .threshold     = threshold,
            .score_min     = stats.score_min,
            .score_max     = stats.score_max,
            .n_branches    = n_branches,
            .n_leaves      = stats.n_leaves,
            .n_leaves_phy  = stats.n_leaves_phy,
            .n_leaves_surv = world_tree.get_size(),
        };
        // Write stats to log
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary();
        log.close();
        m_pstats.update_stats(pstats_cur, stats.batch_timers);

        // Check if no survivors
        if (world_tree.get_size() == 0) {
            m_prune_complete = true;
            spdlog::info("Pruning run complete at level {} - no survivors",
                         m_prune_level);
            return;
        }
    }

    // Iteration flow: Branch -> Validate -> Resolve -> Load/Shift/Add -> Score
    // -> Filter -> Transform -> Add to buffer.
    // Buffer manages space via trimming; advances consumption post-batch.
    void execute_iteration_batched(std::span<const FoldType> ffa_fold,
                                   SizeType seg_idx_cur,
                                   float threshold,
                                   IterationStats& stats) {
        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        auto& prune_ws   = ws.prune;
        auto& branch_ws  = ws.branch;

        // Get coordinates
        const auto coord_init = m_snail_scheme.get_coord(0);
        const auto coord_prev = m_snail_scheme.get_coord(m_prune_level - 1);
        const auto coord_next = m_snail_scheme.get_coord(m_prune_level);
        const auto coord_cur  = m_snail_scheme.get_current_coord(m_prune_level);
        const auto coord_add  = m_snail_scheme.get_segment_coord(m_prune_level);

        // Load fold segment for current level
        const auto ffa_fold_segment =
            m_prune_funcs->load_segment(ffa_fold, seg_idx_cur);

        auto current_threshold = threshold;

        const auto n_branches = world_tree.get_size_old();
        const auto batch_size =
            std::max(1UL, std::min(m_batch_size, n_branches));

        timing::SimpleTimer timer;

        // Process branches in batches
        // Process branches in potentially split batches to handle wraps
        SizeType total_processed = 0;
        while (total_processed < n_branches) {
            const SizeType remaining       = n_branches - total_processed;
            const SizeType this_batch_size = std::min(batch_size, remaining);
            // Get contiguous span; it may be smaller if wrap occurs
            // Read from the beginning of unconsumed data
            auto [leaves_tree_span, current_batch_size] =
                world_tree.get_leaves_span(this_batch_size);
            if (current_batch_size == 0) {
                throw std::runtime_error(
                    std::format("Loaded batch size is 0: total_processed={}, "
                                "this_batch_size={}, remaining={}",
                                total_processed, this_batch_size, remaining));
            }
            total_processed += current_batch_size;

            // Branch
            timer.start();
            const auto n_leaves_batch = m_prune_funcs->branch(
                leaves_tree_span, prune_ws.branched_leaves,
                prune_ws.branched_indices, coord_cur, coord_prev,
                current_batch_size, branch_ws);
            stats.batch_timers["branch"] += timer.stop();
            stats.n_leaves += n_leaves_batch;
            if (n_leaves_batch == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_batch, prune_ws.max_branched_leaves,
                "Branch factor exceeded workspace size:n_leaves_batch <= "
                "max_branched_leaves");

            // Validation
            timer.start();
            const auto n_leaves_after_validation = m_prune_funcs->validate(
                prune_ws.branched_leaves, prune_ws.branched_indices, coord_cur,
                n_leaves_batch);
            stats.batch_timers["validate"] += timer.stop();
            stats.n_leaves_phy += n_leaves_after_validation;
            if (n_leaves_after_validation == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }

            // Resolve
            timer.start();
            m_prune_funcs->resolve(
                prune_ws.branched_leaves, prune_ws.branched_param_idx,
                prune_ws.branched_phase_shift, coord_add, coord_cur, coord_init,
                n_leaves_after_validation);
            stats.batch_timers["resolve"] += timer.stop();

            // Load, shift, add (Map branched_itree to physical indices)
            timer.start();
            m_prune_funcs->shift_add(
                world_tree.get_folds(), prune_ws.branched_indices,
                ffa_fold_segment, prune_ws.branched_param_idx,
                prune_ws.branched_phase_shift, prune_ws.branched_folds,
                n_leaves_after_validation, world_tree.get_physical_start_idx(),
                world_tree.get_capacity());
            stats.batch_timers["shift_add"] += timer.stop();

            // Score and filter
            timer.start();
            const SizeType n_leaves_passing = m_prune_funcs->score_and_filter(
                prune_ws.branched_folds, prune_ws.branched_scores,
                prune_ws.branched_indices, current_threshold,
                n_leaves_after_validation);
            auto branched_scores_span =
                std::span<const float>(prune_ws.branched_scores)
                    .first(n_leaves_after_validation);
            const auto [min_it, max_it] =
                std::ranges::minmax_element(branched_scores_span);
            stats.score_min = std::min(stats.score_min, *min_it);
            stats.score_max = std::max(stats.score_max, *max_it);
            stats.batch_timers["score"] += timer.stop();

            if (n_leaves_passing == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_passing, prune_ws.max_branched_leaves,
                "n_leaves_passing <= max_branched_leaves");

            // Transform
            timer.start();
            m_prune_funcs->transform(prune_ws.branched_leaves,
                                     prune_ws.branched_indices, coord_next,
                                     coord_cur, n_leaves_passing);
            stats.batch_timers["transform"] += timer.stop();

            // Add batch to output suggestions
            timer.start();
            current_threshold = world_tree.add_batch_scattered(
                prune_ws.branched_leaves, prune_ws.branched_folds,
                prune_ws.branched_scores, prune_ws.branched_indices,
                current_threshold, n_leaves_passing);
            stats.batch_timers["batch_add"] += timer.stop();
            // Notify the buffer that a batch of the old suggestions has been
            // consumed
            world_tree.consume_read(current_batch_size);
        }
    }
}; // End Prune::Impl definition

} // End anonymous namespace

// EPMultiPass::Impl implementation
template <SupportedFoldType FoldType> class EPMultiPass<FoldType>::Impl {
public:
    // Self-owned workspace constructor
    Impl(search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         bool show_progress)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()) {
        // Create branching pattern and branch max
        m_branching_pattern   = m_ffa_plan.get_branching_pattern(m_poly_basis);
        const auto branch_max = *std::ranges::max_element(m_branching_pattern);
        m_branch_max =
            std::max(static_cast<SizeType>(std::ceil(branch_max * 2)), 32UL);

        // Allocate workspaces
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        m_workspace_storage.reserve(m_nthreads);
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            for (SizeType i = 0; i < static_cast<SizeType>(m_nthreads); ++i) {
                m_workspace_storage.emplace_back(
                    m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                    m_cfg.get_nparams(), m_cfg.get_nbins_f(), nsegments);
            }
        } else {
            for (SizeType i = 0; i < static_cast<SizeType>(m_nthreads); ++i) {
                m_workspace_storage.emplace_back(
                    m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                    m_cfg.get_nparams(), m_cfg.get_nbins(), nsegments);
            }
        }
        // Validate storage
        error_check::check_greater_equal(
            m_workspace_storage.size(), static_cast<SizeType>(m_nthreads),
            "EPMultiPass: Allocated workspaces size is less than requested "
            "nthreads.");
        // Point the span to our internal storage
        // INVARIANT: m_workspace_storage must not be modified after
        // m_workspaces_view is set. Both move and copy of Impl are deleted to
        // enforce this.
        m_workspaces_view = std::span(m_workspace_storage);
    }

    // External workspace constructor
    Impl(std::span<memory::EPWorkspace<FoldType>> external_workspaces,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         bool show_progress)
        : m_workspaces_view(external_workspaces),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()) {
        // Create branching pattern and branch max
        m_branching_pattern   = m_ffa_plan.get_branching_pattern(m_poly_basis);
        const auto branch_max = *std::ranges::max_element(m_branching_pattern);
        m_branch_max =
            std::max(static_cast<SizeType>(std::ceil(branch_max * 2)), 32UL);
        // Validate workspaces
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        const auto nsegments   = m_ffa_plan.get_nsegments().back();
        error_check::check_greater_equal(
            m_workspaces_view.size(), static_cast<SizeType>(m_nthreads),
            "EPMultiPass: Provided external workspaces is less than requested "
            "nthreads.");
        const SizeType nbins = std::is_same_v<FoldType, ComplexType>
                                   ? m_cfg.get_nbins_f()
                                   : m_cfg.get_nbins();
        for (const auto& ws : m_workspaces_view) {
            ws.validate(m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                        m_cfg.get_nparams(), nbins, nsegments);
        }
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) {
        timing::SimpleTimer timer;
        timer.start();
        spdlog::info("EPMultiPass: Initializing with FFA");
        // Create appropriate FFA fold
        std::tuple<std::vector<FoldType>, plans::FFAPlan<FoldType>> result =
            compute_ffa<FoldType>(ts_e, ts_v, m_cfg, /*quiet=*/false,
                                  m_show_progress);
        const std::vector<FoldType> ffa_fold = std::get<0>(result);
        plans::FFAPlan<FoldType> ffa_plan    = std::move(std::get<1>(result));

        // Setup output files and directory
        const auto nsegments = ffa_plan.get_nsegments().back();
        const std::string filebase =
            std::format("{}_pruning_nstages_{}", file_prefix, nsegments);
        const auto log_file =
            (outdir / std::format("{}_log.txt", filebase)).lexically_normal();
        const auto result_file =
            (outdir / std::format("{}_results.h5", filebase))
                .lexically_normal();

        // Create output directory
        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
        if (!std::filesystem::exists(outdir)) {
            throw std::runtime_error(
                std::format("EPMultiPass::execute: Failed to create output "
                            "directory '{}': {}",
                            outdir.string(), ec.message()));
        }

        // Determine ref_segs to process
        auto ref_segs_to_process =
            utils::determine_ref_segs_pareto(nsegments, m_n_runs, m_ref_segs);
        spdlog::info("Starting Pruning for {} runs, with {} threads",
                     ref_segs_to_process.size(), m_nthreads);

        // Initialize log file
        std::ofstream log(log_file);
        log << "Pruning log\n";
        log.close();

        // Write metadata to result file
        auto writer = cands::PruneResultWriter(
            result_file, cands::PruneResultWriter::Mode::kWrite);
        writer.write_metadata(m_cfg.get_param_names(), nsegments, m_max_sugg,
                              m_threshold_scheme);
        // Execute based on thread count
        if (m_nthreads == 1) {
            execute_single_threaded(ffa_fold, ref_segs_to_process, outdir,
                                    log_file, result_file);
        } else {
            execute_multi_threaded(ffa_fold, ref_segs_to_process, outdir,
                                   log_file);
            cands::merge_prune_result_files(outdir, log_file, result_file);
        }
        const auto ep_time = timer.stop();
        // Write final runtime to result file
        auto writer_final = cands::PruneResultWriter(
            result_file, cands::PruneResultWriter::Mode::kAppend);
        writer_final.write_runtime(ep_time);
        spdlog::info("Pruning complete. Results saved to {}",
                     result_file.string());
        spdlog::info("Pruning time: {:.2f} seconds", ep_time);
    }

private:
    // Pool of owned workspaces
    std::vector<memory::EPWorkspace<FoldType>> m_workspace_storage;
    std::span<memory::EPWorkspace<FoldType>> m_workspaces_view;

    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    std::optional<SizeType> m_n_runs;
    std::optional<std::vector<SizeType>> m_ref_segs;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string m_poly_basis;
    bool m_show_progress;

    plans::FFAPlan<FoldType> m_ffa_plan;
    int m_nthreads;
    std::vector<double> m_branching_pattern;
    SizeType m_branch_max{0};

    // Safely get the workspace for a specific thread index
    [[nodiscard]] memory::EPWorkspace<FoldType>&
    get_thread_workspace(SizeType thread_idx = 0) noexcept {
        return m_workspaces_view[thread_idx];
    }

    void execute_single_threaded(std::span<const FoldType> ffa_fold,
                                 std::span<const SizeType> ref_segs,
                                 const std::filesystem::path& outdir,
                                 const std::filesystem::path& log_file,
                                 const std::filesystem::path& result_file) {
        auto& ws = get_thread_workspace(0);
        auto prune =
            PruneImpl<FoldType>(ws, m_cfg, m_threshold_scheme, m_max_sugg,
                                m_batch_size, m_branch_max, m_poly_basis);
        for (const auto ref_seg : ref_segs) {
            prune.execute(ffa_fold, ref_seg, outdir, log_file, result_file,
                          /*tracker=*/nullptr, /*task_id=*/0, m_show_progress);
        }
    }

    void execute_multi_threaded(std::span<const FoldType> ffa_fold,
                                const std::vector<SizeType>& ref_segs,
                                const std::filesystem::path& outdir,
                                const std::filesystem::path& log_file) {
        // Only create progress tracker if show_progress is true
        std::unique_ptr<progress::MultiprocessProgressTracker> tracker;
        if (m_show_progress) {
            tracker = std::make_unique<progress::MultiprocessProgressTracker>(
                "Pruning tree");
            tracker->start();
        }
        // Create thread pool
        BS::thread_pool pool(m_nthreads);

        // Submit tasks for each ref_seg
        std::vector<std::future<void>> futures;
        futures.reserve(ref_segs.size());
        std::vector<int> task_ids;
        task_ids.reserve(ref_segs.size());

        const auto nsegments = m_ffa_plan.get_nsegments().back();
        int id               = 0;
        for (const auto ref_seg : ref_segs) {
            if (tracker) {
                id = tracker->add_task(
                    std::format("Pruning segment {:03d}", ref_seg),
                    nsegments - 1,
                    /*transient=*/true);
            } else {
                id++;
            }
            task_ids.push_back(id);

            auto future = pool.submit_task([this, ref_seg, outdir,
                                            tracker_ptr = tracker.get(), id,
                                            &ffa_fold]() mutable {
                const auto thread_idx = BS::this_thread::get_index().value();
                auto& ws              = get_thread_workspace(thread_idx);
                auto prune = PruneImpl<FoldType>(ws, m_cfg, m_threshold_scheme,
                                                 m_max_sugg, m_batch_size,
                                                 m_branch_max, m_poly_basis);

                prune.execute(
                    ffa_fold, ref_seg, outdir, /*log_file=*/std::nullopt,
                    /*result_file=*/std::nullopt, /*tracker=*/tracker_ptr,
                    /*task_id=*/id, /*show_progress=*/m_show_progress);
            });
            futures.push_back(std::move(future));
        }
        // Wait for all tasks to complete and handle exceptions
        std::vector<std::pair<SizeType, std::string>> errors;
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                futures[i].get();
            } catch (const std::exception& e) {
                const std::string error_msg = std::format(
                    "Error in ref_seg {}: {}", ref_segs[i], e.what());
                errors.emplace_back(ref_segs[i], error_msg);
            }
        }

        if (errors.empty()) {
            spdlog::info("All tasks completed successfully.");
        } else {
            spdlog::warn("Completed with {} errors out of {} tasks.",
                         errors.size(), ref_segs.size());
        }

        // Log errors to file
        if (!errors.empty()) {
            std::ofstream log(log_file, std::ios::app);
            for (const auto& [ref_seg, error_msg] : errors) {
                log << std::format("Error processing ref_seg {}: {}\n", ref_seg,
                                   error_msg);
            }
            log.close();
            spdlog::error("Errors occurred during processing. Last error: {}",
                          errors.back().second);

            throw std::runtime_error(
                std::format("Multi-threaded execution failed: {} out of {} "
                            "tasks failed",
                            errors.size(), ref_segs.size()));
        }
        if (tracker) {
            tracker->stop();
        }
    }
}; // End EPMultiPass::Impl definition

// --- Definitions for EPMultiPass ---
template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    bool show_progress)
    : m_impl(std::make_unique<Impl>(std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    show_progress)) {}
template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(
    std::span<memory::EPWorkspace<FoldType>> workspaces,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    bool show_progress)
    : m_impl(std::make_unique<Impl>(workspaces,
                                    std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    show_progress)) {}
template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::~EPMultiPass() = default;
template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(EPMultiPass&& other) noexcept = default;
template <SupportedFoldType FoldType>
EPMultiPass<FoldType>&
EPMultiPass<FoldType>::operator=(EPMultiPass&& other) noexcept = default;

template <SupportedFoldType FoldType>
void EPMultiPass<FoldType>::execute(std::span<const float> ts_e,
                                    std::span<const float> ts_v,
                                    const std::filesystem::path& outdir,
                                    std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}

// Explicit instantiation
template class EPMultiPass<float>;
template class EPMultiPass<ComplexType>;

} // namespace loki::algorithms