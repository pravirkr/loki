#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/core/dynamic.hpp"
#include "loki/exceptions.hpp"
#include "loki/progress.hpp"
#include "loki/prune_mask.hpp"
#include "loki/psr_utils.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::algorithms {

namespace {

// Number of individual harvests echoed to the log at info level per run;
// beyond this only per-level summaries are emitted.
constexpr SizeType kMaxHarvestLogLines = 16;

template <SupportedFoldType FoldType> class PruneImpl {
public:
    // External workspace constructor only
    PruneImpl(memory::EPWorkspace<FoldType>& workspace,
              search::PulsarSearchConfig cfg,
              std::span<const float> threshold_scheme,
              SizeType max_sugg,
              SizeType batch_size,
              SizeType branch_max,
              std::string_view poly_basis,
              const PruneRFIConfig& rfi_config,
              const GridMask& mask_base)
        : m_workspace_ptr(&workspace),
          m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_branch_max(branch_max),
          m_poly_basis(poly_basis),
          m_total_levels(m_threshold_scheme.size()),
          m_rfi(&rfi_config),
          m_mask_base(&mask_base) {
        error_check::check_less_equal(m_cfg.get_nparams(), 5,
                                      "Pruning not supported for nparams > 5.");
        m_prune_funcs = core::create_prune_dp_functs<FoldType>(
            m_poly_basis, m_ffa_plan.get_param_counts().back(),
            m_ffa_plan.get_dparams_actual().back(),
            m_ffa_plan.get_nsegments().back(),
            m_ffa_plan.get_tsegments().back(), m_cfg, m_batch_size,
            m_branch_max);
        const auto n_params = m_cfg.get_nparams();
        error_check::check_equal(
            m_mask_base->get_n_accel(),
            m_ffa_plan.get_param_counts().back()[n_params - 2],
            "PruneImpl: mask acceleration grid does not match the FFA plan");
        error_check::check_equal(
            m_mask_base->get_n_freq(),
            m_ffa_plan.get_param_counts().back()[n_params - 1],
            "PruneImpl: mask frequency grid does not match the FFA plan");
        m_leaf_scratch.resize(m_workspace_ptr->world_tree.get_leaves_stride());
    }

    ~PruneImpl()                           = default;
    PruneImpl(const PruneImpl&)            = delete;
    PruneImpl& operator=(const PruneImpl&) = delete;
    PruneImpl(PruneImpl&&)                 = delete;
    PruneImpl& operator=(PruneImpl&&)      = delete;

    [[nodiscard]] float get_memory_usage_gib() const noexcept {
        return get_workspace().get_memory_usage_gib() +
               m_prune_funcs->get_irfft_scratch_memory_gib();
    }

    void execute(std::span<const FoldType> ffa_fold,
                 SizeType ref_seg,
                 std::span<const SizeType> ascend_levels,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file,
                 progress::MultiprocessProgressTracker* tracker,
                 int task_id,
                 bool show_progress) {
        const std::string run_name =
            std::format("{:03d}_{:02d}", ref_seg, task_id);

        const auto& ws                 = get_workspace();
        const auto memory_tree_gb      = ws.world_tree.get_memory_usage_gib();
        const auto memory_workspace_gb = ws.prune.get_memory_usage_gib();
        const auto memory_irfft_scratch_gb =
            m_prune_funcs->get_irfft_scratch_memory_gib();
        const auto memory_total_gb = get_memory_usage_gib();
        spdlog::info(
            "Pruning run {:03d}: Memory Usage: {:.2f} GB total "
            "({:.2f} GB tree + {:.2f} GB workspace + {:.3f} GB irfft scratch)",
            ref_seg, memory_total_gb, memory_tree_gb, memory_workspace_gb,
            memory_irfft_scratch_gb);

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
            // Should we reintegrate survivors at this level?
            if (!ascend_levels.empty() &&
                std::ranges::find(ascend_levels, m_prune_level) !=
                    ascend_levels.end()) {
                ascend_survivors_batched(ffa_fold);
            }
            if (bar) {
                bar->set_score(ws.world_tree.get_score_max());
                bar->set_leaves(ws.world_tree.get_size_lb());
                bar->set_progress(iter + 1);
            }
        }

        // Final ascend if needed
        if (ascend_levels.empty() ||
            std::ranges::find(ascend_levels, m_prune_level) ==
                ascend_levels.end()) {
            ascend_survivors_batched(ffa_fold);
        }

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

    // RFI control (see PruneRFIConfig). The base mask is shared (read-only)
    // across runs; m_mask is the per-run overlay that also accumulates the
    // windows of harvested candidates.
    const PruneRFIConfig* m_rfi;
    const GridMask* m_mask_base;
    GridMask m_mask;
    cands::HarvestBuffer<FoldType> m_harvest;
    SizeType m_n_harvested_total{}; ///< Includes harvests beyond the cap
    std::vector<double> m_leaf_scratch;

    [[nodiscard]] memory::EPWorkspace<FoldType>& get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::EPWorkspace<FoldType>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    /// Duration of the data covered by the middle-out scheme (seconds).
    [[nodiscard]] double get_tobs_scheme() const noexcept {
        return static_cast<double>(m_ffa_plan.get_nsegments().back()) *
               m_ffa_plan.get_tsegments().back();
    }

    [[nodiscard]] bool veto_enabled_at_level(SizeType level) const noexcept {
        return m_rfi->impulsive_veto && level >= m_rfi->impulsive_min_level;
    }

    [[nodiscard]] float get_harvest_threshold(SizeType level) const noexcept {
        if (m_rfi->harvest_scheme.empty() ||
            level > m_rfi->harvest_scheme.size()) {
            return kHarvestDisabled;
        }
        return m_rfi->harvest_scheme[level - 1];
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

        // Per-run RFI state: start from the shared static mask
        m_mask.assign(*m_mask_base);
        m_harvest = cands::HarvestBuffer<FoldType>(
            world_tree.get_leaves_stride(), world_tree.get_folds_stride(),
            m_rfi->harvest_store_folds);
        m_n_harvested_total = 0;
        spdlog::info("Pruning run {:03d}: initialized", ref_seg);

        // Initialize the world tree with the first segment
        const auto fold_segment =
            m_prune_funcs->load_segment(ffa_fold, m_snail_scheme.get_ref_idx());
        const auto coord_init = m_snail_scheme.get_coord(m_prune_level);
        const auto n_leaves   = m_ffa_plan.get_ncoords().back();
        m_prune_funcs->seed(fold_segment, ws.seed_leaves, ws.seed_scores,
                            coord_init);
        SizeType n_seeds_masked = 0;
        if (m_mask.empty()) {
            world_tree.add_initial(ws.seed_leaves, fold_segment, ws.seed_scores,
                                   n_leaves);
        } else {
            // Seed leaf i sits on base-grid cell i: drop the masked ones.
            const auto n_keep =
                m_mask.select_seeds(ws.seed_keep_indices, n_leaves);
            n_seeds_masked = n_leaves - n_keep;
            world_tree.add_initial_scattered(ws.seed_leaves, fold_segment,
                                             ws.seed_scores,
                                             ws.seed_keep_indices, n_keep);
            spdlog::info("Pruning run {:03d}: pulsar mask removed {} of {} "
                         "seeds ({} grid cells masked)",
                         ref_seg, n_seeds_masked, n_leaves, m_mask.count());
        }

        // Initialize the prune stats
        m_pstats = cands::PruneStatsCollection();
        const cands::PruneStats pstats_cur{
            .level           = m_prune_level,
            .seg_idx         = m_snail_scheme.get_segment_idx(m_prune_level),
            .threshold       = 0,
            .score_min       = world_tree.get_score_min(),
            .score_max       = world_tree.get_score_max(),
            .n_branches      = n_leaves,
            .n_leaves        = n_leaves,
            .n_leaves_phy    = world_tree.get_size(),
            .n_leaves_surv   = world_tree.get_size(),
            .n_leaves_masked = n_seeds_masked,
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
        spdlog::info("Ascending survivors at level {}", m_prune_level);
        auto& prune_ws       = ws.prune;
        const auto coord_mid = m_snail_scheme.get_coord(m_prune_level);
        const auto segment_coords_so_far =
            m_snail_scheme.get_segment_coords_so_far(m_prune_level);
        const auto& idx_segments   = segment_coords_so_far.first;
        const auto& coord_segments = segment_coords_so_far.second;
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
        if (m_rfi->has_harvest()) {
            result_writer.write_run_harvest(
                run_name, m_harvest, m_cfg.get_nparams(), m_n_harvested_total);
        }
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
        const auto n_biases =
            static_cast<double>(m_cfg.get_n_boxcar_kadane_biases());
        const auto conservative_tile =
            static_cast<double>(m_cfg.get_use_conservative_tile());

        auto score_flops = [&](double n_leaves) {
            return n_leaves *
                   ((3.0 * nbins) +
                    (n_widths * ((2.0 * nbins) + (nbins / 4) + 11.0)));
        };
        auto score_flops_kadane = [&](double n_leaves) {
            return n_leaves * (((3.0 * nbins) + 1.0) +
                               (n_biases * ((4.0 * nbins) + 11.0)));
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

            // n_leaves_phy is the post-mask count: masked leaves are resolved
            // but never shift-added or scored.
            total_flops += branch_flops(n_branches, n_leaves);
            total_flops +=
                resolve_flops(static_cast<double>(stats.n_leaves_resolved));
            total_flops += shift_add_flops(n_leaves_phy);
            total_flops += irfft_flops(n_leaves_phy);
            if (m_cfg.get_use_boxcar_kadane()) {
                total_flops += score_flops_kadane(n_leaves_phy);
            } else {
                total_flops += score_flops(n_leaves_phy);
            }
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

        cands::PruneIterationStats stats;
        const auto seg_idx_cur = m_snail_scheme.get_segment_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches = world_tree.get_size_old();

        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats);

        // Finalize: make new region active, defragment for contiguous access.
        world_tree.finalize_in_place_update();

        // Update statistics
        stats.norm_scores(world_tree.get_size());
        const cands::PruneStats pstats_cur{
            .level             = m_prune_level,
            .seg_idx           = seg_idx_cur,
            .threshold         = threshold,
            .score_min         = stats.score_min,
            .score_max         = stats.score_max,
            .n_branches        = n_branches,
            .n_leaves          = stats.n_leaves,
            .n_leaves_resolved = stats.n_leaves_resolved,
            .n_leaves_phy      = stats.n_leaves_phy,
            .n_leaves_surv     = world_tree.get_size(),
            .n_leaves_masked   = stats.n_leaves_masked,
            .n_leaves_vetoed   = stats.n_leaves_vetoed,
            .n_harvested       = stats.n_harvested,
        };
        // Write stats to log
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary();
        log.close();
        m_pstats.update_stats(pstats_cur, stats.batch_timers);
        if (stats.n_harvested > 0) {
            spdlog::info("Pruning level {:3d}: harvested {} candidate(s) "
                         "(run total {}, recorded {}, mask cells {})",
                         m_prune_level, stats.n_harvested, m_n_harvested_total,
                         m_harvest.size(), m_mask.count());
        }

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
                                   cands::PruneIterationStats& stats) {
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

        // RFI-control state for this level
        const auto leaves_stride     = world_tree.get_leaves_stride();
        const bool veto_active       = veto_enabled_at_level(m_prune_level);
        const auto harvest_threshold = get_harvest_threshold(m_prune_level);
        const bool harvest_active    = is_harvest_enabled(harvest_threshold);
        // The post-score pass is only needed when it can remove leaves; the
        // mask itself is applied before shift-add (and only grows here through
        // harvesting).
        const bool post_score_active = veto_active || harvest_active;

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
            stats.n_leaves_resolved += n_leaves_after_validation;

            // Pulsar mask: drop leaves resolving to a masked base-grid cell
            // before the expensive shift-add and scoring stages.
            auto n_leaves_active = n_leaves_after_validation;
            if (!m_mask.empty()) {
                timer.start();
                n_leaves_active = m_mask.filter_resolved(
                    prune_ws.branched_leaves, prune_ws.branched_indices,
                    prune_ws.branched_param_idx, prune_ws.branched_phase_shift,
                    leaves_stride, n_leaves_after_validation);
                stats.n_leaves_masked +=
                    n_leaves_after_validation - n_leaves_active;
                stats.batch_timers["rfi"] += timer.stop();
            }
            stats.n_leaves_phy += n_leaves_active;
            if (n_leaves_active == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }

            // Stage-consistency veto needs the parent score of each leaf;
            // gather it now while branched_indices still holds tree origins
            // (score_and_filter overwrites them with passing local ids).
            const auto physical_start_idx = world_tree.get_physical_start_idx();
            const auto capacity           = world_tree.get_capacity();
            if (veto_active) {
                timer.start();
                gather_parent_scores(
                    world_tree.get_scores(), prune_ws.branched_indices,
                    prune_ws.branched_parent_scores, n_leaves_active,
                    physical_start_idx, capacity);
                stats.batch_timers["rfi"] += timer.stop();
            }

            // Load, shift, add (Map branched_itree to physical indices)
            timer.start();
            m_prune_funcs->shift_add(
                world_tree.get_folds(), prune_ws.branched_indices,
                ffa_fold_segment, prune_ws.branched_param_idx,
                prune_ws.branched_phase_shift, prune_ws.branched_folds,
                n_leaves_active, physical_start_idx, capacity);
            stats.batch_timers["shift_add"] += timer.stop();

            // Score and filter
            timer.start();
            SizeType n_leaves_passing = m_prune_funcs->score_and_filter(
                prune_ws.branched_folds, prune_ws.branched_scores,
                prune_ws.branched_indices, current_threshold, n_leaves_active);
            auto branched_scores_span =
                std::span<const float>(prune_ws.branched_scores)
                    .first(n_leaves_active);
            const auto [min_it, max_it] =
                std::ranges::minmax_element(branched_scores_span);
            stats.score_min = std::min(stats.score_min, *min_it);
            stats.score_max = std::max(stats.score_max, *max_it);
            stats.batch_timers["score"] += timer.stop();

            // Veto and early harvest on the passing list (index-only work)
            if (post_score_active && n_leaves_passing > 0) {
                timer.start();
                n_leaves_passing = apply_rfi_post_score(
                    prune_ws, coord_cur, seg_idx_cur, veto_active,
                    harvest_threshold, n_leaves_passing, stats);
                stats.batch_timers["rfi"] += timer.stop();
            }

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

    // Gather the tree score of the parent of each branched leaf. Uses the same
    // logical -> physical circular index mapping as shift_add.
    static void gather_parent_scores(std::span<const float> tree_scores,
                                     std::span<const SizeType> origins,
                                     std::span<float> parent_scores,
                                     SizeType n_leaves,
                                     SizeType physical_start_idx,
                                     SizeType capacity) noexcept {
        for (SizeType i = 0; i < n_leaves; ++i) {
            const auto logical = origins[i] + physical_start_idx;
            const auto physical =
                logical < capacity ? logical : logical - capacity;
            parent_scores[i] = tree_scores[physical];
        }
    }

    /**
     * @brief Veto and early-harvest over passing leaves.
     *
     * @details Operates on `branched_indices[0..n_passing)`, which after
     * score_and_filter() holds the local ids of the leaves above threshold.
     * Leaves are sorted by score descending so the brightest member of a
     * cluster is harvested. Then: (1) drop if the cell is already masked,
     * (2) drop if the stage-consistency veto fires, (3) harvest leaves at or
     * above the harvest threshold. The index list is compacted in place so that
     * transform()/add_batch_scattered() only see survivors.
     *
     * @return Number of surviving leaves.
     */
    SizeType apply_rfi_post_score(memory::PruneWorkspace<FoldType>& prune_ws,
                                  std::pair<double, double> coord_cur,
                                  SizeType seg_idx_cur,
                                  bool veto_active,
                                  float harvest_threshold,
                                  SizeType n_passing,
                                  cands::PruneIterationStats& stats) {
        auto& indices         = prune_ws.branched_indices;
        const auto& scores    = prune_ws.branched_scores;
        const auto& parents   = prune_ws.branched_parent_scores;
        const auto& param_idx = prune_ws.branched_param_idx;
        const auto n_seg      = m_prune_level; // segments in the parent
        const bool harvest_on = is_harvest_enabled(harvest_threshold);
        const auto n_cells    = m_mask.get_n_cells();

        auto first = indices.begin();
        auto last  = first + static_cast<std::ptrdiff_t>(n_passing);
        std::stable_sort(first, last, [&](SizeType lhs, SizeType rhs) {
            return scores[lhs] > scores[rhs];
        });

        auto cell_masked = [&](SizeType local_idx) {
            if (!harvest_on) {
                return false;
            }
            const auto cell = param_idx[local_idx];
            return cell < n_cells && m_mask.is_masked(cell);
        };

        // (1) already-masked cells, (2) impulsive veto
        SizeType write = 0;
        for (SizeType k = 0; k < n_passing; ++k) {
            const auto idx = indices[k];
            if (cell_masked(idx)) {
                continue;
            }
            if (veto_active &&
                is_impulsive_segment(scores[idx], parents[idx], n_seg,
                                     m_rfi->impulsive_kappa,
                                     m_rfi->impulsive_min_snr)) {
                ++stats.n_leaves_vetoed;
                continue;
            }
            indices[write++] = idx;
        }
        n_passing = write;

        // (3) early harvest (mask is live, so same-batch neighbours of a
        // just-harvested source are dropped without a second recording)
        write = 0;
        for (SizeType k = 0; k < n_passing; ++k) {
            const auto idx   = indices[k];
            const auto score = scores[idx];
            if (cell_masked(idx)) {
                continue;
            }
            if (harvest_on && score >= harvest_threshold) {
                harvest_leaf(prune_ws, idx, score, coord_cur, seg_idx_cur);
                ++stats.n_harvested;
                continue;
            }
            indices[write++] = idx;
        }
        return write;
    }

    /// Record a harvested leaf from the branch workspace and extend the mask.
    void harvest_leaf(const memory::PruneWorkspace<FoldType>& prune_ws,
                      SizeType idx,
                      float score,
                      std::pair<double, double> coord_cur,
                      SizeType seg_idx_cur) {
        const auto leaves_stride = prune_ws.leaves_stride;
        const auto folds_stride  = prune_ws.folds_stride;
        const auto leaf_src = std::span<const double>(prune_ws.branched_leaves)
                                  .subspan(idx * leaves_stride, leaves_stride);
        const auto fold_src = std::span<const FoldType>(prune_ws.branched_folds)
                                  .subspan(idx * folds_stride, folds_stride);
        harvest_candidate(leaf_src, fold_src, score, coord_cur, seg_idx_cur);
    }

    /// Convert a leaf to physical parameters, record it, and widen the mask.
    void harvest_candidate(std::span<const double> leaf,
                           std::span<const FoldType> fold,
                           float score,
                           std::pair<double, double> coord_cur,
                           SizeType seg_idx_cur) {
        constexpr SizeType kParamStride = 2U;
        const auto n_params             = m_cfg.get_nparams();

        std::ranges::copy(leaf, m_leaf_scratch.begin());
        m_prune_funcs->report(m_leaf_scratch, coord_cur, 1);
        const auto a = m_leaf_scratch[(n_params - 2) * kParamStride];
        const auto da =
            std::abs(m_leaf_scratch[((n_params - 2) * kParamStride) + 1]);
        const auto f = m_leaf_scratch[(n_params - 1) * kParamStride];
        const auto df =
            std::abs(m_leaf_scratch[((n_params - 1) * kParamStride) + 1]);
        const auto t_ref = coord_cur.first;

        const auto window = make_harvest_window(
            f, a, df, da, t_ref, get_tobs_scheme(), m_rfi->harvest_mask_ntiles);
        m_mask.add_window(window, m_rfi->n_harmonics);
        ++m_n_harvested_total;

        if (m_harvest.size() < m_rfi->max_harvests) {
            m_harvest.push(m_leaf_scratch, fold, score, m_prune_level,
                           seg_idx_cur, t_ref);
            if (m_harvest.size() <= kMaxHarvestLogLines) {
                spdlog::info("Pruning level {:3d}: harvested candidate "
                             "score {:.2f}, f = {:.9f} Hz, a = {:.4e} m/s^2 "
                             "(t_ref = {:.1f} s)",
                             m_prune_level, score, f, a, t_ref);
            }
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
         std::span<const SizeType> ascend_levels,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         bool show_progress,
         PruneRFIConfig rfi_config)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_ascend_levels(ascend_levels.begin(), ascend_levels.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_show_progress(show_progress),
          m_rfi_config(std::move(rfi_config)),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()) {
        // Create branching pattern and branch max
        m_branching_pattern   = m_ffa_plan.get_branching_pattern(m_poly_basis);
        const auto branch_max = *std::ranges::max_element(m_branching_pattern);
        m_branch_max =
            std::max(static_cast<SizeType>(std::ceil(branch_max * 2)), 32UL);

        // Allocate workspaces
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        setup_rfi_control(nsegments);
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
         std::span<const SizeType> ascend_levels,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         bool show_progress,
         PruneRFIConfig rfi_config)
        : m_workspaces_view(external_workspaces),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_ascend_levels(ascend_levels.begin(), ascend_levels.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_show_progress(show_progress),
          m_rfi_config(std::move(rfi_config)),
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
        setup_rfi_control(nsegments);
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
            utils::determine_ref_segs(nsegments, m_n_runs, m_ref_segs);
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
                              m_threshold_scheme, m_rfi_config);
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
    std::vector<SizeType> m_ascend_levels;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string m_poly_basis;
    bool m_show_progress;
    PruneRFIConfig m_rfi_config;

    plans::FFAPlan<FoldType> m_ffa_plan;
    int m_nthreads;
    std::vector<double> m_branching_pattern;
    SizeType m_branch_max{0};
    // Static pulsar mask on the FFA base grid, shared read-only by all runs
    GridMask m_mask_base;

    // Safely get the workspace for a specific thread index
    [[nodiscard]] memory::EPWorkspace<FoldType>&
    get_thread_workspace(SizeType thread_idx = 0) noexcept {
        return m_workspaces_view[thread_idx];
    }

    // Validate the RFI configuration and rasterise the static pulsar mask
    // onto the (accel, freq) grid that resolve() snaps to.
    void setup_rfi_control(SizeType nsegments) {
        m_rfi_config.validate(nsegments);
        const auto n_params = m_cfg.get_nparams();
        error_check::check_greater_equal(
            n_params, SizeType{2},
            "EPMultiPass: pruning requires at least 2 parameters");
        const auto& counts = m_ffa_plan.get_param_counts().back();
        const auto limits  = m_cfg.get_param_limits();
        m_mask_base = GridMask(limits[n_params - 2], counts[n_params - 2],
                               limits[n_params - 1], counts[n_params - 1]);
        m_mask_base.add_windows(m_rfi_config.pulsar_mask,
                                m_rfi_config.n_harmonics);
        if (m_rfi_config.is_active()) {
            spdlog::info(
                "EPMultiPass: RFI control: {} mask window(s) ({} harmonics) "
                "-> {} of {} grid cells masked; harvesting {}; impulsive "
                "veto {}",
                m_rfi_config.pulsar_mask.size(), m_rfi_config.n_harmonics,
                m_mask_base.count(), m_mask_base.get_n_cells(),
                m_rfi_config.has_harvest() ? "on" : "off",
                m_rfi_config.impulsive_veto ? "on" : "off");
        }
    }

    void execute_single_threaded(std::span<const FoldType> ffa_fold,
                                 std::span<const SizeType> ref_segs,
                                 const std::filesystem::path& outdir,
                                 const std::filesystem::path& log_file,
                                 const std::filesystem::path& result_file) {
        auto& ws   = get_thread_workspace(0);
        auto prune = PruneImpl<FoldType>(
            ws, m_cfg, m_threshold_scheme, m_max_sugg, m_batch_size,
            m_branch_max, m_poly_basis, m_rfi_config, m_mask_base);
        for (const auto ref_seg : ref_segs) {
            prune.execute(ffa_fold, ref_seg, m_ascend_levels, outdir, log_file,
                          result_file,
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
                auto prune            = PruneImpl<FoldType>(
                    ws, m_cfg, m_threshold_scheme, m_max_sugg, m_batch_size,
                    m_branch_max, m_poly_basis, m_rfi_config, m_mask_base);

                prune.execute(
                    ffa_fold, ref_seg, m_ascend_levels, outdir,
                    /*log_file=*/std::nullopt,
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
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    bool show_progress,
    PruneRFIConfig rfi_config)
    : m_impl(std::make_unique<Impl>(std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    ascend_levels,
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    show_progress,
                                    std::move(rfi_config))) {}
template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(
    std::span<memory::EPWorkspace<FoldType>> workspaces,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    bool show_progress,
    PruneRFIConfig rfi_config)
    : m_impl(std::make_unique<Impl>(workspaces,
                                    std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    ascend_levels,
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    show_progress,
                                    std::move(rfi_config))) {}
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