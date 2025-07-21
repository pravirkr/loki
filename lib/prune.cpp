#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <utility>

#include <BS_thread_pool.hpp>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/core/dynamic.hpp"
#include "loki/progress.hpp"
#include "loki/psr_utils.hpp"
#include "loki/timing.hpp"
#include "loki/utils/suggestions.hpp"

namespace loki::algorithms {

template <typename FoldType> class PruningManager<FoldType>::Impl {
public:
    Impl(search::PulsarSearchConfig cfg,
         const std::vector<float>& threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         SizeType max_sugg,
         SizeType batch_size)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_nthreads(m_cfg.get_nthreads()),
          m_ffa_plan(m_cfg) {
        // Create appropriate FFA instance based on FoldType
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_the_ffa_complex = std::make_unique<algorithms::FFACOMPLEX>(m_cfg);
            m_ffa_fold.resize(m_ffa_plan.get_fold_size_complex());
        } else {
            m_the_ffa = std::make_unique<algorithms::FFA>(m_cfg);
            m_ffa_fold.resize(m_ffa_plan.get_fold_size());
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
                 std::string_view file_prefix,
                 std::string_view kind) {
        // Execute appropriate FFA instance
        spdlog::info("PruningManager::execute: Initializing with FFA");
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_the_ffa_complex->execute(ts_e, ts_v,
                                       std::span<ComplexType>(m_ffa_fold));
        } else {
            m_the_ffa->execute(ts_e, ts_v, std::span<float>(m_ffa_fold));
        }
        // Setup output files and directory
        const auto nsegments = m_ffa_plan.nsegments.back();
        const std::string filebase =
            std::format("{}_pruning_nstages_{}", file_prefix, nsegments);
        const auto log_file = outdir / std::format("{}_log.txt", filebase);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);

        // Create output directory
        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
        if (!std::filesystem::exists(outdir)) {
            throw std::runtime_error(
                std::format("PruningManager::execute: Failed to create output "
                            "directory '{}': {}",
                            outdir.string(), ec.message()));
        }

        // Determine ref_segs to process
        auto ref_segs_to_process = determine_ref_segs(nsegments);
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
            execute_single_threaded(ref_segs_to_process, outdir, log_file,
                                    result_file, kind);
        } else {
            execute_multi_threaded(ref_segs_to_process, outdir, log_file, kind);
            cands::merge_prune_result_files(outdir, log_file, result_file);
        }
        spdlog::info("Pruning complete. Results saved to {}",
                     result_file.string());
    }

private:
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    std::optional<SizeType> m_n_runs;
    std::optional<std::vector<SizeType>> m_ref_segs;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    int m_nthreads;

    // Type-specific FFA instances
    plans::FFAPlan m_ffa_plan;
    std::unique_ptr<algorithms::FFA> m_the_ffa;
    std::unique_ptr<algorithms::FFACOMPLEX> m_the_ffa_complex;
    std::vector<FoldType> m_ffa_fold;

    std::vector<SizeType> determine_ref_segs(SizeType nsegments) const {
        if (m_n_runs.has_value()) {
            // n_runs takes precedence over ref_segs
            const auto n_runs = m_n_runs.value();
            if (n_runs < 1 || n_runs > nsegments) {
                throw std::runtime_error(
                    std::format("n_runs must be between 1 and {}, got {}",
                                nsegments, n_runs));
            }

            std::vector<SizeType> ref_segs(n_runs);
            if (n_runs == 1) {
                ref_segs[0] = 0;
            } else {
                const auto denom = static_cast<double>(n_runs - 1);
                const auto max   = static_cast<double>(nsegments - 1);
                for (SizeType i = 0; i < n_runs; ++i) {
                    ref_segs[i] = static_cast<SizeType>(
                        std::round(static_cast<double>(i) * max / denom));
                }
            }
            return ref_segs;
        }
        if (m_ref_segs.has_value()) {
            return m_ref_segs.value();
        }
        throw std::runtime_error("Either n_runs or ref_segs must be provided");
    }

    void execute_single_threaded(const std::vector<SizeType>& ref_segs,
                                 const std::filesystem::path& outdir,
                                 const std::filesystem::path& log_file,
                                 const std::filesystem::path& result_file,
                                 std::string_view kind) {
        auto prune = Prune<FoldType>(m_ffa_plan, m_cfg, m_threshold_scheme,
                                     m_max_sugg, m_batch_size, kind);
        // Log detailed memory usage
        const auto memory_usage = prune.get_memory_usage();
        const auto memory_gb =
            static_cast<float>(memory_usage) / (1024.0F * 1024.0F * 1024.0F);
        spdlog::info("Pruning Memory Usage: {:.2f} GB", memory_gb);
        for (const auto ref_seg : ref_segs) {
            spdlog::info("Processing ref segment {} (single-threaded)",
                         ref_seg);
            prune.execute(m_ffa_fold, ref_seg, outdir, log_file, result_file,
                          /*tracker=*/nullptr, /*task_id=*/0);
        }
    }

    void execute_multi_threaded(const std::vector<SizeType>& ref_segs,
                                const std::filesystem::path& outdir,
                                const std::filesystem::path& log_file,
                                std::string_view kind) {
        // Create thread pool
        progress::MultiprocessProgressTracker tracker("Pruning tree");
        tracker.start();
        BS::thread_pool pool(m_nthreads);

        // Submit tasks for each ref_seg
        std::vector<std::future<void>> futures;
        futures.reserve(ref_segs.size());
        std::vector<int> task_ids;
        task_ids.reserve(ref_segs.size());

        const auto& ffa_plan = m_ffa_plan;
        const auto nsegments = ffa_plan.nsegments.back();
        for (const auto ref_seg : ref_segs) {
            const auto id = tracker.add_task(
                std::format("Pruning segment {:03d}", ref_seg), nsegments - 1,
                /*transient=*/true);
            task_ids.push_back(id);

            auto future = pool.submit_task(
                [this, ref_seg, outdir, kind, &tracker, id]() mutable {
                    // Create a Prune instance for each thread
                    auto prune =
                        Prune<FoldType>(m_ffa_plan, m_cfg, m_threshold_scheme,
                                        m_max_sugg, m_batch_size, kind);

                    prune.execute(
                        m_ffa_fold, ref_seg, outdir, /*log_file=*/std::nullopt,
                        /*result_file=*/std::nullopt, /*tracker=*/&tracker,
                        /*task_id=*/id);
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
        tracker.stop();

        // Log errors to file
        if (!errors.empty()) {
            std::ofstream log(log_file, std::ios::app);
            for (const auto& [ref_seg, error_msg] : errors) {
                log << std::format("Error processing ref_seg {}: {}\n", ref_seg,
                                   error_msg);
            }
            log.close();
            spdlog::error("{}", errors.back().second);
        }

        spdlog::info("All tasks completed and progress tracker stopped.");
    }
};

struct IterationStats {
    SizeType n_leaves     = 0;
    SizeType n_leaves_phy = 0;
    float score_min       = std::numeric_limits<float>::max();
    float score_max       = std::numeric_limits<float>::lowest();
    cands::TimerStats batch_timers;
};

template <typename FoldType> struct PruningWorkspace {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType max_batch_size;
    SizeType nparams;
    SizeType nbins;
    SizeType leaves_stride;
    SizeType folds_stride;

    std::vector<double> batch_leaves;
    std::vector<FoldType> batch_folds;
    std::vector<float> batch_scores;
    std::vector<SizeType> batch_isuggest;
    std::vector<SizeType> batch_passing_indices;

    PruningWorkspace(SizeType max_batch_size, SizeType nparams, SizeType nbins)
        : max_batch_size(max_batch_size),
          nparams(nparams),
          nbins(nbins),
          leaves_stride((nparams + 2) * kLeavesParamStride),
          folds_stride(2 * nbins),
          batch_leaves(max_batch_size * leaves_stride),
          batch_folds(max_batch_size * folds_stride),
          batch_scores(max_batch_size),
          batch_isuggest(max_batch_size),
          batch_passing_indices(max_batch_size) {}

    // Call this after filling batch_scores and batch_passing_indices
    void filter_batch(SizeType n_leaves_passing) noexcept {
        for (SizeType dst_idx = 0; dst_idx < n_leaves_passing; ++dst_idx) {
            const SizeType src_idx           = batch_passing_indices[dst_idx];
            batch_scores[dst_idx]            = batch_scores[src_idx];
            const SizeType leaves_src_offset = src_idx * leaves_stride;
            const SizeType leaves_dst_offset = dst_idx * leaves_stride;
            std::copy(batch_leaves.begin() + leaves_src_offset,
                      batch_leaves.begin() + leaves_src_offset + leaves_stride,
                      batch_leaves.begin() + leaves_dst_offset);
            const SizeType folds_src_offset = src_idx * folds_stride;
            const SizeType folds_dst_offset = dst_idx * folds_stride;
            std::copy(batch_folds.begin() + folds_src_offset,
                      batch_folds.begin() + folds_src_offset + folds_stride,
                      batch_folds.begin() + folds_dst_offset);
        }
    }

    SizeType get_memory_usage() const noexcept {
        return (batch_leaves.size() * sizeof(double)) +
               (batch_folds.size() * sizeof(FoldType)) +
               (batch_scores.size() * sizeof(float)) +
               (batch_isuggest.size() * sizeof(SizeType)) +
               (batch_passing_indices.size() * sizeof(SizeType));
    }
};

template <typename FoldType> class Prune<FoldType>::Impl {
public:
    Impl(plans::FFAPlan ffa_plan,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view kind)
        : m_ffa_plan(std::move(ffa_plan)),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_kind(kind) {
        // Allocate suggestion buffer
        m_suggestions = std::make_unique<utils::SuggestionTree<FoldType>>(
            m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins());

        // Allocate iteration workspace
        const auto max_batch_size = m_batch_size * m_cfg.get_branch_max();
        m_pruning_workspace = std::make_unique<PruningWorkspace<FoldType>>(
            max_batch_size, m_cfg.get_nparams(), m_cfg.get_nbins());

        // Setup pruning functions
        setup_pruning();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_memory_usage() const noexcept {
        return m_pruning_workspace->get_memory_usage() +
               m_suggestions->get_memory_usage();
    }

    void execute(std::span<const FoldType> ffa_fold,
                 SizeType ref_seg,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file,
                 progress::MultiprocessProgressTracker* tracker,
                 int task_id) {
        const std::string run_name =
            std::format("{:03d}_{:02d}", ref_seg, task_id);
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / std::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / std::format("tmp_{}_results.h5", run_name));
        std::ofstream log(actual_log_file, std::ios::app);
        log << std::format("Pruning log for ref segment: {}\n", ref_seg);
        log.close();

        const auto nsegments = m_ffa_plan.nsegments.back();
        progress::ProgressGuard progress_guard(true);
        progress::ProgressTracker bar(
            std::format("Pruning segment {:03d}", ref_seg), nsegments - 1,
            tracker, task_id);

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
            bar.set_score(m_suggestions->get_score_max());
            bar.set_leaves(m_suggestions->get_nsugg_lb());
            bar.set_progress(iter + 1);
        }

        // Transform the suggestion params to middle of the data
        const auto delta_t = m_scheme->get_delta(m_prune_level);

        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        result_writer.write_run_results(run_name, m_scheme->get_data(),
                                        m_suggestions->get_transformed(delta_t),
                                        m_suggestions->get_scores(),
                                        m_suggestions->get_nsugg(), *m_pstats);

        // Final log entries
        std::ofstream final_log(actual_log_file, std::ios::app);
        final_log << std::format("Pruning complete for ref segment {}\n",
                                 ref_seg);
        final_log << std::format("Time: {}\n", m_pstats->get_timer_summary());
        final_log.close();
        spdlog::info("Pruning complete for ref segment {}", ref_seg);
        spdlog::info("Pruning stats: {}", m_pstats->get_stats_summary());
        spdlog::info("Pruning time: {}", m_pstats->get_concise_timer_summary());
    }

private:
    plans::FFAPlan m_ffa_plan;
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string_view m_kind;

    bool m_prune_complete{false};
    SizeType m_prune_level{};
    std::unique_ptr<psr_utils::SnailScheme> m_scheme;
    std::unique_ptr<core::PruneTaylorDPFuncts<FoldType>> m_prune_funcs;
    std::unique_ptr<cands::PruneStatsCollection> m_pstats;

    // A single, in-place (circular) suggestion buffer
    std::unique_ptr<utils::SuggestionTree<FoldType>> m_suggestions;
    std::unique_ptr<PruningWorkspace<FoldType>> m_pruning_workspace;

    void initialize(std::span<const FoldType> ffa_fold,
                    SizeType ref_seg,
                    const std::filesystem::path& log_file) {
        timing::ScopeTimer timer("Prune::initialize");
        // Reset the suggestion buffer state
        m_suggestions->reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.nsegments.back();
        const auto tseg      = m_ffa_plan.tsegments.back();
        m_scheme =
            std::make_unique<psr_utils::SnailScheme>(nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;
        spdlog::info("Initializing pruning run for ref segment: {}",
                     m_scheme->get_ref_idx());

        // Initialize the suggestions with the first segment
        const auto fold_segment =
            m_prune_funcs->load(ffa_fold, m_scheme->get_ref_idx());
        const auto coord_init = m_scheme->get_coord(m_prune_level);
        m_prune_funcs->suggest(fold_segment, coord_init, *m_suggestions);

        // Initialize the prune stats
        m_pstats = std::make_unique<cands::PruneStatsCollection>();
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = m_scheme->get_idx(m_prune_level),
            .threshold     = 0,
            .score_min     = m_suggestions->get_score_min(),
            .score_max     = m_suggestions->get_score_max(),
            .n_branches    = m_suggestions->get_nsugg(),
            .n_leaves      = m_suggestions->get_nsugg(),
            .n_leaves_phy  = m_suggestions->get_nsugg(),
            .n_leaves_surv = m_suggestions->get_nsugg(),
        };
        m_pstats->update_stats(pstats_cur);

        // Write the initial prune stats to the log file
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary();
        log.close();
    }

    void execute_iteration(std::span<const FoldType> ffa_fold,
                           const std::filesystem::path& log_file) {
        if (m_prune_complete) {
            return;
        }
        ++m_prune_level;
        if (m_prune_level > m_threshold_scheme.size()) {
            throw std::runtime_error(
                std::format("Pruning complete - exceeded threshold scheme "
                            "length at level {}",
                            m_prune_level));
        }
        // Prepare for in-place update: mark start of write region, reset size
        // for new suggestions.
        m_suggestions->prepare_for_in_place_update();

        IterationStats stats;
        const auto seg_idx_cur = m_scheme->get_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches = m_suggestions->get_nsugg_old();

        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats);

        // Finalize: make new region active, defragment for contiguous access.
        m_suggestions->finalize_in_place_update();

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
            .n_leaves_surv = m_suggestions->get_nsugg(),
        };
        // Write stats to log
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary();
        log.close();
        m_pstats->update_stats(pstats_cur, stats.batch_timers);

        // Check if no survivors
        if (m_suggestions->get_nsugg() == 0) {
            m_prune_complete = true;
            spdlog::info("Pruning complete at level {} - no survivors",
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
        // Get coordinates
        const auto coord_init  = m_scheme->get_coord(0);
        const auto coord_cur   = m_scheme->get_coord(m_prune_level);
        const auto coord_prev  = m_scheme->get_coord(m_prune_level - 1);
        const auto coord_add   = m_scheme->get_seg_coord(m_prune_level);
        const auto coord_valid = m_scheme->get_valid(m_prune_level);

        // Load fold segment for current level
        const auto ffa_fold_segment =
            m_prune_funcs->load(ffa_fold, seg_idx_cur);

        auto current_threshold = threshold;
        const auto trans_matrix =
            m_prune_funcs->get_transform_matrix(coord_cur, coord_prev);
        const auto validation_params =
            m_prune_funcs->get_validation_params(coord_valid);
        const bool validation_check = false;

        const auto n_branches = m_suggestions->get_nsugg_old();
        const auto n_params   = m_cfg.get_nparams();
        const auto batch_size =
            std::max(1UL, std::min(m_batch_size, n_branches));

        timing::SimpleTimer timer;

        // Process branches in batches
        for (SizeType i_batch_start = 0; i_batch_start < n_branches;
             i_batch_start += batch_size) {
            const auto i_batch_end =
                std::min(i_batch_start + batch_size, n_branches);
            const auto current_batch_size = i_batch_end - i_batch_start;

            // Branch
            timer.start();
            auto batch_leaves_span = m_suggestions->get_leaves_span(
                i_batch_start, current_batch_size);
            const auto batch_leaf_origins = m_prune_funcs->branch(
                batch_leaves_span, coord_cur, m_pruning_workspace->batch_leaves,
                current_batch_size, n_params);
            const auto n_leaves_batch = batch_leaf_origins.size();
            stats.batch_timers["branch"] += timer.stop();
            stats.n_leaves += n_leaves_batch;
            if (n_leaves_batch == 0) {
                m_suggestions->advance_read_consumed(current_batch_size);
                continue;
            }
            if (n_leaves_batch > m_pruning_workspace->max_batch_size) {
                throw std::runtime_error(std::format(
                    "Branch factor exceeded workspace size: n_leaves_batch={} "
                    "> max_batch_size={}",
                    n_leaves_batch, m_pruning_workspace->max_batch_size));
            }

            // Validation
            timer.start();
            auto n_leaves_after_validation = n_leaves_batch;
            if (validation_check) {
                n_leaves_after_validation = m_prune_funcs->validate(
                    m_pruning_workspace->batch_leaves, coord_valid,
                    validation_params, n_leaves_batch, n_params);
            }
            stats.batch_timers["validate"] += timer.stop();
            stats.n_leaves_phy += n_leaves_after_validation;
            if (n_leaves_after_validation == 0) {
                m_suggestions->advance_read_consumed(current_batch_size);
                continue;
            }

            // Resolve
            timer.start();
            const auto [batch_param_idx, batch_phase_shift] =
                m_prune_funcs->resolve(m_pruning_workspace->batch_leaves,
                                       coord_add, coord_init,
                                       n_leaves_after_validation, n_params);
            stats.batch_timers["resolve"] += timer.stop();

            // Load, shift, add (Map batch_leaf_origins to global indices)
            timer.start();
            auto batch_isuggest_span =
                std::span(m_pruning_workspace->batch_isuggest)
                    .first(n_leaves_after_validation);
            for (SizeType i = 0; i < n_leaves_after_validation; ++i) {
                batch_isuggest_span[i] = batch_leaf_origins[i] + i_batch_start;
            }
            m_prune_funcs->load_shift_add(
                m_suggestions->get_folds(), batch_isuggest_span,
                ffa_fold_segment, batch_param_idx, batch_phase_shift,
                m_pruning_workspace->batch_folds, n_leaves_after_validation);
            stats.batch_timers["shift_add"] += timer.stop();

            // Score
            timer.start();
            auto batch_folds_span =
                std::span<FoldType>(m_pruning_workspace->batch_folds)
                    .first(n_leaves_after_validation *
                           m_pruning_workspace->folds_stride);
            auto batch_scores_span =
                std::span(m_pruning_workspace->batch_scores)
                    .first(n_leaves_after_validation);
            m_prune_funcs->score(batch_folds_span, batch_scores_span,
                                 n_leaves_after_validation);
            const auto [min_it, max_it] =
                std::ranges::minmax_element(batch_scores_span);
            stats.score_min = std::min(stats.score_min, *min_it);
            stats.score_max = std::max(stats.score_max, *max_it);
            stats.batch_timers["score"] += timer.stop();

            // Thresholding & filtering (direct memory operations)
            timer.start();
            auto batch_passing_indices_span =
                std::span(m_pruning_workspace->batch_passing_indices)
                    .first(n_leaves_after_validation);
            SizeType n_leaves_passing = 0;
            for (SizeType i = 0; i < n_leaves_after_validation; ++i) {
                if (batch_scores_span[i] >= threshold) {
                    batch_passing_indices_span[n_leaves_passing] = i;
                    ++n_leaves_passing;
                }
            }
            stats.batch_timers["filter"] += timer.stop();

            if (n_leaves_passing == 0) {
                m_suggestions->advance_read_consumed(current_batch_size);
                continue;
            }
            if (n_leaves_passing > m_pruning_workspace->max_batch_size) {
                throw std::runtime_error(std::format(
                    "n_leaves_passing={} > max_batch_size={}", n_leaves_passing,
                    m_pruning_workspace->max_batch_size));
            }

            timer.start();
            if (n_leaves_passing != n_leaves_after_validation) {
                m_pruning_workspace->filter_batch(n_leaves_passing);
            }
            stats.batch_timers["filter"] += timer.stop();

            // Transform
            timer.start();
            m_prune_funcs->transform(m_pruning_workspace->batch_leaves,
                                     coord_cur, trans_matrix, n_leaves_passing,
                                     n_params);
            stats.batch_timers["transform"] += timer.stop();

            // Add batch to output suggestions
            timer.start();
            current_threshold = m_suggestions->add_batch(
                m_pruning_workspace->batch_leaves,
                m_pruning_workspace->batch_folds, batch_scores_span,
                current_threshold, n_leaves_passing);
            stats.batch_timers["batch_add"] += timer.stop();
            // Notify the buffer that a batch of the old suggestions has been
            // consumed
            m_suggestions->advance_read_consumed(current_batch_size);
        }
    }

    void setup_pruning() {
        if (m_cfg.get_nparams() > 4) {
            throw std::runtime_error(
                std::format("Pruning not supported for nparams > 4."));
        }

        if (m_kind == "taylor") {
            m_prune_funcs =
                std::make_unique<core::PruneTaylorDPFuncts<FoldType>>(
                    m_ffa_plan.params.back(), m_ffa_plan.dparams.back(),
                    m_ffa_plan.tsegments.back(), m_cfg);
        } else {
            throw std::runtime_error(
                std::format("Invalid pruning kind: {}", m_kind));
        }
    }
};

template <typename FoldType>
PruningManager<FoldType>::PruningManager(
    const search::PulsarSearchConfig& cfg,
    const std::vector<float>& threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size)
    : m_impl(std::make_unique<Impl>(cfg,
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    max_sugg,
                                    batch_size)) {}
template <typename FoldType>
PruningManager<FoldType>::~PruningManager() = default;
template <typename FoldType>
PruningManager<FoldType>::PruningManager(PruningManager&& other) noexcept =
    default;
template <typename FoldType>
PruningManager<FoldType>&
PruningManager<FoldType>::operator=(PruningManager&& other) noexcept = default;
template <typename FoldType>
void PruningManager<FoldType>::execute(std::span<const float> ts_e,
                                       std::span<const float> ts_v,
                                       const std::filesystem::path& outdir,
                                       std::string_view file_prefix,
                                       std::string_view kind) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix, kind);
}

template <typename FoldType>
Prune<FoldType>::Prune(const plans::FFAPlan& ffa_plan,
                       const search::PulsarSearchConfig& cfg,
                       std::span<const float> threshold_scheme,
                       SizeType max_sugg,
                       SizeType batch_size,
                       std::string_view kind)
    : m_impl(std::make_unique<Impl>(
          ffa_plan, cfg, threshold_scheme, max_sugg, batch_size, kind)) {}
template <typename FoldType> Prune<FoldType>::~Prune() = default;
template <typename FoldType>
Prune<FoldType>::Prune(Prune&& other) noexcept = default;
template <typename FoldType>
Prune<FoldType>& Prune<FoldType>::operator=(Prune&& other) noexcept = default;

template <typename FoldType>
SizeType Prune<FoldType>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}

template <typename FoldType>
void Prune<FoldType>::execute(
    std::span<const FoldType> ffa_fold,
    SizeType ref_seg,
    const std::filesystem::path& outdir,
    const std::optional<std::filesystem::path>& log_file,
    const std::optional<std::filesystem::path>& result_file,
    progress::MultiprocessProgressTracker* tracker,
    int task_id) {
    m_impl->execute(ffa_fold, ref_seg, outdir, log_file, result_file, tracker,
                    task_id);
}

// Template instantiations
template class PruningManager<float>;
template class PruningManager<ComplexType>;

template class Prune<float>;
template class Prune<ComplexType>;

} // namespace loki::algorithms