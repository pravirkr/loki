#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
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
#include "loki/utils/world_tree.hpp"

namespace loki::algorithms {

class PruningManager::BaseImpl {
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
                         std::string_view file_prefix,
                         std::string_view kind,
                         bool show_progress) = 0;
};

template <SupportedFoldType FoldType>
class PruningManagerTypedImpl final : public PruningManager::BaseImpl {
public:
    PruningManagerTypedImpl(search::PulsarSearchConfig cfg,
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
          m_nthreads(m_cfg.get_nthreads()) {}

    ~PruningManagerTypedImpl() final                        = default;
    PruningManagerTypedImpl(const PruningManagerTypedImpl&) = delete;
    PruningManagerTypedImpl& operator=(const PruningManagerTypedImpl&) = delete;
    PruningManagerTypedImpl(PruningManagerTypedImpl&&)                 = delete;
    PruningManagerTypedImpl& operator=(PruningManagerTypedImpl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix,
                 std::string_view poly_basis,
                 bool show_progress) override {
        spdlog::info("PruningManager: Initializing with FFA");
        // Create appropriate FFA fold
        std::tuple<std::vector<FoldType>, plans::FFAPlan<FoldType>> result =
            compute_ffa<FoldType>(ts_e, ts_v, m_cfg, /*quiet=*/false,
                                  show_progress);
        const std::vector<FoldType> ffa_fold    = std::get<0>(result);
        const plans::FFAPlan<FoldType> ffa_plan = std::get<1>(result);
        // Setup output files and directory
        const auto nsegments = ffa_plan.get_nsegments().back();
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
                              m_threshold_scheme);
        // Execute based on thread count
        if (m_nthreads == 1) {
            execute_single_threaded(ffa_fold, ffa_plan, ref_segs_to_process,
                                    outdir, log_file, result_file, poly_basis,
                                    show_progress);
        } else {
            execute_multi_threaded(ffa_fold, ffa_plan, ref_segs_to_process,
                                   outdir, log_file, poly_basis, show_progress);
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
    // plans::FFAPlan m_ffa_plan;
    // std::unique_ptr<algorithms::FFA> m_the_ffa;
    // std::unique_ptr<algorithms::FFACOMPLEX> m_the_ffa_complex;
    // std::vector<FoldType> m_ffa_fold;

    void execute_single_threaded(std::span<const FoldType> ffa_fold,
                                 const plans::FFAPlan<FoldType>& ffa_plan,
                                 const std::vector<SizeType>& ref_segs,
                                 const std::filesystem::path& outdir,
                                 const std::filesystem::path& log_file,
                                 const std::filesystem::path& result_file,
                                 std::string_view kind,
                                 bool show_progress) {
        auto prune = Prune<FoldType>(ffa_plan, m_cfg, m_threshold_scheme,
                                     m_max_sugg, m_batch_size, kind);
        for (const auto ref_seg : ref_segs) {
            prune.execute(ffa_fold, ref_seg, outdir, log_file, result_file,
                          /*tracker=*/nullptr, /*task_id=*/0, show_progress);
        }
    }

    void execute_multi_threaded(std::span<const FoldType> ffa_fold,
                                const plans::FFAPlan<FoldType>& ffa_plan,
                                const std::vector<SizeType>& ref_segs,
                                const std::filesystem::path& outdir,
                                const std::filesystem::path& log_file,
                                std::string_view kind,
                                bool show_progress) {
        // Only create progress tracker if show_progress is true
        std::unique_ptr<progress::MultiprocessProgressTracker> tracker;
        if (show_progress) {
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

        const auto nsegments = ffa_plan.get_nsegments().back();
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

            auto future = pool.submit_task(
                [this, ref_seg, outdir, kind, tracker_ptr = tracker.get(), id,
                 show_progress, &ffa_plan, &ffa_fold]() mutable {
                    // Create a Prune instance for each thread
                    auto prune =
                        Prune<FoldType>(ffa_plan, m_cfg, m_threshold_scheme,
                                        m_max_sugg, m_batch_size, kind);

                    prune.execute(
                        ffa_fold, ref_seg, outdir, /*log_file=*/std::nullopt,
                        /*result_file=*/std::nullopt, /*tracker=*/tracker_ptr,
                        /*task_id=*/id, /*show_progress=*/show_progress);
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
}; // End PruningManagerTypedImpl implementation

struct IterationStats {
    SizeType n_leaves     = 0;
    SizeType n_leaves_phy = 0;
    float score_min       = std::numeric_limits<float>::max();
    float score_max       = std::numeric_limits<float>::lowest();
    cands::PruneTimerStats batch_timers;
};

template <SupportedFoldType FoldType> struct PruningWorkspace {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size;
    SizeType branch_max;
    SizeType nparams;
    SizeType nbins;
    SizeType max_branched_leaves;
    SizeType leaves_stride{};
    SizeType folds_stride;

    std::vector<double> branched_leaves;
    std::vector<FoldType> branched_folds;
    std::vector<float> branched_scores;
    // Scratch space for indices
    std::vector<SizeType> branched_indices;
    // Scratch space for resolving parameters
    std::vector<SizeType> branched_param_idx;
    std::vector<float> branched_phase_shift;

    PruningWorkspace(SizeType batch_size,
                     SizeType branch_max,
                     SizeType nparams,
                     SizeType nbins)
        : batch_size(batch_size),
          branch_max(branch_max),
          nparams(nparams),
          nbins(nbins),
          max_branched_leaves(batch_size * branch_max),
          leaves_stride((nparams + 2) * kLeavesParamStride),
          folds_stride(2 * nbins),
          branched_leaves(max_branched_leaves * leaves_stride),
          branched_folds(max_branched_leaves * folds_stride),
          branched_scores(max_branched_leaves),
          branched_indices(max_branched_leaves),
          branched_param_idx(max_branched_leaves),
          branched_phase_shift(max_branched_leaves) {}

    // Call this after filling branched_scores and branched_indices
    void filter_batch(SizeType n_leaves_passing) noexcept {
        for (SizeType dst_idx = 0; dst_idx < n_leaves_passing; ++dst_idx) {
            const SizeType src_idx   = branched_indices[dst_idx];
            branched_scores[dst_idx] = branched_scores[src_idx];
            std::copy_n(branched_leaves.begin() + (src_idx * leaves_stride),
                        leaves_stride,
                        branched_leaves.begin() + (dst_idx * leaves_stride));
            std::copy_n(branched_folds.begin() + (src_idx * folds_stride),
                        folds_stride,
                        branched_folds.begin() + (dst_idx * folds_stride));
        }
    }

    float get_memory_usage() const noexcept {
        const auto total_memory =
            (branched_leaves.size() * sizeof(double)) +
            (branched_folds.size() * sizeof(FoldType)) +
            (branched_scores.size() * sizeof(float)) +
            (branched_indices.size() * sizeof(SizeType)) +
            (branched_param_idx.size() * sizeof(SizeType)) +
            (branched_phase_shift.size() * sizeof(float));
        return static_cast<float>(total_memory) /
               static_cast<float>(1ULL << 30U);
    }
}; // End PruningWorkspace definition

template <SupportedFoldType FoldType> class Prune<FoldType>::Impl {
public:
    Impl(plans::FFAPlan<FoldType> ffa_plan,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis)
        : m_ffa_plan(std::move(ffa_plan)),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_total_levels(m_threshold_scheme.size()) {
        // Setup pruning functions
        setup_pruning();

        const auto max_batch_size = m_batch_size * m_branch_max;
        // Allocate suggestion buffer
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_world_tree = std::make_unique<utils::WorldTree<FoldType>>(
                m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins_f(),
                max_batch_size);
        } else {
            m_world_tree = std::make_unique<utils::WorldTree<FoldType>>(
                m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins(),
                max_batch_size);
        }

        // Allocate iteration workspace
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_pruning_workspace = std::make_unique<PruningWorkspace<FoldType>>(
                m_batch_size, m_branch_max, m_cfg.get_nparams(),
                m_cfg.get_nbins_f());
        } else {
            m_pruning_workspace = std::make_unique<PruningWorkspace<FoldType>>(
                m_batch_size, m_branch_max, m_cfg.get_nparams(),
                m_cfg.get_nbins());
        }

        m_branching_workspace = std::make_unique<utils::BranchingWorkspace>(
            m_batch_size, m_branch_max, m_cfg.get_nparams());

        // Allocate buffers for seeding the world tree and scoring
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        m_seed_leaves.resize(ncoords_ffa * m_world_tree->get_leaves_stride());
        m_seed_scores.resize(ncoords_ffa);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_memory_usage() const noexcept {
        return m_pruning_workspace->get_memory_usage() +
               m_world_tree->get_memory_usage() +
               m_branching_workspace->get_memory_usage();
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
        const auto memory_workspace_gb =
            m_pruning_workspace->get_memory_usage();
        const auto memory_tree_gb = m_world_tree->get_memory_usage();
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
                bar->set_score(m_world_tree->get_score_max());
                bar->set_leaves(m_world_tree->get_size_lb());
                bar->set_progress(iter + 1);
            }
        }

        // Transform the suggestion params to middle of the data
        const auto coord_mid = m_snail_scheme->get_coord(m_prune_level);

        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        auto leaves_report = m_world_tree->get_leaves_contiguous_span();
        m_prune_funcs->report(leaves_report, coord_mid,
                              m_world_tree->get_size());
        result_writer.write_run_results(
            run_name, m_snail_scheme->get_data(), leaves_report,
            m_world_tree->get_scores_contiguous_span(),
            m_world_tree->get_size(), m_cfg.get_nparams(), *m_pstats);

        // Final log entries
        std::ofstream final_log(actual_log_file, std::ios::app);
        final_log << std::format("Pruning run complete for ref segment {}\n",
                                 ref_seg);
        final_log << std::format("Time: {}\n\n", m_pstats->get_timer_summary());
        final_log.close();
        spdlog::info("Pruning run {:03d}: complete", ref_seg);
        spdlog::info("Pruning run {:03d}: stats: {}", ref_seg,
                     m_pstats->get_stats_summary());
        spdlog::info("Pruning run {:03d}: timer: {}", ref_seg,
                     m_pstats->get_concise_timer_summary());
    }

private:
    plans::FFAPlan<FoldType> m_ffa_plan;
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string_view m_poly_basis;
    SizeType m_total_levels;

    std::vector<double> m_branching_pattern;
    SizeType m_branch_max{0};
    bool m_prune_complete{false};
    SizeType m_prune_level{};
    std::unique_ptr<psr_utils::MiddleOutScheme> m_snail_scheme;
    std::unique_ptr<core::PruneDPFuncts<FoldType>> m_prune_funcs;
    std::unique_ptr<cands::PruneStatsCollection> m_pstats;

    // Active, bounded container of hierarchical search candidates
    std::unique_ptr<utils::WorldTree<FoldType>> m_world_tree;
    std::unique_ptr<PruningWorkspace<FoldType>> m_pruning_workspace;
    std::unique_ptr<utils::BranchingWorkspace> m_branching_workspace;

    // Buffers for seeding the world tree and scoring
    std::vector<double> m_seed_leaves;
    std::vector<float> m_seed_scores;

    void initialize(std::span<const FoldType> ffa_fold,
                    SizeType ref_seg,
                    const std::filesystem::path& log_file) {
        // timing::ScopeTimer timer("Prune::initialize");
        //  Reset the world tree state
        m_world_tree->reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        const auto tseg      = m_ffa_plan.get_tsegments().back();
        m_snail_scheme       = std::make_unique<psr_utils::MiddleOutScheme>(
            nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;
        spdlog::info("Pruning run {:03d}: initialized", ref_seg);

        // Initialize the world tree with the first segment
        const auto fold_segment = m_prune_funcs->load_segment(
            ffa_fold, m_snail_scheme->get_ref_idx());
        const auto coord_init = m_snail_scheme->get_coord(m_prune_level);
        const auto n_leaves   = m_ffa_plan.get_ncoords().back();
        m_prune_funcs->seed(fold_segment, m_seed_leaves, m_seed_scores,
                            coord_init);
        // Initialize the WorldTree with the generated data
        m_world_tree->add_initial(m_seed_leaves, fold_segment, m_seed_scores,
                                  n_leaves);

        // Initialize the prune stats
        m_pstats = std::make_unique<cands::PruneStatsCollection>();
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = m_snail_scheme->get_segment_idx(m_prune_level),
            .threshold     = 0,
            .score_min     = m_world_tree->get_score_min(),
            .score_max     = m_world_tree->get_score_max(),
            .n_branches    = m_world_tree->get_size(),
            .n_leaves      = m_world_tree->get_size(),
            .n_leaves_phy  = m_world_tree->get_size(),
            .n_leaves_surv = m_world_tree->get_size(),
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
        error_check::check_less_equal(
            m_prune_level, m_threshold_scheme.size(),
            "Pruning complete - exceeded threshold scheme length");
        // Prepare for in-place update: mark start of write region, reset size
        // for new suggestions.
        m_world_tree->prepare_in_place_update();

        IterationStats stats;
        const auto seg_idx_cur = m_snail_scheme->get_segment_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches = m_world_tree->get_size_old();

        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats);

        // Finalize: make new region active, defragment for contiguous access.
        m_world_tree->finalize_in_place_update();

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
            .n_leaves_surv = m_world_tree->get_size(),
        };
        // Write stats to log
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary();
        log.close();
        m_pstats->update_stats(pstats_cur, stats.batch_timers);

        // Check if no survivors
        if (m_world_tree->get_size() == 0) {
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
        // Get coordinates
        const auto coord_init = m_snail_scheme->get_coord(0);
        const auto coord_prev = m_snail_scheme->get_coord(m_prune_level - 1);
        const auto coord_next = m_snail_scheme->get_coord(m_prune_level);
        const auto coord_cur = m_snail_scheme->get_current_coord(m_prune_level);
        const auto coord_add = m_snail_scheme->get_segment_coord(m_prune_level);

        // Load fold segment for current level
        const auto ffa_fold_segment =
            m_prune_funcs->load_segment(ffa_fold, seg_idx_cur);

        auto current_threshold = threshold;

        const auto n_branches = m_world_tree->get_size_old();
        const auto batch_size =
            std::max(1UL, std::min(m_batch_size, n_branches));
        auto branch_ws = m_branching_workspace->get_view();

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
                m_world_tree->get_leaves_span(this_batch_size);
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
                leaves_tree_span, m_pruning_workspace->branched_leaves,
                m_pruning_workspace->branched_indices, coord_cur, coord_prev,
                current_batch_size, branch_ws);
            stats.batch_timers["branch"] += timer.stop();
            stats.n_leaves += n_leaves_batch;
            if (n_leaves_batch == 0) {
                m_world_tree->consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_batch, m_pruning_workspace->max_branched_leaves,
                "Branch factor exceeded workspace size:n_leaves_batch <= "
                "max_branched_leaves");

            // Validation
            timer.start();
            const auto n_leaves_after_validation =
                m_prune_funcs->validate(m_pruning_workspace->branched_leaves,
                                        m_pruning_workspace->branched_indices,
                                        coord_cur, n_leaves_batch);
            stats.batch_timers["validate"] += timer.stop();
            stats.n_leaves_phy += n_leaves_after_validation;
            if (n_leaves_after_validation == 0) {
                m_world_tree->consume_read(current_batch_size);
                continue;
            }

            // Resolve
            timer.start();
            m_prune_funcs->resolve(m_pruning_workspace->branched_leaves,
                                   m_pruning_workspace->branched_param_idx,
                                   m_pruning_workspace->branched_phase_shift,
                                   coord_add, coord_cur, coord_init,
                                   n_leaves_after_validation);
            stats.batch_timers["resolve"] += timer.stop();

            // Load, shift, add (Map branched_itree to physical indices)
            timer.start();
            m_world_tree->convert_to_physical_indices(
                m_pruning_workspace->branched_indices,
                n_leaves_after_validation);
            m_prune_funcs->shift_add(
                m_world_tree->get_folds(),
                m_pruning_workspace->branched_indices, ffa_fold_segment,
                m_pruning_workspace->branched_param_idx,
                m_pruning_workspace->branched_phase_shift,
                m_pruning_workspace->branched_folds, n_leaves_after_validation);
            stats.batch_timers["shift_add"] += timer.stop();

            // Score
            timer.start();
            m_prune_funcs->score(m_pruning_workspace->branched_folds,
                                 m_pruning_workspace->branched_scores,
                                 n_leaves_after_validation);
            auto branched_scores_span =
                std::span<const float>(m_pruning_workspace->branched_scores)
                    .first(n_leaves_after_validation);
            const auto [min_it, max_it] =
                std::ranges::minmax_element(branched_scores_span);
            stats.score_min = std::min(stats.score_min, *min_it);
            stats.score_max = std::max(stats.score_max, *max_it);
            stats.batch_timers["score"] += timer.stop();

            // Thresholding & filtering (direct memory operations)
            timer.start();
            SizeType n_leaves_passing = 0;
            for (SizeType i = 0; i < n_leaves_after_validation; ++i) {
                if (branched_scores_span[i] >= current_threshold) {
                    m_pruning_workspace->branched_indices[n_leaves_passing] = i;
                    ++n_leaves_passing;
                }
            }
            stats.batch_timers["filter"] += timer.stop();

            if (n_leaves_passing == 0) {
                m_world_tree->consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_passing, m_pruning_workspace->max_branched_leaves,
                "n_leaves_passing <= max_branched_leaves");

            timer.start();
            if (n_leaves_passing != n_leaves_after_validation) {
                m_pruning_workspace->filter_batch(n_leaves_passing);
            }
            stats.batch_timers["filter"] += timer.stop();

            // Transform
            timer.start();
            m_prune_funcs->transform(m_pruning_workspace->branched_leaves,
                                     coord_next, coord_cur, n_leaves_passing);
            stats.batch_timers["transform"] += timer.stop();

            // Add batch to output suggestions
            timer.start();
            current_threshold =
                m_world_tree->add_batch(m_pruning_workspace->branched_leaves,
                                        m_pruning_workspace->branched_folds,
                                        m_pruning_workspace->branched_scores,
                                        current_threshold, n_leaves_passing);
            stats.batch_timers["batch_add"] += timer.stop();
            // Notify the buffer that a batch of the old suggestions has been
            // consumed
            m_world_tree->consume_read(current_batch_size);
        }
    }

    void setup_pruning() {
        error_check::check_less_equal(m_cfg.get_nparams(), 5,
                                      "Pruning not supported for nparams > 5.");
        m_branching_pattern   = m_ffa_plan.get_branching_pattern(m_poly_basis);
        const auto branch_max = *std::ranges::max_element(m_branching_pattern);
        m_branch_max =
            std::max(static_cast<SizeType>(std::ceil(branch_max * 2)), 32UL);
        m_prune_funcs = core::create_prune_dp_functs<FoldType>(
            m_poly_basis, m_ffa_plan.get_params().back(),
            m_ffa_plan.get_dparams_lim().back(),
            m_ffa_plan.get_nsegments().back(),
            m_ffa_plan.get_tsegments().back(), m_cfg, m_batch_size,
            m_branch_max);
    }
}; // End Prune::Impl definition

PruningManager::PruningManager(const search::PulsarSearchConfig& cfg,
                               const std::vector<float>& threshold_scheme,
                               std::optional<SizeType> n_runs,
                               std::optional<std::vector<SizeType>> ref_segs,
                               SizeType max_sugg,
                               SizeType batch_size) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<PruningManagerTypedImpl<ComplexType>>(
            cfg, threshold_scheme, n_runs, std::move(ref_segs), max_sugg,
            batch_size);
    } else {
        m_impl = std::make_unique<PruningManagerTypedImpl<float>>(
            cfg, threshold_scheme, n_runs, std::move(ref_segs), max_sugg,
            batch_size);
    }
}
PruningManager::~PruningManager()                               = default;
PruningManager::PruningManager(PruningManager&& other) noexcept = default;
PruningManager&
PruningManager::operator=(PruningManager&& other) noexcept = default;
void PruningManager::execute(std::span<const float> ts_e,
                             std::span<const float> ts_v,
                             const std::filesystem::path& outdir,
                             std::string_view file_prefix,
                             std::string_view poly_basis,
                             bool show_progress) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix, poly_basis, show_progress);
}

template <SupportedFoldType FoldType>
Prune<FoldType>::Prune(const plans::FFAPlan<FoldType>& ffa_plan,
                       const search::PulsarSearchConfig& cfg,
                       std::span<const float> threshold_scheme,
                       SizeType max_sugg,
                       SizeType batch_size,
                       std::string_view poly_basis)
    : m_impl(std::make_unique<Impl>(
          ffa_plan, cfg, threshold_scheme, max_sugg, batch_size, poly_basis)) {}
template <SupportedFoldType FoldType> Prune<FoldType>::~Prune() = default;
template <SupportedFoldType FoldType>
Prune<FoldType>::Prune(Prune&& other) noexcept = default;
template <SupportedFoldType FoldType>
Prune<FoldType>& Prune<FoldType>::operator=(Prune&& other) noexcept = default;

template <SupportedFoldType FoldType>
SizeType Prune<FoldType>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
template <SupportedFoldType FoldType>
void Prune<FoldType>::execute(
    std::span<const FoldType> ffa_fold,
    SizeType ref_seg,
    const std::filesystem::path& outdir,
    const std::optional<std::filesystem::path>& log_file,
    const std::optional<std::filesystem::path>& result_file,
    progress::MultiprocessProgressTracker* tracker,
    int task_id,
    bool show_progress) {
    m_impl->execute(ffa_fold, ref_seg, outdir, log_file, result_file, tracker,
                    task_id, show_progress);
}

template class Prune<float>;
template class Prune<ComplexType>;

} // namespace loki::algorithms