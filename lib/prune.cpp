#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <utility>

#include <BS_thread_pool.hpp>
#include <spdlog/spdlog.h>
#include <xtensor/views/xview.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/core/dynamic.hpp"
#include "loki/psr_utils.hpp"
#include "loki/timing.hpp"
#include "loki/utils/suggestions.hpp"

namespace loki::algorithms {

class PruningManager::Impl {
public:
    Impl(search::PulsarSearchConfig cfg,
         const std::vector<float>& threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         SizeType max_sugg,
         SizeType batch_size,
         int nthreads)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_nthreads(nthreads) {
        m_the_ffa            = std::make_unique<algorithms::FFA>(m_cfg);
        const auto& ffa_plan = m_the_ffa->get_plan();
        m_ffa_fold.resize(ffa_plan.get_fold_size(), 0.0F);
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
        // Compute FFA fold
        m_the_ffa->execute(ts_e, ts_v, std::span<float>(m_ffa_fold));

        // Setup output files and directory
        const auto& ffa_plan = m_the_ffa->get_plan();
        const auto nsegments = ffa_plan.fold_shapes.back()[0];
        const std::string filebase =
            std::format("{}_pruning_nstages_{}", file_prefix, nsegments);
        const auto log_file = outdir / std::format("{}_log.txt", filebase);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);

        // Create output directory
        std::filesystem::create_directories(outdir);

        // Determine ref_segs to process
        std::vector<SizeType> ref_segs_to_process =
            determine_ref_segs(nsegments);

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

    // FFA plan and buffer
    std::unique_ptr<algorithms::FFA> m_the_ffa;
    std::vector<float> m_ffa_fold;

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
            for (SizeType i = 0; i < n_runs; ++i) {
                ref_segs[i] = static_cast<SizeType>(
                    std::round(static_cast<double>(i * (nsegments - 1)) /
                               static_cast<double>(n_runs - 1)));
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
        const auto& ffa_plan = m_the_ffa->get_plan();

        auto prune = Prune<float>(ffa_plan, m_cfg, m_threshold_scheme,
                                  m_max_sugg, m_batch_size, kind);
        for (const auto ref_seg : ref_segs) {
            spdlog::info("Processing ref segment {} (single-threaded)",
                         ref_seg);
            prune.execute(m_ffa_fold, ref_seg, outdir, log_file, result_file);
        }
    }

    void execute_multi_threaded(const std::vector<SizeType>& ref_segs,
                                const std::filesystem::path& outdir,
                                const std::filesystem::path& log_file,
                                std::string_view kind) {

        const auto& ffa_plan = m_the_ffa->get_plan();

        // Create thread pool
        BS::thread_pool pool(m_nthreads);

        // Submit tasks for each ref_seg
        std::vector<std::future<void>> futures;
        futures.reserve(ref_segs.size());

        for (const auto ref_seg : ref_segs) {
            auto future = pool.submit_task(
                [this, ffa_plan, ref_seg, outdir, kind]() mutable {
                    execute_single_ref_seg(ffa_plan, ref_seg, outdir, kind);
                });
            futures.push_back(std::move(future));
        }
        // Wait for all tasks to complete and handle exceptions
        std::vector<std::pair<SizeType, std::string>> errors;
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                futures[i].get();
                spdlog::info("Completed ref segment {}", ref_segs[i]);
            } catch (const std::exception& e) {
                const std::string error_msg = std::format(
                    "Error in ref_seg {}: {}", ref_segs[i], e.what());
                errors.emplace_back(ref_segs[i], error_msg);
                spdlog::error(error_msg);
            }
        }

        // Log errors to file
        if (!errors.empty()) {
            std::ofstream log(log_file, std::ios::app);
            for (const auto& [ref_seg, error_msg] : errors) {
                log << std::format("Error processing ref_seg {}: {}\n", ref_seg,
                                   error_msg);
            }
            log.close();
        }
    }
    void execute_single_ref_seg(const plans::FFAPlan& ffa_plan,
                                SizeType ref_seg,
                                const std::filesystem::path& outdir,
                                std::string_view kind) {

        try {
            // Create separate Prune instance for this thread
            auto prune = Prune<float>(ffa_plan, m_cfg, m_threshold_scheme,
                                      m_max_sugg, m_batch_size, kind);

            // Generate unique temporary files for this thread
            const std::string run_name =
                std::format("{:03d}_{:03d}", ref_seg, ref_seg);
            const auto temp_log_file =
                outdir / std::format("tmp_{}_log.txt", run_name);
            const auto temp_result_file =
                outdir / std::format("tmp_{}_results.h5", run_name);

            spdlog::info("Processing ref segment {} (thread {})", ref_seg,
                         std::this_thread::get_id());

            prune.execute(m_ffa_fold, ref_seg, outdir, temp_log_file,
                          temp_result_file);
        } catch (const std::exception& e) {
            spdlog::error("Error in ref_seg {}: {}", ref_seg, e.what());
            throw; // Re-throw to be caught by the future
        }
    }
};

struct IterationStats {
    SizeType n_leaves     = 0;
    SizeType n_leaves_phy = 0;
    float score_min       = std::numeric_limits<float>::infinity();
    float score_max       = -std::numeric_limits<float>::infinity();
    std::array<double, 7> timers{};
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
        // Allocate suggestion buffers
        m_suggestions_in = std::make_unique<utils::SuggestionStruct<FoldType>>(
            m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins());
        m_suggestions_out = std::make_unique<utils::SuggestionStruct<FoldType>>(
            m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins());

        // Setup pruning functions
        setup_pruning();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void execute(std::span<const FoldType> ffa_fold,
                 SizeType ref_seg,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file) {
        const std::string run_name =
            std::format("{:03d}_{:02d}", ref_seg, ref_seg);
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / std::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / std::format("tmp_{}_results.h5", run_name));
        std::ofstream log(actual_log_file, std::ios::app);
        log << std::format("Pruning log for ref segment: {}\n", ref_seg);
        log.close();

        initialize(ffa_fold, ref_seg, actual_log_file);

        const auto nsegments = m_ffa_plan.fold_shapes.back()[0];
        for (SizeType iter = 0; iter < nsegments - 1; ++iter) {
            execute_iteration(ffa_fold, actual_log_file);
            // Check for early termination (no survivors)
            if (m_prune_complete) {
                spdlog::info(
                    "Pruning terminated early at iteration {} - no survivors",
                    iter + 1);
                break;
            }
        }
        // Determine which suggestion buffer contains the final results
        // After ping-pong, the "input" buffer contains the latest results
        auto* final_suggestions = m_suggestions_in.get();

        // Transform the suggestion params to middle of the data
        const auto delta_t = m_scheme->get_delta(m_prune_level);

        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        result_writer.write_run_results(
            run_name, m_scheme->get_data(),
            final_suggestions->get_transformed(delta_t),
            final_suggestions->get_scores(), *m_pstats);

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

    // Suggestion buffers (ping-pong strategy)
    std::unique_ptr<utils::SuggestionStruct<FoldType>> m_suggestions_in;
    std::unique_ptr<utils::SuggestionStruct<FoldType>> m_suggestions_out;

    void initialize(std::span<const float> ffa_fold,
                    SizeType ref_seg,
                    const std::filesystem::path& log_file) {
        ScopeTimer timer("Prune::initialize");
        // Initialize suggestion buffers
        m_suggestions_in->reset();
        m_suggestions_out->reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.fold_shapes.back()[0];
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
        m_prune_funcs->suggest(fold_segment, coord_init, *m_suggestions_in);

        // Initialize the prune stats
        m_pstats = std::make_unique<cands::PruneStatsCollection>();
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = m_scheme->get_idx(m_prune_level),
            .threshold     = 0,
            .score_min     = m_suggestions_in->get_score_min(),
            .score_max     = m_suggestions_in->get_score_max(),
            .n_branches    = m_suggestions_in->get_nsugg(),
            .n_leaves      = m_suggestions_in->get_nsugg(),
            .n_leaves_phy  = m_suggestions_in->get_nsugg(),
            .n_leaves_surv = m_suggestions_in->get_nsugg(),
        };
        m_pstats->update_stats(pstats_cur);

        // Write the initial prune stats to the log file
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary() << '\n';
        log.close();
    }

    void execute_iteration(std::span<const FoldType> ffa_fold,
                           const std::filesystem::path& log_file) {
        if (m_prune_complete) {
            return;
        }
        ++m_prune_level;
        if (m_prune_level >= m_threshold_scheme.size()) {
            m_prune_complete = true;
            spdlog::info("Pruning complete - exceeded threshold scheme length "
                         "at level {}",
                         m_prune_level);
            return;
        }
        // Reset output suggestion buffer
        m_suggestions_out->reset();

        IterationStats stats;
        const auto seg_idx_cur = m_scheme->get_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats);

        // Update statistics
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = seg_idx_cur,
            .threshold     = threshold,
            .score_min     = stats.score_min,
            .score_max     = stats.score_max,
            .n_branches    = m_suggestions_in->get_nsugg(),
            .n_leaves      = stats.n_leaves,
            .n_leaves_phy  = stats.n_leaves_phy,
            .n_leaves_surv = m_suggestions_out->get_nsugg(),
        };
        // Write stats to log
        std::ofstream log(log_file, std::ios::app);
        log << pstats_cur.get_summary() << '\n';
        log.close();
        m_pstats->update_stats(pstats_cur);

        // Check if no survivors
        if (m_suggestions_in->get_nsugg() == 0) {
            m_prune_complete = true;
            spdlog::info("Pruning complete at level {} - no survivors",
                         m_prune_level);
            return;
        }
        // Ping-pong: swap input and output buffers
        std::swap(m_suggestions_in, m_suggestions_out);
    }

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

        const auto n_branches = m_suggestions_in->get_nsugg();
        const auto batch_size =
            std::max(1UL, std::min(m_batch_size, n_branches));

        std::vector<SizeType> isuggest_batch(batch_size);
        xt::xtensor<FoldType, 3> batch_combined_res(
            {batch_size, 2, m_cfg.get_nbins()});
        std::vector<float> batch_scores(batch_size);
        std::vector<SizeType> filtered_indices(batch_size);

        auto t_start = std::chrono::steady_clock::now();

        // Process branches in batches
        for (SizeType i_batch_start = 0; i_batch_start < n_branches;
             i_batch_start += batch_size) {
            const auto i_batch_end =
                std::min(i_batch_start + batch_size, n_branches);
            const auto current_batch_size = i_batch_end - i_batch_start;

            // Branch
            t_start                = std::chrono::steady_clock::now();
            const auto param_batch = xt::view(
                m_suggestions_in->get_param_sets(),
                xt::range(i_batch_start, i_batch_end), xt::all(), xt::all());
            auto [batch_leaves, batch_leaf_origins] =
                m_prune_funcs->branch(param_batch, coord_cur);
            const auto n_leaves_batch = batch_leaves.shape(0);
            stats.n_leaves += n_leaves_batch;
            stats.timers[0] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();
            if (n_leaves_batch == 0) {
                continue;
            }

            // Validation
            t_start = std::chrono::steady_clock::now();
            if (validation_check) {
                batch_leaves = m_prune_funcs->validate(
                    batch_leaves, coord_valid, validation_params);
            }
            const auto n_leaves_after_validation = batch_leaves.shape(0);
            stats.n_leaves_phy += n_leaves_after_validation;
            stats.timers[1] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();

            // Resolve
            t_start = std::chrono::steady_clock::now();
            auto [batch_param_idx, batch_phase_shift] =
                m_prune_funcs->resolve(batch_leaves, coord_add, coord_init);
            stats.timers[2] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();

            // Load, shift, add
            t_start = std::chrono::steady_clock::now();
            // Map batch_leaf_origins to global indices
            for (SizeType i = 0; i < batch_leaf_origins.size(); ++i) {
                isuggest_batch[i] = batch_leaf_origins[i] + i_batch_start;
            }
            m_prune_funcs->load_shift_add(ffa_fold_segment, batch_param_idx,
                                          batch_phase_shift,
                                          m_suggestions_in->get_folds(),
                                          isuggest_batch, batch_combined_res);
            stats.timers[3] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();

            // Score
            t_start = std::chrono::steady_clock::now();
            m_prune_funcs->score(batch_combined_res, batch_scores);
            if (n_leaves_after_validation > 0) {
                const auto [min_it, max_it] =
                    std::ranges::minmax_element(batch_scores);
                stats.score_min = std::min(stats.score_min, *min_it);
                stats.score_max = std::max(stats.score_max, *max_it);
            }
            stats.timers[4] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();

            // Thresholding & filtering
            t_start = std::chrono::steady_clock::now();
            filtered_indices.clear();
            for (SizeType i = 0; i < n_leaves_after_validation; ++i) {
                if (batch_scores[i] >= threshold) {
                    filtered_indices.push_back(i);
                }
            }
            const auto num_passing = filtered_indices.size();
            if (num_passing == 0) {
                stats.timers[6] +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start)
                        .count();
                continue;
            }
            // Filter results
            auto filtered_leaves = xt::view(
                batch_leaves, xt::keep(filtered_indices), xt::all(), xt::all());
            auto filtered_combined_res =
                xt::view(batch_combined_res, xt::keep(filtered_indices),
                         xt::all(), xt::all());

            std::vector<float> filtered_scores(num_passing);
            std::vector<SizeType> filtered_param_idx(num_passing);
            std::vector<double> filtered_phase_shift(num_passing);
            std::vector<SizeType> filtered_isuggest(num_passing);

            for (SizeType i = 0; i < num_passing; ++i) {
                const auto idx          = filtered_indices[i];
                filtered_scores[i]      = batch_scores[idx];
                filtered_param_idx[i]   = batch_param_idx[idx];
                filtered_phase_shift[i] = batch_phase_shift[idx];
                filtered_isuggest[i]    = isuggest_batch[idx];
            }

            stats.timers[6] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();

            // Transform
            t_start                          = std::chrono::steady_clock::now();
            const auto filtered_leaves_trans = m_prune_funcs->transform(
                xt::view(filtered_leaves, xt::all(), xt::all(), 0), coord_cur,
                trans_matrix);
            stats.timers[5] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();

            // Construct backtrack
            t_start               = std::chrono::steady_clock::now();
            const auto bt_nparams = m_cfg.get_nparams();
            xt::xtensor<SizeType, 2> batch_backtrack(
                {num_passing, bt_nparams + 2});

            for (SizeType i = 0; i < num_passing; ++i) {
                batch_backtrack(i, 0) = filtered_isuggest[i];
                batch_backtrack(i, 1) = filtered_param_idx[i];
                for (SizeType j = 2; j < bt_nparams + 1; ++j) {
                    batch_backtrack(i, j) = 0;
                }
                batch_backtrack(i, bt_nparams + 1) =
                    static_cast<SizeType>(std::round(filtered_phase_shift[i]));
            }
            // Add batch to output suggestions
            current_threshold = m_suggestions_out->add_batch(
                filtered_leaves_trans, filtered_combined_res, filtered_scores,
                batch_backtrack, current_threshold);

            stats.timers[6] += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();
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

PruningManager::PruningManager(const search::PulsarSearchConfig& cfg,
                               const std::vector<float>& threshold_scheme,
                               std::optional<SizeType> n_runs,
                               std::optional<std::vector<SizeType>> ref_segs,
                               SizeType max_sugg,
                               SizeType batch_size,
                               int nthreads)
    : m_impl(std::make_unique<Impl>(cfg,
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    max_sugg,
                                    batch_size,
                                    nthreads)) {}
PruningManager::~PruningManager()                               = default;
PruningManager::PruningManager(PruningManager&& other) noexcept = default;
PruningManager&
PruningManager::operator=(PruningManager&& other) noexcept = default;
void PruningManager::execute(std::span<const float> ts_e,
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
void Prune<FoldType>::execute(
    std::span<const FoldType> ffa_fold,
    SizeType ref_seg,
    const std::filesystem::path& outdir,
    const std::optional<std::filesystem::path>& log_file,
    const std::optional<std::filesystem::path>& result_file) {
    m_impl->execute(ffa_fold, ref_seg, outdir, log_file, result_file);
}

} // namespace loki::algorithms