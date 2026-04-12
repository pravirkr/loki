#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/core/dynamic.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/psr_utils.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"
#include "loki/utils/world_tree.hpp"

namespace loki::algorithms {

namespace {
struct IterationStats {
    SizeType n_leaves     = 0;
    SizeType n_leaves_phy = 0;
    float score_min       = std::numeric_limits<float>::max();
    float score_max       = std::numeric_limits<float>::lowest();
};

struct ExecutionStream {
    cudaStream_t stream{nullptr};
    bool owns{false};

    explicit ExecutionStream(int device_id) {
        cuda_utils::CudaSetDeviceGuard guard(device_id);
        cuda_utils::check_cuda_call(cudaStreamCreate(&stream),
                                    "EPMultiPassCUDA: cudaStreamCreate failed");
        owns = true;
    }

    explicit ExecutionStream(cudaStream_t external_stream)
        : stream(external_stream),
          owns(false) {
        if (stream == nullptr) {
            throw std::invalid_argument(
                "EPMultiPassCUDA: execution_stream must be non-null when using "
                "an external EPWorkspaceCUDA");
        }
    }

    ~ExecutionStream() {
        if (owns && stream != nullptr) {
            cuda_utils::check_cuda_call(
                cudaStreamSynchronize(stream),
                "EPMultiPassCUDA: cudaStreamSynchronize before destroy failed");
            cuda_utils::check_cuda_call(
                cudaStreamDestroy(stream),
                "EPMultiPassCUDA: cudaStreamDestroy failed");
        }
    }

    ExecutionStream(const ExecutionStream&)            = delete;
    ExecutionStream& operator=(const ExecutionStream&) = delete;
    ExecutionStream(ExecutionStream&&)                 = delete;
    ExecutionStream& operator=(ExecutionStream&&)      = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream; }
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> class PruneCUDAImpl {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;
    PruneCUDAImpl(memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
                  search::PulsarSearchConfig cfg,
                  std::span<const float> threshold_scheme,
                  SizeType max_sugg,
                  SizeType batch_size,
                  SizeType branch_max,
                  std::string_view poly_basis,
                  int device_id,
                  cudaStream_t stream)
        : m_workspace_ptr(&workspace),
          m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_branch_max(branch_max),
          m_poly_basis(poly_basis),
          m_total_levels(m_threshold_scheme.size()),
          m_device_id(device_id),
          m_stream(stream) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        error_check::check_less_equal(m_cfg.get_nparams(), 5,
                                      "Pruning not supported for nparams > 5.");
        m_prune_funcs = core::create_prune_dp_functs_cuda<FoldTypeCUDA>(
            m_poly_basis, m_ffa_plan.get_param_counts().back(),
            m_ffa_plan.get_dparams_lim().back(),
            m_ffa_plan.get_nsegments().back(),
            m_ffa_plan.get_tsegments().back(), m_cfg, m_batch_size,
            m_branch_max);
    }

    ~PruneCUDAImpl()                               = default;
    PruneCUDAImpl(const PruneCUDAImpl&)            = delete;
    PruneCUDAImpl& operator=(const PruneCUDAImpl&) = delete;
    PruneCUDAImpl(PruneCUDAImpl&&)                 = delete;
    PruneCUDAImpl& operator=(PruneCUDAImpl&&)      = delete;

    SizeType get_memory_usage() const noexcept {
        return get_workspace().get_memory_usage_gib();
    }

    void execute(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                 SizeType ref_seg,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file,
                 int task_id) {
        timing::SimpleTimer timer;
        timer.start();
        const std::string run_name =
            std::format("{:03d}_{:02d}", ref_seg, task_id);

        // Log detailed memory usage
        const auto& ws                   = get_workspace();
        const auto memory_workspace_gb   = ws.prune.get_memory_usage_gib();
        const auto memory_suggestions_gb = ws.world_tree.get_memory_usage_gib();
        spdlog::info("Pruning run {:03d}: Memory Usage: {:.2f} GB "
                     "(suggestions) + {:.2f} GB (workspace)",
                     ref_seg, memory_suggestions_gb, memory_workspace_gb);

        // Setup log and result files
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / std::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / std::format("tmp_{}_results.h5", run_name));

        const auto nsegments = m_ffa_plan.get_nsegments().back();

        initialize(ffa_fold, ref_seg, m_stream);
        spdlog::info("Pruning run {:03d}: initialized", ref_seg);

        SizeType iterations_completed = 0;
        for (SizeType iter = 0; iter < nsegments - 1; ++iter) {
            execute_iteration(ffa_fold, m_stream);
            iterations_completed = iter + 1;
            // Check for early termination (no survivors)
            if (m_prune_complete) {
                break;
            }
        }

        // Log early exit ONCE after loop if needed
        if (m_prune_complete && iterations_completed < nsegments - 1) {
            spdlog::info(
                "Pruning terminated early at iteration {} - no survivors",
                iterations_completed);
        }

        // Transform the suggestion params to middle of the data
        const auto coord_mid = m_snail_scheme.get_coord(m_prune_level);

        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        auto& ws_final         = get_workspace();
        auto& world_tree_final = ws_final.world_tree;
        memory::CircularViewCUDA<double> leaves_view =
            world_tree_final.get_leaves_circular_view();
        memory::CircularViewCUDA<float> scores_view =
            world_tree_final.get_scores_circular_view();
        const auto leaves_stride = world_tree_final.get_leaves_stride();
        const auto n_leaves      = world_tree_final.get_size();
        const auto n1            = leaves_view.first.size() / leaves_stride;
        const auto n2            = leaves_view.second.size() / leaves_stride;
        m_prune_funcs->report(leaves_view.first, coord_mid, n1, m_stream);
        if (n2 > 0) {
            m_prune_funcs->report(leaves_view.second, coord_mid, n2, m_stream);
        }
        std::vector<double> leaves_report_h(n_leaves * leaves_stride);
        std::vector<float> scores_report_h(n_leaves);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_report_h.data(), leaves_view.first.data(),
                            (n1 * leaves_stride) * sizeof(double),
                            cudaMemcpyDeviceToHost, m_stream),
            "cudaMemcpyAsync leaves_report failed");
        if (n2 > 0) {
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(leaves_report_h.data() + (n1 * leaves_stride),
                                leaves_view.second.data(),
                                (n2 * leaves_stride) * sizeof(double),
                                cudaMemcpyDeviceToHost, m_stream),
                "cudaMemcpyAsync leaves_report failed");
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(scores_report_h.data(), scores_view.first.data(),
                            n1 * sizeof(float), cudaMemcpyDeviceToHost,
                            m_stream),
            "cudaMemcpyAsync scores_report part 1 failed");
        if (n2 > 0) {
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(scores_report_h.data() + n1,
                                scores_view.second.data(), n2 * sizeof(float),
                                cudaMemcpyDeviceToHost, m_stream),
                "cudaMemcpyAsync scores_report part 2 failed");
        }

        cuda_utils::check_cuda_call(
            cudaStreamSynchronize(m_stream),
            "cudaStreamSynchronize write results failed");

        // Both host vectors are now contiguous — wrap as single-span views
        memory::CircularView<double> leaves_report_view{
            std::span<double>(leaves_report_h), std::span<double>{}};
        memory::CircularView<float> scores_report_view{
            std::span<float>(scores_report_h), std::span<float>{}};
        result_writer.write_run_results(
            run_name, m_snail_scheme.get_data(), leaves_report_view,
            scores_report_view, n_leaves, m_cfg.get_nparams(), m_pstats);

        // Final log entries
        std::ofstream log(actual_log_file, std::ios::app);
        log << std::format("Pruning log for ref segment: {}\n", ref_seg);
        log << m_pstats.get_all_summaries();
        log << std::format("Pruning run complete for ref segment {}\n",
                           ref_seg);
        log.close();

        spdlog::info("Pruning run {:03d}: complete", ref_seg);
        spdlog::info("Pruning run {:03d}: stats: {}", ref_seg,
                     m_pstats.get_stats_summary_cuda(timer.stop()));
    }

private:
    // The observer pointer that always points to the active workspace.
    memory::EPWorkspaceCUDA<FoldTypeCUDA>* m_workspace_ptr{nullptr};

    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan<HostFoldT> m_ffa_plan;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    SizeType m_branch_max;
    std::string_view m_poly_basis;
    SizeType m_total_levels;
    int m_device_id;
    cudaStream_t m_stream{nullptr};

    bool m_prune_complete{false};
    SizeType m_prune_level{};
    psr_utils::MiddleOutScheme m_snail_scheme;
    cands::PruneStatsCollection m_pstats;
    std::unique_ptr<core::PruneDPFunctsCUDA<FoldTypeCUDA>> m_prune_funcs;

    [[nodiscard]] memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    void initialize(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                    SizeType ref_seg,
                    cudaStream_t stream) {
        //  Reset the suggestion buffer state
        auto& ws         = get_workspace();
        auto& world_tree = ws.world_tree;
        world_tree.reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        const auto tseg      = m_ffa_plan.get_tsegments().back();
        m_snail_scheme = psr_utils::MiddleOutScheme(nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;

        // Initialize the world tree with the first segment
        const auto fold_segment =
            m_prune_funcs->load_segment(ffa_fold, m_snail_scheme.get_ref_idx());
        const auto coord_init = m_snail_scheme.get_coord(m_prune_level);
        const auto n_leaves   = m_ffa_plan.get_ncoords().back();
        m_prune_funcs->seed(fold_segment, cuda_utils::as_span(ws.seed_leaves_d),
                            cuda_utils::as_span(ws.seed_scores_d), coord_init,
                            stream);
        // Initialize the WorldTree with the generated data
        world_tree.add_initial(
            cuda_utils::as_span(ws.seed_leaves_d), fold_segment,
            cuda_utils::as_span(ws.seed_scores_d), n_leaves, stream);

        // Initialize the prune stats
        m_pstats = cands::PruneStatsCollection();
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = m_snail_scheme.get_segment_idx(m_prune_level),
            .threshold     = 0,
            .score_min     = world_tree.get_score_min(stream),
            .score_max     = world_tree.get_score_max(stream),
            .n_branches    = world_tree.get_size(),
            .n_leaves      = world_tree.get_size(),
            .n_leaves_phy  = world_tree.get_size(),
            .n_leaves_surv = world_tree.get_size(),
        };
        m_pstats.update_stats(pstats_cur);
    }

    void execute_iteration(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                           cudaStream_t stream) {
        if (m_prune_complete) {
            return;
        }
        ++m_prune_level;
        error_check::check_less_equal(
            m_prune_level, m_total_levels,
            std::format("Pruning complete - exceeded total levels at level "
                        "{}",
                        m_prune_level));
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

        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats,
                                  stream);

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
        m_pstats.update_stats(pstats_cur);

        // Check if no survivors
        if (world_tree.get_size() == 0) {
            m_prune_complete = true;
            return;
        }
    }

    // Iteration flow: Branch -> Validate -> Resolve -> Load/Shift/Add -> Score
    // -> Filter -> Transform -> Add to buffer.
    void execute_iteration_batched(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                                   SizeType seg_idx_cur,
                                   float threshold,
                                   IterationStats& stats,
                                   cudaStream_t stream) {

        auto& ws            = get_workspace();
        auto& world_tree    = ws.world_tree;
        auto& prune_ws      = ws.prune;
        auto& scratch_ws    = ws.scratch;
        auto branch_ws_view = ws.branch.get_view();

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
            const auto n_leaves_batch = m_prune_funcs->branch(
                leaves_tree_span,
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                cuda_utils::as_span(prune_ws.validation_mask_d), coord_cur,
                coord_prev, current_batch_size, branch_ws_view, scratch_ws,
                stream);
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
            const auto n_leaves_after_validation = m_prune_funcs->validate(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                cuda_utils::as_span(prune_ws.validation_mask_d), coord_cur,
                n_leaves_batch, scratch_ws, stream);
            stats.n_leaves_phy += n_leaves_after_validation;
            if (n_leaves_after_validation == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }

            // Resolve kernel
            m_prune_funcs->resolve(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.validation_mask_d),
                cuda_utils::as_span(prune_ws.branched_param_idx_d),
                cuda_utils::as_span(prune_ws.branched_phase_shift_d), coord_add,
                coord_cur, coord_init, n_leaves_batch, stream);

            // Shift and add kernel
            // branched_indices_d are logical indices, convert to physical
            // indices in shift_add kernel
            m_prune_funcs->shift_add(
                world_tree.get_folds_span(),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                cuda_utils::as_span(prune_ws.validation_mask_d),
                ffa_fold_segment,
                cuda_utils::as_span(prune_ws.branched_param_idx_d),
                cuda_utils::as_span(prune_ws.branched_phase_shift_d),
                cuda_utils::as_span(prune_ws.branched_folds_d), n_leaves_batch,
                world_tree.get_physical_start_idx(), world_tree.get_capacity(),
                stream);

            // Score and prune kernel
            const auto n_leaves_passing = m_prune_funcs->score_and_filter(
                cuda_utils::as_span(prune_ws.branched_folds_d),
                cuda_utils::as_span(prune_ws.branched_scores_d),
                cuda_utils::as_span(prune_ws.validation_mask_d),
                cuda_utils::as_span(prune_ws.filtered_mask_d),
                current_threshold, n_leaves_batch, scratch_ws, stream);

            // Compute min and max scores
            MinMaxFloat minmax_scores;
            scratch_ws.compute_min_max_scores(
                cuda_utils::as_span(prune_ws.branched_scores_d),
                cuda_utils::as_span(prune_ws.validation_mask_d), &minmax_scores,
                n_leaves_batch, stream);
            stats.score_min = std::min(stats.score_min, minmax_scores.min);
            stats.score_max = std::max(stats.score_max, minmax_scores.max);

            if (n_leaves_passing == 0) {
                world_tree.consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_passing, prune_ws.max_branched_leaves,
                "n_leaves_passing <= max_branched_leaves");

            // Transform kernel
            m_prune_funcs->transform(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.filtered_mask_d), coord_next,
                coord_cur, n_leaves_batch, stream);

            // Convert validation mask to indices
            scratch_ws.convert_mask_to_indices(
                cuda_utils::as_span(prune_ws.filtered_mask_d),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                n_leaves_batch, stream);

            // Add batch to output suggestions
            current_threshold = world_tree.add_batch_scattered(
                cuda_utils::as_span(prune_ws.branched_leaves_d),
                cuda_utils::as_span(prune_ws.branched_folds_d),
                cuda_utils::as_span(prune_ws.branched_scores_d),
                cuda_utils::as_span(prune_ws.branched_indices_d),
                current_threshold, n_leaves_passing, stream);

            // Notify the buffer that a batch of the old suggestions has been
            // consumed
            world_tree.consume_read(current_batch_size);
        }
    }

}; // End PruneCUDAImpl implementation

} // End of anonymous namespace

// EPMultiPassCUDA::Impl implementation
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class EPMultiPassCUDA<FoldTypeCUDA>::Impl {
public:
    using HostFoldT = HostFoldType<FoldTypeCUDA>;

    Impl(search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         int device_id)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_device_id(device_id),
          m_execution_stream(device_id),
          m_ffa_plan(m_cfg),
          m_branching_pattern(m_ffa_plan.get_branching_pattern(m_poly_basis)),
          m_branch_max(
              std::max(static_cast<SizeType>(std::ceil(
                           *std::ranges::max_element(m_branching_pattern) * 2)),
                       32UL)),
          m_workspace_storage(m_batch_size,
                              m_branch_max,
                              m_max_sugg,
                              m_ffa_plan.get_ncoords().back(),
                              m_cfg.get_nparams(),
                              std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                  ? m_cfg.get_nbins_f()
                                  : m_cfg.get_nbins(),
                              m_execution_stream.get()),
          m_workspace_ptr(&m_workspace_storage) {
        // Validate workspace
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        const auto nbins       = std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                     ? m_cfg.get_nbins_f()
                                     : m_cfg.get_nbins();
        const auto& ws         = get_workspace();
        ws.validate(m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                    m_cfg.get_nparams(), nbins);
    }

    Impl(memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
         cudaStream_t execution_stream,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         std::optional<SizeType> n_runs,
         std::optional<std::vector<SizeType>> ref_segs,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         int device_id)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_device_id(device_id),
          m_execution_stream(execution_stream),
          m_ffa_plan(m_cfg),
          m_workspace_storage(),
          m_workspace_ptr(&workspace) {
        // Create branching pattern and branch max
        m_branching_pattern   = m_ffa_plan.get_branching_pattern(m_poly_basis);
        const auto branch_max = *std::ranges::max_element(m_branching_pattern);
        m_branch_max =
            std::max(static_cast<SizeType>(std::ceil(branch_max * 2)), 32UL);

        // Validate workspaces
        const auto& ws         = get_workspace();
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        const SizeType nbins   = std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                                     ? m_cfg.get_nbins_f()
                                     : m_cfg.get_nbins();
        ws.validate(m_batch_size, m_branch_max, m_max_sugg, ncoords_ffa,
                    m_cfg.get_nparams(), nbins);
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
        spdlog::info("EPMultiPassCUDA: Initializing with FFA");
        // Create appropriate FFA fold
        std::tuple<thrust::device_vector<FoldTypeCUDA>,
                   plans::FFAPlan<HostFoldT>>
            result = compute_ffa_cuda_device<FoldTypeCUDA>(ts_e, ts_v, m_cfg,
                                                           m_device_id);
        const thrust::device_vector<FoldTypeCUDA> ffa_fold_d =
            std::get<0>(result);
        plans::FFAPlan<HostFoldT> ffa_plan = std::move(std::get<1>(result));
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
                std::format("EPMultiPassCUDA::execute: Failed to create output "
                            "directory '{}': {}",
                            outdir.string(), ec.message()));
        }

        // Determine ref_segs to process
        auto ref_segs_to_process =
            utils::determine_ref_segs(nsegments, m_n_runs, m_ref_segs);
        spdlog::info("Starting Pruning for {} runs, on CUDA device {}",
                     ref_segs_to_process.size(), m_device_id);

        // Initialize log file
        std::ofstream log(log_file);
        log << "Pruning log\n";
        log.close();

        // Write metadata to result file
        auto writer = cands::PruneResultWriter(
            result_file, cands::PruneResultWriter::Mode::kWrite);
        writer.write_metadata(m_cfg.get_param_names(), nsegments, m_max_sugg,
                              m_threshold_scheme);
        auto& ws   = get_workspace();
        auto prune = PruneCUDAImpl<FoldTypeCUDA>(
            ws, m_cfg, m_threshold_scheme, m_max_sugg, m_batch_size,
            m_branch_max, m_poly_basis, m_device_id, m_execution_stream.get());
        for (const auto ref_seg : ref_segs_to_process) {
            prune.execute(cuda_utils::as_span(ffa_fold_d), ref_seg, outdir,
                          log_file, result_file,
                          /*task_id=*/0);
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
    std::string m_poly_basis;
    int m_device_id;

    plans::FFAPlan<HostFoldT> m_ffa_plan;
    std::vector<double> m_branching_pattern;
    SizeType m_branch_max{0};

    // EP workspace ownership
    ExecutionStream m_execution_stream;
    memory::EPWorkspaceCUDA<FoldTypeCUDA> m_workspace_storage;
    // The observer pointer that always points to the active workspace.
    memory::EPWorkspaceCUDA<FoldTypeCUDA>* m_workspace_ptr{nullptr};

    [[nodiscard]] memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::EPWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

}; // End EPMultiPassCUDATypedImpl implementation

// --- Definitions for EPMultiPassCUDA ---
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCUDA<FoldTypeCUDA>::EPMultiPassCUDA(
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    int device_id)
    : m_impl(std::make_unique<Impl>(std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCUDA<FoldTypeCUDA>::EPMultiPassCUDA(
    memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
    cudaStream_t execution_stream,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    int device_id)
    : m_impl(std::make_unique<Impl>(workspace,
                                    execution_stream,
                                    std::move(cfg),
                                    threshold_scheme,
                                    n_runs,
                                    std::move(ref_segs),
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCUDA<FoldTypeCUDA>::~EPMultiPassCUDA() = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCUDA<FoldTypeCUDA>::EPMultiPassCUDA(
    EPMultiPassCUDA<FoldTypeCUDA>&& other) noexcept = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPMultiPassCUDA<FoldTypeCUDA>& EPMultiPassCUDA<FoldTypeCUDA>::operator=(
    EPMultiPassCUDA<FoldTypeCUDA>&& other) noexcept = default;

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void EPMultiPassCUDA<FoldTypeCUDA>::execute(std::span<const float> ts_e,
                                            std::span<const float> ts_v,
                                            const std::filesystem::path& outdir,
                                            std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}

// Explicit instantiation
template class EPMultiPassCUDA<float>;
template class EPMultiPassCUDA<ComplexTypeCUDA>;

} // namespace loki::algorithms