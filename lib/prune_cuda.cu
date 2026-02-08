#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
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

class PruningManagerCUDA::BaseImpl {
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
                         std::string_view poly_basis) = 0;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PruningManagerCUDATypedImpl final : public PruningManagerCUDA::BaseImpl {
public:
    using HostFoldType = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    PruningManagerCUDATypedImpl(search::PulsarSearchConfig cfg,
                                std::span<const float> threshold_scheme,
                                std::optional<SizeType> n_runs,
                                std::optional<std::vector<SizeType>> ref_segs,
                                SizeType max_sugg,
                                SizeType batch_size,
                                int device_id)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_n_runs(n_runs),
          m_ref_segs(std::move(ref_segs)),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_device_id(device_id) {}

    ~PruningManagerCUDATypedImpl() final                            = default;
    PruningManagerCUDATypedImpl(const PruningManagerCUDATypedImpl&) = delete;
    PruningManagerCUDATypedImpl&
    operator=(const PruningManagerCUDATypedImpl&)              = delete;
    PruningManagerCUDATypedImpl(PruningManagerCUDATypedImpl&&) = delete;
    PruningManagerCUDATypedImpl&
    operator=(PruningManagerCUDATypedImpl&&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix,
                 std::string_view poly_basis) override {
        spdlog::info("PruningManagerCUDA: Initializing with FFA");
        // Create appropriate FFA fold
        std::tuple<thrust::device_vector<FoldTypeCUDA>,
                   plans::FFAPlan<HostFoldType>>
            result = compute_ffa_cuda_device<FoldTypeCUDA>(ts_e, ts_v, m_cfg,
                                                           m_device_id);
        const thrust::device_vector<FoldTypeCUDA> ffa_fold_d =
            std::get<0>(result);
        plans::FFAPlan<HostFoldType> ffa_plan = std::move(std::get<1>(result));
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
            throw std::runtime_error(std::format(
                "PruningManagerCUDA::execute: Failed to create output "
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
        auto prune = PruneCUDA<FoldTypeCUDA>(
            std::move(ffa_plan), m_cfg, m_threshold_scheme, m_max_sugg,
            m_batch_size, poly_basis, m_device_id);
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
    int m_device_id;

}; // End PruningManagerCUDATypedImpl implementation

struct IterationStats {
    SizeType n_leaves     = 0;
    SizeType n_leaves_phy = 0;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct PruningWorkspaceCUDA {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size;
    SizeType branch_max;
    SizeType nparams;
    SizeType nbins;
    SizeType max_branched_leaves;
    SizeType leaves_stride{};
    SizeType folds_stride;

    thrust::device_vector<double> branched_leaves_d;
    thrust::device_vector<FoldTypeCUDA> branched_folds_d;
    thrust::device_vector<float> branched_scores_d;
    // Scratch space for indices
    thrust::device_vector<uint32_t> branched_indices_d;
    // Scratch space for resolving parameters
    thrust::device_vector<uint32_t> branched_param_idx_d;
    thrust::device_vector<float> branched_phase_shift_d;

    PruningWorkspaceCUDA(SizeType batch_size,
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
          branched_leaves_d(max_branched_leaves * leaves_stride),
          branched_folds_d(max_branched_leaves * folds_stride),
          branched_scores_d(max_branched_leaves),
          branched_indices_d(max_branched_leaves),
          branched_param_idx_d(max_branched_leaves),
          branched_phase_shift_d(max_branched_leaves) {}

    float get_memory_usage() const noexcept {
        const auto total_memory =
            (branched_leaves_d.size() * sizeof(double)) +
            (branched_folds_d.size() * sizeof(FoldTypeCUDA)) +
            (branched_scores_d.size() * sizeof(float)) +
            (branched_indices_d.size() * sizeof(uint32_t)) +
            (branched_param_idx_d.size() * sizeof(uint32_t)) +
            (branched_phase_shift_d.size() * sizeof(float));
        return static_cast<float>(total_memory) /
               static_cast<float>(1ULL << 30U);
    }
}; // End PruningWorkspace definition

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PruneCUDA<FoldTypeCUDA>::Impl {
public:
    using HostFoldType   = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = typename FoldTypeTraits<FoldTypeCUDA>::DeviceType;
    Impl(plans::FFAPlan<HostFoldType> ffa_plan,
         search::PulsarSearchConfig cfg,
         std::span<const float> threshold_scheme,
         SizeType max_sugg,
         SizeType batch_size,
         std::string_view poly_basis,
         int device_id)
        : m_ffa_plan(std::move(ffa_plan)),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_poly_basis(poly_basis),
          m_total_levels(m_threshold_scheme.size()),
          m_device_id(device_id) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        // Setup pruning functions
        setup_pruning();

        const auto max_batch_size = m_batch_size * m_branch_max;
        // Allocate suggestion buffer
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            m_world_tree = std::make_unique<utils::WorldTreeCUDA<FoldTypeCUDA>>(
                m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins_f(),
                max_batch_size);
        } else {
            m_world_tree = std::make_unique<utils::WorldTreeCUDA<FoldTypeCUDA>>(
                m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins(),
                max_batch_size);
        }

        // Allocate iteration workspace
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            m_pruning_workspace =
                std::make_unique<PruningWorkspaceCUDA<FoldTypeCUDA>>(
                    m_batch_size, m_branch_max, m_cfg.get_nparams(),
                    m_cfg.get_nbins_f());
        } else {
            m_pruning_workspace =
                std::make_unique<PruningWorkspaceCUDA<FoldTypeCUDA>>(
                    m_batch_size, m_branch_max, m_cfg.get_nparams(),
                    m_cfg.get_nbins());
        }

        m_branching_workspace = std::make_unique<utils::BranchingWorkspaceCUDA>(
            m_batch_size, m_branch_max, m_cfg.get_nparams());

        // Allocate buffers for seeding the world tree and scoring
        const auto ncoords_ffa = m_ffa_plan.get_ncoords().back();
        m_seed_leaves_d.resize(ncoords_ffa * m_world_tree->get_leaves_stride());
        m_seed_scores_d.resize(ncoords_ffa);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    SizeType get_memory_usage() const noexcept {
        return m_pruning_workspace->get_memory_usage() +
               m_world_tree->get_memory_usage();
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
        const auto memory_workspace_gb =
            m_pruning_workspace->get_memory_usage();
        const auto memory_suggestions_gb = m_world_tree->get_memory_usage();
        spdlog::info("Pruning run {:03d}: Memory Usage: {:.2f} GB "
                     "(suggestions) + {:.2f} GB (workspace)",
                     ref_seg, memory_suggestions_gb, memory_workspace_gb);

        // Setup log and result files
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / std::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / std::format("tmp_{}_results.h5", run_name));

        const auto nsegments = m_ffa_plan.get_nsegments().back();

        // cudaStream_t stream = nullptr;
        cudaStream_t stream = nullptr;
        cuda_utils::check_cuda_call(cudaStreamCreate(&stream),
                                    "cudaStreamCreate failed");
        initialize(ffa_fold, ref_seg, stream);
        spdlog::info("Pruning run {:03d}: initialized", ref_seg);

        SizeType iterations_completed = 0;
        for (SizeType iter = 0; iter < nsegments - 1; ++iter) {
            execute_iteration(ffa_fold, stream);
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
        const auto coord_mid = m_snail_scheme->get_coord(m_prune_level);

        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        auto leaves_report_d_span =
            m_world_tree->get_leaves_contiguous_span(stream);
        m_prune_funcs->report(leaves_report_d_span, coord_mid,
                              m_world_tree->get_size(), stream);
        auto scores_report_d_span =
            m_world_tree->get_scores_contiguous_span(stream);
        std::vector<double> leaves_report_h(leaves_report_d_span.size());
        std::vector<float> scores_report_h(scores_report_d_span.size());
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(scores_report_h.data(), scores_report_d_span.data(),
                            scores_report_d_span.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync scores_report failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_report_h.data(), leaves_report_d_span.data(),
                            leaves_report_d_span.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync leaves_report failed");
        cuda_utils::check_cuda_call(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize write results failed");
        cuda_utils::check_cuda_call(cudaStreamDestroy(stream),
                                    "cudaStreamDestroy failed");
        result_writer.write_run_results(run_name, m_snail_scheme->get_data(),
                                        leaves_report_h, scores_report_h,
                                        m_world_tree->get_size(),
                                        m_cfg.get_nparams(), *m_pstats);

        // Final log entries
        std::ofstream log(actual_log_file, std::ios::app);
        log << std::format("Pruning log for ref segment: {}\n", ref_seg);
        log << m_pstats->get_all_summaries();
        log << std::format("Pruning run complete for ref segment {}\n",
                           ref_seg);
        log.close();

        spdlog::info("Pruning run {:03d}: complete", ref_seg);
        spdlog::info("Pruning run {:03d}: stats: {}", ref_seg,
                     m_pstats->get_stats_summary_cuda(timer.stop()));
    }

private:
    plans::FFAPlan<HostFoldType> m_ffa_plan;
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string_view m_poly_basis;
    SizeType m_total_levels;
    int m_device_id;

    std::vector<double> m_branching_pattern;
    SizeType m_branch_max{0};
    bool m_prune_complete{false};
    SizeType m_prune_level{};
    std::unique_ptr<psr_utils::MiddleOutScheme> m_snail_scheme;
    std::unique_ptr<core::PruneDPFunctsCUDA<FoldTypeCUDA>> m_prune_funcs;
    std::unique_ptr<cands::PruneStatsCollection> m_pstats;

    std::unique_ptr<utils::WorldTreeCUDA<FoldTypeCUDA>> m_world_tree;
    std::unique_ptr<PruningWorkspaceCUDA<FoldTypeCUDA>> m_pruning_workspace;
    std::unique_ptr<utils::BranchingWorkspaceCUDA> m_branching_workspace;

    // Buffers for seeding the world tree and scoring
    thrust::device_vector<double> m_seed_leaves_d;
    thrust::device_vector<float> m_seed_scores_d;

    // Counter for tracking the number of passing leaves
    utils::DeviceCounter m_passing_counter;

    void initialize(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                    SizeType ref_seg,
                    cudaStream_t stream) {
        //  Reset the suggestion buffer state
        m_world_tree->reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        const auto tseg      = m_ffa_plan.get_tsegments().back();
        m_snail_scheme       = std::make_unique<psr_utils::MiddleOutScheme>(
            nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;

        // Initialize the world tree with the first segment
        const auto fold_segment = m_prune_funcs->load_segment(
            ffa_fold, m_snail_scheme->get_ref_idx());
        const auto coord_init = m_snail_scheme->get_coord(m_prune_level);
        const auto n_leaves   = m_ffa_plan.get_ncoords().back();
        m_prune_funcs->seed(fold_segment, cuda_utils::as_span(m_seed_leaves_d),
                            cuda_utils::as_span(m_seed_scores_d), coord_init,
                            stream);
        // Initialize the WorldTree with the generated data
        m_world_tree->add_initial(
            cuda_utils::as_span(m_seed_leaves_d), fold_segment,
            cuda_utils::as_span(m_seed_scores_d), n_leaves, stream);

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
        // Prepare for in-place update: mark start of write region, reset size
        // for new suggestions.
        m_world_tree->prepare_in_place_update();

        IterationStats stats;
        const auto seg_idx_cur = m_snail_scheme->get_segment_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches = m_world_tree->get_size_old();

        execute_iteration_batched(ffa_fold, seg_idx_cur, threshold, stats,
                                  stream);

        // Finalize: make new region active, defragment for contiguous access.
        m_world_tree->finalize_in_place_update();

        // Update statistics
        const cands::PruneStats pstats_cur{
            .level         = m_prune_level,
            .seg_idx       = seg_idx_cur,
            .threshold     = threshold,
            .score_min     = m_world_tree->get_score_min(),
            .score_max     = m_world_tree->get_score_max(),
            .n_branches    = n_branches,
            .n_leaves      = stats.n_leaves,
            .n_leaves_phy  = stats.n_leaves_phy,
            .n_leaves_surv = m_world_tree->get_size(),
        };
        m_pstats->update_stats(pstats_cur);

        // Check if no survivors
        if (m_world_tree->get_size() == 0) {
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
            const auto n_leaves_batch = m_prune_funcs->branch(
                leaves_tree_span,
                cuda_utils::as_span(m_pruning_workspace->branched_leaves_d),
                cuda_utils::as_span(m_pruning_workspace->branched_indices_d),
                coord_cur, coord_prev, current_batch_size, branch_ws, stream);
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
            const auto n_leaves_after_validation = m_prune_funcs->validate(
                cuda_utils::as_span(m_pruning_workspace->branched_leaves_d),
                cuda_utils::as_span(m_pruning_workspace->branched_indices_d),
                coord_cur, n_leaves_batch, stream);
            stats.n_leaves_phy += n_leaves_after_validation;
            if (n_leaves_after_validation == 0) {
                m_world_tree->consume_read(current_batch_size);
                continue;
            }

            // Resolve kernel
            m_prune_funcs->resolve(
                cuda_utils::as_span(m_pruning_workspace->branched_leaves_d),
                cuda_utils::as_span(m_pruning_workspace->branched_param_idx_d),
                cuda_utils::as_span(
                    m_pruning_workspace->branched_phase_shift_d),
                coord_add, coord_cur, coord_init, n_leaves_after_validation,
                stream);

            // Shift and add kernel
            // branched_indices_d are logical indices, convert to physical
            // indices in shift_add kernel
            m_prune_funcs->shift_add(
                m_world_tree->get_folds_span(),
                cuda_utils::as_span(m_pruning_workspace->branched_indices_d),
                ffa_fold_segment,
                cuda_utils::as_span(m_pruning_workspace->branched_param_idx_d),
                cuda_utils::as_span(
                    m_pruning_workspace->branched_phase_shift_d),
                cuda_utils::as_span(m_pruning_workspace->branched_folds_d),
                n_leaves_after_validation,
                m_world_tree->get_physical_start_idx(),
                m_world_tree->get_capacity(), stream);

            // Score and prune kernel
            const auto n_leaves_passing = m_prune_funcs->score_and_filter(
                cuda_utils::as_span(m_pruning_workspace->branched_folds_d),
                cuda_utils::as_span(m_pruning_workspace->branched_scores_d),
                cuda_utils::as_span(m_pruning_workspace->branched_indices_d),
                current_threshold, n_leaves_after_validation, m_passing_counter,
                stream);

            if (n_leaves_passing == 0) {
                m_world_tree->consume_read(current_batch_size);
                continue;
            }
            error_check::check_less_equal(
                n_leaves_passing, m_pruning_workspace->max_branched_leaves,
                "n_leaves_passing <= max_branched_leaves");

            // Transform kernel
            m_prune_funcs->transform(
                cuda_utils::as_span(m_pruning_workspace->branched_leaves_d),
                cuda_utils::as_span(m_pruning_workspace->branched_indices_d),
                coord_next, coord_cur, n_leaves_passing, stream);

            // Add batch to output suggestions
            current_threshold = m_world_tree->add_batch_scattered(
                cuda_utils::as_span(m_pruning_workspace->branched_leaves_d),
                cuda_utils::as_span(m_pruning_workspace->branched_folds_d),
                cuda_utils::as_span(m_pruning_workspace->branched_scores_d),
                cuda_utils::as_span(m_pruning_workspace->branched_indices_d),
                current_threshold, n_leaves_passing, stream);

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
        m_prune_funcs = core::create_prune_dp_functs_cuda<FoldTypeCUDA>(
            m_poly_basis, m_ffa_plan.get_param_counts().back(),
            m_ffa_plan.get_dparams_lim().back(),
            m_ffa_plan.get_nsegments().back(),
            m_ffa_plan.get_tsegments().back(), m_cfg, m_batch_size,
            m_branch_max);
    }

}; // End PruneCUDA::Impl implementation

PruningManagerCUDA::PruningManagerCUDA(
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    int device_id) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<PruningManagerCUDATypedImpl<ComplexTypeCUDA>>(
            std::move(cfg), threshold_scheme, n_runs, std::move(ref_segs),
            max_sugg, batch_size, device_id);
    } else {
        m_impl = std::make_unique<PruningManagerCUDATypedImpl<float>>(
            std::move(cfg), threshold_scheme, n_runs, std::move(ref_segs),
            max_sugg, batch_size, device_id);
    }
}
PruningManagerCUDA::~PruningManagerCUDA() = default;
PruningManagerCUDA::PruningManagerCUDA(PruningManagerCUDA&& other) noexcept =
    default;
PruningManagerCUDA&
PruningManagerCUDA::operator=(PruningManagerCUDA&& other) noexcept = default;
void PruningManagerCUDA::execute(std::span<const float> ts_e,
                                 std::span<const float> ts_v,
                                 const std::filesystem::path& outdir,
                                 std::string_view file_prefix,
                                 std::string_view poly_basis) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix, poly_basis);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneCUDA<FoldTypeCUDA>::PruneCUDA(
    plans::FFAPlan<typename FoldTypeTraits<FoldTypeCUDA>::HostType> ffa_plan,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    int device_id)
    : m_impl(std::make_unique<Impl>(std::move(ffa_plan),
                                    std::move(cfg),
                                    threshold_scheme,
                                    max_sugg,
                                    batch_size,
                                    poly_basis,
                                    device_id)) {}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneCUDA<FoldTypeCUDA>::~PruneCUDA() = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneCUDA<FoldTypeCUDA>::PruneCUDA(PruneCUDA&& other) noexcept = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneCUDA<FoldTypeCUDA>&
PruneCUDA<FoldTypeCUDA>::operator=(PruneCUDA&& other) noexcept = default;

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType PruneCUDA<FoldTypeCUDA>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneCUDA<FoldTypeCUDA>::execute(
    cuda::std::span<const FoldTypeCUDA> ffa_fold,
    SizeType ref_seg,
    const std::filesystem::path& outdir,
    const std::optional<std::filesystem::path>& log_file,
    const std::optional<std::filesystem::path>& result_file,
    int task_id) {
    m_impl->execute(ffa_fold, ref_seg, outdir, log_file, result_file, task_id);
}

template class PruneCUDA<float>;
template class PruneCUDA<ComplexTypeCUDA>;

} // namespace loki::algorithms