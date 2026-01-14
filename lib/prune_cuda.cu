#include "loki/algorithms/prune.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <utility>

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/core/dynamic.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/progress.hpp"
#include "loki/psr_utils.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"
#include "loki/utils/suggestions.hpp"

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
                         std::string_view kind,
                         bool show_progress) = 0;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PruningManagerCUDATypedImpl final : public PruningManagerCUDA::BaseImpl {
public:
    using HostFoldType = typename FoldTypeTraits<FoldTypeCUDA>::HostType;
    PruningManagerCUDATypedImpl(search::PulsarSearchConfig cfg,
                                const std::vector<float>& threshold_scheme,
                                std::optional<SizeType> n_runs,
                                std::optional<std::vector<SizeType>> ref_segs,
                                SizeType max_sugg,
                                SizeType batch_size,
                                int device_id)
        : m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme),
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
                 std::string_view kind,
                 bool show_progress) override {
        spdlog::info("PruningManagerCUDA: Initializing with FFA");
        // Create appropriate FFA fold
        std::tuple<thrust::device_vector<FoldTypeCUDA>,
                   plans::FFAPlan<HostFoldType>>
            result = compute_ffa_cuda_device<FoldTypeCUDA>(ts_e, ts_v, m_cfg,
                                                           m_device_id);
        const thrust::device_vector<FoldTypeCUDA> ffa_fold_d =
            std::get<0>(result);
        const plans::FFAPlan<HostFoldType> ffa_plan = std::get<1>(result);
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

        auto prune =
            PruneCUDA<FoldTypeCUDA>(ffa_plan, m_cfg, m_threshold_scheme,
                                    m_max_sugg, m_batch_size, kind);
        for (const auto ref_seg : ref_segs_to_process) {
            prune.execute(cuda::std::span<const FoldTypeCUDA>(
                              thrust::raw_pointer_cast(ffa_fold_d.data()),
                              ffa_fold_d.size()),
                          ref_seg, outdir, log_file, result_file,
                          show_progress);
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

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct PruningWorkspaceCUDA {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType max_batch_size;
    SizeType nparams;
    SizeType nbins;
    SizeType leaves_stride{};
    SizeType folds_stride;

    thrust::device_vector<double> batch_leaves_d;
    thrust::device_vector<FoldTypeCUDA> batch_folds_d;
    thrust::device_vector<float> batch_scores_d;
    thrust::device_vector<SizeType> batch_param_idx_d;
    thrust::device_vector<float> batch_phase_shift_d;
    thrust::device_vector<SizeType> batch_isuggest_d;
    thrust::device_vector<SizeType> batch_passing_indices_d;

    PruningWorkspaceCUDA(SizeType max_batch_size,
                         SizeType nparams,
                         SizeType nbins)
        : max_batch_size(max_batch_size),
          nparams(nparams),
          nbins(nbins),
          leaves_stride((nparams + 2) * kLeavesParamStride),
          folds_stride(2 * nbins),
          batch_leaves_d(max_batch_size * leaves_stride),
          batch_folds_d(max_batch_size * folds_stride),
          batch_scores_d(max_batch_size),
          batch_param_idx_d(max_batch_size),
          batch_phase_shift_d(max_batch_size),
          batch_isuggest_d(max_batch_size),
          batch_passing_indices_d(max_batch_size) {}

    // Call this after filling batch_scores and batch_passing_indices
    void filter_batch(SizeType n_leaves_passing) noexcept;

    float get_memory_usage() const noexcept {
        const auto total_memory =
            (batch_leaves_d.size() * sizeof(double)) +
            (batch_folds_d.size() * sizeof(FoldTypeCUDA)) +
            (batch_scores_d.size() * sizeof(float)) +
            (batch_param_idx_d.size() * sizeof(SizeType)) +
            (batch_phase_shift_d.size() * sizeof(float)) +
            (batch_isuggest_d.size() * sizeof(SizeType)) +
            (batch_passing_indices_d.size() * sizeof(SizeType));
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
         std::string_view kind)
        : m_ffa_plan(std::move(ffa_plan)),
          m_cfg(std::move(cfg)),
          m_threshold_scheme(threshold_scheme.begin(), threshold_scheme.end()),
          m_max_sugg(max_sugg),
          m_batch_size(batch_size),
          m_kind(kind),
          m_total_levels(m_threshold_scheme.size()) {
        // Setup pruning functions
        setup_pruning();

        // Allocate suggestion buffer
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_suggestions = std::make_unique<utils::SuggestionTree<FoldType>>(
                m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins_f(), kind);
        } else {
            m_suggestions = std::make_unique<utils::SuggestionTree<FoldType>>(
                m_max_sugg, m_cfg.get_nparams(), m_cfg.get_nbins(), kind);
        }

        // Allocate iteration workspace
        const auto branch_max     = 1; // m_prune_funcs->get_branch_max();
        const auto max_batch_size = m_batch_size * branch_max;
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            m_pruning_workspace =
                std::make_unique<PruningWorkspaceCUDA<FoldTypeCUDA>>(
                    max_batch_size, m_cfg.get_nparams(), m_cfg.get_nbins_f());
        } else {
            m_pruning_workspace =
                std::make_unique<PruningWorkspaceCUDA<FoldTypeCUDA>>(
                    max_batch_size, m_cfg.get_nparams(), m_cfg.get_nbins());
        }
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

    void execute(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                 SizeType ref_seg,
                 const std::filesystem::path& outdir,
                 const std::optional<std::filesystem::path>& log_file,
                 const std::optional<std::filesystem::path>& result_file) {
        const std::string run_name = std::format("{:03d}_{:02d}", ref_seg, 0);

        // Log detailed memory usage
        const auto memory_workspace_gb =
            m_pruning_workspace->get_memory_usage();
        const auto memory_suggestions_gb = m_suggestions->get_memory_usage();
        spdlog::info("Pruning run {:03d}: Memory Usage: {:.2f} GB "
                     "(suggestions) + {:.2f} GB (workspace)",
                     ref_seg, memory_suggestions_gb, memory_workspace_gb);

        // Setup log and result files
        std::filesystem::path actual_log_file =
            log_file.value_or(outdir / std::format("tmp_{}_log.txt", run_name));
        std::filesystem::path actual_result_file = result_file.value_or(
            outdir / std::format("tmp_{}_results.h5", run_name));
        std::ofstream log(actual_log_file, std::ios::app);
        log << std::format("Pruning log for ref segment: {}\n", ref_seg);
        log.close();

        const auto nsegments = m_ffa_plan.get_nsegments().back();
        initialize(ffa_fold, ref_seg, actual_log_file);

        for (SizeType iter = 0; iter < nsegments - 1; ++iter) {
            execute_iteration(ffa_fold);
            // Check for early termination (no survivors)
            if (m_prune_complete) {
                spdlog::info(
                    "Pruning terminated early at iteration {} - no survivors",
                    iter + 1);
                break;
            }
        }

        // Transform the suggestion params to middle of the data
        const auto coord_mid = m_snail_scheme->get_coord(m_prune_level);

        // Write results
        auto result_writer = cands::PruneResultWriter(
            actual_result_file, cands::PruneResultWriter::Mode::kAppend);
        result_writer.write_run_results(
            run_name, m_snail_scheme->get_data(),
            m_suggestions->get_transformed(coord_mid),
            m_suggestions->get_scores(), m_suggestions->get_nsugg(),
            m_cfg.get_nparams(), *m_pstats);

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
    plans::FFAPlan<HostFoldType> m_ffa_plan;
    search::PulsarSearchConfig m_cfg;
    std::vector<float> m_threshold_scheme;
    SizeType m_max_sugg;
    SizeType m_batch_size;
    std::string m_kind;
    SizeType m_total_levels;

    bool m_prune_complete{false};
    SizeType m_prune_level{};
    std::unique_ptr<psr_utils::MiddleOutScheme> m_snail_scheme;
    std::unique_ptr<cands::PruneStatsCollection> m_pstats;

    std::unique_ptr<utils::SuggestionTree<FoldType>> m_suggestions;
    std::unique_ptr<PruningWorkspaceCUDA<FoldTypeCUDA>> m_pruning_workspace;

    void initialize(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                    SizeType ref_seg,
                    const std::filesystem::path& log_file) {
        //  Reset the suggestion buffer state
        m_suggestions->reset();

        // Initialize snail scheme for current ref_seg
        const auto nsegments = m_ffa_plan.get_nsegments().back();
        const auto tseg      = m_ffa_plan.get_tsegments().back();
        m_snail_scheme       = std::make_unique<psr_utils::MiddleOutScheme>(
            nsegments, ref_seg, tseg);

        m_prune_level    = 0;
        m_prune_complete = false;
        spdlog::info("Pruning run {:03d}: initialized", ref_seg);

        // Initialize the suggestions with the first segment
        const auto fold_segment =
            m_prune_funcs->load(ffa_fold, m_snail_scheme->get_ref_idx());
        const auto coord_init = m_snail_scheme->get_coord(m_prune_level);
        m_prune_funcs->suggest(fold_segment, coord_init, *m_suggestions);
    }

    void execute_iteration(cuda::std::span<const FoldTypeCUDA> ffa_fold) {
        if (m_prune_complete) {
            return;
        }
        ++m_prune_level;
        if (m_prune_level > m_total_levels) {
            throw std::runtime_error(
                std::format("Pruning complete - exceeded total levels at level "
                            "{}",
                            m_prune_level));
        }
        // Prepare for in-place update: mark start of write region, reset size
        // for new suggestions.
        m_suggestions->prepare_for_in_place_update();

        const auto seg_idx_cur = m_snail_scheme->get_segment_idx(m_prune_level);
        const auto threshold   = m_threshold_scheme[m_prune_level - 1];
        // Capture the number of branches *before* finalizing the update
        const auto n_branches = m_suggestions->get_nsugg_old();

        execute_iteration_batched_kernel(ffa_fold, seg_idx_cur, threshold);

        // Finalize: make new region active, defragment for contiguous access.
        m_suggestions->finalize_in_place_update();

        // Check if no survivors
        if (m_suggestions->get_nsugg() == 0) {
            m_prune_complete = true;
            spdlog::info("Pruning run complete at level {} - no survivors",
                         m_prune_level);
            return;
        }
    }

}; // End Prune::Impl implementation

PruningManagerCUDA::PruningManagerCUDA(
    const search::PulsarSearchConfig& cfg,
    const std::vector<float>& threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    SizeType max_sugg,
    SizeType batch_size,
    int device_id) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<PruningManagerCUDATypedImpl<ComplexTypeCUDA>>(
            cfg, threshold_scheme, n_runs, std::move(ref_segs), max_sugg,
            batch_size, device_id);
    } else {
        m_impl = std::make_unique<PruningManagerCUDATypedImpl<float>>(
            cfg, threshold_scheme, n_runs, std::move(ref_segs), max_sugg,
            batch_size, device_id);
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
                                 std::string_view kind,
                                 bool show_progress) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix, kind, show_progress);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneCUDA<FoldTypeCUDA>::PruneCUDA(
    const plans::FFAPlan<typename FoldTypeTraits<FoldTypeCUDA>::HostType>&
        ffa_plan,
    const search::PulsarSearchConfig& cfg,
    std::span<const float> threshold_scheme,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view kind)
    : m_impl(std::make_unique<PruningManagerCUDATypedImpl<FoldTypeCUDA>>(
          ffa_plan, cfg, threshold_scheme, max_sugg, batch_size, kind)) {}
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
    bool show_progress) {
    m_impl->execute(ffa_fold, ref_seg, outdir, log_file, result_file,
                    show_progress);
}

template class PruneCUDA<float>;
template class PruneCUDA<ComplexTypeCUDA>;

} // namespace loki::algorithms