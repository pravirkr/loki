#include <cstddef>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/algorithms/planner_memory.hpp"
#include "lib/algorithms/prune_engine.hpp"
#include "lib/detail/timing.hpp"
#include "lib/pipelines/ep_freq_sweep_engine.hpp"
#include "lib/pipelines/ep_sweep_common.hpp"
#include "lib/utils/fft_impl.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::pipelines {

namespace {

template <SupportedFoldType FoldType>
class EPFreqSweepCpuEngine final : public detail::EPFreqSweepEngine {
public:
    EPFreqSweepCpuEngine(
        search::PulsarSearchConfig cfg,
        bool show_progress,
        float min_pd,
        std::string_view poly_basis,
        float ref_ducy,
        algorithms::PruneRFIConfig rfi_config,
        const std::optional<std::filesystem::path>& plan_cache_file,
        std::optional<SizeType> n_runs,
        const std::optional<std::vector<SizeType>>& ref_segs)
        : m_base_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_min_pd(min_pd),
          m_poly_basis(poly_basis),
          m_ref_ducy(ref_ducy),
          m_rfi_config(std::move(rfi_config)),
          m_n_runs(n_runs),
          m_ref_segs(ref_segs),
          m_n_workers(algorithms::detail::compute_ep_n_workers(
              m_base_cfg.get_nthreads(), m_n_runs, m_ref_segs)),
          m_memory{
              .nparams   = m_base_cfg.get_nparams(),
              .nsamps    = m_base_cfg.get_nsamps(),
              .n_workers = static_cast<int>(m_n_workers),
              .harvest = algorithms::detail::EPHarvestBound::from(m_rfi_config),
          },
          m_region_planner(m_base_cfg,
                           min_pd,
                           poly_basis,
                           ref_ducy,
                           plan_cache_file,
                           m_rfi_config,
                           m_n_workers) {
        const auto& stats = m_region_planner.get_stats();
        spdlog::info(
            "EPFreqSweep initialized: {} chunks planned, max memory: {:.2f} GB "
            "(process limit: {:.2f} GB), max_sugg: {}",
            m_region_planner.get_nchunks(), stats.get_max_memory_gb(),
            m_base_cfg.get_max_process_memory_gb(), stats.get_max_sugg());

        // Allocate shared FFA workspace and fold buffer once for the entire
        // sweep
        m_ffa_workspace = memory::FFAWorkspaceCPU<FoldType>(
            stats.get_max_buffer_size(), stats.get_max_coord_size(),
            m_base_cfg.get_nparams());
        m_ffa_fold.resize(stats.get_max_buffer_size(), FoldType{});
        check_shared_allocation(stats.get_max_buffer_size(),
                                stats.get_max_coord_size());

        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            const auto& chunk_cfgs = m_region_planner.get_chunk_cfgs();
            std::vector<SizeType> n_reals;
            n_reals.reserve(chunk_cfgs.size());
            for (const auto& chunk : chunk_cfgs) {
                n_reals.push_back(chunk.cfg.get_nbins());
            }
            m_fft_manager.prepare_plans(n_reals);

            // The pruning functors of every chunk share these plans: one exact
            // manager per worker, prepared once for all chunk nbins, instead of
            // one per run. Each worker keeps its own, since plans are created
            // lazily.
            m_prune_fft.resize(m_n_workers);
            m_prune_fft_ptrs.reserve(m_n_workers);
            for (auto& fft : m_prune_fft) {
                fft.prepare_exact_plans(n_reals);
                m_prune_fft_ptrs.push_back(&fft);
            }
        }

        spdlog::info(
            "EPFreqSweep allocated shared FFA workspace: buffer_size={}, "
            "coord_size={}",
            stats.get_max_buffer_size(), stats.get_max_coord_size());
    }

    ~EPFreqSweepCpuEngine() final                                = default;
    EPFreqSweepCpuEngine(const EPFreqSweepCpuEngine&)            = delete;
    EPFreqSweepCpuEngine& operator=(const EPFreqSweepCpuEngine&) = delete;
    EPFreqSweepCpuEngine(EPFreqSweepCpuEngine&&)                 = delete;
    EPFreqSweepCpuEngine& operator=(EPFreqSweepCpuEngine&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        timing::SimpleTimer sweep_timer;
        sweep_timer.start();

        std::error_code ec;
        std::filesystem::create_directories(outdir, ec);
        if (!std::filesystem::exists(outdir)) {
            throw std::runtime_error(std::format(
                "EPFreqSweep::execute: Failed to create output directory '{}': "
                "{}",
                outdir.string(), ec.message()));
        }

        const auto result_file =
            (outdir / std::format("{}_ep_results.h5", file_prefix))
                .lexically_normal();
        const auto tmp_dir =
            (outdir / std::format(".tmp_{}_ep_chunks", file_prefix))
                .lexically_normal();
        std::filesystem::create_directories(tmp_dir, ec);

        const auto& chunk_cfgs = m_region_planner.get_chunk_cfgs();
        const SizeType nchunks = chunk_cfgs.size();
        spdlog::info("EPFreqSweep: starting sweep over {} chunks", nchunks);

        constexpr SizeType kBatchSize = algorithms::detail::kEPBatchSize;
        const auto& stats             = m_region_planner.get_stats();

        for (const auto& group : algorithms::detail::ep_chunk_groups(
                 std::span<const algorithms::EPChunkConfig>(chunk_cfgs))) {
            const SizeType chunk_idx = group.begin;
            const SizeType range_end = group.end;

            // Guard against plans (e.g. stale caches) whose branch_max is
            // smaller than what the chunk's own plan needs.
            for (SizeType i = chunk_idx; i < range_end; ++i) {
                const plans::FFAPlan<FoldType> plan(chunk_cfgs[i].cfg);
                const auto needed = algorithms::detail::compute_branch_max(
                    plan.get_branching_pattern(m_poly_basis));
                if (needed > group.branch_max) {
                    throw std::runtime_error(std::format(
                        "EPFreqSweep: chunk {} needs branch_max={} but the "
                        "plan provides {}. Re-plan (stale plan cache?).",
                        i, needed, group.branch_max));
                }
            }

            const auto model_thread_bytes =
                algorithms::detail::ep_group_thread_bytes<FoldType>(m_memory,
                                                                    group);
            check_group_budget(group, model_thread_bytes,
                               stats.get_max_buffer_size(),
                               stats.get_max_coord_size());

            const SizeType effective_nbins =
                std::is_same_v<FoldType, ComplexType>
                    ? chunk_cfgs[chunk_idx].cfg.get_nbins_f()
                    : group.nbins;

            // Allocate workspaces once for this band (one per worker)
            std::vector<memory::EPWorkspaceCPU<FoldType>> workspaces;
            workspaces.reserve(m_n_workers);
            for (SizeType t = 0; t < m_n_workers; ++t) {
                workspaces.emplace_back(
                    kBatchSize, group.branch_max, group.max_sugg, group.ncoords,
                    m_base_cfg.get_nparams(), effective_nbins, group.nsegments);
            }
            std::vector<memory::EPWorkspaceCPU<FoldType>*> workspace_ptrs;
            workspace_ptrs.reserve(workspaces.size());
            for (auto& ws : workspaces) {
                workspace_ptrs.push_back(&ws);
            }
            check_workspace_allocation(group, workspaces.front());

            spdlog::info(
                "EPFreqSweep: allocated {} workspaces for chunks {}..{} "
                "(nbins={}, max_sugg={}, {:.2f} GB per workspace)",
                m_n_workers, chunk_idx, range_end - 1, effective_nbins,
                group.max_sugg, workspaces.front().get_memory_usage_gib());

            // Process all chunks in this nbins band reusing workspaces
            for (SizeType i = chunk_idx; i < range_end; ++i) {
                const auto& chunk = chunk_cfgs[i];
                spdlog::info(
                    "EPFreqSweep: processing chunk {}/{} - nominal "
                    "f=[{:08.3f}, "
                    "{:08.3f}] Hz, actual f=[{:08.3f}, {:08.3f}] Hz (nbins={}, "
                    "max_sugg={})",
                    i + 1, nchunks, chunk.nominal_f_start, chunk.nominal_f_end,
                    chunk.actual_f_start, chunk.actual_f_end,
                    chunk.cfg.get_nbins(), chunk.max_sugg);

                const std::string chunk_prefix = std::format("chunk_{:04d}", i);
                // Internal pipeline: build the CPU engine directly on the
                // shared workspaces, FFA buffers and plan cache.
                const auto chunk_ep = algorithms::detail::make_ep_cpu<FoldType>(
                    std::span(workspace_ptrs), m_ffa_workspace, m_fft_manager,
                    std::span(m_ffa_fold), std::span(m_prune_fft_ptrs),
                    chunk.cfg, chunk.threshold_scheme, m_n_runs, m_ref_segs,
                    /*ascend_levels=*/{}, chunk.max_sugg, kBatchSize,
                    m_poly_basis, m_show_progress && (nchunks == 1),
                    m_rfi_config);

                chunk_ep->execute(ts_e, ts_v, tmp_dir, chunk_prefix);
            }
        }

        const auto total_runtime = sweep_timer.stop();

        // Batch merge all chunk HDF5 files into unified result file
        const double accumulated_flops = detail::merge_ep_sweep_results(
            tmp_dir, result_file, chunk_cfgs, m_base_cfg, m_min_pd,
            m_poly_basis, m_ref_ducy, total_runtime);

        // Clean up temporary chunk output directory on successful completion
        std::filesystem::remove_all(tmp_dir, ec);

        spdlog::info(
            "EPFreqSweep complete: {} chunks processed in {:.2f} seconds "
            "({:.2f} GFLOPs). Results saved to '{}'",
            nchunks, total_runtime, accumulated_flops, result_file.string());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    bool m_show_progress;
    float m_min_pd;
    std::string m_poly_basis;
    float m_ref_ducy;
    algorithms::PruneRFIConfig m_rfi_config;
    std::optional<SizeType> m_n_runs;
    std::optional<std::vector<SizeType>> m_ref_segs;
    // Workers pruning at the same time: min(nthreads, runs).
    SizeType m_n_workers;
    algorithms::detail::EPMemoryContext m_memory;
    algorithms::EPRegionPlanner<FoldType> m_region_planner;

    memory::FFAWorkspaceCPU<FoldType> m_ffa_workspace;
    math::FFTWManager m_fft_manager;
    // Pruning FFT managers, one per worker (complex folds only), and the
    // pointers the chunk engines take. Built once in the constructor.
    std::vector<math::FFTWManager> m_prune_fft;
    std::vector<math::FFTWManager*> m_prune_fft_ptrs;
    std::vector<FoldType> m_ffa_fold;

    // Runtime checks of the plan against the memory model (docs/memory.md).
    // They only fire on a bug or a model drift: the planner guarantees the
    // budget and the model mirrors the allocations exactly.

    /// Before allocating a group: the model total must fit the budget.
    void check_group_budget(const algorithms::detail::EPChunkGroup& group,
                            SizeType thread_bytes,
                            SizeType buffer_size,
                            SizeType coord_size) const {
        const double total_gb = algorithms::detail::ep_total_gb(
            m_memory.n_workers, thread_bytes,
            algorithms::detail::ep_fixed_bytes<FoldType>(m_memory, buffer_size,
                                                         coord_size));
        detail::check_ep_group_budget(
            group, total_gb,
            algorithms::detail::effective_memory_limit_gb(
                m_base_cfg.get_max_process_memory_gb()),
            m_memory.n_workers);
    }

    /// After allocating a group: a workspace must not exceed the model.
    void check_workspace_allocation(
        const algorithms::detail::EPChunkGroup& group,
        const memory::EPWorkspaceCPU<FoldType>& ws) const {
        const auto model = static_cast<double>(
            algorithms::detail::ep_workspace_bytes<FoldType>(
                m_memory.nparams, group.nbins, group.nsegments, group.ncoords,
                group.max_sugg, group.branch_max, m_memory.batch_size));
        const double actual = static_cast<double>(ws.get_memory_usage_gib()) *
                              algorithms::detail::kBytesPerGiB;
        detail::check_ep_workspace_vs_model(group, actual, model);
    }

    /// After allocating the shared FFA buffers: they must match the model.
    void check_shared_allocation(SizeType buffer_size,
                                 SizeType coord_size) const {
        const SizeType actual =
            (m_ffa_workspace.fold_internal.size() * sizeof(FoldType)) +
            (m_ffa_workspace.coords.size() * sizeof(coord::FFACoord)) +
            (m_ffa_workspace.coords_freq.size() * sizeof(coord::FFACoordFreq)) +
            (m_ffa_fold.size() * sizeof(FoldType));
        const SizeType model = algorithms::detail::ep_shared_bytes<FoldType>(
            m_memory.nparams, buffer_size, coord_size);
        detail::check_ep_shared_vs_model(actual, model);
    }
};

} // namespace

namespace detail {
std::unique_ptr<EPFreqSweepEngine> make_ep_freq_sweep_cpu(
    const search::PulsarSearchConfig& cfg,
    bool show_progress,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const algorithms::PruneRFIConfig& rfi_config,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    const std::optional<std::vector<SizeType>>& ref_segs) {
    if (cfg.get_use_fourier()) {
        return std::make_unique<EPFreqSweepCpuEngine<ComplexType>>(
            cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
            plan_cache_file, n_runs, ref_segs);
    }
    return std::make_unique<EPFreqSweepCpuEngine<float>>(
        cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
        plan_cache_file, n_runs, ref_segs);
}
} // namespace detail

} // namespace loki::pipelines
