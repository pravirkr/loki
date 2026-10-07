#include "loki/pipelines/ep_freq_sweep.hpp" // NOLINT(misc-include-cleaner): facade header first

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

#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/prune_engine.hpp"
#include "lib/detail/timing.hpp"
#include "lib/pipelines/ep_freq_sweep_engine.hpp"
#include "lib/utils/fft_impl.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::pipelines {

namespace {

double
merge_ep_sweep_results(const std::filesystem::path& tmp_dir,
                       const std::filesystem::path& result_file,
                       const std::vector<algorithms::EPChunkConfig>& chunk_cfgs,
                       const search::PulsarSearchConfig& base_cfg,
                       float min_pd,
                       std::string_view poly_basis,
                       float ref_ducy,
                       float total_runtime) {
    HighFive::File main_h5(result_file.string(), HighFive::File::Overwrite);
    main_h5.createAttribute("ep_sweep_version", std::string("1.0.0-cpp"));
    main_h5.createAttribute("param_names", base_cfg.get_param_names());
    main_h5.createAttribute("nchunks", chunk_cfgs.size());
    main_h5.createAttribute("f_min", base_cfg.get_f_min());
    main_h5.createAttribute("f_max", base_cfg.get_f_max());
    main_h5.createAttribute("tobs", base_cfg.get_tobs());
    main_h5.createAttribute("tsamp", base_cfg.get_tsamp());
    main_h5.createAttribute("min_pd", min_pd);
    main_h5.createAttribute("poly_basis", std::string(poly_basis));
    main_h5.createAttribute("ref_ducy", ref_ducy);
    main_h5.createAttribute("total_runtime", total_runtime);

    auto main_chunks_group   = main_h5.createGroup("chunks");
    double accumulated_flops = 0.0;
    const SizeType nchunks   = chunk_cfgs.size();

    for (SizeType i = 0; i < nchunks; ++i) {
        const auto& chunk              = chunk_cfgs[i];
        const std::string chunk_prefix = std::format("chunk_{:04d}", i);
        auto chunk_group = main_chunks_group.createGroup(chunk_prefix);

        chunk_group.createAttribute("chunk_id", i);
        chunk_group.createAttribute("nominal_f_start", chunk.nominal_f_start);
        chunk_group.createAttribute("nominal_f_end", chunk.nominal_f_end);
        chunk_group.createAttribute("actual_f_start", chunk.actual_f_start);
        chunk_group.createAttribute("actual_f_end", chunk.actual_f_end);
        chunk_group.createAttribute("nbins", chunk.cfg.get_nbins());
        chunk_group.createAttribute("eta", chunk.cfg.get_eta());
        chunk_group.createAttribute("max_sugg", chunk.max_sugg);
        chunk_group.createAttribute("branch_max", chunk.branch_max);
        chunk_group.createAttribute("peak_complexity", chunk.peak_complexity);
        chunk_group.createAttribute("chunk_memory_gb", chunk.chunk_memory_gb);
        chunk_group.createAttribute("nsegments", chunk.nsegments);

        chunk_group.createDataSet("threshold_scheme", chunk.threshold_scheme);
        chunk_group.createDataSet("branching_pattern", chunk.branching_pattern);

        const auto chunk_result_file =
            tmp_dir / std::format("{}_pruning_nstages_{}_results.h5",
                                  chunk_prefix, chunk.nsegments);

        if (std::filesystem::exists(chunk_result_file)) {
            HighFive::File const chunk_h5(chunk_result_file.string(),
                                          HighFive::File::ReadOnly);
            if (chunk_h5.exist("runs")) {
                HighFive::Group const chunk_runs = chunk_h5.getGroup("runs");
                HighFive::Group const dst_runs =
                    chunk_group.createGroup("runs");
                for (const auto& run_name : chunk_runs.listObjectNames()) {
                    auto const run_grp = chunk_runs.getGroup(run_name);
                    if (run_grp.hasAttribute("total_pruning_gflops")) {
                        double run_gflops{};
                        run_grp.getAttribute("total_pruning_gflops")
                            .read(run_gflops);
                        accumulated_flops += run_gflops;
                    }
                    herr_t const status = H5Ocopy(
                        chunk_runs.getId(), run_name.c_str(), dst_runs.getId(),
                        run_name.c_str(), H5P_DEFAULT, H5P_DEFAULT);
                    if (status < 0) {
                        throw std::runtime_error(std::format(
                            "EPFreqSweep: failed to copy run '{}' for chunk {}",
                            run_name, i));
                    }
                }
            }
        }
    }

    main_h5.createAttribute("total_pruning_gflops", accumulated_flops);
    return accumulated_flops;
}

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
          m_region_planner(
              m_base_cfg, min_pd, poly_basis, ref_ducy, plan_cache_file) {
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

        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            const auto& chunk_cfgs = m_region_planner.get_chunk_cfgs();
            std::vector<SizeType> n_reals;
            n_reals.reserve(chunk_cfgs.size());
            for (const auto& chunk : chunk_cfgs) {
                n_reals.push_back(chunk.cfg.get_nbins());
            }
            m_fft_manager.prepare_plans(n_reals);
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

        constexpr SizeType kBatchSize = 1024U;
        const auto nthreads           = m_base_cfg.get_nthreads();

        SizeType chunk_idx = 0;
        while (chunk_idx < nchunks) {
            // Find contiguous run of chunks sharing the same nbins
            const auto cur_nbins = chunk_cfgs[chunk_idx].cfg.get_nbins();
            SizeType range_end   = chunk_idx + 1;
            while (range_end < nchunks &&
                   chunk_cfgs[range_end].cfg.get_nbins() == cur_nbins) {
                ++range_end;
            }

            // Find max requirements across all chunks in this nbins band
            SizeType group_max_sugg    = 0;
            SizeType group_max_branch  = 0;
            SizeType group_max_ncoords = 0;
            SizeType group_max_nseg    = 0;
            for (SizeType i = chunk_idx; i < range_end; ++i) {
                group_max_sugg =
                    std::max(group_max_sugg, chunk_cfgs[i].max_sugg);
                group_max_branch =
                    std::max(group_max_branch, chunk_cfgs[i].branch_max);
                group_max_ncoords =
                    std::max(group_max_ncoords, chunk_cfgs[i].ncoords);
                group_max_nseg =
                    std::max(group_max_nseg, chunk_cfgs[i].nsegments);
            }

            const SizeType effective_nbins =
                std::is_same_v<FoldType, ComplexType>
                    ? chunk_cfgs[chunk_idx].cfg.get_nbins_f()
                    : cur_nbins;

            // Allocate workspaces once for this band (one per thread)
            std::vector<memory::EPWorkspaceCPU<FoldType>> workspaces;
            workspaces.reserve(static_cast<size_t>(nthreads));
            for (int t = 0; t < nthreads; ++t) {
                workspaces.emplace_back(kBatchSize, group_max_branch,
                                        group_max_sugg, group_max_ncoords,
                                        m_base_cfg.get_nparams(),
                                        effective_nbins, group_max_nseg);
            }
            std::vector<memory::EPWorkspaceCPU<FoldType>*> workspace_ptrs;
            workspace_ptrs.reserve(workspaces.size());
            for (auto& ws : workspaces) {
                workspace_ptrs.push_back(&ws);
            }

            spdlog::info(
                "EPFreqSweep: allocated {} workspaces for chunks {}..{} "
                "(nbins={}, max_sugg={}, {:.2f} GB per workspace)",
                nthreads, chunk_idx, range_end - 1, effective_nbins,
                group_max_sugg, workspaces.front().get_memory_usage_gib());

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
                    std::span(m_ffa_fold), chunk.cfg, chunk.threshold_scheme,
                    m_n_runs, m_ref_segs,
                    /*ascend_levels=*/{}, chunk.max_sugg, kBatchSize,
                    m_poly_basis, m_show_progress && (nchunks == 1),
                    m_rfi_config);

                chunk_ep->execute(ts_e, ts_v, tmp_dir, chunk_prefix);
            }

            chunk_idx = range_end;
        }

        const auto total_runtime = sweep_timer.stop();

        // Batch merge all chunk HDF5 files into unified result file
        const double accumulated_flops = merge_ep_sweep_results(
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
    algorithms::EPRegionPlanner<FoldType> m_region_planner;

    memory::FFAWorkspaceCPU<FoldType> m_ffa_workspace;
    math::FFTWManager m_fft_manager;
    std::vector<FoldType> m_ffa_fold;
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
