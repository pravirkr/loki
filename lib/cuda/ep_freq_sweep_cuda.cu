#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/algorithms/ep_memory_cuda.hpp"
#include "lib/algorithms/planner_memory.hpp"
#include "lib/algorithms/prune_engine.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/prune_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/detail/timing.hpp"
#include "lib/pipelines/ep_freq_sweep_engine.hpp"
#include "lib/pipelines/ep_sweep_common.hpp"

namespace loki::pipelines {

namespace {

/// A CUDA stream owned for the lifetime of the engine.
class OwnedStream {
public:
    explicit OwnedStream(int device_id) {
        cuda_utils::CudaSetDeviceGuard guard(device_id);
        cuda_utils::check_cuda_call(cudaStreamCreate(&m_stream),
                                    "EPFreqSweep: cudaStreamCreate failed");
    }
    ~OwnedStream() {
        if (m_stream != nullptr) {
            cudaStreamSynchronize(m_stream);
            cudaStreamDestroy(m_stream);
        }
    }
    OwnedStream(const OwnedStream&)            = delete;
    OwnedStream& operator=(const OwnedStream&) = delete;
    OwnedStream(OwnedStream&&)                 = delete;
    OwnedStream& operator=(OwnedStream&&)      = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return m_stream; }

private:
    cudaStream_t m_stream{nullptr};
};

/**
 * @brief EPFreqSweep on one CUDA device.
 *
 * Chunks come from EPRegionPlanner(Exec::cuda), so the plan is fitted to the
 * device memory. The sweep owns every shared buffer: the input series, the
 * FFA workspace and the fold buffer for the whole sweep, and one EP workspace
 * per run of chunks with the same nbins, sized from the run's maxima. The
 * runs (reference segments) of a chunk are pruned serially on one stream, so
 * a single worker's buffers are all there is to budget.
 */
template <SupportedFoldType FoldType>
class EPFreqSweepCudaEngine final : public detail::EPFreqSweepEngine {
    using FoldTypeCUDA = CudaFoldType<FoldType>;
    using DeviceFoldT  = DeviceFoldType<FoldTypeCUDA>;

public:
    EPFreqSweepCudaEngine(
        search::PulsarSearchConfig cfg,
        float min_pd,
        std::string_view poly_basis,
        float ref_ducy,
        const std::optional<std::filesystem::path>& plan_cache_file,
        std::optional<SizeType> n_runs,
        const std::optional<std::vector<SizeType>>& ref_segs,
        int device_id)
        : m_base_cfg(std::move(cfg)),
          m_min_pd(min_pd),
          m_poly_basis(poly_basis),
          m_ref_ducy(ref_ducy),
          m_n_runs(n_runs),
          m_ref_segs(ref_segs),
          m_device_id(device_id),
          m_memory{
              .nparams    = m_base_cfg.get_nparams(),
              .nsamps     = m_base_cfg.get_nsamps(),
              .n_workers  = 1,
              .batch_size = algorithms::detail::kEPBatchSizeCuda,
              .harvest    = {},
              .kind       = algorithms::detail::EPMemoryKind::kCuda,
              .device     = device_id,
              .cub_scratch_bytes =
                  &algorithms::detail::ep_cuda_cub_scratch_bytes,
          },
          m_region_planner(m_base_cfg,
                           min_pd,
                           poly_basis,
                           ref_ducy,
                           plan_cache_file,
                           algorithms::PruneRFIConfig{},
                           std::nullopt,
                           Exec::cuda(device_id)),
          m_fft_manager(device_id),
          m_prune_fft(device_id),
          m_stream(device_id) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        const auto& stats      = m_region_planner.get_stats();
        const auto& chunk_cfgs = m_region_planner.get_chunk_cfgs();
        spdlog::info(
            "EPFreqSweep (CUDA {}): {} chunks planned, peak {:.2f} GB of "
            "device memory (limit {:.2f} GB), max_sugg: {}",
            m_device_id, m_region_planner.get_nchunks(),
            stats.get_max_memory_gb(), stats.get_memory_limit_gb(),
            stats.get_max_sugg());

        // The FFA workspace needs the deepest chunk's level count.
        SizeType max_levels = 0;
        for (const auto& chunk : chunk_cfgs) {
            const plans::FFAPlan<FoldType> plan(chunk.cfg);
            max_levels = std::max(max_levels, plan.get_n_levels());
        }

        // Shared buffers, held for the whole sweep.
        m_ffa_workspace = memory::FFAWorkspaceCUDA<FoldTypeCUDA>(
            stats.get_max_buffer_size(), stats.get_max_coord_size(), max_levels,
            m_base_cfg.get_nparams());
        m_fold_d.resize(stats.get_max_buffer_size(), DeviceFoldT{});
        m_ts_e_d.resize(m_base_cfg.get_nsamps());
        m_ts_v_d.resize(m_base_cfg.get_nsamps());
        check_shared_allocation(stats.get_max_buffer_size(),
                                stats.get_max_coord_size());

        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            std::vector<SizeType> n_reals;
            n_reals.reserve(chunk_cfgs.size());
            for (const auto& chunk : chunk_cfgs) {
                n_reals.push_back(chunk.cfg.get_nbins());
            }
            m_fft_manager.prepare_plans(n_reals);
            // The pruning functors of every chunk share these exact-batch
            // plans, instead of building their own per chunk.
            m_prune_fft.prepare_exact_plans(n_reals);
        }

        // What the sweep still has to allocate on top of the shared buffers.
        SizeType peak_group_bytes = 0;
        SizeType peak_transient   = 0;
        for (const auto& group : algorithms::detail::ep_chunk_groups(
                 std::span<const algorithms::EPChunkConfig>(chunk_cfgs))) {
            peak_group_bytes =
                std::max(peak_group_bytes,
                         algorithms::detail::ep_group_thread_bytes<FoldType>(
                             m_memory, group));
        }
        for (const auto& chunk : chunk_cfgs) {
            peak_transient =
                std::max(peak_transient, chunk.ffa_transient_bytes);
        }
        m_extra_peak_gb =
            static_cast<double>(peak_group_bytes + peak_transient) /
            algorithms::detail::kBytesPerGiB;

        spdlog::info("EPFreqSweep (CUDA {}): allocated the shared buffers "
                     "(buffer_size={}, coord_size={}, {} levels)",
                     m_device_id, stats.get_max_buffer_size(),
                     stats.get_max_coord_size(), max_levels);
    }

    ~EPFreqSweepCudaEngine() final                                 = default;
    EPFreqSweepCudaEngine(const EPFreqSweepCudaEngine&)            = delete;
    EPFreqSweepCudaEngine& operator=(const EPFreqSweepCudaEngine&) = delete;
    EPFreqSweepCudaEngine(EPFreqSweepCudaEngine&&)                 = delete;
    EPFreqSweepCudaEngine& operator=(EPFreqSweepCudaEngine&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        timing::SimpleTimer sweep_timer;
        sweep_timer.start();
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);

        if (ts_e.size() != m_base_cfg.get_nsamps() ||
            ts_v.size() != m_base_cfg.get_nsamps()) {
            throw std::invalid_argument(std::format(
                "EPFreqSweep::execute: ts_e and ts_v must have nsamps={} "
                "samples, got {} and {}",
                m_base_cfg.get_nsamps(), ts_e.size(), ts_v.size()));
        }
        check_device_has_room();

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

        // The inputs stay on the device for the whole sweep.
        const cudaStream_t stream = m_stream.get();
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_e_d.data()),
                            ts_e.data(), ts_e.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "EPFreqSweep: cudaMemcpyAsync ts_e failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_v_d.data()),
                            ts_v.data(), ts_v.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "EPFreqSweep: cudaMemcpyAsync ts_v failed");
        cuda_utils::check_cuda_call(
            cudaStreamSynchronize(stream),
            "EPFreqSweep: input copy synchronization failed");

        const auto& chunk_cfgs = m_region_planner.get_chunk_cfgs();
        const SizeType nchunks = chunk_cfgs.size();
        spdlog::info("EPFreqSweep (CUDA {}): starting sweep over {} chunks",
                     m_device_id, nchunks);

        constexpr SizeType kBatchSize = algorithms::detail::kEPBatchSizeCuda;
        const auto& stats             = m_region_planner.get_stats();
        const double limit_gb = algorithms::detail::effective_memory_limit_gb(
            stats.get_memory_limit_gb());
        SizeType sweep_transient = 0;
        for (const auto& chunk : chunk_cfgs) {
            sweep_transient =
                std::max(sweep_transient, chunk.ffa_transient_bytes);
        }

        for (const auto& group : algorithms::detail::ep_chunk_groups(
                 std::span<const algorithms::EPChunkConfig>(chunk_cfgs))) {
            check_branch_max(chunk_cfgs, group);

            // Guard against plans (e.g. stale caches) that no longer fit.
            const SizeType thread_bytes =
                algorithms::detail::ep_group_thread_bytes<FoldType>(m_memory,
                                                                    group);
            detail::check_ep_group_budget(
                group,
                algorithms::detail::ep_total_gb(
                    m_memory.n_workers, thread_bytes,
                    algorithms::detail::ep_fixed_bytes<FoldType>(
                        m_memory, stats.get_max_buffer_size(),
                        stats.get_max_coord_size(), sweep_transient)),
                limit_gb, m_memory.n_workers);

            const SizeType effective_nbins =
                std::is_same_v<FoldType, ComplexType>
                    ? chunk_cfgs[group.begin].cfg.get_nbins_f()
                    : group.nbins;
            const SizeType capacity =
                algorithms::detail::ep_cuda_effective_max_sugg(
                    kBatchSize, group.branch_max, group.max_sugg);

            // One workspace for this run of chunks.
            memory::EPWorkspaceCUDA<FoldTypeCUDA> workspace(
                kBatchSize, group.branch_max, capacity, group.ncoords,
                m_base_cfg.get_nparams(), effective_nbins, group.nsegments,
                stream);
            cuda_utils::check_cuda_call(
                cudaStreamSynchronize(stream),
                "EPFreqSweep: workspace allocation failed");
            detail::check_ep_workspace_vs_model(
                group, static_cast<double>(workspace.get_memory_usage_bytes()),
                static_cast<double>(
                    algorithms::detail::ep_cuda_ep_workspace_bytes<FoldType>(
                        m_memory, group.nbins, group.nsegments, group.ncoords,
                        group.max_sugg, group.branch_max)));
            spdlog::info(
                "EPFreqSweep (CUDA {}): allocated a workspace for chunks "
                "{}..{} (nbins={}, max_sugg={}, {:.2f} GB)",
                m_device_id, group.begin, group.end - 1, effective_nbins,
                capacity, workspace.get_memory_usage_gib());

            algorithms::EPCudaSharedPipeline<FoldTypeCUDA> pipeline{
                .ep_workspace      = &workspace,
                .ffa_workspace     = &m_ffa_workspace,
                .fft_manager       = &m_fft_manager,
                .prune_fft_manager = &m_prune_fft,
                .fold_d            = cuda_utils::as_span(m_fold_d),
                .ts_e_d            = cuda::std::span<const float>(
                    thrust::raw_pointer_cast(m_ts_e_d.data()), m_ts_e_d.size()),
                .ts_v_d = cuda::std::span<const float>(
                    thrust::raw_pointer_cast(m_ts_v_d.data()), m_ts_v_d.size()),
                .stream        = stream,
                .ws_max_sugg   = capacity,
                .ws_branch_max = group.branch_max,
                .ws_ncoords    = group.ncoords,
                .ws_nsegments  = group.nsegments,
            };

            for (SizeType i = group.begin; i < group.end; ++i) {
                const auto& chunk = chunk_cfgs[i];
                spdlog::info("EPFreqSweep: processing chunk {}/{} - nominal "
                             "f=[{:08.3f}, {:08.3f}] Hz, actual f=[{:08.3f}, "
                             "{:08.3f}] Hz (nbins={}, max_sugg={})",
                             i + 1, nchunks, chunk.nominal_f_start,
                             chunk.nominal_f_end, chunk.actual_f_start,
                             chunk.actual_f_end, chunk.cfg.get_nbins(),
                             chunk.max_sugg);

                const std::string chunk_prefix = std::format("chunk_{:04d}", i);
                algorithms::EPMultiPassCudaCore<FoldTypeCUDA> chunk_ep(
                    pipeline, chunk.cfg, chunk.threshold_scheme, m_n_runs,
                    m_ref_segs, /*ascend_levels=*/{}, chunk.max_sugg,
                    kBatchSize, m_poly_basis, m_device_id);
                chunk_ep.execute_device(tmp_dir, chunk_prefix);
            }
            cuda_utils::check_cuda_call(
                cudaStreamSynchronize(stream),
                "EPFreqSweep: chunk group synchronization failed");
        }

        const auto total_runtime = sweep_timer.stop();

        // Batch merge all chunk HDF5 files into the unified result file
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
    float m_min_pd;
    std::string m_poly_basis;
    float m_ref_ducy;
    std::optional<SizeType> m_n_runs;
    std::optional<std::vector<SizeType>> m_ref_segs;
    int m_device_id;
    algorithms::detail::EPMemoryContext m_memory;
    algorithms::EPRegionPlanner<FoldType> m_region_planner;

    memory::FFAWorkspaceCUDA<FoldTypeCUDA> m_ffa_workspace;
    math::CUFFTManager m_fft_manager;
    // The FFT plans of the pruning functors, prepared for every chunk's nbins.
    math::CUFFTManager m_prune_fft;
    // Declared before the device buffers so that it outlives them.
    OwnedStream m_stream;
    thrust::device_vector<DeviceFoldT> m_fold_d;
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    /// Device memory the sweep still takes on top of the shared buffers: the
    /// largest group's workspace and the transient FFA scratch (GiB).
    double m_extra_peak_gb{0.0};

    // Runtime checks of the plan against the memory model (docs/memory.md).
    // They only fire on a bug, a stale cache or a model drift: the planner
    // guarantees the budget and the model mirrors the allocations exactly.

    /// The device must still have room for what the sweep allocates next.
    void check_device_has_room() const {
        const double free_gb =
            algorithms::detail::ep_cuda_free_memory_gb(m_device_id);
        if (free_gb < m_extra_peak_gb) {
            throw std::runtime_error(std::format(
                "EPFreqSweep: device {} has {:.2f} GB free but the sweep "
                "still needs up to {:.2f} GB (plan peak {:.2f} GB). Free "
                "device memory or re-plan with a smaller "
                "max_process_memory_gb.",
                m_device_id, free_gb, m_extra_peak_gb,
                m_region_planner.get_stats().get_max_memory_gb()));
        }
    }

    /// Guard against plans whose branch_max is smaller than what a chunk's
    /// own plan needs.
    void
    check_branch_max(const std::vector<algorithms::EPChunkConfig>& chunk_cfgs,
                     const algorithms::detail::EPChunkGroup& group) const {
        for (SizeType i = group.begin; i < group.end; ++i) {
            const plans::FFAPlan<FoldType> plan(chunk_cfgs[i].cfg);
            const auto needed = algorithms::detail::compute_branch_max(
                plan.get_branching_pattern(m_poly_basis));
            if (needed > group.branch_max) {
                throw std::runtime_error(std::format(
                    "EPFreqSweep: chunk {} needs branch_max={} but the plan "
                    "provides {}. Re-plan (stale plan cache?).",
                    i, needed, group.branch_max));
            }
        }
    }

    /// After allocating the shared FFA buffers: they must match the model.
    void check_shared_allocation(SizeType buffer_size,
                                 SizeType coord_size) const {
        const SizeType actual = m_ffa_workspace.get_buffers_bytes() +
                                (m_fold_d.size() * sizeof(DeviceFoldT));
        const SizeType model =
            algorithms::detail::ep_cuda_shared_bytes<FoldType>(
                m_memory.nparams, buffer_size, coord_size);
        detail::check_ep_shared_vs_model(actual, model);
    }
};

} // namespace

namespace detail {
std::unique_ptr<EPFreqSweepEngine> make_ep_freq_sweep_gpu(
    const search::PulsarSearchConfig& cfg,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    const std::optional<std::vector<SizeType>>& ref_segs,
    int device_id) {
    if (cfg.get_use_fourier()) {
        return std::make_unique<EPFreqSweepCudaEngine<ComplexType>>(
            cfg, min_pd, poly_basis, ref_ducy, plan_cache_file, n_runs,
            ref_segs, device_id);
    }
    return std::make_unique<EPFreqSweepCudaEngine<float>>(
        cfg, min_pd, poly_basis, ref_ducy, plan_cache_file, n_runs, ref_segs,
        device_id);
}
} // namespace detail

} // namespace loki::pipelines
