#include "loki/pipelines/ffa_manager.hpp"

#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/plans.hpp"
#include "loki/cands.hpp"
#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"

namespace loki::algorithms {

class FFAManagerCUDA::BaseImpl {
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
                         std::string_view file_prefix) = 0;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class FFAManagerCUDATypedImpl final : public FFAManagerCUDA::BaseImpl {
public:
    using HostFoldType   = FoldTypeTraits<FoldTypeCUDA>::HostType;
    using DeviceFoldType = FoldTypeTraits<FoldTypeCUDA>::DeviceType;

    FFAManagerCUDATypedImpl(search::PulsarSearchConfig cfg, int device_id)
        : m_base_cfg(std::move(cfg)),
          m_device_id(device_id),
          m_region_planner(create_region_planner(m_base_cfg, m_device_id)) {
        const auto& planner_stats = m_region_planner.get_stats();
        // Allocate buffers once, sized for the largest chunk
        m_ffa_workspace = FFAWorkspaceCUDA<FoldTypeCUDA>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(), m_base_cfg.get_nparams());
        m_fold_d_time.resize(planner_stats.get_max_buffer_size_time());
        m_scores.resize(planner_stats.get_max_scores_size());
        m_scores_d.resize(planner_stats.get_max_scores_size());
        m_passing_indices.resize(planner_stats.get_max_scores_size());
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_write_scores_batch.resize(planner_stats.get_write_scores_size());
        m_ffa_stats = std::make_unique<cands::FFAStatsCollection>();

        // Copy scoring widths to device
        m_widths_d = m_base_cfg.get_scoring_widths();

        // Log the actual memory usage for the allocated buffers
        spdlog::info("FFAManagerCUDA allocated {:.2f} GB ({:.2f} GB buffers "
                     "+ {:.2f} GB coords + {:.2f} GB extra)",
                     planner_stats.get_manager_memory_usage(),
                     planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_extra_memory_usage());
        spdlog::info("FFAManagerCUDA will process {} chunks",
                     m_region_planner.get_nregions());
    }

    ~FFAManagerCUDATypedImpl() final                        = default;
    FFAManagerCUDATypedImpl(const FFAManagerCUDATypedImpl&) = delete;
    FFAManagerCUDATypedImpl& operator=(const FFAManagerCUDATypedImpl&) = delete;
    FFAManagerCUDATypedImpl(FFAManagerCUDATypedImpl&&)                 = delete;
    FFAManagerCUDATypedImpl& operator=(FFAManagerCUDATypedImpl&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        timing::ScopeTimer timer("FFAManagerCUDA::execute");
        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = cands::FFAResultWriter(
            result_file, cands::FFAResultWriter::Mode::kWrite);
        auto param_names = m_base_cfg.get_param_names();
        param_names.emplace_back("width");
        writer.write_metadata(param_names, m_base_cfg.get_scoring_widths());

        // Copy input data to device
        cudaStream_t stream = nullptr;
        m_ts_e_d.resize(ts_e.size());
        m_ts_v_d.resize(ts_v.size());
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_e_d.data()), ts_e.data(),
                        ts_e.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream);
        cuda_utils::check_last_cuda_error("cudaMemcpyAsync failed");
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_v_d.data()), ts_v.data(),
                        ts_v.size() * sizeof(float), cudaMemcpyHostToDevice,
                        stream);
        cuda_utils::check_last_cuda_error("cudaMemcpyAsync failed");
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error(
            "Input data copy stream synchronization failed");

        for (const auto& cfg_cur : m_region_planner.get_cfgs()) {
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.front(), freq_limits.back());
            cands::FFATimerStats ffa_timer_stats;
            float accumulated_flops = 0.0F;
            execute_ffa_region(cfg_cur, writer, ffa_timer_stats,
                               accumulated_flops, stream);
            // Log per-chunk timing summary
            spdlog::info("FFA Chunk: timer: {}",
                         ffa_timer_stats.get_concise_timer_summary());
            // Update accumulated stats
            m_ffa_stats->update_stats(ffa_timer_stats, accumulated_flops);
        }
        writer.write_ffa_stats(*m_ffa_stats);
        spdlog::info("FFA Manager complete.");
        spdlog::info("FFA Manager: timer: {}",
                     m_ffa_stats->get_concise_timer_summary());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    int m_device_id;
    plans::FFARegionPlanner<HostFoldType> m_region_planner;

    algorithms::FFAWorkspaceCUDA<FoldTypeCUDA> m_ffa_workspace;
    std::vector<float> m_scores;
    std::vector<uint32_t> m_passing_indices;
    std::vector<double> m_write_param_sets_batch; // includes width
    std::vector<float> m_write_scores_batch;      // passing scores

    std::unique_ptr<cands::FFAStatsCollection> m_ffa_stats;

    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_d_time;
    thrust::device_vector<float> m_scores_d;
    thrust::device_vector<SizeType> m_widths_d;

    // Helper function to create region planner with GPU memory considerations
    static plans::FFARegionPlanner<HostFoldType>
    create_region_planner(const search::PulsarSearchConfig& base_cfg,
                          int device_id) {
        cuda_utils::CudaSetDeviceGuard device_guard(device_id);
        // Query CUDA memory usage
        const auto [free_mem_gb, total_mem_gb] =
            cuda_utils::get_cuda_memory_usage();

        // Reserve memory for overhead
        constexpr double kReservedGB = 1.0; // For CUDA runtime, kernels, etc.
        const double usable_gpu_gb   = free_mem_gb - kReservedGB;

        spdlog::info("GPU Memory: {:.2f} GB total, {:.2f} GB free, {:.2f} GB "
                     "usable for chunking",
                     total_mem_gb, free_mem_gb, usable_gpu_gb);

        if (usable_gpu_gb < 1.0) {
            throw std::runtime_error(
                std::format("Insufficient GPU memory: {:.2f} GB available, "
                            "need at least 1 GB",
                            usable_gpu_gb));
        }

        // Override max_process_memory_gb for GPU-based chunking
        auto cfg_with_gpu_mem = base_cfg;
        // cfg_with_gpu_mem.set_max_process_memory_gb(usable_gpu_gb);

        // Create region planner with GPU memory limit
        return plans::FFARegionPlanner<HostFoldType>(cfg_with_gpu_mem,
                                                     /*use_gpu=*/true);
    }

    void execute_ffa_region(const search::PulsarSearchConfig& cfg,
                            cands::FFAResultWriter& result_writer,
                            cands::FFATimerStats& ffa_timer_stats,
                            float& accumulated_flops,
                            cudaStream_t stream) {
        timing::SimpleTimer timer;
        // Create FFA with shared workspace
        timer.start();
        auto the_ffa = FFACUDA<FoldTypeCUDA>(cfg, m_ffa_workspace, m_device_id);
        const plans::FFAPlan<HostFoldType>& ffa_plan = the_ffa.get_plan();
        const auto buffer_size_time = ffa_plan.get_buffer_size_time();
        const auto fold_size_time   = ffa_plan.get_fold_size_time();
        the_ffa.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_ts_e_d.data()), m_ts_e_d.size()),
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_ts_v_d.data()), m_ts_v_d.size()),
            cuda::std::span<float>(
                thrust::raw_pointer_cast(m_fold_d_time.data()),
                buffer_size_time),
            stream);
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error("FFA synchronization failed");
        const auto brutefold_time = the_ffa.get_brute_fold_timing();
        ffa_timer_stats["brutefold"] += brutefold_time;
        ffa_timer_stats["ffa"] += timer.stop() - brutefold_time;
        accumulated_flops += ffa_plan.get_gflops(/*return_in_time=*/true);

        // Compute scores
        timer.start();
        const auto nsegments       = ffa_plan.get_nsegments().back();
        const auto ncoords         = ffa_plan.get_ncoords().back();
        const auto n_params        = m_base_cfg.get_nparams();
        const auto& scoring_widths = m_base_cfg.get_scoring_widths();
        const SizeType n_widths    = scoring_widths.size();
        const auto n_scores        = ncoords * n_widths;
        error_check::check_equal(nsegments, 1U,
                                 "FFAManager::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        detection::snr_boxcar_3d_cuda_d(
            cuda::std::span(thrust::raw_pointer_cast(m_fold_d_time.data()),
                            fold_size_time),
            ncoords,
            cuda::std::span<const SizeType>(
                thrust::raw_pointer_cast(m_widths_d.data()), m_widths_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(m_scores_d.data()),
                                   n_scores),
            m_device_id);
        // Synchronize to wait for kernel completion
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error(
            "scoring kernel synchronization failed");

        // Copy scores to host
        timer.start();
        cudaMemcpyAsync(
            m_scores.data(), thrust::raw_pointer_cast(m_scores_d.data()),
            n_scores * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cuda_utils::check_last_cuda_error("scores copy failed");
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error("stream synchronization failed");
        auto scores_span = std::span(m_scores).first(n_scores);
        ffa_timer_stats["score"] += timer.stop();
        const auto score_flops =
            (ncoords * 2) * (n_widths * 2 * m_base_cfg.get_nbins());
        accumulated_flops += score_flops * 1e-9; // convert to GFLOPS

        // Fast scan to identify passing candidates
        timer.start();
        const SizeType total_params = n_params + 1;
        const auto& param_arr       = ffa_plan.get_params().back();
        const auto& param_strides   = ffa_plan.get_param_cart_strides().back();
        const auto snr_min          = m_base_cfg.get_snr_min();

        SizeType n_passing = 0;
        for (SizeType score_idx = 0; score_idx < n_scores; ++score_idx) {
            if (scores_span[score_idx] >= snr_min) {
                m_passing_indices[n_passing] = score_idx;
                n_passing++;
            }
        }

        // Process in batches and write incrementally
        SizeType batch_start = 0;
        while (batch_start < n_passing) {
            const SizeType batch_end = std::min(
                batch_start + plans::kFFAManagerWriteBatchSize, n_passing);
            const SizeType batch_count = batch_end - batch_start;

            // Fill batch buffer
            for (SizeType i = 0; i < batch_count; ++i) {
                const SizeType score_idx = m_passing_indices[batch_start + i];
                const SizeType coord_idx = score_idx / n_widths;
                const SizeType width_idx = score_idx % n_widths;

                // Reconstruct parameters from coord_idx using index arithmetic
                SizeType remaining = coord_idx;
                for (SizeType j = 0; j < n_params; ++j) {
                    const SizeType param_idx = remaining / param_strides[j];
                    remaining -= param_idx * param_strides[j];
                    m_write_param_sets_batch[(i * total_params) + j] =
                        param_arr[j][param_idx];
                }
                m_write_param_sets_batch[(i * total_params) + n_params] =
                    static_cast<double>(scoring_widths[width_idx]);
                m_write_scores_batch[i] = scores_span[score_idx];
            }

            // Write batch
            result_writer.write_results(
                std::span(m_write_param_sets_batch)
                    .first(batch_count * total_params),
                std::span(m_write_scores_batch).first(batch_count), batch_count,
                total_params);
            batch_start = batch_end;
        }

        ffa_timer_stats["filter"] += timer.stop();
    }

}; // End FFAManagerCUDATypedImpl definition

FFAManagerCUDA::FFAManagerCUDA(const search::PulsarSearchConfig& cfg,
                               int device_id) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<FFAManagerCUDATypedImpl<ComplexTypeCUDA>>(
            cfg, device_id);
    } else {
        m_impl =
            std::make_unique<FFAManagerCUDATypedImpl<float>>(cfg, device_id);
    }
}
FFAManagerCUDA::~FFAManagerCUDA()                               = default;
FFAManagerCUDA::FFAManagerCUDA(FFAManagerCUDA&& other) noexcept = default;
FFAManagerCUDA&
FFAManagerCUDA::operator=(FFAManagerCUDA&& other) noexcept = default;

void FFAManagerCUDA::execute(std::span<const float> ts_e,
                             std::span<const float> ts_v,
                             const std::filesystem::path& outdir,
                             std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}
} // namespace loki::algorithms