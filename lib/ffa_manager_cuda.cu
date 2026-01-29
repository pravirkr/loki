#include "loki/pipelines/ffa_manager.hpp"

#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/plans.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/cands.hpp"
#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"
#include "loki/psr_utils.hpp"

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
            planner_stats.get_max_coord_size(),
            planner_stats.get_max_ffa_levels(), m_base_cfg.get_nparams());
        m_scores.resize(planner_stats.get_max_scores_size());
        m_passing_indices.resize(planner_stats.get_max_scores_size());
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_n_passing_scores_per_region.resize(m_region_planner.get_nregions());
        m_ffa_stats = std::make_unique<cands::FFAStatsCollection>();

        m_fold_time_d.resize(planner_stats.get_max_buffer_size_time());
        m_scores_d.resize(planner_stats.get_max_scores_size());
        m_passing_indices_d.resize(planner_stats.get_max_scores_size());

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
        timing::SimpleTimer timer;
        cands::FFATimerStats ffa_timer_stats_pipeline;
        timer.start();
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
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_e_d.data()),
                            ts_e.data(), ts_e.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync ts_e failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ts_v_d.data()),
                            ts_v.data(), ts_v.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync ts_v failed");
        cuda_utils::check_cuda_call(
            cudaStreamSynchronize(stream),
            "Input data copy stream synchronization failed");
        ffa_timer_stats_pipeline["io"] += timer.stop();

        m_total_passing_scores       = 0; // reset total passing scores
        const auto& ffa_regions_cfgs = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::PulsarSearchConfig& cfg_cur = ffa_regions_cfgs[i];
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.min, freq_limits.max);
            cands::FFATimerStats ffa_timer_stats;
            const SizeType n_passing =
                execute_ffa_region(cfg_cur, ffa_timer_stats, stream);
            m_n_passing_scores_per_region[i] = n_passing;
            // Log per-chunk timing summary
            spdlog::info("FFA Chunk: timer: {}",
                         ffa_timer_stats.get_concise_timer_summary());
            // Update accumulated stats
            m_ffa_stats->update_stats(ffa_timer_stats);
        }

        // Copy scores to host
        timer.start();
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(m_scores.data(),
                            thrust::raw_pointer_cast(m_scores_d.data()),
                            m_total_passing_scores * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "scores copy failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                m_passing_indices.data(),
                thrust::raw_pointer_cast(m_passing_indices_d.data()),
                m_total_passing_scores * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream),
            "passing indices copy failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "stream synchronization failed");

        const float accumulated_flops = save_results(writer);
        ffa_timer_stats_pipeline["io"] += timer.stop();
        m_ffa_stats->update_stats(ffa_timer_stats_pipeline, accumulated_flops);
        writer.write_ffa_stats(*m_ffa_stats);
        spdlog::info("FFA Manager complete.");
        spdlog::info("FFA Manager: timer: {}",
                     m_ffa_stats->get_concise_timer_summary());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    int m_device_id;
    regions::FFARegionPlanner<HostFoldType> m_region_planner;

    algorithms::FFAWorkspaceCUDA<FoldTypeCUDA> m_ffa_workspace;
    SizeType m_total_passing_scores{};
    std::vector<float> m_scores;
    std::vector<uint32_t> m_passing_indices;
    std::vector<double> m_write_param_sets_batch; // includes width
    std::vector<SizeType> m_n_passing_scores_per_region;

    std::unique_ptr<cands::FFAStatsCollection> m_ffa_stats;
    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_time_d;
    thrust::device_vector<float> m_scores_d;
    thrust::device_vector<uint32_t> m_widths_d;
    thrust::device_vector<uint32_t> m_passing_indices_d;

    cuda_utils::DeviceCounter m_passing_counter;

    // Helper function to create region planner with GPU memory considerations
    static regions::FFARegionPlanner<HostFoldType>
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
        return regions::FFARegionPlanner<HostFoldType>(cfg_with_gpu_mem,
                                                       /*use_gpu=*/true);
    }

    SizeType execute_ffa_region(const search::PulsarSearchConfig& cfg,
                                cands::FFATimerStats& ffa_timer_stats,
                                cudaStream_t stream) {
        timing::SimpleTimer timer;
        // Create FFA with shared workspace
        timer.start();
        auto the_ffa = FFACUDA<FoldTypeCUDA>(cfg, m_ffa_workspace, m_device_id);
        const plans::FFAPlan<HostFoldType>& ffa_plan = the_ffa.get_plan();
        const auto buffer_size_time = ffa_plan.get_buffer_size_time();
        const auto fold_size_time   = ffa_plan.get_fold_size_time();
        the_ffa.execute(
            cuda_utils::as_span(m_ts_e_d), cuda_utils::as_span(m_ts_v_d),
            cuda_utils::as_span(m_fold_time_d, buffer_size_time), stream);
        const auto brutefold_time = the_ffa.get_brute_fold_timing();
        ffa_timer_stats["brutefold"] += brutefold_time;
        ffa_timer_stats["ffa"] += timer.stop() - brutefold_time;

        // Compute scores
        timer.start();
        const auto nsegments = ffa_plan.get_nsegments().back();
        const auto ncoords   = ffa_plan.get_ncoords().back();
        const auto n_widths  = cfg.get_scoring_widths().size();
        const auto n_scores  = ncoords * n_widths;
        error_check::check_equal(nsegments, 1U,
                                 "FFAManager::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        // Calculate available space in buffers
        const SizeType available_space =
            m_scores_d.size() - m_total_passing_scores;
        error_check::check_greater_equal(
            available_space, n_scores,
            std::format("Buffer overflow: {} candidates already accumulated, "
                        "{} more needed, but only {} space available. "
                        "Options: (1) Increase snr_min threshold, "
                        "(2) Add max_passing_candidates config parameter.",
                        m_total_passing_scores, n_scores, available_space));
        // Pass incremental spans with offset
        const SizeType n_passing = detection::score_and_filter_cuda_d(
            cuda_utils::as_span(m_fold_time_d, fold_size_time),
            cuda_utils::as_span(m_widths_d),
            cuda_utils::as_span(m_scores_d)
                .subspan(m_total_passing_scores, n_scores),
            cuda_utils::as_span(m_passing_indices_d)
                .subspan(m_total_passing_scores, n_scores),
            cfg.get_snr_min(), ncoords, cfg.get_nbins(), stream,
            m_passing_counter);
        m_total_passing_scores += n_passing;

        ffa_timer_stats["score"] += timer.stop();
        return n_passing;
    }

    float save_results(cands::FFAResultWriter& result_writer) {
        const auto n_params         = m_base_cfg.get_nparams();
        const SizeType total_params = n_params + 1;
        const auto& scoring_widths  = m_base_cfg.get_scoring_widths();
        const SizeType n_widths     = scoring_widths.size();

        float accumulated_flops        = 0.0F;
        SizeType global_passing_offset = 0; // Track cumulative offset
        const auto& ffa_regions_cfgs   = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::PulsarSearchConfig& cfg_cur = ffa_regions_cfgs[i];
            plans::FFAPlan<HostFoldType> ffa_plan(cfg_cur);
            const auto& param_limits = cfg_cur.get_param_limits();
            const auto& param_counts = ffa_plan.get_param_counts().back();
            const auto& param_strides =
                ffa_plan.get_param_cart_strides().back();
            const auto n_passing = m_n_passing_scores_per_region[i];

            // Compute flops
            accumulated_flops += ffa_plan.get_gflops(/*return_in_time=*/true);
            const auto ncoords = ffa_plan.get_ncoords().back();
            const auto score_flops =
                (ncoords * 2) * (n_widths * 2 * cfg_cur.get_nbins());
            accumulated_flops += score_flops * 1e-9; // convert to GFLOPS

            // Process in batches and write incrementally
            SizeType batch_start = 0;
            while (batch_start < n_passing) {
                const SizeType batch_end =
                    std::min(batch_start + regions::kFFAManagerWriteBatchSize,
                             n_passing);
                const SizeType batch_count = batch_end - batch_start;

                // Fill batch buffer
                for (SizeType i = 0; i < batch_count; ++i) {
                    // Access from global buffer with proper offset
                    const SizeType global_idx =
                        global_passing_offset + batch_start + i;
                    const SizeType score_idx = m_passing_indices[global_idx];
                    const SizeType coord_idx = score_idx / n_widths;
                    const SizeType width_idx = score_idx % n_widths;

                    // Reconstruct parameters from coord_idx using index
                    // arithmetic
                    SizeType remaining = coord_idx;
                    for (SizeType j = 0; j < n_params; ++j) {
                        const SizeType param_idx = remaining / param_strides[j];
                        remaining -= param_idx * param_strides[j];
                        m_write_param_sets_batch[(i * total_params) + j] =
                            psr_utils::get_param_val_at_idx(
                                param_limits[j], param_counts[j], param_idx);
                    }
                    m_write_param_sets_batch[(i * total_params) + n_params] =
                        static_cast<double>(scoring_widths[width_idx]);
                }

                // Write batch
                result_writer.write_results(
                    std::span(m_write_param_sets_batch)
                        .first(batch_count * total_params),
                    std::span(m_scores).subspan(
                        global_passing_offset + batch_start, batch_count),
                    batch_count, total_params);
                batch_start = batch_end;
            }
            global_passing_offset += n_passing;
        }
        return accumulated_flops;
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