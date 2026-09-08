#include "loki/pipelines/ffa_freq_sweep.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/cands.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/ffa_sweep_candidates.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::algorithms {

class FFAFreqSweepCUDA::BaseImpl {
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
class FFAFreqSweepCUDATypedImpl final : public FFAFreqSweepCUDA::BaseImpl {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;

    FFAFreqSweepCUDATypedImpl(search::PulsarSearchConfig cfg, int device_id)
        : m_base_cfg(std::move(cfg)),
          m_device_id(device_id),
          m_region_planner(create_region_planner(m_base_cfg, m_device_id)),
          m_region_decode(build_region_decode_table<HostFoldT>(
              m_region_planner.get_cfgs())),
          m_cands(m_region_planner.get_stats().get_max_candidates()),
          m_fft_manager(device_id) {
        const auto& planner_stats = m_region_planner.get_stats();
        // Allocate buffers once, sized for the largest chunk
        m_ffa_workspace = memory::FFAWorkspaceCUDA<FoldTypeCUDA>(
            planner_stats.get_max_buffer_size(),
            planner_stats.get_max_coord_size(),
            planner_stats.get_max_ffa_levels(), m_base_cfg.get_nparams());
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            const auto& cfgs = m_region_planner.get_cfgs();
            std::vector<SizeType> n_reals;
            n_reals.reserve(cfgs.size());
            for (const auto& cfg : cfgs) {
                n_reals.push_back(cfg.get_nbins());
            }
            m_fft_manager.prepare_plans(n_reals);
        }
        const auto scratch_size = planner_stats.get_max_scores_scratch_size();
        m_scores_staging.resize(scratch_size);
        m_indices_staging.resize(scratch_size);
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_ffa_stats = std::make_unique<cands::FFAStatsCollection>();

        m_fold_time_d.resize(planner_stats.get_max_buffer_size_time());
        // The filter kernel claims output slots with an unbounded atomic, so
        // the device buffers must cover the worst case of every score passing.
        m_scores_d.resize(scratch_size);
        m_passing_indices_d.resize(scratch_size);
        validate_scratch_sizes(scratch_size);

        // Log the actual memory usage for the allocated buffers
        spdlog::info("FFAFreqSweepCUDA allocated {:.2f} GB ({:.2f} GB buffers "
                     "+ {:.2f} GB coords + {:.2f} GB extra)",
                     planner_stats.get_freq_sweep_memory_usage(),
                     planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_device_extra_memory_usage());
        spdlog::info("FFAFreqSweepCUDA will process {} chunks, keeping up to "
                     "{} candidates in RAM before flushing to disk",
                     m_region_planner.get_nregions(), m_cands.get_capacity());
    }

    ~FFAFreqSweepCUDATypedImpl() final                          = default;
    FFAFreqSweepCUDATypedImpl(const FFAFreqSweepCUDATypedImpl&) = delete;
    FFAFreqSweepCUDATypedImpl&
    operator=(const FFAFreqSweepCUDATypedImpl&)                       = delete;
    FFAFreqSweepCUDATypedImpl(FFAFreqSweepCUDATypedImpl&&)            = delete;
    FFAFreqSweepCUDATypedImpl& operator=(FFAFreqSweepCUDATypedImpl&&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) override {
        timing::SimpleTimer timer;
        cands::FFATimerStats ffa_timer_stats_pipeline;
        timer.start();
        // Reset accumulated state so repeated execute() calls are independent
        m_ffa_stats = std::make_unique<cands::FFAStatsCollection>();
        m_cands.clear();
        m_total_passing_scores = 0;

        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = cands::FFAResultWriter(
            result_file, cands::FFAResultWriter::Mode::kWrite);
        auto param_names = m_base_cfg.get_param_names();
        param_names.emplace_back("width");
        writer.write_metadata(param_names, m_base_cfg.get_nbins(),
                              m_base_cfg.get_ducy_max(), m_base_cfg.get_wtsp());

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

        double accumulated_flops     = 0.0;
        const auto& ffa_regions_cfgs = m_region_planner.get_cfgs();
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::PulsarSearchConfig& cfg_cur = ffa_regions_cfgs[i];
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::info("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                         freq_limits.min, freq_limits.max);
            cands::FFATimerStats ffa_timer_stats;
            execute_ffa_region(cfg_cur, i, writer, ffa_timer_stats, stream);
            accumulated_flops += m_region_decode[i].gflops;
            // Log per-chunk timing summary
            spdlog::info("FFA Chunk: timer: {}",
                         ffa_timer_stats.get_concise_timer_summary());
            // Update accumulated stats
            m_ffa_stats->update_stats(ffa_timer_stats);
        }

        // Drain whatever is still in RAM
        timer.start();
        flush_candidates(m_cands, m_region_decode, writer,
                         m_write_param_sets_batch, m_base_cfg.get_nparams());
        ffa_timer_stats_pipeline["io"] += timer.stop();
        m_ffa_stats->update_stats(ffa_timer_stats_pipeline,
                                  static_cast<float>(accumulated_flops));
        writer.write_ffa_stats(*m_ffa_stats);
        spdlog::info("FFA Freq Sweep complete: {} candidates above S/N {:.2f}",
                     m_total_passing_scores, m_base_cfg.get_snr_min());
        spdlog::info("FFA Freq Sweep: timer: {}",
                     m_ffa_stats->get_concise_timer_summary());
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    int m_device_id;
    regions::FFARegionPlanner<HostFoldT> m_region_planner;
    std::vector<RegionDecode> m_region_decode;
    // Fixed-capacity accumulator; drained to disk whenever it fills up.
    CandidateBuffer m_cands;

    memory::FFAWorkspaceCUDA<FoldTypeCUDA> m_ffa_workspace;
    math::CUFFTManager m_fft_manager;
    SizeType m_total_passing_scores{};
    // Host staging for one chunk's compacted device output.
    std::vector<float> m_scores_staging;
    std::vector<uint32_t> m_indices_staging;
    std::vector<double> m_write_param_sets_batch; // includes width

    std::unique_ptr<cands::FFAStatsCollection> m_ffa_stats;
    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_time_d;
    thrust::device_vector<float> m_scores_d;
    thrust::device_vector<uint32_t> m_widths_d;
    thrust::device_vector<uint32_t> m_passing_indices_d;

    memory::DeviceCounter m_passing_counter;

    // Helper function to create region planner with GPU memory considerations
    static regions::FFARegionPlanner<HostFoldT>
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

        const double user_limit   = base_cfg.get_max_process_memory_gb();
        const double gpu_limit_gb = std::min(user_limit, usable_gpu_gb);
        spdlog::info("Using {:.2f} GB GPU chunking budget (user limit {:.2f} "
                     "GB, usable GPU {:.2f} GB)",
                     gpu_limit_gb, user_limit, usable_gpu_gb);

        auto cfg_with_gpu_mem = base_cfg;
        cfg_with_gpu_mem.set_max_process_memory_gb(gpu_limit_gb);

        // Create region planner with GPU memory limit
        return regions::FFARegionPlanner<HostFoldT>(cfg_with_gpu_mem,
                                                    /*use_gpu=*/true);
    }

    void execute_ffa_region(const search::PulsarSearchConfig& cfg,
                            SizeType region_id,
                            cands::FFAResultWriter& writer,
                            cands::FFATimerStats& ffa_timer_stats,
                            cudaStream_t stream) {
        timing::SimpleTimer timer;
        // Create FFA with shared workspace
        timer.start();
        auto the_ffa = FFACUDA<FoldTypeCUDA>(m_ffa_workspace, m_fft_manager,
                                             cfg, m_device_id);
        const plans::FFAPlan<HostFoldT>& ffa_plan = the_ffa.get_plan();
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
        const auto& dec     = m_region_decode[region_id];
        const auto n_scores = dec.get_n_scores();
        error_check::check_equal(
            dec.nsegments, SizeType{1},
            "FFAFreqSweepCUDA::execute_ffa_region: nsegments "
            "must be 1 to call scoring function");
        // The decode strides must describe the fold layout we just produced.
        error_check::check_equal(ffa_plan.get_ncoords().back(), dec.ncoords,
                                 "FFAFreqSweepCUDA::execute_ffa_region: decode "
                                 "table is out of sync with the FFA plan");
        // Boxcar widths follow nbins, which differs between regions, so they
        // must be re-uploaded per chunk rather than once at construction.
        m_widths_d = dec.widths;

        // Device buffers are per-chunk scratch: the offset no longer depends
        // on how many candidates have already survived.
        const SizeType n_passing = detection::score_and_filter_cuda_d(
            cuda_utils::as_span(m_fold_time_d, fold_size_time),
            cuda_utils::as_span(m_widths_d),
            cuda_utils::as_span(m_scores_d, n_scores),
            cuda_utils::as_span(m_passing_indices_d, n_scores),
            static_cast<float>(cfg.get_snr_min()), dec.ncoords, dec.nbins,
            stream, m_passing_counter);
        ffa_timer_stats["score"] += timer.stop();

        timer.start();
        copy_candidates_to_host(n_passing, region_id, writer, stream);
        m_total_passing_scores += n_passing;
        ffa_timer_stats["io"] += timer.stop();
    }

    /**
     * @brief Move this chunk's compacted candidates from device to the host
     * accumulator, draining the accumulator whenever it fills up.
     *
     * @note The filter kernel claims output slots via atomicAdd, so the order
     * within a chunk (and hence the row order in the result file) is not
     * reproducible run to run.
     */
    void copy_candidates_to_host(SizeType n_passing,
                                 SizeType region_id,
                                 cands::FFAResultWriter& writer,
                                 cudaStream_t stream) {
        if (n_passing == 0) {
            return;
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(m_scores_staging.data(),
                            thrust::raw_pointer_cast(m_scores_d.data()),
                            n_passing * sizeof(float), cudaMemcpyDeviceToHost,
                            stream),
            "scores copy failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                m_indices_staging.data(),
                thrust::raw_pointer_cast(m_passing_indices_d.data()),
                n_passing * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
            "passing indices copy failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "stream synchronization failed");

        SizeType copied = 0;
        while (copied < n_passing) {
            if (m_cands.is_full()) {
                flush_candidates(m_cands, m_region_decode, writer,
                                 m_write_param_sets_batch,
                                 m_base_cfg.get_nparams());
            }
            const SizeType chunk =
                std::min(m_cands.get_space(), n_passing - copied);
            std::copy_n(m_scores_staging.begin() +
                            static_cast<IndexType>(copied),
                        chunk, m_cands.get_scores_tail(chunk).begin());
            std::copy_n(m_indices_staging.begin() +
                            static_cast<IndexType>(copied),
                        chunk, m_cands.get_indices_tail(chunk).begin());
            m_cands.commit(chunk, static_cast<uint32_t>(region_id));
            copied += chunk;
        }
    }

    /// @brief Assert the planner-derived scratch sizes cover every chunk.
    void validate_scratch_sizes(SizeType scratch_size) const {
        for (SizeType i = 0; i < m_region_decode.size(); ++i) {
            error_check::check_less_equal(
                m_region_decode[i].get_n_scores(), scratch_size,
                std::format("FFAFreqSweepCUDA: chunk {} needs {} score slots "
                            "but the planner only sized the scratch for {}",
                            i, m_region_decode[i].get_n_scores(),
                            scratch_size));
        }
    }

}; // End FFAFreqSweepCUDATypedImpl definition

FFAFreqSweepCUDA::FFAFreqSweepCUDA(const search::PulsarSearchConfig& cfg,
                                   int device_id) {
    if (cfg.get_use_fourier()) {
        m_impl = std::make_unique<FFAFreqSweepCUDATypedImpl<ComplexTypeCUDA>>(
            cfg, device_id);
    } else {
        m_impl =
            std::make_unique<FFAFreqSweepCUDATypedImpl<float>>(cfg, device_id);
    }
}
FFAFreqSweepCUDA::~FFAFreqSweepCUDA()                                 = default;
FFAFreqSweepCUDA::FFAFreqSweepCUDA(FFAFreqSweepCUDA&& other) noexcept = default;
FFAFreqSweepCUDA&
FFAFreqSweepCUDA::operator=(FFAFreqSweepCUDA&& other) noexcept = default;

void FFAFreqSweepCUDA::execute(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               const std::filesystem::path& outdir,
                               std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}
} // namespace loki::algorithms