#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/regions.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/planner_memory.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_cuda.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/score_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/detail/error_check.hpp"
#include "lib/detail/nvtx.hpp"
#include "lib/detail/timing.hpp"
#include "lib/pipelines/ffa_freq_sweep_engine.hpp"
#include "lib/search/cands.hpp"
#include "lib/search/ffa_sweep_candidates.hpp"

namespace loki::pipelines {

template <SupportedFoldTypeCUDA FoldTypeCUDA>
class FFAFreqSweepCudaEngine final : public detail::FFAFreqSweepEngine {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;

    FFAFreqSweepCudaEngine(search::FFASearchConfig cfg, int device_id)
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
                math::check_gpu_fourier_nbins(cfg.get_nbins(),
                                              "FFAFreqSweep (CUDA)");
                n_reals.push_back(cfg.get_nbins());
            }
            m_fft_manager.prepare_plans(n_reals);
        }
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            // The fused Fourier scorer needs a Bluestein table for most
            // nbins; every chunk's is uploaded here, before the pipeline.
            cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
            for (const auto& cfg : m_region_planner.get_cfgs()) {
                const auto nbins = cfg.get_nbins();
                if (core::ffa_final_score_fourier_supported(nbins) &&
                    core::ffa_fourier_needs_bluestein(nbins)) {
                    (void)m_ffa_workspace.bluestein_table(nbins);
                }
            }
        }
        // Per-chunk widths as the kernels take them (host source of the
        // async uploads).
        m_widths_u32.reserve(m_region_decode.size());
        for (const auto& dec : m_region_decode) {
            m_widths_u32.emplace_back(dec.widths.begin(), dec.widths.end());
        }
        const auto scratch_size = planner_stats.get_max_scores_scratch_size();
        m_scores_staging.resize(scratch_size);
        m_indices_staging.resize(scratch_size);
        m_write_param_sets_batch.resize(
            planner_stats.get_write_param_sets_size());
        m_width_batch.resize(algorithms::kFFAFreqSweepWriteBatchSize);
        m_nbins_batch.resize(algorithms::kFFAFreqSweepWriteBatchSize);
        m_ffa_stats = std::make_unique<search::FFAStatsCollection>();
        cuda_utils::check_cuda_call(cudaStreamCreate(&m_stream),
                                    "cudaStreamCreate failed");
        if (m_base_cfg.get_use_boxcar_kadane()) {
            throw std::invalid_argument(
                "use_boxcar_kadane is not supported for FFA frequency sweep");
        }

        m_fold_time_d.resize(planner_stats.get_max_buffer_size_time());
        // Two score-scratch slots: the GPU fills one while the host copies
        // the other out. Fused chunks drop pairs past the capacity (and are
        // rescored); unfused chunks are planned to fit.
        cuda_utils::check_cuda_call(
            cudaStreamCreateWithFlags(&m_copy_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags failed");
        for (auto& slot : m_slots) {
            slot.scores_d.resize(scratch_size);
            slot.indices_d.resize(scratch_size);
            slot.widths_d.resize(algorithms::detail::kGpuFusedMaxWidths);
            cuda_utils::check_cuda_call(
                cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming),
                "cudaEventCreate failed");
            cuda_utils::check_cuda_call(
                cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming),
                "cudaEventCreate failed");
        }

        // Device memory the planner budgets for the sweep. "Other" is the
        // brute-fold tables.
        const float total_gb = planner_stats.get_freq_sweep_memory_usage();
        const float other_gb = total_gb -
                               planner_stats.get_buffer_memory_usage() -
                               planner_stats.get_coord_memory_usage() -
                               planner_stats.get_device_extra_memory_usage();
        spdlog::info("FfaFreqSweep device memory {:.2f} GB ({:.2f} GB fold "
                     "buffers + {:.2f} GB coords + {:.2f} GB extra + {:.2f} "
                     "GB other)",
                     total_gb, planner_stats.get_buffer_memory_usage(),
                     planner_stats.get_coord_memory_usage(),
                     planner_stats.get_device_extra_memory_usage(), other_gb);
        spdlog::info("FfaFreqSweep will process {} chunks, keeping up to "
                     "{} candidates in RAM before flushing to disk",
                     m_region_planner.get_nregions(), m_cands.get_capacity());
    }

    ~FFAFreqSweepCudaEngine() final {
        for (auto& slot : m_slots) {
            if (slot.done != nullptr) {
                cudaEventDestroy(slot.done);
            }
            if (slot.copied != nullptr) {
                cudaEventDestroy(slot.copied);
            }
        }
        if (m_copy_stream != nullptr) {
            cudaStreamDestroy(m_copy_stream);
        }
        if (m_stream != nullptr) {
            cudaStreamDestroy(m_stream);
        }
    }
    FFAFreqSweepCudaEngine(const FFAFreqSweepCudaEngine&)            = delete;
    FFAFreqSweepCudaEngine& operator=(const FFAFreqSweepCudaEngine&) = delete;
    FFAFreqSweepCudaEngine(FFAFreqSweepCudaEngine&&)                 = delete;
    FFAFreqSweepCudaEngine& operator=(FFAFreqSweepCudaEngine&&)      = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix,
                 std::string_view config_toml) override {
        const nvtx::Range range("ffa sweep (cuda)");
        timing::SimpleTimer sweep_timer;
        sweep_timer.start();
        timing::SimpleTimer timer;
        search::FFATimerStats ffa_timer_stats_pipeline;
        timer.start();
        // Reset accumulated state so repeated execute() calls are independent
        m_ffa_stats = std::make_unique<search::FFAStatsCollection>();
        m_cands.clear();
        m_total_passing_scores = 0;
        m_io_time              = 0.0F;

        // Write metadata to result file
        const std::string filebase = std::format("{}_ffa", file_prefix);
        const auto result_file =
            outdir / std::format("{}_results.h5", filebase);
        auto writer = search::FFAResultWriter(
            result_file, search::FFAResultWriter::Mode::kWrite);
        writer.write_metadata(
            search::FFAResultMetadata(m_base_cfg, config_toml));

        // Copy input data to device
        cudaStream_t stream = m_stream;
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
        // Two-slot pipeline: chunk i runs on the GPU while the host
        // collects chunk i - 1's candidates and writes them out.
        std::optional<InFlight> prev;
        std::vector<SizeType> redo;
        for (SizeType i = 0; i < ffa_regions_cfgs.size(); ++i) {
            const search::FFASearchConfig& cfg_cur = ffa_regions_cfgs[i];
            const auto& freq_limits = cfg_cur.get_param_limits().back();
            spdlog::debug("Processing chunk f0 (Hz): [{:08.3f}, {:08.3f}]",
                          freq_limits.min, freq_limits.max);
            accumulated_flops += m_region_decode[i].gflops;
            const nvtx::Range chunk_range("ffa sweep: chunk");
            timer.start();
            auto core = std::make_unique<algorithms::FFACudaCore<FoldTypeCUDA>>(
                m_ffa_workspace, m_fft_manager, cfg_cur, m_device_id,
                /*sweep_buffers=*/true);
            if (!algorithms::detail::gpu_fused_score_applies(
                    core->get_plan())) {
                // Unfused chunk: drain the pipeline and run it in order.
                if (prev) {
                    finish_in_flight(*prev, writer, redo);
                    prev.reset();
                }
                search::FFATimerStats ffa_timer_stats;
                execute_ffa_region(*core, i, writer, ffa_timer_stats, stream);
                m_ffa_stats->update_stats(ffa_timer_stats);
                continue;
            }
            const int slot = prev ? 1 - prev->slot : 0;
            InFlight cur{.region = i, .core = std::move(core), .slot = slot};
            launch_in_flight(cur, stream);
            ffa_timer_stats_pipeline["ffa"] += timer.stop();
            if (prev) {
                finish_in_flight(*prev, writer, redo);
            }
            prev = std::move(cur);
        }
        if (prev) {
            finish_in_flight(*prev, writer, redo);
            prev.reset();
        }
        // Chunks whose candidates overflowed the scratch, scored again in
        // profile ranges.
        for (const SizeType i : redo) {
            algorithms::FFACudaCore<FoldTypeCUDA> core(
                m_ffa_workspace, m_fft_manager, ffa_regions_cfgs[i],
                m_device_id, /*sweep_buffers=*/true);
            search::FFATimerStats ffa_timer_stats;
            execute_ffa_region(core, i, writer, ffa_timer_stats, stream);
            m_ffa_stats->update_stats(ffa_timer_stats);
        }

        ffa_timer_stats_pipeline["io"] += m_io_time;
        // Drain whatever is still in RAM
        timer.start();
        search::flush_candidates(m_cands, m_region_decode, writer,
                                 m_write_param_sets_batch, m_width_batch,
                                 m_nbins_batch, m_base_cfg.get_nparams());
        ffa_timer_stats_pipeline["io"] += timer.stop();
        m_ffa_stats->update_stats(ffa_timer_stats_pipeline,
                                  static_cast<float>(accumulated_flops));
        const float total_runtime = sweep_timer.stop();
        writer.write_ffa_stats(*m_ffa_stats, total_runtime);
        writer.finalize();
        spdlog::info("FFA Freq Sweep complete: {} candidates above S/N {:.2f}",
                     m_total_passing_scores, m_base_cfg.get_snr_min());
        // The stage timers do not cover every wait (GPU kernels overlap the
        // host), so the wall-clock is logged next to them.
        spdlog::info("FFA Freq Sweep: wall {:.1f} s; timer: {}", total_runtime,
                     m_ffa_stats->get_concise_timer_summary());
    }

private:
    search::FFASearchConfig m_base_cfg;
    int m_device_id;
    algorithms::FFARegionPlanner<HostFoldT> m_region_planner;
    std::vector<search::RegionDecode> m_region_decode;
    // Fixed-capacity accumulator; drained to disk whenever it fills up.
    search::CandidateBuffer m_cands;

    memory::FFAWorkspaceCUDA<FoldTypeCUDA> m_ffa_workspace;
    math::GpuFFTManager m_fft_manager;
    SizeType m_total_passing_scores{};
    // Host staging for one chunk's compacted device output.
    std::vector<float> m_scores_staging;
    std::vector<uint32_t> m_indices_staging;
    std::vector<double> m_write_param_sets_batch;
    std::vector<std::uint16_t> m_width_batch;
    std::vector<std::uint16_t> m_nbins_batch;
    cudaStream_t m_stream{nullptr};

    std::unique_ptr<search::FFAStatsCollection> m_ffa_stats;
    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<float> m_fold_time_d;
    thrust::device_vector<uint32_t> m_widths_d;

    // Host rows buffered before a write while the GPU runs the next chunk.
    static constexpr SizeType kFlushRows = SizeType{1} << 16U;
    struct ScoreSlot {
        cuda_utils::DeviceVectorNoInit<float> scores_d;
        cuda_utils::DeviceVectorNoInit<uint32_t> indices_d;
        thrust::device_vector<uint32_t> widths_d;
        memory::DeviceCounter counter;
        core::FFAFinalScoreArgs args;
        cudaEvent_t done{nullptr};   // chunk scored, count copied
        cudaEvent_t copied{nullptr}; // candidates copied out
    };
    std::array<ScoreSlot, 2> m_slots;
    std::vector<std::vector<uint32_t>> m_widths_u32;
    cudaStream_t m_copy_stream{nullptr};
    float m_io_time{0.0F};

    memory::DeviceCounter m_passing_counter;

    // The plan of FFARegionPlanner(Exec::cuda): the memory limit capped by
    // the device's free memory less the reserve (shared with the dry run).
    static algorithms::FFARegionPlanner<HostFoldT>
    create_region_planner(const search::FFASearchConfig& base_cfg,
                          int device_id) {
        return algorithms::FFARegionPlanner<HostFoldT>(base_cfg,
                                                       Exec::cuda(device_id));
    }

    // A fused chunk whose GPU work is queued: its core, and the scratch slot
    // its candidates and count land in.
    struct InFlight {
        SizeType region{};
        std::unique_ptr<algorithms::FFACudaCore<FoldTypeCUDA>> core;
        int slot{};
    };

    /// Queue a fused chunk on `stream`; nothing here waits for the GPU.
    void launch_in_flight(InFlight& f, cudaStream_t stream) {
        const auto& dec = m_region_decode[f.region];
        const auto& cfg = m_region_planner.get_cfgs()[f.region];
        auto& slot      = m_slots[static_cast<size_t>(f.slot)];
        error_check::check_equal(f.core->get_plan().get_ncoords().back(),
                                 dec.ncoords,
                                 "FfaFreqSweep: decode table is out of sync "
                                 "with the FFA plan");
        error_check::check_less_equal(dec.widths.size(), slot.widths_d.size(),
                                      "FfaFreqSweep: too many boxcar widths");
        // The slot's scratch is free once the host copied its previous
        // chunk out (a no-op before the slot is first used).
        cuda_utils::check_cuda_call(cudaStreamWaitEvent(stream, slot.copied, 0),
                                    "FfaFreqSweep: cudaStreamWaitEvent failed");
        // Every upload of this chunk goes through the slot's pinned staging
        // (truly asynchronous; the slot's previous chunk has finished).
        m_ffa_workspace.select_staging_slot(f.slot);
        const auto& widths_u32 = m_widths_u32[f.region];
        const void* widths_h   = m_ffa_workspace.stage(
            widths_u32.data(), widths_u32.size() * sizeof(uint32_t));
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(slot.widths_d.data()),
                            widths_h, widths_u32.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "FfaFreqSweep: widths upload failed");
        slot.counter.reset(stream);
        const float fold_scale =
            std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                ? 1.0F / static_cast<float>(cfg.get_nbins())
                : 1.0F;
        slot.args = core::FFAFinalScoreArgs{
            .widths     = thrust::raw_pointer_cast(slot.widths_d.data()),
            .nwidths    = static_cast<uint32_t>(dec.widths.size()),
            .threshold  = static_cast<float>(cfg.get_snr_min()),
            .fold_scale = fold_scale,
            .scores     = thrust::raw_pointer_cast(slot.scores_d.data()),
            .indices    = thrust::raw_pointer_cast(slot.indices_d.data()),
            .counter    = slot.counter.data(),
            .capacity   = static_cast<uint32_t>(slot.scores_d.size()),
        };
        const cuda::std::span<DeviceFoldT> work(
            reinterpret_cast<DeviceFoldT*>(
                thrust::raw_pointer_cast(m_fold_time_d.data())),
            algorithms::detail::gpu_sweep_buffer_size(f.core->get_plan()));
        f.core->execute_score(cuda_utils::as_span(m_ts_e_d),
                              cuda_utils::as_span(m_ts_v_d), work, slot.args,
                              stream);
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(slot.counter.h_ptr, slot.counter.d_ptr,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
            "FfaFreqSweep: counter copy failed");
        cuda_utils::check_cuda_call(cudaEventRecord(slot.done, stream),
                                    "FfaFreqSweep: cudaEventRecord failed");
        m_ffa_workspace.select_staging_slot(-1);
    }

    /// Wait for a queued chunk and move its candidates to the host buffer,
    /// writing the buffer out once it holds kFlushRows candidates.
    void finish_in_flight(InFlight& f,
                          search::FFAResultWriter& writer,
                          std::vector<SizeType>& redo) {
        const nvtx::Range range("ffa sweep: collect candidates");
        auto& slot = m_slots[static_cast<size_t>(f.slot)];
        cuda_utils::check_cuda_call(
            cudaEventSynchronize(slot.done),
            "FfaFreqSweep: cudaEventSynchronize failed");
        const SizeType n_passing = *slot.counter.h_ptr;
        cuda_utils::check_cuda_call(
            cudaStreamWaitEvent(m_copy_stream, slot.done, 0),
            "FfaFreqSweep: cudaStreamWaitEvent failed");
        if (n_passing > slot.args.capacity) {
            // Rescored in ranges after the sweep (the parent level is gone).
            redo.push_back(f.region);
        } else if (n_passing > 0) {
            // Pageable destination: the copy has completed on return.
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(m_scores_staging.data(),
                                thrust::raw_pointer_cast(slot.scores_d.data()),
                                n_passing * sizeof(float),
                                cudaMemcpyDeviceToHost, m_copy_stream),
                "FfaFreqSweep: scores copy failed");
            cuda_utils::check_cuda_call(
                cudaMemcpyAsync(m_indices_staging.data(),
                                thrust::raw_pointer_cast(slot.indices_d.data()),
                                n_passing * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, m_copy_stream),
                "FfaFreqSweep: indices copy failed");
        }
        cuda_utils::check_cuda_call(cudaEventRecord(slot.copied, m_copy_stream),
                                    "FfaFreqSweep: cudaEventRecord failed");
        if (n_passing > slot.args.capacity) {
            return;
        }
        timing::SimpleTimer timer;
        timer.start();
        stage_candidates(n_passing, f.region, writer);
        m_total_passing_scores += n_passing;
        if (m_cands.get_size() >= kFlushRows) {
            search::flush_candidates(m_cands, m_region_decode, writer,
                                     m_write_param_sets_batch, m_width_batch,
                                     m_nbins_batch, m_base_cfg.get_nparams());
        }
        m_io_time += timer.stop();
    }

    void execute_ffa_region(algorithms::FFACudaCore<FoldTypeCUDA>& the_ffa,
                            SizeType region_id,
                            search::FFAResultWriter& writer,
                            search::FFATimerStats& ffa_timer_stats,
                            cudaStream_t stream) {
        timing::SimpleTimer timer;
        timer.start();
        const auto& cfg = m_region_planner.get_cfgs()[region_id];
        const plans::FFAPlan<HostFoldT>& ffa_plan = the_ffa.get_plan();
        const auto& dec                           = m_region_decode[region_id];
        error_check::check_equal(dec.nsegments, SizeType{1},
                                 "FfaFreqSweep::execute_ffa_region: nsegments "
                                 "must be 1 to call scoring function");
        // The decode strides must describe the fold layout we just produced.
        error_check::check_equal(ffa_plan.get_ncoords().back(), dec.ncoords,
                                 "FfaFreqSweep::execute_ffa_region: decode "
                                 "table is out of sync with the FFA plan");
        // Boxcar widths follow nbins, which differs between regions, so they
        // must be re-uploaded per chunk rather than once at construction.
        m_widths_d = dec.widths;
        // Fourier folds skip the IRFFT 1/nbins pass; the scorer applies it
        // to each value as it loads it (the same multiply, so the same bits).
        const float fold_scale =
            std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>
                ? 1.0F / static_cast<float>(cfg.get_nbins())
                : 1.0F;
        const auto threshold = static_cast<float>(cfg.get_snr_min());

        if (algorithms::detail::gpu_fused_score_applies(ffa_plan)) {
            // The last level is merged and scored on chip; the final fold
            // is never stored. The work buffer is the second ping-pong
            // buffer, in fold units.
            core::FFAFinalScoreArgs args{
                .widths     = thrust::raw_pointer_cast(m_widths_d.data()),
                .nwidths    = static_cast<uint32_t>(m_widths_d.size()),
                .threshold  = threshold,
                .fold_scale = fold_scale,
                .scores = thrust::raw_pointer_cast(m_slots[0].scores_d.data()),
                .indices =
                    thrust::raw_pointer_cast(m_slots[0].indices_d.data()),
                .counter  = m_passing_counter.data(),
                .capacity = static_cast<uint32_t>(m_slots[0].scores_d.size()),
            };
            const cuda::std::span<DeviceFoldT> work(
                reinterpret_cast<DeviceFoldT*>(
                    thrust::raw_pointer_cast(m_fold_time_d.data())),
                algorithms::detail::gpu_sweep_buffer_size(ffa_plan));
            m_passing_counter.reset(stream);
            the_ffa.execute_score(cuda_utils::as_span(m_ts_e_d),
                                  cuda_utils::as_span(m_ts_v_d), work, args,
                                  stream);
            const SizeType n_passing  = m_passing_counter.value_sync(stream);
            const auto brutefold_time = the_ffa.get_brute_fold_timing();
            ffa_timer_stats["brutefold"] += brutefold_time;
            ffa_timer_stats["ffa"] += timer.stop() - brutefold_time;

            timer.start();
            if (n_passing <= args.capacity) {
                copy_candidates_to_host(n_passing, region_id, writer, stream);
            } else {
                rescore_in_ranges(the_ffa, args, dec.ncoords, n_passing,
                                  region_id, writer, stream);
            }
            m_total_passing_scores += n_passing;
            ffa_timer_stats["io"] += timer.stop();
            return;
        }

        // Unfused: store the final fold, then score it.
        const auto n_scores = dec.get_n_scores();
        error_check::check_less_equal(
            n_scores, SizeType{m_slots[0].scores_d.size()},
            "FfaFreqSweep::execute_ffa_region: score scratch too small for "
            "an unfused chunk");
        const auto fold_size_time = ffa_plan.get_fold_size_time();
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            the_ffa.execute(cuda_utils::as_span(m_ts_e_d),
                            cuda_utils::as_span(m_ts_v_d),
                            cuda_utils::as_span(
                                m_fold_time_d, ffa_plan.get_buffer_size_time()),
                            stream, /*normalize=*/false);
        } else {
            the_ffa.execute(
                cuda_utils::as_span(m_ts_e_d), cuda_utils::as_span(m_ts_v_d),
                cuda_utils::as_span(m_fold_time_d, ffa_plan.get_buffer_size()),
                stream);
        }
        const auto brutefold_time = the_ffa.get_brute_fold_timing();
        ffa_timer_stats["brutefold"] += brutefold_time;
        ffa_timer_stats["ffa"] += timer.stop() - brutefold_time;

        timer.start();
        const SizeType n_passing = detection::score_and_filter_cuda_d(
            cuda_utils::as_span(m_fold_time_d, fold_size_time),
            cuda_utils::as_span(m_widths_d),
            cuda_utils::as_span(m_slots[0].scores_d, n_scores),
            cuda_utils::as_span(m_slots[0].indices_d, n_scores), threshold,
            dec.ncoords, dec.nbins, stream, m_passing_counter, fold_scale);
        ffa_timer_stats["score"] += timer.stop();

        timer.start();
        copy_candidates_to_host(n_passing, region_id, writer, stream);
        m_total_passing_scores += n_passing;
        ffa_timer_stats["io"] += timer.stop();
    }

    /**
     * @brief More pairs passed than the score scratch holds: score the final
     * level again in profile ranges, halving a range until it fits. The
     * parent level is still intact, so only the fused final level reruns.
     */
    void rescore_in_ranges(algorithms::FFACudaCore<FoldTypeCUDA>& the_ffa,
                           core::FFAFinalScoreArgs args,
                           SizeType ncoords,
                           SizeType n_passing,
                           SizeType region_id,
                           search::FFAResultWriter& writer,
                           cudaStream_t stream) {
        const SizeType cap = args.capacity;
        const SizeType pieces =
            std::min<SizeType>(ncoords, ((2 * n_passing) + cap - 1) / cap);
        std::vector<std::pair<SizeType, SizeType>> todo;
        for (SizeType i = pieces; i-- > 0;) {
            todo.emplace_back((ncoords * i) / pieces,
                              (ncoords * (i + 1)) / pieces);
        }
        SizeType total = 0;
        while (!todo.empty()) {
            const auto [begin, end] = todo.back();
            todo.pop_back();
            args.profile_begin = static_cast<uint32_t>(begin);
            args.profile_end   = static_cast<uint32_t>(end);
            m_passing_counter.reset(stream);
            the_ffa.rescore_final(args, stream);
            const SizeType n = m_passing_counter.value_sync(stream);
            if (n > cap) {
                // One profile has at most kGpuFusedMaxWidths pairs.
                error_check::check_greater(
                    end - begin, SizeType{1},
                    "FfaFreqSweep: score scratch below one profile");
                const SizeType mid = begin + ((end - begin) / 2);
                todo.emplace_back(mid, end);
                todo.emplace_back(begin, mid);
                continue;
            }
            copy_candidates_to_host(n, region_id, writer, stream);
            total += n;
        }
        error_check::check_equal(total, n_passing,
                                 "FfaFreqSweep: range rescoring lost pairs");
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
                                 search::FFAResultWriter& writer,
                                 cudaStream_t stream) {
        if (n_passing == 0) {
            return;
        }
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                m_scores_staging.data(),
                thrust::raw_pointer_cast(m_slots[0].scores_d.data()),
                n_passing * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "scores copy failed");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                m_indices_staging.data(),
                thrust::raw_pointer_cast(m_slots[0].indices_d.data()),
                n_passing * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
            "passing indices copy failed");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "stream synchronization failed");
        stage_candidates(n_passing, region_id, writer);
    }

    /// Append the staged candidates of one chunk to the host accumulator,
    /// draining it to disk whenever it fills up.
    void stage_candidates(SizeType n_passing,
                          SizeType region_id,
                          search::FFAResultWriter& writer) {
        SizeType copied = 0;
        while (copied < n_passing) {
            if (m_cands.is_full()) {
                search::flush_candidates(
                    m_cands, m_region_decode, writer, m_write_param_sets_batch,
                    m_width_batch, m_nbins_batch, m_base_cfg.get_nparams());
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

}; // End FFAFreqSweepCudaEngine definition

namespace detail {
std::unique_ptr<FFAFreqSweepEngine> make_ffa_freq_sweep_gpu(
    const search::FFASearchConfig& cfg, int device_id, bool show_progress) {
    (void)show_progress;
    if (cfg.get_use_fourier()) {
        return std::make_unique<FFAFreqSweepCudaEngine<ComplexTypeCUDA>>(
            cfg, device_id);
    }
    return std::make_unique<FFAFreqSweepCudaEngine<float>>(cfg, device_id);
}
} // namespace detail

} // namespace loki::pipelines
