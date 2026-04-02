#include "loki/algorithms/ffa.hpp"

#include <memory>

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/core/taylor_ffa.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
#include "loki/timing.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::algorithms {

// FFACUDA::Impl implementation
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class FFACUDA<FoldTypeCUDA>::Impl {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;

    explicit Impl(search::PulsarSearchConfig cfg, int device_id)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_device_id(device_id),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_workspace_storage(m_ffa_plan),
          m_workspace_ptr(&m_workspace_storage) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        // Validate workspace
        const auto& ws = get_workspace();
        ws.validate(m_ffa_plan);
        // Initialize BruteFold
        initialize_brute_fold();
        log_info();
    }

    explicit Impl(memory::FFAWorkspaceCUDA<FoldTypeCUDA>& workspace,
                  search::PulsarSearchConfig cfg,
                  int device_id)
        : m_cfg(std::move(cfg)),
          m_ffa_plan(m_cfg),
          m_device_id(device_id),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_workspace_storage(),
          m_workspace_ptr(&workspace) {
        cuda_utils::CudaSetDeviceGuard device_guard(m_device_id);
        // Validate workspace
        const auto& ws = get_workspace();
        ws.validate(m_ffa_plan);
        // Initialize BruteFold
        initialize_brute_fold();
        log_info();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan<HostFoldT>& get_plan() const { return m_ffa_plan; }
    [[nodiscard]] plans::FFAPlan<HostFoldT> extract_plan() && noexcept {
        return std::move(m_ffa_plan);
    }

    float get_brute_fold_timing() const noexcept { return m_brutefold_time; }

    void execute_h(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   std::span<HostFoldT> fold) {
        // timing::ScopeTimer timer("FFACUDA::execute_h");
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACUDA::execute_h: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACUDA::execute_h: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_buffer_size(),
            "FFACUDA::execute_h: fold must have size buffer_size");

        // Resize buffers only if needed
        if (m_ts_e_d.size() < ts_e.size()) {
            m_ts_e_d.resize(ts_e.size());
            m_ts_v_d.resize(ts_v.size());
        }
        if (m_fold_d.size() < fold.size()) {
            m_fold_d.resize(fold.size());
        }
        // Copy input data to device
        cudaStream_t stream = nullptr;
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
        // Execute FFA on device using persistent buffers
        execute_d(cuda_utils::as_span(m_ts_e_d), cuda_utils::as_span(m_ts_v_d),
                  cuda_utils::as_span(m_fold_d), stream);

        // Copy result back to host
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(fold.data(),
                            thrust::raw_pointer_cast(m_fold_d.data()),
                            fold.size() * sizeof(DeviceFoldT),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync fold failed");
        // Synchronize stream before returning to host
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
    }

    void execute_h(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   cuda::std::span<DeviceFoldT> fold_d) {
        // timing::ScopeTimer timer("FFACUDA::execute_h");
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACUDA::execute_h: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACUDA::execute_h: ts_v must have size nsamps");
        error_check::check_equal(
            fold_d.size(), m_ffa_plan.get_buffer_size(),
            "FFACUDA::execute_d: fold must have size buffer_size");

        // Resize buffers only if needed
        if (m_ts_e_d.size() < ts_e.size()) {
            m_ts_e_d.resize(ts_e.size());
            m_ts_v_d.resize(ts_v.size());
        }
        // Copy input data to device
        cudaStream_t stream = nullptr;
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
        // Execute FFA on device using persistent buffers
        execute_d(cuda_utils::as_span(m_ts_e_d), cuda_utils::as_span(m_ts_v_d),
                  fold_d, stream);

        // Synchronize stream before returning to host
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
    }

    void execute_d(cuda::std::span<const float> ts_e_d,
                   cuda::std::span<const float> ts_v_d,
                   cuda::std::span<DeviceFoldT> fold_d,
                   cudaStream_t stream) {
        // timing::ScopeTimer timer("FFACUDA::execute_d");
        error_check::check_equal(
            ts_e_d.size(), m_cfg.get_nsamps(),
            "FFACUDA::execute_d: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v_d.size(), ts_e_d.size(),
            "FFACUDA::execute_d: ts_v must have size nsamps");
        error_check::check_equal(
            fold_d.size(), m_ffa_plan.get_buffer_size(),
            "FFACUDA::execute_d: fold must have size buffer_size");

        auto& ws = get_workspace();
        // Resolve the coordinates into the workspace for the FFA plan
        if (m_is_freq_only) {
            ws.resolve_coordinates_freq(m_ffa_plan, stream);
        } else {
            ws.resolve_coordinates(m_ffa_plan, stream);
        }

        // Execute the FFA plan
        execute_unified_device(ts_e_d, ts_v_d, fold_d, stream);
    }

    void execute_h(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   std::span<float> fold)
        requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>)
    {
        // timing::ScopeTimer timer("FFACUDA::execute_h");
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACUDA::execute_h: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACUDA::execute_h: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), 2 * m_ffa_plan.get_buffer_size(),
            "FFACUDA::execute_h: fold must have size 2*buffer_size");

        // Resize buffers only if needed
        if (m_ts_e_d.size() < ts_e.size()) {
            m_ts_e_d.resize(ts_e.size());
            m_ts_v_d.resize(ts_v.size());
        }
        if (m_fold_d_time.size() < fold.size()) {
            m_fold_d_time.resize(fold.size());
        }
        // Copy input data to device
        cudaStream_t stream = nullptr;
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
        // Execute FFA on device using persistent buffers
        execute_d(cuda_utils::as_span(m_ts_e_d), cuda_utils::as_span(m_ts_v_d),
                  cuda_utils::as_span(m_fold_d_time), stream);

        // Copy result back to host
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(
                fold.data(), thrust::raw_pointer_cast(m_fold_d_time.data()),
                fold.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync fold failed");
        // Synchronize stream before returning to host
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "cudaStreamSynchronize failed");
    }

    void execute_d(cuda::std::span<const float> ts_e_d,
                   cuda::std::span<const float> ts_v_d,
                   cuda::std::span<float> fold_d,
                   cudaStream_t stream)
        requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>)
    {
        // timing::ScopeTimer timer("FFACUDA::execute_d");
        const auto fold_size_time      = m_ffa_plan.get_fold_size_time();
        const auto fold_size_fourier   = m_ffa_plan.get_fold_size();
        const auto buffer_size_fourier = m_ffa_plan.get_buffer_size();

        error_check::check_equal(
            ts_e_d.size(), m_cfg.get_nsamps(),
            "FFACUDA::execute_d: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v_d.size(), ts_e_d.size(),
            "FFACUDA::execute_d: ts_v must have size nsamps");
        error_check::check_equal(
            fold_d.size(), 2 * buffer_size_fourier,
            "FFACUDA::execute_d: fold must have size 2*buffer_size_fourier");

        auto& ws = get_workspace();
        // Resolve the coordinates for the FFA plan
        if (m_is_freq_only) {
            ws.resolve_coordinates_freq(m_ffa_plan, stream);
        } else {
            ws.resolve_coordinates(m_ffa_plan, stream);
        }

        auto fold_complex = cuda::std::span<ComplexTypeCUDA>(
            reinterpret_cast<ComplexTypeCUDA*>(fold_d.data()),
            buffer_size_fourier);
        // Execute the FFA plan
        execute_unified_device(ts_e_d, ts_v_d, fold_complex, stream,
                               /*output_in_internal_buffer=*/true);
        // IRFFT
        const auto nfft = fold_size_time / m_cfg.get_nbins();
        math::irfft_batch_cuda(
            cuda_utils::as_span(ws.fold_internal_d).first(fold_size_fourier),
            fold_d.first(fold_size_time), static_cast<int>(nfft),
            static_cast<int>(m_cfg.get_nbins()), stream);
    }

private:
    search::PulsarSearchConfig m_cfg;
    plans::FFAPlan<HostFoldT> m_ffa_plan;
    int m_device_id;
    bool m_is_freq_only;

    // Brute fold for the initial time-domain folding
    std::unique_ptr<BruteFoldCUDA<FoldTypeCUDA>> m_the_bf;
    std::unique_ptr<BruteFoldCUDA<float>> m_the_bf_float; // For lossy init
    bool m_use_lossy_init{false};
    float m_brutefold_time{0.0F};

    // Persistent input/output buffers
    thrust::device_vector<float> m_ts_e_d;
    thrust::device_vector<float> m_ts_v_d;
    thrust::device_vector<DeviceFoldT> m_fold_d;
    thrust::device_vector<float> m_fold_d_time;

    // FFA workspace ownership
    memory::FFAWorkspaceCUDA<FoldTypeCUDA> m_workspace_storage;
    // The observer pointer that always points to the active workspace.
    memory::FFAWorkspaceCUDA<FoldTypeCUDA>* m_workspace_ptr{nullptr};

    [[nodiscard]] memory::FFAWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() noexcept {
        return *m_workspace_ptr;
    }
    [[nodiscard]] const memory::FFAWorkspaceCUDA<FoldTypeCUDA>&
    get_workspace() const noexcept {
        return *m_workspace_ptr;
    }

    void log_info() {
        // Log iniital and final fold shapes
        const auto& fold_shapes = m_ffa_plan.get_fold_shapes();
        spdlog::info("P-FFA [{}] -> [{}]", fmt::join(fold_shapes.front(), ", "),
                     fmt::join(fold_shapes.back(), ", "));
        const auto memory_buffer_gb = m_ffa_plan.get_buffer_memory_usage();
        const auto memory_coord_gb  = m_ffa_plan.get_coord_memory_usage();
        spdlog::info("FFACUDA Memory: {:.2f} GB + {:.2f} GB (coords)",
                     memory_buffer_gb, memory_coord_gb);
    }

    void initialize_brute_fold() {
        const auto t_ref =
            m_is_freq_only ? 0.0 : m_ffa_plan.get_tsegments()[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.compute_param_grid(0).back();

        // Check if we need lossy initialization (ComplexTypeCUDA with large
        // nbins)
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            if (m_cfg.get_nbins() > m_cfg.get_nbins_min_lossy_bf()) {
                m_use_lossy_init = true;
                m_the_bf_float   = std::make_unique<BruteFoldCUDA<float>>(
                    freqs_arr, m_ffa_plan.get_segment_lens()[0],
                    m_cfg.get_nbins(), m_cfg.get_nsamps(), m_cfg.get_tsamp(),
                    t_ref, m_device_id);
                spdlog::debug(
                    "Using lossy initialization (time->freq) for nbins={}",
                    m_cfg.get_nbins());
                return;
            }
        }

        // Normal initialization
        m_the_bf = std::make_unique<BruteFoldCUDA<FoldTypeCUDA>>(
            freqs_arr, m_ffa_plan.get_segment_lens()[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_device_id);
    }

    void initialize_device(cuda::std::span<const float> ts_e_d,
                           cuda::std::span<const float> ts_v_d,
                           DeviceFoldT* init_buffer_d,
                           DeviceFoldT* temp_buffer_d,
                           cudaStream_t stream) {
        timing::SimpleTimer timer;
        timer.start();
        if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
            if (m_use_lossy_init) {
                // Lossy path: use time-domain BruteFold, then RFFT to frequency
                // domain
                const auto brute_fold_size_time =
                    m_the_bf_float->get_fold_size();

                // Use temp_buffer for time-domain output
                // temp_buffer_d is DeviceFoldT*, reinterpret as float* for
                // time-domain data
                auto real_temp_view = cuda::std::span<float>(
                    reinterpret_cast<float*>(temp_buffer_d),
                    brute_fold_size_time);

                m_the_bf_float->execute(ts_e_d, ts_v_d, real_temp_view, stream);

                // Out-of-place RFFT from temp_buffer (real) to init_buffer
                // (complex)
                const auto nfft = brute_fold_size_time / m_cfg.get_nbins();
                const auto brute_fold_size_fourier =
                    nfft * ((m_cfg.get_nbins() / 2) + 1);
                math::rfft_batch_cuda(
                    real_temp_view,
                    cuda::std::span<ComplexTypeCUDA>(init_buffer_d,
                                                     brute_fold_size_fourier),
                    static_cast<int>(nfft), static_cast<int>(m_cfg.get_nbins()),
                    stream);
                m_brutefold_time += timer.stop();
                return;
            }
        }
        // Normal path (float or ComplexType with nbins <= 64)
        m_the_bf->execute(
            ts_e_d, ts_v_d,
            cuda::std::span(init_buffer_d, m_the_bf->get_fold_size()), stream);
        cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                    "brute fold synchronization failed");
        m_brutefold_time += timer.stop();
    }

    void execute_unified_device(cuda::std::span<const float> ts_e_d,
                                cuda::std::span<const float> ts_v_d,
                                cuda::std::span<DeviceFoldT> fold_d,
                                cudaStream_t stream,
                                bool output_in_internal_buffer = false) {
        const auto levels = m_cfg.get_niters_ffa() + 1;
        error_check::check_greater_equal(
            levels, 2,
            "FFACUDA::execute_unified_device: levels must be greater "
            "than or equal to 2");

        auto& ws = get_workspace();
        // Use fold_internal from workspace and output fold for ping-pong
        DeviceFoldT* fold_internal_ptr =
            thrust::raw_pointer_cast(ws.fold_internal_d.data());
        DeviceFoldT* fold_result_ptr = thrust::raw_pointer_cast(fold_d.data());

        DeviceFoldT* current_in_ptr  = nullptr;
        DeviceFoldT* current_out_ptr = nullptr;

        // Number of internal ping-pong iterations (excluding the final write)
        const SizeType internal_iters = levels - 2;
        // Determine starting configuration to ensure final result lands in the
        // correct side of the ping-pong table
        const bool odd_swaps        = (internal_iters % 2) == 1;
        const bool init_in_internal = (odd_swaps == output_in_internal_buffer);
        if (init_in_internal) {
            // init -> internal,
            // odd swaps -> ends in result, even swaps -> ends in internal
            current_in_ptr  = fold_internal_ptr;
            current_out_ptr = fold_result_ptr;
        } else {
            // init -> result,
            // even swaps -> ends in result, odd swaps -> ends in internal
            current_in_ptr  = fold_result_ptr;
            current_out_ptr = fold_internal_ptr;
        }

        // Initialize in the current buffer
        initialize_device(ts_e_d, ts_v_d, current_in_ptr, current_out_ptr,
                          stream);

        // FFA iterations
        if (m_is_freq_only) {
            auto coords_base = ws.coords_freq_d.get_raw_ptrs();
            for (SizeType i_level = 1; i_level < levels; ++i_level) {
                execute_iter_freq(current_in_ptr, current_out_ptr, coords_base,
                                  i_level, stream);
                if (i_level < levels - 1) {
                    std::swap(current_in_ptr, current_out_ptr);
                }
            }
        } else {
            auto coords_base = ws.coords_d.get_raw_ptrs();
            for (SizeType i_level = 1; i_level < levels; ++i_level) {
                execute_iter(current_in_ptr, current_out_ptr, coords_base,
                             i_level, stream);
                if (i_level < levels - 1) {
                    std::swap(current_in_ptr, current_out_ptr);
                }
            }
        }
    }

    void execute_iter_freq(const DeviceFoldT* __restrict__ fold_in,
                           DeviceFoldT* __restrict__ fold_out,
                           coord::FFACoordFreqDPtrs coords_base,
                           SizeType i_level,
                           cudaStream_t stream) {
        const auto nbins        = m_cfg.get_nbins();
        const auto nbins_f      = m_cfg.get_nbins_f();
        const auto nsegments    = m_ffa_plan.get_fold_shapes_time()[i_level][0];
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];
        // Get the coordinates for the current level
        const coord::FFACoordFreqDPtrs coords =
            coords_base.offset(ncoords_offset);

        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            kernels::ffa_iter_freq_cuda(fold_in, fold_out, coords, ncoords_cur,
                                        ncoords_prev, nsegments, nbins, stream);

        } else {
            kernels::ffa_complex_iter_freq_cuda(
                fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
                nbins_f, nbins, stream);
        }
    }

    void execute_iter(const DeviceFoldT* __restrict__ fold_in,
                      DeviceFoldT* __restrict__ fold_out,
                      coord::FFACoordDPtrs coords_base,
                      SizeType i_level,
                      cudaStream_t stream) {
        const auto nbins        = m_cfg.get_nbins();
        const auto nbins_f      = m_cfg.get_nbins_f();
        const auto nsegments    = m_ffa_plan.get_fold_shapes_time()[i_level][0];
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];
        // Get the coordinates for the current level
        const coord::FFACoordDPtrs coords = coords_base.offset(ncoords_offset);

        if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
            kernels::ffa_iter_cuda(fold_in, fold_out, coords, ncoords_cur,
                                   ncoords_prev, nsegments, nbins, stream);

        } else {
            kernels::ffa_complex_iter_cuda(fold_in, fold_out, coords,
                                           ncoords_cur, ncoords_prev, nsegments,
                                           nbins_f, nbins, stream);
        }
    }

}; // End FFACUDA::Impl definition

// --- Definitions for FFACUDA ---
template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFACUDA<FoldTypeCUDA>::FFACUDA(const search::PulsarSearchConfig& cfg,
                               int device_id)
    : m_impl(std::make_unique<Impl>(cfg, device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFACUDA<FoldTypeCUDA>::FFACUDA(
    memory::FFAWorkspaceCUDA<FoldTypeCUDA>& workspace,
    const search::PulsarSearchConfig& cfg,
    int device_id)
    : m_impl(std::make_unique<Impl>(workspace, cfg, device_id)) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFACUDA<FoldTypeCUDA>::~FFACUDA() = default;

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFACUDA<FoldTypeCUDA>::FFACUDA(FFACUDA<FoldTypeCUDA>&& other) noexcept =
    default;

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFACUDA<FoldTypeCUDA>& FFACUDA<FoldTypeCUDA>::operator=(
    FFACUDA<FoldTypeCUDA>&& other) noexcept = default;

template <SupportedFoldTypeCUDA FoldTypeCUDA>
auto FFACUDA<FoldTypeCUDA>::get_plan() const noexcept
    -> const plans::FFAPlan<HostFoldT>& {
    return m_impl->get_plan();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
auto FFACUDA<FoldTypeCUDA>::extract_plan() && noexcept
    -> plans::FFAPlan<HostFoldT> {
    return std::move(*m_impl).extract_plan();
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
float FFACUDA<FoldTypeCUDA>::get_brute_fold_timing() const noexcept {
    return m_impl->get_brute_fold_timing();
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFACUDA<FoldTypeCUDA>::execute(
    std::span<const float> ts_e,
    std::span<const float> ts_v,
    std::span<typename FoldTypeTraits<FoldTypeCUDA>::HostType> fold) {
    m_impl->execute_h(ts_e, ts_v, fold);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFACUDA<FoldTypeCUDA>::execute(
    std::span<const float> ts_e,
    std::span<const float> ts_v,
    cuda::std::span<typename FFACUDA<FoldTypeCUDA>::DeviceFoldT> fold_d) {
    m_impl->execute_h(ts_e, ts_v, fold_d);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFACUDA<FoldTypeCUDA>::execute(
    cuda::std::span<const float> ts_e,
    cuda::std::span<const float> ts_v,
    cuda::std::span<typename FFACUDA<FoldTypeCUDA>::DeviceFoldT> fold,
    cudaStream_t stream) {
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFACUDA<FoldTypeCUDA>::execute(std::span<const float> ts_e,
                                    std::span<const float> ts_v,
                                    std::span<float> fold)
    requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>)
{
    m_impl->execute_h(ts_e, ts_v, fold);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFACUDA<FoldTypeCUDA>::execute(cuda::std::span<const float> ts_e,
                                    cuda::std::span<const float> ts_v,
                                    cuda::std::span<float> fold,
                                    cudaStream_t stream)
    requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>)
{
    m_impl->execute_d(ts_e, ts_v, fold, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::tuple<std::vector<HostFoldType<FoldTypeCUDA>>,
           plans::FFAPlan<HostFoldType<FoldTypeCUDA>>>
compute_ffa_cuda(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const search::PulsarSearchConfig& cfg,
                 int device_id,
                 bool quiet) {
    using HostFoldT = HostFoldType<FoldTypeCUDA>;
    timing::ScopedLogLevel scoped_log_level(quiet);
    FFACUDA<FoldTypeCUDA> ffa(cfg, device_id);
    const plans::FFAPlan<HostFoldT>& ffa_plan = ffa.get_plan();
    const auto buffer_size                    = ffa_plan.get_buffer_size();
    std::vector<HostFoldT> fold(buffer_size, HostFoldT{});
    ffa.execute(ts_e, ts_v, std::span<HostFoldT>(fold));
    // RESIZE to actual result size
    const auto fold_size = ffa_plan.get_fold_size();
    fold.resize(fold_size);
    return {std::move(fold), std::move(ffa).extract_plan()};
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::tuple<thrust::device_vector<FoldTypeCUDA>,
           plans::FFAPlan<HostFoldType<FoldTypeCUDA>>>
compute_ffa_cuda_device(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        const search::PulsarSearchConfig& cfg,
                        int device_id) {
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;
    FFACUDA<FoldTypeCUDA> ffa(cfg, device_id);
    const plans::FFAPlan<HostFoldT>& ffa_plan = ffa.get_plan();
    const auto buffer_size                    = ffa_plan.get_buffer_size();
    thrust::device_vector<DeviceFoldT> fold(buffer_size, DeviceFoldT{});
    ffa.execute(ts_e, ts_v, cuda_utils::as_span(fold));
    // RESIZE to actual result size
    const auto fold_size = ffa_plan.get_fold_size();
    fold.resize(fold_size);
    return {std::move(fold), std::move(ffa).extract_plan()};
}

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_fourier_return_to_time_cuda(std::span<const float> ts_e,
                                        std::span<const float> ts_v,
                                        const search::PulsarSearchConfig& cfg,
                                        int device_id,
                                        bool quiet) {
    timing::ScopedLogLevel scoped_log_level(quiet);
    FFACUDA<ComplexTypeCUDA> ffa(cfg, device_id);
    const plans::FFAPlan<ComplexType>& ffa_plan = ffa.get_plan();
    const auto buffer_size_time = ffa_plan.get_buffer_size_time();
    std::vector<float> fold(buffer_size_time);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    // RESIZE to actual result size
    const auto fold_size_time = ffa_plan.get_fold_size_time();
    fold.resize(fold_size_time);
    // Get the plan for the time domain
    plans::FFAPlan<float> ffa_plan_time(cfg);
    return {std::move(fold), std::move(ffa_plan_time)};
}

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores_cuda(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        const search::PulsarSearchConfig& cfg,
                        int device_id,
                        bool quiet) {
    timing::ScopedLogLevel scoped_log_level(quiet);
    auto [fold, ffa_plan] =
        cfg.get_use_fourier()
            ? compute_ffa_fourier_return_to_time_cuda(ts_e, ts_v, cfg,
                                                      device_id, quiet)
            : compute_ffa_cuda<float>(ts_e, ts_v, cfg, device_id, quiet);
    const auto nsegments = ffa_plan.get_nsegments().back();
    const auto ncoords   = ffa_plan.get_ncoords().back();
    error_check::check_equal(
        nsegments, 1U, "compute_ffa_scores: nsegments must be 1 for scores");
    const auto& score_widths = cfg.get_scoring_widths();
    const auto nscores       = ncoords * score_widths.size();
    std::vector<float> scores(nscores);
    detection::snr_boxcar_3d_cuda(fold, score_widths, scores, ncoords,
                                  cfg.get_nbins(), device_id);
    return {std::move(scores), std::move(ffa_plan)};
}

// Explicit instantiation
template class FFACUDA<float>;
template class FFACUDA<ComplexTypeCUDA>;

template std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_cuda<float>(std::span<const float>,
                        std::span<const float>,
                        const search::PulsarSearchConfig&,
                        int,
                        bool);

template std::tuple<std::vector<ComplexType>, plans::FFAPlan<ComplexType>>
compute_ffa_cuda<ComplexTypeCUDA>(std::span<const float>,
                                  std::span<const float>,
                                  const search::PulsarSearchConfig&,
                                  int,
                                  bool);

template std::tuple<thrust::device_vector<float>, plans::FFAPlan<float>>
compute_ffa_cuda_device<float>(std::span<const float>,
                               std::span<const float>,
                               const search::PulsarSearchConfig&,
                               int);

template std::tuple<thrust::device_vector<ComplexTypeCUDA>,
                    plans::FFAPlan<ComplexType>>
compute_ffa_cuda_device<ComplexTypeCUDA>(std::span<const float>,
                                         std::span<const float>,
                                         const search::PulsarSearchConfig&,
                                         int);

} // namespace loki::algorithms