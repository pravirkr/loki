#include "loki/algorithms/ffa.hpp"

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
#include "loki/progress.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"
#include "loki/utils/fft.hpp"

namespace loki::algorithms {

// FFAWorkspace::Data implementation
template <SupportedFoldType FoldType> struct FFAWorkspace<FoldType>::Data {
    std::vector<FoldType> fold_internal;
    std::vector<plans::FFACoord> coords;
    std::vector<plans::FFACoordFreq> coords_freq;

    Data() = default;

    explicit Data(const plans::FFAPlan<FoldType>& ffa_plan) {
        const auto buffer_size  = ffa_plan.get_buffer_size();
        const auto coord_size   = ffa_plan.get_coord_size();
        const bool is_freq_only = ffa_plan.get_n_params() == 1;
        fold_internal.resize(buffer_size, FoldType{});
        if (is_freq_only) {
            coords_freq.resize(coord_size);
        } else {
            coords.resize(coord_size);
        }
    }

    explicit Data(SizeType buffer_size,
                  SizeType coord_size,
                  SizeType n_params) {
        const bool is_freq_only = n_params == 1;
        fold_internal.resize(buffer_size, FoldType{});
        if (is_freq_only) {
            coords_freq.resize(coord_size);
        } else {
            coords.resize(coord_size);
        }
    }

    void validate(const plans::FFAPlan<FoldType>& ffa_plan) const {
        const auto buffer_size  = ffa_plan.get_buffer_size();
        const bool is_freq_only = ffa_plan.get_n_params() == 1;
        error_check::check_greater_equal(
            fold_internal.size(), buffer_size,
            "FFAWorkspace: fold_internal buffer too small");
        if (is_freq_only) {
            error_check::check_greater_equal(
                coords_freq.size(), ffa_plan.get_coord_size(),
                "FFAWorkspace: coordinates not allocated for enough levels");
        } else {
            error_check::check_greater_equal(
                coords.size(), ffa_plan.get_coord_size(),
                "FFAWorkspace: coordinates not allocated for enough levels");
        }
    }
};

// FFA::Impl implementation
template <SupportedFoldType FoldType> class FFA<FoldType>::Impl {
public:
    using WorkspaceData = FFAWorkspace<FoldType>::Data;

    explicit Impl(search::PulsarSearchConfig cfg, bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_owns_workspace(true),
          m_ffa_workspace_owned(m_ffa_plan),
          m_ffa_workspace_external(nullptr) {
        // Validate workspace
        m_ffa_workspace_owned.validate(m_ffa_plan);
        // Initialize BruteFold
        initialize_brute_fold();
        log_info();
    }

    explicit Impl(search::PulsarSearchConfig cfg,
                  FFAWorkspace<FoldType>& workspace,
                  bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_is_freq_only(m_cfg.get_nparams() == 1),
          m_owns_workspace(false),
          m_ffa_workspace_external(&workspace) {
        // Validate workspace
        m_ffa_workspace_external->validate(m_ffa_plan);
        // Initialize BruteFold
        initialize_brute_fold();
        log_info();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan<FoldType>& get_plan() const { return m_ffa_plan; }

    float get_brute_fold_timing() const noexcept { return m_brutefold_time; }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) {
        error_check::check_equal(ts_e.size(), m_cfg.get_nsamps(),
                                 "FFA::execute: ts_e must have size nsamps");
        error_check::check_equal(ts_v.size(), ts_e.size(),
                                 "FFA::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_buffer_size(),
            "FFA::execute: fold must have size buffer_size");

        auto* ws = get_workspace_data();
        // Resolve the coordinates into the workspace for the FFA plan
        if (m_is_freq_only) {
            m_ffa_plan.resolve_coordinates_freq(ws->coords_freq);
        } else {
            m_ffa_plan.resolve_coordinates(ws->coords);
        }

        // Execute the FFA plan
        execute_unified(ts_e, ts_v, fold);
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold)
        requires(std::is_same_v<FoldType, ComplexType>)
    {
        const auto fold_size_time      = m_ffa_plan.get_fold_size_time();
        const auto fold_size_fourier   = m_ffa_plan.get_fold_size();
        const auto buffer_size_fourier = m_ffa_plan.get_buffer_size();

        error_check::check_equal(ts_e.size(), m_cfg.get_nsamps(),
                                 "FFA::execute: ts_e must have size nsamps");
        error_check::check_equal(ts_v.size(), ts_e.size(),
                                 "FFA::execute: ts_v must have size nsamps");
        error_check::check_equal(fold.size(), 2 * buffer_size_fourier,
                                 "FFA::execute: fold must have "
                                 "size 2*buffer_size_fourier");

        auto* ws = get_workspace_data();
        // Resolve the coordinates for the FFA plan
        if (m_is_freq_only) {
            m_ffa_plan.resolve_coordinates_freq(ws->coords_freq);
        } else {
            m_ffa_plan.resolve_coordinates(ws->coords);
        }

        auto fold_complex = std::span<ComplexType>(
            reinterpret_cast<ComplexType*>(fold.data()), buffer_size_fourier);
        // Execute the FFA plan
        execute_unified(ts_e, ts_v, fold_complex,
                        /*output_in_internal_buffer=*/true);
        // IRFFT
        const auto nfft = fold_size_time / m_cfg.get_nbins();
        utils::irfft_batch(
            std::span(ws->fold_internal.data(), fold_size_fourier),
            fold.first(fold_size_time), static_cast<int>(nfft),
            static_cast<int>(m_cfg.get_nbins()), m_nthreads);
    }

private:
    search::PulsarSearchConfig m_cfg;
    bool m_show_progress;
    plans::FFAPlan<FoldType> m_ffa_plan;
    int m_nthreads;
    bool m_is_freq_only;
    bool m_owns_workspace;

    // Brute fold for the initial time-domain folding
    std::unique_ptr<BruteFold<FoldType>> m_the_bf;
    std::unique_ptr<BruteFold<float>> m_the_bf_float; // For lossy init
    bool m_use_lossy_init{false};
    float m_brutefold_time{0.0F};

    // FFA workspace ownership
    FFAWorkspace<FoldType> m_ffa_workspace_owned;
    FFAWorkspace<FoldType>* m_ffa_workspace_external;

    WorkspaceData* get_workspace_data() noexcept {
        return m_owns_workspace ? m_ffa_workspace_owned.data()
                                : m_ffa_workspace_external->data();
    }

    void log_info() {
        // Log iniital and final fold shapes
        const auto& fold_shapes = m_ffa_plan.get_fold_shapes();
        spdlog::info("P-FFA [{}] -> [{}]", fmt::join(fold_shapes.front(), ", "),
                     fmt::join(fold_shapes.back(), ", "));
        //  Log memory usage
        const auto memory_buffer_gb = m_ffa_plan.get_buffer_memory_usage();
        const auto memory_coord_gb  = m_ffa_plan.get_coord_memory_usage();
        spdlog::info("FFA Memory Usage: {:.2f} GB + {:.2f} GB (coords)",
                     memory_buffer_gb, memory_coord_gb);
    }

    void initialize_brute_fold() {
        const auto t_ref =
            m_is_freq_only ? 0.0 : m_ffa_plan.get_tsegments()[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.get_params()[0].back();

        // Check if we need lossy initialization (ComplexType with large nbins)
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            if (m_cfg.get_nbins() > m_cfg.get_nbins_min_lossy_bf()) {
                m_use_lossy_init = true;
                m_the_bf_float   = std::make_unique<BruteFold<float>>(
                    freqs_arr, m_ffa_plan.get_segment_lens()[0],
                    m_cfg.get_nbins(), m_cfg.get_nsamps(), m_cfg.get_tsamp(),
                    t_ref, m_nthreads);
                spdlog::debug(
                    "Using lossy initialization (time->freq) for nbins={}",
                    m_cfg.get_nbins());
                return;
            }
        }

        // Normal initialization
        m_the_bf = std::make_unique<BruteFold<FoldType>>(
            freqs_arr, m_ffa_plan.get_segment_lens()[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);
    }

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    FoldType* init_buffer,
                    FoldType* temp_buffer) {
        timing::SimpleTimer timer;
        timer.start();
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            if (m_use_lossy_init) {
                // Lossy path: use time-domain BruteFold, then RFFT to frequency
                // domain
                const auto brute_fold_size_time =
                    m_the_bf_float->get_fold_size();

                // Use temp_buffer for time-domain output
                // temp_buffer is ComplexType*, reinterpret as float* for
                // time-domain data
                auto real_temp_view =
                    std::span<float>(reinterpret_cast<float*>(temp_buffer),
                                     brute_fold_size_time);
                m_the_bf_float->execute(ts_e, ts_v, real_temp_view);

                // Out-of-place RFFT from temp_buffer (real) to init_buffer
                // (complex)
                const auto nfft = brute_fold_size_time / m_cfg.get_nbins();
                const auto brute_fold_size_fourier =
                    nfft * ((m_cfg.get_nbins() / 2) + 1);
                utils::rfft_batch(real_temp_view,
                                  std::span<ComplexType>(
                                      init_buffer, brute_fold_size_fourier),
                                  static_cast<int>(nfft),
                                  static_cast<int>(m_cfg.get_nbins()),
                                  m_nthreads);
                m_brutefold_time += timer.stop();
                return;
            }
        }
        // Normal path (float or ComplexType with nbins <= 64)
        m_the_bf->execute(ts_e, ts_v,
                          std::span(init_buffer, m_the_bf->get_fold_size()));
        m_brutefold_time += timer.stop();
    }

    void execute_unified(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<FoldType> fold,
                         bool output_in_internal_buffer = false) {
        const auto levels = m_cfg.get_niters_ffa() + 1;
        error_check::check_greater_equal(
            levels, 2,
            "FFA::execute_unified: levels must be greater than or equal "
            "to 2");

        auto* ws = get_workspace_data();
        // Use fold_internal from workspace and output fold for ping-pong
        FoldType* fold_internal_ptr = ws->fold_internal.data();
        FoldType* fold_result_ptr   = fold.data();

        FoldType* current_in_ptr  = nullptr;
        FoldType* current_out_ptr = nullptr;

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

        // Initialize in the current buffer (using optional temp buffer)
        initialize(ts_e, ts_v, current_in_ptr, current_out_ptr);

        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        if (m_is_freq_only) {
            for (SizeType i_level = 1; i_level < levels; ++i_level) {
                execute_iter_freq(current_in_ptr, current_out_ptr, i_level);
                // Ping-pong buffers (unless it's the final iteration)
                if (i_level < levels - 1) {
                    std::swap(current_in_ptr, current_out_ptr);
                }
                if (m_show_progress) {
                    bar->set_leaves(m_ffa_plan.get_ncoords_lb()[i_level]);
                    bar->set_progress(i_level);
                }
            }
        } else {
            for (SizeType i_level = 1; i_level < levels; ++i_level) {
                execute_iter(current_in_ptr, current_out_ptr, i_level);
                // Ping-pong buffers (unless it's the final iteration)
                if (i_level < levels - 1) {
                    std::swap(current_in_ptr, current_out_ptr);
                }
                if (m_show_progress) {
                    bar->set_leaves(m_ffa_plan.get_ncoords_lb()[i_level]);
                    bar->set_progress(i_level);
                }
            }
        }
        bar->mark_as_completed();
    }

    void execute_iter_freq(const FoldType* __restrict__ fold_in,
                           FoldType* __restrict__ fold_out,
                           SizeType i_level) {
        const auto nbins        = m_cfg.get_nbins();
        const auto nbins_f      = m_cfg.get_nbins_f();
        const auto nsegments    = m_ffa_plan.get_fold_shapes_time()[i_level][0];
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];
        // Get the coordinates for the current level
        auto* ws = get_workspace_data();
        const auto coords_cur_span =
            std::span(ws->coords_freq).subspan(ncoords_offset, ncoords_cur);

        if constexpr (std::is_same_v<FoldType, float>) {
            kernels::ffa_iter_freq(fold_in, fold_out, coords_cur_span.data(),
                                   ncoords_cur, ncoords_prev, nsegments, nbins,
                                   m_nthreads);
        } else {
            kernels::ffa_complex_iter_freq(
                fold_in, fold_out, coords_cur_span.data(), ncoords_cur,
                ncoords_prev, nsegments, nbins_f, nbins, m_nthreads);
        }
    }

    void execute_iter(const FoldType* __restrict__ fold_in,
                      FoldType* __restrict__ fold_out,
                      SizeType i_level) {
        const auto nbins        = m_cfg.get_nbins();
        const auto nbins_f      = m_cfg.get_nbins_f();
        const auto nsegments    = m_ffa_plan.get_fold_shapes_time()[i_level][0];
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];
        // Get the coordinates for the current level
        auto* ws = get_workspace_data();
        const auto coords_cur_span =
            std::span(ws->coords).subspan(ncoords_offset, ncoords_cur);

        if constexpr (std::is_same_v<FoldType, float>) {
            kernels::ffa_iter(fold_in, fold_out, coords_cur_span.data(),
                              ncoords_cur, ncoords_prev, nsegments, nbins,
                              m_nthreads);

        } else {
            kernels::ffa_complex_iter(fold_in, fold_out, coords_cur_span.data(),
                                      ncoords_cur, ncoords_prev, nsegments,
                                      nbins_f, nbins, m_nthreads);
        }
    }
}; // End FFA::Impl definition

// --- Definitions for FFAWorkspace ---
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace() : m_data(std::make_unique<Data>()) {}
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan)
    : m_data(std::make_unique<Data>(ffa_plan)) {}
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(SizeType buffer_size,
                                     SizeType coord_size,
                                     SizeType n_params)
    : m_data(std::make_unique<Data>(buffer_size, coord_size, n_params)) {}
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::~FFAWorkspace() = default;
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(FFAWorkspace&& other) noexcept = default;
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>&
FFAWorkspace<FoldType>::operator=(FFAWorkspace&& other) noexcept = default;
template <SupportedFoldType FoldType>
void FFAWorkspace<FoldType>::validate(
    const plans::FFAPlan<FoldType>& ffa_plan) const {
    m_data->validate(ffa_plan);
}

// --- Definitions for FFA ---
template <SupportedFoldType FoldType>
FFA<FoldType>::FFA(const search::PulsarSearchConfig& cfg, bool show_progress)
    : m_impl(std::make_unique<Impl>(cfg, show_progress)) {}
template <SupportedFoldType FoldType>
FFA<FoldType>::FFA(const search::PulsarSearchConfig& cfg,
                   FFAWorkspace<FoldType>& workspace,
                   bool show_progress)
    : m_impl(std::make_unique<Impl>(cfg, workspace, show_progress)) {}
template <SupportedFoldType FoldType> FFA<FoldType>::~FFA() = default;
template <SupportedFoldType FoldType>
FFA<FoldType>::FFA(FFA&& other) noexcept = default;
template <SupportedFoldType FoldType>
FFA<FoldType>& FFA<FoldType>::operator=(FFA&& other) noexcept = default;

template <SupportedFoldType FoldType>
const plans::FFAPlan<FoldType>& FFA<FoldType>::get_plan() const noexcept {
    return m_impl->get_plan();
}
template <SupportedFoldType FoldType>
float FFA<FoldType>::get_brute_fold_timing() const noexcept {
    return m_impl->get_brute_fold_timing();
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            std::span<FoldType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}
template <SupportedFoldType FoldType>
void FFA<FoldType>::execute(std::span<const float> ts_e,
                            std::span<const float> ts_v,
                            std::span<float> fold)
    requires(std::is_same_v<FoldType, ComplexType>)
{
    m_impl->execute(ts_e, ts_v, fold);
}

template <SupportedFoldType FoldType>
std::tuple<std::vector<FoldType>, plans::FFAPlan<FoldType>>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::PulsarSearchConfig& cfg,
            bool quiet,
            bool show_progress) {
    timing::ScopedLogLevel scoped_log_level(quiet);
    FFA<FoldType> ffa(cfg, show_progress);
    const plans::FFAPlan<FoldType>& ffa_plan = ffa.get_plan();
    const auto buffer_size                   = ffa_plan.get_buffer_size();
    std::vector<FoldType> fold(buffer_size, FoldType{});
    ffa.execute(ts_e, ts_v, std::span<FoldType>(fold));
    // RESIZE to actual result size
    const auto fold_size = ffa_plan.get_fold_size();
    fold.resize(fold_size);
    return {std::move(fold), ffa_plan};
}

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_fourier_return_to_time(std::span<const float> ts_e,
                                   std::span<const float> ts_v,
                                   const search::PulsarSearchConfig& cfg,
                                   bool quiet,
                                   bool show_progress) {
    timing::ScopedLogLevel scoped_log_level(quiet);
    FFA<ComplexType> ffa(cfg, show_progress);
    const plans::FFAPlan<ComplexType>& ffa_plan = ffa.get_plan();
    const auto buffer_size_time = ffa_plan.get_buffer_size_time();
    std::vector<float> fold(buffer_size_time);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    // RESIZE to actual result size
    const auto fold_size_time = ffa_plan.get_fold_size_time();
    fold.resize(fold_size_time);
    // Get the plan for the time domain
    plans::FFAPlan<float> ffa_plan_time(cfg);
    return {std::move(fold), ffa_plan_time};
}

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   const search::PulsarSearchConfig& cfg,
                   bool quiet,
                   bool show_progress) {
    timing::ScopedLogLevel scoped_log_level(quiet);
    const auto [fold, ffa_plan] =
        cfg.get_use_fourier()
            ? compute_ffa_fourier_return_to_time(ts_e, ts_v, cfg, quiet,
                                                 show_progress)
            : compute_ffa<float>(ts_e, ts_v, cfg, quiet, show_progress);
    const auto nsegments = ffa_plan.get_nsegments().back();
    const auto ncoords   = ffa_plan.get_ncoords().back();
    error_check::check_equal(
        nsegments, 1U, "compute_ffa_scores: nsegments must be 1 for scores");
    const auto& score_widths = cfg.get_scoring_widths();
    const auto nscores       = ncoords * score_widths.size();
    std::vector<float> scores(nscores);
    detection::snr_boxcar_3d(fold, score_widths, scores, ncoords,
                             cfg.get_nbins(), cfg.get_nthreads());
    return {std::move(scores), ffa_plan};
}

// Explicit instantiation
template class FFAWorkspace<float>;
template class FFAWorkspace<ComplexType>;
template class FFA<float>;
template class FFA<ComplexType>;

template std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa(std::span<const float>,
            std::span<const float>,
            const search::PulsarSearchConfig&,
            bool,
            bool);
template std::tuple<std::vector<ComplexType>, plans::FFAPlan<ComplexType>>
compute_ffa(std::span<const float>,
            std::span<const float>,
            const search::PulsarSearchConfig&,
            bool,
            bool);

} // namespace loki::algorithms