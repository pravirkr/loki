#include "loki/algorithms/ffa.hpp"

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

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

template <SupportedFoldType FoldType> class FFAWorkspace<FoldType>::Impl {
public:
    Impl() = default;
    explicit Impl(const plans::FFAPlan<FoldType>& ffa_plan) {
        const auto buffer_size = ffa_plan.get_buffer_size();
        m_fold_internal.resize(buffer_size, default_fold_value<FoldType>());
        const bool is_freq_only = ffa_plan.get_n_params() == 1;
        if (is_freq_only) {
            m_coords_freq.resize(ffa_plan.get_coord_size());
        } else {
            m_coords.resize(ffa_plan.get_coord_size());
        }
    }

    explicit Impl(SizeType buffer_size,
                  SizeType coord_size,
                  SizeType n_params) {
        m_fold_internal.resize(buffer_size, default_fold_value<FoldType>());
        const bool is_freq_only = n_params == 1;
        if (is_freq_only) {
            m_coords_freq.resize(coord_size);
        } else {
            m_coords.resize(coord_size);
        }
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    std::vector<FoldType>& get_fold_buffer() noexcept {
        return m_fold_internal;
    }
    std::vector<plans::FFACoord>& get_coords() noexcept { return m_coords; }
    std::vector<plans::FFACoordFreq>& get_coords_freq() noexcept {
        return m_coords_freq;
    }

    void validate(const plans::FFAPlan<FoldType>& ffa_plan) const {
        const auto buffer_size  = ffa_plan.get_buffer_size();
        const bool is_freq_only = ffa_plan.get_n_params() == 1;
        error_check::check_greater_equal(
            m_fold_internal.size(), buffer_size,
            "FFAWorkspace: fold_internal buffer too small");
        if (is_freq_only) {
            error_check::check_greater_equal(
                m_coords_freq.size(), ffa_plan.get_coord_size(),
                "FFAWorkspace: coordinates not allocated for enough levels");
        } else {
            error_check::check_greater_equal(
                m_coords.size(), ffa_plan.get_coord_size(),
                "FFAWorkspace: coordinates not allocated for enough levels");
        }
    }

private:
    std::vector<FoldType> m_fold_internal;
    std::vector<plans::FFACoord> m_coords;
    std::vector<plans::FFACoordFreq> m_coords_freq;
};

template <SupportedFoldType FoldType> class FFA<FoldType>::Impl {
public:
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
        log_memory();
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
        log_memory();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan<FoldType>& get_plan() const { return m_ffa_plan; }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold) {
        timing::ScopeTimer timer("FFA::execute");
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_buffer_size(),
            "FFA::Impl::execute: fold must have size buffer_size");

        auto& ws = get_workspace();
        // Resolve the coordinates into the workspace for the FFA plan
        if (m_is_freq_only) {
            m_ffa_plan.resolve_coordinates_freq(ws.get_coords_freq());
        } else {
            m_ffa_plan.resolve_coordinates(ws.get_coords());
        }

        // Execute the FFA plan
        execute_unified(ts_e, ts_v, fold);
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold)
        requires(std::is_same_v<FoldType, ComplexType>)
    {
        static_assert(std::is_same_v<FoldType, ComplexType>,
                      "This overload is for ComplexType only");
        timing::ScopeTimer timer("FFA::execute");
        const auto fold_size_time      = m_ffa_plan.get_fold_size_time();
        const auto fold_size_fourier   = m_ffa_plan.get_fold_size();
        const auto buffer_size_fourier = m_ffa_plan.get_buffer_size();

        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(fold.size(), 2 * buffer_size_fourier,
                                 "FFACOMPLEX::Impl::execute: fold must have "
                                 "size 2 * buffer_size_fourier");

        auto& ws = get_workspace();
        // Resolve the coordinates for the FFA plan
        if (m_is_freq_only) {
            m_ffa_plan.resolve_coordinates_freq(ws.get_coords_freq());
        } else {
            m_ffa_plan.resolve_coordinates(ws.get_coords());
        }

        auto fold_complex = std::span<ComplexType>(
            reinterpret_cast<ComplexType*>(fold.data()), buffer_size_fourier);
        // Execute the FFA plan
        execute_unified(ts_e, ts_v, fold_complex,
                        /*output_in_internal_buffer=*/true);
        // IRFFT
        const auto nfft = fold_size_time / m_cfg.get_nbins();
        utils::irfft_batch(
            std::span(ws.get_fold_buffer().data(), fold_size_fourier),
            std::span(fold.data(), fold_size_time), static_cast<int>(nfft),
            static_cast<int>(m_cfg.get_nbins()), m_nthreads);
    }

private:
    search::PulsarSearchConfig m_cfg;
    bool m_show_progress;
    plans::FFAPlan<FoldType> m_ffa_plan;
    int m_nthreads;
    bool m_is_freq_only;
    bool m_owns_workspace;
    std::unique_ptr<algorithms::BruteFold<FoldType>> m_the_bf;

    // FFA workspace ownership
    FFAWorkspace<FoldType> m_ffa_workspace_owned;
    FFAWorkspace<FoldType>* m_ffa_workspace_external;

    FFAWorkspace<FoldType>& get_workspace() {
        return m_owns_workspace ? m_ffa_workspace_owned
                                : *m_ffa_workspace_external;
    }

    void log_memory() {
        const auto memory_buffer_gb = m_ffa_plan.get_buffer_memory_usage();
        const auto memory_coord_gb  = m_ffa_plan.get_coord_memory_usage();
        spdlog::info("FFA Memory Usage: {:.2f} GB + {:.2f} GB (coords)",
                     memory_buffer_gb, memory_coord_gb);
    }

    void initialize_brute_fold() {
        const auto t_ref =
            m_is_freq_only ? 0.0 : m_ffa_plan.get_tsegments()[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.get_params()[0].back();
        m_the_bf = std::make_unique<algorithms::BruteFold<FoldType>>(
            freqs_arr, m_ffa_plan.get_segment_lens()[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);
    }

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    FoldType* init_buffer) {
        timing::ScopeTimer timer("FFA::initialize");
        m_the_bf->execute(ts_e, ts_v,
                          std::span(init_buffer, m_the_bf->get_fold_size()));
    }

    void execute_unified(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<FoldType> fold,
                         bool output_in_internal_buffer = false) {
        const auto levels = m_cfg.get_niters_ffa() + 1;
        error_check::check_greater_equal(
            levels, 2,
            "FFA::Impl::execute_unified: levels must be greater than or equal "
            "to 2");

        auto& ws = get_workspace();
        // Use fold_internal from workspace and output fold for ping-pong
        FoldType* fold_internal_ptr = ws.get_fold_buffer().data();
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

        // Initialize in the current buffer
        initialize(ts_e, ts_v, current_in_ptr);

        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(current_in_ptr, current_out_ptr, i_level);
            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                std::swap(current_in_ptr, current_out_ptr);
            }
            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.get_ncoords_lb()[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
    }

    void execute_iter(const FoldType* __restrict__ fold_in,
                      FoldType* __restrict__ fold_out,
                      SizeType i_level) {
        auto& ws                = get_workspace();
        const auto nsegments    = m_ffa_plan.get_fold_shapes()[i_level][0];
        const auto nbins        = m_ffa_plan.get_fold_shapes()[i_level].back();
        const auto ncoords_cur  = m_ffa_plan.get_ncoords()[i_level];
        const auto ncoords_prev = m_ffa_plan.get_ncoords()[i_level - 1];
        const auto ncoords_offset = m_ffa_plan.get_ncoords_offsets()[i_level];

        // Choose strategy based on level characteristics
        if constexpr (std::is_same_v<FoldType, float>) {
            if (m_is_freq_only) {
                const auto coords_cur_span =
                    std::span(ws.get_coords_freq())
                        .subspan(ncoords_offset, ncoords_cur);
                if (nsegments >= 256) {
                    kernels::ffa_iter_segment_freq(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                } else {
                    kernels::ffa_iter_standard_freq(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                }
            } else {
                const auto coords_cur_span =
                    std::span(ws.get_coords())
                        .subspan(ncoords_offset, ncoords_cur);
                if (nsegments >= 256) {
                    kernels::ffa_iter_segment(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                } else {
                    kernels::ffa_iter_standard(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                }
            }
        } else {
            if (m_is_freq_only) {
                const auto coords_cur_span =
                    std::span(ws.get_coords_freq())
                        .subspan(ncoords_offset, ncoords_cur);
                if (nsegments >= 256) {
                    kernels::ffa_complex_iter_segment_freq(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                } else {
                    kernels::ffa_complex_iter_standard_freq(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                }
            } else {
                const auto coords_cur_span =
                    std::span(ws.get_coords())
                        .subspan(ncoords_offset, ncoords_cur);
                if (nsegments >= 256) {
                    kernels::ffa_complex_iter_segment(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                } else {
                    kernels::ffa_complex_iter_standard(
                        fold_in, fold_out, coords_cur_span.data(), nsegments,
                        nbins, ncoords_cur, ncoords_prev, m_nthreads);
                }
            }
        }
    }
}; // End FFA::Impl definition

// --- Definitions for FFAWorkspace ---
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace() : m_impl(std::make_unique<Impl>()) {}
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan)
    : m_impl(std::make_unique<Impl>(ffa_plan)) {}
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(SizeType buffer_size,
                                     SizeType coord_size,
                                     SizeType n_params)
    : m_impl(std::make_unique<Impl>(buffer_size, coord_size, n_params)) {}
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::~FFAWorkspace() = default;
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(FFAWorkspace&& other) noexcept = default;
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>&
FFAWorkspace<FoldType>::operator=(FFAWorkspace&& other) noexcept = default;
template <SupportedFoldType FoldType>
std::vector<FoldType>& FFAWorkspace<FoldType>::get_fold_buffer() noexcept {
    return m_impl->get_fold_buffer();
}
template <SupportedFoldType FoldType>
std::vector<plans::FFACoord>& FFAWorkspace<FoldType>::get_coords() noexcept {
    return m_impl->get_coords();
}
template <SupportedFoldType FoldType>
std::vector<plans::FFACoordFreq>&
FFAWorkspace<FoldType>::get_coords_freq() noexcept {
    return m_impl->get_coords_freq();
}
template <SupportedFoldType FoldType>
void FFAWorkspace<FoldType>::validate(
    const plans::FFAPlan<FoldType>& ffa_plan) const {
    m_impl->validate(ffa_plan);
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
    std::vector<FoldType> fold(buffer_size, default_fold_value<FoldType>());
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
    detection::snr_boxcar_3d(fold, ncoords, score_widths, scores,
                             cfg.get_nthreads());
    return {std::move(scores), ffa_plan};
}

// Explicit instantiation
template class FFAWorkspace<float>;
template class FFAWorkspace<ComplexType>;
template class FFA<float>;
template class FFA<ComplexType>;

template std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::PulsarSearchConfig& cfg,
            bool quiet,
            bool show_progress);
template std::tuple<std::vector<ComplexType>, plans::FFAPlan<ComplexType>>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::PulsarSearchConfig& cfg,
            bool quiet,
            bool show_progress);

} // namespace loki::algorithms