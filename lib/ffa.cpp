#include "loki/algorithms/ffa.hpp"

#include <algorithm>
#include <memory>
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

class FFA::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg, bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_use_single_buffer(m_ffa_plan.get_fold_size() >=
                              m_ffa_plan.get_buffer_size()) {
        // Allocate memory for the FFA buffers
        m_fold_in.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        if (!m_use_single_buffer) {
            m_fold_out.resize(m_ffa_plan.get_buffer_size(), 0.0F);
        }
        // Allocate memory for the FFA coordinates
        m_coordinates.resize(m_ffa_plan.n_levels);
        for (SizeType i_level = 0; i_level < m_ffa_plan.n_levels; ++i_level) {
            m_coordinates[i_level].resize(m_ffa_plan.ncoords[i_level]);
        }
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);

        // Log detailed memory usage
        const auto memory_buffer_gb = m_ffa_plan.get_buffer_memory_usage();
        const auto memory_coord_gb  = m_ffa_plan.get_coord_memory_usage();
        spdlog::info("FFA Memory Usage: {:.2f} GB ({} buffers) + {:.2f} GB "
                     "(coords)",
                     memory_buffer_gb, m_use_single_buffer ? 1 : 2,
                     memory_coord_gb);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan& get_plan() const { return m_ffa_plan; }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold) {
        timing::ScopeTimer timer("FFA::execute");
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_fold_size(),
            "FFA::Impl::execute: fold must have size fold_size");

        // Resolve the coordinates for the FFA plan
        m_ffa_plan.resolve_coordinates(m_coordinates);
        // Execute the FFA plan
        if (m_use_single_buffer) {
            execute_single_buffer(ts_e, ts_v, fold);
        } else {
            execute_double_buffer(ts_e, ts_v, fold);
        }
    }

private:
    search::PulsarSearchConfig m_cfg;
    bool m_show_progress;
    plans::FFAPlan m_ffa_plan;
    int m_nthreads;
    bool m_use_single_buffer;
    std::unique_ptr<algorithms::BruteFold> m_the_bf;

    // Buffers for the FFA plan
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;
    std::vector<std::vector<plans::FFACoord>> m_coordinates;

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    float* init_buffer) {
        timing::ScopeTimer timer("FFA::initialize");
        m_the_bf->execute(ts_e, ts_v,
                          std::span(init_buffer, m_the_bf->get_fold_size()));
    }

    void execute_double_buffer(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<float> fold) {
        initialize(ts_e, ts_v, m_fold_in.data());

        float* fold_in_ptr     = m_fold_in.data();
        float* fold_out_ptr    = m_fold_out.data();
        float* fold_result_ptr = fold.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;
        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            // Determine output buffer: final iteration writes to output buffer
            const bool is_last     = i_level == levels - 1;
            float* current_out_ptr = is_last ? fold_result_ptr : fold_out_ptr;
            execute_iter(fold_in_ptr, current_out_ptr, i_level);
            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                std::swap(fold_in_ptr, fold_out_ptr);
            }
            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.ncoords_lb[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
    }

    void execute_single_buffer(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<float> fold) {
        const auto levels = m_cfg.get_niters_ffa() + 1;

        // Use m_fold_in and output fold buffer for ping-pong
        float* fold_internal_ptr = m_fold_in.data();
        float* fold_result_ptr   = fold.data();

        // Calculate the number of internal iterations (excluding final write)
        const SizeType internal_iters = levels - 2;

        // Determine starting configuration to ensure final result lands in fold
        bool odd_swaps = (internal_iters % 2) == 1;
        float *current_in_ptr, *current_out_ptr;
        if (odd_swaps) {
            // Initialize in fold, will end up in fold after odd swaps
            initialize(ts_e, ts_v, fold_result_ptr);
            current_in_ptr  = fold_result_ptr;
            current_out_ptr = fold_internal_ptr;
        } else {
            // Initialize in internal, will end up in fold after even swaps
            initialize(ts_e, ts_v, fold_internal_ptr);
            current_in_ptr  = fold_internal_ptr;
            current_out_ptr = fold_result_ptr;
        }

        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(current_in_ptr, current_out_ptr, i_level);
            if (!is_last) {
                std::swap(current_in_ptr, current_out_ptr);
            }
            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.ncoords_lb[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
    }

    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_level) {
        const auto coords_cur   = m_coordinates[i_level];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto ncoords_cur  = m_ffa_plan.ncoords[i_level];
        const auto ncoords_prev = m_ffa_plan.ncoords[i_level - 1];

        // Choose strategy based on level characteristics
        if (nsegments >= 256) {
            kernels::ffa_iter_segment(fold_in, fold_out, coords_cur.data(),
                                      nsegments, nbins, ncoords_cur,
                                      ncoords_prev, m_nthreads);
        } else {
            kernels::ffa_iter_standard(fold_in, fold_out, coords_cur.data(),
                                       nsegments, nbins, ncoords_cur,
                                       ncoords_prev, m_nthreads);
        }
    }
}; // End FFA::Impl definition

class FFACOMPLEX::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg, bool show_progress)
        : m_cfg(std::move(cfg)),
          m_show_progress(show_progress),
          m_ffa_plan(m_cfg),
          m_nthreads(m_cfg.get_nthreads()),
          m_use_single_buffer(m_ffa_plan.get_fold_size_complex() >=
                              m_ffa_plan.get_buffer_size_complex()) {
        // Allocate memory for the FFA buffers
        m_fold_in.resize(m_ffa_plan.get_buffer_size_complex(),
                         ComplexType(0.0F, 0.0F));
        if (!m_use_single_buffer) {
            m_fold_out.resize(m_ffa_plan.get_buffer_size_complex(),
                              ComplexType(0.0F, 0.0F));
        }
        // Allocate memory for the FFA coordinates
        m_coordinates.resize(m_ffa_plan.n_levels);
        for (SizeType i_level = 0; i_level < m_ffa_plan.n_levels; ++i_level) {
            m_coordinates[i_level].resize(m_ffa_plan.ncoords[i_level]);
        }
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);

        // Log detailed memory usage
        const auto memory_buffer_gb = m_ffa_plan.get_buffer_memory_usage();
        const auto memory_coord_gb  = m_ffa_plan.get_coord_memory_usage();
        spdlog::info(
            "FFACOMPLEX Memory Usage: {:.2f} GB ({} buffers) + {:.2f} GB "
            "(coords)",
            memory_buffer_gb, m_use_single_buffer ? 1 : 2, memory_coord_gb);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FFAPlan& get_plan() const { return m_ffa_plan; }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold) {
        timing::ScopeTimer timer("FFACOMPLEX::execute");
        const auto fold_size         = m_ffa_plan.get_fold_size();
        const auto fold_size_complex = m_ffa_plan.get_fold_size_complex();

        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(fold.size(), 2 * fold_size_complex,
                                 "FFACOMPLEX::Impl::execute: fold must have "
                                 "size 2 * fold_size_complex");

        // Resolve the coordinates for the FFA plan
        m_ffa_plan.resolve_coordinates(m_coordinates);

        const auto nfft = fold_size / m_cfg.get_nbins();
        if (m_use_single_buffer) {
            auto fold_complex = std::span<ComplexType>(
                reinterpret_cast<ComplexType*>(fold.data()),
                m_ffa_plan.get_fold_size_complex());
            execute_single_buffer(ts_e, ts_v, fold_complex, true);
            // IRFFT
            utils::irfft_batch(std::span(m_fold_in.data(), fold_size_complex),
                               std::span(fold.data(), fold_size),
                               static_cast<int>(nfft),
                               static_cast<int>(m_cfg.get_nbins()), m_nthreads);
        } else {
            ComplexType* result_ptr = execute_double_buffer(ts_e, ts_v);
            // IRFFT
            utils::irfft_batch(std::span(result_ptr, fold_size_complex),
                               std::span(fold.data(), fold_size),
                               static_cast<int>(nfft),
                               static_cast<int>(m_cfg.get_nbins()), m_nthreads);
        }
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<ComplexType> fold_complex) {
        timing::ScopeTimer timer("FFACOMPLEX::execute");
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold_complex.size(), m_ffa_plan.get_fold_size_complex(),
            "FFACOMPLEX::Impl::execute: fold must have size fold_size_complex");

        // Resolve the coordinates for the FFA plan
        m_ffa_plan.resolve_coordinates(m_coordinates);

        if (m_use_single_buffer) {
            execute_single_buffer(ts_e, ts_v, fold_complex, false);
        } else {
            execute_double_buffer(ts_e, ts_v, fold_complex);
        }
    }

private:
    search::PulsarSearchConfig m_cfg;
    bool m_show_progress;
    plans::FFAPlan m_ffa_plan;
    int m_nthreads;
    bool m_use_single_buffer;
    std::unique_ptr<algorithms::BruteFold> m_the_bf;

    // Buffers for the FFA plan
    std::vector<ComplexType> m_fold_in;
    std::vector<ComplexType> m_fold_out;
    std::vector<std::vector<plans::FFACoord>> m_coordinates;

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    ComplexType* init_buffer,
                    ComplexType* temp_buffer) {
        timing::ScopeTimer timer("FFACOMPLEX::initialize");
        // Use temp_buffer for the initial real-valued brute fold output
        auto real_temp_view =
            std::span<float>(reinterpret_cast<float*>(temp_buffer),
                             m_ffa_plan.get_brute_fold_size());
        m_the_bf->execute(ts_e, ts_v, real_temp_view);

        // Out-of-place RFFT from temp_buffer (real) to init_buffer (complex)
        const auto nfft         = m_the_bf->get_fold_size() / m_cfg.get_nbins();
        const auto complex_size = nfft * ((m_cfg.get_nbins() / 2) + 1);
        utils::rfft_batch(real_temp_view,
                          std::span<ComplexType>(init_buffer, complex_size),
                          static_cast<int>(nfft),
                          static_cast<int>(m_cfg.get_nbins()), m_nthreads);
    }

    void execute_double_buffer(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<ComplexType> fold_complex) {
        initialize(ts_e, ts_v, m_fold_in.data(), m_fold_out.data());

        ComplexType* fold_in_ptr     = m_fold_in.data();
        ComplexType* fold_out_ptr    = m_fold_out.data();
        ComplexType* fold_result_ptr = fold_complex.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;
        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            // Determine output buffer: final iteration writes to output buffer
            const bool is_last = i_level == levels - 1;
            ComplexType* current_out_ptr =
                is_last ? fold_result_ptr : fold_out_ptr;
            execute_iter(fold_in_ptr, current_out_ptr, i_level);
            // Ping-pong buffers (unless it's the final iteration)
            if (!is_last) {
                std::swap(fold_in_ptr, fold_out_ptr);
            }
            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.ncoords_lb[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
    }

    // This is a special case for the double buffer implementation where we
    // want to return the buffer that was the output of the last iteration.
    // This is used for the IRFFT.
    ComplexType* execute_double_buffer(std::span<const float> ts_e,
                                       std::span<const float> ts_v) {
        initialize(ts_e, ts_v, m_fold_in.data(), m_fold_out.data());

        ComplexType* fold_in_ptr  = m_fold_in.data();
        ComplexType* fold_out_ptr = m_fold_out.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;
        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            // Always ping-pong between internal buffers
            execute_iter(fold_in_ptr, fold_out_ptr, i_level);
            std::swap(fold_in_ptr, fold_out_ptr);

            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.ncoords_lb[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
        return fold_in_ptr;
    }

    void execute_single_buffer(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<ComplexType> fold_complex,
                               bool output_in_internal_buffer) {
        const auto levels = m_cfg.get_niters_ffa() + 1;

        // Use m_fold_in and output fold buffer for ping-pong
        ComplexType* fold_internal_ptr = m_fold_in.data();
        ComplexType* fold_result_ptr   = fold_complex.data();

        // Calculate the number of internal iterations (excluding final write)
        const SizeType internal_iters = levels - 2;

        // Determine starting configuration to ensure final result lands in the
        // correct side of the ping-pong table
        bool odd_swaps = (internal_iters % 2) == 1;
        const bool init_in_internal =
            (odd_swaps ^ output_in_internal_buffer) == 0;
        ComplexType *current_in_ptr, *current_out_ptr;
        if (init_in_internal) {
            // Initialize in internal, will end up in the desired buffer
            current_in_ptr  = fold_internal_ptr;
            current_out_ptr = fold_result_ptr;
        } else {
            // Initialize in fold, will end up in the desired buffer
            current_in_ptr  = fold_result_ptr;
            current_out_ptr = fold_internal_ptr;
        }

        // Initialize the current buffer
        initialize(ts_e, ts_v, current_in_ptr, current_out_ptr);

        progress::ProgressGuard progress_guard(m_show_progress);
        auto bar = progress::make_ffa_bar("Computing FFA", levels - 1);

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(current_in_ptr, current_out_ptr, i_level);
            if (!is_last) {
                std::swap(current_in_ptr, current_out_ptr);
            }
            if (m_show_progress) {
                bar->set_leaves(m_ffa_plan.ncoords_lb[i_level]);
                bar->set_progress(i_level);
            }
        }
        bar->mark_as_completed();
    }

    void execute_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      SizeType i_level) {
        const auto coords_cur   = m_coordinates[i_level];
        const auto coords_prev  = m_coordinates[i_level - 1];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto nbins_f      = (nbins / 2) + 1;
        const auto ncoords_cur  = m_ffa_plan.ncoords[i_level];
        const auto ncoords_prev = m_ffa_plan.ncoords[i_level - 1];

        // Choose strategy based on level characteristics
        if (nsegments >= 256) {
            kernels::ffa_complex_iter_segment(
                fold_in, fold_out, coords_cur.data(), nsegments, nbins_f, nbins,
                ncoords_cur, ncoords_prev, m_nthreads);
        } else {
            kernels::ffa_complex_iter_standard(
                fold_in, fold_out, coords_cur.data(), nsegments, nbins_f, nbins,
                ncoords_cur, ncoords_prev, m_nthreads);
        }
    }
}; // End FFACOMPLEX::Impl definition

FFA::FFA(const search::PulsarSearchConfig& cfg, bool show_progress)
    : m_impl(std::make_unique<Impl>(cfg, show_progress)) {}
FFA::~FFA()                               = default;
FFA::FFA(FFA&& other) noexcept            = default;
FFA& FFA::operator=(FFA&& other) noexcept = default;
const plans::FFAPlan& FFA::get_plan() const noexcept {
    return m_impl->get_plan();
}
void FFA::execute(std::span<const float> ts_e,
                  std::span<const float> ts_v,
                  std::span<float> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}

FFACOMPLEX::FFACOMPLEX(const search::PulsarSearchConfig& cfg,
                       bool show_progress)
    : m_impl(std::make_unique<Impl>(cfg, show_progress)) {}
FFACOMPLEX::~FFACOMPLEX()                                      = default;
FFACOMPLEX::FFACOMPLEX(FFACOMPLEX&& other) noexcept            = default;
FFACOMPLEX& FFACOMPLEX::operator=(FFACOMPLEX&& other) noexcept = default;
const plans::FFAPlan& FFACOMPLEX::get_plan() const noexcept {
    return m_impl->get_plan();
}
void FFACOMPLEX::execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<float> fold) {
    m_impl->execute(ts_e, ts_v, fold);
}
void FFACOMPLEX::execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         std::span<ComplexType> fold_complex) {
    m_impl->execute(ts_e, ts_v, fold_complex);
}

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::PulsarSearchConfig& cfg,
            bool show_progress) {
    FFA ffa(cfg, show_progress);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return {std::move(fold), ffa_plan};
}

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_complex(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    const search::PulsarSearchConfig& cfg,
                    bool show_progress) {
    FFACOMPLEX ffa(cfg, show_progress);
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(2 * ffa_plan.get_fold_size_complex(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    // RESIZE to actual result size
    fold.resize(ffa_plan.get_fold_size());
    return {std::move(fold), ffa_plan};
}

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_scores(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   const search::PulsarSearchConfig& cfg,
                   bool show_progress) {
    const auto [fold, ffa_plan] =
        cfg.get_use_fft_shifts()
            ? compute_ffa_complex(ts_e, ts_v, cfg, show_progress)
            : compute_ffa(ts_e, ts_v, cfg, show_progress);
    const auto nsegments    = ffa_plan.nsegments.back();
    const auto n_param_sets = ffa_plan.ncoords.back();
    error_check::check_equal(
        nsegments, 1U, "compute_ffa_scores: nsegments must be 1 for scores");
    const auto& score_widths = cfg.get_score_widths();
    const auto nscores       = n_param_sets * score_widths.size();
    std::vector<float> scores(nscores);
    detection::snr_boxcar_3d(fold, n_param_sets, score_widths, scores,
                             cfg.get_nthreads());
    return {std::move(scores), ffa_plan};
}

} // namespace loki::algorithms