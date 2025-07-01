#include "loki/algorithms/ffa.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include <indicators/cursor_control.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
#include "loki/search/configs.hpp"
#include "loki/timing.hpp"
#include "loki/utils.hpp"
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
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);

        // Log detailed memory usage
        const auto memory_usage = m_ffa_plan.get_memory_usage();
        const auto memory_gb =
            static_cast<float>(memory_usage) / (1024.0F * 1024.0F * 1024.0F);
        spdlog::info("FFA Memory Usage: {:.2f} GB ({} buffers)", memory_gb,
                     m_use_single_buffer ? 1 : 2);
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
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFA::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFA::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_fold_size(),
            "FFA::Impl::execute: fold must have size fold_size");

        ScopeTimer timer("FFA::execute");
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

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    float* init_buffer) {
        ScopeTimer timer("FFA::initialize");
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
        utils::ProgressGuard progress_guard(m_show_progress);
        auto bar = utils::make_standard_bar("Computing FFA...");

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
                const auto progress = static_cast<float>(i_level) /
                                      static_cast<float>(levels - 1) * 100.0F;
                bar.set_progress(static_cast<SizeType>(progress));
            }
        }
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

        utils::ProgressGuard progress_guard(m_show_progress);
        auto bar = utils::make_standard_bar("Computing FFA...");

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(current_in_ptr, current_out_ptr, i_level);
            if (!is_last) {
                std::swap(current_in_ptr, current_out_ptr);
            }
            if (m_show_progress) {
                const auto progress = static_cast<float>(i_level) /
                                      static_cast<float>(levels - 1) * 100.0F;
                bar.set_progress(static_cast<SizeType>(progress));
            }
        }
    }

    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_level) {
        const auto coords_cur   = m_ffa_plan.coordinates[i_level];
        const auto coords_prev  = m_ffa_plan.coordinates[i_level - 1];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto ncoords_cur  = coords_cur.size();
        const auto ncoords_prev = coords_prev.size();

        // Choose strategy based on level characteristics
        if (nsegments >= 256) {
            execute_iter_segment(fold_in, fold_out, coords_cur, nsegments,
                                 nbins, ncoords_cur, ncoords_prev);
        } else {
            execute_iter_standard(fold_in, fold_out, coords_cur, nsegments,
                                  nbins, ncoords_cur, ncoords_prev);
        }
    }

    void execute_iter_segment(const float* __restrict__ fold_in,
                              float* __restrict__ fold_out,
                              const auto& coords_cur,
                              SizeType nsegments,
                              SizeType nbins,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev) {
        // Process one segment at a time to keep data in cache
        constexpr SizeType kBlockSize = 32;

        std::vector<float> temp_buffer(2 * nbins);
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, ncoords_cur,       \
               ncoords_prev) firstprivate(temp_buffer)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            // Process coordinates in blocks within each segment
            for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
                 icoord_block += kBlockSize) {
                SizeType block_end =
                    std::min(icoord_block + kBlockSize, ncoords_cur);
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_tail * 2 * nbins);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_head * 2 * nbins);
                    const auto out_offset =
                        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);
                    const float* __restrict__ fold_tail = &fold_in[tail_offset];
                    const float* __restrict__ fold_head = &fold_in[head_offset];
                    float* __restrict__ fold_sum        = &fold_out[out_offset];
                    float* __restrict__ temp_buffer_ptr = temp_buffer.data();
                    kernels::shift_add_buffer(fold_tail, coord_cur.shift_tail,
                                              fold_head, coord_cur.shift_head,
                                              fold_sum, temp_buffer_ptr, nbins);
                }
            }
        }
    }

    void execute_iter_standard(const float* __restrict__ fold_in,
                               float* __restrict__ fold_out,
                               const auto& coords_cur,
                               SizeType nsegments,
                               SizeType nbins,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev) {
        constexpr SizeType kBlockSize = 32;

        std::vector<float> temp_buffer(2 * nbins);
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, ncoords_cur,       \
               ncoords_prev) firstprivate(temp_buffer)
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_tail * 2 * nbins);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                        (coord_cur.i_head * 2 * nbins);
                    const auto out_offset =
                        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

                    const float* __restrict__ fold_tail = &fold_in[tail_offset];
                    const float* __restrict__ fold_head = &fold_in[head_offset];
                    float* __restrict__ fold_sum        = &fold_out[out_offset];
                    float* __restrict__ temp_buffer_ptr = temp_buffer.data();
                    kernels::shift_add_buffer(fold_tail, coord_cur.shift_tail,
                                              fold_head, coord_cur.shift_head,
                                              fold_sum, temp_buffer_ptr, nbins);
                }
            }
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
        // Initialize the brute fold
        const auto t_ref =
            m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
        const auto freqs_arr = m_ffa_plan.params[0].back();

        m_the_bf = std::make_unique<algorithms::BruteFold>(
            freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
            m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref, m_nthreads);

        // Log detailed memory usage
        const auto memory_usage = m_ffa_plan.get_memory_usage();
        const auto memory_gb =
            static_cast<float>(memory_usage) / (1024.0F * 1024.0F * 1024.0F);
        spdlog::info("FFACOMPLEX Memory Usage: {:.2f} GB ({} buffers)",
                     memory_gb, m_use_single_buffer ? 1 : 2);
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
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(fold.size(),
                                 2 * m_ffa_plan.get_fold_size_complex(),
                                 "FFACOMPLEX::Impl::execute: fold must have "
                                 "size 2 * fold_size_complex");

        ScopeTimer timer("FFACOMPLEX::execute");
        auto fold_complex =
            std::span<ComplexType>(reinterpret_cast<ComplexType*>(fold.data()),
                                   m_ffa_plan.get_fold_size_complex());
        if (m_use_single_buffer) {
            execute_single_buffer(ts_e, ts_v, fold_complex);
        } else {
            execute_double_buffer(ts_e, ts_v, fold_complex);
        }
        // IRFFT in-place
        const auto nfft = m_ffa_plan.get_fold_size() / m_cfg.get_nbins();
        utils::irfft_batch_inplace(fold_complex, static_cast<int>(nfft),
                                   static_cast<int>(m_cfg.get_nbins()),
                                   m_nthreads);
    }

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<ComplexType> fold) {
        error_check::check_equal(
            ts_e.size(), m_cfg.get_nsamps(),
            "FFACOMPLEX::Impl::execute: ts_e must have size nsamps");
        error_check::check_equal(
            ts_v.size(), ts_e.size(),
            "FFACOMPLEX::Impl::execute: ts_v must have size nsamps");
        error_check::check_equal(
            fold.size(), m_ffa_plan.get_fold_size_complex(),
            "FFACOMPLEX::Impl::execute: fold must have size fold_size_complex");

        ScopeTimer timer("FFACOMPLEX::execute");
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
    std::vector<ComplexType> m_fold_in;
    std::vector<ComplexType> m_fold_out;

    void initialize(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    ComplexType* init_buffer) {
        ScopeTimer timer("FFACOMPLEX::initialize");

        const auto required_floats = m_ffa_plan.get_brute_fold_size_complex();

        // Cast complex buffer to float for brute folding
        auto real_view = std::span<float>(reinterpret_cast<float*>(init_buffer),
                                          required_floats);
        m_the_bf->execute_stride(ts_e, ts_v, real_view);

        // In-place RFFT
        const auto nfft         = m_the_bf->get_fold_size() / m_cfg.get_nbins();
        const auto complex_size = nfft * ((m_cfg.get_nbins() / 2) + 1);
        utils::rfft_batch_inplace(
            std::span<ComplexType>(init_buffer, complex_size),
            static_cast<int>(nfft), static_cast<int>(m_cfg.get_nbins()),
            m_nthreads);
    }

    void execute_double_buffer(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<ComplexType> fold_complex) {
        initialize(ts_e, ts_v, m_fold_in.data());

        ComplexType* fold_in_ptr     = m_fold_in.data();
        ComplexType* fold_out_ptr    = m_fold_out.data();
        ComplexType* fold_result_ptr = fold_complex.data();

        const auto levels = m_cfg.get_niters_ffa() + 1;
        utils::ProgressGuard progress_guard(m_show_progress);
        auto bar = utils::make_standard_bar("Computing FFA...");

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
                const auto progress = static_cast<float>(i_level) /
                                      static_cast<float>(levels - 1) * 100.0F;
                bar.set_progress(static_cast<SizeType>(progress));
            }
        }
    }

    void execute_single_buffer(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               std::span<ComplexType> fold_complex) {
        const auto levels = m_cfg.get_niters_ffa() + 1;

        // Use m_fold_in and output fold buffer for ping-pong
        ComplexType* fold_internal_ptr = m_fold_in.data();
        ComplexType* fold_result_ptr   = fold_complex.data();

        // Calculate the number of internal iterations (excluding final write)
        const SizeType internal_iters = levels - 2;

        // Determine starting configuration to ensure final result lands in fold
        bool odd_swaps = (internal_iters % 2) == 1;
        ComplexType *current_in_ptr, *current_out_ptr;
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

        utils::ProgressGuard progress_guard(m_show_progress);
        auto bar = utils::make_standard_bar("Computing FFA...");

        for (SizeType i_level = 1; i_level < levels; ++i_level) {
            const bool is_last = i_level == levels - 1;
            execute_iter(current_in_ptr, current_out_ptr, i_level);
            if (!is_last) {
                std::swap(current_in_ptr, current_out_ptr);
            }
            if (m_show_progress) {
                const auto progress = static_cast<float>(i_level) /
                                      static_cast<float>(levels - 1) * 100.0F;
                bar.set_progress(static_cast<SizeType>(progress));
            }
        }
    }

    void execute_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      SizeType i_level) {
        const auto coords_cur   = m_ffa_plan.coordinates[i_level];
        const auto coords_prev  = m_ffa_plan.coordinates[i_level - 1];
        const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
        const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
        const auto nbins_f      = (nbins / 2) + 1;
        const auto ncoords_cur  = coords_cur.size();
        const auto ncoords_prev = coords_prev.size();

        // Choose strategy based on level characteristics
        if (nsegments >= 256) {
            execute_iter_segment(fold_in, fold_out, coords_cur, nsegments,
                                 nbins_f, nbins, ncoords_cur, ncoords_prev);
        } else {
            execute_iter_standard(fold_in, fold_out, coords_cur, nsegments,
                                  nbins_f, nbins, ncoords_cur, ncoords_prev);
        }
    }

    void execute_iter_segment(const ComplexType* __restrict__ fold_in,
                              ComplexType* __restrict__ fold_out,
                              const auto& coords_cur,
                              SizeType nsegments,
                              SizeType nbins_f,
                              SizeType nbins,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev) {
        // Process one segment at a time to keep data in cache
        constexpr SizeType kBlockSize = 32;
#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, nbins_f,           \
               ncoords_cur, ncoords_prev)
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            // Process coordinates in blocks within each segment
            for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
                 icoord_block += kBlockSize) {
                SizeType block_end =
                    std::min(icoord_block + kBlockSize, ncoords_cur);
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins_f) +
                        (coord_cur.i_tail * 2 * nbins_f);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
                        (coord_cur.i_head * 2 * nbins_f);
                    const auto out_offset = (iseg * ncoords_cur * 2 * nbins_f) +
                                            (icoord * 2 * nbins_f);
                    const ComplexType* __restrict__ fold_tail =
                        &fold_in[tail_offset];
                    const ComplexType* __restrict__ fold_head =
                        &fold_in[head_offset];
                    ComplexType* __restrict__ fold_sum = &fold_out[out_offset];
                    kernels::shift_add_complex_recurrence(
                        fold_tail, coord_cur.shift_tail, fold_head,
                        coord_cur.shift_head, fold_sum, nbins_f, nbins);
                }
            }
        }
    }

    void execute_iter_standard(const ComplexType* __restrict__ fold_in,
                               ComplexType* __restrict__ fold_out,
                               const auto& coords_cur,
                               SizeType nsegments,
                               SizeType nbins_f,
                               SizeType nbins,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev) {
        constexpr SizeType kBlockSize = 32;

#pragma omp parallel for num_threads(m_nthreads) default(none)                 \
    shared(fold_in, fold_out, coords_cur, nsegments, nbins, nbins_f,           \
               ncoords_cur, ncoords_prev)
        for (SizeType icoord_block = 0; icoord_block < ncoords_cur;
             icoord_block += kBlockSize) {
            SizeType block_end =
                std::min(icoord_block + kBlockSize, ncoords_cur);
            for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
                for (SizeType icoord = icoord_block; icoord < block_end;
                     ++icoord) {
                    const auto& coord_cur = coords_cur[icoord];
                    const auto tail_offset =
                        ((iseg * 2) * ncoords_prev * 2 * nbins_f) +
                        (coord_cur.i_tail * 2 * nbins_f);
                    const auto head_offset =
                        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
                        (coord_cur.i_head * 2 * nbins_f);
                    const auto out_offset = (iseg * ncoords_cur * 2 * nbins_f) +
                                            (icoord * 2 * nbins_f);

                    const ComplexType* __restrict__ fold_tail =
                        &fold_in[tail_offset];
                    const ComplexType* __restrict__ fold_head =
                        &fold_in[head_offset];
                    ComplexType* __restrict__ fold_sum = &fold_out[out_offset];
                    kernels::shift_add_complex_recurrence(
                        fold_tail, coord_cur.shift_tail, fold_head,
                        coord_cur.shift_head, fold_sum, nbins_f, nbins);
                }
            }
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
                         std::span<ComplexType> fold) {
    m_impl->execute(ts_e, ts_v, fold);
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
    detection::snr_boxcar_3d(fold, n_param_sets, score_widths, scores, 1.0F,
                             cfg.get_nthreads());
    return {std::move(scores), ffa_plan};
}

} // namespace loki::algorithms