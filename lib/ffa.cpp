#include "loki/algorithms/ffa.hpp"

#include <chrono>
#include <format>
#include <numeric>
#include <ranges>
#include <utility>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>
#include <xsimd/xsimd.hpp>

#include "loki/algorithms/fold.hpp"
#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/psr_utils.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

FFAPlan::FFAPlan(search::PulsarSearchConfig cfg) : m_cfg(std::move(cfg)) {
    configure_plan();
}

SizeType FFAPlan::get_buffer_size() const noexcept {
    return std::ranges::max(
        fold_shapes | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<>());
        }));
}

SizeType FFAPlan::get_memory_usage() const noexcept {
    return get_buffer_size() * sizeof(float);
}

SizeType FFAPlan::get_fold_size() const noexcept {
    return std::accumulate(fold_shapes.back().begin(), fold_shapes.back().end(),
                           1, std::multiplies<>());
}

void FFAPlan::configure_plan() {
    const auto levels = m_cfg.get_niters_ffa() + 1;
    segment_lens.resize(levels);
    tsegments.resize(levels);
    params.resize(levels);
    dparams.resize(levels);
    fold_shapes.resize(levels);
    coordinates.resize(levels);

    for (SizeType i_level = 0; i_level < levels; ++i_level) {
        const auto segment_len = m_cfg.get_bseg_brute() * (1U << i_level);
        const auto tsegment =
            static_cast<double>(segment_len) * m_cfg.get_tsamp();
        const auto nsegments  = m_cfg.get_nsamps() / segment_len;
        const auto dparam_arr = m_cfg.get_dparams(tsegment);

        segment_lens[i_level] = segment_len;
        tsegments[i_level]    = tsegment;
        dparams[i_level]      = dparam_arr;
        params[i_level].resize(m_cfg.get_nparams());
        fold_shapes[i_level].resize(m_cfg.get_nparams() + 3);

        fold_shapes[i_level][0] = nsegments;
        for (SizeType iparam = 0; iparam < m_cfg.get_nparams(); ++iparam) {
            const auto param_arr = psr_utils::range_param(
                m_cfg.get_param_limits()[iparam][0],
                m_cfg.get_param_limits()[iparam][1], dparam_arr[iparam]);
            params[i_level][iparam]          = param_arr;
            fold_shapes[i_level][iparam + 1] = param_arr.size();
        }
        fold_shapes[i_level][m_cfg.get_nparams() + 1] = 2;
        fold_shapes[i_level][m_cfg.get_nparams() + 2] = m_cfg.get_nbins();
    }
    // Check param arr lengths for the initialization
    if (m_cfg.get_nparams() > 1) {
        for (SizeType iparam = 0; iparam < m_cfg.get_nparams() - 1; ++iparam) {
            if (params[0][iparam].size() != 1) {
                throw std::runtime_error(
                    "Only one parameter for higher order derivatives is "
                    "supported for the initialization");
            }
        }
    }

    // Now resolve the params for the FFA plan
    for (SizeType i_level = 1; i_level < levels; ++i_level) {
        std::vector<double> p_set_cur(m_cfg.get_nparams());
        const auto strides = calculate_strides(params[i_level - 1]);
        for (const auto& p_set_view :
             utils::cartesian_product_view(params[i_level])) {
            for (SizeType iparam = 0; iparam < m_cfg.get_nparams(); ++iparam) {
                p_set_cur[iparam] = p_set_view[iparam];
            }
            const auto [p_idx_tail, phase_shift_tail] =
                core::ffa_taylor_resolve(p_set_cur, params[i_level - 1],
                                         i_level, 0, m_cfg.get_tseg_brute(),
                                         m_cfg.get_nbins());
            const auto [p_idx_head, phase_shift_head] =
                core::ffa_taylor_resolve(p_set_cur, params[i_level - 1],
                                         i_level, 1, m_cfg.get_tseg_brute(),
                                         m_cfg.get_nbins());
            const auto i_tail =
                std::inner_product(p_idx_tail.begin(), p_idx_tail.end(),
                                   strides.begin(), SizeType{0});
            const auto i_head =
                std::inner_product(p_idx_head.begin(), p_idx_head.end(),
                                   strides.begin(), SizeType{0});

            const auto coord_cur = FFACoord{.i_tail     = i_tail,
                                            .shift_tail = phase_shift_tail,
                                            .i_head     = i_head,
                                            .shift_head = phase_shift_head};
            coordinates[i_level].emplace_back(coord_cur);
        }
    }
}

std::vector<SizeType>
FFAPlan::calculate_strides(std::span<const std::vector<double>> p_arr) {
    const auto nparams = p_arr.size();
    std::vector<SizeType> strides(nparams);
    strides[nparams - 1] = 1;
    for (int i = static_cast<int>(nparams) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * p_arr[i + 1].size();
    }
    return strides;
}

FFA::FFA(search::PulsarSearchConfig cfg)
    : m_cfg(std::move(cfg)),
      m_ffa_plan(m_cfg),
      m_nthreads(m_cfg.get_nthreads()) {
    // Allocate memory for the FFA buffers
    m_fold_in.resize(m_ffa_plan.get_buffer_size(), 0.0F);
    m_fold_out.resize(m_ffa_plan.get_buffer_size(), 0.0F);
}

namespace {
inline void shift_add(const float* __restrict__ data_tail,
                      SizeType phase_shift_tail,
                      const float* __restrict__ data_head,
                      SizeType phase_shift_head,
                      float* __restrict__ out,
                      SizeType nbins) {

    const SizeType shift_tail = phase_shift_tail % nbins;
    const SizeType shift_head = phase_shift_head % nbins;

    const float* __restrict__ data_tail_row1 = data_tail + nbins;
    const float* __restrict__ data_head_row1 = data_head + nbins;
    float* __restrict__ out_row1             = out + nbins;
    for (SizeType j = 0; j < nbins; ++j) {
        const SizeType idx_tail =
            (j < shift_tail) ? (j + nbins - shift_tail) : (j - shift_tail);
        const SizeType idx_head =
            (j < shift_head) ? (j + nbins - shift_head) : (j - shift_head);
        out[j]      = data_tail[idx_tail] + data_head[idx_head];
        out_row1[j] = data_tail_row1[idx_tail] + data_head_row1[idx_head];
    }
}
} // namespace

const FFAPlan& FFA::get_plan() const { return m_ffa_plan; }

void FFA::initialize(std::span<const float> ts_e, std::span<const float> ts_v) {
    auto start = std::chrono::steady_clock::now();
    spdlog::info("FFA initialize");
    const auto t_ref =
        m_cfg.get_nparams() == 1 ? 0.0 : m_ffa_plan.tsegments[0] / 2.0;
    const auto freqs_arr = m_ffa_plan.params[0].back();
    algorithms::BruteFold bf(freqs_arr, m_ffa_plan.segment_lens[0],
                             m_cfg.get_nbins(), m_cfg.get_nsamps(),
                             m_cfg.get_tsamp(), t_ref, m_cfg.get_nthreads());
    bf.execute(ts_e, ts_v, std::span(m_fold_in.data(), bf.get_fold_size()));
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    spdlog::info("FFA::initialize took {} ms", elapsed_ms);
}

void FFA::execute(std::span<const float> ts_e,
                  std::span<const float> ts_v,
                  std::span<float> fold) {
    if (ts_e.size() != m_cfg.get_nsamps()) {
        throw std::runtime_error(std::format("ts must have size nsamps (got "
                                             "{} != {})",
                                             ts_e.size(), m_cfg.get_nsamps()));
    }
    if (ts_v.size() != ts_e.size()) {
        throw std::runtime_error(
            std::format("ts variance must have size nsamps "
                        "(got {} != {})",
                        ts_v.size(), ts_e.size()));
    }
    if (fold.size() != m_ffa_plan.get_fold_size()) {
        throw std::runtime_error(std::format("Output array has wrong size (got "
                                             "{} != {})",
                                             fold.size(),
                                             m_ffa_plan.get_fold_size()));
    }
    auto start = std::chrono::steady_clock::now();
    indicators::show_console_cursor(false);
    indicators::ProgressBar bar{
        indicators::option::PrefixText{"Computing FFA"},
        indicators::option::ShowPercentage(true),
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
    };
    initialize(ts_e, ts_v);
    // Use raw pointers for swapping buffers
    float* fold_in_ptr  = m_fold_in.data();
    float* fold_out_ptr = m_fold_out.data();
    const auto levels   = m_cfg.get_niters_ffa() + 1;

    for (SizeType i_level = 1; i_level < levels - 1; ++i_level) {
        execute_iter(fold_in_ptr, fold_out_ptr, i_level);
        std::swap(fold_in_ptr, fold_out_ptr);
        const auto progress = static_cast<float>(i_level) /
                              static_cast<float>(levels - 1) * 100.0F;
        bar.set_progress(static_cast<SizeType>(progress));
    }
    // Last iteration directly writes to the output buffer
    execute_iter(fold_in_ptr, fold.data(), m_cfg.get_niters_ffa());
    bar.set_progress(100);
    indicators::show_console_cursor(true);
    spdlog::info("FFA finished");
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    spdlog::info("FFA::execute took {} ms", elapsed_ms);
}

void FFA::execute_iter(const float* __restrict__ fold_in,
                       float* __restrict__ fold_out,
                       SizeType i_level) {
    const auto& coords_cur  = m_ffa_plan.coordinates[i_level];
    const auto& coords_prev = m_ffa_plan.coordinates[i_level - 1];
    const auto nsegments    = m_ffa_plan.fold_shapes[i_level][0];
    const auto nbins        = m_ffa_plan.fold_shapes[i_level].back();
    const auto ncoords_cur  = coords_cur.size();
    const auto ncoords_prev = coords_prev.size();
#pragma omp parallel for num_threads(m_nthreads)
    for (SizeType icoord = 0; icoord < ncoords_cur; ++icoord) {
        const auto& coord_cur = coords_cur[icoord];
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            const auto tail_offset = ((iseg * 2) * ncoords_prev * 2 * nbins) +
                                     (coord_cur.i_tail * 2 * nbins);
            const auto head_offset =
                ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                (coord_cur.i_head * 2 * nbins);
            const auto out_offset =
                (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);
            shift_add(fold_in + tail_offset, coord_cur.shift_tail,
                      fold_in + head_offset, coord_cur.shift_head,
                      fold_out + out_offset, nbins);
        }
    }
}

std::vector<float> compute_ffa(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               search::PulsarSearchConfig cfg) {
    FFA ffa(std::move(cfg));
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}

} // namespace loki::algorithms