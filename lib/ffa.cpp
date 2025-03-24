#include "loki/loki_types.hpp"
#include <loki/ffa.hpp>

#include <numeric>
#include <ranges>
#include <utility>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <loki/basic.hpp>
#include <loki/cartesian.hpp>
#include <loki/configs.hpp>
#include <loki/defaults.hpp>
#include <loki/fold.hpp>
#include <loki/psr_utils.hpp>
#include <loki/utils.hpp>

FFAPlan::FFAPlan(PulsarSearchConfig cfg) : m_cfg(std::move(cfg)) {
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
        const auto segment_len = m_cfg.get_bseg_brute() * (1 << i_level);
        const auto tsegment =
            static_cast<float>(segment_len) * m_cfg.get_tsamp();
        const auto nsegments  = m_cfg.get_nsamps() / segment_len;
        const auto dparam_arr = m_cfg.get_dparams(tsegment);

        segment_lens[i_level] = segment_len;
        tsegments[i_level]    = tsegment;
        dparams[i_level]      = dparam_arr;
        params[i_level].resize(m_cfg.get_nparams());
        fold_shapes[i_level].resize(m_cfg.get_nparams() + 2);

        fold_shapes[i_level][0] = nsegments;
        for (SizeType iparam = 0; iparam < m_cfg.get_nparams(); ++iparam) {
            const auto param_arr = loki::utils::range_param(
                m_cfg.get_param_limits()[iparam][0],
                m_cfg.get_param_limits()[iparam][1], dparam_arr[iparam]);
            params[i_level][iparam]          = param_arr;
            fold_shapes[i_level][iparam + 1] = param_arr.size();
        }
        fold_shapes[i_level][m_cfg.get_nparams() + 1] = m_cfg.get_nbins();
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
        std::vector<FloatType> p_set_cur(m_cfg.get_nparams());
        const auto strides = calculate_strides(params[i_level - 1]);
        for (const auto& p_set_view : cartesian_product_view(params[i_level])) {
            for (SizeType iparam = 0; iparam < m_cfg.get_nparams(); ++iparam) {
                p_set_cur[iparam] = p_set_view[iparam];
            }
            const auto [p_idx_tail, phase_shift_tail] =
                loki::ffa_resolve(p_set_cur, params[i_level - 1], i_level, 0,
                                  m_cfg.get_tseg_brute(), m_cfg.get_nbins());
            const auto [p_idx_head, phase_shift_head] =
                loki::ffa_resolve(p_set_cur, params[i_level - 1], i_level, 1,
                                  m_cfg.get_tseg_brute(), m_cfg.get_nbins());
            const auto& i_tail =
                std::inner_product(p_idx_tail.begin(), p_idx_tail.end(),
                                   strides.begin(), SizeType{0});
            const auto& i_head =
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
FFAPlan::calculate_strides(std::span<const std::vector<FloatType>> p_arr) {
    const auto nparams = p_arr.size();
    std::vector<SizeType> strides(nparams);
    strides[nparams - 1] = 1;
    for (SizeType i = nparams - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * p_arr[i + 1].size();
    }
    return strides;
}

FFA::FFA(PulsarSearchConfig cfg) : m_cfg(std::move(cfg)), m_ffa_plan(m_cfg) {
    // Allocate memory for the FFA buffers
    m_fold_in.resize(m_ffa_plan.get_buffer_size(), 0.0F);
    m_fold_out.resize(m_ffa_plan.get_buffer_size(), 0.0F);
}

namespace {
inline void shift_add(const float* __restrict__ data1,
                      SizeType shift1,
                      const float* __restrict__ data2,
                      SizeType shift2,
                      float* __restrict__ out,
                      SizeType nbins) {
    constexpr SizeType kSimdSize = xsimd::simd_type<float>::size;
    SizeType vec_size            = (nbins / kSimdSize) * kSimdSize;
    // Vectorized loop
    for (SizeType i = 0; i < vec_size; i += kSimdSize) {
        SizeType idx1 = (i + shift1 + nbins) % nbins;
        SizeType idx2 = (i + shift2 + nbins) % nbins;
        auto v1       = xsimd::load_unaligned(&data1[idx1]);
        auto v2       = xsimd::load_unaligned(&data2[idx2]);
        (v1 + v2).store_unaligned(&out[i]);
    }
    for (SizeType i = vec_size; i < nbins; ++i) {
        SizeType idx1 = (i + shift1 + nbins) % nbins;
        SizeType idx2 = (i + shift2 + nbins) % nbins;
        out[i]        = data1[idx1] + data2[idx2];
    }
}
} // namespace

const FFAPlan& FFA::get_plan() const { return m_ffa_plan; }

void FFA::initialize(std::span<const float> ts_e, std::span<const float> ts_v) {
    spdlog::debug("Initializing FFA");
    const auto t_ref     = m_ffa_plan.tsegments[0] / 2.0F;
    const auto freqs_arr = m_ffa_plan.params[0].back();
    BruteFold bf(freqs_arr, m_ffa_plan.segment_lens[0], m_cfg.get_nbins(),
                 m_cfg.get_nsamps(), m_cfg.get_tsamp(), t_ref,
                 m_cfg.get_nthreads());
    auto fold_init = std::span(m_fold_in.data(), bf.get_fold_size());
    bf.execute(ts_e, ts_v, fold_init);
}

void FFA::execute(std::span<const float> ts_e,
                  std::span<const float> ts_v,
                  std::span<float> fold) {
    if (ts_e.size() != m_cfg.get_nsamps()) {
        throw std::runtime_error("ts must have size nsamps");
    }
    if (ts_v.size() != ts_e.size()) {
        throw std::runtime_error("ts variance must have size nsamps");
    }
    if (fold.size() != m_ffa_plan.get_fold_size()) {
        throw std::runtime_error("Output array has wrong size");
    }
    spdlog::info("Running FFA");
    indicators::show_console_cursor(false);
    indicators::ProgressBar bar{
        indicators::option::PrefixText{"Computing"},
        indicators::option::ShowPercentage(true),
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
    };
    initialize(ts_e, ts_v);
    // Use raw pointers for swapping buffers
    float* fold_in_ptr  = m_fold_in.data();
    float* fold_out_ptr = m_fold_out.data();

    for (SizeType i_iter = 1; i_iter < m_cfg.get_niters_ffa() + 1; ++i_iter) {
        execute_iter(fold_in_ptr, fold_out_ptr, i_iter);
        std::swap(fold_in_ptr, fold_out_ptr);
        const auto progress = static_cast<float>(i_iter) /
                              static_cast<float>(m_cfg.get_niters_ffa()) *
                              100.0F;
        bar.set_progress(static_cast<SizeType>(progress));
    }
    // Last iteration directly writes to the output buffer
    execute_iter(fold_in_ptr, fold.data(), m_cfg.get_niters_ffa());
    bar.set_progress(100);
    indicators::show_console_cursor(true);
    spdlog::info("FFA finished");
}

void FFA::execute_iter(const float* __restrict__ fold_in,
                       float* __restrict__ fold_out,
                       SizeType i_iter) {
    const auto& coords_cur  = m_ffa_plan.coordinates[i_iter];
    const auto& coords_prev = m_ffa_plan.coordinates[i_iter - 1];
    const auto nsegments    = m_ffa_plan.fold_shapes[i_iter][0];
    const auto nbins        = m_ffa_plan.fold_shapes[i_iter].back();
    const auto ncoords_cur  = coords_cur.size();
    const auto ncoords_prev = coords_prev.size();
#pragma omp parallel for num_threads(m_cfg.get_nthreads())
    for (SizeType icoord = 0; icoord < ncoords_cur; ++icoord) {
        const auto& coord_cur = coords_cur[icoord];
        for (SizeType iseg = 0; iseg < nsegments; ++iseg) {
            SizeType tail_offset = ((iseg * 2) * ncoords_prev * 2 * nbins) +
                                   (coord_cur.i_tail * 2 * nbins);
            SizeType head_offset = ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) +
                                   (coord_cur.i_head * 2 * nbins);
            SizeType out_offset =
                (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

            shift_add(fold_in + tail_offset, coord_cur.shift_tail,
                      fold_in + head_offset, coord_cur.shift_head,
                      fold_out + out_offset, nbins);
        }
    }
}

std::vector<float> compute_ffa(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               PulsarSearchConfig cfg) {
    FFA ffa(std::move(cfg));
    const auto& ffa_plan = ffa.get_plan();
    std::vector<float> fold(ffa_plan.get_fold_size(), 0.0F);
    ffa.execute(ts_e, ts_v, std::span<float>(fold));
    return fold;
}