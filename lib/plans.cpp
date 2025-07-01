#include "loki/algorithms/plans.hpp"

#include <cstdint>
#include <numeric>
#include <ranges>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/cartesian.hpp"
#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/search/configs.hpp"

namespace loki::plans {

FFAPlan::FFAPlan(search::PulsarSearchConfig cfg) : m_cfg(std::move(cfg)) {
    configure_plan();
    validate_plan();
}

SizeType FFAPlan::get_buffer_size() const noexcept {
    return std::ranges::max(
        fold_shapes | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<>());
        }));
}

SizeType FFAPlan::get_brute_fold_size() const noexcept {
    return std::accumulate(fold_shapes.front().begin(),
                           fold_shapes.front().end(), 1, std::multiplies<>());
}

SizeType FFAPlan::get_brute_fold_size_complex() const noexcept {
    const auto nfft = get_brute_fold_size() / m_cfg.get_nbins();
    // For in-place RFFT, FFTW needs nfft * (nbins + 2) floats total
    return nfft * (m_cfg.get_nbins() + 2);
}

SizeType FFAPlan::get_fold_size() const noexcept {
    return std::accumulate(fold_shapes.back().begin(), fold_shapes.back().end(),
                           1, std::multiplies<>());
}

SizeType FFAPlan::get_buffer_size_complex() const noexcept {
    // Calculate the standard complex buffer size (max of all FFA levels)
    const auto standard_complex_size = std::ranges::max(
        fold_shapes_complex | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<>());
        }));
    const auto required_complex = (get_brute_fold_size_complex() + 1) / 2;
    return std::max(static_cast<SizeType>(standard_complex_size),
                    required_complex);
}

SizeType FFAPlan::get_fold_size_complex() const noexcept {
    return std::accumulate(fold_shapes_complex.back().begin(),
                           fold_shapes_complex.back().end(), 1,
                           std::multiplies<>());
}

SizeType FFAPlan::get_memory_usage() const noexcept {
    SizeType total_memory = 0;
    if (m_cfg.get_use_fft_shifts()) {
        const auto complex_buffer_size = get_buffer_size_complex();
        total_memory += 2 * complex_buffer_size * sizeof(ComplexType);
        const auto temp_complex_size = get_fold_size_complex();
        total_memory += temp_complex_size * sizeof(ComplexType);
    } else {
        const auto float_buffer_size = get_buffer_size();
        total_memory += 2 * float_buffer_size * sizeof(float);
    }

    return total_memory;
}

std::map<std::string, std::vector<double>> FFAPlan::get_params_dict() const {
    const auto param_names = m_cfg.get_param_names();
    const auto params_arr  = params.back();
    error_check::check_equal(params_arr.size(), param_names.size(),
                             "Number of parameters in the last level of the "
                             "FFA plan does not match the number of parameter "
                             "names");
    std::map<std::string, std::vector<double>> result;
    for (SizeType i = 0; i < param_names.size(); ++i) {
        result[param_names[i]] = params_arr[i];
    }
    return result;
}

void FFAPlan::configure_plan() {
    const auto levels = m_cfg.get_niters_ffa() + 1;
    segment_lens.resize(levels);
    nsegments.resize(levels);
    tsegments.resize(levels);
    ncoords.resize(levels);
    params.resize(levels);
    dparams.resize(levels);
    fold_shapes.resize(levels);
    fold_shapes_complex.resize(levels);
    coordinates.resize(levels);

    for (SizeType i_level = 0; i_level < levels; ++i_level) {
        const auto segment_len = m_cfg.get_bseg_brute() * (1U << i_level);
        const auto tsegment =
            static_cast<double>(segment_len) * m_cfg.get_tsamp();
        const auto nsegments_cur = m_cfg.get_nsamps() / segment_len;
        const auto dparam_arr    = m_cfg.get_dparams(tsegment);

        segment_lens[i_level] = segment_len;
        nsegments[i_level]    = nsegments_cur;
        tsegments[i_level]    = tsegment;
        dparams[i_level]      = dparam_arr;
        params[i_level].resize(m_cfg.get_nparams());
        fold_shapes[i_level].resize(m_cfg.get_nparams() + 3);
        fold_shapes_complex[i_level].resize(m_cfg.get_nparams() + 3);

        fold_shapes[i_level][0]         = nsegments_cur;
        fold_shapes_complex[i_level][0] = nsegments_cur;
        for (SizeType iparam = 0; iparam < m_cfg.get_nparams(); ++iparam) {
            const auto param_arr = psr_utils::range_param(
                m_cfg.get_param_limits()[iparam][0],
                m_cfg.get_param_limits()[iparam][1], dparam_arr[iparam]);
            params[i_level][iparam]                  = param_arr;
            fold_shapes[i_level][iparam + 1]         = param_arr.size();
            fold_shapes_complex[i_level][iparam + 1] = param_arr.size();
        }
        fold_shapes[i_level][m_cfg.get_nparams() + 1]         = 2;
        fold_shapes_complex[i_level][m_cfg.get_nparams() + 1] = 2;
        fold_shapes[i_level][m_cfg.get_nparams() + 2] = m_cfg.get_nbins();
        fold_shapes_complex[i_level][m_cfg.get_nparams() + 2] =
            m_cfg.get_nbins_f();
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
    // First do level 0 (important for book-keeping, ncoords_prev!)
    std::vector<double> p_set_cur(m_cfg.get_nparams());
    for (const auto& p_set_view : utils::cartesian_product_view(params[0])) {
        for (SizeType iparam = 0; iparam < m_cfg.get_nparams(); ++iparam) {
            p_set_cur[iparam] = p_set_view[iparam];
        }
        const auto coord_cur = FFACoord{.i_tail     = SIZE_MAX,
                                        .shift_tail = 0,
                                        .i_head     = SIZE_MAX,
                                        .shift_head = 0};
        coordinates[0].emplace_back(coord_cur);
    }
    ncoords[0] = coordinates[0].size();

    for (SizeType i_level = 1; i_level < levels; ++i_level) {
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
        ncoords[i_level] = coordinates[i_level].size();
    }
}

void FFAPlan::validate_plan() const {
    error_check::check(get_buffer_size() > 0,
                       "FFAPlan::validate_plan: buffer size is zero");
    error_check::check(get_buffer_size_complex() > 0,
                       "FFAPlan::validate_plan: buffer size (complex) is zero");
    error_check::check(get_fold_size() > 0,
                       "FFAPlan::validate_plan: fold size is zero");
    error_check::check(get_fold_size_complex() > 0,
                       "FFAPlan::validate_plan: fold size (complex) is zero");
    error_check::check(get_brute_fold_size() > 0,
                       "FFAPlan::validate_plan: brute fold size is zero");
    error_check::check(
        get_brute_fold_size_complex() > 0,
        "FFAPlan::validate_plan: brute fold size (complex) is zero");

    // For the first level, only the freqs array should have size > 1
    for (SizeType iparam = 0; iparam < m_cfg.get_nparams() - 1; ++iparam) {
        if (params[0][iparam].size() != 1) {
            throw std::runtime_error(
                "FFAPlan::validate_plan: Only one parameter for higher order "
                "derivatives is supported for the initialization for the "
                "first level");
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

} // namespace loki::plans