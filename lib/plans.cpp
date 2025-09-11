#include "loki/algorithms/plans.hpp"

#include <numeric>
#include <ranges>
#include <utility>

#include <spdlog/spdlog.h>

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
            return std::accumulate(shape.begin(), shape.end(), SizeType{1},
                                   std::multiplies<>());
        }));
}

SizeType FFAPlan::get_brute_fold_size() const noexcept {
    return std::accumulate(fold_shapes.front().begin(),
                           fold_shapes.front().end(), SizeType{1},
                           std::multiplies<>());
}

SizeType FFAPlan::get_fold_size() const noexcept {
    return std::accumulate(fold_shapes.back().begin(), fold_shapes.back().end(),
                           SizeType{1}, std::multiplies<>());
}

SizeType FFAPlan::get_buffer_size_complex() const noexcept {
    // Calculate the standard complex buffer size (max of all FFA levels)
    return std::ranges::max(
        fold_shapes_complex | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), SizeType{1},
                                   std::multiplies<>());
        }));
}

SizeType FFAPlan::get_fold_size_complex() const noexcept {
    return std::accumulate(fold_shapes_complex.back().begin(),
                           fold_shapes_complex.back().end(), SizeType{1},
                           std::multiplies<>());
}

SizeType FFAPlan::get_coord_size() const noexcept {
    return std::accumulate(ncoords.begin(), ncoords.end(), 0, std::plus<>());
}

float FFAPlan::get_buffer_memory_usage() const noexcept {
    const auto internal_buffers = get_fold_size() >= get_buffer_size() ? 1 : 2;
    SizeType total_memory       = 0;
    if (m_cfg.get_use_fft_shifts()) {
        const auto complex_buffer_size = get_buffer_size_complex();
        total_memory +=
            internal_buffers * complex_buffer_size * sizeof(ComplexType);
    } else {
        const auto float_buffer_size = get_buffer_size();
        total_memory += internal_buffers * float_buffer_size * sizeof(float);
    }
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}

float FFAPlan::get_coord_memory_usage() const noexcept {
    const auto total_memory = get_coord_size() * sizeof(FFACoord);
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
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
    n_params          = m_cfg.get_nparams();
    n_levels          = levels;
    segment_lens.resize(levels);
    nsegments.resize(levels);
    tsegments.resize(levels);
    ncoords.resize(levels);
    ncoords_lb.resize(levels);
    params.resize(levels);
    param_cart_strides.resize(levels);
    dparams.resize(levels);
    dparams_lim.resize(levels);
    fold_shapes.resize(levels);
    fold_shapes_complex.resize(levels);

    for (SizeType i_level = 0; i_level < levels; ++i_level) {
        const auto segment_len = m_cfg.get_bseg_brute() * (1U << i_level);
        const auto tsegment =
            static_cast<double>(segment_len) * m_cfg.get_tsamp();
        const auto nsegments_cur  = m_cfg.get_nsamps() / segment_len;
        const auto dparam_arr     = m_cfg.get_dparams(tsegment);
        const auto dparam_arr_lim = m_cfg.get_dparams_lim(tsegment);

        segment_lens[i_level] = segment_len;
        nsegments[i_level]    = nsegments_cur;
        tsegments[i_level]    = tsegment;
        dparams[i_level]      = dparam_arr;
        dparams_lim[i_level]  = dparam_arr_lim;
        params[i_level].resize(n_params);
        fold_shapes[i_level].resize(n_params + 3);
        fold_shapes_complex[i_level].resize(n_params + 3);

        fold_shapes[i_level][0]         = nsegments_cur;
        fold_shapes_complex[i_level][0] = nsegments_cur;
        for (SizeType iparam = 0; iparam < n_params; ++iparam) {
            const auto param_arr = psr_utils::range_param(
                m_cfg.get_param_limits()[iparam][0],
                m_cfg.get_param_limits()[iparam][1], dparam_arr[iparam]);
            params[i_level][iparam]                  = param_arr;
            fold_shapes[i_level][iparam + 1]         = param_arr.size();
            fold_shapes_complex[i_level][iparam + 1] = param_arr.size();
        }
        param_cart_strides[i_level]        = calculate_strides(params[i_level]);
        fold_shapes[i_level][n_params + 1] = 2;
        fold_shapes_complex[i_level][n_params + 1] = 2;
        fold_shapes[i_level][n_params + 2]         = m_cfg.get_nbins();
        fold_shapes_complex[i_level][n_params + 2] = m_cfg.get_nbins_f();
        ncoords[i_level] =
            std::accumulate(params[i_level].begin(), params[i_level].end(), 1UL,
                            [](SizeType acc, const auto& param_vec) {
                                return acc * param_vec.size();
                            });
        ncoords_lb[i_level] =
            ncoords[i_level] > 0
                ? std::log2(static_cast<float>(ncoords[i_level]))
                : 0.0F;
    }
    // Check param arr lengths for the initialization
    if (n_params > 1) {
        for (SizeType iparam = 0; iparam < n_params - 1; ++iparam) {
            if (params[0][iparam].size() != 1) {
                throw std::runtime_error(
                    "Only one parameter for higher order derivatives is "
                    "supported for the initialization");
            }
        }
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

void FFAPlan::resolve_coordinates_freq(
    std::span<std::vector<FFACoordFreq>> coordinates) {
    error_check::check_equal(n_params, 1,
                             "resolve_coordinates_freq() only supports "
                             "nparams=1");
    // Resolve the frequency coordinates for the FFA plan
    const auto ncoords_max = std::ranges::max(ncoords);
    std::vector<float> relative_phase_batch(ncoords_max);
    std::vector<uint32_t> pindex_prev_flat(ncoords_max);
    for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
        const auto ncoords_cur = ncoords[i_level];
        auto& coords_cur       = coordinates[i_level];
        error_check::check_greater_equal(coords_cur.size(), ncoords_cur,
                                         "FFAPlan::resolve_coordinates: "
                                         "coords_cur must have size >= "
                                         "ncoords_cur");
        auto relative_phase_batch_span =
            std::span(relative_phase_batch).first(ncoords_cur);
        auto pindex_prev_flat_span =
            std::span(pindex_prev_flat).first(ncoords_cur);
        core::ffa_taylor_resolve_freq_batch(
            params[i_level], params[i_level - 1], pindex_prev_flat_span,
            relative_phase_batch_span, i_level, m_cfg.get_tseg_brute(),
            m_cfg.get_nbins());
        // Generate coordinates for the head
        for (SizeType coord_idx = 0; coord_idx < ncoords_cur; ++coord_idx) {
            coords_cur[coord_idx].idx   = pindex_prev_flat[coord_idx];
            coords_cur[coord_idx].shift = relative_phase_batch[coord_idx];
        }
    }
}

void FFAPlan::resolve_coordinates(
    std::span<std::vector<FFACoord>> coordinates) {
    error_check::check_greater_equal(
        n_params, 2U,
        "resolve_coordinates only "
        "supports nparams>=2. For frequency "
        "coordinates, use resolve_coordinates_freq() instead.");

    error_check::check_less_equal(
        n_params, 3U,
        "resolve_coordinates only "
        "supports nparams<=3. Larger values are "
        "not supported and advised. Use Pruning instead.");

    using ResolveFunc =
        void (*)(std::span<const std::vector<double>>,
                 std::span<const std::vector<double>>, std::span<uint32_t>,
                 std::span<float>, SizeType, SizeType, double, SizeType);

    constexpr std::array<ResolveFunc, 3> kResolveFuncs = {
        core::ffa_taylor_resolve_accel_batch, // nparams == 2
        core::ffa_taylor_resolve_jerk_batch,  // nparams == 3
    };

    const auto resolve_func = kResolveFuncs[n_params - 2];

    // Resolve the params for the FFA plan
    const auto ncoords_max = std::ranges::max(ncoords);
    std::vector<float> relative_phase_batch(ncoords_max);
    std::vector<uint32_t> pindex_prev_flat(ncoords_max);

    for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
        const auto ncoords_cur = ncoords[i_level];
        auto& coords_cur       = coordinates[i_level];
        error_check::check_greater_equal(coords_cur.size(), ncoords_cur,
                                         "FFAPlan::resolve_coordinates: "
                                         "coords_cur must have size >= "
                                         "ncoords_cur");
        auto relative_phase_batch_span =
            std::span(relative_phase_batch).first(ncoords_cur);
        auto pindex_prev_flat_span =
            std::span(pindex_prev_flat).first(ncoords_cur);

        // Tail coordinates
        resolve_func(params[i_level], params[i_level - 1],
                     pindex_prev_flat_span, relative_phase_batch_span, i_level,
                     0, m_cfg.get_tseg_brute(), m_cfg.get_nbins());
        for (SizeType coord_idx = 0; coord_idx < ncoords_cur; ++coord_idx) {
            coords_cur[coord_idx].i_tail     = pindex_prev_flat[coord_idx];
            coords_cur[coord_idx].shift_tail = relative_phase_batch[coord_idx];
        }

        // Head coordinates
        resolve_func(params[i_level], params[i_level - 1],
                     pindex_prev_flat_span, relative_phase_batch_span, i_level,
                     1, m_cfg.get_tseg_brute(), m_cfg.get_nbins());
        for (SizeType coord_idx = 0; coord_idx < ncoords_cur; ++coord_idx) {
            coords_cur[coord_idx].i_head     = pindex_prev_flat[coord_idx];
            coords_cur[coord_idx].shift_head = relative_phase_batch[coord_idx];
        }
    }
}

std::vector<std::vector<FFACoordFreq>> FFAPlan::resolve_coordinates_freq() {
    std::vector<std::vector<FFACoordFreq>> coordinates(n_levels);
    for (SizeType i_level = 0; i_level < n_levels; ++i_level) {
        coordinates[i_level].resize(ncoords[i_level]);
    }
    resolve_coordinates_freq(std::span(coordinates));
    return coordinates;
}

std::vector<std::vector<FFACoord>> FFAPlan::resolve_coordinates() {
    std::vector<std::vector<FFACoord>> coordinates(n_levels);
    for (SizeType i_level = 0; i_level < n_levels; ++i_level) {
        coordinates[i_level].resize(ncoords[i_level]);
    }
    resolve_coordinates(std::span(coordinates));
    return coordinates;
}

std::vector<double> FFAPlan::generate_branching_pattern() const {
    return psr_utils::generate_branching_pattern(
        params.back(), dparams_lim.back(), m_cfg.get_param_limits(),
        m_cfg.get_tseg_ffa(), nsegments.back() - 1, m_cfg.get_nbins(),
        m_cfg.get_tol_bins());
}

} // namespace loki::plans