#include "loki/algorithms/plans.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include "loki/common/types.hpp"
#include "loki/core/circular.hpp"
#include "loki/core/taylor.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/search/configs.hpp"

namespace loki::plans {
namespace detail {

// --- FFAPlanBase implementation ---
FFAPlanBase::FFAPlanBase(search::PulsarSearchConfig cfg)
    : m_cfg(std::move(cfg)) {
    configure_base_plan();
    validate_base_plan();
}
SizeType FFAPlanBase::get_coord_size() const noexcept {
    return std::accumulate(m_ncoords.begin(), m_ncoords.end(), 0,
                           std::plus<>());
}
SizeType FFAPlanBase::get_total_params_flat_count() const noexcept {
    return std::accumulate(m_params_flat_sizes.begin(),
                           m_params_flat_sizes.end(), 0, std::plus<>());
}
float FFAPlanBase::get_coord_memory_usage() const noexcept {
    SizeType total_memory;
    if (m_cfg.get_nparams() == 1) {
        total_memory = get_coord_size() * sizeof(coord::FFACoordFreq);
    } else {
        total_memory = get_coord_size() * sizeof(coord::FFACoord);
    }
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}

void FFAPlanBase::configure_base_plan() {
    const auto levels = m_cfg.get_niters_ffa() + 1;
    m_n_params        = m_cfg.get_nparams();
    m_n_levels        = levels;
    m_segment_lens.resize(levels);
    m_nsegments.resize(levels);
    m_tsegments.resize(levels);
    m_ncoords.resize(levels);
    m_ncoords_lb.resize(levels);
    m_ncoords_offsets.resize(levels + 1); // Include final sentinel
    m_params_flat_offsets.resize(levels);
    m_params_flat_sizes.resize(levels);
    m_param_counts.resize(levels);
    m_param_cart_strides.resize(levels);
    m_dparams.resize(levels);
    m_dparams_lim.resize(levels);

    for (SizeType i_level = 0; i_level < levels; ++i_level) {
        const auto segment_len = m_cfg.get_bseg_brute() * (1U << i_level);
        const auto tsegment =
            static_cast<double>(segment_len) * m_cfg.get_tsamp();
        const auto nsegments_cur  = m_cfg.get_nsamps() / segment_len;
        const auto dparam_arr     = m_cfg.get_dparams(tsegment);
        const auto dparam_arr_lim = m_cfg.get_dparams_lim(tsegment);

        m_segment_lens[i_level] = segment_len;
        m_nsegments[i_level]    = nsegments_cur;
        m_tsegments[i_level]    = tsegment;
        m_dparams[i_level]      = dparam_arr;
        m_dparams_lim[i_level]  = dparam_arr_lim;
        m_param_counts[i_level].resize(m_n_params);

        SizeType param_size_level = 0;
        SizeType ncoords_level    = 1;
        for (SizeType iparam = 0; iparam < m_n_params; ++iparam) {
            const auto param_size_cur = psr_utils::range_param_count(
                m_cfg.get_param_limits()[iparam][0],
                m_cfg.get_param_limits()[iparam][1], dparam_arr[iparam]);
            m_param_counts[i_level][iparam] = param_size_cur;
            param_size_level += param_size_cur;
            ncoords_level *= param_size_cur;
        }
        m_param_cart_strides[i_level] =
            calculate_strides(m_param_counts[i_level]);
        m_params_flat_sizes[i_level] = param_size_level;
        m_ncoords[i_level]           = ncoords_level;
        m_ncoords_lb[i_level] =
            m_ncoords[i_level] > 0
                ? std::log2(static_cast<float>(m_ncoords[i_level]))
                : 0.0F;
        if (i_level == 0) {
            m_ncoords_offsets[i_level]     = 0;
            m_params_flat_offsets[i_level] = 0;
        } else {
            m_ncoords_offsets[i_level] =
                m_ncoords_offsets[i_level - 1] + m_ncoords[i_level - 1];
            m_params_flat_offsets[i_level] =
                m_params_flat_offsets[i_level - 1] +
                m_params_flat_sizes[i_level - 1];
        }
    }
    m_ncoords_offsets[levels] =
        m_ncoords_offsets[levels - 1] + m_ncoords[levels - 1];
}

void FFAPlanBase::validate_base_plan() const {
    // For the first level, only the freqs array should have size > 1
    for (SizeType iparam = 0; iparam < m_n_params - 1; ++iparam) {
        if (m_param_counts[0][iparam] != 1) {
            throw std::runtime_error(
                "FFAPlan::validate_plan: Only one parameter for higher "
                "order derivatives is supported for the initialization for "
                "the first level");
        }
    }
}
std::vector<SizeType>
FFAPlanBase::calculate_strides(std::span<const SizeType> p_arr_counts) {
    const auto nparams = p_arr_counts.size();
    std::vector<SizeType> strides(nparams);
    strides[nparams - 1] = 1;
    for (int i = static_cast<int>(nparams) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * p_arr_counts[i + 1];
    }
    return strides;
}

} // namespace detail

// --- FFAPlanMetadata implementation ---
template <SupportedFoldType FoldType>
FFAPlanMetadata<FoldType>::FFAPlanMetadata(search::PulsarSearchConfig cfg)
    : FFAPlanBase(std::move(cfg)) {
    configure_fold_shapes();
    compute_flops();
    validate_metadata();
}
template <SupportedFoldType FoldType>
SizeType FFAPlanMetadata<FoldType>::get_brute_fold_size() const noexcept {
    return std::accumulate(m_fold_shapes.front().begin(),
                           m_fold_shapes.front().end(), SizeType{1},
                           std::multiplies<>());
}

template <SupportedFoldType FoldType>
SizeType FFAPlanMetadata<FoldType>::get_fold_size() const noexcept {
    return std::accumulate(m_fold_shapes.back().begin(),
                           m_fold_shapes.back().end(), SizeType{1},
                           std::multiplies<>());
}
template <SupportedFoldType FoldType>
SizeType FFAPlanMetadata<FoldType>::get_fold_size_time() const noexcept {
    return std::accumulate(m_fold_shapes_time.back().begin(),
                           m_fold_shapes_time.back().end(), SizeType{1},
                           std::multiplies<>());
}
template <SupportedFoldType FoldType>
SizeType FFAPlanMetadata<FoldType>::get_buffer_size() const noexcept {
    return std::ranges::max(
        m_fold_shapes | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), SizeType{1},
                                   std::multiplies<>());
        }));
}
template <SupportedFoldType FoldType>
SizeType FFAPlanMetadata<FoldType>::get_buffer_size_time() const noexcept {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        return 2 * get_buffer_size();
    } else {
        return get_buffer_size();
    }
}
template <SupportedFoldType FoldType>
float FFAPlanMetadata<FoldType>::get_buffer_memory_usage() const noexcept {
    const auto total_memory = 2 * get_buffer_size() * sizeof(FoldType);
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}
template <SupportedFoldType FoldType>
float FFAPlanMetadata<FoldType>::get_gflops(
    bool return_in_time) const noexcept {
    return (return_in_time ? m_flops_required_return_in_time
                           : m_flops_required) *
           1.0e-9F;
}

template <SupportedFoldType FoldType>
void FFAPlanMetadata<FoldType>::configure_fold_shapes() {
    m_fold_shapes.resize(m_n_levels);
    m_fold_shapes_time.resize(m_n_levels);
    for (SizeType i_level = 0; i_level < m_n_levels; ++i_level) {
        m_fold_shapes[i_level].resize(m_n_params + 3);
        m_fold_shapes_time[i_level].resize(m_n_params + 3);
        m_fold_shapes[i_level][0]      = m_nsegments[i_level];
        m_fold_shapes_time[i_level][0] = m_nsegments[i_level];
        for (SizeType iparam = 0; iparam < m_n_params; ++iparam) {
            m_fold_shapes[i_level][iparam + 1] =
                m_param_counts[i_level][iparam];
            m_fold_shapes_time[i_level][iparam + 1] =
                m_param_counts[i_level][iparam];
        }
        m_fold_shapes[i_level][m_n_params + 1]      = 2;
        m_fold_shapes_time[i_level][m_n_params + 1] = 2;
        m_fold_shapes_time[i_level][m_n_params + 2] = m_cfg.get_nbins();
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_fold_shapes[i_level][m_n_params + 2] = m_cfg.get_nbins_f();
        } else {
            m_fold_shapes[i_level][m_n_params + 2] = m_cfg.get_nbins();
        }
    }
}

template <SupportedFoldType FoldType>
void FFAPlanMetadata<FoldType>::compute_flops() {
    const auto nbins   = m_cfg.get_nbins();
    const auto nsamps  = m_cfg.get_nsamps();
    const auto nfreqs  = m_ncoords[0];
    const auto nbins_f = (nbins / 2) + 1;

    SizeType m_flops_brutefold{0U}, m_flops_ffa{0U};
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        if (nbins > m_cfg.get_nbins_min_lossy_bf()) {
            // O(nsamps) - Lossy BruteFold + RFFT
            m_flops_brutefold =
                (nsamps * 2 * nfreqs) + (nfreqs * nbins * std::log2(nbins));
        } else {
            // Direct DFT
            m_flops_brutefold = 8 * nsamps * 2 * nfreqs * nbins_f;
        }
        for (SizeType i_level = 1; i_level < m_n_levels; ++i_level) {
            const auto nsegments_cur = m_nsegments[i_level];
            const auto ncoords_cur   = m_ncoords[i_level];
            // 2 profiles (e+v) * nbins_f * 8 FLOPs per complex
            // shift-add
            m_flops_ffa += ncoords_cur * nsegments_cur * 2 * nbins_f * 8;
        }
    } else {
        // O(nsamps)
        m_flops_brutefold = (nsamps * 2 * nfreqs);
        for (SizeType i_level = 1; i_level < m_n_levels; ++i_level) {
            const auto nsegments_cur = m_nsegments[i_level];
            const auto ncoords_cur   = m_ncoords[i_level];
            // 2 profiles (e+v) * nbins * 1 FLOP per add
            m_flops_ffa += ncoords_cur * nsegments_cur * 2 * nbins;
        }
    }
    m_flops_required = m_flops_brutefold + m_flops_ffa;
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        // IRFFT
        const auto nfft = get_fold_size_time() / nbins;
        m_flops_required_return_in_time =
            m_flops_required + (nfft * nbins * std::log2(nbins));
    } else {
        m_flops_required_return_in_time = m_flops_required;
    }
}

template <SupportedFoldType FoldType>
void FFAPlanMetadata<FoldType>::validate_metadata() const {
    error_check::check(
        get_buffer_size() > 0,
        "FFAPlanMetadata::validate_metadata: buffer size is zero");
    error_check::check(get_fold_size() > 0,
                       "FFAPlanMetadata::validate_metadata: fold size is zero");
    error_check::check(
        get_brute_fold_size() > 0,
        "FFAPlanMetadata::validate_metadata: brute fold size is zero");
}

// --- FFAPlan implementation ---
template <SupportedFoldType FoldType>
FFAPlan<FoldType>::FFAPlan(search::PulsarSearchConfig cfg)
    : FFAPlanMetadata<FoldType>(std::move(cfg)) {
    configure_params();
}
template <SupportedFoldType FoldType>
std::map<std::string, std::vector<double>>
FFAPlan<FoldType>::get_params_dict() const {
    const auto param_names = this->m_cfg.get_param_names();
    const auto params_arr  = m_params.back();
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

template <SupportedFoldType FoldType>
void FFAPlan<FoldType>::resolve_coordinates(std::span<coord::FFACoord> coords) {
    error_check::check_greater_equal(
        this->m_n_params, 2U,
        "resolve_coordinates only supports nparams>=2. For frequency "
        "coordinates, use resolve_coordinates_freq() instead.");
    error_check::check_less_equal(this->m_n_params, 5U,
                                  "resolve_coordinates only supports "
                                  "nparams<=5. Larger values are "
                                  "not supported and advised.");
    error_check::check_greater_equal(coords.size(), this->get_coord_size(),
                                     "FFAPlan::resolve_coordinates: coords "
                                     "must have size >= coord_size");

    for (SizeType i_level = 1; i_level < this->m_n_levels; ++i_level) {
        const auto ncoords_cur    = this->m_ncoords[i_level];
        const auto ncoords_offset = this->m_ncoords_offsets[i_level];
        auto coords_span          = coords.subspan(ncoords_offset, ncoords_cur);

        // Tail coordinates
        core::ffa_taylor_resolve_poly_batch(
            m_params[i_level], m_params[i_level - 1], coords_span, i_level, 0,
            this->m_cfg.get_tseg_brute(), this->m_cfg.get_nbins(),
            this->m_n_params);

        // Head coordinates
        core::ffa_taylor_resolve_poly_batch(
            m_params[i_level], m_params[i_level - 1], coords_span, i_level, 1,
            this->m_cfg.get_tseg_brute(), this->m_cfg.get_nbins(),
            this->m_n_params);
    }
}
template <SupportedFoldType FoldType>
std::vector<std::vector<coord::FFACoord>>
FFAPlan<FoldType>::resolve_coordinates() {
    std::vector<coord::FFACoord> coords_flat(this->get_coord_size());
    resolve_coordinates(coords_flat);
    std::vector<std::vector<coord::FFACoord>> coords(this->m_n_levels);
    for (SizeType i_level = 0; i_level < this->m_n_levels; ++i_level) {
        const auto ncoords_cur    = this->m_ncoords[i_level];
        const auto ncoords_offset = this->m_ncoords_offsets[i_level];
        coords[i_level].resize(ncoords_cur);
        std::copy(coords_flat.begin() + static_cast<IndexType>(ncoords_offset),
                  coords_flat.begin() + static_cast<IndexType>(ncoords_offset) +
                      static_cast<IndexType>(ncoords_cur),
                  coords[i_level].begin());
    }
    return coords;
}

template <SupportedFoldType FoldType>
void FFAPlan<FoldType>::resolve_coordinates_freq(
    std::span<coord::FFACoordFreq> coords_freq) {
    error_check::check_equal(this->m_n_params, 1,
                             "resolve_coordinates_freq() only supports "
                             "nparams=1");
    // Resolve the frequency coordinates for the FFA plan
    error_check::check_greater_equal(
        coords_freq.size(), this->get_coord_size(),
        "FFAPlan::resolve_coordinates_freq: "
        "coords_freq must have size >= coord_size");

    for (SizeType i_level = 1; i_level < this->m_n_levels; ++i_level) {
        const auto ncoords_cur    = this->m_ncoords[i_level];
        const auto ncoords_offset = this->m_ncoords_offsets[i_level];
        auto coords_freq_span =
            coords_freq.subspan(ncoords_offset, ncoords_cur);
        core::ffa_taylor_resolve_freq_batch(
            m_params[i_level], m_params[i_level - 1], coords_freq_span, i_level,
            this->m_cfg.get_tseg_brute(), this->m_cfg.get_nbins());
    }
}
template <SupportedFoldType FoldType>
std::vector<std::vector<coord::FFACoordFreq>>
FFAPlan<FoldType>::resolve_coordinates_freq() {
    std::vector<coord::FFACoordFreq> coords_freq_flat(this->get_coord_size());
    resolve_coordinates_freq(coords_freq_flat);
    std::vector<std::vector<coord::FFACoordFreq>> coords_freq(this->m_n_levels);
    for (SizeType i_level = 0; i_level < this->m_n_levels; ++i_level) {
        const auto ncoords_cur    = this->m_ncoords[i_level];
        const auto ncoords_offset = this->m_ncoords_offsets[i_level];
        coords_freq[i_level].resize(ncoords_cur);
        std::copy(
            coords_freq_flat.begin() + static_cast<IndexType>(ncoords_offset),
            coords_freq_flat.begin() + static_cast<IndexType>(ncoords_offset) +
                static_cast<IndexType>(ncoords_cur),
            coords_freq[i_level].begin());
    }
    return coords_freq;
}

template <SupportedFoldType FoldType>
std::vector<double> FFAPlan<FoldType>::get_branching_pattern_approx(
    std::string_view poly_basis, SizeType ref_seg, IndexType isuggest) const {
    if (poly_basis == "taylor") {
        return core::generate_bp_poly_taylor_approx(
            m_params.back(), this->m_dparams_lim.back(),
            this->m_cfg.get_param_limits(), this->m_cfg.get_tseg_ffa(),
            this->m_nsegments.back(), this->m_cfg.get_nbins(),
            this->m_cfg.get_eta(), ref_seg, isuggest,
            this->m_cfg.get_use_conservative_tile());
    }
    throw std::invalid_argument(
        std::format("Invalid poly_basis ({}) for branching pattern generation",
                    poly_basis));
}

template <SupportedFoldType FoldType>
std::vector<double>
FFAPlan<FoldType>::get_branching_pattern(std::string_view poly_basis,
                                         SizeType ref_seg) const {
    if (poly_basis == "taylor") {
        if (this->m_n_params <= 4) {
            return core::generate_bp_poly_taylor(
                m_params.back(), this->m_dparams_lim.back(),
                this->m_cfg.get_param_limits(), this->m_cfg.get_tseg_ffa(),
                this->m_nsegments.back(), this->m_cfg.get_nbins(),
                this->m_cfg.get_eta(), ref_seg,
                this->m_cfg.get_use_conservative_tile());
        }
        if (this->m_n_params == 5) {
            return core::generate_bp_circ_taylor(
                m_params.back(), this->m_dparams_lim.back(),
                this->m_cfg.get_param_limits(), this->m_cfg.get_tseg_ffa(),
                this->m_nsegments.back(), this->m_cfg.get_nbins(),
                this->m_cfg.get_eta(), ref_seg, this->m_cfg.get_p_orb_min(),
                this->m_cfg.get_minimum_snap_cells(),
                this->m_cfg.get_use_conservative_tile());
        }
        throw std::invalid_argument("nparams > 5 not supported for "
                                    "branching pattern generation");
    }
    throw std::invalid_argument(
        std::format("Invalid poly_basis ({}) for branching pattern generation",
                    poly_basis));
}

template <SupportedFoldType FoldType>
std::vector<double> FFAPlan<FoldType>::get_params_flat() const noexcept {
    const SizeType total = this->get_total_params_flat_count();
    std::vector<double> params_flat;
    params_flat.reserve(total);
    for (const auto& level : m_params) {
        for (const auto& p : level) {
            params_flat.insert(params_flat.end(), p.begin(), p.end());
        }
    }
    return params_flat;
}

template <SupportedFoldType FoldType>
std::vector<uint32_t>
FFAPlan<FoldType>::get_param_counts_flat() const noexcept {
    std::vector<uint32_t> param_counts_flat(this->m_n_levels *
                                            this->m_n_params);
    for (const auto& p : this->m_param_counts) {
        param_counts_flat.insert(param_counts_flat.end(), p.begin(), p.end());
    }
    return param_counts_flat;
}

template <SupportedFoldType FoldType>
void FFAPlan<FoldType>::configure_params() {
    m_params.resize(this->m_n_levels);

    for (SizeType i_level = 0; i_level < this->m_n_levels; ++i_level) {
        m_params[i_level].resize(this->m_n_params);
        for (SizeType iparam = 0; iparam < this->m_n_params; ++iparam) {
            m_params[i_level][iparam] = psr_utils::range_param(
                this->m_cfg.get_param_limits()[iparam][0],
                this->m_cfg.get_param_limits()[iparam][1],
                this->m_dparams[i_level][iparam]);
        }
    }
}

// --- Explicit template instantiations ---
template class FFAPlanMetadata<float>;
template class FFAPlanMetadata<ComplexType>;
template class FFAPlan<float>;
template class FFAPlan<ComplexType>;
} // namespace loki::plans