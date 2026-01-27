#include "loki/algorithms/plans.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/core/circular.hpp"
#include "loki/core/taylor.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils.hpp"

namespace loki::plans {

// --- FFAPlanBase implementation ---
FFAPlanBase::FFAPlanBase(search::PulsarSearchConfig cfg)
    : m_cfg(std::move(cfg)) {
    configure_base_plan();
    validate_base_plan();
}
const search::PulsarSearchConfig& FFAPlanBase::get_config() const noexcept {
    return m_cfg;
}
SizeType FFAPlanBase::get_n_params() const noexcept { return m_n_params; }
SizeType FFAPlanBase::get_n_levels() const noexcept { return m_n_levels; }
std::span<const SizeType> FFAPlanBase::get_segment_lens() const noexcept {
    return m_segment_lens;
}
std::span<const SizeType> FFAPlanBase::get_nsegments() const noexcept {
    return m_nsegments;
}
std::span<const double> FFAPlanBase::get_tsegments() const noexcept {
    return m_tsegments;
}
std::span<const SizeType> FFAPlanBase::get_ncoords() const noexcept {
    return m_ncoords;
}
std::span<const float> FFAPlanBase::get_ncoords_lb() const noexcept {
    return m_ncoords_lb;
}
std::span<const SizeType> FFAPlanBase::get_ncoords_offsets() const noexcept {
    return m_ncoords_offsets;
}
const std::vector<std::vector<SizeType>>&
FFAPlanBase::get_param_counts() const noexcept {
    return m_param_counts;
}
const std::vector<std::vector<SizeType>>&
FFAPlanBase::get_param_cart_strides() const noexcept {
    return m_param_cart_strides;
}
const std::vector<std::vector<double>>&
FFAPlanBase::get_dparams() const noexcept {
    return m_dparams;
}
const std::vector<std::vector<double>>&
FFAPlanBase::get_dparams_lim() const noexcept {
    return m_dparams_lim;
}
SizeType FFAPlanBase::get_coord_size() const noexcept {
    return std::accumulate(m_ncoords.begin(), m_ncoords.end(), 0,
                           std::plus<>());
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

        for (SizeType iparam = 0; iparam < m_n_params; ++iparam) {
            m_param_counts[i_level][iparam] = psr_utils::range_param_count(
                m_cfg.get_param_limits()[iparam][0],
                m_cfg.get_param_limits()[iparam][1], dparam_arr[iparam]);
        }
        m_param_cart_strides[i_level] =
            calculate_strides(m_param_counts[i_level]);
        m_ncoords[i_level] = std::accumulate(
            m_param_counts[i_level].begin(), m_param_counts[i_level].end(), 1UL,
            [](SizeType acc, SizeType param_count) {
                return acc * param_count;
            });
        m_ncoords_lb[i_level] =
            m_ncoords[i_level] > 0
                ? std::log2(static_cast<float>(m_ncoords[i_level]))
                : 0.0F;
        if (i_level == 0) {
            m_ncoords_offsets[i_level] = 0;
        } else {
            m_ncoords_offsets[i_level] =
                m_ncoords_offsets[i_level - 1] + m_ncoords[i_level - 1];
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

// --- FFAPlanMetadata implementation ---
template <SupportedFoldType FoldType>
FFAPlanMetadata<FoldType>::FFAPlanMetadata(search::PulsarSearchConfig cfg)
    : FFAPlanBase(std::move(cfg)) {
    configure_fold_shapes();
    compute_flops();
    validate_metadata();
}
template <SupportedFoldType FoldType>
const std::vector<std::vector<SizeType>>&
FFAPlanMetadata<FoldType>::get_fold_shapes() const noexcept {
    return m_fold_shapes;
}

template <SupportedFoldType FoldType>
const std::vector<std::vector<SizeType>>&
FFAPlanMetadata<FoldType>::get_fold_shapes_time() const noexcept {
    return m_fold_shapes_time;
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
const std::vector<std::vector<std::vector<double>>>&
FFAPlan<FoldType>::get_params() const noexcept {
    return m_params;
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

    // Resolve the params for the FFA plan
    const auto ncoords_max = std::ranges::max(this->m_ncoords);
    std::vector<float> relative_phase_batch(ncoords_max);
    std::vector<uint32_t> pindex_prev_flat(ncoords_max);

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
    SizeType total = 0;
    for (const auto& level : m_params) {
        for (const auto& p : level) {
            total += p.size();
        }
    }
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
std::vector<SizeType>
FFAPlan<FoldType>::get_param_counts_flat() const noexcept {
    std::vector<SizeType> param_counts_flat(this->m_n_levels *
                                            this->m_n_params);
    for (SizeType i_level = 0; i_level < this->m_n_levels; ++i_level) {
        for (SizeType i_param = 0; i_param < this->m_n_params; ++i_param) {
            param_counts_flat[(i_level * this->m_n_params) + i_param] =
                m_params[i_level][i_param].size();
        }
    }
    return param_counts_flat;
}

template <SupportedFoldType FoldType>
std::pair<std::vector<SizeType>, std::vector<SizeType>>
FFAPlan<FoldType>::get_params_flat_sizes() const noexcept {
    std::vector<SizeType> params_flat_offsets(this->m_n_levels);
    std::vector<SizeType> params_flat_sizes(this->m_n_levels);
    for (SizeType i_level = 0; i_level < this->m_n_levels; ++i_level) {
        for (const auto& p : m_params[i_level]) {
            params_flat_sizes[i_level] += p.size();
        }
        if (i_level == 0) {
            params_flat_offsets[i_level] = 0;
        } else {
            params_flat_offsets[i_level] = params_flat_offsets[i_level - 1] +
                                           params_flat_sizes[i_level - 1];
        }
    }
    return {params_flat_offsets, params_flat_sizes};
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

std::vector<coord::FFARegion> generate_ffa_regions(double p_min,
                                                   double p_max,
                                                   double tsamp,
                                                   SizeType nbins_min,
                                                   double eta_min,
                                                   double octave_scale,
                                                   SizeType nbins_max) {
    error_check::check_greater(p_min, 0.0, "p_min must be positive.");
    error_check::check_greater(p_max, p_min, "p_max must be > p_min.");
    error_check::check_greater(tsamp, 0.0, "tsamp must be positive.");
    error_check::check_greater_equal(nbins_min, 2, "nbins_min must be >= 2.");
    error_check::check_greater(eta_min, 0.0, "eta_min must be positive.");
    error_check::check_greater_equal(octave_scale, 1.0,
                                     "octave_scale must be >= 1.0.");
    error_check::check_greater_equal(nbins_max, nbins_min,
                                     "nbins_max must be >= nbins_min.");

    std::vector<coord::FFARegion> regions;
    SizeType nbins_cur =
        std::min(nbins_min, static_cast<SizeType>(p_min / tsamp));
    double rho = eta_min / static_cast<double>(nbins_cur);

    // Core invariant: physical time width of a single folding bin (in
    // seconds). Core invariant: duty-cycle resolution (rho) in bins.
    const double bin_width = p_min / static_cast<double>(nbins_cur);
    auto p_cur_low         = p_min;
    while (p_cur_low < p_max) {
        auto nbins_k = nbins_cur;
        if (nbins_k >= nbins_max) {
            double eta_k = std::round(rho * static_cast<double>(nbins_max));
            regions.push_back({1.0 / p_max, 1.0 / p_cur_low, nbins_max, eta_k});
            break;
        }
        const double p_cur_high = std::min(p_cur_low * octave_scale, p_max);
        double eta_k            = rho * static_cast<double>(nbins_k);
        regions.push_back({1.0 / p_cur_high, 1.0 / p_cur_low, nbins_k, eta_k});
        p_cur_low = p_cur_high;
        nbins_cur = static_cast<SizeType>(p_cur_low / bin_width);
    }
    return regions;
}

template <SupportedFoldType FoldType> struct FFARegionStats<FoldType>::Impl {
    SizeType max_buffer_size{};
    SizeType max_coord_size{};
    SizeType max_ncoords{}; // maximum number of coordinates in the last level
    SizeType n_widths{};
    SizeType n_params{};
    SizeType n_samps{}; // ts_e.size()
    SizeType m_max_passing_candidates{};
    bool use_gpu{};

    Impl(SizeType max_buffer_size,
         SizeType max_coord_size,
         SizeType max_ncoords,
         SizeType n_widths,
         SizeType n_params,
         SizeType n_samps,
         SizeType max_passing_candidates,
         bool use_gpu)
        : max_buffer_size(max_buffer_size),
          max_coord_size(max_coord_size),
          max_ncoords(max_ncoords),
          n_widths(n_widths),
          n_params(n_params),
          n_samps(n_samps),
          m_max_passing_candidates(max_passing_candidates),
          use_gpu(use_gpu) {}

    ~Impl()                      = default;
    Impl(const Impl&)            = default;
    Impl& operator=(const Impl&) = default;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    SizeType get_max_buffer_size_time() const noexcept {
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            return 2 * max_buffer_size;
        } else {
            return max_buffer_size;
        }
    }
    SizeType get_max_scores_size() const noexcept {
        return std::max(max_ncoords * n_widths, m_max_passing_candidates);
    }
    SizeType get_write_param_sets_size() const noexcept {
        return kFFAManagerWriteBatchSize * (n_params + 1); // includes width
    }
    float get_buffer_memory_usage() const noexcept {
        return static_cast<float>(2 * max_buffer_size * sizeof(FoldType)) /
               static_cast<float>(1ULL << 30U);
    }
    float get_coord_memory_usage() const noexcept {
        SizeType coord_size;
        if (n_params == 1) {
            coord_size = max_coord_size * sizeof(coord::FFACoordFreq);
        } else {
            coord_size = max_coord_size * sizeof(coord::FFACoord);
        }
        return static_cast<float>(coord_size) / static_cast<float>(1ULL << 30U);
    }
    float get_extra_memory_usage() const noexcept {
        return static_cast<float>(
                   (get_write_param_sets_size() * sizeof(double)) +
                   (get_max_scores_size() * sizeof(uint32_t)) +
                   (get_max_scores_size() * sizeof(float))) /
               static_cast<float>(1ULL << 30U);
    }
    float get_cpu_memory_usage() const noexcept {
        return get_buffer_memory_usage() + get_coord_memory_usage() +
               get_extra_memory_usage();
    }
    // Use this for GPU memory usage calculation
    float get_device_memory_usage() const noexcept {
        // ts_e_d + ts_v_d + scores_d (widths_d is negligible)
        const float device_extra_gb =
            ((2 * n_samps + get_max_scores_size()) * sizeof(float) +
             (get_max_scores_size() * sizeof(uint32_t))) /
            static_cast<float>(1ULL << 30U);
        // m_fold_d_time
        const float fold_d_time_gb = get_max_buffer_size_time() *
                                     sizeof(float) /
                                     static_cast<float>(1ULL << 30U);
        return device_extra_gb + fold_d_time_gb + get_buffer_memory_usage() +
               get_coord_memory_usage();
    }
    float get_manager_memory_usage() const noexcept {
        if (use_gpu) {
            return get_device_memory_usage();
        }
        return get_cpu_memory_usage();
    }
}; // End FFARegionStats::Impl definition

template <SupportedFoldType FoldType> class FFARegionPlanner<FoldType>::Impl {
public:
    explicit Impl(search::PulsarSearchConfig cfg, bool use_gpu)
        : m_base_cfg(std::move(cfg)),
          m_use_gpu(use_gpu) {
        plan_regions();
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const std::vector<search::PulsarSearchConfig>& get_cfgs() const noexcept {
        return m_cfgs;
    }
    SizeType get_nregions() const noexcept { return m_cfgs.size(); }
    const FFARegionStats<FoldType>& get_stats() const noexcept {
        return m_stats;
    }
    const std::vector<coord::FFAChunkStats>& get_chunk_stats() const noexcept {
        return m_chunk_stats;
    }

private:
    search::PulsarSearchConfig m_base_cfg;
    bool m_use_gpu;

    std::vector<search::PulsarSearchConfig> m_cfgs;
    FFARegionStats<FoldType> m_stats{0, 0, 0, 0, 0, 0, 0, m_use_gpu};
    std::vector<coord::FFAChunkStats> m_chunk_stats;

    double calculate_max_drift(const search::PulsarSearchConfig& cfg) const {
        if (cfg.get_nparams() <= 1) {
            return 0.0;
        }

        const auto& param_limits = cfg.get_param_limits();
        // Drift from center to edge
        const auto t_half = cfg.get_tobs() / 2.0;
        if (cfg.get_nparams() == 2) {
            const auto max_accel = std::max(std::abs(param_limits[0][0]),
                                            std::abs(param_limits[0][1]));
            const auto drift     = max_accel * t_half;
            return drift / utils::kCval;
        }
        if (cfg.get_nparams() == 3) {
            const auto max_jerk  = std::max(std::abs(param_limits[0][0]),
                                            std::abs(param_limits[0][1]));
            const auto max_accel = std::max(std::abs(param_limits[1][0]),
                                            std::abs(param_limits[1][1]));
            const auto drift =
                (max_accel * t_half) + (max_jerk * t_half * t_half / 2.0);
            return drift / utils::kCval;
        }
        throw std::runtime_error(
            "Unsupported number of parameters for drift calculation");
    }

    void plan_regions() {
        // Step 1: Generate FFA regions based on frequency-dependent
        // nbins/eta
        const double f_min     = m_base_cfg.get_f_min();
        const double f_max     = m_base_cfg.get_f_max();
        const double p_min     = 1.0 / f_max;
        const double p_max     = 1.0 / f_min;
        const auto ffa_regions = plans::generate_ffa_regions(
            p_min, p_max, m_base_cfg.get_tsamp(), m_base_cfg.get_nbins(),
            m_base_cfg.get_eta(), m_base_cfg.get_octave_scale(),
            m_base_cfg.get_nbins_max());
        // Print region info
        spdlog::info("FFARegionPlanner - planned regions:");
        for (const auto& region : ffa_regions) {
            spdlog::info("Region: f=[{:08.3f}, {:08.3f}] Hz, nbins={:04d}, "
                         "eta={:04.1f}",
                         region.f_start, region.f_end, region.nbins,
                         region.eta);
        }

        // Step 2: For each region, subdivide in frequency if it doesn't fit
        // in memory
        const auto& base_param_limits = m_base_cfg.get_param_limits();
        const auto max_drift          = calculate_max_drift(m_base_cfg);
        // Log drift information
        if (max_drift > 0 && m_base_cfg.get_nparams() > 1) {
            spdlog::info("Drift-aware chunking: max_drift={:.6f} ({:.4f}%) "
                         "for tobs={:.1f}s, n_params={}",
                         max_drift, max_drift * 100.0, m_base_cfg.get_tobs(),
                         m_base_cfg.get_nparams());
        }

        SizeType max_buffer_size{};
        SizeType max_coord_size{};
        SizeType max_ncoords{};
        for (const auto& region : ffa_regions) {
            // Create a config for this region
            auto region_param_limits   = base_param_limits;
            region_param_limits.back() = {region.f_start, region.f_end};
            subdivide_region_by_memory(region_param_limits, region.nbins,
                                       region.eta, max_drift, max_buffer_size,
                                       max_coord_size, max_ncoords);
        }
        m_stats = FFARegionStats<FoldType>(
            max_buffer_size, max_coord_size, max_ncoords,
            m_base_cfg.get_n_scoring_widths(), m_base_cfg.get_nparams(),
            m_base_cfg.get_nsamps(), m_base_cfg.get_max_passing_candidates(),
            m_use_gpu);

        // Log summary statistics
        log_planning_summary();
    }

    void
    subdivide_region_by_memory(const std::vector<ParamLimitType>& param_limits,
                               SizeType nbins,
                               double eta,
                               double max_drift,
                               SizeType& max_buffer_size,
                               SizeType& max_coord_size,
                               SizeType& max_ncoords) {
        // For GPU, this is the device memory limit. For CPU, this is the
        // process memory limit.
        const auto max_memory_gb = m_base_cfg.get_max_process_memory_gb();
        // Reserve some headroom for OS, Python interpreter, and rounding
        // errors
        constexpr double kSafetyMarginGB = 0.5; // 500 MB safety margin
        const auto effective_limit_gb    = max_memory_gb - kSafetyMarginGB;

        const auto [f_start, f_end] = param_limits.back();

        constexpr double kMinViableRange  = 0.1; // Minimum possible range in Hz
        constexpr double kMinChunkSize    = 0.01; // Merge threshold (Hz)
        constexpr double kSearchTolerance = 0.01; // Binary search stop (Hz)
        constexpr SizeType kMaxProbes     = 20;   // Avoid infinite loops

        // Pre-flight check: can we fit minimum viable chunk at worst case
        // (high freq)?
        {
            auto min_check_limits   = param_limits;
            min_check_limits.back() = {(f_end - kMinViableRange) *
                                           (1.0 - max_drift),
                                       f_end * (1.0 + max_drift)};
            auto check_cfg =
                m_base_cfg.get_updated_config(nbins, eta, min_check_limits);
            FFAPlanMetadata<FoldType> check_plan(std::move(check_cfg));

            FFARegionStats<FoldType> min_stats{
                check_plan.get_buffer_size(),
                check_plan.get_coord_size(),
                check_plan.get_ncoords().back(),
                m_base_cfg.get_n_scoring_widths(),
                m_base_cfg.get_nparams(),
                m_base_cfg.get_nsamps(),
                m_base_cfg.get_max_passing_candidates(),
                m_use_gpu};

            if (min_stats.get_manager_memory_usage() > effective_limit_gb) {
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: Cannot fit minimum viable chunk at "
                    "highest frequency.\n"
                    "  Nominal range: {:.3f} Hz, with drift: {:.3f} Hz\n"
                    "  Required memory: {:.2f} GB, Available: {:.2f} GB\n"
                    "  Suggestion: Increase max_memory_gb or reduce "
                    "parameter "
                    "searchranges.",
                    kMinViableRange,
                    min_check_limits.back()[1] - min_check_limits.back()[0],
                    min_stats.get_manager_memory_usage(), effective_limit_gb));
            }
        }

        // Reverse iteration: go from high frequency to low (f_end →
        // f_start)
        double current_f_end = f_end;
        while (current_f_end > f_start) {
            // Early exit for tiny remaining ranges
            const double range_width = current_f_end - f_start;
            if (range_width < kMinViableRange) {
                spdlog::debug(
                    "Skipping remaining range [{:08.3f}, {:08.3f}] Hz "
                    "(only {:.3f} Hz wide, below minimum {:.3f} Hz)",
                    f_start, current_f_end, range_width, kMinViableRange);
                break;
            }
            // Binary search for largest chunk that fits (working backwards
            // from current_f_end)
            double f_low        = f_start;
            double f_high       = current_f_end;
            double best_f_start = current_f_end; // Start from the top

            // Start with a Heuristic probe (10% of remaining range or 1 Hz)
            double f_probe = std::max(
                current_f_end - std::max(0.1 * range_width, 1.0), f_start);
            SizeType probe_count = 0;

            while (probe_count < kMaxProbes &&
                   (f_high - f_low) > kSearchTolerance) {
                // Create test config for this chunk
                auto test_param_limits   = param_limits;
                test_param_limits.back() = {f_probe * (1.0 - max_drift),
                                            current_f_end * (1.0 + max_drift)};
                // Simulate the chunk
                auto test_cfg = m_base_cfg.get_updated_config(
                    nbins, eta, test_param_limits);
                FFAPlanMetadata<FoldType> test_plan(std::move(test_cfg));
                FFARegionStats<FoldType> sim_stats{
                    std::max(max_buffer_size, test_plan.get_buffer_size()),
                    std::max(max_coord_size, test_plan.get_coord_size()),
                    std::max(max_ncoords, test_plan.get_ncoords().back()),
                    m_base_cfg.get_n_scoring_widths(),
                    m_base_cfg.get_nparams(),
                    m_base_cfg.get_nsamps(),
                    m_base_cfg.get_max_passing_candidates(),
                    m_use_gpu};
                if (sim_stats.get_manager_memory_usage() <=
                    effective_limit_gb) {
                    // Fits! Try to include more (go lower in frequency)
                    best_f_start = f_probe;
                    f_high       = f_probe;

                    if (f_probe <= f_start + kSearchTolerance) {
                        best_f_start = f_start; // Close enough, use f_start
                        break;
                    }
                    // Probe lower
                    f_probe = std::max(std::midpoint(f_low, f_high), f_start);
                } else {
                    // Doesn't fit, need smaller chunk (go higher in
                    // frequency)
                    f_low   = f_probe;
                    f_probe = std::midpoint(f_low, f_high);
                }

                probe_count++;
            }
            // Check if remaining range is too small to warrant separate
            // chunk
            const double remaining_range = best_f_start - f_start;
            if (remaining_range > 0 && remaining_range < kMinChunkSize &&
                best_f_start < current_f_end) {
                // Extend current chunk to include small remainder
                best_f_start = f_start;
            }
            if (best_f_start >= current_f_end) {
                throw std::runtime_error(std::format(
                    "FFARegionPlanner: Cannot fit any chunk in range "
                    "[{:08.3f}, {:08.3f}] Hz with nbins={} into memory "
                    "limit "
                    "{:.2f} GB.\n"
                    "  Suggestion: Increase max_memory_gb or reduce "
                    "parameter "
                    "searchranges.",
                    f_start, current_f_end, nbins, max_memory_gb));
            }

            // Finalize this chunk
            const double nominal_start = best_f_start;
            const double nominal_end   = current_f_end;
            const double nominal_width = nominal_end - nominal_start;
            const double actual_start  = nominal_start * (1.0 - max_drift);
            const double actual_end    = nominal_end * (1.0 + max_drift);
            const double actual_width  = actual_end - actual_start;
            const double overlap_fraction =
                (actual_width - nominal_width) / actual_width;

            auto chunk_param_limits   = param_limits;
            chunk_param_limits.back() = {actual_start, actual_end};
            auto chunk_cfg =
                m_base_cfg.get_updated_config(nbins, eta, chunk_param_limits);
            FFAPlanMetadata<FoldType> chunk_plan(std::move(chunk_cfg));
            const double chunk_memory_gb =
                chunk_plan.get_buffer_memory_usage() +
                chunk_plan.get_coord_memory_usage();

            // Store config and stats
            m_cfgs.push_back(chunk_cfg);
            m_chunk_stats.push_back(
                coord::FFAChunkStats{.nominal_f_start  = nominal_start,
                                     .nominal_f_end    = nominal_end,
                                     .actual_f_start   = actual_start,
                                     .actual_f_end     = actual_end,
                                     .nominal_width    = nominal_width,
                                     .actual_width     = actual_width,
                                     .total_memory_gb  = chunk_memory_gb,
                                     .overlap_fraction = overlap_fraction});
            spdlog::debug("Chunk: nominal=[{:08.3f}, {:08.3f}] ({:.3f} Hz), "
                          "actual=[{:08.3f}, {:08.3f}] ({:.3f} Hz), "
                          "overlap={:.1f}%, mem={:.2f} GB",
                          nominal_start, nominal_end, nominal_width,
                          actual_start, actual_end, actual_width,
                          overlap_fraction * 100.0, chunk_memory_gb);
            max_buffer_size =
                std::max(max_buffer_size, chunk_plan.get_buffer_size());
            max_coord_size =
                std::max(max_coord_size, chunk_plan.get_coord_size());
            max_ncoords =
                std::max(max_ncoords, chunk_plan.get_ncoords().back());

            // Move to next chunk (going backwards in frequency)
            current_f_end = best_f_start;
        }
    }

    void log_planning_summary() const {
        if (m_chunk_stats.empty()) {
            return;
        }

        // Compute aggregate statistics
        double total_nominal_width = 0.0;
        double total_actual_width  = 0.0;
        double max_memory          = 0.0;
        double avg_memory          = 0.0;
        double max_overlap_pct     = 0.0;
        double avg_overlap_pct     = 0.0;

        for (const auto& stat : m_chunk_stats) {
            total_nominal_width += stat.nominal_width;
            total_actual_width += stat.actual_width;
            max_memory = std::max(max_memory, stat.total_memory_gb);
            avg_memory += stat.total_memory_gb;
            max_overlap_pct =
                std::max(max_overlap_pct, stat.overlap_fraction * 100.0);
            avg_overlap_pct += stat.overlap_fraction * 100.0;
        }
        avg_memory /= static_cast<float>(m_chunk_stats.size());
        avg_overlap_pct /= static_cast<double>(m_chunk_stats.size());

        const double redundancy_factor =
            total_actual_width / total_nominal_width;
        if (m_use_gpu) {
            spdlog::info("FFARegionPlanner - GPU Summary:");
        } else {
            spdlog::info("FFARegionPlanner - CPU Summary:");
        }
        spdlog::info("  Total chunks: {}", m_chunk_stats.size());
        spdlog::info(
            "  Nominal coverage: {:.3f} Hz, Actual coverage: {:.3f} Hz",
            total_nominal_width, total_actual_width);
        spdlog::info("  Redundancy factor: {:.3f}x ({:.1f}% extra computation)",
                     redundancy_factor, (redundancy_factor - 1.0) * 100.0);
        spdlog::info("  Overlap: avg={:.1f}%, max={:.1f}%", avg_overlap_pct,
                     max_overlap_pct);
        spdlog::info("  Memory per chunk: avg={:.2f} GB, max={:.2f} GB",
                     avg_memory, max_memory);
        spdlog::info("  Manager allocated: {:.2f} GB (limit: {:.2f} GB)",
                     m_stats.get_manager_memory_usage(),
                     m_base_cfg.get_max_process_memory_gb());
    }

}; // End FFARegionPlanner::Impl definition

// --- Definitions for FFARegionStats ---
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::FFARegionStats(SizeType max_buffer_size,
                                         SizeType max_coord_size,
                                         SizeType max_ncoords,
                                         SizeType n_widths,
                                         SizeType n_params,
                                         SizeType n_samps,
                                         SizeType max_passing_candidates,
                                         bool use_gpu)
    : m_impl(std::make_unique<Impl>(max_buffer_size,
                                    max_coord_size,
                                    max_ncoords,
                                    n_widths,
                                    n_params,
                                    n_samps,
                                    max_passing_candidates,
                                    use_gpu)) {}
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::~FFARegionStats() = default;
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::FFARegionStats(FFARegionStats&&) noexcept = default;
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>&
FFARegionStats<FoldType>::operator=(FFARegionStats&&) noexcept = default;
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>::FFARegionStats(const FFARegionStats& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}
template <SupportedFoldType FoldType>
FFARegionStats<FoldType>&
FFARegionStats<FoldType>::operator=(const FFARegionStats& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_buffer_size() const noexcept {
    return m_impl->max_buffer_size;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_coord_size() const noexcept {
    return m_impl->max_coord_size;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_ncoords() const noexcept {
    return m_impl->max_ncoords;
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_buffer_size_time() const noexcept {
    return m_impl->get_max_buffer_size_time();
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_max_scores_size() const noexcept {
    return m_impl->get_max_scores_size();
}
template <SupportedFoldType FoldType>
SizeType FFARegionStats<FoldType>::get_write_param_sets_size() const noexcept {
    return m_impl->get_write_param_sets_size();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_buffer_memory_usage() const noexcept {
    return m_impl->get_buffer_memory_usage();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_coord_memory_usage() const noexcept {
    return m_impl->get_coord_memory_usage();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_extra_memory_usage() const noexcept {
    return m_impl->get_extra_memory_usage();
}
template <SupportedFoldType FoldType>
float FFARegionStats<FoldType>::get_manager_memory_usage() const noexcept {
    return m_impl->get_manager_memory_usage();
}

// --- Definitions for FFARegionPlanner ---
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>::FFARegionPlanner(
    const search::PulsarSearchConfig& cfg, bool use_gpu)
    : m_impl(std::make_unique<Impl>(cfg, use_gpu)) {}
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>::~FFARegionPlanner() = default;
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>::FFARegionPlanner(FFARegionPlanner&&) noexcept =
    default;
template <SupportedFoldType FoldType>
FFARegionPlanner<FoldType>& FFARegionPlanner<FoldType>::operator=(
    FFARegionPlanner<FoldType>&&) noexcept = default;
template <SupportedFoldType FoldType>
const std::vector<search::PulsarSearchConfig>&
FFARegionPlanner<FoldType>::get_cfgs() const noexcept {
    return m_impl->get_cfgs();
}
template <SupportedFoldType FoldType>
SizeType FFARegionPlanner<FoldType>::get_nregions() const noexcept {
    return m_impl->get_nregions();
}
template <SupportedFoldType FoldType>
const FFARegionStats<FoldType>&
FFARegionPlanner<FoldType>::get_stats() const noexcept {
    return m_impl->get_stats();
}

// --- Explicit template instantiations ---
template class FFAPlanMetadata<float>;
template class FFAPlanMetadata<ComplexType>;
template class FFAPlan<float>;
template class FFAPlan<ComplexType>;
template class FFARegionStats<float>;
template class FFARegionStats<ComplexType>;
template class FFARegionPlanner<float>;
template class FFARegionPlanner<ComplexType>;
} // namespace loki::plans