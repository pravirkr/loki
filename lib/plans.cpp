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

struct FFAPlanBase::Impl {
    SizeType n_params{}; // # of parameters to search over (..., a, f)
    SizeType n_levels{}; // # of FFA merge levels
    std::vector<SizeType> segment_lens;      // Segment lengths
    std::vector<SizeType> nsegments;         // # of segments
    std::vector<double> tsegments;           // Segment lengths in seconds
    std::vector<SizeType> ncoords;           // # of coordinates
    std::vector<float> ncoords_lb;           // Log2 of # of coordinates
    std::vector<uint32_t> ncoords_offsets;   // Offset # of coordinates
    std::vector<uint32_t> param_counts_flat; // Flattened parameter counts
    std::vector<std::vector<SizeType>> param_counts; // param count per level
    std::vector<std::vector<SizeType>> param_cart_strides; // Cartesian strides
    std::vector<std::vector<double>> dparams;              // Grid step sizes
    std::vector<std::vector<double>> dparams_lim; // Grid step size (limits)

    explicit Impl(search::PulsarSearchConfig cfg) : m_cfg(std::move(cfg)) {
        configure_plan();
        validate_plan();
    }

    ~Impl()                                = default;
    Impl(Impl&& other) noexcept            = default;
    Impl& operator=(Impl&& other) noexcept = default;
    Impl(const Impl& other)                = default;
    Impl& operator=(const Impl& other)     = default;

    const search::PulsarSearchConfig& get_config() const noexcept {
        return m_cfg;
    }

    SizeType get_coord_size() const noexcept {
        return std::accumulate(ncoords.begin(), ncoords.end(), 0,
                               std::plus<>());
    }

    float get_coord_memory_usage() const noexcept {
        SizeType total_memory;
        if (m_cfg.get_nparams() == 1) {
            total_memory = get_coord_size() * sizeof(coord::FFACoordFreq);
        } else {
            total_memory = get_coord_size() * sizeof(coord::FFACoord);
        }
        return static_cast<float>(total_memory) /
               static_cast<float>(1ULL << 30U);
    }

    void resolve_coordinates(std::span<coord::FFACoord> coords) {
        error_check::check_greater_equal(
            n_params, 2U,
            "resolve_coordinates only supports nparams>=2. For frequency "
            "coordinates, use resolve_coordinates_freq() instead.");
        error_check::check_less_equal(n_params, 5U,
                                      "resolve_coordinates only supports "
                                      "nparams<=5. Larger values are "
                                      "not supported and advised.");
        error_check::check_greater_equal(coords.size(), get_coord_size(),
                                         "FFAPlan::resolve_coordinates: coords "
                                         "must have size >= coord_size");

        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto ncoords_cur    = ncoords[i_level];
            const auto ncoords_offset = ncoords_offsets[i_level];
            auto coords_span = coords.subspan(ncoords_offset, ncoords_cur);

            // Tail coordinates
            core::ffa_taylor_resolve_poly_batch(
                param_counts[i_level], param_counts[i_level - 1],
                m_cfg.get_param_limits(), coords_span, i_level, 0,
                m_cfg.get_tseg_brute(), m_cfg.get_nbins(), n_params);

            // Head coordinates
            core::ffa_taylor_resolve_poly_batch(
                param_counts[i_level], param_counts[i_level - 1],
                m_cfg.get_param_limits(), coords_span, i_level, 1,
                m_cfg.get_tseg_brute(), m_cfg.get_nbins(), n_params);
        }
    }
    std::vector<std::vector<coord::FFACoord>> resolve_coordinates() {
        std::vector<coord::FFACoord> coords_flat(get_coord_size());
        resolve_coordinates(coords_flat);
        std::vector<std::vector<coord::FFACoord>> coords(n_levels);
        for (SizeType i_level = 0; i_level < n_levels; ++i_level) {
            const auto ncoords_cur    = ncoords[i_level];
            const auto ncoords_offset = ncoords_offsets[i_level];
            coords[i_level].resize(ncoords_cur);
            std::copy(
                coords_flat.begin() + static_cast<IndexType>(ncoords_offset),
                coords_flat.begin() + static_cast<IndexType>(ncoords_offset) +
                    static_cast<IndexType>(ncoords_cur),
                coords[i_level].begin());
        }
        return coords;
    }

    void resolve_coordinates_freq(std::span<coord::FFACoordFreq> coords_freq) {
        error_check::check_equal(n_params, 1,
                                 "resolve_coordinates_freq() only supports "
                                 "nparams=1");
        // Resolve the frequency coordinates for the FFA plan
        error_check::check_greater_equal(
            coords_freq.size(), get_coord_size(),
            "FFAPlan::resolve_coordinates_freq: "
            "coords_freq must have size >= coord_size");

        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto ncoords_cur    = ncoords[i_level];
            const auto ncoords_offset = ncoords_offsets[i_level];
            auto coords_freq_span =
                coords_freq.subspan(ncoords_offset, ncoords_cur);
            core::ffa_taylor_resolve_freq_batch(
                param_counts[i_level][0], param_counts[i_level - 1][0],
                m_cfg.get_param_limits()[0], coords_freq_span, i_level,
                m_cfg.get_tseg_brute(), m_cfg.get_nbins());
        }
    }

    std::vector<std::vector<coord::FFACoordFreq>> resolve_coordinates_freq() {
        std::vector<coord::FFACoordFreq> coords_freq_flat(get_coord_size());
        resolve_coordinates_freq(coords_freq_flat);
        std::vector<std::vector<coord::FFACoordFreq>> coords_freq(n_levels);
        for (SizeType i_level = 0; i_level < n_levels; ++i_level) {
            const auto ncoords_cur    = ncoords[i_level];
            const auto ncoords_offset = ncoords_offsets[i_level];
            coords_freq[i_level].resize(ncoords_cur);
            std::copy(coords_freq_flat.begin() +
                          static_cast<IndexType>(ncoords_offset),
                      coords_freq_flat.begin() +
                          static_cast<IndexType>(ncoords_offset) +
                          static_cast<IndexType>(ncoords_cur),
                      coords_freq[i_level].begin());
        }
        return coords_freq;
    }

    std::vector<std::vector<double>>
    compute_param_grid(SizeType ffa_level) const noexcept {
        std::vector<std::vector<double>> param_grid(n_params);
        for (SizeType iparam = 0; iparam < n_params; ++iparam) {
            param_grid[iparam] =
                psr_utils::range_param(m_cfg.get_param_limits()[iparam].min,
                                       m_cfg.get_param_limits()[iparam].max,
                                       dparams[ffa_level][iparam]);
        }
        return param_grid;
    }

    std::vector<std::vector<std::vector<double>>>
    compute_param_grid_full() const noexcept {
        std::vector<std::vector<std::vector<double>>> param_grid(n_levels);
        for (SizeType i_level = 0; i_level < n_levels; ++i_level) {
            param_grid[i_level].resize(n_params);
            for (SizeType iparam = 0; iparam < n_params; ++iparam) {
                param_grid[i_level][iparam] =
                    psr_utils::range_param(m_cfg.get_param_limits()[iparam].min,
                                           m_cfg.get_param_limits()[iparam].max,
                                           dparams[i_level][iparam]);
            }
        }
        return param_grid;
    }

    std::map<std::string, std::vector<double>> get_params_dict() const {
        const auto param_names = m_cfg.get_param_names();
        std::map<std::string, std::vector<double>> result;
        for (SizeType iparam = 0; iparam < param_names.size(); ++iparam) {
            result[param_names[iparam]] =
                psr_utils::range_param(m_cfg.get_param_limits()[iparam].min,
                                       m_cfg.get_param_limits()[iparam].max,
                                       dparams[n_levels - 1][iparam]);
        }
        return result;
    }

    std::vector<double>
    get_branching_pattern_approx(std::string_view poly_basis,
                                 SizeType ref_seg,
                                 IndexType isuggest) const {
        const auto param_arr = compute_param_grid(n_levels - 1);
        if (poly_basis == "taylor") {
            return core::generate_bp_poly_taylor_approx(
                param_arr, dparams_lim[n_levels - 1], m_cfg.get_param_limits(),
                m_cfg.get_tseg_ffa(), nsegments[n_levels - 1],
                m_cfg.get_nbins(), m_cfg.get_eta(), ref_seg, isuggest,
                m_cfg.get_use_conservative_tile());
        }
        throw std::invalid_argument(std::format(
            "Invalid poly_basis ({}) for branching pattern generation",
            poly_basis));
    }

    std::vector<double> get_branching_pattern(std::string_view poly_basis,
                                              SizeType ref_seg) const {
        const auto param_arr = compute_param_grid(n_levels - 1);
        if (poly_basis == "taylor") {
            if (n_params <= 4) {
                return core::generate_bp_poly_taylor(
                    param_arr, dparams_lim[n_levels - 1],
                    m_cfg.get_param_limits(), m_cfg.get_tseg_ffa(),
                    nsegments[n_levels - 1], m_cfg.get_nbins(), m_cfg.get_eta(),
                    ref_seg, m_cfg.get_use_conservative_tile());
            }
            if (n_params == 5) {
                return core::generate_bp_circ_taylor(
                    param_arr, dparams_lim[n_levels - 1],
                    m_cfg.get_param_limits(), m_cfg.get_tseg_ffa(),
                    nsegments[n_levels - 1], m_cfg.get_nbins(), m_cfg.get_eta(),
                    ref_seg, m_cfg.get_p_orb_min(),
                    m_cfg.get_minimum_snap_cells(),
                    m_cfg.get_use_conservative_tile());
            }
            throw std::invalid_argument("nparams > 5 not supported for "
                                        "branching pattern generation");
        }
        throw std::invalid_argument(std::format(
            "Invalid poly_basis ({}) for branching pattern generation",
            poly_basis));
    }

private:
    search::PulsarSearchConfig m_cfg;

    void configure_plan() {
        const auto levels = m_cfg.get_niters_ffa() + 1;
        n_params          = m_cfg.get_nparams();
        n_levels          = levels;
        segment_lens.resize(levels);
        nsegments.resize(levels);
        tsegments.resize(levels);
        ncoords.resize(levels);
        ncoords_lb.resize(levels);
        ncoords_offsets.resize(levels + 1); // Include final sentinel
        param_counts_flat.resize(levels * n_params);
        param_counts.resize(levels);
        param_cart_strides.resize(levels);
        dparams.resize(levels);
        dparams_lim.resize(levels);

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
            param_counts[i_level].resize(n_params);

            SizeType ncoords_level = 1;
            for (SizeType iparam = 0; iparam < n_params; ++iparam) {
                const auto param_size_cur = psr_utils::range_param_count(
                    m_cfg.get_param_limits()[iparam].min,
                    m_cfg.get_param_limits()[iparam].max, dparam_arr[iparam]);
                param_counts[i_level][iparam] = param_size_cur;
                ncoords_level *= param_size_cur;
                param_counts_flat[(i_level * n_params) + iparam] =
                    param_size_cur;
            }
            param_cart_strides[i_level] =
                calculate_strides(param_counts[i_level]);
            ncoords[i_level] = ncoords_level;
            ncoords_lb[i_level] =
                ncoords[i_level] > 0
                    ? std::log2(static_cast<float>(ncoords[i_level]))
                    : 0.0F;
            if (i_level == 0) {
                ncoords_offsets[i_level] = 0;
            } else {
                ncoords_offsets[i_level] =
                    ncoords_offsets[i_level - 1] + ncoords[i_level - 1];
            }
        }
        ncoords_offsets[levels] =
            ncoords_offsets[levels - 1] + ncoords[levels - 1];
    }

    void validate_plan() const {
        // For the first level, only the freqs array should have size > 1
        for (SizeType iparam = 0; iparam < n_params - 1; ++iparam) {
            if (param_counts[0][iparam] != 1) {
                throw std::runtime_error(
                    "FFAPlan::validate_plan: Only one parameter for higher "
                    "order derivatives is supported for the initialization "
                    "for "
                    "the first level");
            }
        }
    }
    static std::vector<SizeType>
    calculate_strides(std::span<const SizeType> p_arr_counts) {
        const auto nparams = p_arr_counts.size();
        std::vector<SizeType> strides(nparams);
        strides[nparams - 1] = 1;
        for (int i = static_cast<int>(nparams) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * p_arr_counts[i + 1];
        }
        return strides;
    }
}; // End FFAPlanBase::Impl definition

// --- Definitions for FFAPlanBase ---
FFAPlanBase::FFAPlanBase(const search::PulsarSearchConfig& cfg)
    : m_impl(std::make_unique<Impl>(cfg)) {}
FFAPlanBase::~FFAPlanBase()                                 = default;
FFAPlanBase::FFAPlanBase(FFAPlanBase&&) noexcept            = default;
FFAPlanBase& FFAPlanBase::operator=(FFAPlanBase&&) noexcept = default;

const search::PulsarSearchConfig& FFAPlanBase::get_config() const noexcept {
    return m_impl->get_config();
}
SizeType FFAPlanBase::get_n_params() const noexcept { return m_impl->n_params; }
SizeType FFAPlanBase::get_n_levels() const noexcept { return m_impl->n_levels; }
std::span<const SizeType> FFAPlanBase::get_segment_lens() const noexcept {
    return m_impl->segment_lens;
}
std::span<const SizeType> FFAPlanBase::get_nsegments() const noexcept {
    return m_impl->nsegments;
}
std::span<const double> FFAPlanBase::get_tsegments() const noexcept {
    return m_impl->tsegments;
}
std::span<const SizeType> FFAPlanBase::get_ncoords() const noexcept {
    return m_impl->ncoords;
}
std::span<const float> FFAPlanBase::get_ncoords_lb() const noexcept {
    return m_impl->ncoords_lb;
}
std::span<const uint32_t> FFAPlanBase::get_ncoords_offsets() const noexcept {
    return m_impl->ncoords_offsets;
}
std::span<const uint32_t> FFAPlanBase::get_param_counts_flat() const noexcept {
    return m_impl->param_counts_flat;
}
const std::vector<std::vector<SizeType>>&
FFAPlanBase::get_param_counts() const noexcept {
    return m_impl->param_counts;
}
const std::vector<std::vector<SizeType>>&
FFAPlanBase::get_param_cart_strides() const noexcept {
    return m_impl->param_cart_strides;
}
const std::vector<std::vector<double>>&
FFAPlanBase::get_dparams() const noexcept {
    return m_impl->dparams;
}
const std::vector<std::vector<double>>&
FFAPlanBase::get_dparams_lim() const noexcept {
    return m_impl->dparams_lim;
}
SizeType FFAPlanBase::get_coord_size() const noexcept {
    return m_impl->get_coord_size();
}
float FFAPlanBase::get_coord_memory_usage() const noexcept {
    return m_impl->get_coord_memory_usage();
}
void FFAPlanBase::resolve_coordinates(std::span<coord::FFACoord> coords) {
    m_impl->resolve_coordinates(coords);
}
std::vector<std::vector<coord::FFACoord>> FFAPlanBase::resolve_coordinates() {
    return m_impl->resolve_coordinates();
}
void FFAPlanBase::resolve_coordinates_freq(
    std::span<coord::FFACoordFreq> coords_freq) {
    m_impl->resolve_coordinates_freq(coords_freq);
}
std::vector<std::vector<coord::FFACoordFreq>>
FFAPlanBase::resolve_coordinates_freq() {
    return m_impl->resolve_coordinates_freq();
}
std::vector<std::vector<double>>
FFAPlanBase::compute_param_grid(SizeType ffa_level) const noexcept {
    return m_impl->compute_param_grid(ffa_level);
}
std::vector<std::vector<std::vector<double>>>
FFAPlanBase::compute_param_grid_full() const noexcept {
    return m_impl->compute_param_grid_full();
}
std::map<std::string, std::vector<double>>
FFAPlanBase::get_params_dict() const {
    return m_impl->get_params_dict();
}
std::vector<double> FFAPlanBase::get_branching_pattern_approx(
    std::string_view poly_basis, SizeType ref_seg, IndexType isuggest) const {
    return m_impl->get_branching_pattern_approx(poly_basis, ref_seg, isuggest);
}
std::vector<double>
FFAPlanBase::get_branching_pattern(std::string_view poly_basis,
                                   SizeType ref_seg) const {
    return m_impl->get_branching_pattern(poly_basis, ref_seg);
}

// --- Implementation for FFAPlan ---
template <SupportedFoldType FoldType>
FFAPlan<FoldType>::FFAPlan(const search::PulsarSearchConfig& cfg)
    : FFAPlanBase(cfg) {
    configure_fold_shapes();
    compute_flops();
    validate();
}
template <SupportedFoldType FoldType>
SizeType FFAPlan<FoldType>::get_brute_fold_size() const noexcept {
    return std::accumulate(m_fold_shapes.front().begin(),
                           m_fold_shapes.front().end(), SizeType{1},
                           std::multiplies<>());
}
template <SupportedFoldType FoldType>
SizeType FFAPlan<FoldType>::get_fold_size() const noexcept {
    return std::accumulate(m_fold_shapes.back().begin(),
                           m_fold_shapes.back().end(), SizeType{1},
                           std::multiplies<>());
}
template <SupportedFoldType FoldType>
SizeType FFAPlan<FoldType>::get_fold_size_time() const noexcept {
    return std::accumulate(m_fold_shapes_time.back().begin(),
                           m_fold_shapes_time.back().end(), SizeType{1},
                           std::multiplies<>());
}
template <SupportedFoldType FoldType>
SizeType FFAPlan<FoldType>::get_buffer_size() const noexcept {
    return std::ranges::max(
        m_fold_shapes | std::views::transform([](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end(), SizeType{1},
                                   std::multiplies<>());
        }));
}
template <SupportedFoldType FoldType>
SizeType FFAPlan<FoldType>::get_buffer_size_time() const noexcept {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        return 2 * get_buffer_size();
    } else {
        return get_buffer_size();
    }
}
template <SupportedFoldType FoldType>
float FFAPlan<FoldType>::get_buffer_memory_usage() const noexcept {
    const auto total_memory = 2 * get_buffer_size() * sizeof(FoldType);
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}
template <SupportedFoldType FoldType>
float FFAPlan<FoldType>::get_gflops(bool return_in_time) const noexcept {
    return (return_in_time ? m_flops_required_return_in_time
                           : m_flops_required) *
           1.0e-9F;
}

template <SupportedFoldType FoldType>
void FFAPlan<FoldType>::configure_fold_shapes() noexcept {
    const auto n_levels = get_n_levels();
    const auto n_params = get_n_params();
    const auto& cfg     = get_config();
    m_fold_shapes.resize(n_levels);
    m_fold_shapes_time.resize(n_levels);
    for (SizeType i_level = 0; i_level < n_levels; ++i_level) {
        m_fold_shapes[i_level].resize(n_params + 3);
        m_fold_shapes_time[i_level].resize(n_params + 3);
        m_fold_shapes[i_level][0]      = get_nsegments()[i_level];
        m_fold_shapes_time[i_level][0] = get_nsegments()[i_level];
        for (SizeType iparam = 0; iparam < n_params; ++iparam) {
            m_fold_shapes[i_level][iparam + 1] =
                get_param_counts()[i_level][iparam];
            m_fold_shapes_time[i_level][iparam + 1] =
                get_param_counts()[i_level][iparam];
        }
        m_fold_shapes[i_level][n_params + 1]      = 2;
        m_fold_shapes_time[i_level][n_params + 1] = 2;
        m_fold_shapes_time[i_level][n_params + 2] = cfg.get_nbins();
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            m_fold_shapes[i_level][n_params + 2] = cfg.get_nbins_f();
        } else {
            m_fold_shapes[i_level][n_params + 2] = cfg.get_nbins();
        }
    }
}

template <SupportedFoldType FoldType>
void FFAPlan<FoldType>::compute_flops() noexcept {
    const auto& cfg     = get_config();
    const auto nbins    = cfg.get_nbins();
    const auto nsamps   = cfg.get_nsamps();
    const auto nfreqs   = get_ncoords()[0];
    const auto n_levels = get_n_levels();
    const auto nbins_f  = (nbins / 2) + 1;

    SizeType flops_brutefold{0U}, flops_ffa{0U};
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        if (nbins > cfg.get_nbins_min_lossy_bf()) {
            // O(nsamps) - Lossy BruteFold + RFFT
            flops_brutefold =
                (nsamps * 2 * nfreqs) + (nfreqs * nbins * std::log2(nbins));
        } else {
            // Direct DFT
            flops_brutefold = 8 * nsamps * 2 * nfreqs * nbins_f;
        }
        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto nsegments   = get_nsegments()[i_level];
            const auto ncoords_cur = get_ncoords()[i_level];
            // 2 profiles (e+v) * nbins_f * 8 FLOPs per complex
            // shift-add
            flops_ffa += ncoords_cur * nsegments * 2 * nbins_f * 8;
        }
    } else {
        // O(nsamps)
        flops_brutefold = (nsamps * 2 * nfreqs);
        for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
            const auto nsegments   = get_nsegments()[i_level];
            const auto ncoords_cur = get_ncoords()[i_level];
            // 2 profiles (e+v) * nbins * 1 FLOP per add
            flops_ffa += ncoords_cur * nsegments * 2 * nbins;
        }
    }
    m_flops_required = flops_brutefold + flops_ffa;
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        // IRFFT
        const auto nfft = get_fold_size_time() / nbins;
        m_flops_required_return_in_time =
            m_flops_required + (nfft * nbins * std::log2(nbins));
    } else {
        m_flops_required_return_in_time = m_flops_required;
    }
}

template <SupportedFoldType FoldType> void FFAPlan<FoldType>::validate() const {
    error_check::check(get_buffer_size() > 0,
                       "FFAPlan::validate_plan: buffer size is zero");
    error_check::check(get_fold_size() > 0,
                       "FFAPlan::validate_plan: fold size is zero");
    error_check::check(get_brute_fold_size() > 0,
                       "FFAPlan::validate_plan: brute fold size is zero");
}

// --- Explicit template instantiations ---
template class FFAPlan<float>;
template class FFAPlan<ComplexType>;
} // namespace loki::plans