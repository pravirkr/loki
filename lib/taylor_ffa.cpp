#include "loki/core/taylor_ffa.hpp"

#include <cmath>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"
#include "loki/utils.hpp"

namespace loki::core {

namespace {

template <int LATTER>
void ffa_taylor_resolve_accel_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins) {
    constexpr SizeType kParams = 2;
    error_check::check_equal(param_grid_count_cur.size(), kParams,
                             "param_grid_count_cur should have 2 elements");
    error_check::check_equal(param_grid_count_prev.size(), kParams,
                             "param_grid_count_prev should have 2 elements");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits should have 2 elements");

    const SizeType n_accel_cur  = param_grid_count_cur[0];
    const SizeType n_freq_cur   = param_grid_count_cur[1];
    const SizeType n_accel_prev = param_grid_count_prev[0];
    const SizeType n_freq_prev  = param_grid_count_prev[1];
    const ParamLimit& lim_accel = param_limits[0];
    const ParamLimit& lim_freq  = param_limits[1];
    const auto ncoords          = n_accel_cur * n_freq_cur;
    error_check::check_equal(coords.size(), ncoords, "coords size mismatch");

    const double tsegment =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));
    const auto delta_t = (static_cast<double>(LATTER) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto half_delta_t_sq = 0.5 * delta_t * delta_t;

    for (SizeType accel_idx = 0; accel_idx < n_accel_cur; ++accel_idx) {
        // Generate parameters on the fly
        const auto a_cur =
            psr_utils::get_param_val_at_idx(lim_accel, n_accel_cur, accel_idx);
        const auto a_new = a_cur;
        const auto v_new = a_cur * delta_t;
        const auto d_new = a_cur * half_delta_t_sq;
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_prev);
        const auto coord_a_offset = accel_idx * n_freq_cur;
        const auto idx_a_offset   = idx_a * n_freq_prev;

        for (SizeType freq_idx = 0; freq_idx < n_freq_cur; ++freq_idx) {
            const auto f_cur =
                psr_utils::get_param_val_at_idx(lim_freq, n_freq_cur, freq_idx);
            const auto f_new     = f_cur * (1.0 - (v_new * utils::kInvCval));
            const auto delay_rel = d_new * utils::kInvCval;

            const auto relative_phase =
                psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);
            const auto idx_f = psr_utils::get_nearest_idx_analytical(
                f_new, lim_freq, n_freq_prev);

            const auto final_idx = static_cast<uint32_t>(idx_a_offset + idx_f);
            const auto coord_idx = coord_a_offset + freq_idx;
            if constexpr (LATTER == 0) {
                coords[coord_idx].i_tail     = final_idx;
                coords[coord_idx].shift_tail = relative_phase;
            } else {
                coords[coord_idx].i_head     = final_idx;
                coords[coord_idx].shift_head = relative_phase;
            }
        }
    }
}

template <int LATTER>
void ffa_taylor_resolve_jerk_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins,
    SizeType param_stride) {
    constexpr SizeType kParams = 3;
    error_check::check_greater_equal(
        param_grid_count_cur.size(), kParams,
        "param_grid_count_cur should have 3 elements");
    error_check::check_greater_equal(
        param_grid_count_prev.size(), kParams,
        "param_grid_count_prev should have 3 elements");
    error_check::check_greater_equal(param_limits.size(), kParams,
                                     "param_limits should have 3 elements");
    error_check::check_greater_equal(
        param_stride, kParams,
        "param_stride should be greater than or equal to 3");

    const SizeType po           = param_stride - kParams;
    const SizeType n_jerk_cur   = param_grid_count_cur[po + 0];
    const SizeType n_accel_cur  = param_grid_count_cur[po + 1];
    const SizeType n_freq_cur   = param_grid_count_cur[po + 2];
    const SizeType n_jerk_prev  = param_grid_count_prev[po + 0];
    const SizeType n_accel_prev = param_grid_count_prev[po + 1];
    const SizeType n_freq_prev  = param_grid_count_prev[po + 2];
    const ParamLimit& lim_jerk  = param_limits[po + 0];
    const ParamLimit& lim_accel = param_limits[po + 1];
    const ParamLimit& lim_freq  = param_limits[po + 2];
    const auto ncoords          = n_jerk_cur * n_accel_cur * n_freq_cur;
    error_check::check_equal(coords.size(), ncoords, "coords size mismatch");

    const double tsegment =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));
    const auto delta_t = (static_cast<double>(LATTER) - 0.5) * tsegment;

    // Pre-compute constants to avoid repeated calculations
    const auto delta_t_sq          = delta_t * delta_t;
    const auto delta_t_cubed       = delta_t_sq * delta_t;
    const auto half_delta_t_sq     = 0.5 * delta_t_sq;
    const auto sixth_delta_t_cubed = delta_t_cubed / 6.0;

    for (SizeType jerk_idx = 0; jerk_idx < n_jerk_cur; ++jerk_idx) {
        const auto j_cur =
            psr_utils::get_param_val_at_idx(lim_jerk, n_jerk_cur, jerk_idx);
        const auto j_new = j_cur; // No transformation needed

        // Pre-compute jerk-related terms for this jerk value
        const auto j_delta_t             = j_cur * delta_t;
        const auto half_j_delta_t_sq     = 0.5 * j_cur * delta_t_sq;
        const auto j_sixth_delta_t_cubed = j_cur * sixth_delta_t_cubed;

        const auto idx_j =
            psr_utils::get_nearest_idx_analytical(j_new, lim_jerk, n_jerk_prev);
        const auto coord_j_offset = jerk_idx * n_accel_cur * n_freq_cur;
        const auto idx_j_offset   = idx_j * n_accel_prev * n_freq_prev;

        for (SizeType accel_idx = 0; accel_idx < n_accel_cur; ++accel_idx) {
            const auto a_cur = psr_utils::get_param_val_at_idx(
                lim_accel, n_accel_cur, accel_idx);
            const auto a_new = a_cur + j_delta_t;
            const auto v_new = (a_cur * delta_t) + half_j_delta_t_sq;
            const auto d_new =
                (a_cur * half_delta_t_sq) + j_sixth_delta_t_cubed;

            // Find accel index once per (jerk_idx, accel_idx) pair
            const auto idx_a = psr_utils::get_nearest_idx_analytical(
                a_new, lim_accel, n_accel_prev);
            const auto coord_a_offset =
                coord_j_offset + (accel_idx * n_freq_cur);
            const auto idx_a_offset = idx_j_offset + (idx_a * n_freq_prev);

            for (SizeType freq_idx = 0; freq_idx < n_freq_cur; ++freq_idx) {
                const auto f_cur = psr_utils::get_param_val_at_idx(
                    lim_freq, n_freq_cur, freq_idx);
                const auto f_new = f_cur * (1.0 - (v_new * utils::kInvCval));
                const auto delay_rel = d_new * utils::kInvCval;

                const auto relative_phase =
                    psr_utils::get_phase_idx(delta_t, f_cur, nbins, delay_rel);
                const auto idx_f = psr_utils::get_nearest_idx_analytical(
                    f_new, lim_freq, n_freq_prev);

                const auto final_idx =
                    static_cast<uint32_t>(idx_a_offset + idx_f);
                const auto coord_idx = coord_a_offset + freq_idx;
                if constexpr (LATTER == 0) {
                    coords[coord_idx].i_tail     = final_idx;
                    coords[coord_idx].shift_tail = relative_phase;
                } else {
                    coords[coord_idx].i_head     = final_idx;
                    coords[coord_idx].shift_head = relative_phase;
                }
            }
        }
    }
}

template <SizeType NPARAMS, int LATTER>
void ffa_taylor_resolve_poly_batch_impl(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins) {
    static_assert(NPARAMS > 1 && NPARAMS <= 5 && LATTER >= 0 && LATTER <= 1,
                  "Unsupported number of parameters or latter");

    if constexpr (NPARAMS == 2) {
        ffa_taylor_resolve_accel_batch<LATTER>(
            param_grid_count_cur, param_grid_count_prev, param_limits, coords,
            ffa_level, tseg_brute, nbins);
    } else {
        ffa_taylor_resolve_jerk_batch<LATTER>(
            param_grid_count_cur, param_grid_count_prev, param_limits, coords,
            ffa_level, tseg_brute, nbins, NPARAMS);
    }
}

} // namespace

std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve_generic(std::span<const double> pset_cur,
                           std::span<const SizeType> param_grid_count_prev,
                           std::span<const ParamLimit> param_limits,
                           SizeType ffa_level,
                           SizeType latter,
                           double tseg_brute,
                           SizeType nbins) {
    const auto nparams = pset_cur.size();
    std::vector<double> pset_prev(nparams, 0.0);
    const double tsegment =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));
    double delta_t{}, delay{};

    if (nparams == 1) {
        delta_t      = static_cast<double>(latter) * tsegment;
        pset_prev[0] = pset_cur[0];
        delay        = 0.0;
    } else {
        delta_t = (static_cast<double>(latter) - 0.5) * tsegment;
        std::tie(pset_prev, delay) =
            transforms::shift_taylor_params_d_f(pset_cur, delta_t);
    }
    const auto relative_phase =
        psr_utils::get_phase_idx(delta_t, pset_cur[nparams - 1], nbins, delay);

    std::vector<SizeType> pindex_prev(nparams);
    for (SizeType ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] = psr_utils::get_nearest_idx_analytical(
            pset_prev[ip], param_limits[ip], param_grid_count_prev[ip]);
    }
    return {pindex_prev, relative_phase};
}

void ffa_taylor_resolve_freq_batch(SizeType n_freqs_cur,
                                   SizeType n_freqs_prev,
                                   const ParamLimit& lim_freq,
                                   std::span<coord::FFACoordFreq> coords,
                                   SizeType ffa_level,
                                   double tseg_brute,
                                   SizeType nbins) {
    error_check::check_equal(coords.size(), n_freqs_cur,
                             "coords size mismatch");

    const double delta_t =
        std::ldexp(tseg_brute, static_cast<int>(ffa_level - 1));

    // Calculate relative phases and flattened parameter indices
    for (SizeType i = 0; i < n_freqs_cur; ++i) {
        const double f_cur =
            psr_utils::get_param_val_at_idx(lim_freq, n_freqs_cur, i);
        const SizeType idx_f = psr_utils::get_nearest_idx_analytical(
            f_cur, lim_freq, n_freqs_prev);
        coords[i].idx   = static_cast<uint32_t>(idx_f);
        coords[i].shift = psr_utils::get_phase_idx(delta_t, f_cur, nbins, 0.0);
    }
}

void ffa_taylor_resolve_poly_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins,
    SizeType n_params) {
    auto dispatch = [&]<SizeType N, int L>() {
        return ffa_taylor_resolve_poly_batch_impl<N, L>(
            param_grid_count_cur, param_grid_count_prev, param_limits, coords,
            ffa_level, tseg_brute, nbins);
    };
    auto launch = [&](bool latter) {
        switch (n_params) {
        case 2:
            latter ? dispatch.template operator()<2, 0>()
                   : dispatch.template operator()<2, 1>();
            break;
        case 3:
            latter ? dispatch.template operator()<3, 0>()
                   : dispatch.template operator()<3, 1>();
            break;
        case 4:
            latter ? dispatch.template operator()<4, 0>()
                   : dispatch.template operator()<4, 1>();
            break;
        case 5:
            latter ? dispatch.template operator()<5, 0>()
                   : dispatch.template operator()<5, 1>();
            break;
        default:
            throw std::invalid_argument("Unsupported Taylor order");
        }
    };
    launch(latter != 0U);
}

} // namespace loki::core
