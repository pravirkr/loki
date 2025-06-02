#include "loki/psr_utils.hpp"

#include <cmath>
#include <format>

#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::psr_utils {

SizeType
get_phase_idx(double proper_time, double freq, SizeType nbins, double delay) {
    if (freq <= 0.0) {
        throw std::invalid_argument(
            std::format("Frequency must be positive (got {})", freq));
    }
    if (nbins <= 0) {
        throw std::invalid_argument(
            std::format("Number of bins must be positive (got {})", nbins));
    }
    // Ensure norm_phase is non-negative
    const auto phase      = std::fmod((proper_time + delay) * freq, 1.0);
    const auto norm_phase = phase < 0.0 ? phase + 1.0 : phase;
    // phase is in [0, 1). Round and wrap to ensure it is in [0, nbins).
    const auto iphase = static_cast<SizeType>(std::round(
                            norm_phase * static_cast<double>(nbins))) %
                        nbins;
    return iphase;
}

double get_phase_idx_complete(double proper_time,
                              double freq,
                              SizeType nbins,
                              double delay) {
    if (freq <= 0.0) {
        throw std::invalid_argument(
            std::format("Frequency must be positive (got {})", freq));
    }
    if (nbins <= 0) {
        throw std::invalid_argument(
            std::format("Number of bins must be positive (got {})", nbins));
    }
    const auto phase      = std::fmod((proper_time + delay) * freq, 1.0);
    const auto norm_phase = phase < 0.0 ? phase + 1.0 : phase;
    return norm_phase * static_cast<double>(nbins);
}

std::vector<double> poly_taylor_step_f(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double t_ref) {
    const auto dphi = tol_bins / static_cast<double>(fold_bins);
    const auto dt   = tobs - t_ref;
    std::vector<double> dparams_f(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        dparams_f[nparams - 1 - i] =
            dphi * math::factorial(static_cast<double>(i + 1)) /
            std::pow(dt, i + 1);
    }
    return dparams_f;
}

std::vector<double> poly_taylor_step_d(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double f_max,
                                       double t_ref) {
    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    std::vector<double> dparams_d(nparams);
    for (SizeType i = 0; i < nparams - 1; ++i) {
        dparams_d[i] = dparams_f[i] * utils::kCval / f_max;
    }
    dparams_d[nparams - 1] = dparams_f[nparams - 1];
    return dparams_d;
}

std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType fold_bins,
                                        double f_cur,
                                        double t_ref) {
    const auto nparams = dparam_old.size();
    const auto dt      = tobs_new - t_ref;
    std::vector<double> shift(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        auto factor = std::pow(dt, i + 1) * static_cast<double>(fold_bins) /
                      math::factorial(static_cast<double>(i + 1));
        if (i > 0) {
            factor *= f_cur / utils::kCval;
        }
        shift[nparams - 1 - i] =
            std::abs(dparam_old[i] - dparam_new[i]) * factor;
    }
    return shift;
}

std::vector<double> shift_params_d(std::span<const double> param_vec,
                                   double delta_t,
                                   SizeType n_out) {
    const auto nparams   = param_vec.size();
    const auto n_out_min = std::min(n_out, nparams);
    std::vector<double> result(n_out_min, 0.0);

    for (SizeType i = 0; i < nparams; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto k = i - j;
            auto term =
                std::pow(delta_t, k) / static_cast<double>(math::factorial(k));
            result[i] += param_vec[j] * term;
        }
    }
    return result;
}

std::tuple<std::vector<double>, double>
shift_params(std::span<const double> param_vec, double delta_t) {
    const auto nparams = param_vec.size();
    std::vector<double> dvec_cur(nparams + 1, 0.0);
    // Copy till acceleration
    if (nparams > 1) {
        std::copy(param_vec.begin(), param_vec.end() - 1, dvec_cur.begin());
    }
    std::vector<double> dvec_new =
        shift_params_d(dvec_cur, delta_t, nparams + 1);
    std::vector<double> param_vec_new(param_vec.begin(), param_vec.end());
    if (nparams > 1) {
        std::copy(dvec_new.begin(), dvec_new.end() - 2, param_vec_new.begin());
    }
    param_vec_new.back() =
        param_vec.back() * (1.0 + dvec_new[nparams - 1] / utils::kCval);
    const auto delay_rel = dvec_new.back() / utils::kCval;
    return {param_vec_new, delay_rel};
}

std::tuple<std::vector<double>, double> branch_param(double param_cur,
                                                     double dparam_cur,
                                                     double dparam_new,
                                                     double param_min,
                                                     double param_max) {
    if (dparam_cur <= 0.0 || dparam_new <= 0.0) {
        throw std::invalid_argument(
            std::format("Both dparam_cur and dparam_new must be positive (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }
    if (param_cur < param_min || param_cur > param_max) {
        throw std::invalid_argument(std::format(
            "param_cur must be within [param_min, param_max] (got {})",
            param_cur));
    }

    // If desired step size is too large, return current value
    if (dparam_new > (param_max - param_min) / 2.0) {
        return {{param_cur}, dparam_new};
    }
    const auto n = 2 + static_cast<int>(std::ceil(dparam_cur / dparam_new));
    if (n < 3) {
        throw std::invalid_argument(
            "Invalid input: ensure dparam_cur > dparam_new");
    }

    // 0.5 < confidence_const < 1
    const auto confidence_const =
        0.5 * (1.0 + 1.0 / static_cast<double>(n - 2));
    const auto half_range = confidence_const * dparam_cur;
    const auto step       = 2.0 * half_range / static_cast<double>(n - 1);

    // Generate array excluding first and last points
    std::vector<double> param_arr_new;
    param_arr_new.reserve(n - 2);
    for (int i = 1; i < n - 1; ++i) {
        param_arr_new.push_back(param_cur - half_range +
                                (step * static_cast<double>(i)));
    }

    const auto dparam_new_actual = dparam_cur / static_cast<double>(n - 2);
    return {param_arr_new, dparam_new_actual};
}

std::vector<double> range_param(double vmin, double vmax, double dv) {
    if (vmin > vmax) {
        throw std::invalid_argument(
            std::format("vmin must be less than or equal to vmax (got {}, {})",
                        vmin, vmax));
    }
    if (dv <= 0) {
        throw std::invalid_argument(
            std::format("dv must be positive (got {})", dv));
    }
    // Check if step size is larger than half the range
    if (dv > (vmax - vmin) / 2.0) {
        return {(vmax + vmin) / 2.0};
    }
    const auto npoints = static_cast<SizeType>((vmax - vmin) / dv);
    const auto step    = (vmax - vmin) / static_cast<double>(npoints + 1);
    std::vector<double> grid_points;
    grid_points.reserve(npoints);
    for (SizeType i = 1; i <= npoints; ++i) {
        grid_points.push_back(vmin + (step * static_cast<double>(i)));
    }
    return grid_points;
}

} // namespace loki::psr_utils
