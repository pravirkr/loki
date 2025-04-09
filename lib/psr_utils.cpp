#include "loki/psr_utils.hpp"

#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::psr_utils {

SizeType get_phase_idx(FloatType proper_time,
                       FloatType freq,
                       SizeType nbins,
                       FloatType delay) {
    if (freq <= 0.0) {
        throw std::invalid_argument("Frequency must be positive");
    }
    if (nbins == 0) {
        throw std::invalid_argument("Number of bins must be positive");
    }
    auto phase = std::fmod((proper_time + delay) * freq, 1.0);
    if (phase < 0.0) {
        phase += 1.0; // Ensure phase is non-negative
    }
    // Round and wrap to ensure it is in [0, nbins)
    const auto iphase = static_cast<SizeType>(
        std::round(phase * static_cast<FloatType>(nbins)));
    return (iphase == nbins) ? 0 : iphase;
}

std::vector<FloatType> poly_taylor_step_f(SizeType nparams,
                                          FloatType tobs,
                                          SizeType fold_bins,
                                          FloatType tol_bins,
                                          FloatType t_ref) {
    const auto dphi = tol_bins / static_cast<FloatType>(fold_bins);
    const auto dt   = tobs - t_ref;
    std::vector<FloatType> dparams_f(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        dparams_f[nparams - 1 - i] =
            dphi * loki::math::factorial(static_cast<FloatType>(i + 1)) /
            std::pow(dt, static_cast<FloatType>(i + 1));
    }
    return dparams_f;
}

std::vector<FloatType> poly_taylor_step_d(SizeType nparams,
                                          FloatType tobs,
                                          SizeType fold_bins,
                                          FloatType tol_bins,
                                          FloatType f_max,
                                          FloatType t_ref) {
    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    std::vector<FloatType> dparams_d(nparams);
    for (SizeType i = 0; i < nparams - 1; ++i) {
        dparams_d[i] = dparams_f[i] * utils::kCval / f_max;
    }
    dparams_d[nparams - 1] = dparams_f[nparams - 1];
    return dparams_d;
}

std::vector<FloatType>
poly_taylor_shift_d(std::span<const FloatType> dparam_cur,
                    std::span<const FloatType> dparam_new,
                    FloatType tobs_new,
                    SizeType fold_bins,
                    FloatType f_cur,
                    FloatType t_ref) {
    const auto nparams = dparam_cur.size();
    const auto dt      = tobs_new - t_ref;
    std::vector<FloatType> shift(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        auto factor = std::pow(dt, static_cast<FloatType>(i + 1)) *
                      static_cast<FloatType>(fold_bins) /
                      loki::math::factorial(static_cast<FloatType>(i + 1));
        if (i > 0) {
            factor *= f_cur / utils::kCval;
        }
        shift[nparams - 1 - i] =
            std::abs(dparam_cur[i] - dparam_new[i]) * factor;
    }
    return shift;
}

std::vector<FloatType> shift_params(std::span<const FloatType> param_vec,
                                    FloatType delta_t) {
    const auto nparams = param_vec.size();
    std::vector<FloatType> result(nparams, 0.0);

    for (SizeType i = 0; i < nparams; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto k = i - j;
            auto term    = std::pow(delta_t, static_cast<FloatType>(k)) /
                        static_cast<FloatType>(loki::math::factorial(k));
            result[i] += param_vec[j] * term;
        }
    }
    return result;
}

std::tuple<std::vector<FloatType>, FloatType>
branch_param(FloatType param_cur,
             FloatType dparam_cur,
             FloatType dparam_new,
             FloatType param_min,
             FloatType param_max) {
    if (dparam_cur <= 0.0 || dparam_new <= 0.0) {
        throw std::invalid_argument(
            "Both dparam_cur and dparam_new must be positive");
    }
    if (param_cur < param_min || param_cur > param_max) {
        throw std::invalid_argument(
            "param_cur must be within [param_min, param_max]");
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
        0.5 * (1.0 + 1.0 / static_cast<FloatType>(n - 2));
    const auto half_range = confidence_const * dparam_cur;
    const auto step       = 2.0 * half_range / static_cast<FloatType>(n - 1);

    // Generate array excluding first and last points
    std::vector<FloatType> param_arr_new;
    param_arr_new.reserve(n - 2);
    for (int i = 1; i < n - 1; ++i) {
        param_arr_new.push_back(param_cur - half_range +
                                (step * static_cast<FloatType>(i)));
    }

    const auto dparam_new_actual = dparam_cur / static_cast<FloatType>(n - 2);
    return {param_arr_new, dparam_new_actual};
}

std::vector<FloatType>
range_param(FloatType vmin, FloatType vmax, FloatType dv) {
    if (vmin > vmax) {
        throw std::invalid_argument("vmin must be less than or equal to vmax");
    }
    if (dv <= 0) {
        throw std::invalid_argument("dv must be positive");
    }
    // Check if step size is larger than half the range
    if (dv > (vmax - vmin) / 2.0) {
        return {(vmax + vmin) / 2.0};
    }
    const auto npoints = static_cast<SizeType>((vmax - vmin) / dv);
    const auto step    = (vmax - vmin) / static_cast<FloatType>(npoints + 1);
    std::vector<FloatType> grid_points;
    grid_points.reserve(npoints);
    for (SizeType i = 1; i <= npoints; ++i) {
        grid_points.push_back(vmin + (step * static_cast<FloatType>(i)));
    }
    return grid_points;
}

} // namespace loki::psr_utils
