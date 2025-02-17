

#include <loki/math.hpp>
#include <loki/utils.hpp>

#include <loki/psr_utils.hpp>

SizeType loki::utils::get_phase_idx(float proper_time,
                                    float freq,
                                    SizeType nbins,
                                    float delay) {
    if (freq <= 0.0F) {
        throw std::invalid_argument("Frequency must be positive");
    }
    if (nbins == 0) {
        throw std::invalid_argument("Number of bins must be positive");
    }
    auto phase = std::fmod((proper_time + delay) * freq, 1.0F);
    if (phase < 0.0F) {
        phase += 1.0F; // Ensure phase is non-negative
    }
    // Round and wrap to ensure it is in [0, nbins)
    const auto iphase =
        static_cast<SizeType>(std::round(phase * static_cast<float>(nbins)));
    return (iphase == nbins) ? 0 : iphase;
}

std::vector<float> loki::utils::poly_taylor_step_f(SizeType nparams,
                                                   float tobs,
                                                   SizeType fold_bins,
                                                   float tol_bins,
                                                   float t_ref) {
    const auto dphi = tol_bins / static_cast<float>(fold_bins);
    const auto dt   = tobs - t_ref;
    std::vector<float> dparams_f(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        dparams_f[nparams - 1 - i] =
            dphi * loki::math::factorial(static_cast<float>(i + 1)) /
            std::pow(dt, static_cast<float>(i + 1));
    }
    return dparams_f;
}

std::vector<float> loki::utils::poly_taylor_step_d(SizeType nparams,
                                                   float tobs,
                                                   SizeType fold_bins,
                                                   float tol_bins,
                                                   float f_max,
                                                   float t_ref) {
    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    std::vector<float> dparams_d(nparams);
    for (SizeType i = 0; i < nparams - 1; ++i) {
        dparams_d[i] = dparams_f[i] * loki::kCval / f_max;
    }
    dparams_d[nparams - 1] = dparams_f[nparams - 1];
    return dparams_d;
}

std::vector<float>
loki::utils::poly_taylor_shift_d(std::span<const float> dparam_cur,
                                 std::span<const float> dparam_new,
                                 float tobs_new,
                                 SizeType fold_bins,
                                 float f_cur,
                                 float t_ref) {
    const auto nparams = dparam_cur.size();
    const auto dt      = tobs_new - t_ref;
    std::vector<float> shift(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        auto factor = std::pow(dt, static_cast<float>(i + 1)) *
                      static_cast<float>(fold_bins) /
                      loki::math::factorial(static_cast<float>(i + 1));
        if (i > 0) {
            factor *= f_cur / loki::kCval;
        }
        shift[nparams - 1 - i] =
            std::abs(dparam_cur[i] - dparam_new[i]) * factor;
    }
    return shift;
}

std::vector<float> loki::utils::shift_params(std::span<const float> param_vec,
                                             float delta_t) {
    const auto nparams = param_vec.size();
    std::vector<float> result(nparams, 0.0F);

    for (SizeType i = 0; i < nparams; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto k = i - j;
            auto term    = std::pow(delta_t, static_cast<float>(k)) /
                        static_cast<float>(loki::math::factorial(k));
            result[i] += param_vec[j] * term;
        }
    }
    return result;
}

std::tuple<std::vector<float>, float>
loki::utils::branch_param(float param_cur,
                          float dparam_cur,
                          float dparam_new,
                          float param_min,
                          float param_max) {
    if (dparam_cur <= 0.0F || dparam_new <= 0.0F) {
        throw std::invalid_argument(
            "Both dparam_cur and dparam_new must be positive");
    }
    if (param_cur < param_min || param_cur > param_max) {
        throw std::invalid_argument(
            "param_cur must be within [param_min, param_max]");
    }

    // If desired step size is too large, return current value
    if (dparam_new > (param_max - param_min) / 2.0F) {
        return {{param_cur}, dparam_new};
    }
    const auto n = 2 + static_cast<int>(std::ceil(dparam_cur / dparam_new));
    if (n < 3) {
        throw std::invalid_argument(
            "Invalid input: ensure dparam_cur > dparam_new");
    }

    // 0.5 < confidence_const < 1
    const auto confidence_const =
        0.5F * (1.0F + 1.0F / static_cast<float>(n - 2));
    const auto half_range = confidence_const * dparam_cur;
    const auto step       = 2.0F * half_range / static_cast<float>(n - 1);

    // Generate array excluding first and last points
    std::vector<float> param_arr_new;
    param_arr_new.reserve(n - 2);
    for (int i = 1; i < n - 1; ++i) {
        param_arr_new.push_back(param_cur - half_range +
                                (step * static_cast<float>(i)));
    }

    const auto dparam_new_actual = dparam_cur / static_cast<float>(n - 2);
    return {param_arr_new, dparam_new_actual};
}

std::vector<float> loki::utils::range_param(float vmin, float vmax, float dv) {
    if (vmin > vmax) {
        throw std::invalid_argument("vmin must be less than or equal to vmax");
    }
    if (dv <= 0) {
        throw std::invalid_argument("dv must be positive");
    }
    // Check if step size is larger than half the range
    if (dv > (vmax - vmin) / 2.0F) {
        return {(vmax + vmin) / 2.0F};
    }
    const auto npoints = static_cast<SizeType>((vmax - vmin) / dv);
    const auto step    = (vmax - vmin) / static_cast<float>(npoints + 1);
    std::vector<float> grid_points;
    grid_points.reserve(npoints);
    for (SizeType i = 1; i <= npoints; ++i) {
        grid_points.push_back(vmin + (step * static_cast<float>(i)));
    }
    return grid_points;
}
