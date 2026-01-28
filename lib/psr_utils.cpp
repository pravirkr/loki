#include "loki/psr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <format>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::psr_utils {

std::vector<double> poly_taylor_step_f(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double t_ref) {
    const auto dphi = tol_bins / static_cast<double>(fold_bins);
    const auto dt   = tobs - t_ref;
    std::vector<double> dparams_f(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        const auto orth_factor = std::pow(2.0, i);
        dparams_f[nparams - 1 - i] =
            dphi * orth_factor * math::factorial(static_cast<double>(i + 1)) /
            std::pow(dt, i + 1);
    }
    return dparams_f;
}

std::vector<double> poly_taylor_step_d_f(SizeType nparams,
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

std::vector<double> poly_taylor_step_d(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double f_max,
                                       double t_ref) {
    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    std::vector<double> dparams_d(nparams);
    for (SizeType i = 0; i < nparams; ++i) {
        dparams_d[i] = dparams_f[i] * utils::kCval / f_max;
    }
    return dparams_d;
}

void poly_taylor_step_d_vec(SizeType nparams,
                            double tobs,
                            SizeType fold_bins,
                            double tol_bins,
                            std::span<const double> f_max,
                            std::span<double> dparams_batch,
                            double t_ref) {

    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    const auto n_batch = f_max.size();
    error_check::check_equal(dparams_batch.size(), n_batch * nparams,
                             "dparams_batch must be of size nbatch * nparams");
    for (SizeType i = 0; i < n_batch; ++i) {
        const auto factor = utils::kCval / f_max[i];
        for (SizeType j = 0; j < nparams; ++j) {
            dparams_batch[(i * nparams) + j] = dparams_f[j] * factor;
        }
    }
}

bool split_f(double df_old,
             double df_new,
             double tobs_new,
             SizeType k,
             double fold_bins,
             double tol_bins,
             double t_ref) {
    const auto dt     = tobs_new - t_ref;
    const auto factor = std::pow(dt, k + 1) * fold_bins /
                        math::factorial(static_cast<double>(k + 1));
    const auto factor_opt = factor / std::pow(2.0, k);
    constexpr double kEps = 1e-6;
    return std::abs(df_old - df_new) * factor_opt > (tol_bins - kEps);
}

std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType fold_bins,
                                        double f_cur,
                                        double t_ref) {
    const auto n_params = dparam_old.size();
    const auto dt       = tobs_new - t_ref;
    std::vector<double> shift(n_params);
    for (SizeType i = 0; i < n_params; ++i) {
        const auto orth_factor = std::pow(2.0, i);
        auto factor =
            std::pow(dt, i + 1) * static_cast<double>(fold_bins) /
            (math::factorial(static_cast<double>(i + 1)) * orth_factor);
        shift[n_params - 1 - i] = std::abs(dparam_old[i] - dparam_new[i]) *
                                  factor * (f_cur / utils::kCval);
    }
    return shift;
}

void poly_taylor_shift_d_vec(std::span<const double> dparam_old,
                             std::span<const double> dparam_new,
                             double tobs_new,
                             SizeType fold_bins,
                             std::span<const double> f_cur,
                             double t_ref,
                             std::span<double> shift_bins_batch,
                             SizeType nbatch,
                             SizeType nparams) {

    const auto dt          = tobs_new - t_ref;
    const auto fold_bins_d = static_cast<double>(fold_bins);
    error_check::check_equal(
        shift_bins_batch.size(), nbatch * nparams,
        "shift_bins_batch must be of size nbatch * nparams");

    // Pre-compute dt powers and factorials
    std::vector<double> dt_powers(nparams);
    std::vector<double> factorials(nparams);

    double dt_power = dt;
    for (SizeType i = 0; i < nparams; ++i) {
        dt_powers[i]  = dt_power;
        factorials[i] = math::factorial(static_cast<double>(i + 1));
        dt_power *= dt;
    }

    // Optimized computation
    for (SizeType i = 0; i < nparams; ++i) {
        const auto k            = nparams - 1 - i;
        const auto factor_base  = dt_powers[k] * fold_bins_d / factorials[k];
        const auto factor_opt   = factor_base / std::pow(2.0, k);
        const auto scale_factor = factor_opt / utils::kCval;
        for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
            const SizeType idx    = (batch_idx * nparams) + i;
            const double diff     = std::abs(dparam_old[idx] - dparam_new[idx]);
            shift_bins_batch[idx] = diff * scale_factor * f_cur[batch_idx];
        }
    }
}

std::tuple<std::vector<double>, double> branch_param(double param_cur,
                                                     double dparam_cur,
                                                     double dparam_new,
                                                     double param_min,
                                                     double param_max) {
    constexpr double kEps = 1e-12;
    if (dparam_cur <= kEps || dparam_new <= kEps) {
        throw std::invalid_argument(
            std::format("Both dparam_cur and dparam_new must be positive (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }
    if (param_max <= param_min + kEps) {
        throw std::invalid_argument(
            std::format("param_max must be greater than param_min (got {}, {})",
                        param_max, param_min));
    }
    if (param_cur < (param_min + kEps) || param_cur > (param_max - kEps)) {
        throw std::invalid_argument(
            std::format("param_cur ({}) must be within [param_min ({}), "
                        "param_max ({})]",
                        param_cur, param_min, param_max));
    }
    const double param_range = (param_max - param_min) / 2.0;
    if (dparam_new > (param_range + kEps)) {
        // Step size too large, fallback to current value
        return {{param_cur}, dparam_new};
    }
    const int num_points =
        static_cast<int>(std::ceil(((dparam_cur + kEps) / dparam_new) - kEps));
    if (num_points <= 0) {
        throw std::invalid_argument(
            std::format("Invalid input: ensure dparam_cur > dparam_new (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }

    // Confidence-based symmetric range shrinkage
    // 0.5 < confidence_const < 1
    const double confidence_const =
        0.5 * (1.0 + 1.0 / static_cast<double>(num_points));
    const double half_range = confidence_const * dparam_cur;

    // Generate array excluding first and last points
    const int n = num_points + 2;
    auto grid_points =
        utils::linspace(param_cur - half_range, param_cur + half_range, n);
    const double dparam_new_actual =
        dparam_cur / static_cast<double>(num_points);
    return {std::vector<double>(grid_points.begin() + 1, grid_points.end() - 1),
            dparam_new_actual};
}

std::pair<double, SizeType> branch_param_padded(std::span<double> out_values,
                                                double param_cur,
                                                double dparam_cur,
                                                double dparam_new,
                                                double param_min,
                                                double param_max) {
    constexpr double kEps = 1e-12;
    if (dparam_cur <= kEps || dparam_new <= kEps) {
        throw std::invalid_argument(
            std::format("Both dparam_cur and dparam_new must be positive (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }

    if (param_max <= param_min + kEps) {
        throw std::invalid_argument(
            std::format("param_max must be greater than param_min (got {}, {})",
                        param_max, param_min));
    }
    const double param_range = (param_max - param_min) / 2.0;
    if (dparam_new > (param_range + kEps)) {
        // Step size too large, fallback to current value
        out_values[0] = param_cur;
        return {dparam_new, 1};
    }

    const auto num_points =
        static_cast<int>(std::ceil(((dparam_cur + kEps) / dparam_new) - kEps));
    if (num_points <= 0) {
        throw std::invalid_argument(std::format(
            "Invalid input: ensure dparam_cur > dparam_new (got {}, {})",
            dparam_cur, dparam_new));
    }

    // Calculate the actual branched values
    // 0.5 < confidence_const < 1
    const int n = num_points + 2;
    const double confidence_const =
        0.5 * (1.0 + 1.0 / static_cast<double>(num_points));
    const double half_range = confidence_const * dparam_cur;
    const double start      = param_cur - half_range;
    const double stop       = param_cur + half_range;
    const int num_intervals = n - 1;
    const double step = (stop - start) / static_cast<double>(num_intervals);

    // Generate points and fill the start of the padded array
    const SizeType count =
        std::min(static_cast<SizeType>(num_points), out_values.size());
    for (SizeType i = 0; i < count; ++i) {
        out_values[i] = start + (static_cast<double>(i + 1) * step);
    }

    // Calculate actual dparam based on generated points
    const double dparam_new_actual =
        dparam_cur / static_cast<double>(num_points);
    return {dparam_new_actual, count};
}

SizeType range_param_count(double vmin, double vmax, double dv) {
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
        return 1;
    }
    // np.linspace(vmin, vmax, npoints + 2)[1:-1]
    return static_cast<SizeType>((vmax - vmin) / dv);
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
    // np.linspace(vmin, vmax, npoints + 2)[1:-1]
    const auto npoints = static_cast<SizeType>((vmax - vmin) / dv);

    std::vector<double> result(npoints);
    const auto step = (vmax - vmin) / static_cast<double>(npoints + 1);
    // Start from i=1, end at i=total_points-1 (exclusive)
    for (SizeType i = 0; i < npoints; ++i) {
        result[i] = vmin + (step * static_cast<double>(i + 1));
    }
    return result;
}

} // namespace loki::psr_utils
