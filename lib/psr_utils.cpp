#include "loki/psr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <format>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/transforms.hpp"
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
                            SizeType nbins,
                            double eta,
                            std::span<const double> f_max,
                            std::span<double> dparams_batch,
                            double t_ref) {

    const auto dparams_f = poly_taylor_step_f(nparams, tobs, nbins, eta, t_ref);
    const auto n_batch   = f_max.size();
    error_check::check_equal(dparams_batch.size(), n_batch * nparams,
                             "dparams_batch must be of size nbatch * nparams");
    for (SizeType i = 0; i < n_batch; ++i) {
        const auto factor = utils::kCval / f_max[i];
        for (SizeType j = 0; j < nparams; ++j) {
            dparams_batch[(i * nparams) + j] = dparams_f[j] * factor;
        }
    }
}

void poly_taylor_step_d_vec_limited(SizeType nparams,
                                    double tobs,
                                    SizeType nbins,
                                    double eta,
                                    std::span<const double> f_max,
                                    std::span<const ParamLimit> param_limits,
                                    std::span<double> dparams_batch,
                                    double t_ref) {

    const auto dparams_f = poly_taylor_step_f(nparams, tobs, nbins, eta, t_ref);
    const auto n_batch   = f_max.size();
    error_check::check_equal(dparams_batch.size(), n_batch * nparams,
                             "dparams_batch must be of size nbatch * nparams");
    for (SizeType i = 0; i < n_batch; ++i) {
        const auto factor = utils::kCval / f_max[i];
        for (SizeType j = 0; j < nparams - 1; ++j) {
            const double param_range =
                (param_limits[j].max - param_limits[j].min);
            dparams_batch[(i * nparams) + j] =
                std::min(dparams_f[j] * factor, param_range);
        }
        const double d1_range = factor * (param_limits[nparams - 1].max -
                                          param_limits[nparams - 1].min);
        dparams_batch[(i * nparams) + nparams - 1] =
            std::min(dparams_f[nparams - 1] * factor, d1_range);
    }
}

bool split_f(double df_old,
             double df_new,
             double tobs_new,
             SizeType k,
             double nbins,
             double eta,
             double t_ref) {
    const auto dt         = tobs_new - t_ref;
    const auto factor     = std::pow(dt, k + 1) * nbins /
                            math::factorial(static_cast<double>(k + 1));
    const auto factor_opt = factor / std::pow(2.0, k);
    return std::abs(df_old - df_new) * factor_opt > (eta - utils::kEps);
}

std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType nbins,
                                        double f_cur,
                                        double t_ref) {
    const auto n_params = dparam_old.size();
    const auto dt       = tobs_new - t_ref;
    std::vector<double> shift(n_params);
    for (SizeType i = 0; i < n_params; ++i) {
        const auto orth_factor = std::pow(2.0, i);
        auto factor =
            std::pow(dt, i + 1) * static_cast<double>(nbins) /
            (math::factorial(static_cast<double>(i + 1)) * orth_factor);
        shift[n_params - 1 - i] = std::abs(dparam_old[i] - dparam_new[i]) *
                                  factor * (f_cur / utils::kCval);
    }
    return shift;
}

void poly_taylor_shift_d_vec(std::span<const double> dparam_old,
                             std::span<const double> dparam_new,
                             double tobs_new,
                             SizeType nbins,
                             std::span<const double> f_cur,
                             double t_ref,
                             std::span<double> shift_bins_batch,
                             SizeType nbatch,
                             SizeType nparams) {

    const auto dt      = tobs_new - t_ref;
    const auto nbins_d = static_cast<double>(nbins);
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
        const auto factor_base  = dt_powers[k] * nbins_d / factorials[k];
        const auto factor_opt   = factor_base / std::pow(2.0, k);
        const auto scale_factor = factor_opt / utils::kCval;
        for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
            const SizeType idx    = (batch_idx * nparams) + i;
            const double diff     = std::abs(dparam_old[idx] - dparam_new[idx]);
            shift_bins_batch[idx] = diff * scale_factor * f_cur[batch_idx];
        }
    }
}

void poly_cheb_step_vec(SizeType n_params,
                        SizeType nbins,
                        double eta,
                        std::span<const double> f0_batch,
                        std::span<double> dparams_batch) {
    const auto n_batch = f0_batch.size();
    const auto dphi    = eta / static_cast<double>(nbins);
    error_check::check_equal(
        dparams_batch.size(), n_batch * n_params,
        "dparams_batch must be of size n_batch * n_params");
    for (SizeType i = 0; i < n_batch; ++i) {
        const double factor = utils::kCval / f0_batch[i];
        for (SizeType j = 0; j < n_params; ++j) {
            dparams_batch[((i * n_params) + j)] = dphi * factor;
        }
    }
}

void poly_cheb_step_vec_limited(SizeType n_params,
                                double scale_cur,
                                SizeType nbins,
                                double eta,
                                std::span<const double> f0_batch,
                                std::span<const ParamLimit> param_limits,
                                std::span<double> dparams_batch) {
    const auto n_batch = f0_batch.size();
    const auto dphi    = eta / static_cast<double>(nbins);
    error_check::check_equal(
        dparams_batch.size(), n_batch * n_params,
        "dparams_batch must be of size n_batch * n_params");
    // Build taylor_limits: shape [n_batch, n_params, 2]
    std::vector<double> taylor_limits(n_batch * n_params * 2);
    for (SizeType i = 0; i < n_batch; ++i) {
        const double f0 = f0_batch[i];
        for (SizeType j = 0; j < n_params - 1; ++j) {
            taylor_limits[(((i * n_params) + j) * 2) + 0] = param_limits[j].min;
            taylor_limits[(((i * n_params) + j) * 2) + 1] = param_limits[j].max;
        }
        const SizeType last = n_params - 1;
        taylor_limits[(((i * n_params) + last) * 2) + 0] =
            (1.0 - (param_limits[last].max / f0)) * utils::kCval;
        taylor_limits[(((i * n_params) + last) * 2) + 1] =
            (1.0 - (param_limits[last].min / f0)) * utils::kCval;
    }

    std::vector<double> cheby_limits(n_batch * n_params * 2);
    transforms::taylor_to_chebyshev_limits_full(
        taylor_limits, n_batch, n_params, scale_cur, cheby_limits);

    // Compute dparams and clamp against Chebyshev ranges
    for (SizeType i = 0; i < n_batch; ++i) {
        const double factor = utils::kCval / f0_batch[i];
        for (SizeType j = 0; j < n_params; ++j) {
            const double dparam_unclamped = dphi * factor;
            const double cheby_min =
                cheby_limits[(((i * n_params) + j) * 2) + 0];
            const double cheby_max =
                cheby_limits[(((i * n_params) + j) * 2) + 1];
            const double cheby_range = cheby_max - cheby_min;
            dparams_batch[((i * n_params) + j)] =
                std::min(dparam_unclamped, cheby_range);
        }
    }
}

void poly_cheb_shift_vec(std::span<const double> dparam_old,
                         std::span<const double> dparam_new,
                         SizeType nbins,
                         std::span<const double> f_cur,
                         std::span<double> shift_bins_batch,
                         SizeType nbatch,
                         SizeType nparams) {
    const auto nbins_d = static_cast<double>(nbins);
    error_check::check_equal(
        shift_bins_batch.size(), nbatch * nparams,
        "shift_bins_batch must be of size nbatch * nparams");
    for (SizeType i = 0; i < nparams; ++i) {
        const auto scale_factor = nbins_d / utils::kCval;
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
    if (dparam_cur <= utils::kEps || dparam_new <= utils::kEps) {
        throw std::invalid_argument(
            std::format("Both dparam_cur and dparam_new must be positive (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }
    if (param_max <= param_min + utils::kEps) {
        throw std::invalid_argument(
            std::format("param_max must be greater than param_min (got {}, {})",
                        param_max, param_min));
    }
    if (param_cur < (param_min + utils::kEps) ||
        param_cur > (param_max - utils::kEps)) {
        throw std::invalid_argument(
            std::format("param_cur ({}) must be within [param_min ({}), "
                        "param_max ({})]",
                        param_cur, param_min, param_max));
    }
    const double param_range = (param_max - param_min) / 2.0;
    if (dparam_new > (param_range + utils::kEps)) {
        // Step size too large, fallback to current value
        return {{param_cur}, dparam_new};
    }
    const int num_points = static_cast<int>(
        std::ceil(((dparam_cur + utils::kEps) / dparam_new) - utils::kEps));
    if (num_points <= 0) {
        throw std::invalid_argument(
            std::format("Invalid input: ensure dparam_cur > dparam_new (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }

    // Confidence-based symmetric range shrinkage
    // 0.5 < confidence_const < 1
    const double confidence_const =
        0.5 * (1.0 + (1.0 / static_cast<double>(num_points)));
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
                                                SizeType branch_max) {
    if (dparam_cur <= utils::kEps || dparam_new <= utils::kEps) {
        throw std::invalid_argument(
            std::format("Both dparam_cur and dparam_new must be positive (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }
    const auto num_points = static_cast<int>(
        std::ceil(((dparam_cur + utils::kEps) / dparam_new) - utils::kEps));
    error_check::check_greater(
        num_points, 0,
        std::format("num_points must be positive. Invalid input: ensure "
                    "dparam_cur > dparam_new (got {}, {})",
                    dparam_cur, dparam_new));

    // Calculate the actual branched values
    // 0.5 < confidence_const < 1
    const int n = num_points + 2;
    const double confidence_const =
        0.5 * (1.0 + (1.0 / static_cast<double>(num_points)));
    const double half_range = confidence_const * dparam_cur;
    const double start      = param_cur - half_range;
    const double stop       = param_cur + half_range;
    const int num_intervals = n - 1;
    const double step = (stop - start) / static_cast<double>(num_intervals);

    error_check::check_less_equal(num_points, static_cast<int>(branch_max),
                                  "num_points must be less than or equal to "
                                  "branch_max");

    // Generate points and fill the start of the padded array
    for (SizeType i = 0; i < static_cast<SizeType>(num_points); ++i) {
        out_values[i] = start + (static_cast<double>(i + 1) * step);
    }

    // Calculate actual dparam based on generated points
    const double dparam_new_actual =
        dparam_cur / static_cast<double>(num_points);
    return {dparam_new_actual, num_points};
}

std::pair<double, SizeType> branch_dparam_crackle(double dparam_cur,
                                                  double dparam_new,
                                                  SizeType branch_max) {
    if (dparam_cur <= utils::kEps || dparam_new <= utils::kEps) {
        throw std::invalid_argument(
            std::format("Both dparam_cur and dparam_new must be positive (got "
                        "{}, {})",
                        dparam_cur, dparam_new));
    }
    const auto num_points = static_cast<int>(
        std::ceil(((dparam_cur + utils::kEps) / dparam_new) - utils::kEps));
    error_check::check_greater(
        num_points, 0,
        std::format("num_points must be positive. Invalid input: ensure "
                    "dparam_cur > dparam_new (got {}, {})",
                    dparam_cur, dparam_new));
    error_check::check_less_equal(num_points, static_cast<int>(branch_max),
                                  "num_points must be less than or equal to "
                                  "branch_max");

    // Calculate actual dparam based on generated points
    const double dparam_new_actual =
        dparam_cur / static_cast<double>(num_points);
    return {dparam_new_actual, num_points};
}

void branch_crackle_padded(std::span<double> out_values,
                           double param_cur,
                           double dparam_cur,
                           SizeType num_points) {
    // Calculate the actual branched values
    // 0.5 < confidence_const < 1
    const auto n = num_points + 2;
    const double confidence_const =
        0.5 * (1.0 + (1.0 / static_cast<double>(num_points)));
    const double half_range  = confidence_const * dparam_cur;
    const double start       = param_cur - half_range;
    const double stop        = param_cur + half_range;
    const auto num_intervals = n - 1;
    const double step = (stop - start) / static_cast<double>(num_intervals);

    // Generate points and fill the start of the padded array
    for (SizeType i = 0; i < num_points; ++i) {
        out_values[i] = start + (static_cast<double>(i + 1) * step);
    }
}

} // namespace loki::psr_utils
