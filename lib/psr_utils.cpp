#include "loki/psr_utils.hpp"

#include <cmath>
#include <format>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xview.hpp>

#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::psr_utils {

double
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
    return norm_phase * static_cast<double>(nbins);
}

SizeType get_phase_idx_int(double proper_time,
                           double freq,
                           SizeType nbins,
                           double delay) {
    const auto phase = get_phase_idx(proper_time, freq, nbins, delay);
    auto iphase      = static_cast<SizeType>(std::round(phase));
    if (iphase == nbins) {
        iphase = 0;
    }
    return iphase;
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

xt::xtensor<double, 2>
poly_taylor_step_d_vec(SizeType nparams,
                       double tobs,
                       SizeType fold_bins,
                       double tol_bins,
                       const xt::xtensor<double, 1>& f_max,
                       double t_ref) {

    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    const auto nbatch = f_max.size();

    xt::xtensor<double, 2> dparams = xt::zeros<double>({nbatch, nparams});
    for (SizeType i = 0; i < nparams - 1; ++i) {
        xt::view(dparams, xt::all(), i) = dparams_f[i] * utils::kCval / f_max;
    }
    xt::view(dparams, xt::all(), nparams - 1) = dparams_f[nparams - 1];
    return dparams;
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
    return std::abs(df_old - df_new) * factor > tol_bins;
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

xt::xtensor<double, 2>
poly_taylor_shift_d_vec(const xt::xtensor<double, 2>& dparam_old,
                        const xt::xtensor<double, 2>& dparam_new,
                        double tobs_new,
                        SizeType fold_bins,
                        const xt::xtensor<double, 1>& f_cur,
                        double t_ref) {

    const auto nbatch      = dparam_old.shape()[0];
    const auto nparams     = dparam_old.shape()[1];
    const auto dt          = tobs_new - t_ref;
    const auto fold_bins_d = static_cast<double>(fold_bins);

    xt::xtensor<double, 2> result     = xt::zeros<double>({nbatch, nparams});
    xt::xtensor<double, 1> dt_powers  = xt::zeros<double>({nparams});
    xt::xtensor<double, 1> factorials = xt::zeros<double>({nparams});

    double dt_power = dt;
    for (SizeType i = 0; i < nparams; ++i) {
        dt_powers(i)  = dt_power;
        factorials(i) = math::factorial(static_cast<double>(i + 1));
        dt_power *= dt;
    }
    for (SizeType i = 0; i < nparams; ++i) {
        const auto k     = nparams - 1 - i;
        auto factor_base = dt_powers(i) * fold_bins_d / factorials(i);

        if (i > 0) {
            // Scale by f_cur / C_VAL for all but last parameter
            auto factors = factor_base * f_cur / utils::kCval;
            xt::view(result, xt::all(), k) =
                xt::abs(xt::view(dparam_old, xt::all(), i) -
                        xt::view(dparam_new, xt::all(), i)) *
                factors;
        } else {
            // No scaling for last parameter
            xt::view(result, xt::all(), k) =
                xt::abs(xt::view(dparam_old, xt::all(), i) -
                        xt::view(dparam_new, xt::all(), i)) *
                factor_base;
        }
    }

    return result;
}

std::vector<double> shift_params_d(std::span<const double> param_vec,
                                   double delta_t,
                                   SizeType n_out) {
    const auto nparams   = param_vec.size();
    const auto n_out_min = std::min(n_out, nparams);
    std::vector<double> result(n_out_min, 0.0);

    // We want the last n_out_min rows of the transformation matrix
    const auto start_row = nparams - n_out_min;

    for (SizeType i = 0; i < n_out_min; ++i) {
        const auto row = start_row + i;
        for (SizeType j = 0; j <= row; ++j) {
            const auto k = row - j;
            auto term =
                std::pow(delta_t, k) / static_cast<double>(math::factorial(k));
            result[i] += param_vec[j] * term;
        }
    }
    return result;
}

xt::xtensor<double, 2>
shift_params_d_batch(const xt::xtensor<double, 2>& param_vec_batch,
                     double delta_t,
                     SizeType n_out) {
    const auto nparams      = param_vec_batch.shape()[1];
    const auto n_out_actual = std::min(n_out, nparams);

    xt::xtensor<double, 2> powers = xt::zeros<double>({nparams, nparams});
    for (SizeType i = 0; i < nparams; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            powers(i, j) = static_cast<double>(i - j);
        }
    }

    // Compute transformation matrix: t_mat = delta_t^powers / factorial(powers)
    xt::xtensor<double, 2> t_mat = xt::zeros<double>({nparams, nparams});
    for (SizeType i = 0; i < nparams; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto power = static_cast<SizeType>(powers(i, j));
            t_mat(i, j)      = std::pow(delta_t, power) /
                          math::factorial(static_cast<double>(power));
        }
    }
    // Take only the last n_out rows
    auto t_mat_subset =
        xt::view(t_mat, xt::range(nparams - n_out_actual, nparams), xt::all());
    return xt::linalg::dot(param_vec_batch, xt::transpose(t_mat_subset));
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

std::tuple<xt::xtensor<double, 3>, xt::xtensor<double, 1>>
shift_params_batch(const xt::xtensor<double, 3>& param_vec_batch,
                   double delta_t) {
    const auto size    = param_vec_batch.shape()[0];
    const auto nparams = param_vec_batch.shape()[1];

    xt::xtensor<double, 2> dvec_cur = xt::zeros<double>({size, nparams + 1});

    // Copy till acceleration: dvec_cur[:, :-2] = param_vec_batch[:, :-1, 0]
    if (nparams > 1) {
        xt::view(dvec_cur, xt::all(), xt::range(0, nparams - 1)) =
            xt::view(param_vec_batch, xt::all(), xt::range(0, nparams - 1), 0);
    }

    // Transform using batch operation
    auto dvec_new      = shift_params_d_batch(dvec_cur, delta_t, nparams + 1);
    auto param_vec_new = param_vec_batch; // Copy input
    xt::xtensor<double, 1> delay_rel = xt::zeros<double>({size});

    // Update parameters: param_vec_new[:, :-1, 0] = dvec_new[:, :-2]
    if (nparams > 1) {
        xt::view(param_vec_new, xt::all(), xt::range(0, nparams - 1), 0) =
            xt::view(dvec_new, xt::all(), xt::range(0, nparams - 1));
    }

    // Update frequency: param_vec_new[:, -1, 0] = param_vec_batch[:, -1, 0] *
    // (1 + dvec_new[:, -2] / C_VAL)
    auto freq_correction =
        1.0 + xt::view(dvec_new, xt::all(), nparams - 1) / utils::kCval;
    xt::view(param_vec_new, xt::all(), nparams - 1, 0) =
        xt::view(param_vec_batch, xt::all(), nparams - 1, 0) * freq_correction;

    // Compute delay: delay_rel = dvec_new[:, -1] / C_VAL
    delay_rel = xt::view(dvec_new, xt::all(), nparams) / utils::kCval;

    return {param_vec_new, delay_rel};
}

std::tuple<xt::xtensor<double, 3>, xt::xtensor<double, 1>>
shift_params_circular_batch(const xt::xtensor<double, 3>& param_vec_batch,
                            double delta_t) {

    const auto size    = param_vec_batch.shape()[0];
    const auto nparams = param_vec_batch.shape()[1];

    if (nparams != 4) {
        throw std::invalid_argument(
            "4 parameters are needed for circular orbit resolve.");
    }

    // Extract parameters for omega calculation
    auto snap_values    = xt::view(param_vec_batch, xt::all(), 0, 0);
    auto accel_values   = xt::view(param_vec_batch, xt::all(), 2, 0);
    auto minus_omega_sq = snap_values / accel_values;
    auto omega_batch    = xt::sqrt(-minus_omega_sq);

    // Compute required order
    auto max_omega      = xt::amax(omega_batch)();
    auto required_order = std::min(
        static_cast<SizeType>((max_omega * std::abs(delta_t) * M_E) + 10),
        100UL);

    // Create dvec_cur with extended size
    xt::xtensor<double, 2> dvec_cur =
        xt::zeros<double>({size, required_order + 1});

    // Copy base parameters: dvec_cur[:, -5:-2] = param_vec_batch[:, :-1, 0]
    xt::view(dvec_cur, xt::all(),
             xt::range(required_order - 4, required_order - 1)) =
        xt::view(param_vec_batch, xt::all(), xt::range(0, 3), 0);

    // Fill higher order derivatives
    for (SizeType power = 5; power <= required_order; ++power) {
        const auto col_idx = required_order - power;

        if (power % 2 == 0) {
            // Even powers: coefficients based on accel
            const auto exponent = (power - 2) / 2;
            auto coefs          = xt::pow(minus_omega_sq, exponent) *
                         xt::view(param_vec_batch, xt::all(), 2, 0);
            xt::view(dvec_cur, xt::all(), col_idx) = coefs;
        } else {
            // Odd powers: coefficients based on velocity
            const auto exponent = (power - 3) / 2;
            auto coefs          = xt::pow(minus_omega_sq, exponent) *
                         xt::view(param_vec_batch, xt::all(), 1, 0);
            xt::view(dvec_cur, xt::all(), col_idx) = coefs;
        }
    }

    // Transform using batch operation
    auto dvec_new = shift_params_d_batch(dvec_cur, delta_t, nparams + 1);

    // Create output arrays
    auto param_vec_new               = param_vec_batch; // Copy input
    xt::xtensor<double, 1> delay_rel = xt::zeros<double>({size});

    // Update parameters: param_vec_new[:, :-1, 0] = dvec_new[:, :-2]
    xt::view(param_vec_new, xt::all(), xt::range(0, nparams - 1), 0) =
        xt::view(dvec_new, xt::all(), xt::range(0, nparams - 1));

    // Update frequency with correction
    auto freq_correction =
        1.0 + xt::view(dvec_new, xt::all(), nparams - 1) / utils::kCval;
    xt::view(param_vec_new, xt::all(), nparams - 1, 0) =
        xt::view(param_vec_batch, xt::all(), nparams - 1, 0) * freq_correction;

    // Compute delay
    delay_rel = xt::view(dvec_new, xt::all(), nparams) / utils::kCval;

    return {param_vec_new, delay_rel};
}

xt::xtensor<double, 3>
convert_taylor_to_circular(const xt::xtensor<double, 3>& param_sets) {

    const auto size = param_sets.shape()[0];

    // Extract Taylor parameters
    auto snap  = xt::view(param_sets, xt::all(), 0, 0);
    auto jerk  = xt::view(param_sets, xt::all(), 1, 0);
    auto accel = xt::view(param_sets, xt::all(), 2, 0);
    auto freq  = xt::view(param_sets, xt::all(), 3, 0);

    auto dsnap  = xt::view(param_sets, xt::all(), 0, 1);
    auto djerk  = xt::view(param_sets, xt::all(), 1, 1);
    auto daccel = xt::view(param_sets, xt::all(), 2, 1);
    auto dfreq  = xt::view(param_sets, xt::all(), 3, 1);

    // Compute omega_sq and intermediate values
    auto omega_sq       = -snap / accel;
    auto omega          = xt::sqrt(omega_sq);
    auto omega_sq_cubed = omega_sq * omega;

    // Create output array
    auto out = xt::zeros_like(param_sets);

    // Circular parameters: [omega, freq, x_cos_phi, x_sin_phi]
    xt::view(out, xt::all(), 0, 0) = omega;
    xt::view(out, xt::all(), 1, 0) =
        freq * (1.0 - (-jerk / omega_sq) / utils::kCval);
    xt::view(out, xt::all(), 2, 0) = -accel / (omega_sq * utils::kCval);
    xt::view(out, xt::all(), 3, 0) = -jerk / (omega_sq_cubed * utils::kCval);

    // Uncertainties
    auto d_omega_sq = xt::sqrt(xt::pow(dsnap / accel, 2) +
                               xt::pow((snap * daccel) / xt::pow(accel, 2), 2));

    xt::view(out, xt::all(), 0, 1) = 0.5 * d_omega_sq / omega;

    // Complex uncertainty calculations using xtensor operations
    auto freq_term1 =
        xt::pow((1.0 + jerk / (omega_sq * utils::kCval)) * dfreq, 2);
    auto freq_term2 = xt::pow((freq / (omega_sq * utils::kCval)) * djerk, 2);
    auto freq_term3 = xt::pow(
        (freq * jerk / (xt::pow(omega_sq, 2) * utils::kCval)) * d_omega_sq, 2);
    xt::view(out, xt::all(), 1, 1) =
        xt::sqrt(freq_term1 + freq_term2 + freq_term3);

    auto x_cos_term1 = xt::pow(daccel / (omega_sq * utils::kCval), 2);
    auto x_cos_term2 = xt::pow(
        (accel * d_omega_sq) / (xt::pow(omega_sq, 2) * utils::kCval), 2);
    xt::view(out, xt::all(), 2, 1) = xt::sqrt(x_cos_term1 + x_cos_term2);

    auto x_sin_term1 = xt::pow(djerk / (omega_sq_cubed * utils::kCval), 2);
    auto x_sin_term2 = xt::pow(
        (1.5 * jerk * d_omega_sq) / (utils::kCval * xt::pow(omega_sq, 2.5)), 2);
    xt::view(out, xt::all(), 3, 1) = xt::sqrt(x_sin_term1 + x_sin_term2);

    return out;
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
    // np.linspace(vmin, vmax, npoints + 2)[1:-1]
    const auto npoints   = static_cast<SizeType>((vmax - vmin) / dv);
    auto grid_points     = xt::linspace(vmin, vmax, npoints + 2);
    const auto grid_view = xt::view(grid_points, xt::range(1, npoints + 1));
    return {grid_view.begin(), grid_view.end()};
}

} // namespace loki::psr_utils
