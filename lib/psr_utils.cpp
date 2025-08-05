#include "loki/psr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <format>

#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::psr_utils {

float get_phase_idx(double proper_time,
                    double freq,
                    SizeType nbins,
                    double delay) {
    if (freq <= 0.0) {
        throw std::invalid_argument(
            std::format("Frequency must be positive (got {})", freq));
    }
    if (nbins == 0) {
        throw std::invalid_argument(
            std::format("Number of bins must be positive (got {})", nbins));
    }
    // Calculate the total phase in cycles (can be negative or > 1)
    const double total_phase = (proper_time + delay) * freq;
    // Normalize phase to [0, 1) interval
    double norm_phase = std::fmod(total_phase, 1.0);
    // Handle negative phases by wrapping to positive equivalent
    // This ensures the result is always in [0, 1)
    if (norm_phase < 0.0) {
        norm_phase += 1.0;
    }
    // Scale the normalized phase to [0, nbins) and convert to float
    auto scaled_phase =
        static_cast<float>(norm_phase * static_cast<double>(nbins));
    if (scaled_phase >= static_cast<float>(nbins)) {
        scaled_phase = 0.0F;
    }
    return scaled_phase;
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

void poly_taylor_step_d_vec(SizeType nparams,
                            double tobs,
                            SizeType fold_bins,
                            double tol_bins,
                            std::span<const double> f_max,
                            std::span<double> dparams_batch,
                            double t_ref) {

    const auto dparams_f =
        poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref);
    const auto nbatch = f_max.size();
    error_check::check_equal(dparams_batch.size(), nbatch * nparams,
                             "dparams_batch must be of size nbatch * nparams");
    for (SizeType i = 0; i < nbatch; ++i) {
        const auto factor = utils::kCval / f_max[i];
        for (SizeType j = 0; j < nparams - 1; ++j) {
            dparams_batch[(i * nparams) + j] = dparams_f[j] * factor;
        }
        dparams_batch[(i * nparams) + (nparams - 1)] = dparams_f[nparams - 1];
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
        const auto k           = nparams - 1 - i;
        const auto factor_base = dt_powers[k] * fold_bins_d / factorials[k];

        if (i < nparams - 1) {
            // Scale by f_cur / C_VAL for all but last parameter
            const auto scale_factor = factor_base / utils::kCval;
            for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
                const SizeType idx = (batch_idx * nparams) + i;
                const double diff = std::abs(dparam_old[idx] - dparam_new[idx]);
                shift_bins_batch[idx] = diff * scale_factor * f_cur[batch_idx];
            }
        } else {
            // No scaling for last parameter (frequency)
            for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
                const SizeType idx = (batch_idx * nparams) + i;
                const double diff = std::abs(dparam_old[idx] - dparam_new[idx]);
                shift_bins_batch[idx] = diff * factor_base;
            }
        }
    }
}

std::vector<std::vector<double>> precompute_shift_matrix(SizeType nparams,
                                                         double delta_t) {
    std::vector<std::vector<double>> trans_matrix(
        nparams, std::vector<double>(nparams, 0.0));
    for (SizeType i = 0; i < nparams; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto k = i - j;
            const auto term =
                std::pow(delta_t, k) / static_cast<double>(math::factorial(k));
            trans_matrix[i][j] = term;
        }
    }
    return trans_matrix;
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

// Optimized flat version of shift_params_batch operations
void shift_params_batch(std::span<const double> param_vec_data,
                        double delta_t,
                        SizeType nbatch,
                        SizeType nparams,
                        SizeType param_vec_stride,
                        std::span<double> kvec_new_data,
                        std::span<double> delay_batch) {
    error_check::check(nparams > 1, "nparams must be greater than 1");

    const SizeType transform_dim = nparams + 1;
    // Pre-compute transformation coefficients (factorial terms)
    std::vector<double> taylor_coeffs(transform_dim);
    double dt_power = 1.0;
    for (SizeType k = 0; k < transform_dim; ++k) {
        taylor_coeffs[k] = dt_power / math::factorial(static_cast<double>(k));
        dt_power *= delta_t;
    }
    std::vector<double> dvec_cur(transform_dim, 0.0);
    std::vector<double> dvec_new(transform_dim, 0.0);

    // Process each batch item
    for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
        const SizeType batch_offset = batch_idx * param_vec_stride;
        const SizeType kvec_offset  = batch_idx * nparams;

        // Reset arrays
        std::ranges::fill(dvec_cur, 0.0);
        std::ranges::fill(dvec_new, 0.0);

        // Extract current parameters (excluding frequency for dvec_cur)
        for (SizeType j = 0; j < nparams - 1; ++j) {
            dvec_cur[j] = param_vec_data[batch_offset + (j * 2)];
        }

        // Apply transformation: dvec_new = T_mat * dvec_cur
        for (SizeType i = 0; i < transform_dim; ++i) {
            for (SizeType j = 0; j <= i && j < transform_dim; ++j) {
                const SizeType power = i - j;
                dvec_new[i] += dvec_cur[j] * taylor_coeffs[power];
            }
        }
        // Update parameter values (except frequency)
        for (SizeType j = 0; j < nparams - 1; ++j) {
            kvec_new_data[kvec_offset + j] = dvec_new[j];
        }

        // Update frequency with relativistic correction
        const double freq_old =
            param_vec_data[batch_offset + ((nparams - 1) * 2)];
        kvec_new_data[kvec_offset + (nparams - 1)] =
            freq_old * (1.0 + (dvec_new[nparams - 1] / utils::kCval));

        // Compute delay
        delay_batch[batch_idx] = dvec_new[nparams] / utils::kCval;
    }
}

// Optimized flat version of shift_params_circular_batch
void shift_params_circular_batch(std::span<const double> param_vec_data,
                                 double delta_t,
                                 SizeType nbatch,
                                 SizeType nparams,
                                 SizeType param_vec_stride,
                                 std::span<double> kvec_new_data,
                                 std::span<double> delay_batch) {
    error_check::check_equal(nparams, 4U,
                             "nparams should be 4 for circular orbit resolve");

    // Calculate omega values for all batches
    std::vector<double> omega_batch(nbatch);
    double max_omega = 0.0;

    for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
        const SizeType batch_offset = batch_idx * param_vec_stride;
        const double minus_omega_sq =
            param_vec_data[batch_offset] / param_vec_data[batch_offset + 4];
        const double omega     = std::sqrt(-minus_omega_sq);
        omega_batch[batch_idx] = omega;
        max_omega              = std::max(max_omega, omega);
    }

    const auto required_order = std::min(
        static_cast<SizeType>((max_omega * std::abs(delta_t) * M_E) + 10),
        100UL);

    // Pre-compute transformation coefficients for the extended matrix
    const SizeType transform_dim = required_order + 1;
    std::vector<double> taylor_coeffs(transform_dim);
    // double dt_power = 1.0;
    // for (SizeType k = 0; k < transform_dim; ++k) {
    //     taylor_coeffs[k] = dt_power /
    //     math::factorial(static_cast<double>(k)); dt_power *= delta_t;
    // }
    taylor_coeffs[0] = 1.0;
    for (SizeType k = 1; k < transform_dim; ++k) {
        taylor_coeffs[k] =
            taylor_coeffs[k - 1] * delta_t / static_cast<double>(k);
    }

    std::vector<double> dvec_cur(transform_dim, 0.0);
    std::vector<double> dvec_new(nparams + 1, 0.0);

    // Process each batch item
    for (SizeType batch_idx = 0; batch_idx < nbatch; ++batch_idx) {
        const SizeType batch_offset = batch_idx * param_vec_stride;
        const SizeType kvec_offset  = batch_idx * nparams;
        const double omega          = omega_batch[batch_idx];
        const double minus_omega_sq = -(omega * omega);

        // Reset arrays
        std::ranges::fill(dvec_cur, 0.0);
        std::ranges::fill(dvec_new, 0.0);

        // Copy base parameters: dvec_cur[:, -5:-2] = param_vec_batch[:, :-1, 0]
        // This corresponds to required_order-4 to required_order-1
        const SizeType base_start = required_order - 4; // transform_dim - 5
        for (SizeType j = 0; j < 3; ++j) {
            dvec_cur[base_start + j] = param_vec_data[batch_offset + (j * 2)];
        }

        // Fill higher order derivatives
        for (SizeType power = 5; power <= required_order; ++power) {
            const SizeType col_idx = required_order - power;

            if (power % 2 == 0) {
                // Even powers: coefficients based on accel
                const SizeType exponent = (power - 2) / 2;
                const double coef =
                    std::pow(minus_omega_sq, static_cast<double>(exponent)) *
                    param_vec_data[batch_offset + 4];
                dvec_cur[col_idx] = coef;
            } else {
                // Odd powers: coefficients based on velocity
                const SizeType exponent = (power - 3) / 2;
                const double coef =
                    std::pow(minus_omega_sq, static_cast<double>(exponent)) *
                    param_vec_data[batch_offset + 2];
                dvec_cur[col_idx] = coef;
            }
        }

        // Apply transformation matrix
        for (SizeType i = 0; i < nparams + 1; ++i) {
            const SizeType row = transform_dim - (nparams + 1) + i;
            double sum         = 0.0;
            for (SizeType col = 0; col <= row; ++col) {
                const SizeType expo = row - col;
                sum += dvec_cur[col] * taylor_coeffs[expo];
            }
            dvec_new[i] = sum;
        }

        // Update parameter values (except frequency)
        for (SizeType j = 0; j < nparams - 1; ++j) {
            kvec_new_data[kvec_offset + j] = dvec_new[j];
        }

        // Update frequency with relativistic correction
        const double freq_old =
            param_vec_data[batch_offset + ((nparams - 1) * 2)];
        kvec_new_data[kvec_offset + (nparams - 1)] =
            freq_old * (1.0 + (dvec_new[nparams - 1] / utils::kCval));

        // Compute delay
        delay_batch[batch_idx] = dvec_new[nparams] / utils::kCval;
    }
}

/*
xt::xtensor<double, 3>
convert_taylor_to_circular(const xt::xtensor<double, 3>& param_sets) {
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
*/

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
            "param_cur ({}) must be within [param_min ({}), param_max ({})]",
            param_cur, param_min, param_max));
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

    // Generate array excluding first and last points
    auto grid_points =
        utils::linspace(param_cur - half_range, param_cur + half_range, n);
    const auto dparam_new_actual = dparam_cur / static_cast<double>(n - 2);
    return {std::vector<double>(grid_points.begin() + 1, grid_points.end() - 1),
            dparam_new_actual};
}

std::pair<double, SizeType> branch_param_padded(std::span<double> out_values,
                                                double param_cur,
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
            "param_cur ({}) must be within [param_min ({}), param_max ({})]",
            param_cur, param_min, param_max));
    }

    SizeType count           = 0;
    double dparam_new_actual = dparam_new; // Default if no branching occurs

    // If desired step size is too large, return current value
    if (dparam_new > (param_max - param_min) / 2.0) {
        out_values[0]     = param_cur;
        count             = 1;
        dparam_new_actual = dparam_cur;
        return {dparam_new_actual, count};
    }

    const auto n = 2 + static_cast<int>(std::ceil(dparam_cur / dparam_new));
    const auto num_points = n - 2;
    if (num_points <= 0) {
        throw std::invalid_argument(std::format(
            "Invalid input: ensure dparam_cur > dparam_new (got {}, {})",
            dparam_cur, dparam_new));
    }

    // Calculate the actual branched values
    // 0.5 < confidence_const < 1
    const double confidence_const =
        0.5 * (1.0 + 1.0 / static_cast<double>(num_points));
    const double half_range  = confidence_const * dparam_cur;
    const double start       = param_cur - half_range;
    const double stop        = param_cur + half_range;
    const auto num_intervals = n - 1;
    const double step = (stop - start) / static_cast<double>(num_intervals);

    // Generate points and fill the start of the padded array
    double current_val = start + step;
    count = std::min(static_cast<SizeType>(num_points), out_values.size());

    for (SizeType i = 0; i < count; ++i) {
        out_values[i] = current_val;
        current_val += step;
    }

    // Calculate actual dparam based on generated points
    dparam_new_actual = dparam_cur / static_cast<double>(num_points);

    return {dparam_new_actual, count};
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
    auto grid_points =
        utils::linspace(vmin, vmax, npoints + 2, /*endpoint=*/true);
    return {grid_points.begin() + 1, grid_points.end() - 1};
}

} // namespace loki::psr_utils
