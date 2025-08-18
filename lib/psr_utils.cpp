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
    const double total_phase = (proper_time - delay) * freq;
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
        const auto orth_factor = std::pow(2.0, i);
        dparams_f[nparams - 1 - i] =
            dphi * orth_factor * math::factorial(static_cast<double>(i + 1)) /
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
    const auto factor_opt = factor / std::pow(2.0, k);
    return std::abs(df_old - df_new) * factor_opt > tol_bins;
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
        const auto orth_factor = std::pow(2.0, i);
        auto factor =
            std::pow(dt, i + 1) * static_cast<double>(fold_bins) /
            (math::factorial(static_cast<double>(i + 1)) * orth_factor);
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
        const auto factor_opt  = factor_base / std::pow(2.0, k);

        if (i < nparams - 1) {
            // Scale by f_cur / C_VAL for all but last parameter
            const auto scale_factor = factor_opt / utils::kCval;
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
                shift_bins_batch[idx] = diff * factor_opt;
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
        param_vec.back() * (1.0 - dvec_new[nparams - 1] / utils::kCval);
    const auto delay_rel = dvec_new.back() / utils::kCval;
    return {param_vec_new, delay_rel};
}

// Optimized flat version of shift_params_batch operations
void shift_params_batch(std::span<double> param_vec_data,
                        double delta_t,
                        SizeType n_batch,
                        SizeType n_params) {
    error_check::check(n_params > 1, "nparams must be greater than 1");
    constexpr auto kParamsStride = 2UL;
    const auto batch_stride      = n_params * kParamsStride;

    const SizeType transform_dim = n_params + 1;
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
    for (SizeType batch_idx = 0; batch_idx < n_batch; ++batch_idx) {
        const SizeType batch_offset = batch_idx * batch_stride;
        // Reset arrays
        std::ranges::fill(dvec_cur, 0.0);
        std::ranges::fill(dvec_new, 0.0);

        // Extract current parameters (excluding frequency for dvec_cur)
        for (SizeType j = 0; j < n_params - 1; ++j) {
            dvec_cur[j] = param_vec_data[batch_offset + (j * kParamsStride)];
        }

        // Apply transformation: dvec_new = T_mat * dvec_cur
        for (SizeType i = 0; i < transform_dim; ++i) {
            for (SizeType j = 0; j <= i && j < transform_dim; ++j) {
                const SizeType power = i - j;
                dvec_new[i] += dvec_cur[j] * taylor_coeffs[power];
            }
        }
        // Update parameter values (except frequency)
        for (SizeType j = 0; j < n_params - 1; ++j) {
            param_vec_data[batch_offset + (j * kParamsStride)] = dvec_new[j];
        }
        // Update frequency with relativistic correction
        param_vec_data[batch_offset + ((n_params - 1) * kParamsStride)] *=
            (1.0 - (dvec_new[n_params - 1] / utils::kCval));
    }
}

// Optimized flat version of shift_params_circular_batch
void shift_params_circular_batch(std::span<double> param_vec_data,
                                 double delta_t,
                                 SizeType n_batch,
                                 SizeType n_params) {
    error_check::check_equal(n_params, 4UL,
                             "nparams should be 4 for circular orbit resolve");
    constexpr auto kParamsStride = 2UL;
    constexpr auto kBatchStride  = 4UL * kParamsStride;
    // Process each batch item
    for (SizeType batch_idx = 0; batch_idx < n_batch; ++batch_idx) {
        const SizeType batch_offset = batch_idx * kBatchStride;

        const auto s_cur = param_vec_data[batch_offset + (0 * kParamsStride)];
        const auto j_cur = param_vec_data[batch_offset + (2 * kParamsStride)];
        const auto a_cur = param_vec_data[batch_offset + (4 * kParamsStride)];
        const auto f_cur = param_vec_data[batch_offset + (6 * kParamsStride)];

        // Circular orbit mask condition
        const auto minus_omega_sq = s_cur / a_cur;
        const auto omega_orb      = std::sqrt(-minus_omega_sq);
        const auto omega_orb_sq   = -minus_omega_sq;
        // Evolve the phase to the new time t_j = t_i + delta_t
        const auto omega_dt = omega_orb * delta_t;
        const auto cos_odt  = std::cos(omega_dt);
        const auto sin_odt  = std::sin(omega_dt);
        const auto a_new = (a_cur * cos_odt) + ((j_cur / omega_orb) * sin_odt);
        const auto j_new = (j_cur * cos_odt) - ((a_cur * omega_orb) * sin_odt);
        const auto s_new = -omega_orb_sq * a_new;
        const auto delta_v = ((a_cur / omega_orb) * sin_odt) -
                             ((j_cur / omega_orb_sq) * (cos_odt - 1.0));
        const auto f_new = f_cur * (1.0 - delta_v * utils::kCval);
        // Update parameter values
        param_vec_data[batch_offset + (0 * kParamsStride)] = s_new;
        param_vec_data[batch_offset + (2 * kParamsStride)] = j_new;
        param_vec_data[batch_offset + (4 * kParamsStride)] = a_new;
        param_vec_data[batch_offset + (6 * kParamsStride)] = f_new;
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

std::vector<double>
generate_branching_pattern(std::span<const std::vector<double>> param_arr,
                           std::span<const double> dparams,
                           const std::vector<ParamLimitType>& param_limits,
                           double tseg_ffa,
                           SizeType nstages,
                           SizeType fold_bins,
                           double tol_bins) {
    error_check::check_equal(param_arr.size(), dparams.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    const auto poly_order = dparams.size();
    const auto& freq_arr  = param_arr.back(); // Last array is frequency
    const auto n0         = freq_arr.size();  // Number of frequency bins

    // Initialize dparam_cur_batch - each frequency gets the same dparams
    std::vector<double> dparam_cur_batch(n0 * poly_order);
    for (SizeType i = 0; i < n0; ++i) {
        std::ranges::copy(dparams, dparam_cur_batch.begin() +
                                       static_cast<IndexType>(i * poly_order));
    }

    // Initialize weights and branching pattern
    std::vector<SizeType> weights(n0, 1);
    std::vector<double> branching_pattern(nstages);

    // Pre-compute parameter ranges
    std::vector<double> param_ranges(poly_order);
    for (SizeType i = 0; i < poly_order; ++i) {
        param_ranges[i] = (param_limits[i][1] - param_limits[i][0]) / 2.0;
    }

    // Current frequency array and arrays for next iteration
    std::vector<double> freq_arr_current(freq_arr.begin(), freq_arr.end());

    for (SizeType prune_level = 1; prune_level <= nstages; ++prune_level) {
        const auto nfreq    = freq_arr_current.size();
        const auto tseg_cur = tseg_ffa * static_cast<double>(prune_level + 1);
        const auto t_ref    = tseg_cur / 2.0;

        // Calculate optimal parameter steps
        std::vector<double> dparam_opt_batch(nfreq * poly_order);
        poly_taylor_step_d_vec(poly_order, tseg_cur, fold_bins, tol_bins,
                               std::span<const double>(freq_arr_current),
                               std::span<double>(dparam_opt_batch), t_ref);

        // Calculate shift bins
        std::vector<double> shift_bins_batch(nfreq * poly_order);
        poly_taylor_shift_d_vec(
            std::span<const double>(dparam_cur_batch),
            std::span<const double>(dparam_opt_batch), tseg_cur, fold_bins,
            std::span<const double>(freq_arr_current), t_ref,
            std::span<double>(shift_bins_batch), nfreq, poly_order);

        // Initialize arrays for next iteration
        std::vector<double> dparam_next_tmp(nfreq * poly_order);
        std::vector<SizeType> n_branch_freq(nfreq, 1);
        std::vector<SizeType> n_branch_nonfreq(nfreq, 1);

        // Determine branching needs
        for (SizeType i = 0; i < nfreq; ++i) {
            for (SizeType j = 0; j < poly_order; ++j) {
                const auto idx             = (i * poly_order) + j;
                const auto needs_branching = shift_bins_batch[idx] >= tol_bins;
                const auto too_large_step =
                    dparam_opt_batch[idx] > param_ranges[j];

                if (!needs_branching || too_large_step) {
                    dparam_next_tmp[idx] = dparam_cur_batch[idx];
                    continue;
                }

                const auto num_points = std::max(
                    1, static_cast<int>(std::ceil(dparam_cur_batch[idx] /
                                                  dparam_opt_batch[idx])));

                if (j == poly_order - 1) { // Frequency parameter
                    n_branch_freq[i] = static_cast<SizeType>(num_points);
                } else {
                    n_branch_nonfreq[i] *= static_cast<SizeType>(num_points);
                }

                dparam_next_tmp[idx] =
                    dparam_cur_batch[idx] / static_cast<double>(num_points);
            }
        }

        // Calculate weighted branching factor
        double weighted_sum          = 0.0;
        SizeType total_weight        = 0;
        SizeType total_freq_branches = 0;

        for (SizeType i = 0; i < nfreq; ++i) {
            const auto weight        = weights[i];
            const auto branch_factor = n_branch_nonfreq[i] * n_branch_freq[i];

            total_weight += weight;
            weighted_sum += static_cast<double>(weight * branch_factor);
            total_freq_branches += n_branch_freq[i];
        }

        branching_pattern[prune_level - 1] =
            weighted_sum / static_cast<double>(total_weight);

        // Prepare arrays for next iteration
        if (prune_level < nstages) {
            std::vector<double> freq_arr_next(total_freq_branches);
            std::vector<SizeType> weights_next(total_freq_branches);
            std::vector<double> dparam_cur_next(total_freq_branches *
                                                poly_order);

            SizeType pos = 0;
            for (SizeType i = 0; i < nfreq; ++i) {
                const auto cfreq  = n_branch_freq[i];
                const auto weight = weights[i] * n_branch_nonfreq[i];

                if (cfreq == 1) {
                    freq_arr_next[pos] = freq_arr_current[i];
                    weights_next[pos]  = weight;
                    std::copy(dparam_next_tmp.begin() +
                                  static_cast<IndexType>(i * poly_order),
                              dparam_next_tmp.begin() +
                                  static_cast<IndexType>((i + 1) * poly_order),
                              dparam_cur_next.begin() +
                                  static_cast<IndexType>(pos * poly_order));
                    ++pos;
                } else if (cfreq == 2) {
                    const auto dparam_cur_freq =
                        dparam_cur_batch[(i * poly_order) + poly_order - 1];
                    const auto delta = 0.25 * dparam_cur_freq;
                    const auto f     = freq_arr_current[i];

                    // First branch
                    freq_arr_next[pos] = f - delta;
                    weights_next[pos]  = weight;
                    std::copy(dparam_next_tmp.begin() +
                                  static_cast<IndexType>(i * poly_order),
                              dparam_next_tmp.begin() +
                                  static_cast<IndexType>((i + 1) * poly_order),
                              dparam_cur_next.begin() +
                                  static_cast<IndexType>(pos * poly_order));
                    ++pos;

                    // Second branch
                    freq_arr_next[pos] = f + delta;
                    weights_next[pos]  = weight;
                    std::copy(dparam_next_tmp.begin() +
                                  static_cast<IndexType>(i * poly_order),
                              dparam_next_tmp.begin() +
                                  static_cast<IndexType>((i + 1) * poly_order),
                              dparam_cur_next.begin() +
                                  static_cast<IndexType>(pos * poly_order));
                    ++pos;
                } else {
                    throw std::runtime_error(
                        std::format("cfreq == {} is not supported", cfreq));
                }
            }

            // Update for next iteration
            freq_arr_current = std::move(freq_arr_next);
            weights          = std::move(weights_next);
            dparam_cur_batch = std::move(dparam_cur_next);
        }
    }

    return branching_pattern;
}

} // namespace loki::psr_utils
