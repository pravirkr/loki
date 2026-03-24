#include "loki/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::transforms {

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

std::vector<double> shift_taylor_params(std::span<const double> param_vec,
                                        double delta_t,
                                        SizeType n_out) {
    const auto nparams      = param_vec.size();
    const auto n_out_actual = (n_out == 0) ? nparams : std::min(n_out, nparams);
    std::vector<double> result(n_out_actual, 0.0);

    // We want the last n_out_min rows of the transformation matrix
    const auto start_row = nparams - n_out_actual;

    // Compute result = param_vec @ t_mat.T (for the last n_out_actual rows)
    for (SizeType i = 0; i < n_out_actual; ++i) {
        const auto row = start_row + i;
        double sum     = 0.0;
        for (SizeType j = 0; j <= row; ++j) {
            const auto k = row - j;
            const auto t_elem =
                std::pow(delta_t, k) / static_cast<double>(math::factorial(k));
            sum += param_vec[j] * t_elem;
        }
        result[i] = sum;
    }
    return result;
}

std::vector<double>
shift_taylor_errors_batch(std::span<const double> taylor_error_vec,
                          double delta_t,
                          bool use_conservative_tile,
                          SizeType n_batch,
                          SizeType n_params) {
    std::vector<double> result(n_batch * n_params, 0.0);

    if (!use_conservative_tile) {
        // Non-conservative: taylor_error_vec * abs(diag(t_mat))
        std::ranges::copy(taylor_error_vec, result.begin());
        return result;
    }

    // Conservative: sqrt((taylor_error_vec^2) @ (t_mat^2).T)
    // Only compute lower triangular part since t_mat[i,j] = 0 for j > i
    std::vector<double> t_coeffs_sq(n_params * n_params, 0.0);

    for (SizeType i = 0; i < n_params; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto k = i - j;
            const auto t_elem =
                std::pow(delta_t, k) / static_cast<double>(math::factorial(k));
            t_coeffs_sq[(i * n_params) + j] = t_elem * t_elem;
        }
    }
    // Apply transformation to all batches
    for (SizeType i_batch = 0; i_batch < n_batch; ++i_batch) {
        const SizeType batch_offset = i_batch * n_params;

        for (SizeType i = 0; i < n_params; ++i) {
            double sum_squared = 0.0;

            for (SizeType j = 0; j <= i; ++j) {
                const auto error_j = taylor_error_vec[batch_offset + j];
                const auto t_sq    = t_coeffs_sq[(i * n_params) + j];
                sum_squared += error_j * error_j * t_sq;
            }

            result[batch_offset + i] = std::sqrt(sum_squared);
        }
    }

    return result;
}

std::tuple<std::vector<double>, double>
shift_taylor_params_d_f(std::span<const double> param_vec, double delta_t) {
    const auto n_params = param_vec.size();
    // Create extended taylor parameter vector (n_params + 1)
    std::vector<double> taylor_param_vec(n_params + 1, 0.0);
    // Copy till acceleration (all except last parameter which is frequency)
    std::copy(param_vec.begin(), param_vec.end() - 1, taylor_param_vec.begin());
    // Shift parameters (requesting n_params + 1 outputs)
    std::vector<double> taylor_param_vec_new =
        shift_taylor_params(taylor_param_vec, delta_t, n_params + 1);
    std::vector<double> param_vec_new(param_vec.begin(), param_vec.end());
    // Update kinematic parameters (all except frequency)
    std::copy(taylor_param_vec_new.begin(),
              taylor_param_vec_new.begin() +
                  static_cast<IndexType>(n_params - 1),
              param_vec_new.begin());
    const double v_over_c = taylor_param_vec_new[n_params - 1] / utils::kCval;
    param_vec_new[n_params - 1] = param_vec[n_params - 1] * (1.0 - v_over_c);
    const auto delay_rel        = taylor_param_vec_new[n_params] / utils::kCval;
    return {std::move(param_vec_new), delay_rel};
}

std::vector<double>
shift_taylor_circular_errors_batch(std::span<const double> taylor_error_vec,
                                   double delta_t,
                                   double p_orb_min,
                                   bool use_conservative_tile,
                                   SizeType n_batch,
                                   SizeType n_params) {
    constexpr SizeType kParamsExpected = 6U;
    error_check::check_equal(
        n_params, kParamsExpected,
        "nparams should be 6 for circular orbit propagation");
    if (use_conservative_tile) {
        throw std::logic_error("Conservative tile not implemented for circular "
                               "orbit propagation.");
    }
    std::vector<double> result(n_batch * kParamsExpected, 0.0);
    const double omega_max_sq = std::pow(2.0 * std::numbers::pi / p_orb_min, 2);
    for (SizeType i_batch = 0; i_batch < n_batch; ++i_batch) {
        const SizeType batch_offset = i_batch * n_params;
        const auto sig_d3_i         = taylor_error_vec[batch_offset + 2];
        const auto sig_d2_i         = taylor_error_vec[batch_offset + 3];
        const auto sig_d1_i         = taylor_error_vec[batch_offset + 4];

        const auto sig_d2_j =
            std::sqrt((sig_d2_i * sig_d2_i) +
                      ((delta_t * sig_d3_i) * (delta_t * sig_d3_i)));
        const auto sig_d3_j = std::sqrt((sig_d3_i * sig_d3_i) +
                                        (sig_d2_i * sig_d2_i * omega_max_sq));
        const auto sig_d1_j =
            std::sqrt((sig_d1_i * sig_d1_i) +
                      ((delta_t * sig_d3_i / 2) * (delta_t * sig_d3_i / 2)) +
                      ((delta_t * sig_d2_i) * (delta_t * sig_d2_i)));
        result[batch_offset + 0] = omega_max_sq * sig_d3_j;
        result[batch_offset + 1] = omega_max_sq * sig_d2_j;
        result[batch_offset + 2] = sig_d3_j;
        result[batch_offset + 3] = sig_d2_j;
        result[batch_offset + 4] = sig_d1_j;
        result[batch_offset + 5] = 0;
    }
    return result;
}

void shift_cheb_errors_batch(std::span<double> cheb_error_batch,
                             double scale_next,
                             double scale_cur,
                             SizeType n_batch,
                             SizeType n_params) {
    const auto p = scale_next / scale_cur;
    for (SizeType i_batch = 0; i_batch < n_batch; ++i_batch) {
        const SizeType batch_offset = i_batch * n_params;
        for (SizeType i = 0; i < n_params; ++i) {
            const auto k = n_params - i;
            cheb_error_batch[batch_offset + i] =
                cheb_error_batch[batch_offset + i] * std::pow(p, k);
        }
    }
}

void taylor_to_chebyshev_limits_full(std::span<const double> taylor_limits,
                                     SizeType n_batch,
                                     SizeType n_params,
                                     double ts,
                                     std::span<double> out) {
    auto tl = [&](SizeType i, SizeType j, SizeType k) -> double {
        return taylor_limits[(((i * n_params) + j) * 2) + k];
    };
    auto ol = [&](SizeType i, SizeType j, SizeType k) -> double& {
        return out[(((i * n_params) + j) * 2) + k];
    };

    const double ts2 = ts * ts;
    const double ts3 = ts2 * ts;
    const double ts4 = ts3 * ts;
    const double ts5 = ts4 * ts;

    if (n_params == 2) {
        for (SizeType i = 0; i < n_batch; ++i) {
            const double d1_min = tl(i, 1, 0) * ts;
            const double d1_max = tl(i, 1, 1) * ts;
            const double d2_min = tl(i, 0, 0) * ts2 * 0.5;
            const double d2_max = tl(i, 0, 1) * ts2 * 0.5;
            ol(i, 1, 0)         = d1_min;
            ol(i, 1, 1)         = d1_max;
            ol(i, 0, 0)         = 0.5 * d2_min;
            ol(i, 0, 1)         = 0.5 * d2_max;
        }
        return;
    }
    if (n_params == 3) {
        for (SizeType i = 0; i < n_batch; ++i) {
            const double d1_min = tl(i, 2, 0) * ts;
            const double d1_max = tl(i, 2, 1) * ts;
            const double d2_min = tl(i, 1, 0) * ts2 * 0.5;
            const double d2_max = tl(i, 1, 1) * ts2 * 0.5;
            const double d3_min = tl(i, 0, 0) * ts3 / 6.0;
            const double d3_max = tl(i, 0, 1) * ts3 / 6.0;
            ol(i, 2, 0)         = d1_min + (0.75 * d3_min);
            ol(i, 2, 1)         = d1_max + (0.75 * d3_max);
            ol(i, 1, 0)         = 0.5 * d2_min;
            ol(i, 1, 1)         = 0.5 * d2_max;
            ol(i, 0, 0)         = 0.25 * d3_min;
            ol(i, 0, 1)         = 0.25 * d3_max;
        }
        return;
    }
    if (n_params == 4) {
        for (SizeType i = 0; i < n_batch; ++i) {
            const double d1_min = tl(i, 3, 0) * ts;
            const double d1_max = tl(i, 3, 1) * ts;
            const double d2_min = tl(i, 2, 0) * ts2 * 0.5;
            const double d2_max = tl(i, 2, 1) * ts2 * 0.5;
            const double d3_min = tl(i, 1, 0) * ts3 / 6.0;
            const double d3_max = tl(i, 1, 1) * ts3 / 6.0;
            const double d4_min = tl(i, 0, 0) * ts4 / 24.0;
            const double d4_max = tl(i, 0, 1) * ts4 / 24.0;
            ol(i, 3, 0)         = d1_min + (0.75 * d3_min);
            ol(i, 3, 1)         = d1_max + (0.75 * d3_max);
            ol(i, 2, 0)         = (0.5 * d2_min) + (0.5 * d4_min);
            ol(i, 2, 1)         = (0.5 * d2_max) + (0.5 * d4_max);
            ol(i, 1, 0)         = 0.25 * d3_min;
            ol(i, 1, 1)         = 0.25 * d3_max;
            ol(i, 0, 0)         = 0.125 * d4_min;
            ol(i, 0, 1)         = 0.125 * d4_max;
        }
        return;
    }
    if (n_params == 5) {
        for (SizeType i = 0; i < n_batch; ++i) {
            const double d1_min = tl(i, 4, 0) * ts;
            const double d1_max = tl(i, 4, 1) * ts;
            const double d2_min = tl(i, 3, 0) * ts2 * 0.5;
            const double d2_max = tl(i, 3, 1) * ts2 * 0.5;
            const double d3_min = tl(i, 2, 0) * ts3 / 6.0;
            const double d3_max = tl(i, 2, 1) * ts3 / 6.0;
            const double d4_min = tl(i, 1, 0) * ts4 / 24.0;
            const double d4_max = tl(i, 1, 1) * ts4 / 24.0;
            const double d5_min = tl(i, 0, 0) * ts5 / 120.0;
            const double d5_max = tl(i, 0, 1) * ts5 / 120.0;
            ol(i, 4, 0)         = d1_min + (0.75 * d3_min) + (0.625 * d5_min);
            ol(i, 4, 1)         = d1_max + (0.75 * d3_max) + (0.625 * d5_max);
            ol(i, 3, 0)         = (0.5 * d2_min) + (0.5 * d4_min);
            ol(i, 3, 1)         = (0.5 * d2_max) + (0.5 * d4_max);
            ol(i, 2, 0)         = (0.25 * d3_min) + (0.3125 * d5_min);
            ol(i, 2, 1)         = (0.25 * d3_max) + (0.3125 * d5_max);
            ol(i, 1, 0)         = 0.125 * d4_min;
            ol(i, 1, 1)         = 0.125 * d4_max;
            ol(i, 0, 0)         = 0.0625 * d5_min;
            ol(i, 0, 1)         = 0.0625 * d5_max;
        }
        return;
    }
    throw std::invalid_argument("n_params > 5 not supported");
}

} // namespace loki::transforms