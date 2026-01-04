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

void report_leaves_taylor_batch(std::span<double> leaves_batch,
                                SizeType n_batch,
                                SizeType n_params) {
    constexpr SizeType kParamStride = 2U;
    const SizeType leaves_stride    = (n_params + 2) * kParamStride;
    error_check::check_equal(leaves_batch.size(), n_batch * leaves_stride,
                             "leaves_batch size mismatch");
    for (SizeType i = 0; i < n_batch; ++i) {
        const auto leaf_offset = i * leaves_stride;
        const auto v_final =
            leaves_batch[leaf_offset + ((n_params - 1) * kParamStride) + 0];
        const auto dv_final =
            leaves_batch[leaf_offset + ((n_params - 1) * kParamStride) + 1];
        const auto f0_batch =
            leaves_batch[leaf_offset + ((n_params + 1) * kParamStride) + 0];
        const auto s_factor = 1.0 - (v_final / utils::kCval);
        // Gauge transform + error propagation
        for (SizeType j = 0; j < n_params - 1; ++j) {
            const auto param_offset        = leaf_offset + (j * kParamStride);
            const auto param_val           = leaves_batch[param_offset + 0];
            const auto param_err           = leaves_batch[param_offset + 1];
            leaves_batch[param_offset + 0] = param_val / s_factor;
            leaves_batch[param_offset + 1] = std::sqrt(
                std::pow(param_err / s_factor, 2) +
                (std::pow(param_val / (utils::kCval * s_factor * s_factor), 2) *
                 std::pow(dv_final, 2)));
        }
        leaves_batch[leaf_offset + ((n_params - 1) * kParamStride) + 0] =
            f0_batch * s_factor;
        leaves_batch[leaf_offset + ((n_params - 1) * kParamStride) + 1] =
            f0_batch * dv_final / utils::kCval;
    }
}

} // namespace loki::transforms