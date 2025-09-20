#include "loki/transforms.hpp"

#include <algorithm>
#include <cmath>

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
    const auto nparams   = param_vec.size();
    const auto n_out_min = nparams == 0 ? nparams : std::min(n_out, nparams);
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

std::vector<double>
shift_taylor_errors(std::span<const double> taylor_error_vec,
                    double delta_t,
                    bool conservative_errors) {
    const auto nparams = taylor_error_vec.size();
    std::vector<double> result(nparams, 0.0);
    if (conservative_errors) {
        // Conservative: sqrt((taylor_error_vec^2) @ (t_mat^2).T)
        for (SizeType i = 0; i < nparams; ++i) {
            double sum_squared = 0.0;
            for (SizeType j = 0; j <= i; ++j) {
                const auto k      = i - j;
                const auto t_elem = std::pow(delta_t, k) /
                                    static_cast<double>(math::factorial(k));
                const auto error_contrib = taylor_error_vec[j] * t_elem;
                sum_squared += error_contrib * error_contrib;
            }
            result[i] = std::sqrt(sum_squared);
        }
    } else {
        // Non-conservative: taylor_error_vec * abs(diag(t_mat))
        for (SizeType i = 0; i < nparams; ++i) {
            result[i] = taylor_error_vec[i];
        }
    }

    return result;
}

std::vector<double>
shift_taylor_errors_batch(std::span<const double> taylor_error_vec,
                          double delta_t,
                          bool conservative_errors,
                          SizeType n_batch,
                          SizeType n_params) {
    std::vector<double> result(n_batch * n_params, 0.0);

    if (!conservative_errors) {
        // Non-conservative: just copy (diagonal elements = 1.0)
        std::ranges::copy(taylor_error_vec, result.begin());
        return result;
    }

    // Conservative case: precompute transformation coefficients
    // Only compute lower triangular part since t_mat[i,j] = 0 for j > i
    std::vector<double> t_coeffs(n_params * n_params, 0.0);

    for (SizeType i = 0; i < n_params; ++i) {
        for (SizeType j = 0; j <= i; ++j) {
            const auto k = i - j;
            const auto t_elem =
                std::pow(delta_t, k) / static_cast<double>(math::factorial(k));
            t_coeffs[(i * n_params) + j] = t_elem;
        }
    }
    // Apply transformation to all batches
    for (SizeType batch = 0; batch < n_batch; ++batch) {
        const SizeType batch_offset = batch * n_params;

        for (SizeType i = 0; i < n_params; ++i) {
            double sum_squared = 0.0;

            for (SizeType j = 0; j <= i; ++j) {
                const auto t_elem = t_coeffs[(i * n_params) + j];
                const auto error_contrib =
                    taylor_error_vec[batch_offset + j] * t_elem;
                sum_squared += error_contrib * error_contrib;
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
    // Copy till acceleration
    for (SizeType i = 0; i < n_params - 1; ++i) {
        taylor_param_vec[i] = param_vec[i];
    }
    std::vector<double> taylor_param_vec_new =
        shift_taylor_params(taylor_param_vec, delta_t, n_params + 1);
    std::vector<double> param_vec_new(param_vec.begin(), param_vec.end());
    for (SizeType i = 0; i < n_params - 1; ++i) {
        param_vec_new[i] = taylor_param_vec_new[i];
    }
    const double v_over_c = taylor_param_vec_new[n_params - 1] / utils::kCval;
    param_vec_new[n_params - 1] = param_vec[n_params - 1] * (1.0 - v_over_c);
    const auto delay_rel        = taylor_param_vec_new[n_params] / utils::kCval;
    return {std::move(param_vec_new), delay_rel};
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

void report_leaves_chebyshev_batch(std::span<double> leaves_batch,
                                   std::pair<double, double> coord_mid,
                                   SizeType n_batch,
                                   SizeType n_params) {
    // TODO: Implement
}
} // namespace loki::transforms