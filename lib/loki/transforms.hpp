#pragma once

#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::transforms {

std::vector<std::vector<double>> precompute_shift_matrix(SizeType nparams,
                                                         double delta_t);

// Shift the kinematic Taylor parameters to a new reference time
// Ordering is [d_k_max, ..., d_1, d_0] where d_k is coefficient of (t -
// t_c)^k/k!
std::vector<double>
shift_taylor_params(std::span<const double> taylor_param_vec,
                    double delta_t,
                    SizeType n_out = 0);

std::vector<double>
shift_taylor_errors(std::span<const double> taylor_error_vec,
                    double delta_t,
                    bool conservative_errors);

std::vector<double>
shift_taylor_errors_batch(std::span<const double> taylor_error_vec,
                          double delta_t,
                          bool conservative_errors,
                          SizeType n_batch,
                          SizeType n_params);

// Shift the kinematic Taylor parameters (with frequency) to a new reference
// time Ordering is [..., j, a, f]
std::tuple<std::vector<double>, double>
shift_taylor_params_d_f(std::span<const double> param_vec, double delta_t);

void report_leaves_taylor_batch(std::span<double> leaves_batch,
                                SizeType n_batch,
                                SizeType n_params);
void report_leaves_chebyshev_batch(std::span<double> leaves_batch,
                                   std::pair<double, double> coord_mid,
                                   SizeType n_batch,
                                   SizeType n_params);

} // namespace loki::transforms