#pragma once

#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "loki/common/types.hpp"

namespace loki::psr_utils {

// Calculate the phase index of the proper time in the folded profile.
double
get_phase_idx(double proper_time, double freq, SizeType nbins, double delay);

// Calculate the rounded integer phase index of the proper time.
SizeType get_phase_idx_int(double proper_time,
                           double freq,
                           SizeType nbins,
                           double delay);

// Grid size for frequency and its derivatives {f_k, ..., f}.
std::vector<double> poly_taylor_step_f(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double t_ref = 0.0);

// Grid for parameters {d_k,... d_2, f} based on the Taylor expansion.
std::vector<double> poly_taylor_step_d(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double f_max,
                                       double t_ref = 0.0);

xt::xtensor<double, 2>
poly_taylor_step_d_vec(SizeType nparams,
                       double tobs,
                       SizeType fold_bins,
                       double tol_bins,
                       const xt::xtensor<double, 1>& f_max,
                       double t_ref = 0.0);

// Check if a parameter should be split
bool split_f(double df_old,
             double df_new,
             double tobs_new,
             SizeType k,
             double fold_bins,
             double tol_bins,
             double t_ref = 0.0);

// Compute the bin shift for parameters {d_k,... d_2, f}.
std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType fold_bins,
                                        double f_cur,
                                        double t_ref = 0.0);

xt::xtensor<double, 2>
poly_taylor_shift_d_vec(const xt::xtensor<double, 2>& dparam_old,
                        const xt::xtensor<double, 2>& dparam_new,
                        double tobs_new,
                        SizeType fold_bins,
                        const xt::xtensor<double, 1>& f_cur,
                        double t_ref = 0.0);

// Shift the kinematical parameters to a new reference time.
std::vector<double> shift_params_d(std::span<const double> param_vec,
                                   double delta_t,
                                   SizeType n_out = 3);

// Shift the search parameters vector to a new reference time.
std::tuple<std::vector<double>, double>
shift_params(std::span<const double> param_vec, double delta_t);

xt::xtensor<double, 2> shift_params_d_batch(
    const xt::xtensor<double, 2>& param_vec_batch,
    double delta_t,
    SizeType n_out);

std::tuple<xt::xtensor<double, 3>, xt::xtensor<double, 1>>
shift_params_batch(const xt::xtensor<double, 3>& param_vec_batch,
                   double delta_t);

// Circular orbit batch shifting
std::tuple<xt::xtensor<double, 3>, xt::xtensor<double, 1>>
shift_params_circular_batch(const xt::xtensor<double, 3>& param_vec_batch,
                            double delta_t);

// Conversion from Taylor to circular parameters
xt::xtensor<double, 3> convert_taylor_to_circular(
    const xt::xtensor<double, 3>& param_sets);

// Refine a parameter range around a current value with a finer step size.
std::tuple<std::vector<double>, double>
branch_param(double param_cur,
             double dparam_cur,
             double dparam_new,
             double param_min = std::numeric_limits<double>::lowest(),
             double param_max = std::numeric_limits<double>::max());

// Generate an evenly spaced array of values between vmin and vmax.
std::vector<double> range_param(double vmin, double vmax, double dv);

} // namespace loki::psr_utils