#pragma once

#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::psr_utils {

// Calculate the phase index of the proper time in the folded profile.
SizeType
get_phase_idx(double proper_time, double freq, SizeType nbins, double delay);

// Calculate the (unrounded) phase index of the proper time.
double get_phase_idx_complete(double proper_time,
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

// Compute the bin shift for parameters {d_k,... d_2, f}.
std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType fold_bins,
                                        double f_cur,
                                        double t_ref = 0.0);

// Shift the kinematical parameters to a new reference time.
std::vector<double> shift_params_d(std::span<const double> param_vec,
                                   double delta_t,
                                   SizeType n_out = 3);

// Shift the search parameters vector to a new reference time.
std::tuple<std::vector<double>, double>
shift_params(std::span<const double> param_vec, double delta_t);

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