#pragma once

#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include "loki/loki_types.hpp"

namespace loki::utils {

// Calculate the phase index of the proper time in the folded profile.
SizeType get_phase_idx(FloatType proper_time,
                       FloatType freq,
                       SizeType nbins,
                       FloatType delay);

// Grid size for frequency and its derivatives {f_k, ..., f}.
std::vector<FloatType> poly_taylor_step_f(SizeType nparams,
                                          FloatType tobs,
                                          SizeType fold_bins,
                                          FloatType tol_bins,
                                          FloatType t_ref = 0.0F);

// Grid for parameters {d_k,... d_2, f} based on the Taylor expansion.
std::vector<FloatType> poly_taylor_step_d(SizeType nparams,
                                          FloatType tobs,
                                          SizeType fold_bins,
                                          FloatType tol_bins,
                                          FloatType f_max,
                                          FloatType t_ref = 0.0F);

// Compute the bin shift for parameters {d_k,... d_2, f}.
std::vector<FloatType>
poly_taylor_shift_d(std::span<const FloatType> dparam_cur,
                    std::span<const FloatType> dparam_new,
                    FloatType tobs_new,
                    SizeType fold_bins,
                    FloatType f_cur,
                    FloatType t_ref = 0.0F);

// Shift the kinematical parameters to a new reference time.
std::vector<FloatType> shift_params(std::span<const FloatType> param_vec,
                                    FloatType delta_t);

// Refine a parameter range around a current value with a finer step size.
std::tuple<std::vector<FloatType>, FloatType>
branch_param(FloatType param_cur,
             FloatType dparam_cur,
             FloatType dparam_new,
             FloatType param_min = std::numeric_limits<FloatType>::lowest(),
             FloatType param_max = std::numeric_limits<FloatType>::max());

// Generate an evenly spaced array of values between vmin and vmax.
std::vector<FloatType>
range_param(FloatType vmin, FloatType vmax, FloatType dv);

} // namespace loki::utils