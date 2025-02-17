#pragma once

#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include <loki/loki_types.hpp>

namespace loki::utils {

// Calculate the phase index of the proper time in the folded profile.
SizeType
get_phase_idx(float proper_time, float freq, SizeType nbins, float delay);

// Grid size for frequency and its derivatives {f_k, ..., f}.
std::vector<float> poly_taylor_step_f(SizeType nparams,
                                      float tobs,
                                      SizeType fold_bins,
                                      float tol_bins,
                                      float t_ref = 0.0F);

// Grid for parameters {d_k,... d_2, f} based on the Taylor expansion.
std::vector<float> poly_taylor_step_d(SizeType nparams,
                                      float tobs,
                                      SizeType fold_bins,
                                      float tol_bins,
                                      float f_max,
                                      float t_ref = 0.0F);

// Compute the bin shift for parameters {d_k,... d_2, f}.
std::vector<float> poly_taylor_shift_d(std::span<const float> dparam_cur,
                                       std::span<const float> dparam_new,
                                       float tobs_new,
                                       SizeType fold_bins,
                                       float f_cur,
                                       float t_ref = 0.0F);

// Shift the kinematical parameters to a new reference time.
std::vector<float> shift_params(std::span<const float> param_vec,
                                float delta_t);

// Refine a parameter range around a current value with a finer step size.
std::tuple<std::vector<float>, float>
branch_param(float param_cur,
             float dparam_cur,
             float dparam_new,
             float param_min = std::numeric_limits<float>::lowest(),
             float param_max = std::numeric_limits<float>::max());

// Generate an evenly spaced array of values between vmin and vmax.
std::vector<float> range_param(float vmin, float vmax, float dv);

} // namespace loki::utils