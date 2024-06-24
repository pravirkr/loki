#pragma once

#include <cstddef>
#include <span>

#include <loki/math.hpp>

namespace loki {

std::vector<float> ffa_shift_ref(std::span<const float> param_vec, float t_ref);

std::tuple<std::vector<int>, float>
ffa_resolve(std::span<const float> pset_cur,
            const std::vector<std::vector<float>>& parr_prev,
            int ffa_level,
            int latter,
            float tchunk_init,
            int nbins);

void ffa_init(std::span<const float> ts,
              std::span<float> fold,
              std::span<const float> param_arr,
              float tsegment_cur,
              size_t nbins);

float freq_step(float tobs, size_t nbins, float f_max, float tol_bins);

float deriv_step(
    float tobs, float tsamp, size_t deriv, float tol_bins, float t_ref = 0.0F);

} // namespace loki