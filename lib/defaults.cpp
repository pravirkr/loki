#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include <loki/defaults.hpp>
#include <loki/math.hpp>
#include <loki/utils.hpp>

std::vector<float> loki::ffa_shift_ref(std::span<const float> param_vec,
                                       float t_ref) {
    int nparams = static_cast<int>(param_vec.size());

    Eigen::MatrixXf coeffs = Eigen::MatrixXf::Zero(nparams, nparams);
    for (int i = 0; i < nparams; ++i) {
        for (int j = 0; j <= i; ++j) {
            const int power = i - j;
            coeffs(i, j) =
                static_cast<float>(std::pow(t_ref, power) / factorial(power));
        }
    }
    Eigen::VectorXf param_vec_eigen = Eigen::Map<const Eigen::VectorXf>(
        param_vec.data(), static_cast<int>(param_vec.size()));
    Eigen::VectorXf shifted = coeffs * param_vec_eigen;

    return {shifted.data(), shifted.data() + shifted.size()};
}

std::tuple<std::vector<int>, float>
loki::ffa_resolve(std::span<const float> pset_cur,
                  const std::vector<std::vector<float>>& parr_prev,
                  int ffa_level,
                  int latter,
                  float tchunk_init,
                  int nbins) {
    int nparams = static_cast<int>(pset_cur.size());
    double t_ref_prev =
        (latter - 0.5) * std::pow(2.0, ffa_level - 1) * tchunk_init;
    std::vector<float> pset_prev(nparams);
    double delay_rel;

    if (nparams == 1) {
        pset_prev[0] = pset_cur[0];
        delay_rel    = 0.0;
    } else {
        std::vector<double> kvec_cur(nparams + 1, 0.0);
        std::copy(pset_cur.begin(), pset_cur.end() - 1, kvec_cur.begin());

        std::vector<float> kvec_cur_float(kvec_cur.begin(), kvec_cur.end());
        std::vector<float> kvec_prev_float =
            ffa_shift_ref(kvec_cur_float, static_cast<float>(t_ref_prev));

        std::vector<double> kvec_prev(kvec_prev_float.begin(),
                                      kvec_prev_float.end());
        std::copy(kvec_prev.begin(), kvec_prev.end() - 1, pset_prev.begin());
        pset_prev.back() =
            pset_cur.back() * (1.0 + kvec_prev[nparams - 1] / loki::kCval);
        delay_rel = kvec_prev.back() / loki::kCval;
    }

    double phase_rel = loki::get_phase_idx(
        t_ref_prev, pset_prev.back(), nbins, delay_rel);

    std::vector<int> pindex_prev(nparams);
    for (int ip = 0; ip < nparams; ++ip) {
        pindex_prev[ip] =
            loki::find_nearest_sorted_idx(std::span(parr_prev[ip]), pset_prev[ip]);
    }

    return {pindex_prev, phase_rel};
}

void loki::ffa_init(std::span<const float> ts,
                    std::span<float> fold,
                    std::span<const float> param_arr,
                    float tsegment_cur,
                    size_t nbins) {
    //fold_brute_start(ts, std::span<const float> freq_arr, std::span<float> fold,
    //                 size_t chunk_len, size_t nbins, float dt, float t_ref);
}

float loki::freq_step(float tobs, size_t nbins, float f_max, float tol_bins) {
    const auto m_cycle   = tobs * f_max;
    const auto tsamp_min = 1.0F / (f_max * static_cast<float>(nbins));
    return tol_bins * static_cast<float>(std::pow(f_max, 2)) * tsamp_min /
           (m_cycle - 1);
}

float loki::deriv_step(
    float tobs, float tsamp, size_t deriv, float tol_bins, float t_ref) {
    if (deriv < 2) {
        throw std::invalid_argument("deriv must be >= 2");
    }
    const auto dparam = tsamp * static_cast<float>(factorial(deriv)) *
                        loki::kCval / std::pow(tobs - t_ref, deriv);
    return tol_bins * static_cast<float>(dparam);
}
