#pragma once

#include <cstddef>

#include <loki/fold.hpp>
#include <loki/loki_types.hpp>

struct PulsarSearchConfig {
    PulsarSearchConfig() = delete;
    PulsarSearchConfig(SizeType nsamps,
                       float tsamp,
                       SizeType nbins,
                       float tol_bins,
                       const std::vector<ParamLimitType>& param_limits,
                       float ducy_max                     = 0.2F,
                       float wtsp                         = 1.5F,
                       std::optional<SizeType> bseg_brute = std::nullopt,
                       std::optional<SizeType> bseg_ffa   = std::nullopt);

    std::vector<float> get_dparams_f(float tseg_cur) const;
    std::vector<float> get_dparams(float tseg_cur) const;
    std::vector<float> get_dparams_lim(float tseg_cur) const;

    SizeType nsamps;
    float tsamp;
    SizeType nbins;
    float tol_bins;
    std::vector<ParamLimitType> param_limits;
    float ducy_max;
    float wtsp;
    SizeType bseg_brute;
    SizeType bseg_ffa;

    float tseg_brute{};
    float tseg_ffa{};
    SizeType niters_ffa{};
    SizeType nparams{};
    float f_min{};
    float f_max{};

private:
    void validate() const;
    SizeType get_bseg_brute_default() const;
    SizeType get_bseg_ffa_default() const;
};
