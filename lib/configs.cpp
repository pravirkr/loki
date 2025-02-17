#include "loki/loki_types.hpp"
#include <loki/configs.hpp>

#include <spdlog/spdlog.h>

#include <loki/defaults.hpp>
#include <loki/ffa.hpp>
#include <loki/psr_utils.hpp>
#include <loki/utils.hpp>

PulsarSearchConfig::PulsarSearchConfig(
    SizeType nsamps,
    float tsamp,
    SizeType nbins,
    float tol_bins,
    const std::vector<ParamLimitType>& param_limits,
    float ducy_max,
    float wtsp,
    std::optional<SizeType> bseg_brute,
    std::optional<SizeType> bseg_ffa)
    : nsamps(nsamps),
      tsamp(tsamp),
      nbins(nbins),
      tol_bins(tol_bins),
      param_limits(param_limits),
      ducy_max(ducy_max),
      wtsp(wtsp),
      bseg_brute(bseg_brute.value_or(get_bseg_brute_default())),
      bseg_ffa(bseg_ffa.value_or(get_bseg_ffa_default())) {
    if (param_limits.empty()) {
        throw std::runtime_error("coord_limits must be non-empty");
    }
    nparams = param_limits.size();
    validate();

    this->tseg_brute = static_cast<float>(this->bseg_brute) * this->tsamp;
    this->tseg_ffa   = static_cast<float>(this->bseg_ffa) * this->tsamp;
    this->niters_ffa =
        static_cast<SizeType>(std::log2(static_cast<float>(this->bseg_ffa) /
                                        static_cast<float>(this->bseg_brute)));
    this->f_min = param_limits[this->nparams - 1][0];
    this->f_max = param_limits[this->nparams - 1][1];
}

std::vector<float> PulsarSearchConfig::get_dparams_f(float tseg_cur) const {
    const float t_ref = (nparams == 1) ? 0.0F : tseg_cur / 2.0F;
    return loki::utils::poly_taylor_step_f(nparams, tseg_cur, nbins, tol_bins,
                                           t_ref);
};

std::vector<float> PulsarSearchConfig::get_dparams(float tseg_cur) const {
    const float t_ref = (nparams == 1) ? 0.0F : tseg_cur / 2.0F;
    return loki::utils::poly_taylor_step_d(nparams, tseg_cur, nbins, tol_bins,
                                           f_max, t_ref);
};

std::vector<float> PulsarSearchConfig::get_dparams_lim(float tseg_cur) const {
    const std::vector<float> dparams = get_dparams(tseg_cur);
    std::vector<float> dparams_lim(dparams.size());
    for (SizeType iparam = 0; iparam < nparams; ++iparam) {
        if (iparam == nparams - 1) {
            dparams_lim[iparam] = dparams[iparam];
        }
        dparams_lim[iparam] = std::min(
            dparams[iparam], param_limits[iparam][1] - param_limits[iparam][0]);
    }
    return dparams_lim;
};

SizeType PulsarSearchConfig::get_bseg_brute_default() const {
    const SizeType init_levels = (nsamps == 1) ? 1 : 5;
    const auto levels          = static_cast<SizeType>(
        std::log2(static_cast<float>(nsamps) * tsamp * f_min));
    return static_cast<SizeType>(nsamps / (1 << (levels - init_levels)));
}

SizeType PulsarSearchConfig::get_bseg_ffa_default() const { return nsamps; }

void PulsarSearchConfig::validate() const {
    if (nsamps <= 0) {
        throw std::runtime_error("nsamps must be positive");
    }
    if ((nsamps & (nsamps - 1)) != 0) {
        throw std::runtime_error("nsamps must be power of 2");
    }
    if (tsamp <= 0) {
        throw std::runtime_error("tsamp must be positive");
    }
    if (nbins <= 0) {
        throw std::runtime_error("nbins must be positive");
    }
    if (tol_bins <= 0) {
        throw std::runtime_error("tolerance must be positive");
    }
    if ((bseg_brute & (bseg_brute - 1)) != 0) {
        throw std::runtime_error("bseg_brute must be power of 2");
    }
    if ((bseg_ffa & (bseg_ffa - 1)) != 0) {
        throw std::runtime_error("bseg_ffa must be power of 2");
    }
    if (bseg_brute >= nsamps) {
        throw std::runtime_error("bseg_brute must be less than nsamps");
    }
    if (bseg_ffa >= nsamps) {
        throw std::runtime_error("bseg_ffa must be less than nsamps");
    }
    if (bseg_ffa <= bseg_brute) {
        throw std::runtime_error("bseg_ffa must be greater than bseg_brute");
    }
    if (nparams < 1 || nparams > 4) {
        throw std::runtime_error("nparams must be between 1 and 4");
    }
    for (const auto& param_limit : param_limits) {
        if (param_limit[0] >= param_limit[1]) {
            throw std::runtime_error("coord_limits must be increasing");
        }
    }
}