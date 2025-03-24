#include <loki/configs.hpp>

#include <format>
#include <thread>

#include <spdlog/spdlog.h>

#include <loki/psr_utils.hpp>

PulsarSearchConfig::PulsarSearchConfig(
    SizeType nsamps,
    FloatType tsamp,
    SizeType nbins,
    FloatType tol_bins,
    const std::vector<ParamLimitType>& param_limits,
    FloatType ducy_max,
    FloatType wtsp,
    std::optional<SizeType> bseg_brute,
    std::optional<SizeType> bseg_ffa,
    SizeType nthreads)
    : m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_nbins(nbins),
      m_tol_bins(tol_bins),
      m_param_limits(param_limits),
      m_ducy_max(ducy_max),
      m_wtsp(wtsp),
      m_bseg_brute(bseg_brute.value_or(get_bseg_brute_default())),
      m_bseg_ffa(bseg_ffa.value_or(get_bseg_ffa_default())),
      m_nthreads(nthreads) {
    spdlog::info(
        "PulsarSearchConfigClass: nsamps={}, tsamp={}, nbins={}, tol_bins={}, "
        "ducy_max={}, wtsp={}, bseg_brute={}, bseg_ffa={}, nthreads={}",
        m_nsamps, m_tsamp, m_nbins, m_tol_bins, m_ducy_max, m_wtsp,
        m_bseg_brute, m_bseg_ffa, m_nthreads);
    if (m_param_limits.empty()) {
        throw std::runtime_error("coord_limits must be non-empty");
    }
    m_nparams  = m_param_limits.size();
    m_nthreads = std::max<SizeType>(m_nthreads, 1);
    m_nthreads =
        std::min<SizeType>(m_nthreads, std::thread::hardware_concurrency());
    validate();
    m_tseg_brute = static_cast<FloatType>(m_bseg_brute) * m_tsamp;
    m_tseg_ffa   = static_cast<FloatType>(m_bseg_ffa) * m_tsamp;
    m_niters_ffa =
        static_cast<SizeType>(std::log2(static_cast<FloatType>(m_bseg_ffa) /
                                        static_cast<FloatType>(m_bseg_brute)));
    m_f_min = m_param_limits[m_nparams - 1][0];
    m_f_max = m_param_limits[m_nparams - 1][1];
}

std::vector<FloatType>
PulsarSearchConfig::get_dparams_f(FloatType tseg_cur) const {
    const FloatType t_ref = (m_nparams == 1) ? 0.0 : tseg_cur / 2.0;
    return loki::utils::poly_taylor_step_f(m_nparams, tseg_cur, m_nbins,
                                           m_tol_bins, t_ref);
}

std::vector<FloatType>
PulsarSearchConfig::get_dparams(FloatType tseg_cur) const {
    const FloatType t_ref = (m_nparams == 1) ? 0.0 : tseg_cur / 2.0;
    return loki::utils::poly_taylor_step_d(m_nparams, tseg_cur, m_nbins,
                                           m_tol_bins, m_f_max, t_ref);
}

std::vector<FloatType>
PulsarSearchConfig::get_dparams_lim(FloatType tseg_cur) const {
    const std::vector<FloatType> dparams = get_dparams(tseg_cur);
    std::vector<FloatType> dparams_lim(dparams.size());
    for (SizeType iparam = 0; iparam < m_nparams; ++iparam) {
        if (iparam == m_nparams - 1) {
            dparams_lim[iparam] = dparams[iparam];
        }
        dparams_lim[iparam] =
            std::min(dparams[iparam],
                     m_param_limits[iparam][1] - m_param_limits[iparam][0]);
    }
    return dparams_lim;
}

SizeType PulsarSearchConfig::get_bseg_brute_default() const {
    const SizeType init_levels = (m_nsamps == 1) ? 1 : 5;
    const auto levels          = static_cast<SizeType>(
        std::log2(static_cast<FloatType>(m_nsamps) * m_tsamp * m_f_min));
    return static_cast<SizeType>(m_nsamps / (1 << (levels - init_levels)));
}

SizeType PulsarSearchConfig::get_bseg_ffa_default() const { return m_nsamps; }

void PulsarSearchConfig::validate() const {
    if (m_nsamps <= 0) {
        throw std::runtime_error(
            std::format("nsamps must be positive (got {})", m_nsamps));
    }
    if ((m_nsamps & (m_nsamps - 1)) != 0) {
        throw std::runtime_error(
            std::format("nsamps must be power of 2 (got {})", m_nsamps));
    }
    if (m_tsamp <= 0) {
        throw std::runtime_error(
            std::format("tsamp must be positive (got {})", m_tsamp));
    }
    if (m_nbins <= 0) {
        throw std::runtime_error(
            std::format("nbins must be positive (got {})", m_nbins));
    }
    if (m_tol_bins <= 0) {
        throw std::runtime_error(
            std::format("tol_bins must be positive (got {})", m_tol_bins));
    }
    if ((m_bseg_brute & (m_bseg_brute - 1)) != 0) {
        throw std::runtime_error(std::format(
            "bseg_brute must be power of 2 (got {})", m_bseg_brute));
    }
    if ((m_bseg_ffa & (m_bseg_ffa - 1)) != 0) {
        throw std::runtime_error(
            std::format("bseg_ffa must be power of 2 (got {})", m_bseg_ffa));
    }
    if (m_bseg_brute > m_nsamps) {
        throw std::runtime_error(
            std::format("bseg_brute must be less than nsamps (got {} > {})",
                        m_bseg_brute, m_nsamps));
    }
    if (m_bseg_ffa > m_nsamps) {
        throw std::runtime_error(
            std::format("bseg_ffa must be less than nsamps (got {} > {})",
                        m_bseg_ffa, m_nsamps));
    }
    if (m_bseg_ffa <= m_bseg_brute) {
        throw std::runtime_error(std::format(
            "bseg_ffa must be greater than bseg_brute (got {} <= {})",
            m_bseg_ffa, m_bseg_brute));
    }
    if (m_nparams < 1 || m_nparams > 4) {
        throw std::runtime_error(
            std::format("nparams must be between 1 and 4 (got {})", m_nparams));
    }
    for (SizeType iparam = 0; iparam < m_nparams; ++iparam) {
        const auto& param_limit = m_param_limits[iparam];
        if (param_limit[0] >= param_limit[1]) {
            throw std::runtime_error(std::format(
                "param_limits[{}] must be increasing (got [{}, {}])", iparam,
                param_limit[0], param_limit[1]));
        }
    }
}