#pragma once

#include <vector>

#include "loki/loki_types.hpp"

class PulsarSearchConfig {
public:
    PulsarSearchConfig() = delete;
    PulsarSearchConfig(SizeType nsamps,
                       FloatType tsamp,
                       SizeType nbins,
                       FloatType tol_bins,
                       const std::vector<ParamLimitType>& param_limits,
                       FloatType ducy_max                 = 0.2,
                       FloatType wtsp                     = 1.5,
                       std::optional<SizeType> bseg_brute = std::nullopt,
                       std::optional<SizeType> bseg_ffa   = std::nullopt,
                       SizeType nthreads                  = 1);

    // Getters
    [[nodiscard]] SizeType get_nsamps() const { return m_nsamps; }
    [[nodiscard]] FloatType get_tsamp() const { return m_tsamp; }
    [[nodiscard]] SizeType get_nbins() const { return m_nbins; }
    [[nodiscard]] FloatType get_tol_bins() const { return m_tol_bins; }
    [[nodiscard]] const std::vector<ParamLimitType>& get_param_limits() const {
        return m_param_limits;
    }
    [[nodiscard]] FloatType get_ducy_max() const { return m_ducy_max; }
    [[nodiscard]] FloatType get_wtsp() const { return m_wtsp; }
    [[nodiscard]] SizeType get_bseg_brute() const { return m_bseg_brute; }
    [[nodiscard]] SizeType get_bseg_ffa() const { return m_bseg_ffa; }
    [[nodiscard]] SizeType get_nthreads() const { return m_nthreads; }
    [[nodiscard]] FloatType get_tseg_brute() const { return m_tseg_brute; }
    [[nodiscard]] FloatType get_tseg_ffa() const { return m_tseg_ffa; }
    [[nodiscard]] SizeType get_niters_ffa() const { return m_niters_ffa; }
    [[nodiscard]] SizeType get_nparams() const { return m_nparams; }
    [[nodiscard]] FloatType get_f_min() const { return m_f_min; }
    [[nodiscard]] FloatType get_f_max() const { return m_f_max; }

    [[nodiscard]] std::vector<FloatType>
    get_dparams_f(FloatType tseg_cur) const;
    [[nodiscard]] std::vector<FloatType> get_dparams(FloatType tseg_cur) const;
    [[nodiscard]] std::vector<FloatType>
    get_dparams_lim(FloatType tseg_cur) const;

private:
    void validate() const;
    [[nodiscard]] SizeType get_bseg_brute_default() const;
    [[nodiscard]] SizeType get_bseg_ffa_default() const;

    SizeType m_nsamps;
    FloatType m_tsamp;
    SizeType m_nbins;
    FloatType m_tol_bins;
    std::vector<ParamLimitType> m_param_limits;
    FloatType m_ducy_max;
    FloatType m_wtsp;
    SizeType m_bseg_brute;
    SizeType m_bseg_ffa;
    SizeType m_nthreads;
    FloatType m_tseg_brute{};
    FloatType m_tseg_ffa{};
    SizeType m_niters_ffa{};
    SizeType m_nparams{};
    FloatType m_f_min{};
    FloatType m_f_max{};
};
