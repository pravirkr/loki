#pragma once

#include <vector>

#include "loki/common/types.hpp"

namespace loki::search {

class PulsarSearchConfig {
public:
    PulsarSearchConfig() = delete;
    PulsarSearchConfig(SizeType nsamps,
                       double tsamp,
                       SizeType nbins,
                       double tol_bins,
                       const std::vector<ParamLimitType>& param_limits,
                       double ducy_max                    = 0.2,
                       double wtsp                        = 1.5,
                       SizeType prune_poly_order          = 3,
                       SizeType prune_n_derivs            = 3,
                       std::optional<SizeType> bseg_brute = std::nullopt,
                       std::optional<SizeType> bseg_ffa   = std::nullopt,
                       bool use_fft_shifts                = true,
                       SizeType branch_max                = 16,
                       int nthreads                       = 1);

    // Getters
    [[nodiscard]] SizeType get_nsamps() const { return m_nsamps; }
    [[nodiscard]] double get_tsamp() const { return m_tsamp; }
    [[nodiscard]] SizeType get_nbins() const { return m_nbins; }
    [[nodiscard]] double get_tol_bins() const { return m_tol_bins; }
    [[nodiscard]] const std::vector<ParamLimitType>& get_param_limits() const {
        return m_param_limits;
    }
    [[nodiscard]] double get_ducy_max() const { return m_ducy_max; }
    [[nodiscard]] double get_wtsp() const { return m_wtsp; }
    [[nodiscard]] SizeType get_prune_poly_order() const {
        return m_prune_poly_order;
    }
    [[nodiscard]] SizeType get_prune_n_derivs() const {
        return m_prune_n_derivs;
    }
    [[nodiscard]] SizeType get_bseg_brute() const { return m_bseg_brute; }
    [[nodiscard]] SizeType get_bseg_ffa() const { return m_bseg_ffa; }
    [[nodiscard]] bool get_use_fft_shifts() const { return m_use_fft_shifts; }
    [[nodiscard]] SizeType get_branch_max() const { return m_branch_max; }
    [[nodiscard]] int get_nthreads() const { return m_nthreads; }
    [[nodiscard]] double get_tseg_brute() const { return m_tseg_brute; }
    [[nodiscard]] double get_tseg_ffa() const { return m_tseg_ffa; }
    [[nodiscard]] SizeType get_niters_ffa() const { return m_niters_ffa; }
    [[nodiscard]] SizeType get_nparams() const { return m_nparams; }
    [[nodiscard]] double get_f_min() const { return m_f_min; }
    [[nodiscard]] double get_f_max() const { return m_f_max; }

    [[nodiscard]] std::vector<double> get_dparams_f(double tseg_cur) const;
    [[nodiscard]] std::vector<double> get_dparams(double tseg_cur) const;
    [[nodiscard]] std::vector<double> get_dparams_lim(double tseg_cur) const;

private:
    void validate() const;
    [[nodiscard]] SizeType get_bseg_brute_default() const;
    [[nodiscard]] SizeType get_bseg_ffa_default() const;

    SizeType m_nsamps;
    double m_tsamp;
    SizeType m_nbins;
    double m_tol_bins;
    std::vector<ParamLimitType> m_param_limits;
    double m_ducy_max;
    double m_wtsp;
    SizeType m_prune_poly_order;
    SizeType m_prune_n_derivs;
    SizeType m_bseg_brute;
    SizeType m_bseg_ffa;
    bool m_use_fft_shifts;
    SizeType m_branch_max;
    int m_nthreads;
    double m_tseg_brute{};
    double m_tseg_ffa{};
    SizeType m_niters_ffa{};
    SizeType m_nparams{};
    double m_f_min{};
    double m_f_max{};
};

} // namespace loki::search