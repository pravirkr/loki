#include "loki/search/configs.hpp"

#include <format>
#include <omp.h>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"

namespace loki::search {

class PulsarSearchConfig::Impl {
public:
    Impl(SizeType nsamps,
         double tsamp,
         SizeType nbins,
         double eta,
         const std::vector<ParamLimitType>& param_limits,
         double ducy_max,
         double wtsp,
         bool use_fourier,
         int nthreads,
         double max_process_memory_gb,
         double octave_scale,
         SizeType nbins_max,
         SizeType nbins_min_lossy_bf,
         std::optional<SizeType> bseg_brute,
         std::optional<SizeType> bseg_ffa,
         double snr_min,
         SizeType prune_poly_order,
         double p_orb_min,
         double m_c_max,
         double m_p_min,
         double minimum_snap_cells,
         bool use_conservative_tile)
        : m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_nbins(nbins),
          m_eta(eta),
          m_param_limits(param_limits),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_use_fourier(use_fourier),
          m_nthreads(nthreads),
          m_max_process_memory_gb(max_process_memory_gb),
          m_octave_scale(octave_scale),
          m_nbins_max(nbins_max),
          m_nbins_min_lossy_bf(nbins_min_lossy_bf),
          m_snr_min(snr_min),
          m_prune_poly_order(prune_poly_order),
          m_p_orb_min(p_orb_min),
          m_m_c_max(m_c_max),
          m_m_p_min(m_p_min),
          m_minimum_snap_cells(minimum_snap_cells),
          m_use_conservative_tile(use_conservative_tile),
          m_bseg_brute_explicit(bseg_brute),
          m_bseg_ffa_explicit(bseg_ffa) {
        if (m_param_limits.empty()) {
            throw std::runtime_error("coord_limits must be non-empty");
        }
        m_nbins_f = (m_nbins / 2) + 1;
        m_nparams = m_param_limits.size();
        m_param_names.assign(kParamNames.end() - m_nparams, kParamNames.end());
        m_f_min      = m_param_limits[m_nparams - 1][0];
        m_f_max      = m_param_limits[m_nparams - 1][1];
        m_bseg_brute = bseg_brute.value_or(get_bseg_brute_default());
        m_bseg_ffa   = bseg_ffa.value_or(get_bseg_ffa_default());

        m_nthreads = std::clamp(m_nthreads, 1, omp_get_max_threads());
        validate();
        m_tseg_brute = static_cast<double>(m_bseg_brute) * m_tsamp;
        m_tseg_ffa   = static_cast<double>(m_bseg_ffa) * m_tsamp;
        m_niters_ffa =
            static_cast<SizeType>(std::log2(static_cast<double>(m_bseg_ffa) /
                                            static_cast<double>(m_bseg_brute)));
        m_scoring_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);

        spdlog::debug(
            "PulsarSearchConfigClass: nsamps={}, tsamp={}, nbins={}, eta={}, "
            "ducy_max={}, wtsp={}, use_fourier={}, nthreads={}, snr_min={}, "
            "prune_poly_order={}, "
            "bseg_brute={}, bseg_ffa={}, p_orb_min={}, "
            "minimum_snap_cells={}, use_conservative_tile={}",
            m_nsamps, m_tsamp, m_nbins, m_eta, m_ducy_max, m_wtsp,
            m_use_fourier, m_nthreads, m_snr_min, m_prune_poly_order,
            m_bseg_brute, m_bseg_ffa, m_p_orb_min, m_minimum_snap_cells,
            m_use_conservative_tile);
    }

    // Getters
    SizeType get_nsamps() const { return m_nsamps; }
    double get_tsamp() const { return m_tsamp; }
    double get_tobs() const { return static_cast<double>(m_nsamps) * m_tsamp; }
    SizeType get_nbins() const { return m_nbins; }
    SizeType get_nbins_f() const { return m_nbins_f; }
    double get_eta() const { return m_eta; }
    const std::vector<ParamLimitType>& get_param_limits() const {
        return m_param_limits;
    }
    double get_ducy_max() const { return m_ducy_max; }
    double get_wtsp() const { return m_wtsp; }
    bool get_use_fourier() const { return m_use_fourier; }
    int get_nthreads() const { return m_nthreads; }
    double get_max_process_memory_gb() const { return m_max_process_memory_gb; }
    double get_octave_scale() const { return m_octave_scale; }
    SizeType get_nbins_max() const { return m_nbins_max; }
    SizeType get_nbins_min_lossy_bf() const { return m_nbins_min_lossy_bf; }
    SizeType get_bseg_brute() const { return m_bseg_brute; }
    SizeType get_bseg_ffa() const { return m_bseg_ffa; }
    double get_snr_min() const { return m_snr_min; }
    SizeType get_prune_poly_order() const { return m_prune_poly_order; }
    double get_p_orb_min() const { return m_p_orb_min; }
    double get_m_c_max() const { return m_m_c_max; }
    double get_m_p_min() const { return m_m_p_min; }
    double get_minimum_snap_cells() const { return m_minimum_snap_cells; }
    bool get_use_conservative_tile() const { return m_use_conservative_tile; }

    double get_tseg_brute() const { return m_tseg_brute; }
    double get_tseg_ffa() const { return m_tseg_ffa; }
    SizeType get_niters_ffa() const { return m_niters_ffa; }
    SizeType get_nparams() const { return m_nparams; }
    std::vector<std::string> get_param_names() const { return m_param_names; }
    double get_f_min() const { return m_f_min; }
    double get_f_max() const { return m_f_max; }
    std::vector<SizeType> get_scoring_widths() const {
        return m_scoring_widths;
    }
    SizeType get_n_scoring_widths() const { return m_scoring_widths.size(); }

    // Methods
    double get_x_mass_const() const {
        return 0.005 * std::pow(m_m_p_min + m_m_c_max, 1.0 / 3.0) * m_m_c_max /
               (m_m_p_min + m_m_c_max);
    }
    void set_max_process_memory_gb(double max_process_memory_gb) noexcept {
        error_check::check_greater(max_process_memory_gb, 0,
                                   "max_process_memory_gb must be positive");
        m_max_process_memory_gb = max_process_memory_gb;
    }
    std::vector<double> get_dparams_f(double tseg_cur) const {
        const double t_ref = (m_nparams == 1) ? 0.0 : tseg_cur / 2.0;
        return psr_utils::poly_taylor_step_f(m_nparams, tseg_cur, m_nbins,
                                             m_eta, t_ref);
    }

    std::vector<double> get_dparams(double tseg_cur) const {
        const double t_ref = (m_nparams == 1) ? 0.0 : tseg_cur / 2.0;
        return psr_utils::poly_taylor_step_d_f(m_nparams, tseg_cur, m_nbins,
                                               m_eta, m_f_max, t_ref);
    }

    std::vector<double> get_dparams_lim(double tseg_cur) const {
        const std::vector<double> dparams = get_dparams(tseg_cur);
        std::vector<double> dparams_lim(dparams.size());
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

    PulsarSearchConfig
    get_updated_config(SizeType nbins,
                       double eta,
                       const std::vector<ParamLimitType>& param_limits) const {
        return {m_nsamps,
                m_tsamp,
                nbins,
                eta,
                param_limits,
                m_ducy_max,
                m_wtsp,
                m_use_fourier,
                m_nthreads,
                m_max_process_memory_gb,
                m_octave_scale,
                m_nbins_max,
                m_nbins_min_lossy_bf,
                m_bseg_brute_explicit,
                m_bseg_ffa_explicit,
                m_snr_min,
                m_prune_poly_order,
                m_p_orb_min,
                m_m_c_max,
                m_m_p_min,
                m_minimum_snap_cells,
                m_use_conservative_tile};
    }

private:
    SizeType m_nsamps;
    double m_tsamp;
    SizeType m_nbins;
    SizeType m_nbins_f;
    double m_eta;
    std::vector<ParamLimitType> m_param_limits;
    double m_ducy_max;
    double m_wtsp;
    bool m_use_fourier;
    int m_nthreads;
    double m_max_process_memory_gb;
    double m_octave_scale;
    SizeType m_nbins_max;
    SizeType m_nbins_min_lossy_bf;
    SizeType m_bseg_brute;
    SizeType m_bseg_ffa;
    double m_snr_min;
    SizeType m_prune_poly_order;
    double m_p_orb_min;
    double m_m_c_max;
    double m_m_p_min;
    double m_minimum_snap_cells;
    bool m_use_conservative_tile;
    std::optional<SizeType> m_bseg_brute_explicit;
    std::optional<SizeType> m_bseg_ffa_explicit;

    double m_tseg_brute{};
    double m_tseg_ffa{};
    SizeType m_niters_ffa{};
    SizeType m_nparams{};
    std::vector<std::string> m_param_names;
    double m_f_min{};
    double m_f_max{};
    std::vector<SizeType> m_scoring_widths;

    void validate() const {
        error_check::check_greater(m_nsamps, 0, "nsamps must be positive");
        error_check::check_power_of_2(m_nsamps, "nsamps");
        error_check::check_greater(m_tsamp, 0, "tsamp must be positive");
        error_check::check_greater(m_eta, 0,
                                   "eta (tolerance bins) must be positive");
        error_check::check_greater(m_max_process_memory_gb, 0,
                                   "max_process_memory_gb must be positive");
        error_check::check_greater_equal(
            m_nbins_max, m_nbins,
            "nbins_max must be greater than or equal to nbins");
        error_check::check_power_of_2(m_bseg_brute, "bseg_brute");
        error_check::check_power_of_2(m_bseg_ffa, "bseg_ffa");
        error_check::check_less(m_bseg_brute, m_nsamps,
                                "bseg_brute must be less than nsamps");
        error_check::check_less_equal(
            m_bseg_ffa, m_nsamps,
            "bseg_ffa must be less than or equal to nsamps");
        error_check::check_greater_equal(
            m_bseg_ffa, m_bseg_brute,
            "bseg_ffa must be greater than or equal to bseg_brute");
        error_check::check_greater_equal(m_nparams, 1,
                                         "nparams must be at least 1");
        for (SizeType iparam = 0; iparam < m_nparams; ++iparam) {
            const auto& param_limit = m_param_limits[iparam];
            error_check::check_greater_equal(
                param_limit[1], param_limit[0],
                std::format(
                    "param_limits[{}] must be increasing (got [{}, {}])",
                    iparam, param_limit[0], param_limit[1]));
        }
    }

    SizeType get_bseg_brute_default() const {
        const SizeType init_levels = (m_nparams == 1) ? 1 : 5;
        const auto levels          = static_cast<SizeType>(
            std::log2(static_cast<double>(m_nsamps) * m_tsamp * m_f_min));
        return static_cast<SizeType>(m_nsamps / (1U << (levels - init_levels)));
    }

    SizeType get_bseg_ffa_default() const { return m_nsamps; }
}; // End PulsarSearchConfig::Impl definition

// --- Definitions for PulsarSearchConfig ---
PulsarSearchConfig::PulsarSearchConfig(
    SizeType nsamps,
    double tsamp,
    SizeType nbins,
    double eta,
    const std::vector<ParamLimitType>& param_limits,
    double ducy_max,
    double wtsp,
    bool use_fourier,
    int nthreads,
    double max_process_memory_gb,
    double octave_scale,
    SizeType nbins_max,
    SizeType nbins_min_lossy_bf,
    std::optional<SizeType> bseg_brute,
    std::optional<SizeType> bseg_ffa,
    double snr_min,
    SizeType prune_poly_order,
    double p_orb_min,
    double m_c_max,
    double m_p_min,
    double minimum_snap_cells,
    bool use_conservative_tile)
    : m_impl(std::make_unique<Impl>(nsamps,
                                    tsamp,
                                    nbins,
                                    eta,
                                    param_limits,
                                    ducy_max,
                                    wtsp,
                                    use_fourier,
                                    nthreads,
                                    max_process_memory_gb,
                                    octave_scale,
                                    nbins_max,
                                    nbins_min_lossy_bf,
                                    bseg_brute,
                                    bseg_ffa,
                                    snr_min,
                                    prune_poly_order,
                                    p_orb_min,
                                    m_c_max,
                                    m_p_min,
                                    minimum_snap_cells,
                                    use_conservative_tile)) {}
PulsarSearchConfig::~PulsarSearchConfig()                             = default;
PulsarSearchConfig::PulsarSearchConfig(PulsarSearchConfig&&) noexcept = default;
PulsarSearchConfig&
PulsarSearchConfig::operator=(PulsarSearchConfig&&) noexcept = default;
PulsarSearchConfig::PulsarSearchConfig(const PulsarSearchConfig& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

PulsarSearchConfig&
PulsarSearchConfig::operator=(const PulsarSearchConfig& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
SizeType PulsarSearchConfig::get_nsamps() const noexcept {
    return m_impl->get_nsamps();
}
double PulsarSearchConfig::get_tsamp() const noexcept {
    return m_impl->get_tsamp();
}
double PulsarSearchConfig::get_tobs() const noexcept {
    return m_impl->get_tobs();
}
SizeType PulsarSearchConfig::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
SizeType PulsarSearchConfig::get_nbins_f() const noexcept {
    return m_impl->get_nbins_f();
}
double PulsarSearchConfig::get_eta() const noexcept {
    return m_impl->get_eta();
}
const std::vector<ParamLimitType>&
PulsarSearchConfig::get_param_limits() const noexcept {
    return m_impl->get_param_limits();
}
double PulsarSearchConfig::get_ducy_max() const noexcept {
    return m_impl->get_ducy_max();
}
double PulsarSearchConfig::get_wtsp() const noexcept {
    return m_impl->get_wtsp();
}
bool PulsarSearchConfig::get_use_fourier() const noexcept {
    return m_impl->get_use_fourier();
}
int PulsarSearchConfig::get_nthreads() const noexcept {
    return m_impl->get_nthreads();
}
double PulsarSearchConfig::get_max_process_memory_gb() const noexcept {
    return m_impl->get_max_process_memory_gb();
}
double PulsarSearchConfig::get_octave_scale() const noexcept {
    return m_impl->get_octave_scale();
}
SizeType PulsarSearchConfig::get_nbins_max() const noexcept {
    return m_impl->get_nbins_max();
}
SizeType PulsarSearchConfig::get_nbins_min_lossy_bf() const noexcept {
    return m_impl->get_nbins_min_lossy_bf();
}
SizeType PulsarSearchConfig::get_bseg_brute() const noexcept {
    return m_impl->get_bseg_brute();
}
SizeType PulsarSearchConfig::get_bseg_ffa() const noexcept {
    return m_impl->get_bseg_ffa();
}
double PulsarSearchConfig::get_snr_min() const noexcept {
    return m_impl->get_snr_min();
}
SizeType PulsarSearchConfig::get_prune_poly_order() const noexcept {
    return m_impl->get_prune_poly_order();
}
double PulsarSearchConfig::get_p_orb_min() const noexcept {
    return m_impl->get_p_orb_min();
}
double PulsarSearchConfig::get_m_c_max() const noexcept {
    return m_impl->get_m_c_max();
}
double PulsarSearchConfig::get_m_p_min() const noexcept {
    return m_impl->get_m_p_min();
}
double PulsarSearchConfig::get_minimum_snap_cells() const noexcept {
    return m_impl->get_minimum_snap_cells();
}
bool PulsarSearchConfig::get_use_conservative_tile() const noexcept {
    return m_impl->get_use_conservative_tile();
}
double PulsarSearchConfig::get_tseg_brute() const noexcept {
    return m_impl->get_tseg_brute();
}
double PulsarSearchConfig::get_tseg_ffa() const noexcept {
    return m_impl->get_tseg_ffa();
}
SizeType PulsarSearchConfig::get_niters_ffa() const noexcept {
    return m_impl->get_niters_ffa();
}
SizeType PulsarSearchConfig::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
std::vector<std::string> PulsarSearchConfig::get_param_names() const noexcept {
    return m_impl->get_param_names();
}
double PulsarSearchConfig::get_f_min() const noexcept {
    return m_impl->get_f_min();
}
double PulsarSearchConfig::get_f_max() const noexcept {
    return m_impl->get_f_max();
}
std::vector<SizeType> PulsarSearchConfig::get_scoring_widths() const noexcept {
    return m_impl->get_scoring_widths();
}
SizeType PulsarSearchConfig::get_n_scoring_widths() const noexcept {
    return m_impl->get_n_scoring_widths();
}
double PulsarSearchConfig::get_x_mass_const() const noexcept {
    return m_impl->get_x_mass_const();
}
void PulsarSearchConfig::set_max_process_memory_gb(double max_process_memory_gb) noexcept {
    m_impl->set_max_process_memory_gb(max_process_memory_gb);
}
std::vector<double>
PulsarSearchConfig::get_dparams_f(double tseg_cur) const noexcept {
    return m_impl->get_dparams_f(tseg_cur);
}
std::vector<double>
PulsarSearchConfig::get_dparams(double tseg_cur) const noexcept {
    return m_impl->get_dparams(tseg_cur);
}
std::vector<double>
PulsarSearchConfig::get_dparams_lim(double tseg_cur) const noexcept {
    return m_impl->get_dparams_lim(tseg_cur);
}
PulsarSearchConfig PulsarSearchConfig::get_updated_config(
    SizeType nbins,
    double eta,
    const std::vector<ParamLimitType>& param_limits) const noexcept {
    return m_impl->get_updated_config(nbins, eta, param_limits);
}
} // namespace loki::search