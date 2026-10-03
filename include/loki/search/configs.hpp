#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::search {

class FFASearchConfig;

/**
 * @brief Plain struct representation of an FFA TOML configuration file.
 */
struct FFATomlConfig {
    // [input]
    std::string timeseries_path;
    bool preprocess{true};
    double filter_window{1.0};

    // [search]
    double f_min{0.5};
    double f_max{100.0};
    std::optional<double> acc_min;
    std::optional<double> acc_max;
    std::optional<double> jerk_min;
    std::optional<double> jerk_max;

    SizeType nbins{64};
    double eta{1.0};
    double ducy_max{0.2};
    double wtsp{1.5};
    double snr_min{5.0};
    bool use_fourier{true};
    bool use_boxcar_kadane{false};

    // [performance]
    std::optional<SizeType> nsamps;
    std::optional<double> tsamp;
    int nthreads{1};
    double max_process_memory_gb{8.0};
    double octave_scale{2.0};
    SizeType nbins_max{1024};
    SizeType nbins_min_lossy_bf{64};
    std::optional<SizeType> bseg_brute;
    std::optional<SizeType> bseg_ffa;
    SizeType max_passing_candidates{1U << 22U}; // 4M

    // [output]
    std::string outdir{"./"};
    std::string prefix{"loki"};

    // [cuda]
    bool use_cuda{false};
    int device_id{0};

    /// @brief Convert parsed TOML parameters into an FFASearchConfig.
    [[nodiscard]] FFASearchConfig
    to_search_config(std::optional<SizeType> override_nsamps = std::nullopt,
                     std::optional<double> override_tsamp = std::nullopt) const;

    /// @brief Load from a TOML file.
    static FFATomlConfig load(const std::filesystem::path& path);

    /// @brief Parse from a TOML string.
    static FFATomlConfig from_string(std::string_view toml_content);

    /// @brief Write a documented default TOML configuration to disk.
    static void write_default(const std::filesystem::path& path);

    /// @brief Get the documented default TOML configuration text.
    static std::string default_toml_string();
};

/**
 * @brief Configuration for Fast Folding Algorithm (FFA) searches.
 *
 * Contains only parameters required for FFA frequency sweeps and single-region
 * searches.
 */
class FFASearchConfig {
public:
    FFASearchConfig() = delete;
    FFASearchConfig(SizeType nsamps,
                    double tsamp,
                    SizeType nbins,
                    double eta,
                    std::span<const ParamLimit> param_limits,
                    double ducy_max                    = 0.2,
                    double wtsp                        = 1.5,
                    bool use_fourier                   = true,
                    int nthreads                       = 1,
                    double max_process_memory_gb       = 8.0,
                    double octave_scale                = 2.0,
                    SizeType nbins_max                 = 1024,
                    SizeType nbins_min_lossy_bf        = 64,
                    std::optional<SizeType> bseg_brute = std::nullopt,
                    std::optional<SizeType> bseg_ffa   = std::nullopt,
                    double snr_min                     = 5.0,
                    SizeType max_passing_candidates    = 1U << 22U, // 4M
                    bool use_boxcar_kadane             = false);

    virtual ~FFASearchConfig();
    FFASearchConfig(FFASearchConfig&&) noexcept;
    FFASearchConfig& operator=(FFASearchConfig&&) noexcept;
    FFASearchConfig(const FFASearchConfig&);
    FFASearchConfig& operator=(const FFASearchConfig&);

    // --- Getters ---
    SizeType get_nsamps() const noexcept;
    double get_tsamp() const noexcept;
    SizeType get_nbins() const noexcept;
    double get_tobs() const noexcept;
    SizeType get_nbins_f() const noexcept;
    double get_eta() const noexcept;
    std::span<const ParamLimit> get_param_limits() const noexcept;
    double get_ducy_max() const noexcept;
    double get_wtsp() const noexcept;
    bool get_use_fourier() const noexcept;
    int get_nthreads() const noexcept;
    double get_max_process_memory_gb() const noexcept;
    double get_octave_scale() const noexcept;
    SizeType get_nbins_max() const noexcept;
    SizeType get_nbins_min_lossy_bf() const noexcept;
    SizeType get_bseg_brute() const noexcept;
    SizeType get_bseg_ffa() const noexcept;
    double get_snr_min() const noexcept;
    SizeType get_max_passing_candidates() const noexcept;
    bool get_use_boxcar_kadane() const noexcept;
    virtual bool get_use_conservative_tile() const noexcept { return false; }
    double get_tseg_brute() const noexcept;
    double get_tseg_ffa() const noexcept;
    SizeType get_niters_ffa() const noexcept;
    SizeType get_nparams() const noexcept;
    [[nodiscard]] std::vector<std::string> get_param_names() const noexcept;
    double get_f_min() const noexcept;
    double get_f_max() const noexcept;
    [[nodiscard]] std::vector<SizeType> get_scoring_widths() const noexcept;
    SizeType get_n_scoring_widths() const noexcept;
    [[nodiscard]] std::vector<float> get_boxcar_kadane_biases() const noexcept;
    SizeType get_n_boxcar_kadane_biases() const noexcept;

    // --- Setters ---
    void set_max_process_memory_gb(double max_process_memory_gb) noexcept;

    // --- Methods ---
    [[nodiscard]] std::vector<double>
    get_dparams_f(double tseg_cur) const noexcept;
    [[nodiscard]] std::vector<double>
    get_dparams(double tseg_cur) const noexcept;
    [[nodiscard]] std::vector<double>
    get_dparams_actual(double tseg_cur) const noexcept;
    [[nodiscard]] std::vector<SizeType>
    get_param_grid_count(double tseg_cur) const noexcept;

    [[nodiscard]] FFASearchConfig
    get_updated_config(SizeType nbins,
                       double eta,
                       std::span<const ParamLimit> param_limits) const;
    [[nodiscard]] FFASearchConfig get_updated_config(SizeType nbins,
                                                     double eta,
                                                     double f_min,
                                                     double f_max) const;

    // --- TOML Helpers ---
    static FFASearchConfig
    from_toml(const std::filesystem::path& path,
              std::optional<SizeType> nsamps = std::nullopt,
              std::optional<double> tsamp    = std::nullopt);

    static FFASearchConfig
    from_toml_string(std::string_view toml_content,
                     std::optional<SizeType> nsamps = std::nullopt,
                     std::optional<double> tsamp    = std::nullopt);

    static void write_default_toml(const std::filesystem::path& path);
    static std::string default_toml_string();

protected:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Configuration for Extreme Pruning (EP) searches.
 *
 * Superset of FFASearchConfig, adding EP-specific pruning, orbit, and threshold
 * configuration.
 */
class EPSearchConfig : public FFASearchConfig {
public:
    EPSearchConfig() = delete;
    EPSearchConfig(SizeType nsamps,
                   double tsamp,
                   SizeType nbins,
                   double eta,
                   std::span<const ParamLimit> param_limits,
                   double ducy_max                    = 0.2,
                   double wtsp                        = 1.5,
                   bool use_fourier                   = true,
                   int nthreads                       = 1,
                   double max_process_memory_gb       = 8.0,
                   double octave_scale                = 2.0,
                   SizeType nbins_max                 = 1024,
                   SizeType nbins_min_lossy_bf        = 64,
                   std::optional<SizeType> bseg_brute = std::nullopt,
                   std::optional<SizeType> bseg_ffa   = std::nullopt,
                   double snr_min                     = 5.0,
                   SizeType max_passing_candidates    = 1U << 22U, // 4M
                   SizeType prune_poly_order          = 3,
                   double p_orb_min                   = 1e-5,
                   double m_c_max                     = 10.0,
                   double m_p_min                     = 1.4,
                   double propagator_significance     = 2.0,
                   double validation_significance     = 5.0,
                   bool use_conservative_tile         = false,
                   bool use_boxcar_kadane             = false);

    explicit EPSearchConfig(FFASearchConfig ffa_cfg,
                            SizeType prune_poly_order      = 3,
                            double p_orb_min               = 1e-5,
                            double m_c_max                 = 10.0,
                            double m_p_min                 = 1.4,
                            double propagator_significance = 2.0,
                            double validation_significance = 5.0,
                            bool use_conservative_tile     = false);

    ~EPSearchConfig() override;
    EPSearchConfig(EPSearchConfig&&) noexcept;
    EPSearchConfig& operator=(EPSearchConfig&&) noexcept;
    EPSearchConfig(const EPSearchConfig&);
    EPSearchConfig& operator=(const EPSearchConfig&);

    // EP-only Getters
    SizeType get_prune_poly_order() const noexcept;
    double get_p_orb_min() const noexcept;
    double get_m_c_max() const noexcept;
    double get_m_p_min() const noexcept;
    double get_propagator_significance() const noexcept;
    double get_validation_significance() const noexcept;
    bool get_use_conservative_tile() const noexcept override;
    double get_x_mass_const() const noexcept;

    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    [[nodiscard]] EPSearchConfig
    get_updated_config(SizeType nbins,
                       double eta,
                       std::span<const ParamLimit> param_limits) const;

    // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
    [[nodiscard]] EPSearchConfig get_updated_config(SizeType nbins,
                                                    double eta,
                                                    double f_min,
                                                    double f_max) const;

private:
    class EPImpl;
    std::unique_ptr<EPImpl> m_ep_impl;
};

/// Backward-compatibility alias: PulsarSearchConfig is identical to
/// EPSearchConfig
using PulsarSearchConfig = EPSearchConfig;

} // namespace loki::search
