#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::search {

/**
 * @brief Configuration for Pulsar Search algorithms
 *
 * This class contains parameters for both FFA and Pruning algorithms.
 * When using FFA alone, pruning-specific parameters are ignored.
 *
 * EP only Parameters:
 *   - prune_poly_order, p_orb_min, snap_activation_threshold
 *   - use_conservative_grid
 */
class PulsarSearchConfig {
public:
    PulsarSearchConfig() = delete;
    PulsarSearchConfig(SizeType nsamps,
                       double tsamp,
                       SizeType nbins,
                       double eta,
                       const std::vector<ParamLimitType>& param_limits,
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
                       double minimum_snap_cells          = 5.0,
                       bool use_conservative_tile         = false);

    // --- Rule of five: PIMPL ---
    ~PulsarSearchConfig();
    PulsarSearchConfig(PulsarSearchConfig&&) noexcept;
    PulsarSearchConfig& operator=(PulsarSearchConfig&&) noexcept;
    PulsarSearchConfig(const PulsarSearchConfig&);
    PulsarSearchConfig& operator=(const PulsarSearchConfig&);

    // --- Getters ---
    /// @brief Get the number of samples.
    SizeType get_nsamps() const noexcept;
    /// @brief Get the sample time in seconds.
    double get_tsamp() const noexcept;
    /// @brief Get the number of frequency bins.
    SizeType get_nbins() const noexcept;
    /// @brief Get the total observation time in seconds.
    double get_tobs() const noexcept;
    /// @brief Get the number of frequency bins for the FFT.
    SizeType get_nbins_f() const noexcept;
    /// @brief Get the eta parameter.
    double get_eta() const noexcept;
    /// @brief Get the parameter limits (tuples of [min, max] values).
    const std::vector<ParamLimitType>& get_param_limits() const noexcept;
    /// @brief Get the maximum ducy factor.
    double get_ducy_max() const noexcept;
    /// @brief Get the width scale parameter for the boxcar width trials.
    double get_wtsp() const noexcept;
    /// @brief Get whether to use the Fourier domain.
    bool get_use_fourier() const noexcept;
    /// @brief Get the number of threads to use.
    int get_nthreads() const noexcept;
    /// @brief Get the maximum memory per process in GB.
    double get_max_process_memory_gb() const noexcept;
    /// @brief Get the octave scale parameter for the FFA regions.
    double get_octave_scale() const noexcept;
    /// @brief Get the maximum number of fold bins.
    SizeType get_nbins_max() const noexcept;
    /// @brief Get the minimum number of bins for the lossy brute force search.
    SizeType get_nbins_min_lossy_bf() const noexcept;
    /// @brief Get the number of segments for the brute force search.
    SizeType get_bseg_brute() const noexcept;
    /// @brief Get the number of segments for the FFA search.
    SizeType get_bseg_ffa() const noexcept;
    /// @brief Get the minimum SNR threshold.
    double get_snr_min() const noexcept;
    /// @brief Get the maximum number of passing candidates.
    SizeType get_max_passing_candidates() const noexcept;
    /// @brief Get the polynomial order for the EP algorithm.
    SizeType get_prune_poly_order() const noexcept;
    /// @brief Get the minimum orbital period for the circular orbit search.
    double get_p_orb_min() const noexcept;
    /// @brief Get the maximum mass of the companion for the circular orbit
    /// search.
    double get_m_c_max() const noexcept;
    /// @brief Get the minimum mass of the pulsar for the circular orbit search.
    double get_m_p_min() const noexcept;
    /// @brief Get the minimum number of snap cells required for the circular
    /// orbit search to be considered active.
    double get_minimum_snap_cells() const noexcept;
    /// @brief Get whether to use the conservative tile.
    bool get_use_conservative_tile() const noexcept;
    /// @brief Get the segment length for the brute force search.
    double get_tseg_brute() const noexcept;
    /// @brief Get the segment length for the FFA search.
    double get_tseg_ffa() const noexcept;
    /// @brief Get the number of iterations for the FFA search.
    SizeType get_niters_ffa() const noexcept;
    /// @brief Get the number of parameters to search over.
    SizeType get_nparams() const noexcept;
    /// @brief Get the parameter names.
    [[nodiscard]] std::vector<std::string> get_param_names() const noexcept;
    /// @brief Get the minimum frequency.
    double get_f_min() const noexcept;
    /// @brief Get the maximum frequency.
    double get_f_max() const noexcept;
    /// @brief Get the scoring widths.
    [[nodiscard]] std::vector<SizeType> get_scoring_widths() const noexcept;
    /// @brief Get the number of scoring widths.
    SizeType get_n_scoring_widths() const noexcept;
    /// @brief Get the mass constant for the circular orbit search.
    double get_x_mass_const() const noexcept;

    // --- Setters ---
    /// @brief Set the maximum memory per process in GB (Used for GPU memory
    /// allocation).
    void set_max_process_memory_gb(double max_process_memory_gb) noexcept;

    // --- Methods ---
    /// @brief Get the parameter step sizes for the f-based params.
    [[nodiscard]] std::vector<double>
    get_dparams_f(double tseg_cur) const noexcept;
    /// @brief Get the parameter step sizes for the {f,d} based params.
    [[nodiscard]] std::vector<double>
    get_dparams(double tseg_cur) const noexcept;
    /// @brief Get the parameter step sizes (limited) for the {f,d} based
    /// params.
    [[nodiscard]] std::vector<double>
    get_dparams_lim(double tseg_cur) const noexcept;

    /// @brief Get an updated configuration with the given parameters.
    PulsarSearchConfig get_updated_config(
        SizeType nbins,
        double eta,
        const std::vector<ParamLimitType>& param_limits) const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::search