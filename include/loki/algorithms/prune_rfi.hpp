#pragma once

#include <limits>
#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::algorithms {

/// Large finite float sentinel used to disable harvesting at a given stage.
/// Uses std::numeric_limits<float>::max() for full -ffast-math safety.
inline constexpr float kHarvestDisabled = std::numeric_limits<float>::max();

/// Default half-width in acceleration (m/s^2) for terrestrial birdie windows.
inline constexpr double kBirdieAccelPad = 1.0e-3;

/**
 * @brief True if the harvest threshold is active (not disabled).
 */
[[nodiscard]] constexpr bool is_harvest_enabled(float threshold) noexcept {
    return threshold < std::numeric_limits<float>::max();
}

/**
 * @brief An exclusion window in physical parameter space for the EP search.
 *
 * @details Frequencies are in Hz and accelerations in m/s^2 (kinematic units,
 * as used by the FFA base grid). The default acceleration range covers the
 * entire grid, so a window with only [f_lo, f_hi] set masks a frequency band
 * for all accelerations.
 */
struct ParamWindow {
    double f_lo{};
    double f_hi{};
    double a_lo = std::numeric_limits<double>::lowest();
    double a_hi = std::numeric_limits<double>::max();
};

/**
 * @brief RFI-control configuration for the EP (pruning) algorithm.
 *
 * @details All mechanisms are opt-in. With the defaults, the pruning loop is
 * unchanged.
 *
 * - Pulsar mask: static exclusion windows (from a coarse FFA pass, a birdie
 *   list, etc.) rasterised onto the FFA base grid. Seeds inside a window are
 *   dropped, and branched leaves that resolve to a masked grid cell are
 *   rejected before the shift-add and scoring stages.
 * - Early harvesting: candidates whose score exceeds `harvest_scheme[level-1]`
 *   at an intermediate level are recorded to disk, removed from the tree, and
 *   their neighbourhood is appended to the (per-run) mask.
 * - Stage-consistency veto: leaves whose score gain from a single added
 *   segment is implausibly large for a coherent signal are rejected.
 */
struct PruneRFIConfig {
    /// Static exclusion windows (physical units).
    std::vector<ParamWindow> pulsar_mask;
    /// Also mask f*k and f/k for k in [2, n_harmonics] (0 disables).
    SizeType n_harmonics = 0;
    /// Per-level upper score threshold (size nsegments - 1, aligned with the
    /// pruning threshold scheme). Empty disables harvesting; kHarvestDisabled
    /// disables harvesting at a given level.
    std::vector<float> harvest_scheme;
    /// Half-width of a harvested window in units of the leaf's current tile
    /// size (df, da), in addition to the Doppler sweep over the observation.
    double harvest_mask_ntiles = 4.0;
    /// Cap on the number of recorded harvests per run. Masking continues
    /// beyond the cap.
    SizeType max_harvests = 4096;
    /// Also store the (2, nbins) folded profile of each harvested candidate.
    bool harvest_store_folds = true;
    /// Enable the stage-consistency (impulsive segment) veto.
    bool impulsive_veto = false;
    /// Reject if S_child^2 - S_parent^2 > kappa * S_child^2 / (n_seg + 1).
    /// A brand-new spike (S_parent = 0) is vetoed only when n_seg + 1 > kappa.
    double impulsive_kappa = 6.0;
    /// Do not apply the veto below this pruning level.
    SizeType impulsive_min_level = 6;
    /// Only test leaves whose child score is at least this value.
    float impulsive_min_snr = 8.0F;

    /// @brief Returns true if any mechanism is active.
    [[nodiscard]] bool is_active() const noexcept {
        return !pulsar_mask.empty() || !harvest_scheme.empty() ||
               impulsive_veto;
    }
    /// @brief Returns true if candidates may be recorded to the harvest store.
    [[nodiscard]] bool has_harvest() const noexcept {
        return !harvest_scheme.empty();
    }

    /**
     * @brief Validate the configuration against the number of segments.
     * @throws std::runtime_error on inconsistent input.
     */
    void validate(SizeType nsegments) const;
};

/**
 * @brief Build an exclusion window covering the full Doppler sweep of a source.
 *
 * @details The instantaneous frequency of a source with acceleration `a` over
 * an observation of length `tobs` spans f * (1 -/+ |a| tobs / c) relative to
 * its frequency at either end of the observation. The window is the union of
 * both one-sided sweeps (so the reference epoch of `f` does not matter),
 * padded by `f_pad` (Hz) and `a_pad` (m/s^2).
 *
 * @param f Source frequency (Hz).
 * @param a Source acceleration (m/s^2).
 * @param tobs Observation length (s).
 * @param f_pad Extra half-width in frequency (Hz).
 * @param a_pad Half-width in acceleration (m/s^2). Default (std::numeric_limits<double>::max())
 * masks all accelerations.
 */
[[nodiscard]] ParamWindow
make_pulsar_window(double f,
                   double a,
                   double tobs,
                   double f_pad = 0.0,
                   double a_pad = std::numeric_limits<double>::max());

/**
 * @brief Exclusion window for a terrestrial (zero-acceleration) birdie.
 *
 * @param f Centre frequency (Hz).
 * @param f_pad Half-width in frequency (Hz).
 * @param a_pad Half-width in acceleration (m/s^2); default is a small epsilon
 * around a = 0, not the full acceleration grid.
 */
[[nodiscard]] ParamWindow
make_birdie_window(double f, double f_pad, double a_pad = kBirdieAccelPad);

/**
 * @brief Exclusion window around a harvested candidate.
 *
 * @details Pads frequency by `ntiles * df` plus the Doppler sweep over
 * `reach = max(t_ref, tobs - t_ref)`, and acceleration by `ntiles * da`.
 *
 * @param f Source frequency at `t_ref` (Hz).
 * @param a Source acceleration (m/s^2).
 * @param df Frequency tile half-width (Hz).
 * @param da Acceleration tile half-width (m/s^2).
 * @param t_ref Leaf reference time (s).
 * @param tobs Observation length (s).
 * @param ntiles Tile-width multiplier (must be positive).
 */
[[nodiscard]] ParamWindow make_harvest_window(double f,
                                              double a,
                                              double df,
                                              double da,
                                              double t_ref,
                                              double tobs,
                                              double ntiles);

/**
 * @brief Build a conservative harvest threshold scheme derived from the
 * pruning threshold scheme.
 *
 * @details Stages below `min_level` are disabled (set to `kHarvestDisabled`)
 * to avoid truncating marginal signals and prevent broad early-stage tile masking.
 * Subsequent stages are set to `std::max(min_snr, threshold_scheme[s] + offset)`.
 *
 * @param threshold_scheme Pruning threshold scheme (size nsegments - 1).
 * @param min_level First stage (1-indexed) where harvest is permitted (default 10).
 * @param offset Additive offset above the threshold scheme (default 10.0F).
 * @param min_snr Minimum absolute SNR required for harvesting (default 15.0F).
 */
[[nodiscard]] std::vector<float>
make_default_harvest_scheme(std::span<const float> threshold_scheme,
                            SizeType min_level = 10,
                            float offset       = 10.0F,
                            float min_snr      = 15.0F);

/**
 * @brief Stage-consistency test for a single added segment.
 *
 * @details With `n_seg` segments already coherently summed in the parent, a
 * genuine signal adds roughly S_c^2 / (n_seg + 1) to the squared score per
 * segment. Reject if the observed gain exceeds `kappa` times that share.
 * A brand-new spike (S_parent <= 0) satisfies the inequality only when
 * `n_seg + 1 > kappa`. Leaves below `min_snr` are never vetoed. A negative
 * parent score is treated as zero.
 */
[[nodiscard]] bool is_impulsive_segment(float score_child,
                                        float score_parent,
                                        SizeType n_seg,
                                        double kappa,
                                        float min_snr) noexcept;

} // namespace loki::algorithms
