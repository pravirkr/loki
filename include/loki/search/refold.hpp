#pragma once

/**
 * @file refold.hpp
 * @brief Refold a weighted time series at a candidate's parameters and
 * measure its S/N on a local frequency (and optionally fdot) grid.
 *
 * A rejection test: a refold of the searched data that falls far below the
 * search S/N flags a candidate the search scored on a favourable noise path.
 * A pure-noise candidate also refolds near its search S/N, so a refold never
 * confirms one. Host only (CPU).
 */

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::search {

/// How the phase of the target is computed.
enum class RefoldModel : std::uint8_t {
    /// The Taylor polynomial of the target's derivatives, whatever their
    /// count.
    kTaylor,
    /// The search's own leaf model (psr_utils::leaf_phase): in a
    /// five-parameter search, a circular orbit when its terms are
    /// significant at the propagator significance.
    kLeaf,
};

/// What to fold: one leaf in the layout of an EP result's `param_sets` row,
/// n_params + 2 rows of (value, error): the derivatives highest first down
/// to the acceleration (m/s^k), the frequency (Hz), then (delay, error) and
/// (f0, basis). The parameters refer to the reference time given to refold.
struct RefoldTarget {
    std::vector<double> leaf;
    SizeType n_params{1};
    RefoldModel model{RefoldModel::kTaylor};
    /// As the search's (EP config); used by RefoldModel::kLeaf only.
    double propagator_significance{2.0};
};

/// A target at @p freq (Hz) with kinematic derivatives @p derivs from the
/// acceleration upward (m/s^2, m/s^3, ...; at most four), errors zero.
/// Trailing zero derivatives are dropped.
[[nodiscard]] RefoldTarget refold_taylor_target(double freq,
                                                std::span<const double> derivs);

/// A five-parameter target (accel, jerk, snap, crackle at @p freq) with
/// RefoldModel::kLeaf and errors zero: every nonzero term counts as
/// significant, so it folds as a circular orbit when snap and accel (else
/// crackle and jerk) imply a real angular frequency, else as a polynomial.
[[nodiscard]] RefoldTarget
refold_circular_target(double freq,
                       std::span<const double, 4> derivs,
                       double propagator_significance = 2.0);

struct RefoldOptions {
    /// Phase bins; 0 chooses min(64, max(8, floor(1 / (f * tsamp)))).
    SizeType nbins{0};
    /// Contiguous subintegrations folded exactly at the target phase.
    SizeType nsub{128};
    /// Frequency grid: +- df_steps steps of df_step_bins / (nbins * T);
    /// df_steps 0 covers +- df_span_t / T.
    SizeType df_steps{0};
    double df_span_t{3.0};
    double df_step_bins{0.25};
    /// fdot grid: 2 * fd_steps + 1 values over +- fd_span (Hz/s); 0: none.
    SizeType fd_steps{0};
    double fd_span{0.0};
    /// Also score the best point in this many contiguous time parts (0, 1:
    /// none).
    SizeType split{0};
    int nthreads{1};
};

struct RefoldResult {
    SizeType nbins{0};
    std::string model;    ///< "taylor", "circular_snap" or "circular_crackle".
    double freq{0.0};     ///< The target's frequency (Hz).
    double fdot{0.0};     ///< Its -f * accel / c (Hz/s).
    float snr_in{0.0F};   ///< S/N at the target itself.
    SizeType width_in{0}; ///< Its boxcar width (bins).
    float snr{0.0F};      ///< Best S/N on the grid.
    SizeType width{0};
    SizeType start_bin{0};
    double df{0.0};  ///< Best frequency offset (Hz).
    double dfd{0.0}; ///< Best fdot offset (Hz/s).
    /// S/N and start bin of each time part at the best point.
    std::vector<float> split_snr;
    std::vector<SizeType> split_start_bin;
    /// The best fold: signal over the root of its variance, per bin.
    std::vector<float> profile;
};

/**
 * @brief Refold a series at @p target and scan a local grid around it.
 *
 * Sample i is at time i * tsamp (its start, as the search's brute fold);
 * @p t_ref is on that clock (the middle of the searched span for a reported
 * leaf). Each subintegration is folded exactly at the target phase with
 * algorithms::fold_phase; offsets df, dfd shift subintegration s by
 * df * t_s + dfd * t_s^2 / 2 cycles (t_s its mid time from t_ref) before the
 * subintegrations are summed. S/N is the maximum over boxcar widths
 * 1..nbins/2 and start bins of sum(signal) / sqrt(sum(variance)).
 */
[[nodiscard]] RefoldResult refold(std::span<const float> ts_e,
                                  std::span<const float> ts_v,
                                  double tsamp,
                                  double t_ref,
                                  const RefoldTarget& target,
                                  const RefoldOptions& options = {});

} // namespace loki::search
