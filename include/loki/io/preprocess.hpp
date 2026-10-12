#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

namespace loki::io {

/// Preprocessing method applied to a raw timeseries.
enum class PreprocessMethod : std::uint8_t {
    /// Robust dual-array sufficient statistics: `ts_e = (T - mu) g / sigma^2`
    /// and `ts_v = g^2 / sigma^2` with local baseline, local variance and RFI
    /// masking (masked samples have `ts_e = ts_v = 0`).
    kRobust,
    /// Legacy path kept for verification: running-median detrend, one global
    /// z-score, `ts_v = 1`.
    kZScore,
};

/// Instrumental response `g = mu^alpha` of the robust method.
enum class GainModel : std::uint8_t {
    kAdditive,       ///< alpha = 0: g = 1 (calibrated / search data)
    kMultiplicative, ///< alpha = 1: g = mu (uncalibrated total power)
};

/// Known periodic RFI: zap Fourier bins with `|f - freq| <= width / 2`.
struct Birdie {
    double freq{0.0};  ///< Hz
    double width{0.0}; ///< Hz
};

/**
 * @brief Options of preprocess().
 *
 * Windows and block scales are in seconds. See docs/preprocessing.md for the
 * algorithm and guidance on each value.
 */
struct PreprocessOptions {
    PreprocessMethod method{PreprocessMethod::kRobust};
    /// Baseline (running median) window, seconds. Both methods.
    double filter_window{1.0};

    // --- kRobust ---
    GainModel gain_model{GainModel::kAdditive};
    /// Local variance window, seconds; 0 uses the baseline window.
    double variance_window{0.0};
    /// Blocks per window: the baseline is a running median of block medians.
    SizeType window_blocks{101};
    /// Rounds of (statistics -> bad-block flags) before the final statistics.
    SizeType n_iter{2};
    /// Bad-block scales in seconds; empty disables bad-block masking.
    std::vector<double> block_scales{0.016, 0.065, 0.26, 1.05};
    /// Robust z-score above which a block mean or variance is flagged.
    double block_sigma{6.0};
    /// Blocks with fewer valid samples than this fraction are masked.
    double min_good_fraction{0.3};
    /// Samples with |T - mu| / sigma above this are zeroed; 0 disables.
    double clip_sigma{6.0};
    /// Zap Fourier bins whose whitened power exceeds zap_sigma.
    bool zap_periodic{false};
    /// Gaussian-equivalent significance of the periodic zap threshold.
    double zap_sigma{8.0};
    /// Running-median width (Fourier bins) used to whiten the power spectrum.
    SizeType zap_whiten_bins{1001};
    /// Known RFI frequencies, always zapped when non-empty.
    std::vector<Birdie> birdies;

    // --- kZScore ---
    LocMethod loc{LocMethod::kMean};
    ScaleMethod scale{ScaleMethod::kIqr};
    /// Block-averaged running median for long windows (false: exact filter).
    bool fast_median{true};
    SizeType fast_median_min_points{101};

    /// @brief Throws std::invalid_argument on an out-of-range value.
    void validate() const;
};

/**
 * @brief Summary of a preprocess() call.
 *
 * The block-level arrays describe the robust estimates (empty for kZScore):
 * block `b` covers samples `[b * block_size, min((b + 1) * block_size, n))`.
 */
struct PreprocessReport {
    PreprocessMethod method{PreprocessMethod::kRobust};
    SizeType nsamps{0};
    /// Effective baseline / variance windows in samples.
    SizeType baseline_window{0};
    SizeType variance_window{0};
    /// Samples per statistics block (kRobust).
    SizeType block_size{0};
    /// Samples zeroed by bad-block masking.
    SizeType n_masked{0};
    /// Samples zeroed by the per-sample clip (not counting masked ones).
    SizeType n_clipped{0};
    /// Fourier bins zapped (threshold and birdies).
    SizeType n_zapped{0};
    /// Longest contiguous run of zero-weight samples.
    SizeType longest_masked_run{0};
    /// Global robust scale of the raw series.
    double global_scale{0.0};
    /// Output normalisation: `ts_e` and `ts_v` carry `c` and `c^2` so that
    /// the median nonzero `ts_v` is 1. S/N is independent of `c`.
    double norm{1.0};
    std::vector<float> block_mu;
    std::vector<float> block_sigma;
    /// Fraction of valid samples per block.
    std::vector<float> block_good_fraction;

    /// Fraction of samples with zero weight.
    [[nodiscard]] double zero_weight_fraction() const noexcept {
        return nsamps == 0 ? 0.0
                           : static_cast<double>(n_masked + n_clipped) /
                                 static_cast<double>(nsamps);
    }
};

/**
 * @brief Build the folding inputs `ts_e` and `ts_v` from a raw timeseries.
 *
 * Validation happens here, once, so the fold and score hot loops need none:
 * on return `ts_e` is finite, `ts_v` is finite and non-negative, `ts_v == 0`
 * only for masked or clipped samples (where `ts_e == 0` too), and at least one
 * sample has `ts_v > 0`.
 *
 * @param raw Raw samples (finite). May alias @p ts_e.
 * @param tsamp Sample interval in seconds.
 * @param ts_e Output weighted signal, same length as @p raw.
 * @param ts_v Output weights (statistical information), same length.
 * @param options Method and parameters.
 * @param exec Backend; only the CPU is implemented.
 * @throws std::invalid_argument on bad sizes or options, non-finite input,
 * a constant series, or a fully masked series.
 */
PreprocessReport preprocess(std::span<const float> raw,
                            double tsamp,
                            std::span<float> ts_e,
                            std::span<float> ts_v,
                            const PreprocessOptions& options = {},
                            Exec exec                        = {});

} // namespace loki::io
