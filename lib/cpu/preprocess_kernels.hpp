#pragma once

/**
 * @file preprocess_kernels.hpp
 * @brief Stages of the robust CPU preprocessing (loki::io::preprocess).
 * Internal; exposed for white-box tests.
 *
 * Every stage works on a fixed decomposition into blocks and reduces in a
 * fixed order, so results do not depend on the number of threads. The
 * block geometry and the block-level steps shared with the GPU backend are
 * in lib/io/preprocess_robust.hpp.
 */

#include <cstdint>
#include <span>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

#include "lib/io/preprocess_robust.hpp"

namespace loki::io::detail {

/**
 * @brief Linear interpolation of block values to the samples of block @p b.
 *
 * Values sit at block centres; samples outside the first and last centre
 * take the end values. Writes `grid.end(b) - grid.begin(b)` values to @p out.
 */
void interpolate_block(const BlockGrid& grid,
                       std::span<const float> values,
                       SizeType b,
                       float* out) noexcept;

/**
 * @brief Median of the valid samples (`mask != 0`) of every block.
 *
 * @p usable is 1 where the block has at least `min_valid(min_good, len,
 * block)` valid samples (preprocess_robust.hpp), or, with no mask (every
 * sample valid), any sample at all, so a short tail block of an unmasked
 * series stays usable; @p good is the valid fraction.
 */
void block_medians(std::span<const float> x,
                   std::span<const std::uint8_t> mask,
                   const BlockGrid& grid,
                   double min_good,
                   std::span<float> med,
                   std::span<std::uint8_t> usable,
                   std::span<float> good,
                   int nthreads);

/**
 * @brief Robust variance (Gaussian-consistent MAD squared, with a
 * small-sample correction) of the residual
 * `x - mu` over the valid samples of every block.
 *
 * Blocks without enough valid samples, or with zero spread (constant data),
 * get `usable = 0`.
 */
void block_variances(std::span<const float> x,
                     std::span<const std::uint8_t> mask,
                     const BlockGrid& grid,
                     std::span<const float> mu,
                     double min_good,
                     std::span<float> var,
                     std::span<std::uint8_t> usable,
                     int nthreads);

/**
 * @brief Centred running median / mean of the usable entries of @p v.
 *
 * The window has @p k entries (odd). The mean truncates it at the edges; the
 * median extends the series by point reflection about the first and last
 * usable entries, so a linear trend stays unbiased up to the edges. Entries
 * whose window holds no usable value are filled by linear interpolation
 * between their neighbours (end values at the edges).
 * @throws std::invalid_argument if no entry is usable.
 */
[[nodiscard]] std::vector<float>
masked_running_median(std::span<const float> v,
                      std::span<const std::uint8_t> usable,
                      SizeType k,
                      int nthreads);
// masked_running_mean() is in lib/io/preprocess_robust.hpp.

/// `z = (x - mu) / sigma` with block-level mu and variance interpolated.
void whiten(std::span<const float> x,
            const BlockGrid& grid,
            std::span<const float> mu,
            std::span<const float> var,
            std::span<float> z,
            int nthreads);

/**
 * @brief Multi-scale bad-block mask from whitened residuals @p z.
 *
 * Scales run from short to long. At each scale the mean and mean square of
 * the valid `z` of each block (ignoring single samples with `|z| > clip`,
 * which the per-sample clip handles; 0 keeps all) are turned into
 * unit-normal statistics, and each is compared with the robust centre and
 * spread of that statistic over all blocks at the scale (spread floored at
 * the white-noise value 1). Blocks beyond @p nsigma, or with fewer than
 * @p min_good of their samples still valid, are masked.
 *
 * @param scales Block lengths in samples.
 * @param mask Output, 1 = valid; overwritten.
 */
void flag_bad_blocks(std::span<const float> z,
                     std::span<const SizeType> scales,
                     double nsigma,
                     double min_good,
                     double clip,
                     std::span<std::uint8_t> mask,
                     int nthreads);

/// What zap_spectrum() flags and how it replaces the flagged bins.
struct SpectralZap {
    bool use_threshold{false};  ///< flag bins above the threshold
    double sigma{8.0};          ///< threshold (Gaussian-equivalent)
    SizeType whiten_bins{1001}; ///< running-median width in bins
    std::span<const Birdie> birdies;
    ZapMode mode{ZapMode::kNoise};
    std::uint64_t seed{0}; ///< noise stream seed
    /// Removal passes when the series has gaps (hard modes only).
    SizeType gap_passes{kZapGapPasses};
};

/**
 * @brief Zap periodic RFI in the spectrum of @p z, in place.
 *
 * The power spectrum is whitened by its running median over
 * SpectralZap::whiten_bins; with SpectralZap::use_threshold, bins above
 * exp_power_threshold(sigma) are flagged, as are the bins inside the
 * birdies. DC is never zapped.
 *
 * - ZapMode::kPhaseKept: a flagged bin above the local mean power keeps its
 *   phase and gets the local mean power.
 * - ZapMode::kNoise: every flagged bin is replaced by complex Gaussian noise
 *   of the local mean power, drawn from a stream keyed on (seed, bin), so
 *   the result does not depend on the thread count.
 * - ZapMode::kZero: every flagged bin is set to zero.
 *
 * The hard modes (noise, zero) leave the birdie windows out of the local
 * mean. @p samples (size n, or empty) marks the valid samples with 1; the
 * others are gaps, zero in @p z and on return. With a gap fraction of at
 * least kZapMinGapFraction, a hard zap removes the flagged bins
 * SpectralZap::gap_passes times (re-zeroing the gaps in between) before the
 * fill, so the part of a line that the gaps put back is removed too; the fill
 * level then comes from the cleaned spectrum, raised for the part of the fill
 * that zeroing the gaps removes (between 1 / (1 - f)^2 for short gaps and 1 /
 * (1 - f) for long ones, f the gap fraction, from the gaps' spectrum). @p
 * zapped (size n/2 + 1, or empty) receives 1 for every bin in a birdie window
 * or above the threshold, whatever the mode.
 *
 * @return Number of bins replaced.
 */
SizeType zap_spectrum(std::span<float> z,
                      double tsamp,
                      const SpectralZap& zs,
                      std::span<const std::uint8_t> samples,
                      std::span<std::uint8_t> zapped,
                      int nthreads);

/**
 * @brief Self-check of a zap on its output @p ts (n samples).
 *
 * Whitens the power spectrum of @p ts by the running median of the bins
 * not in @p flagged (size n/2 + 1) and returns, per band (edges 0, 10, 20,
 * 50, 100, ... Hz up to Nyquist), the mean whitened power of the flagged
 * bins over that of the others. Bands without flagged bins are left out.
 */
[[nodiscard]] std::vector<PreprocessReport::ZapBand>
zap_bands(std::span<const float> ts,
          double tsamp,
          std::span<const std::uint8_t> flagged,
          SizeType whiten_bins,
          int nthreads);

} // namespace loki::io::detail
