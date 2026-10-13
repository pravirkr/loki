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
 * @p usable is 1 where the block has at least `max(3, ceil(min_good *
 * len))` valid samples; @p good is the valid fraction.
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

/// Power threshold in Exp(1) units with upper-tail probability equal to the
/// Gaussian upper tail at @p sigma.
[[nodiscard]] double exp_power_threshold(double sigma);

/**
 * @brief Zap periodic RFI in the spectrum of @p z, in place.
 *
 * The power spectrum is whitened by its running median over @p whiten_bins;
 * with @p use_threshold, bins above exp_power_threshold(zap_sigma) are
 * zapped, as are all bins inside @p birdies. A zapped bin keeps its phase
 * and gets the local mean power. DC is never zapped.
 *
 * @return Number of zapped bins.
 */
SizeType zap_spectrum(std::span<float> z,
                      double tsamp,
                      bool use_threshold,
                      double zap_sigma,
                      SizeType whiten_bins,
                      std::span<const Birdie> birdies,
                      int nthreads);

} // namespace loki::io::detail
