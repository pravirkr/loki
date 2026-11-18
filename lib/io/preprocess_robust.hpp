#pragma once

/**
 * @file preprocess_robust.hpp
 * @brief Backend-neutral part of the robust preprocessing
 * (loki::io::preprocess, PreprocessMethod::kRobust). Internal.
 *
 * The method alternates per-sample passes (block medians and variances,
 * whitening, bad-block sums, the output pass) with work on the block
 * statistics, which are a few hundred thousand values at most. The
 * per-sample passes are RobustStages, one implementation per backend. The
 * block-level work runs on the host for every backend, in this file, so it
 * is the same code (and the same bits) whatever the backend.
 */

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"

namespace loki::io::detail {

/// Smallest statistics block; robust block estimates need a few samples.
inline constexpr SizeType kMinStatBlock = 32;
/// Shortest series the robust method accepts.
inline constexpr SizeType kMinRobustSamples = 16;
/// Smallest bad-block scale in samples and the fewest blocks per scale.
inline constexpr SizeType kMinFlagBlock  = 8;
inline constexpr SizeType kMinFlagBlocks = 8;
/// Variance floor relative to the median block variance.
inline constexpr double kRelVarFloor = 1e-6;

/// Contiguous blocks of `block` samples; the last one may be shorter.
struct BlockGrid {
    SizeType n{0};
    SizeType block{1};
    SizeType nblocks{0};

    BlockGrid(SizeType n_samples, SizeType block_size);

    [[nodiscard]] SizeType begin(SizeType b) const noexcept {
        return b * block;
    }
    [[nodiscard]] SizeType end(SizeType b) const noexcept {
        return b + 1 == nblocks ? n : (b + 1) * block;
    }
    [[nodiscard]] double center(SizeType b) const noexcept {
        return static_cast<double>(begin(b)) +
               (0.5 * static_cast<double>(end(b) - begin(b) - 1));
    }
};

/// Window length in samples for @p window_sec, clamped to [1, n].
[[nodiscard]] SizeType
window_in_samples(double window_sec, double tsamp, SizeType n);

/// Nearest odd count of blocks covering @p window samples (>= 1).
[[nodiscard]] SizeType blocks_per_window(SizeType window, SizeType block);

/// Fewest valid samples a block of @p len on a grid of @p block needs:
/// max(min(block, 3), ceil(min_good * len)). The floor comes from the grid,
/// not from @p len, so a series' last block of one or two samples is not
/// usable on its own (too noisy to anchor the edge of a running median),
/// while a grid of one-bin blocks (the spectral zap path) still is.
[[nodiscard]] SizeType min_valid(double min_good, SizeType len, SizeType block);

/// Median of v[0, n) (mean of the two middle values for even n). Reorders v.
template <typename T> double median_inplace(T* v, SizeType n) noexcept {
    const SizeType h = n / 2;
    std::nth_element(v, v + h, v + n);
    const auto hi = static_cast<double>(v[h]);
    if (n % 2 == 1) {
        return hi;
    }
    const auto lo = static_cast<double>(*std::max_element(v, v + h));
    return 0.5 * (lo + hi);
}

/// Median of a small vector (copied).
[[nodiscard]] double median_of(std::vector<double> v);

/// Gaussian-consistent robust spread: 1.4826 * MAD about @p med.
[[nodiscard]] double robust_spread(const std::vector<double>& v, double med);

/// Running mean of the usable entries (see preprocess_kernels.hpp).
[[nodiscard]] std::vector<float> masked_running_mean(
    std::span<const float> v, std::span<const std::uint8_t> usable, SizeType k);

/// Centred running mean over @p k entries (odd), extending the series by
/// point reflection about its ends so that linear trends are unbiased.
[[nodiscard]] std::vector<float>
reflected_running_mean(std::span<const float> v, SizeType k);

/// Fills entries with `have == 0` by linear interpolation in place.
/// @throws std::invalid_argument if no entry has a value.
void fill_holes(std::span<float> v, std::span<const std::uint8_t> have);

/// Longest run of zeros in @p v.
[[nodiscard]] SizeType longest_zero_run(std::span<const float> v) noexcept;

/**
 * @brief Bad-block decisions of one flagging scale from the per-block sums.
 *
 * @p cnt, @p s1 and @p s2 are the count, sum and sum of squares of the valid
 * whitened residuals of each block of @p g. Blocks with too few valid
 * samples, or whose mean or mean-square statistic is beyond @p nsigma robust
 * spreads, get `bad = 1`; the others 0.
 */
void flag_decisions(const BlockGrid& g,
                    std::span<const SizeType> cnt,
                    std::span<const double> s1,
                    std::span<const double> s2,
                    double nsigma,
                    double min_good,
                    std::span<std::uint8_t> bad);

// --- Spectral zap: host helpers shared by the backends -------------------

/// make_keyed_pcg32 purpose of the noise drawn into zapped bins.
inline constexpr std::uint32_t kZapNoisePurpose = 1;
/// Default removal passes of a hard zap on data with gaps.
inline constexpr SizeType kZapGapPasses = 3;
/// Gap fraction from which a hard zap makes its extra passes.
inline constexpr double kZapMinGapFraction = 1e-4;
/// Samples per chunk of the input hash that seeds the noise fill.
inline constexpr SizeType kZapSeedChunk = SizeType{1} << 20U;

/// Power threshold in Exp(1) units with upper-tail probability equal to the
/// Gaussian upper tail at @p sigma.
[[nodiscard]] double exp_power_threshold(double sigma);

/// Inclusive range of spectral bins.
struct BinRange {
    SizeType lo{0};
    SizeType hi{0};
};

/// The bin ranges (of @p nf, DC excluded, possibly overlapping) inside
/// @p birdies over an observation of @p t_obs seconds; a birdie narrower
/// than a bin covers the nearest bin. Empty ranges are left out.
[[nodiscard]] std::vector<BinRange>
birdie_ranges(std::span<const Birdie> birdies, double t_obs, SizeType nf);

/// 1 for the bins of birdie_ranges(), 0 elsewhere (size @p nf).
[[nodiscard]] std::vector<std::uint8_t>
birdie_bins(std::span<const Birdie> birdies, double t_obs, SizeType nf);

/// The noise seed of a series of @p n samples from the FNV-1a hashes of its
/// kZapSeedChunk-sample chunks, in order.
[[nodiscard]] std::uint64_t
combine_chunk_hashes(SizeType n, std::span<const std::uint64_t> hashes);

/// Geometry of the local mean power of @p nf bins with a running-median
/// width of @p whiten_bins: blocks of `block` bins, `k` blocks per window.
struct PowerGrid {
    SizeType block{1};
    SizeType k{1};
};
[[nodiscard]] PowerGrid power_grid(SizeType whiten_bins, SizeType nf);

/// Samples per block of the valid-sample indicator that GapRetention
/// transforms (the tail shorter than a block is left out).
[[nodiscard]] inline SizeType gap_retention_block(SizeType n) noexcept {
    return std::max<SizeType>(n >> 20U, 1);
}

/**
 * In-window power that zeroing the gaps leaves of a fill that is white
 * over a run of W flagged bins:
 * sum_{|q| < W} |M(q)|^2 (1 - |q| / W) / n^2, with M the spectrum of the
 * valid-sample indicator. (1 - f)^2 for gaps much shorter than 1 / W (f the
 * gap fraction), 1 - f for much longer ones. M is taken from the indicator
 * averaged over blocks of gap_retention_block(n) samples, which is exact
 * enough at the low q that matter.
 */
class GapRetention {
public:
    /// @p fraction: the valid fraction of each whole block.
    explicit GapRetention(std::span<const float> fraction);

    /// Retention for a run of @p width bins.
    [[nodiscard]] double operator()(SizeType width) const;

private:
    std::vector<double> m_s0; // prefix sums of |M(q)|^2 / n^2
    std::vector<double> m_s1; // prefix sums of q |M(q)|^2 / n^2
    double m_dc{0.0};
    double m_total{0.0};
};

/// NormalSampler's inverse-CDF table, for a device noise fill
/// (math::normal_from_lut).
[[nodiscard]] std::span<const float> zap_noise_lut();

/// Edges of the zap self-check bands, Hz: 0, 10, 20, 50, 100, ... up to
/// the Nyquist frequency @p f_nyq.
[[nodiscard]] std::vector<double> zap_band_edges(double f_nyq);

/**
 * @brief The per-sample passes of the robust method on one backend.
 *
 * The series lives in the implementation: the working copy `x` (the input,
 * later the output ts_e), a scratch row `z` (whitened residuals, later the
 * output ts_v) and a sample mask (1 = valid). Block-level arrays are host
 * memory.
 */
class RobustStages {
public:
    RobustStages()                               = default;
    virtual ~RobustStages()                      = default;
    RobustStages(const RobustStages&)            = delete;
    RobustStages& operator=(const RobustStages&) = delete;
    RobustStages(RobustStages&&)                 = delete;
    RobustStages& operator=(RobustStages&&)      = delete;

    /// @throws std::invalid_argument if `x` has a non-finite value.
    virtual void check_finite() = 0;
    /// Sets every sample of the mask to valid.
    virtual void reset_mask() = 0;
    /// Median of the valid samples of every block (see block_medians()).
    virtual void block_medians(const BlockGrid& grid,
                               double min_good,
                               std::span<float> med,
                               std::span<std::uint8_t> usable,
                               std::span<float> good) = 0;
    /// masked_running_median() of the block medians, holes filled.
    [[nodiscard]] virtual std::vector<float>
    running_median(std::span<const float> v,
                   std::span<const std::uint8_t> usable,
                   SizeType k) = 0;
    /// Robust variance of `x - mu` per block (see block_variances()).
    virtual void block_variances(const BlockGrid& grid,
                                 std::span<const float> mu,
                                 double min_good,
                                 std::span<float> var,
                                 std::span<std::uint8_t> usable) = 0;
    /// `z = (x - mu) / sigma` (see whiten()).
    virtual void whiten(const BlockGrid& grid,
                        std::span<const float> mu,
                        std::span<const float> var) = 0;
    /// Count, sum and sum of squares of the valid `z` of every block of
    /// @p g, ignoring `|z| > clip` (0 keeps all).
    virtual void flag_sums(const BlockGrid& g,
                           float clip,
                           std::span<SizeType> cnt,
                           std::span<double> s1,
                           std::span<double> s2) = 0;
    /// Masks every sample of the blocks of @p g with `bad != 0`.
    virtual void mask_blocks(const BlockGrid& g,
                             std::span<const std::uint8_t> bad) = 0;
    /**
     * @brief Zaps periodic RFI and birdies (PreprocessOptions::zap_periodic,
     * birdies) and updates `x`.
     * @return Number of zapped bins.
     */
    virtual SizeType zap(const BlockGrid& grid,
                         std::span<const float> mu,
                         std::span<const float> var,
                         double tsamp,
                         const PreprocessOptions& o) = 0;
    /// Self-check of the last zap() on ts_e (after output()); see
    /// zap_bands(). Backends that do not zap return nothing.
    [[nodiscard]] virtual std::vector<PreprocessReport::ZapBand>
    zap_check(double /*tsamp*/, const PreprocessOptions& /*o*/) {
        return {};
    }
    /// Output pass: ts_e into `x`, ts_v into `z`, with per-block counts of
    /// clipped and masked samples.
    virtual void output(const BlockGrid& grid,
                        std::span<const float> mu,
                        std::span<const float> var,
                        float cf,
                        float clip2,
                        bool mult,
                        std::span<SizeType> n_clip,
                        std::span<SizeType> n_mask) = 0;
    /// Number of samples of the series.
    [[nodiscard]] virtual SizeType nsamps() const = 0;
    /// Longest run of zero-weight samples of ts_v, after output().
    [[nodiscard]] virtual SizeType longest_zero_run() = 0;
};

/// The robust method over the series held by @p stages.
PreprocessReport
run_robust(RobustStages& stages, double tsamp, const PreprocessOptions& o);

} // namespace loki::io::detail
