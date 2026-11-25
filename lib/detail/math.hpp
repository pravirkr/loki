#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include "loki/common/types.hpp"

#include "lib/detail/keyed_rng.hpp"

namespace loki::math {

/**
 * @brief Compute the factorial of a number.
 *
 * @tparam T The type of the number (integer or floating-point).
 * @param n The input value.
 * @return T The factorial of the number \f$ n! \f$.
 *
 * This functions uses Boost's \c boost::math::factorial for integer types and
 * \c std::tgamma for floating-point types.
 */
template <typename T> constexpr T factorial(const T n) {
    static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic.");
    if (n < static_cast<T>(0)) {
        throw std::invalid_argument(
            "Factorial is not defined for negative numbers.");
    }
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(boost::math::factorial<double>(n));
    } else {
        return std::tgamma(n + 1);
    }
}

template <typename T> T norm_isf(T p) {
    const boost::math::normal_distribution<T> norm_dist;
    return boost::math::quantile(boost::math::complement(norm_dist, p));
}

template <std::floating_point T> class StatLookupTables {
public:
    static constexpr T kMaxMinusLogSf     = 400.0;
    static constexpr T kMinusLogSfRes     = 0.1;
    static constexpr T kChiSqMax          = 300.0;
    static constexpr T kChiSqRes          = 0.5;
    static constexpr SizeType kChiSqMaxDf = 64;

    static constexpr SizeType kNormTableSize =
        static_cast<SizeType>(kMaxMinusLogSf / kMinusLogSfRes) + 1;
    static constexpr SizeType kChiSqTableSize =
        static_cast<SizeType>(kChiSqMax / kChiSqRes) + 1;

    StatLookupTables() { initialize_tables(); }

    T norm_isf(T minus_logsf) const {
        if (minus_logsf < 0) {
            // Below the table: continue its first segment downwards, so the
            // result stays monotonic.
            return m_norm_isf_table[0] +
                   ((minus_logsf / kMinusLogSfRes) *
                    (m_norm_isf_table[1] - m_norm_isf_table[0]));
        }
        const auto pos      = minus_logsf / kMinusLogSfRes;
        const auto pos_int  = static_cast<SizeType>(pos);
        const auto frac_pos = pos - static_cast<T>(pos_int);
        if (minus_logsf < kMaxMinusLogSf) {
            return std::lerp(m_norm_isf_table[pos_int],
                             m_norm_isf_table[pos_int + 1], frac_pos);
        }
        return m_norm_isf_table.back() *
               std::sqrt(minus_logsf / kMaxMinusLogSf);
    }

    T chi_sq_minus_logsf(T chi_sq_score, SizeType df) const {
        if (df == 0 || df > kChiSqMaxDf) {
            throw std::out_of_range("Degrees of freedom out of valid range");
        }
        if (chi_sq_score < 0) {
            throw std::out_of_range("chi_sq_score must be non-negative");
        }
        const auto tab_pos     = chi_sq_score / kChiSqRes;
        const auto tab_pos_int = static_cast<SizeType>(tab_pos);
        const auto frac_pos    = tab_pos - static_cast<T>(tab_pos_int);
        if (chi_sq_score < kChiSqMax) {
            return std::lerp(
                m_chi_sq_minus_logsf_table[df - 1][tab_pos_int],
                m_chi_sq_minus_logsf_table[df - 1][tab_pos_int + 1], frac_pos);
        }
        return m_chi_sq_minus_logsf_table[df - 1].back() * chi_sq_score /
               kChiSqMax;
    }

    // Exact normal inverse survival function using Boost, evaluated in
    // double: exp(-400) underflows in float.
    static T exact_norm_isf(T minus_logsf) {
        const boost::math::normal_distribution<double> norm_dist;
        return static_cast<T>(boost::math::quantile(boost::math::complement(
            norm_dist, std::exp(-static_cast<double>(minus_logsf)))));
    }

    // Exact chi-squared minus log survival function using Boost
    static T exact_chi_sq_minus_logsf(T chi_sq_score, SizeType df) {
        if (df == 0) {
            throw std::out_of_range(
                "Degrees of freedom must be greater than 0");
        }
        // In double: the survival function underflows in float.
        const boost::math::chi_squared_distribution<double> chi_sq_dist(
            static_cast<double>(df));
        return static_cast<T>(
            -std::log(boost::math::cdf(boost::math::complement(
                chi_sq_dist, static_cast<double>(chi_sq_score)))));
    }

private:
    void initialize_tables() {
        // Initialize m_norm_isf_table. The exact value at 0 is -inf (the
        // quantile of probability 1); extrapolate the next two entries so
        // interpolation in [0, kMinusLogSfRes) stays finite.
        for (SizeType i = 1; i < kNormTableSize; ++i) {
            T minus_logsf       = static_cast<T>(i) * kMinusLogSfRes;
            m_norm_isf_table[i] = exact_norm_isf(minus_logsf);
        }
        m_norm_isf_table[0] =
            (T(2) * m_norm_isf_table[1]) - m_norm_isf_table[2];

        // Initialize m_chi_sq_minus_logsf_table
        for (SizeType df = 1; df <= kChiSqMaxDf; ++df) {
            for (SizeType i = 0; i < kChiSqTableSize; ++i) {
                T chi_sq_score = static_cast<T>(i) * kChiSqRes;
                m_chi_sq_minus_logsf_table[df - 1][i] =
                    exact_chi_sq_minus_logsf(chi_sq_score, df);
            }
        }
    }

    std::array<T, kNormTableSize> m_norm_isf_table;
    std::array<std::array<T, kChiSqTableSize>, kChiSqMaxDf>
        m_chi_sq_minus_logsf_table;
};

/** Generate a table of Chebyshev polynomial coefficients and their derivatives.
 * @param order_max Maximum polynomial order (T_0 to T_order_max).
 * @param n_derivs Number of derivatives to compute (0th is the polynomial
 * itself).
 * @return 3D tensor of shape (n_derivs + 1, order_max + 1, order_max + 1).
 */
std::vector<float> generate_cheb_table(SizeType order_max, SizeType n_derivs);

/** Generate generalized Chebyshev polynomials with shift and scale.
 * @param poly_order Maximum polynomial order.
 * @param t0 Shift parameter (center of the domain).
 * @param scale Scale parameter for the domain.
 * @return 2D matrix of shape (poly_order + 1, poly_order + 1).
 */
std::vector<float>
generalized_cheb_pols(SizeType poly_order, float t0, float scale);

constexpr bool is_power_of_two(SizeType n) noexcept {
    return (n != 0U) && ((n & (n - 1)) == 0U);
}

// Compute the connection matrix S for the Chebyshev polynomials.
// @param k_max Maximum polynomial order.
// @param coeff_order  0=ascending, 1=descending
// @return 2D matrix of shape (k_max + 1, k_max + 1).
std::vector<double> compute_connection_matrix_s(SizeType k_max,
                                                SizeType coeff_order = 1U);

using StatTables = StatLookupTables<float>;

/**
 * @brief Normal and bounded-uniform samples drawn from a caller-owned PCG32.
 *
 * Stateless: the caller owns the stream, so results depend only on how the
 * stream was seeded, never on which thread runs the code or on earlier
 * calls. Give every independent piece of work its own stream (e.g. keyed on
 * seed, stage and work item) to stay reproducible under any scheduling.
 *
 * Normals use an inverse-CDF lookup table (2^17 + 1 quantiles of N(0, 1))
 * with linear interpolation: ~1e-5 relative error, several times faster than
 * std::normal_distribution.
 */
class NormalSampler {
public:
    /// Fills `out` with samples of N(mean, stddev^2) drawn from `rng`.
    static void generate(PCG32& rng,
                         std::span<float> out,
                         float mean,
                         float stddev) noexcept {
        const float* __restrict__ lut_ptr = lut().data();
        float* __restrict__ out_ptr       = out.data();
        for (SizeType i = 0; i < out.size(); ++i) {
            out_ptr[i] = normal_from_lut(rng(), lut_ptr, mean, stddev);
        }
    }

    /// The inverse-CDF table (kNormalLutSize quantiles of N(0, 1)), for
    /// device code that draws with normal_from_lut().
    [[nodiscard]] static std::span<const float> lut() {
        std::call_once(s_lut_init_flag, &NormalSampler::init_lut);
        return s_lut;
    }

    /// Uniform index in [0, max_value] (Lemire's bounded method).
    [[nodiscard]] static SizeType uniform_index(PCG32& rng,
                                                SizeType max_value) noexcept {
        const uint64_t range = static_cast<uint64_t>(max_value) + 1ULL;
        uint64_t x           = rng();
        uint64_t m           = x * range;
        auto l               = static_cast<uint32_t>(m);
        if (l < range) {
            const uint64_t t = (1ULL << 32U) % range;
            while (l < t) {
                x = rng();
                m = x * range;
                l = static_cast<uint32_t>(m);
            }
        }
        return static_cast<SizeType>(m >> 32U);
    }

private:
    // Shared lookup table (initialised once, read-only thereafter)
    static inline std::vector<float> s_lut;
    static inline std::once_flag s_lut_init_flag;

    // 2^17 + 1 for high resolution for the lookup table
    static constexpr SizeType kTableSize = kNormalLutSize - 1;

    /// Quantiles of N(0, 1) at linearly spaced probabilities in [0, 1].
    static void init_lut() {
        s_lut.resize(kTableSize + 1);
        const boost::math::normal_distribution<double> n01;
        for (SizeType i = 0; i <= kTableSize; ++i) {
            double p = static_cast<double>(i) / static_cast<double>(kTableSize);
            p        = std::clamp(p, 1e-9, 1.0 - 1e-9);
            s_lut[i] = static_cast<float>(boost::math::quantile(n01, p));
        }
    }
};

// ---------------------------------------------------------------------------
// 1D float32 running filters and z-score normalisation (implemented in
// math.cpp).
// ---------------------------------------------------------------------------

// Threading: every function below takes the number of OpenMP threads to use
// (\p nthreads, values < 1 are treated as 1). The count is passed explicitly
// to each parallel region; it is never clamped to the machine's thread count.

/// Statistic computed over each sliding window by running_filter().
enum class FilterMethod : std::uint8_t { kMean, kMedian };

/**
 * @brief Left/right scale of a distribution.
 *
 * Symmetric estimators (std, IQR, MAD) report the same value on both sides.
 * Double MAD reports the scale of the values below and above the median.
 */
struct ScaleEstimate {
    double left;
    double right;
};

/// Location and scale used by zscore().
struct ZScoreResult {
    double loc;
    ScaleEstimate scale;
};

namespace detail {
/// Algorithm switch points. Exposed so tests and benchmarks can force each
/// code path; the defaults are tuned by bench/running_filter_b.cpp.
struct Tuning {
    /// Windows >= this use the double-heap median; smaller use a sorted array.
    /// bench/running_filter_b.cpp: the heap wins from w = 5 upwards.
    SizeType heap_window{8};
    /// Order statistics on arrays >= this use the parallel radix select.
    SizeType radix_select_size{1U << 16U};
};
Tuning& tuning() noexcept;
} // namespace detail

/**
 * @brief Sliding-window mean or median of a 1D series.
 *
 * Output \c i is the statistic over the \c window samples
 * <tt>in[i - window/2 .. i - window/2 + window - 1]</tt>. Samples outside the
 * series are obtained by reflecting about the edges (numpy "symmetric"
 * padding, so the edge sample is repeated). For an odd window the window is
 * centred; for an even window it extends one more sample to the left. The
 * median of an even window is the mean of the two middle values. A window
 * longer than the series is allowed.
 *
 * Median results are exact and independent of the number of OpenMP threads.
 * The mean uses a double precision running sum seeded per thread chunk.
 *
 * @param in Input samples. Must be finite (the build uses -ffast-math).
 * @param out Output, same length as \p in. Must not overlap \p in.
 * @param window Window length in samples (>= 1).
 * @param method Statistic to compute.
 * @throws std::invalid_argument on empty input, size mismatch, aliasing or a
 * zero window.
 */
void running_filter(std::span<const float> in,
                    std::span<float> out,
                    SizeType window,
                    FilterMethod method = FilterMethod::kMean,
                    int nthreads        = 1);

/**
 * @brief Approximate running_filter() for long windows.
 *
 * Block-averages the input by <tt>ds = window / min_points</tt>, applies
 * running_filter() with width \p min_points to the short series and linearly
 * interpolates the result back (edges are clamped). The effective window is
 * <tt>ds * min_points</tt>. Falls back to the exact filter when
 * <tt>ds == 1</tt> or the series has fewer than \p min_points blocks.
 *
 * @throws std::invalid_argument as running_filter(), or if min_points is 0.
 */
void running_filter_fast(std::span<const float> in,
                         std::span<float> out,
                         SizeType window,
                         FilterMethod method = FilterMethod::kMean,
                         SizeType min_points = 101,
                         int nthreads        = 1);

/**
 * @brief In-place baseline removal: <tt>x[i] -= running_filter(x)[i]</tt>.
 *
 * Gives exactly the same result as subtracting the output of
 * running_filter() / running_filter_fast() but needs no series-sized scratch
 * buffer in the exact path and only a short array in the fast path.
 *
 * @param x Series, modified in place.
 * @param window Window length in samples (>= 1).
 * @param method Statistic to subtract.
 * @param fast Use the approximate running_filter_fast().
 * @param min_points See running_filter_fast().
 */
void subtract_running_filter(std::span<float> x,
                             SizeType window,
                             FilterMethod method = FilterMethod::kMedian,
                             bool fast           = true,
                             SizeType min_points = 101,
                             int nthreads        = 1);

/**
 * @brief Location of a series (0 for LocMethod::kNone).
 * @throws std::invalid_argument if \p x is empty.
 */
[[nodiscard]] double
estimate_loc(std::span<const float> x, LocMethod method, int nthreads = 1);

/**
 * @brief Scale of a series, normalised to the standard deviation of a
 * Gaussian (1 for ScaleMethod::kNone).
 *
 * - kStd: population standard deviation.
 * - kIqr: (q75 - q25) / 1.3490, linear-interpolated quantiles.
 * - kMad: 1.4826 * median(|x - median|); when that is zero, the mean absolute
 *   deviation from the median divided by sqrt(2/pi).
 * - kDoubleMad: MAD of the samples <= median (left) and >= median (right),
 *   each with the same zero fallback.
 *
 * @throws std::invalid_argument if \p x is empty.
 */
[[nodiscard]] ScaleEstimate
estimate_scale(std::span<const float> x, ScaleMethod method, int nthreads = 1);

/**
 * @brief In-place z-score: <tt>x = (x - loc) / scale</tt>.
 *
 * A scale that is not strictly positive and finite leaves the series centred
 * but unscaled. For kDoubleMad the left scale is used for samples below the
 * location and the right scale otherwise.
 *
 * @return The location and scale that were estimated.
 * @throws std::invalid_argument if \p x is empty.
 */
ZScoreResult
zscore(std::span<float> x, LocMethod loc, ScaleMethod scale, int nthreads = 1);

} // namespace loki::math
