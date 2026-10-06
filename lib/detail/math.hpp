#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <omp.h>

#include "loki/common/types.hpp"

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
            throw std::out_of_range("minus_logsf must be non-negative");
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
            return std::lerp(m_chi_sq_minus_logsf_table[df][tab_pos_int],
                             m_chi_sq_minus_logsf_table[df][tab_pos_int + 1],
                             frac_pos);
        }
        return m_chi_sq_minus_logsf_table[df - 1].back() * chi_sq_score /
               kChiSqMax;
    }

    // Exact normal inverse survival function using Boost
    static T exact_norm_isf(T minus_logsf) {
        boost::math::normal_distribution<T> norm_dist;
        return boost::math::quantile(
            boost::math::complement(norm_dist, std::exp(-minus_logsf)));
    }

    // Exact chi-squared minus log survival function using Boost
    static T exact_chi_sq_minus_logsf(T chi_sq_score, SizeType df) {
        if (df == 0) {
            throw std::out_of_range(
                "Degrees of freedom must be greater than 0");
        }
        boost::math::chi_squared_distribution<T> chi_sq_dist(
            static_cast<T>(df));
        return -std::log(boost::math::cdf(
            boost::math::complement(chi_sq_dist, chi_sq_score)));
    }

private:
    void initialize_tables() {
        // Initialize m_norm_isf_table
        for (SizeType i = 0; i < kNormTableSize; ++i) {
            T minus_logsf       = static_cast<T>(i) * kMinusLogSfRes;
            m_norm_isf_table[i] = exact_norm_isf(minus_logsf);
        }

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

// A minimal PCG32 random number generator.
class PCG32 {
public:
    using ResultType = uint32_t;
    using StateType  = uint64_t;

    static constexpr StateType kDefaultState  = 0x853c49e6748fea9bULL;
    static constexpr StateType kDefaultStream = 0xda3e39cb94b95bdbULL;
    static constexpr StateType kMult          = 0x5851f42d4c957f2dULL;

    PCG32() { seed_seq(kDefaultState, kDefaultStream); }
    explicit PCG32(StateType seed, StateType stream = kDefaultStream) {
        seed_seq(seed, stream);
    }

    void seed_seq(StateType seed, StateType stream) {
        m_state = 0U;
        m_inc   = (stream << 1U) | 1U;
        operator()();
        m_state += seed;
        operator()();
    }

    // Generates the next random number
    ResultType operator()() {
        const StateType oldstate = m_state;
        m_state                  = (oldstate * kMult) + m_inc;
        const auto xorshifted =
            static_cast<ResultType>(((oldstate >> 18U) ^ oldstate) >> 27U);
        const auto rot = static_cast<ResultType>(oldstate >> 59U);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1U) & 31U));
    }

    static constexpr ResultType min() {
        return std::numeric_limits<ResultType>::min();
    }
    static constexpr ResultType max() {
        return std::numeric_limits<ResultType>::max();
    }

private:
    StateType m_state{};
    StateType m_inc{};
};

/**
 * @brief High-performance thread-safe normal distribution generator.
 *
 * Generates normally distributed random numbers using PCG32 PRNG combined with
 * inverse CDF lookup table and linear interpolation. Each thread maintains
 * independent state, ensuring thread-safety without locks during generation.
 *
 * Performance: 3-10× faster than std::normal_distribution across architectures
 * Quality: Passes Kolmogorov-Smirnov and Anderson-Darling normality tests
 * Accuracy: ~1e-5 relative error for standard normal distribution
 *
 */
class ThreadLocalNormalRNG {
public:
    /**
     * @brief Constructs the generator with a base seed.
     *
     * Each thread will derive its own unique seed from this base seed.
     * The lookup table is initialized on first use (thread-safe).
     *
     * @param base_seed Base seed for seeding per-thread RNGs. Default uses
     * std::random_device for non-deterministic seed.
     */
    explicit ThreadLocalNormalRNG(uint64_t base_seed = std::random_device{}())
        : m_base_seed(base_seed) {
        std::call_once(s_lut_init_flag, &ThreadLocalNormalRNG::init_lut);
    }

    ~ThreadLocalNormalRNG()                                      = default;
    ThreadLocalNormalRNG(const ThreadLocalNormalRNG&)            = delete;
    ThreadLocalNormalRNG& operator=(const ThreadLocalNormalRNG&) = delete;
    ThreadLocalNormalRNG(ThreadLocalNormalRNG&&)                 = delete;
    ThreadLocalNormalRNG& operator=(ThreadLocalNormalRNG&&)      = delete;

    /**
     * @brief Generates normally distributed random numbers.
     *
     * Fills the output span with samples from N(mean, stddev^2).
     * Thread-safe: can be called simultaneously from multiple threads.
     *
     * @param out Output span to fill with generated samples
     * @param mean Mean of the normal distribution
     * @param stddev Standard deviation
     */
    void
    generate(std::span<float> out, float mean, float stddev) const noexcept {
        generate_impl(out.data(), out.size(), mean, stddev);
    }

    /// Fills `out` from `rng` using the shared normal lookup table.
    /// Does not touch the thread-local generator, so each caller can own a
    /// stream. Same interpolation as `generate`.
    static void generate_with(PCG32& rng,
                              std::span<float> out,
                              float mean,
                              float stddev) noexcept {
        std::call_once(s_lut_init_flag, &ThreadLocalNormalRNG::init_lut);
        const auto max_idx = static_cast<float>(s_lut.size() - 2);
        const float* __restrict__ lut_ptr = s_lut.data();
        float* __restrict__ out_ptr       = out.data();
        for (SizeType i = 0; i < out.size(); ++i) {
            const float u_scaled =
                static_cast<float>(rng()) * kInvU32 * max_idx;
            const auto idx   = static_cast<SizeType>(u_scaled);
            const float frac = u_scaled - static_cast<float>(idx);
            const float z =
                std::fma(frac, lut_ptr[idx + 1] - lut_ptr[idx], lut_ptr[idx]);
            out_ptr[i] = std::fma(z, stddev, mean);
        }
    }

    // Generate a random index in [0, max_value]
    [[nodiscard]] SizeType uniform_index(SizeType max_value) const noexcept {
        auto& rng = get_thread_rng(m_base_seed);

        // Lemire's fast bounded random
        const uint64_t range = static_cast<uint64_t>(max_value) + 1ULL;

        uint64_t x = rng();
        uint64_t m = x * range;
        auto l     = static_cast<uint32_t>(m);
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
    // Static shared lookup table (initialized once, read-only thereafter)
    static inline std::vector<float> s_lut;
    static inline std::once_flag s_lut_init_flag;

    // Constants
    static constexpr float kInvU32 =
        1.0F / static_cast<float>(std::numeric_limits<uint32_t>::max());
    // 2^17 + 1 for high resolution for the lookup table
    static constexpr SizeType kTableSize = 1U << 17U;
    static constexpr uint64_t kSeedMix1  = 0x9e3779b97f4a7c15ULL;
    static constexpr uint64_t kSeedMix2  = 0xda3e39cb94b95bdbULL;
    static constexpr uint64_t kSplitMix1 = 0xbf58476d1ce4e5b9ULL;
    static constexpr uint64_t kSplitMix2 = 0x94d049bb133111ebULL;

    uint64_t m_base_seed;

    /**
     * @brief Initializes the lookup table with quantiles of N(0,1).
     *
     * Called exactly once via std::call_once. Precomputes quantiles
     * for uniform probability values in [0, 1] with linear spacing.
     */
    static void init_lut() {
        s_lut.resize(kTableSize + 1);
        const boost::math::normal_distribution<double> n01;
        for (SizeType i = 0; i <= kTableSize; ++i) {
            double p = static_cast<double>(i) / static_cast<double>(kTableSize);
            p        = std::clamp(p, 1e-9, 1.0 - 1e-9);
            s_lut[i] = static_cast<float>(boost::math::quantile(n01, p));
        }
    }

    /**
     * @brief SplitMix64 hash function for seed mixing.
     *
     * High-quality 64-bit hash used to derive per-thread seeds from base seed.
     *
     * @param x Input value to hash
     * @return Hashed 64-bit value
     */
    [[nodiscard]] static uint64_t splitmix64(uint64_t x) noexcept {
        x += kSeedMix1;
        x = (x ^ (x >> 30U)) * kSplitMix1;
        x = (x ^ (x >> 27U)) * kSplitMix2;
        return x ^ (x >> 31U);
    }

    /**
     * @brief Returns thread-local PCG32 generator, initializing if needed.
     *
     * Each thread gets its own PCG32 instance with a unique seed derived
     * from base_seed and thread ID. Initialization happens lazily on first
     * call.
     *
     * @param base_seed Base seed for deriving thread-specific seed
     * @return Reference to thread-local PCG32 generator
     */
    [[nodiscard]] static PCG32& get_thread_rng(uint64_t base_seed) noexcept {
        thread_local PCG32 rng;
        thread_local bool initialized = false;

        if (!initialized) {
            // Derive unique seed for this thread
            const auto tid = static_cast<uint64_t>(omp_get_thread_num());
            const uint64_t thread_seed =
                splitmix64(base_seed ^ (tid * kSeedMix1));
            const uint64_t stream = splitmix64(thread_seed ^ kSeedMix2);

            rng         = PCG32(thread_seed, stream);
            initialized = true;
        }
        return rng;
    }

    // Core generation implementation using LUT + linear interpolation.
    void generate_impl(float* __restrict__ out_ptr,
                       SizeType out_size,
                       float mean,
                       float stddev) const noexcept {
        // Get thread-local RNG
        auto& rng = get_thread_rng(m_base_seed);

        // Maximum valid index for interpolation (we need idx+1 to exist)
        const auto max_idx = static_cast<float>(s_lut.size() - 2);
        const float* __restrict__ lut_ptr = s_lut.data();

        for (SizeType i = 0; i < out_size; ++i) {
            // Get uniform random in [0, max_idx]
            const float u_scaled =
                static_cast<float>(rng()) * kInvU32 * max_idx;
            // Split into integer index and fractional part
            const auto idx   = static_cast<SizeType>(u_scaled);
            const float frac = u_scaled - static_cast<float>(idx);
            // Linear interpolation: z = lut[idx] + frac * (lut[idx+1] -
            // lut[idx])
            const float z =
                std::fma(frac, lut_ptr[idx + 1] - lut_ptr[idx], lut_ptr[idx]);
            // Transform to N(mean, stddev^2): x = mean + stddev * z
            out_ptr[i] = std::fma(z, stddev, mean);
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
