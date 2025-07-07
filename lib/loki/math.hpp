#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/random.hpp>
#include <omp.h>
#include <xsimd/xsimd.hpp>
#include <xtensor/containers/xtensor.hpp>

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
    boost::math::normal_distribution<T> norm_dist;
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
xt::xtensor<float, 3> generate_cheb_table(SizeType order_max,
                                          SizeType n_derivs);

/** Generate generalized Chebyshev polynomials with shift and scale.
 * @param poly_order Maximum polynomial order.
 * @param t0 Shift parameter (center of the domain).
 * @param scale Scale parameter for the domain.
 * @return 2D matrix of shape (poly_order + 1, poly_order + 1).
 */
xt::xtensor<float, 2>
generalized_cheb_pols(SizeType poly_order, float t0, float scale);

constexpr bool is_power_of_two(SizeType n) noexcept {
    return (n != 0U) && ((n & (n - 1)) == 0U);
}

using StatTables = StatLookupTables<float>;

class ThreadSafeRNGBase {
protected:
    // Use boost::random is much faster than std::random.
    std::vector<boost::random::mt19937> m_rngs;

public:
    explicit ThreadSafeRNGBase(
        unsigned int base_seed = std::random_device{}()) {
        const int max_threads = omp_get_max_threads();
        if (max_threads <= 0) {
            throw std::runtime_error("OpenMP: Invalid thread count");
        }
        m_rngs.reserve(max_threads);
        // Use std::seed_seq to generate high-quality, non-correlated seeds for
        // each thread.
        std::seed_seq seq{base_seed};
        std::vector<uint32_t> thread_seeds(max_threads);
        seq.generate(thread_seeds.begin(), thread_seeds.end());
        for (int i = 0; i < max_threads; ++i) {
            m_rngs.emplace_back(thread_seeds[i]);
        }
    }

    ThreadSafeRNGBase(const ThreadSafeRNGBase&)            = delete;
    ThreadSafeRNGBase& operator=(const ThreadSafeRNGBase&) = delete;
    ThreadSafeRNGBase(ThreadSafeRNGBase&&)                 = default;
    ThreadSafeRNGBase& operator=(ThreadSafeRNGBase&&)      = default;
    virtual ~ThreadSafeRNGBase()                           = default;

    boost::random::mt19937& get_engine_for_current_thread() {
        const int tid = omp_get_thread_num();
        if (tid < 0 || tid >= static_cast<int>(m_rngs.size())) {
            throw std::out_of_range("Invalid OpenMP thread id");
        }
        return m_rngs[tid];
    }

    virtual void generate_normal_dist_range(std::span<float> range,
                                            float mean,
                                            float stddev) = 0;
};

class ThreadSafeRNG final : public ThreadSafeRNGBase {
public:
    using ThreadSafeRNGBase::ThreadSafeRNGBase;

    // Fills a range with random numbers using a provided distribution.
    template <class Distribution, typename T>
    void generate_range(Distribution& dist, std::span<T> range) {
        auto& engine = get_engine_for_current_thread();
        for (auto& val : range) {
            val = dist(engine);
        }
    }

    void generate_normal_dist_range(std::span<float> range,
                                    float mean,
                                    float stddev) override {
        auto& engine = get_engine_for_current_thread();
        boost::random::normal_distribution<float> dist(mean, stddev);
        for (auto& val : range) {
            val = dist(engine);
        }
    }
};

/**
 * @brief Thread-safe random number generator using a lookup table with linear
 * interpolation.
 *
 * This class provides a thread-safe implementation of a random number generator
 * that uses a lookup table with linear interpolation to generate normally
 * distributed random numbers. 3x faster than Boost's normal_distribution.
 *
 * The lookup table is initialized once and shared across all threads. It
 * precomputes a large Q-function table of quantiles, then for each uniform draw
 * do one array lookup + one linear interpolation (avoiding expensive erfc_inv
 * calls).
 *
 * The interpolation error is ≲1e-5; visually indistinguishable for Monte-Carlo
 * noise. The lookup table is initialized with 2^16 + 1 entries for good
 * resolution.
 */
class ThreadSafeLUTRNG final : public ThreadSafeRNGBase {
private:
    static std::vector<float> m_lut;
    static std::once_flag m_lut_init_flag;
    static constexpr float kInvUint32Max =
        1.0F / static_cast<float>(std::numeric_limits<uint32_t>::max());

    // This function is called only once to populate the lookup table.
    static void initialize_lut() {
        constexpr SizeType kTableSize = 65537;
        m_lut.resize(kTableSize);
        // Use `double` for the distribution object.
        // This provides the necessary precision to calculate quantiles for
        // probabilities very close to 0.0 or 1.0 without overflowing.
        boost::math::normal_distribution<double> dist;
        for (SizeType i = 0; i < kTableSize; ++i) {
            double probability = static_cast<double>(i) / (kTableSize - 1);
            probability        = std::clamp(probability, 1e-9, 1.0 - 1e-9);
            m_lut[i] =
                static_cast<float>(boost::math::quantile(dist, probability));
        }
    }

public:
    explicit ThreadSafeLUTRNG(unsigned int base_seed = std::random_device{}())
        : ThreadSafeRNGBase(base_seed) {
        std::call_once(m_lut_init_flag, initialize_lut);
    }

    void generate_normal_dist_range(std::span<float> range,
                                    float mean,
                                    float stddev) override {
        generate_normal_dist_range_xsimd(range.data(), range.size(), mean,
                                         stddev);
    }

    void generate_normal_dist_range_xsimd(float* __restrict__ range,
                                          SizeType range_size,
                                          float mean,
                                          float stddev) {
        auto& engine                  = get_engine_for_current_thread();
        using BatchFloat              = xsimd::batch<float>;
        using BatchInt                = xsimd::batch<int32_t>;
        constexpr SizeType kBatchSize = BatchFloat::size;
        const auto max_idx            = static_cast<float>(m_lut.size() - 2);

        std::array<uint32_t, kBatchSize> bits{};
        std::array<float, kBatchSize> bits_float{};
        std::array<float, kBatchSize> idxf_buf{};
        std::array<int32_t, kBatchSize> idxs_buf{};
        const SizeType main_loop = range_size - (range_size % kBatchSize);
        for (SizeType i = 0; i < main_loop; i += kBatchSize) {
            // Generate random bits and convert to uniform floats
            for (SizeType j = 0; j < kBatchSize; ++j) {
                bits[j] = engine();
            }
            // Load bits as floats directly and scale to [0, max_idx]
            for (SizeType j = 0; j < kBatchSize; ++j) {
                bits_float[j] = static_cast<float>(bits[j]);
            }
            BatchFloat u = xsimd::load_unaligned(bits_float.data()) *
                           kInvUint32Max * max_idx;
            // Split into index and fractional parts
            BatchFloat idxf = xsimd::floor(u);
            BatchFloat frac = u - idxf;
            // Convert float indices to integer batch
            idxf.store_unaligned(idxf_buf.data());
            for (SizeType j = 0; j < kBatchSize; ++j) {
                idxs_buf[j] = static_cast<int32_t>(idxf_buf[j]);
            }
            BatchInt idxs = xsimd::load_unaligned(idxs_buf.data());

            // Gather y1 and y2 from lookup table
            const auto dummy_batch = BatchFloat(0.0F);
            BatchFloat y1 = xsimd::kernel::gather(dummy_batch, m_lut.data(),
                                                  idxs, xsimd::default_arch{});
            BatchFloat y2 = xsimd::kernel::gather(dummy_batch, m_lut.data() + 1,
                                                  idxs, xsimd::default_arch{});

            // Interpolate, scale, and shift
            BatchFloat samples = y1 + (y2 - y1) * frac;
            samples = samples * BatchFloat(stddev) + BatchFloat(mean);
            // Store result
            samples.store_unaligned(range + i);
        }
        // Scalar tail
        for (SizeType i = main_loop; i < range_size; ++i) {
            const uint32_t bits = engine();
            const auto u   = static_cast<float>(bits) * kInvUint32Max * max_idx;
            const auto idx = static_cast<SizeType>(std::floor(u));
            const auto frac = u - static_cast<float>(idx);
            const auto y1   = m_lut[idx];
            const auto y2   = m_lut[idx + 1];
            range[i]        = (y1 + (y2 - y1) * frac) * stddev + mean;
        }
    }
};

// Define static members for ThreadSafeLUTRNG
inline std::vector<float> ThreadSafeLUTRNG::m_lut;
inline std::once_flag ThreadSafeLUTRNG::m_lut_init_flag;

} // namespace loki::math
