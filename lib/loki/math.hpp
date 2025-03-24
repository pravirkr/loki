#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <xtensor/xtensor.hpp>

#include <loki/loki_types.hpp>

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

} // namespace loki::math
