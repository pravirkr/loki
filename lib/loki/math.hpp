#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

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

template <std::floating_point T> struct StatLookupTables {
    static constexpr T kMaxMinusLogSf        = 400.0;
    static constexpr T kMinusLogSfRes        = 0.1;
    static constexpr T kChiSqMax             = 300.0;
    static constexpr T kChiSqRes             = 0.5;
    static constexpr std::size_t kChiSqMaxDf = 64;

    static constexpr size_t kNormTableSize =
        static_cast<size_t>(kMaxMinusLogSf / kMinusLogSfRes) + 1;
    static constexpr size_t kChiSqTableSize =
        static_cast<size_t>(kChiSqMax / kChiSqRes) + 1;

    static constexpr auto gen_norm_isf_table() {
        boost::math::normal_distribution<T> norm_dist;
        std::array<T, kNormTableSize> table{};
        for (size_t i = 0; i < kNormTableSize; ++i) {
            table[i] = boost::math::quantile(boost::math::complement(
                norm_dist, std::exp(-static_cast<T>(i) * kMinusLogSfRes)));
        }
        return table;
    }
    static constexpr auto gen_chi_sq_minus_logsf_table() {
        std::array<std::array<T, kChiSqTableSize>, kChiSqMaxDf + 1> table{};
        for (size_t df = 1; df <= kChiSqMaxDf; ++df) {
            boost::math::chi_squared_distribution<T> chi_sq_dist(
                static_cast<T>(df));
            for (size_t i = 0; i < kChiSqTableSize; ++i) {
                table[df][i] =
                    -std::log(boost::math::cdf(boost::math::complement(
                        chi_sq_dist, static_cast<T>(i) * kChiSqRes)));
            }
        }
        return table;
    }
    // Global constexpr tables
    static constexpr auto kNormIsfTable = gen_norm_isf_table();
    static constexpr auto kChiSqMinusLogsfTable =
        gen_chi_sq_minus_logsf_table();

    // Normal inverse survival function
    static constexpr T norm_isf(T minus_logsf) {
        const auto pos      = minus_logsf / kMinusLogSfRes;
        const auto pos_int  = static_cast<size_t>(pos);
        const auto frac_pos = pos - static_cast<T>(pos_int);
        if (minus_logsf < kMaxMinusLogSf) {
            return std::lerp(kNormIsfTable[pos_int], kNormIsfTable[pos_int + 1],
                             frac_pos);
        }
        return kNormIsfTable.back() * std::sqrt(minus_logsf / kMaxMinusLogSf);
    }

    // Chi-squared minus log survival function
    static constexpr T chi_sq_minus_logsf(T chi_sq_score, size_t df) {
        if (df == 0 || df > kChiSqMaxDf) {
            throw std::out_of_range("Degrees of freedom out of valid range");
        }
        const auto tab_pos     = chi_sq_score / kChiSqRes;
        const auto tab_pos_int = static_cast<size_t>(tab_pos);
        const auto frac_pos    = tab_pos - static_cast<T>(tab_pos_int);
        if (chi_sq_score < kChiSqMax) {
            return std::lerp(kChiSqMinusLogsfTable[df][tab_pos_int],
                             kChiSqMinusLogsfTable[df][tab_pos_int + 1],
                             frac_pos);
        }
        return kChiSqMinusLogsfTable[df].back() * chi_sq_score / kChiSqMax;
    }
};

using StatTables = StatLookupTables<float>;
