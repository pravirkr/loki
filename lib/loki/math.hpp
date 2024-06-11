#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

template <typename T, std::size_t N> struct FactorialTable {
    std::array<T, N> values;

    constexpr FactorialTable() : values() {
        values[0] = 1;
        for (std::size_t i = 1; i < N; ++i) {
            values[i] = values[i - 1] * static_cast<T>(i);
        }
    }
    constexpr T operator[](std::size_t n) const { return values[n]; }
};

template <typename T, std::size_t N> constexpr auto make_factorial_table() {
    static_assert(N <= std::numeric_limits<T>::max() / (N - 1),
                  "Factorial table size is too large.");
    return FactorialTable<T, N>();
}

constexpr std::size_t kMaxFactorialTable = 20;
constexpr auto kFactorialTable =
    make_factorial_table<std::size_t, kMaxFactorialTable + 1>();

template <typename T> constexpr T factorial_compute(const T n) {
    if constexpr (std::is_integral_v<T>) {
        const auto n_cal = static_cast<std::size_t>(n);
        if (n_cal <= kMaxFactorialTable) {
            return kFactorialTable[n_cal];
        }
        return n_cal * factorial_compute(n_cal - 1);
    } else {
        return std::tgamma(n + 1);
    }
}

/**
 * @brief Compute the factorial of a number at compile time.
 *
 * @tparam T The type of the number.
 * @param n A real-valued number.
 * @return constexpr T The factorial of the number \f$ n! \f$.
 * When \c n is a real number, <tt>factorial(n) = tgamma(n + 1)</tt>.
 * When \c n is an integer, lookup table is used up to 20 and then computed
 * recursively.
 */
template <typename T> constexpr T factorial(const T n) {
    if (n < 0) {
        throw std::invalid_argument(
            "Factorial is not defined for negative numbers.");
    }
    return factorial_compute(n);
}