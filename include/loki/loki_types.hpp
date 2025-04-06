#pragma once

#include <array>
#include <complex>
#include <concepts>
#include <cstddef>

#include <xtensor/xtensor.hpp>

using SizeType        = std::size_t;
using FloatType       = double;
using IndexType       = std::ptrdiff_t;
using ComplexType     = std::complex<float>;
using ParamLimitType  = std::array<FloatType, 2>;
using SuggestionTypeF = xt::xtensor<float, 3>;
using SuggestionTypeD = xt::xtensor<double, 3>;

namespace loki::backend {

/**
 * @brief Tag struct representing the CPU backend.
 */
struct CPU {};

/**
 * @brief Tag struct representing the CUDA backend.
 */
struct CUDA {};

/**
 * @brief Concept to constrain template parameters to valid execution backends.
 */
template <typename T>
concept ExecutionBackend = std::same_as<T, CPU> || std::same_as<T, CUDA>;

} // namespace loki::backend