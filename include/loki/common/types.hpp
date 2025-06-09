#pragma once

#include <array>
#include <complex>
#include <concepts>
#include <cstddef>

#include <xtensor/containers/xtensor.hpp>

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/complex>
#include <cuda/std/span>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki {

using SizeType        = std::size_t;
using IndexType       = std::ptrdiff_t;
using ComplexType     = std::complex<float>;
using ParamLimitType  = std::array<double, 2>;
using SuggestionTypeF = xt::xtensor<float, 3>;
using SuggestionTypeD = xt::xtensor<double, 3>;

#ifdef LOKI_ENABLE_CUDA
using ComplexTypeCUDA = cuda::std::complex<float>;
#endif // LOKI_ENABLE_CUDA

inline constexpr size_t kUnrollFactor = 4;

#if defined(__clang__)
#define UNROLL_VECTORIZE                                                       \
    _Pragma("clang loop unroll_count(kUnrollFactor) vectorize(enable)")
#elif defined(__GNUC__)
#define UNROLL_VECTORIZE                                                       \
    _Pragma("GCC unroll kUnrollFactor") _Pragma("GCC ivdep")
#else
#define UNROLL_VECTORIZE
#endif

} // namespace loki

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