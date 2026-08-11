#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/complex>
#include <cuda/std/span>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki {

using SizeType    = std::size_t;
using IndexType   = std::ptrdiff_t;
using ComplexType = std::complex<float>;

struct ParamLimit {
    double min;
    double max;
};

/// Holds the minimum and maximum of a set of float scores.
struct MinMaxFloat {
    float min;
    float max;
};

template <typename T>
concept SupportedFoldType =
    std::is_same_v<T, float> || std::is_same_v<T, ComplexType>;

template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

#ifdef LOKI_ENABLE_CUDA
using ComplexTypeCUDA = cuda::std::complex<float>;

template <typename T>
concept SupportedFoldTypeCUDA =
    std::is_same_v<T, float> || std::is_same_v<T, ComplexTypeCUDA>;

template <SupportedFoldTypeCUDA T> struct FoldTypeTraits;
template <> struct FoldTypeTraits<float> {
    using HostType   = float;
    using DeviceType = float;
};

template <> struct FoldTypeTraits<ComplexTypeCUDA> {
    using HostType   = ComplexType;
    using DeviceType = ComplexTypeCUDA;
};

template <SupportedFoldTypeCUDA T>
using HostFoldType = typename FoldTypeTraits<T>::HostType;

template <SupportedFoldTypeCUDA T>
using DeviceFoldType = typename FoldTypeTraits<T>::DeviceType;

#endif // LOKI_ENABLE_CUDA

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
// Helper macro for stringification
#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x) #x

inline constexpr SizeType kUnrollFactor = 8;

#if defined(__clang__)
#define UNROLL_N(N) _Pragma(STRINGIFY(clang loop unroll_count(N)))
#define UNROLL_VECTORIZE_N(N)                                                  \
    _Pragma(STRINGIFY(clang loop unroll_count(N) vectorize(enable)))
#elif defined(__GNUC__)
#define UNROLL_N(N) _Pragma(STRINGIFY(GCC unroll N))
#define UNROLL_VECTORIZE_N(N)                                                  \
    _Pragma(STRINGIFY(GCC unroll N)) _Pragma("GCC ivdep")
#else
#define UNROLL_N(N)
#define UNROLL_VECTORIZE_N(N)
#endif

#define UNROLL_VECTORIZE UNROLL_VECTORIZE_N(kUnrollFactor)
// NOLINTEND(cppcoreguidelines-macro-usage)
// UNROLL_VECTORIZE_N is not supported for gcc < 14.0

#if defined(LOKI_ENABLE_CUDA) && defined(__CUDACC__)
#define LOKI_HD __host__ __device__
#define LOKI_D __device__
#define LOKI_H __host__
#else
#define LOKI_HD
#define LOKI_D
#define LOKI_H
#endif

inline constexpr std::array<std::string, 5> kParamNames = {
    "crackle", "snap", "jerk", "accel", "freq"};

} // namespace loki
