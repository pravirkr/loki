#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <xsimd/xsimd.hpp>

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/complex>
#include <cuda/std/span>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki {

using SizeType       = std::size_t;
using IndexType      = std::ptrdiff_t;
using ComplexType    = std::complex<float>;
using ParamLimitType = std::array<double, 2>;

using AlignedFloatVec = std::vector<float, xsimd::aligned_allocator<float>>;

template <typename T>
concept SupportedFoldType =
    std::is_same_v<T, float> || std::is_same_v<T, ComplexType>;

template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template <SupportedFoldType FoldType> constexpr FoldType default_fold_value() {
    if constexpr (std::is_same_v<FoldType, std::complex<float>>) {
        return FoldType{0.0F, 0.0F};
    } else {
        return FoldType{};
    }
}

#ifdef LOKI_ENABLE_CUDA
using ComplexTypeCUDA = cuda::std::complex<float>;
#endif // LOKI_ENABLE_CUDA

inline constexpr size_t kUnrollFactor = 8;

#if defined(__clang__)
#define UNROLL_VECTORIZE                                                       \
    _Pragma("clang loop unroll_count(kUnrollFactor) vectorize(enable)")
#elif defined(__GNUC__)
#define UNROLL_VECTORIZE                                                       \
    _Pragma("GCC unroll kUnrollFactor") _Pragma("GCC ivdep")
#else
#define UNROLL_VECTORIZE
#endif

inline constexpr std::array<std::string, 5> kParamNames = {
    "crackle", "snap", "jerk", "accel", "freq"};

} // namespace loki
