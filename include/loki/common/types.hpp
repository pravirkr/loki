#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

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

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
// Helper macro for stringification
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define STRINGIFY_IMPL(x) #x

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

// Keyed on the compiler, not on the backend: the header is identical in every
// build, and nvcc-compiled translation units get host/device qualifiers.
#if defined(__CUDACC__)
#define LOKI_HD __host__ __device__
#define LOKI_D __device__
#define LOKI_H __host__
#else
#define LOKI_HD
#define LOKI_D
#define LOKI_H
#endif

inline constexpr std::array<std::string_view, 5> kParamNames = {
    "crackle", "snap", "jerk", "accel", "freq",
};

/// Location estimate used when normalising a loaded timeseries.
/// kNone leaves the location at 0.
enum class LocMethod : std::uint8_t { kMean, kMedian, kNone };

/// Scale estimate used when normalising a loaded timeseries.
/// kNone leaves the scale at 1.
enum class ScaleMethod : std::uint8_t { kStd, kIqr, kMad, kDoubleMad, kNone };

/// Gaussian consistency constants: IQR of N(0,1) is Phi^-1(0.75) -
/// Phi^-1(0.25); kMadScale is 1 / Phi^-1(0.75).
inline constexpr double kIqrScale = 1.3489795003921634;
inline constexpr double kMadScale = 1.482602218505602;

} // namespace loki
