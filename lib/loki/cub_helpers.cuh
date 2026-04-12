#pragma once

#include <cub/cub.cuh>

// Check if we are on a modern CUB version (CCCL 3.0+)
#if CUB_VERSION >= 300000
#include <cuda/functional>
#include <cuda/std/numbers>
template <typename T> using CubMaxOp    = ::cuda::maximum<T>;
template <typename T> using CubMinOp    = ::cuda::minimum<T>;
template <typename T> using ThrustMaxOp = ::cuda::maximum<T>;
template <typename T> using ThrustMinOp = ::cuda::minimum<T>;
inline constexpr double kPI             = cuda::std::numbers::pi_v<double>;
#else
// Fall back to CUB operators for older CCCL
#include <thrust/functional.h>
template <typename T> using CubMaxOp    = cub::Max;
template <typename T> using CubMinOp    = cub::Min;
template <typename T> using ThrustMaxOp = thrust::maximum<T>;
template <typename T> using ThrustMinOp = thrust::minimum<T>;
inline constexpr double kPI             = 3.14159265358979323846; // NOLINT
#endif // CUB_VERSION >= 300000

#include "loki/common/types.hpp"

namespace loki {
// ---------------------------------------------------------------------------
// Functor: uint8_t → uint32_t
//
// Used as a zero-copy cast inside DeviceReduce::Sum so that the mask array
// can remain uint8_t on device while the accumulator operates in uint32_t,
// preventing wrap-around at 255.
// ---------------------------------------------------------------------------
struct Uint8ToUint32 {
    __host__ __device__ uint32_t operator()(uint8_t x) const {
        return static_cast<uint32_t>(x);
    }
};

struct MinMaxReduce {
    __host__ __device__ MinMaxFloat operator()(const MinMaxFloat& a,
                                               const MinMaxFloat& b) const {
        return {fminf(a.min, b.min), fmaxf(a.max, b.max)};
    }
};

struct ScoreToMinMaxFloat {
    const float* __restrict__ scores;
    const uint8_t* __restrict__ mask;

    __host__ __device__ MinMaxFloat operator()(int i) const {
        float s = scores[i];
        return mask[i] ? MinMaxFloat{s, s}
                       : MinMaxFloat{std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::lowest()};
    }
};
} // namespace loki