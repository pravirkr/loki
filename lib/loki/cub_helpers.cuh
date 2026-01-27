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
