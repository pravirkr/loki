#pragma once

/**
 * @file types_cuda.cuh
 * @brief Device-side fold types and host/device type traits (CUDA only).
 */

#include <type_traits>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

namespace loki {

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

/// Device fold type for a public (host) fold type: float -> float,
/// ComplexType -> ComplexTypeCUDA. Same memory layout, so a DeviceSpan of
/// the host type can be reinterpreted as the device type.
template <SupportedFoldType T>
using CudaFoldType =
    std::conditional_t<std::is_same_v<T, float>, float, ComplexTypeCUDA>;

} // namespace loki
