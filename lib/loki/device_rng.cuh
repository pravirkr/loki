#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && !defined(LOKI_FORCE_CURAND_RNG)

#ifndef CURANDDX_SM
#define CURANDDX_SM 700
#endif

#include <curanddx.hpp>

namespace loki::math {

/**
 * @brief cuRANDDx generator configuration for high-performance per-thread RNG.
 *
 * Used on sm_70+ device code paths. Requires NVIDIA MathDX / cuRANDDx headers
 * at compile time for those architecture slices.
 *
 * @tparam Rounds Number of Philox rounds
 * @tparam SM Target SM architecture for cuRANDDx templates
 */
template <uint32_t Rounds = 10, uint32_t SM = CURANDDX_SM>
struct DeviceRNGConfig {
    using Generator = decltype(curanddx::Generator<curanddx::philox4_32>() +
                               curanddx::PhiloxRounds<Rounds>() +
                               curanddx::SM<SM>() + curanddx::Thread());

    using NormalFloat  = curanddx::normal<float, curanddx::box_muller>;
    using UniformFloat = curanddx::uniform<float>;
};

} // namespace loki::math

#else

#include <curand.h>
#include <curand_kernel.h>

namespace loki::math {
    namespace detail {

/**
 * @brief Wrapper for stock cuRAND per-thread Philox4_32_10 state (sm_50+).
 */
struct CurandPhiloxGenerator {
    curandStatePhilox4_32_10_t state;

    __device__ CurandPhiloxGenerator(uint64_t seed,
                                     uint64_t subsequence,
                                     uint64_t offset) {
        curand_init(seed, subsequence, offset, &state);
    }
};

/**
 * @brief Stock cuRAND-backed normal distribution generator yielding float4.
 */
struct CurandNormalFloat {
    float mean;
    float stddev;

    __device__ CurandNormalFloat(float m, float s) : mean(m), stddev(s) {}

    __device__ float4 generate4(CurandPhiloxGenerator& gen) const {
        const float4 r = curand_normal4(&gen.state);
        return make_float4(fmaf(r.x, stddev, mean), fmaf(r.y, stddev, mean),
                           fmaf(r.z, stddev, mean), fmaf(r.w, stddev, mean));
    }
};

/**
 * @brief Stock cuRAND-backed uniform distribution generator yielding float4.
 */
struct CurandUniformFloat {
    float lo;
    float hi;

    __device__ CurandUniformFloat(float l, float h) : lo(l), hi(h) {}

    __device__ float4 generate4(CurandPhiloxGenerator& gen) const {
        const float4 r   = curand_uniform4(&gen.state);
        const float span = hi - lo;
        return make_float4(fmaf(r.x, span, lo), fmaf(r.y, span, lo),
                           fmaf(r.z, span, lo), fmaf(r.w, span, lo));
    }
};

} // namespace detail

/**
 * @brief Stock cuRAND Philox4_32_10 per-thread RNG configuration.
 *
 * Used on pre-Volta device code paths (sm_50–sm_69), on the nvcc host
 * compilation pass, or when LOKI_FORCE_CURAND_RNG is defined. Template
 * parameters Rounds and SM are accepted for API compatibility but ignored
 * (Philox-10 is fixed by cuRAND).
 *
 * @tparam Rounds Number of Philox rounds (ignored)
 * @tparam SM Target SM architecture (ignored)
 */
template <uint32_t Rounds = 10, uint32_t SM = 700>
struct DeviceRNGConfig {
    using Generator    = detail::CurandPhiloxGenerator;
    using NormalFloat  = detail::CurandNormalFloat;
    using UniformFloat = detail::CurandUniformFloat;
};

} // namespace loki::math

#endif

namespace loki::math {
/// Default per-thread RNG: Philox 7 rounds on sm_70+ (cuRANDDx), Philox-10 otherwise.
using DefaultDeviceRNG = DeviceRNGConfig<7>;

} // namespace loki::math
