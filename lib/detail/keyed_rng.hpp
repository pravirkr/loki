#pragma once

/**
 * @file keyed_rng.hpp
 * @brief Keyed PCG32 streams and the inverse-CDF normal draw, for host and
 * device code. Internal.
 *
 * Plain integer and float arithmetic with LOKI_HD qualifiers, so an
 * nvcc-compiled kernel draws exactly the numbers the host draws for the
 * same key (the spectral zap's noise fill runs on either backend). The
 * float steps of the normal draw use explicit IEEE intrinsics on the
 * device, because the CUDA Release flags include -use_fast_math.
 */

#include <cmath>
#include <cstdint>
#include <limits>

#include "loki/common/types.hpp"

namespace loki::math {

// A minimal PCG32 random number generator.
class PCG32 {
public:
    using ResultType = uint32_t;
    using StateType  = uint64_t;

    static constexpr StateType kDefaultState  = 0x853c49e6748fea9bULL;
    static constexpr StateType kDefaultStream = 0xda3e39cb94b95bdbULL;
    static constexpr StateType kMult          = 0x5851f42d4c957f2dULL;

    LOKI_HD PCG32() { seed_seq(kDefaultState, kDefaultStream); }
    LOKI_HD explicit PCG32(StateType seed, StateType stream = kDefaultStream) {
        seed_seq(seed, stream);
    }

    LOKI_HD void seed_seq(StateType seed, StateType stream) {
        m_state = 0U;
        m_inc   = (stream << 1U) | 1U;
        operator()();
        m_state += seed;
        operator()();
    }

    // Generates the next random number
    LOKI_HD ResultType operator()() {
        const StateType oldstate = m_state;
        m_state                  = (oldstate * kMult) + m_inc;
        const auto xorshifted =
            static_cast<ResultType>(((oldstate >> 18U) ^ oldstate) >> 27U);
        const auto rot = static_cast<ResultType>(oldstate >> 59U);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1U) & 31U));
    }

    LOKI_HD static constexpr ResultType min() {
        return std::numeric_limits<ResultType>::min();
    }
    LOKI_HD static constexpr ResultType max() {
        return std::numeric_limits<ResultType>::max();
    }

private:
    StateType m_state{};
    StateType m_inc{};
};

/// SplitMix64 finaliser: a strong 64-bit mix for deriving seeds.
[[nodiscard]] LOKI_HD constexpr uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

/**
 * @brief Independent PCG32 stream for one keyed work item.
 *
 * Hashes the seed and the key fields (a purpose tag plus up to four item
 * coordinates) into both the PCG state and the stream selector, so distinct
 * keys give statistically independent streams, and the same key always
 * gives the same stream. Use it instead of per-thread generators: the draws
 * then depend only on (seed, key), never on threads or scheduling.
 * `branch` must be < 2^24; the other fields may use all 32 bits.
 */
[[nodiscard]] LOKI_HD inline PCG32 make_keyed_pcg32(uint64_t seed,
                                                    uint32_t purpose,
                                                    uint32_t stage,
                                                    uint32_t parent,
                                                    uint32_t branch,
                                                    uint32_t trial) {
    uint64_t mixed = splitmix64(seed ^ (static_cast<uint64_t>(purpose) << 48U));
    mixed          = splitmix64(mixed ^ (static_cast<uint64_t>(stage) << 32U));
    mixed          = splitmix64(mixed ^ parent);
    mixed =
        splitmix64(mixed ^ ((static_cast<uint64_t>(branch) << 40U) ^ trial));
    return PCG32(mixed, splitmix64(mixed ^ 0xda3e39cb94b95bdbULL));
}

/// Quantiles in NormalSampler's lookup table: 2^17 + 1 of N(0, 1).
inline constexpr SizeType kNormalLutSize = (SizeType{1} << 17U) + 1;

/**
 * @brief One N(mean, stddev^2) draw from the uniform word @p r and the
 * inverse-CDF table @p lut (kNormalLutSize quantiles, linear
 * interpolation). NormalSampler::generate on the host; the same operations
 * on the device.
 */
[[nodiscard]] LOKI_HD inline float normal_from_lut(uint32_t r,
                                                   const float* lut,
                                                   float mean,
                                                   float stddev) noexcept {
    constexpr float kInvU32 =
        1.0F / static_cast<float>(std::numeric_limits<uint32_t>::max());
    constexpr auto kMaxIdx = static_cast<float>(kNormalLutSize - 2);
#if defined(__CUDA_ARCH__)
    const float u_scaled =
        __fmul_rn(__fmul_rn(__uint2float_rn(r), kInvU32), kMaxIdx);
    const auto idx   = static_cast<SizeType>(u_scaled);
    const float frac = __fsub_rn(u_scaled, static_cast<float>(idx));
    const float z =
        __fmaf_rn(frac, __fsub_rn(lut[idx + 1], lut[idx]), lut[idx]);
    return __fmaf_rn(z, stddev, mean);
#else
    // Uniform in [0, max_idx], split into index and fraction
    const float u_scaled = static_cast<float>(r) * kInvU32 * kMaxIdx;
    const auto idx       = static_cast<SizeType>(u_scaled);
    const float frac     = u_scaled - static_cast<float>(idx);
    const float z        = std::fma(frac, lut[idx + 1] - lut[idx], lut[idx]);
    return std::fma(z, stddev, mean);
#endif
}

} // namespace loki::math
