#pragma once

/**
 * @file ffa_portable_fft.cuh
 * @brief Block-level power-of-two complex FFT in registers and shared memory
 * (forward, natural order), the on-chip transform of the fused Fourier
 * scorers. Plain CUDA (no library), so it runs on every architecture loki
 * supports (sm_50+). See docs/fft.md. CUDA only.
 *
 * Layout: transform threadIdx.y of the block; thread threadIdx.x holds
 * elements threadIdx.x + i * stride (i < E) on input and on output. The
 * algorithm is a Stockham FFT whose passes are radix-E DFTs in registers
 * (the last pass radix M / E^p if E does not divide evenly), one
 * shared-memory exchange between passes; the first pass reads the input
 * layout directly. Twiddles are FP32 sincospif of the exact fraction, raised
 * to the needed powers by recurrence.
 */

#include <cstdint>

#include <cuda_runtime.h>

namespace loki::core::detail {

// exp(-2 pi i k / 16) for k < 8 (forward twiddles of the register DFTs).
__device__ __forceinline__ constexpr float2 pfft_w16(uint32_t k) {
    switch (k) {
    case 0:
        return {1.0F, 0.0F};
    case 1:
        return {0.92387953251128674F, -0.38268343236508978F};
    case 2:
        return {0.70710678118654752F, -0.70710678118654752F};
    case 3:
        return {0.38268343236508978F, -0.92387953251128674F};
    case 4:
        return {0.0F, -1.0F};
    case 5:
        return {-0.38268343236508978F, -0.92387953251128674F};
    case 6:
        return {-0.70710678118654752F, -0.70710678118654752F};
    default:
        return {-0.92387953251128674F, -0.38268343236508978F};
    }
}

__device__ __forceinline__ float2 pfft_mul(float2 a, float2 b) {
    return {fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x)};
}

constexpr uint32_t pfft_bitrev(uint32_t i, uint32_t n) {
    uint32_t r = 0;
    for (uint32_t b = 1; b < n; b <<= 1U) {
        r = (r << 1U) | ((i & b) != 0 ? 1U : 0U);
    }
    return r;
}

// Forward DFT of v[0, R) in registers (radix-2 decimation in time).
template <uint32_t R> __device__ __forceinline__ void pfft_dft(float2* v) {
    static_assert(R >= 1 && R <= 16 && (R & (R - 1)) == 0, "radix");
#pragma unroll
    for (uint32_t i = 0; i < R; ++i) {
        const uint32_t j = pfft_bitrev(i, R);
        if (i < j) {
            const float2 t = v[i];
            v[i]           = v[j];
            v[j]           = t;
        }
    }
#pragma unroll
    for (uint32_t len = 2; len <= R; len <<= 1U) {
#pragma unroll
        for (uint32_t i = 0; i < R; i += len) {
#pragma unroll
            for (uint32_t k = 0; k < len / 2; ++k) {
                const float2 a = v[i + k];
                float2 b       = v[i + k + (len / 2)];
                if (k != 0) {
                    b = pfft_mul(b, pfft_w16(k * (16 / len)));
                }
                v[i + k]             = {a.x + b.x, a.y + b.y};
                v[i + k + (len / 2)] = {a.x - b.x, a.y - b.y};
            }
        }
    }
}

template <uint32_t M, uint32_t E, uint32_t F> struct PortableFFT {
    static_assert((M & (M - 1)) == 0 && (E & (E - 1)) == 0 && E <= M,
                  "power-of-two sizes");
    static_assert(E >= 2 && E <= 16, "2 to 16 elements per thread");
    static constexpr bool kPortable                 = true;
    using value_type                                = float2;
    static constexpr uint32_t stride                = M / E;
    static constexpr uint32_t elements_per_thread   = E;
    static constexpr uint32_t storage_size          = E;
    static constexpr uint32_t ffts_per_block        = F;
    static constexpr uint32_t max_threads_per_block = stride * F;
    static constexpr dim3 block_dim{stride, F, 1};
    static constexpr uint32_t shared_memory_size =
        F * M * static_cast<uint32_t>(sizeof(float2));
    // A transform of at most 32 threads lies within one warp (stride
    // divides 32), and nothing else touches its buffer: warp barriers do.
    static constexpr bool kWarpLocal = stride <= 32;

    /// Barrier over the threads of one transform (every thread of the
    /// block calls it).
    __device__ __forceinline__ static void sync() {
        if constexpr (kWarpLocal) {
            __syncwarp();
        } else {
            __syncthreads();
        }
    }

    // One Stockham pass of radix R over sub-transforms of length kNs: E / R
    // butterflies per thread, read from and written back to buf.
    template <uint32_t kNs, uint32_t R>
    __device__ __forceinline__ static void pass(float2* v, float2* buf) {
        constexpr uint32_t kB = E / R;
        const uint32_t t      = threadIdx.x;
#pragma unroll
        for (uint32_t b = 0; b < kB; ++b) {
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                v[(b * R) + r] = buf[t + (b * stride) + (r * (M / R))];
            }
        }
        sync();
#pragma unroll
        for (uint32_t b = 0; b < kB; ++b) {
            const uint32_t j = t + (b * stride);
            const uint32_t k = j % kNs;
            float2* u        = v + (b * R);
            float2 w1;
            sincospif(-2.0F * static_cast<float>(k) /
                          static_cast<float>(kNs * R),
                      &w1.y, &w1.x);
            float2 w = w1;
#pragma unroll
            for (uint32_t r = 1; r < R; ++r) {
                u[r] = pfft_mul(u[r], w);
                w    = pfft_mul(w, w1);
            }
            pfft_dft<R>(u);
            const uint32_t base = ((j / kNs) * kNs * R) + k;
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                buf[base + (r * kNs)] = u[r];
            }
        }
        sync();
    }

    template <uint32_t kNs>
    __device__ __forceinline__ static void passes(float2* v, float2* buf) {
        if constexpr (kNs < M) {
            constexpr uint32_t R = (M / kNs) >= E ? E : (M / kNs);
            pass<kNs, R>(v, buf);
            passes<kNs * R>(v, buf);
        }
    }

    __device__ __forceinline__ void execute(float2* v, float2* smem) const {
        float2* buf      = smem + (threadIdx.y * M);
        const uint32_t t = threadIdx.x;
        // The previous transform may still be reading buf.
        sync();
        // Pass 0 (radix E, no twiddles): the input layout is its read
        // pattern, so it runs in registers.
        pfft_dft<E>(v);
#pragma unroll
        for (uint32_t r = 0; r < E; ++r) {
            buf[(t * E) + r] = v[r];
        }
        sync();
        passes<E>(v, buf);
#pragma unroll
        for (uint32_t i = 0; i < E; ++i) {
            v[i] = buf[t + (i * stride)];
        }
    }
};

} // namespace loki::core::detail
