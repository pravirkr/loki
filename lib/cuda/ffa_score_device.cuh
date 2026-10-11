#pragma once

/**
 * @file ffa_score_device.cuh
 * @brief Device helpers shared by the fused final-level scorers
 * (ffa_score_cuda.cu, ffa_score_fourier_cuda.cu). CUDA only.
 *
 * The scoring arithmetic reproduces detection::kernel_snr_boxcar_thread and
 * kernel_snr_boxcar_warp (score_cuda.cu) expression for expression.
 */

#include <cstdint>

#include <cuda/std/complex>
#include <cuda/std/limits>
#include <cuda_runtime.h>

#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/ffa_resolve_device.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core::detail {

constexpr uint32_t kWarpSize = 32;

// Same expression as detection::fold_norm_bin (score_cuda.cu).
__device__ __forceinline__ float fold_norm_bin(float e, float v) {
    return (v > 0.0F) ? (e * rsqrtf(v)) : 0.0F;
}

// Same as detection::warp_inclusive_scan (score_cuda.cu).
// lane_id is the linear lane (blocks may be 2D).
__device__ __forceinline__ float warp_inclusive_scan(float val, int lane_id) {
    constexpr unsigned int kFullMask = 0xFFFFFFFF;
#pragma unroll
    for (int offset = 1; offset < static_cast<int>(kWarpSize); offset <<= 1) {
        const float tmp =
            __shfl_up_sync(kFullMask, val, static_cast<unsigned int>(offset));
        if (lane_id >= offset) {
            val += tmp;
        }
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    constexpr unsigned int kFullMask = 0xFFFFFFFF;
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 16));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 8));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 4));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 2));
    val = fmaxf(val, __shfl_down_sync(kFullMask, val, 1));
    return val;
}

__device__ __forceinline__ size_t profile_offset(uint32_t segment,
                                                 uint32_t ncoords,
                                                 uint32_t icoord,
                                                 uint32_t stride) {
    return ((static_cast<size_t>(segment) * ncoords) + icoord) * stride;
}

__device__ __forceinline__ uint32_t round_shift(float shift, uint32_t nbins) {
    const uint32_t s = __float2uint_rz(shift + 0.5F);
    return (s == nbins) ? 0U : s;
}

__device__ __forceinline__ uint32_t rotate_idx(uint32_t ibin,
                                               uint32_t shift,
                                               uint32_t nbins) {
    return (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);
}

struct TailHead {
    uint32_t i_tail;
    uint32_t shift_tail;
    uint32_t i_head;
    uint32_t shift_head;
};

// kernel_ffa_iter: both halves shifted.
struct AccelParents {
    coord::FFACoordDPtrs c;
    __device__ __forceinline__ TailHead get(uint32_t icoord,
                                            uint32_t nbins) const {
        return {.i_tail     = c.i_tail[icoord],
                .shift_tail = round_shift(c.shift_tail[icoord], nbins),
                .i_head     = c.i_head[icoord],
                .shift_head = round_shift(c.shift_head[icoord], nbins)};
    }
};

// Accel final level resolved on the fly from the level tables (the same
// arithmetic as the resolve kernel, so the same coordinates).
struct AccelParentsComputed {
    FFAAccelResolveLevel lvl;
    __device__ __forceinline__ TailHead get(uint32_t icoord,
                                            uint32_t nbins) const {
        const AccelCoord c = resolve_accel_coord(lvl, icoord);
        return {.i_tail     = c.i_tail,
                .shift_tail = round_shift(c.shift_tail, nbins),
                .i_head     = c.i_head,
                .shift_head = round_shift(c.shift_head, nbins)};
    }
};

// kernel_ffa_freq_iter: one parent, tail unshifted.
struct FreqParents {
    coord::FFACoordFreqDPtrs c;
    __device__ __forceinline__ TailHead get(uint32_t icoord,
                                            uint32_t nbins) const {
        const uint32_t idx = c.idx[icoord];
        return {.i_tail     = idx,
                .shift_tail = 0U,
                .i_head     = idx,
                .shift_head = round_shift(c.shift[icoord], nbins)};
    }
};

// Final coordinate of traversal position s (see FFAFinalScoreArgs).
__device__ __forceinline__ uint32_t tile_order(uint32_t s,
                                               const FFAFinalScoreArgs& args) {
    if (args.grid_rows <= 1 || args.tile_rows <= 1) {
        return s;
    }
    const uint32_t per_group = args.tile_rows * args.grid_cols;
    const uint32_t g         = s / per_group;
    const uint32_t r         = s - (g * per_group);
    const uint32_t row0      = g * args.tile_rows;
    const uint32_t h         = min(args.tile_rows, args.grid_rows - row0);
    const uint32_t f         = r / h;
    const uint32_t a         = row0 + (r - (f * h));
    return (a * args.grid_cols) + f;
}

__device__ __forceinline__ void
emit(const FFAFinalScoreArgs& args, uint32_t index, float snr) {
    if (snr >= args.threshold) {
        const uint32_t slot = atomicAdd(args.counter, 1U);
        if (slot < args.capacity) {
            args.indices[slot] = index;
            args.scores[slot]  = snr;
        }
    }
}

// Widths per launch; a register maximum per width.
constexpr uint32_t kMaxWidths = 32;

struct WidthConsts {
    int w[kMaxWidths];
    float h[kMaxWidths];
    float b[kMaxWidths];
};

// Per-width boxcar constants, the same expressions as the scorers.
__device__ __forceinline__ void
load_width_consts(const FFAFinalScoreArgs& args,
                  int nb,
                  WidthConsts& s_wc,
                  uint32_t tid      = threadIdx.x,
                  uint32_t nthreads = blockDim.x) {
    for (uint32_t iw = tid; iw < args.nwidths; iw += nthreads) {
        const int w = static_cast<int>(args.widths[iw]);
        const float h =
            sqrtf(static_cast<float>(nb - w) / static_cast<float>(nb * w));
        const float b = static_cast<float>(w) * h / static_cast<float>(nb - w);
        s_wc.w[iw]    = w;
        s_wc.h[iw]    = h;
        s_wc.b[iw]    = b;
    }
}

// Boxcar S/N of one profile for every width, one warp per profile. row[i]
// is the inclusive prefix sum of the normalized bins. Each lane loads a
// window start once and updates the maximum of every width; the maximum is
// order-independent. kSerialFormula selects the wrapped-window expression of
// the thread kernel, (total - lo) + hi; the warp kernel uses
// (total + hi) - lo. Chunks of 32 window starts where no window wraps (warp
// uniform) skip the wrap logic.
template <bool kSerialFormula, uint32_t NW>
__device__ __forceinline__ void
score_profile_warp(const float* row,
                   float total_sum,
                   int nb,
                   const WidthConsts& s_wc,
                   uint32_t index_base,
                   const FFAFinalScoreArgs& args,
                   int lane_id) {
    const uint32_t nwidths = args.nwidths;
    float mx[NW];
    int wv[NW];
    int wmax = 1;
#pragma unroll
    for (uint32_t iw = 0; iw < NW; ++iw) {
        mx[iw] = cuda::std::numeric_limits<float>::lowest();
        wv[iw] = (iw < nwidths) ? s_wc.w[iw] : 1;
        wmax   = max(wmax, wv[iw]);
    }
    for (int base = 0; base < nb; base += static_cast<int>(kWarpSize)) {
        const int j = base + lane_id;
        if (base + static_cast<int>(kWarpSize) + wmax - 2 < nb) {
            // No window of this chunk wraps: hi - lo for every width.
            const float lo    = (j > 0) ? row[j - 1] : 0.0F;
            const float* hi_p = row + (j - 1);
#pragma unroll
            for (uint32_t iw = 0; iw < NW; ++iw) {
                if (iw < nwidths) {
                    mx[iw] = fmaxf(mx[iw], hi_p[wv[iw]] - lo);
                }
            }
            continue;
        }
        if (j >= nb) {
            continue;
        }
        const float lo = (j > 0) ? row[j - 1] : 0.0F;
#pragma unroll
        for (uint32_t iw = 0; iw < NW; ++iw) {
            if (iw < nwidths) {
                const int idx   = j + wv[iw] - 1;
                const bool wrap = idx >= nb;
                const float hi  = row[wrap ? idx - nb : idx];
                float window_sum;
                if constexpr (kSerialFormula) {
                    window_sum = wrap ? ((total_sum - lo) + hi) : (hi - lo);
                } else {
                    window_sum = wrap ? ((total_sum + hi) - lo) : (hi - lo);
                }
                mx[iw] = fmaxf(mx[iw], window_sum);
            }
        }
    }
#pragma unroll
    for (uint32_t iw = 0; iw < NW; ++iw) {
        if (iw < nwidths) {
            const float max_diff = warp_reduce_max(mx[iw]);
            if (lane_id == 0) {
                const float h   = s_wc.h[iw];
                const float b   = s_wc.b[iw];
                const float snr = ((h + b) * max_diff) - (b * total_sum);
                emit(args, index_base + iw, snr);
            }
        }
    }
}

// Thread-per-profile score (kernel_snr_boxcar_thread). @p row holds the
// normalized bins and is overwritten with their inclusive prefix sums,
// summed serially in the thread kernel's order (its psum[k] is row[k - 1],
// with psum[0] = 0). Window starts run outermost so each row[j - 1] is read
// once for every width; the window maximum is order-independent.
template <uint32_t NW>
__device__ __forceinline__ void
score_row_serial(float* row,
                 int nb,
                 const WidthConsts& s_wc,
                 uint32_t index_base,
                 const FFAFinalScoreArgs& args) {
    float running = 0.0F;
#pragma unroll 8
    for (int i = 0; i < nb; ++i) {
        running += row[i];
        row[i] = running;
    }
    const float total_sum  = running;
    const uint32_t nwidths = args.nwidths;
    float mx[NW];
    int wv[NW];
    int wmax = 1;
#pragma unroll
    for (uint32_t iw = 0; iw < NW; ++iw) {
        wv[iw] = (iw < nwidths) ? s_wc.w[iw] : 1;
        wmax   = max(wmax, wv[iw]);
        // j = 0: psum[w] - psum[0]
        mx[iw] = row[wv[iw] - 1];
    }
    // Starts with no wrapped window for any width.
    const int no_wrap_end = nb - wmax + 1;
    int j                 = 1;
    for (; j < no_wrap_end; ++j) {
        const float lo    = row[j - 1];
        const float* hi_p = row + (j - 1);
#pragma unroll
        for (uint32_t iw = 0; iw < NW; ++iw) {
            if (iw < nwidths) {
                mx[iw] = fmaxf(mx[iw], hi_p[wv[iw]] - lo);
            }
        }
    }
    for (; j < nb; ++j) {
        const float lo = row[j - 1];
#pragma unroll
        for (uint32_t iw = 0; iw < NW; ++iw) {
            if (iw < nwidths) {
                const int idx = j + wv[iw] - 1;
                const float window_sum =
                    (idx < nb) ? (row[idx] - lo)
                               : ((total_sum - lo) + row[idx - nb]);
                mx[iw] = fmaxf(mx[iw], window_sum);
            }
        }
    }
#pragma unroll
    for (uint32_t iw = 0; iw < NW; ++iw) {
        if (iw < nwidths) {
            const float h   = s_wc.h[iw];
            const float b   = s_wc.b[iw];
            const float snr = ((h + b) * mx[iw]) - (b * total_sum);
            emit(args, index_base + iw, snr);
        }
    }
}

// Warp-per-profile score (kernel_snr_boxcar_warp): a warp scan of the
// normalized bins in chunks of 32, written over the e row.
template <uint32_t NW>
__device__ __forceinline__ void score_ev_warp(float* e_ptr,
                                              const float* v_ptr,
                                              float fold_scale,
                                              int nb,
                                              const WidthConsts& s_wc,
                                              uint32_t index_base,
                                              const FFAFinalScoreArgs& args,
                                              int lane_id) {
    float* s_psum_warp = e_ptr;
    float running_sum  = 0.0F;
    const int num_chunks =
        (nb + static_cast<int>(kWarpSize) - 1) / static_cast<int>(kWarpSize);
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int idx = (chunk * static_cast<int>(kWarpSize)) + lane_id;
        float val     = (idx < nb) ? fold_norm_bin(e_ptr[idx] * fold_scale,
                                                   v_ptr[idx] * fold_scale)
                                   : 0.0F;
        val           = warp_inclusive_scan(val, lane_id);
        val += running_sum;
        if (idx < nb) {
            s_psum_warp[idx] = val;
        }
        running_sum =
            __shfl_sync(0xFFFFFFFF, val, static_cast<int>(kWarpSize) - 1);
    }
    __syncwarp();
    score_profile_warp</*kSerialFormula=*/false, NW>(
        s_psum_warp, running_sum, nb, s_wc, index_base, args, lane_id);
}

// --- Fourier-domain merges (frequency sweep path) ---------------------------

struct Phasor {
    float c;
    float s;
};

// exp(-2 pi i k shift / nbins) in FP32. k * shift is split exactly into
// hi + lo (fmaf), reduced to a fraction of a cycle, then evaluated with
// sincospif, which is accurate for any argument. For a power-of-two nbins
// the scaling is exact. This avoids the FP64 phase of fourier_phase
// (kernels_cuda.cu) and the large-argument error of __sincosf.
__device__ __forceinline__ Phasor fourier_phasor(uint32_t k,
                                                 float shift,
                                                 float inv_nbins) {
    const auto kf    = static_cast<float>(k);
    const float t_hi = kf * shift;
    const float t_lo = fmaf(kf, shift, -t_hi);
    const float c_hi = t_hi * inv_nbins;
    const float c_lo = t_lo * inv_nbins;
    const float frac = (c_hi - rintf(c_hi)) + c_lo;
    Phasor ph;
    sincospif(-2.0F * frac, &ph.s, &ph.c);
    return ph;
}

__device__ __forceinline__ Phasor phasor_mul(const Phasor& a, const Phasor& b) {
    return {fmaf(a.c, b.c, -a.s * b.s), fmaf(a.c, b.s, a.s * b.c)};
}

struct TailHeadF {
    uint32_t i_tail;
    float shift_tail;
    uint32_t i_head;
    float shift_head;
};

// kernel_ffa_complex_iter: both halves are phase shifted.
struct AccelParentsC {
    static constexpr bool kTailShift = true;
    coord::FFACoordDPtrs c;
    __device__ __forceinline__ TailHeadF get(uint32_t icoord) const {
        return {.i_tail     = c.i_tail[icoord],
                .shift_tail = c.shift_tail[icoord],
                .i_head     = c.i_head[icoord],
                .shift_head = c.shift_head[icoord]};
    }
};

// Fourier counterpart of AccelParentsComputed.
struct AccelParentsCComputed {
    static constexpr bool kTailShift = true;
    FFAAccelResolveLevel lvl;
    __device__ __forceinline__ TailHeadF get(uint32_t icoord) const {
        const AccelCoord c = resolve_accel_coord(lvl, icoord);
        return {.i_tail     = c.i_tail,
                .shift_tail = c.shift_tail,
                .i_head     = c.i_head,
                .shift_head = c.shift_head};
    }
};

// kernel_ffa_complex_freq_iter: one parent, the tail unshifted.
struct FreqParentsC {
    static constexpr bool kTailShift = false;
    coord::FFACoordFreqDPtrs c;
    __device__ __forceinline__ TailHeadF get(uint32_t icoord) const {
        const uint32_t idx = c.idx[icoord];
        return {.i_tail     = idx,
                .shift_tail = 0.0F,
                .i_head     = idx,
                .shift_head = c.shift[icoord]};
    }
};

// tail * pt + head * ph (tail unrotated when !kTailShift), with the
// complex products formed as in kernel_ffa_complex_iter.
template <bool kTailShift>
__device__ __forceinline__ ComplexTypeCUDA merge_c(const ComplexTypeCUDA& tail,
                                                   const ComplexTypeCUDA& head,
                                                   const Phasor& pt,
                                                   const Phasor& ph) {
    const float real_head = fmaf(head.real(), ph.c, -head.imag() * ph.s);
    const float imag_head = fmaf(head.real(), ph.s, head.imag() * ph.c);
    if constexpr (kTailShift) {
        const float real_tail = fmaf(tail.real(), pt.c, -tail.imag() * pt.s);
        const float imag_tail = fmaf(tail.real(), pt.s, tail.imag() * pt.c);
        return {real_tail + real_head, imag_tail + imag_head};
    } else {
        return {tail.real() + real_head, tail.imag() + imag_head};
    }
}

} // namespace loki::core::detail
