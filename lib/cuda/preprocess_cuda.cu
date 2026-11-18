#include "loki/io/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/fft_long_cuda.cuh"
#include "lib/cuda/search_series_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/detail/keyed_rng.hpp"
#include "lib/io/preprocess_engine.hpp"
#include "lib/io/preprocess_robust.hpp"

namespace loki::io::detail {

namespace {

// One CUDA block per statistics block (or running-median window).
constexpr int kThreads = 128;
// Values staged in shared memory per row; longer rows use global scratch.
constexpr SizeType kSmemCap = 4096;

/// Order-preserving key of a float; -0 and +0 map to the same key, so the
/// selected value of a zero is +0.
__device__ __forceinline__ uint32_t float_key(float v) {
    uint32_t u = __float_as_uint(v);
    u          = u == 0x80000000U ? 0U : u;
    return (u & 0x80000000U) != 0U ? ~u : (u | 0x80000000U);
}

__device__ __forceinline__ float key_float(uint32_t k) {
    const uint32_t u = (k & 0x80000000U) != 0U ? (k & 0x7FFFFFFFU) : ~k;
    return __uint_as_float(u);
}

struct GridDev {
    SizeType n;
    SizeType block;
    SizeType nblocks;

    __device__ SizeType begin(SizeType b) const { return b * block; }
    __device__ SizeType end(SizeType b) const {
        return b + 1 == nblocks ? n : (b + 1) * block;
    }
    __device__ double center(SizeType b) const {
        return static_cast<double>(begin(b)) +
               (0.5 * static_cast<double>(end(b) - begin(b) - 1));
    }
};

GridDev to_dev(const BlockGrid& g) {
    return {.n = g.n, .block = g.block, .nblocks = g.nblocks};
}

/**
 * @brief interpolate_block() value of sample @p j of block @p b.
 *
 * Same IEEE double operations as the host build: a reciprocal of the centre
 * distance, a slope, then one fused multiply-add.
 */
__device__ float
interp(const GridDev& g, const float* __restrict__ v, SizeType b, SizeType j) {
    const SizeType i0 = g.begin(b);
    const double cb   = g.center(b);
    const auto j_mid  = static_cast<SizeType>(ceil(cb)) - i0;
    const auto j_hi   = static_cast<SizeType>(floor(cb)) + 1 - i0;
    const auto lerp   = [&](SizeType lo, SizeType hi) {
        const double cl    = g.center(lo);
        const double dc    = __dsub_rn(g.center(hi), cl);
        const auto vl      = static_cast<double>(v[lo]);
        const double dv    = __dsub_rn(static_cast<double>(v[hi]), vl);
        const double slope = __dmul_rn(dv, __ddiv_rn(1.0, dc));
        const double t     = __dsub_rn(static_cast<double>(i0 + j), cl);
        return __double2float_rn(__fma_rn(t, slope, vl));
    };
    if (j < j_mid) {
        return b > 0 ? lerp(b - 1, b) : v[b];
    }
    if (j < j_hi) {
        return v[b];
    }
    return b + 1 < g.nblocks ? lerp(b, b + 1) : v[b];
}

struct SelectShared {
    uint32_t hist[256];
    uint32_t digit;
    uint32_t below;
    uint32_t lt;
    uint32_t max_lt;
};

/// Key of the rank-@p r value (0-based) of a[0, m): radix select, 8 bits
/// per pass, over the whole block.
__device__ uint32_t select_key(const float* a,
                               uint32_t m,
                               uint32_t r,
                               SelectShared& s) {
    const auto tid  = static_cast<uint32_t>(threadIdx.x);
    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t i = tid; i < 256; i += kThreads) {
            s.hist[i] = 0;
        }
        __syncthreads();
        for (uint32_t i = tid; i < m; i += kThreads) {
            const uint32_t k = float_key(a[i]);
            if ((k & pmask) == prefix) {
                atomicAdd(&s.hist[(k >> shift) & 0xFFU], 1U);
            }
        }
        __syncthreads();
        if (tid < 32) {
            // Lane L owns digits 8L..8L+7.
            uint32_t c[8];
            uint32_t sum = 0;
            for (uint32_t q = 0; q < 8; ++q) {
                c[q] = s.hist[(8 * tid) + q];
                sum += c[q];
            }
            uint32_t incl = sum;
            for (uint32_t off = 1; off < 32; off <<= 1U) {
                const uint32_t t = __shfl_up_sync(0xFFFFFFFFU, incl, off);
                incl += tid >= off ? t : 0;
            }
            const uint32_t excl = incl - sum;
            if (excl <= r && r < incl) {
                uint32_t cum = excl;
                for (uint32_t q = 0; q < 8; ++q) {
                    if (cum + c[q] > r) {
                        s.digit = (8 * tid) + q;
                        s.below = cum;
                        break;
                    }
                    cum += c[q];
                }
            }
        }
        __syncthreads();
        prefix |= s.digit << static_cast<uint32_t>(shift);
        pmask |= 0xFFU << static_cast<uint32_t>(shift);
        r -= s.below;
        __syncthreads();
    }
    return prefix;
}

/// median_inplace() of a[0, m), m >= 1, as a double.
__device__ double block_median(const float* a, uint32_t m, SelectShared& s) {
    const uint32_t h   = m / 2;
    const uint32_t khi = select_key(a, m, h, s);
    const float hi     = key_float(khi);
    if (m % 2 == 1) {
        return static_cast<double>(hi);
    }
    // Rank h - 1 is h's value unless exactly h values are smaller.
    if (threadIdx.x == 0) {
        s.lt     = 0;
        s.max_lt = 0;
    }
    __syncthreads();
    uint32_t lt     = 0;
    uint32_t max_lt = 0;
    for (auto i = static_cast<uint32_t>(threadIdx.x); i < m; i += kThreads) {
        const uint32_t k = float_key(a[i]);
        if (k < khi) {
            ++lt;
            max_lt = max(max_lt, k);
        }
    }
    if (lt > 0) {
        atomicAdd(&s.lt, lt);
        atomicMax(&s.max_lt, max_lt);
    }
    __syncthreads();
    const float lo = s.lt < h ? hi : key_float(s.max_lt);
    __syncthreads();
    return 0.5 * (static_cast<double>(lo) + static_cast<double>(hi));
}

using BlockScan = cub::BlockScan<uint32_t, kThreads>;

/// Copies the valid values of a row into @p out in index order; returns
/// their count. @p load(j) gives (valid, value) for j in [0, len).
template <class Load>
__device__ uint32_t
compact(Load load, SizeType len, float* out, BlockScan::TempStorage& scan) {
    uint32_t base = 0;
    for (SizeType t0 = 0; t0 < len; t0 += kThreads) {
        const SizeType j = t0 + threadIdx.x;
        bool ok          = false;
        float v          = 0.0F;
        if (j < len) {
            load(j, ok, v);
        }
        uint32_t pos   = 0;
        uint32_t total = 0;
        BlockScan(scan).ExclusiveSum(ok ? 1U : 0U, pos, total);
        if (ok) {
            out[base + pos] = v;
        }
        base += total;
        __syncthreads();
    }
    return base;
}

/// min_valid() on a time-domain grid, whose blocks hold at least 16 samples:
/// the floor min(block, 3) is always 3.
__device__ SizeType min_valid_dev(double min_good, SizeType len) {
    const auto need =
        static_cast<SizeType>(ceil(min_good * static_cast<double>(len)));
    return max(need, SizeType{3});
}

/// Row buffers of one CUDA block: shared memory, or a slice of global
/// scratch when the row is longer than kSmemCap.
__device__ float*
row_buffer(float* smem, float* scratch, SizeType row_cap, int which) {
    if (scratch == nullptr) {
        return smem + (static_cast<SizeType>(which) * row_cap);
    }
    return scratch +
           ((static_cast<SizeType>(blockIdx.x) * 2 + which) * row_cap);
}

__global__ void kernel_check_finite(const float* __restrict__ x,
                                    SizeType n,
                                    unsigned int* bad) {
    unsigned int any = 0;
    for (SizeType i = (blockIdx.x * SizeType{blockDim.x}) + threadIdx.x; i < n;
         i += SizeType{gridDim.x} * blockDim.x) {
        any |= (__float_as_uint(x[i]) & 0x7F800000U) == 0x7F800000U ? 1U : 0U;
    }
    if (__syncthreads_or(static_cast<int>(any)) != 0 && threadIdx.x == 0) {
        atomicOr(bad, 1U);
    }
}

/// block_medians(): @p mask may be null (every value valid); @p floor is
/// min(block, 3), the grid's floor of min_valid().
__global__ void kernel_block_medians(const float* __restrict__ x,
                                     const uint8_t* __restrict__ mask,
                                     GridDev g,
                                     double min_good,
                                     SizeType floor,
                                     float* __restrict__ med,
                                     uint8_t* __restrict__ usable,
                                     float* __restrict__ good,
                                     float* scratch,
                                     SizeType row_cap) {
    extern __shared__ float smem[];
    __shared__ SelectShared sel;
    __shared__ BlockScan::TempStorage scan;
    float* a = row_buffer(smem, scratch, row_cap, 0);
    for (SizeType b = blockIdx.x; b < g.nblocks; b += gridDim.x) {
        const SizeType i0  = g.begin(b);
        const SizeType len = g.end(b) - i0;
        const uint32_t cnt = compact(
            [&](SizeType j, bool& ok, float& v) {
                ok = mask == nullptr || mask[i0 + j] != 0U;
                v  = x[i0 + j];
            },
            len, a, scan);
        const auto need =
            static_cast<SizeType>(ceil(min_good * static_cast<double>(len)));
        const bool ok = cnt >= max(need, floor);
        const float m =
            ok ? static_cast<float>(block_median(a, cnt, sel)) : 0.0F;
        if (threadIdx.x == 0) {
            good[b] =
                __fdiv_rn(static_cast<float>(cnt), static_cast<float>(len));
            usable[b] = ok ? 1U : 0U;
            med[b]    = m;
        }
        __syncthreads();
    }
}

__global__ void kernel_block_variances(const float* __restrict__ x,
                                       const uint8_t* __restrict__ mask,
                                       GridDev g,
                                       const float* __restrict__ mu,
                                       double min_good,
                                       float* __restrict__ var,
                                       uint8_t* __restrict__ usable,
                                       float* scratch,
                                       SizeType row_cap) {
    extern __shared__ float smem[];
    __shared__ SelectShared sel;
    __shared__ BlockScan::TempStorage scan;
    float* a = row_buffer(smem, scratch, row_cap, 0);
    float* d = row_buffer(smem, scratch, row_cap, 1);
    for (SizeType b = blockIdx.x; b < g.nblocks; b += gridDim.x) {
        const SizeType i0  = g.begin(b);
        const SizeType len = g.end(b) - i0;
        const uint32_t cnt = compact(
            [&](SizeType j, bool& ok, float& v) {
                ok = mask[i0 + j] != 0U;
                v  = __fsub_rn(x[i0 + j], interp(g, mu, b, j));
            },
            len, a, scan);
        if (cnt < min_valid_dev(min_good, len)) {
            if (threadIdx.x == 0) {
                usable[b] = 0U;
                var[b]    = 0.0F;
            }
            __syncthreads();
            continue;
        }
        const double med = block_median(a, cnt, sel);
        for (auto i = static_cast<uint32_t>(threadIdx.x); i < cnt;
             i += kThreads) {
            d[i] = __double2float_rn(
                fabs(__dsub_rn(static_cast<double>(a[i]), med)));
        }
        __syncthreads();
        const double med_d = block_median(d, cnt, sel);
        if (threadIdx.x == 0) {
            const auto nd        = static_cast<double>(cnt);
            const double small_n = nd > 9.0 ? __ddiv_rn(nd, nd - 0.8) : 1.2;
            double sigma = __dmul_rn(__dmul_rn(kMadScale, small_n), med_d);
            if (sigma <= 0.0) {
                // Quantised data: mean absolute deviation / sqrt(2 / pi),
                // summed in index order as on the host.
                double abs_sum = 0.0;
                for (uint32_t i = 0; i < cnt; ++i) {
                    abs_sum = __dadd_rn(
                        abs_sum,
                        fabs(__dsub_rn(static_cast<double>(a[i]), med)));
                }
                constexpr double kSqrtHalfPi = 1.2533141373155001;
                sigma = __dmul_rn(__ddiv_rn(abs_sum, nd), kSqrtHalfPi);
            }
            usable[b] = sigma > 0.0 ? 1U : 0U;
            var[b]    = __double2float_rn(__dmul_rn(sigma, sigma));
        }
        __syncthreads();
    }
}

/// masked_running_median() without the hole filling: @p have marks outputs
/// whose window held a usable entry.
__global__ void kernel_running_median(const float* __restrict__ v,
                                      const uint8_t* __restrict__ usable,
                                      SizeType nb,
                                      SizeType half,
                                      SizeType first,
                                      SizeType last,
                                      float* __restrict__ out,
                                      uint8_t* __restrict__ have,
                                      float* scratch,
                                      SizeType row_cap) {
    extern __shared__ float smem[];
    __shared__ SelectShared sel;
    __shared__ BlockScan::TempStorage scan;
    float* a       = row_buffer(smem, scratch, row_cap, 0);
    const float a0 = v[first];
    const float a1 = v[last];
    const auto fi  = static_cast<std::int64_t>(first);
    const auto la  = static_cast<std::int64_t>(last);
    for (SizeType i = blockIdx.x; i < nb; i += gridDim.x) {
        const auto lo =
            static_cast<std::int64_t>(i) - static_cast<std::int64_t>(half);
        const uint32_t cnt = compact(
            [&](SizeType jj, bool& ok, float& val) {
                const std::int64_t j = lo + static_cast<std::int64_t>(jj);
                std::int64_t m       = j;
                val                  = 0.0F;
                if (j < fi) {
                    m   = (2 * fi) - j;
                    val = m <= la ? __fsub_rn(2.0F * a0, v[m]) : 0.0F;
                } else if (j > la) {
                    m   = (2 * la) - j;
                    val = m >= fi ? __fsub_rn(2.0F * a1, v[m]) : 0.0F;
                } else {
                    val = v[m];
                }
                ok = m >= fi && m <= la && usable[m] != 0U;
            },
            (2 * half) + 1, a, scan);
        const float m =
            cnt > 0 ? static_cast<float>(block_median(a, cnt, sel)) : 0.0F;
        if (threadIdx.x == 0) {
            have[i] = cnt > 0 ? 1U : 0U;
            out[i]  = m;
        }
        __syncthreads();
    }
}

__global__ void kernel_whiten(const float* __restrict__ x,
                              GridDev g,
                              const float* __restrict__ mu,
                              const float* __restrict__ var,
                              float* __restrict__ z) {
    for (SizeType b = blockIdx.x; b < g.nblocks; b += gridDim.x) {
        const SizeType i0  = g.begin(b);
        const SizeType len = g.end(b) - i0;
        for (SizeType j = threadIdx.x; j < len; j += kThreads) {
            const float r = __fsub_rn(x[i0 + j], interp(g, mu, b, j));
            z[i0 + j]     = __fdiv_rn(r, __fsqrt_rn(interp(g, var, b, j)));
        }
    }
}

/// flag_sums(): one thread per block, summing in index order.
__global__ void kernel_flag_sums(const float* __restrict__ z,
                                 const uint8_t* __restrict__ mask,
                                 GridDev g,
                                 float cl,
                                 SizeType* __restrict__ cnt_out,
                                 double* __restrict__ s1_out,
                                 double* __restrict__ s2_out) {
    for (SizeType b = (blockIdx.x * SizeType{blockDim.x}) + threadIdx.x;
         b < g.nblocks; b += SizeType{gridDim.x} * blockDim.x) {
        SizeType cnt = 0;
        double s1    = 0.0;
        double s2    = 0.0;
        for (SizeType i = g.begin(b); i < g.end(b); ++i) {
            const float v = z[i];
            if (mask[i] != 0U && (cl == 0.0F || fabsf(v) <= cl)) {
                const auto dv = static_cast<double>(v);
                ++cnt;
                s1 = __dadd_rn(s1, dv);
                s2 = __fma_rn(dv, dv, s2);
            }
        }
        cnt_out[b] = cnt;
        s1_out[b]  = s1;
        s2_out[b]  = s2;
    }
}

__global__ void kernel_mask_blocks(uint8_t* __restrict__ mask,
                                   SizeType n,
                                   SizeType block,
                                   const uint8_t* __restrict__ bad) {
    for (SizeType i = (blockIdx.x * SizeType{blockDim.x}) + threadIdx.x; i < n;
         i += SizeType{gridDim.x} * blockDim.x) {
        if (bad[i / block] != 0U) {
            mask[i] = 0U;
        }
    }
}

__global__ void kernel_output(float* __restrict__ x,
                              float* __restrict__ v_out,
                              const uint8_t* __restrict__ mask,
                              GridDev g,
                              const float* __restrict__ mu,
                              const float* __restrict__ var,
                              float cf,
                              float clip2,
                              bool mult,
                              SizeType* __restrict__ n_clip,
                              SizeType* __restrict__ n_mask) {
    __shared__ unsigned long long s_clip;
    __shared__ unsigned long long s_mask;
    for (SizeType b = blockIdx.x; b < g.nblocks; b += gridDim.x) {
        if (threadIdx.x == 0) {
            s_clip = 0;
            s_mask = 0;
        }
        __syncthreads();
        const SizeType i0     = g.begin(b);
        const SizeType len    = g.end(b) - i0;
        unsigned long long nc = 0;
        unsigned long long nm = 0;
        for (SizeType j = threadIdx.x; j < len; j += kThreads) {
            const SizeType i   = i0 + j;
            const float mub    = interp(g, mu, b, j);
            const float varb   = interp(g, var, b, j);
            const float r      = __fsub_rn(x[i], mub);
            const float gain   = mult ? mub : 1.0F;
            const float cg     = __fmul_rn(cf, gain);
            const float wgt    = __fdiv_rn(cg, varb);
            const bool masked  = mask[i] == 0U;
            const bool clipped = !masked && clip2 > 0.0F &&
                                 __fmul_rn(r, r) > __fmul_rn(clip2, varb);
            nm += masked ? 1 : 0;
            nc += clipped ? 1 : 0;
            const bool keep = !masked && !clipped;
            x[i]            = keep ? __fmul_rn(r, wgt) : 0.0F;
            v_out[i]        = keep ? __fmul_rn(cg, wgt) : 0.0F;
        }
        if (nc > 0) {
            atomicAdd(&s_clip, nc);
        }
        if (nm > 0) {
            atomicAdd(&s_mask, nm);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            n_clip[b] = s_clip;
            n_mask[b] = s_mask;
        }
        __syncthreads();
    }
}

// --- Spectral zap kernels (zap_cpu, zap_spectrum and zap_bands on the
// device). Float steps use explicit IEEE intrinsics (-use_fast_math).

/// Grid-stride index helpers.
__device__ __forceinline__ SizeType gtid() {
    return (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
}
__device__ __forceinline__ SizeType gstride() {
    return static_cast<SizeType>(gridDim.x) * blockDim.x;
}

/// data_seed(): FNV-1a of each kZapSeedChunk-sample chunk, one thread per
/// chunk (integer arithmetic: the host's bits).
__global__ void kernel_chunk_hash(const float* __restrict__ x,
                                  SizeType n,
                                  SizeType nchunks,
                                  std::uint64_t* __restrict__ out) {
    for (SizeType c = gtid(); c < nchunks; c += gstride()) {
        std::uint64_t v    = 0xcbf29ce484222325ULL;
        const SizeType end = min(n, (c + 1) * kZapSeedChunk);
        for (SizeType i = c * kZapSeedChunk; i < end; ++i) {
            v = (v ^ __float_as_uint(x[i])) * 0x100000001b3ULL;
        }
        out[c] = v;
    }
}

/// Outliers (|z| > clip) are marked 2 (still valid); every z that is not
/// mask 1 is zeroed, so masked samples and outliers stay out of the
/// spectrum.
__global__ void kernel_zap_prepare(float* __restrict__ z,
                                   uint8_t* __restrict__ mask,
                                   SizeType n,
                                   float clip) {
    for (SizeType i = gtid(); i < n; i += gstride()) {
        const bool outlier = clip > 0.0F && fabsf(z[i]) > clip;
        const uint8_t m    = (mask[i] != 0U && outlier) ? 2U : mask[i];
        mask[i]            = m;
        z[i]               = m == 1U ? z[i] : 0.0F;
    }
}

/// |spec|^2, bin 0 set to zero.
__global__ void kernel_power(const ComplexTypeCUDA* __restrict__ spec,
                             SizeType nf,
                             float* __restrict__ power) {
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        const float re = spec[k].real();
        const float im = spec[k].imag();
        power[k] =
            k == 0 ? 0.0F : __fadd_rn(__fmul_rn(re, re), __fmul_rn(im, im));
    }
}

/// Sets out[k] = 1 for every k in the inclusive ranges (lo, hi pairs).
__global__ void kernel_mark_ranges(const SizeType* __restrict__ ranges,
                                   SizeType nranges,
                                   uint8_t* __restrict__ out) {
    for (SizeType r = blockIdx.x; r < nranges; r += gridDim.x) {
        const SizeType lo = ranges[2 * r];
        const SizeType hi = ranges[(2 * r) + 1];
        for (SizeType k = lo + threadIdx.x; k <= hi; k += blockDim.x) {
            out[k] = 1U;
        }
    }
}

/// valid[k] = (flags[k] == 0), and 0 below @p k_min.
__global__ void kernel_valid_from(const uint8_t* __restrict__ flags,
                                  SizeType nf,
                                  SizeType k_min,
                                  uint8_t* __restrict__ valid) {
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        valid[k] = (k >= k_min && flags[k] == 0U) ? 1U : 0U;
    }
}

/// Local mean power: the interpolated running median over ln 2.
__global__ void kernel_mean_interp(GridDev g,
                                   const float* __restrict__ fmean,
                                   float* __restrict__ mean) {
    constexpr float kLn2 = 0.693147180559945309F;
    for (SizeType b = blockIdx.x; b < g.nblocks; b += gridDim.x) {
        const SizeType i0  = g.begin(b);
        const SizeType len = g.end(b) - i0;
        for (SizeType j = threadIdx.x; j < len; j += blockDim.x) {
            mean[i0 + j] = __fdiv_rn(interp(g, fmean, b, j), kLn2);
        }
    }
}

/// Zap flags (zap_spectrum): bins above the threshold and the birdie
/// windows (phase_kept: only their bins above the local mean). @p flagged
/// gets zap or window.
__global__ void kernel_zap_flags(const float* __restrict__ power,
                                 const float* __restrict__ mean,
                                 const uint8_t* __restrict__ window,
                                 SizeType nf,
                                 bool use_threshold,
                                 double thr,
                                 bool hard,
                                 uint8_t* __restrict__ zap,
                                 uint8_t* __restrict__ flagged) {
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        bool z = false;
        if (k > 0) {
            const float p  = power[k];
            const float mk = mean[k];
            if (use_threshold && mk > 0.0F &&
                static_cast<double>(p) > __dmul_rn(thr, mk)) {
                z = true;
            }
            if (window[k] != 0U && (hard || p > mk)) {
                z = true;
            }
        }
        zap[k]     = z ? 1U : 0U;
        flagged[k] = (z || window[k] != 0U) ? 1U : 0U;
    }
}

/// Counts the entries of @p v equal to @p value (integer atomics: exact).
__global__ void kernel_count_eq(const uint8_t* __restrict__ v,
                                SizeType n,
                                uint8_t value,
                                unsigned long long* __restrict__ count) {
    unsigned long long c = 0;
    for (SizeType i = gtid(); i < n; i += gstride()) {
        c += v[i] == value ? 1ULL : 0ULL;
    }
    for (int off = 16; off > 0; off /= 2) {
        c += __shfl_down_sync(0xffffffffU, c, off);
    }
    if ((threadIdx.x % 32) == 0 && c > 0) {
        atomicAdd(count, c);
    }
}

/// phase_kept: a flagged bin above the local mean keeps its phase and gets
/// the local mean power.
__global__ void kernel_phase_kept(ComplexTypeCUDA* __restrict__ spec,
                                  const float* __restrict__ mean,
                                  const uint8_t* __restrict__ zap,
                                  SizeType nf,
                                  unsigned long long* __restrict__ count) {
    unsigned long long c = 0;
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        if (k == 0 || zap[k] == 0U) {
            continue;
        }
        const float re = spec[k].real();
        const float im = spec[k].imag();
        const auto p   = static_cast<double>(
            __fadd_rn(__fmul_rn(re, re), __fmul_rn(im, im)));
        if (p > mean[k]) {
            const auto f = __double2float_rn(
                __dsqrt_rn(__ddiv_rn(static_cast<double>(mean[k]), p)));
            spec[k] = ComplexTypeCUDA(__fmul_rn(re, f), __fmul_rn(im, f));
            ++c;
        }
    }
    for (int off = 16; off > 0; off /= 2) {
        c += __shfl_down_sync(0xffffffffU, c, off);
    }
    if ((threadIdx.x % 32) == 0 && c > 0) {
        atomicAdd(count, c);
    }
}

/// Zeroes the zapped bins (k >= 1).
__global__ void kernel_zero_zapped(ComplexTypeCUDA* __restrict__ spec,
                                   const uint8_t* __restrict__ zap,
                                   SizeType nf) {
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        if (k > 0 && zap[k] != 0U) {
            spec[k] = ComplexTypeCUDA(0.0F, 0.0F);
        }
    }
}

/// Zeroes the gaps (mask != 1).
__global__ void kernel_zero_gaps(float* __restrict__ z,
                                 const uint8_t* __restrict__ mask,
                                 SizeType n) {
    for (SizeType i = gtid(); i < n; i += gstride()) {
        if (mask[i] != 1U) {
            z[i] = 0.0F;
        }
    }
}

/// Valid samples (mask 1) per gap-retention block of @p block samples.
__global__ void kernel_gap_counts(const uint8_t* __restrict__ mask,
                                  SizeType nd,
                                  SizeType block,
                                  SizeType* __restrict__ counts) {
    for (SizeType j = blockIdx.x; j < nd; j += gridDim.x) {
        unsigned long long c = 0;
        for (SizeType i = (j * block) + threadIdx.x; i < (j + 1) * block;
             i += blockDim.x) {
            c += mask[i] == 1U ? 1ULL : 0ULL;
        }
        for (int off = 16; off > 0; off /= 2) {
            c += __shfl_down_sync(0xffffffffU, c, off);
        }
        __shared__ unsigned long long warp_sums[32];
        if ((threadIdx.x % 32) == 0) {
            warp_sums[threadIdx.x / 32] = c;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long t = 0;
            for (unsigned int w = 0; w < (blockDim.x + 31) / 32; ++w) {
                t += warp_sums[w];
            }
            counts[j] = static_cast<SizeType>(t);
        }
        __syncthreads();
    }
}

/// Multiplies mean over each flagged run [lo, hi) by its gain.
__global__ void kernel_scale_runs(float* __restrict__ mean,
                                  const SizeType* __restrict__ runs,
                                  const float* __restrict__ gains,
                                  SizeType nruns) {
    for (SizeType r = blockIdx.x; r < nruns; r += gridDim.x) {
        const SizeType lo = runs[2 * r];
        const SizeType hi = runs[(2 * r) + 1];
        const float g     = gains[r];
        for (SizeType k = lo + threadIdx.x; k < hi; k += blockDim.x) {
            mean[k] = __fmul_rn(mean[k], g);
        }
    }
}

/// Hard fill of the zapped bins: zero, or complex Gaussian noise of the
/// local mean power from a stream keyed on (seed, bin) -- the host's draws.
__global__ void kernel_noise_fill(ComplexTypeCUDA* __restrict__ spec,
                                  const float* __restrict__ mean,
                                  const uint8_t* __restrict__ zap,
                                  SizeType nf,
                                  bool noise,
                                  bool even,
                                  std::uint64_t seed,
                                  const float* __restrict__ lut) {
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        if (k == 0 || zap[k] == 0U) {
            continue;
        }
        const float mk = mean[k];
        if (!noise || mk <= 0.0F) {
            spec[k] = ComplexTypeCUDA(0.0F, 0.0F);
            continue;
        }
        auto rng = math::make_keyed_pcg32(
            seed, kZapNoisePurpose, 0U, static_cast<std::uint32_t>(k), 0U, 0U);
        const bool real_bin = even && k == nf - 1; // Nyquist is real
        const float sd      = __fsqrt_rn(real_bin ? mk : __fdiv_rn(mk, 2.0F));
        const float re      = math::normal_from_lut(rng(), lut, 0.0F, sd);
        const float im      = math::normal_from_lut(rng(), lut, 0.0F, sd);
        spec[k]             = ComplexTypeCUDA(re, real_bin ? 0.0F : im);
    }
}

/// After the zap: x = mu + z sigma for the valid samples (when @p update),
/// and mask 2 back to 1.
__global__ void kernel_zap_restore(float* __restrict__ x,
                                   const float* __restrict__ z,
                                   uint8_t* __restrict__ mask,
                                   GridDev g,
                                   const float* __restrict__ mu,
                                   const float* __restrict__ var,
                                   bool update) {
    for (SizeType b = blockIdx.x; b < g.nblocks; b += gridDim.x) {
        const SizeType i0  = g.begin(b);
        const SizeType len = g.end(b) - i0;
        for (SizeType j = threadIdx.x; j < len; j += kThreads) {
            const SizeType i = i0 + j;
            if (update && mask[i] == 1U) {
                x[i] = __fadd_rn(
                    interp(g, mu, b, j),
                    __fmul_rn(z[i], __fsqrt_rn(interp(g, var, b, j))));
            }
            mask[i] = mask[i] == 0U ? 0U : 1U;
        }
    }
}

/// Zero runs of each chunk of @p chunk samples of @p v: the leading and
/// trailing zeros and the longest run inside (one thread per chunk, in
/// index order; combined on the host).
__global__ void kernel_zero_runs(const float* __restrict__ v,
                                 SizeType n,
                                 SizeType chunk,
                                 SizeType nchunks,
                                 SizeType* __restrict__ lead,
                                 SizeType* __restrict__ trail,
                                 SizeType* __restrict__ best) {
    for (SizeType c = gtid(); c < nchunks; c += gstride()) {
        const SizeType i0  = c * chunk;
        const SizeType end = min(n, i0 + chunk);
        SizeType run       = 0;
        SizeType b         = 0;
        SizeType first     = end - i0; // all zeros unless a nonzero is seen
        for (SizeType i = i0; i < end; ++i) {
            if (v[i] == 0.0F) {
                ++run;
                b = max(b, run);
            } else {
                if (first == end - i0) {
                    first = i - i0;
                }
                run = 0;
            }
        }
        lead[c]  = first;
        trail[c] = run;
        best[c]  = b;
    }
}

/// zap_bands(): whitened power of every bin, -1 where the local mean is not
/// positive (left out).
__global__ void kernel_whiten_power(float* __restrict__ power,
                                    const float* __restrict__ mean,
                                    SizeType nf) {
    for (SizeType k = gtid(); k < nf; k += gstride()) {
        const float mk = mean[k];
        power[k] =
            mk > 0.0F
                ? __double2float_rn(__ddiv_rn(static_cast<double>(power[k]),
                                              static_cast<double>(mk)))
                : -1.0F;
    }
}

/// Per band (one CUDA block each, bins [lo[i], hi[i])): sums and counts of
/// the whitened power of the flagged and the other bins. Fixed reduction
/// order, so the result does not change from run to run.
__global__ void kernel_band_sums(const float* __restrict__ w,
                                 const uint8_t* __restrict__ flagged,
                                 const SizeType* __restrict__ lo,
                                 const SizeType* __restrict__ hi,
                                 double* __restrict__ sums,
                                 unsigned long long* __restrict__ counts) {
    using Reduce  = cub::BlockReduce<double, kThreads>;
    using ReduceU = cub::BlockReduce<unsigned long long, kThreads>;
    __shared__ typename Reduce::TempStorage rtmp;
    __shared__ typename ReduceU::TempStorage utmp;
    const SizeType band      = blockIdx.x;
    double s_in              = 0.0;
    double s_out             = 0.0;
    unsigned long long c_in  = 0;
    unsigned long long c_out = 0;
    for (SizeType k = lo[band] + threadIdx.x; k < hi[band]; k += kThreads) {
        const float v = w[k];
        if (v < 0.0F) {
            continue;
        }
        if (flagged[k] != 0U) {
            s_in = __dadd_rn(s_in, static_cast<double>(v));
            ++c_in;
        } else {
            s_out = __dadd_rn(s_out, static_cast<double>(v));
            ++c_out;
        }
    }
    const double t_in = Reduce(rtmp).Sum(s_in);
    __syncthreads();
    const double t_out = Reduce(rtmp).Sum(s_out);
    __syncthreads();
    const unsigned long long n_in = ReduceU(utmp).Sum(c_in);
    __syncthreads();
    const unsigned long long n_out = ReduceU(utmp).Sum(c_out);
    if (threadIdx.x == 0) {
        sums[2 * band]         = t_in;
        sums[(2 * band) + 1]   = t_out;
        counts[2 * band]       = n_in;
        counts[(2 * band) + 1] = n_out;
    }
}

template <typename T> using DVec = cuda_utils::DeviceVectorNoInit<T>;

template <typename T> T* ptr(DVec<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

/// The robust passes on one device. The series stays on the device from
/// the upload of the input to the download of ts_e and ts_v.
class GpuRobustStages final : public RobustStages {
public:
    GpuRobustStages(std::span<const float> raw,
                    std::span<float> ts_e,
                    std::span<float> ts_v)
        : m_n(raw.size()),
          m_device(current_device()),
          m_resident(ts_e.empty()),
          m_ts_e(ts_e),
          m_ts_v(ts_v),
          m_x(raw.size()),
          m_z(raw.size()),
          m_mask(raw.size()) {
        cuda_utils::check_cuda_call(
            cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking),
            "preprocess: cudaStreamCreate failed");
        int sms = 1;
        int dev = 0;
        cuda_utils::check_cuda_call(cudaGetDevice(&dev));
        cuda_utils::check_cuda_call(
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        m_sms = static_cast<SizeType>(sms);
        copy_up(raw.data(), ptr(m_x), m_n);
    }

    ~GpuRobustStages() override {
        release_zap();
        cudaStreamDestroy(m_stream);
    }

    GpuRobustStages(const GpuRobustStages&)            = delete;
    GpuRobustStages& operator=(const GpuRobustStages&) = delete;
    GpuRobustStages(GpuRobustStages&&)                 = delete;
    GpuRobustStages& operator=(GpuRobustStages&&)      = delete;

    void check_finite() override {
        DVec<unsigned int> bad(1);
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(ptr(bad), 0, sizeof(unsigned int), m_stream));
        kernel_check_finite<<<grid_for(m_n), 256, 0, m_stream>>>(ptr(m_x), m_n,
                                                                 ptr(bad));
        cuda_utils::check_last_cuda_error("preprocess: check_finite");
        unsigned int h = 0;
        copy_down(ptr(bad), &h, 1);
        if (h != 0U) {
            throw std::invalid_argument(
                "preprocess: input has non-finite values");
        }
    }

    void reset_mask() override {
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(ptr(m_mask), 1, m_n, m_stream));
    }

    void block_medians(const BlockGrid& grid,
                       double min_good,
                       std::span<float> med,
                       std::span<std::uint8_t> usable,
                       std::span<float> good) override {
        const SizeType nb = grid.nblocks;
        auto& f           = block_buffers(nb);
        const Rows rows   = rows_for(grid.block, nb, 1);
        kernel_block_medians<<<rows.grid, kThreads, rows.smem, m_stream>>>(
            ptr(m_x), ptr(m_mask), to_dev(grid), min_good,
            std::min<SizeType>(grid.block, 3), ptr(f.a), ptr(f.u), ptr(f.b),
            rows.scratch, rows.cap);
        cuda_utils::check_last_cuda_error("preprocess: block_medians");
        copy_down(ptr(f.a), med.data(), nb);
        copy_down(ptr(f.u), usable.data(), nb);
        copy_down(ptr(f.b), good.data(), nb);
    }

    std::vector<float> running_median(std::span<const float> v,
                                      std::span<const std::uint8_t> usable,
                                      SizeType k) override {
        const SizeType nb = v.size();
        SizeType first    = 0;
        while (first < nb && usable[first] == 0U) {
            ++first;
        }
        if (first == nb) {
            throw std::invalid_argument(
                "preprocess: no usable block statistics");
        }
        SizeType last = nb - 1;
        while (usable[last] == 0U) {
            --last;
        }
        auto& f = block_buffers(nb);
        copy_up(v.data(), ptr(f.a), nb);
        copy_up(usable.data(), ptr(f.u), nb);
        const SizeType half = k / 2;
        const Rows rows     = rows_for((2 * half) + 1, nb, 1);
        kernel_running_median<<<rows.grid, kThreads, rows.smem, m_stream>>>(
            ptr(f.a), ptr(f.u), nb, half, first, last, ptr(f.b), ptr(f.u2),
            rows.scratch, rows.cap);
        cuda_utils::check_last_cuda_error("preprocess: running_median");
        std::vector<float> out(nb);
        std::vector<std::uint8_t> have(nb);
        copy_down(ptr(f.b), out.data(), nb);
        copy_down(ptr(f.u2), have.data(), nb);
        fill_holes(out, have);
        return out;
    }

    void block_variances(const BlockGrid& grid,
                         std::span<const float> mu,
                         double min_good,
                         std::span<float> var,
                         std::span<std::uint8_t> usable) override {
        const SizeType nb = grid.nblocks;
        auto& f           = block_buffers(nb);
        copy_up(mu.data(), ptr(f.a), nb);
        const Rows rows = rows_for(grid.block, nb, 2);
        kernel_block_variances<<<rows.grid, kThreads, rows.smem, m_stream>>>(
            ptr(m_x), ptr(m_mask), to_dev(grid), ptr(f.a), min_good, ptr(f.b),
            ptr(f.u), rows.scratch, rows.cap);
        cuda_utils::check_last_cuda_error("preprocess: block_variances");
        copy_down(ptr(f.b), var.data(), nb);
        copy_down(ptr(f.u), usable.data(), nb);
    }

    void whiten(const BlockGrid& grid,
                std::span<const float> mu,
                std::span<const float> var) override {
        const SizeType nb = grid.nblocks;
        auto& f           = block_buffers(nb);
        copy_up(mu.data(), ptr(f.a), nb);
        copy_up(var.data(), ptr(f.b), nb);
        kernel_whiten<<<grid_blocks(nb), kThreads, 0, m_stream>>>(
            ptr(m_x), to_dev(grid), ptr(f.a), ptr(f.b), ptr(m_z));
        cuda_utils::check_last_cuda_error("preprocess: whiten");
    }

    void flag_sums(const BlockGrid& g,
                   float clip,
                   std::span<SizeType> cnt,
                   std::span<double> s1,
                   std::span<double> s2) override {
        const SizeType nb = g.nblocks;
        if (m_cnt.size() < nb) {
            m_cnt.resize(nb);
            m_s1.resize(nb);
            m_s2.resize(nb);
        }
        kernel_flag_sums<<<grid_for(nb), kThreads, 0, m_stream>>>(
            ptr(m_z), ptr(m_mask), to_dev(g), clip, ptr(m_cnt), ptr(m_s1),
            ptr(m_s2));
        cuda_utils::check_last_cuda_error("preprocess: flag_sums");
        copy_down(ptr(m_cnt), cnt.data(), nb);
        copy_down(ptr(m_s1), s1.data(), nb);
        copy_down(ptr(m_s2), s2.data(), nb);
    }

    void mask_blocks(const BlockGrid& g,
                     std::span<const std::uint8_t> bad) override {
        if (std::ranges::none_of(bad, [](std::uint8_t v) { return v != 0; })) {
            return;
        }
        if (m_bad.size() < g.nblocks) {
            m_bad.resize(g.nblocks);
        }
        copy_up(bad.data(), ptr(m_bad), g.nblocks);
        kernel_mask_blocks<<<grid_for(m_n), 256, 0, m_stream>>>(
            ptr(m_mask), m_n, g.block, ptr(m_bad));
        cuda_utils::check_last_cuda_error("preprocess: mask_blocks");
        // The caller's flags may die on return: finish the upload first.
        sync("preprocess: mask_blocks sync failed");
    }

    SizeType zap(const BlockGrid& grid,
                 std::span<const float> mu,
                 std::span<const float> var,
                 double tsamp,
                 const PreprocessOptions& o) override {
        const SizeType n  = m_n;
        const SizeType nf = (n / 2) + 1;
        allocate_zap(nf, o.zap_whiten_bins);
        // The noise seed follows the input (x is still the raw series), so
        // a rerun on the same data reproduces the output bit for bit.
        const std::uint64_t seed = o.zap_seed.value_or(
            o.zap_mode == ZapMode::kNoise ? data_seed() : std::uint64_t{0});
        whiten(grid, mu, var);
        const float clip =
            o.clip_sigma > 0.0 ? static_cast<float>(o.clip_sigma) : 0.0F;
        kernel_zap_prepare<<<grid_for(n), 256, 0, m_stream>>>(
            ptr(m_z), ptr(m_mask), n, clip);
        cuda_utils::check_last_cuda_error("preprocess: zap_prepare");
        const SizeType n_zapped = zap_spectrum(tsamp, o, seed);
        // whiten() and the spectrum's medians reused the block buffers.
        const SizeType nb = grid.nblocks;
        auto& f           = block_buffers(nb);
        copy_up(mu.data(), ptr(f.a), nb);
        copy_up(var.data(), ptr(f.b), nb);
        kernel_zap_restore<<<grid_blocks(nb), kThreads, 0, m_stream>>>(
            ptr(m_x), ptr(m_z), ptr(m_mask), to_dev(grid), ptr(f.a), ptr(f.b),
            n_zapped > 0);
        cuda_utils::check_last_cuda_error("preprocess: zap_restore");
        return n_zapped;
    }

    std::vector<PreprocessReport::ZapBand>
    zap_check(double tsamp, const PreprocessOptions& o) override {
        const SizeType n  = m_n;
        const SizeType nf = (n / 2) + 1;
        // x holds ts_e after output().
        m_fft->rfft(ptr(m_x), ptr(m_spec));
        kernel_power<<<grid_for(nf), 256, 0, m_stream>>>(ptr(m_spec), nf,
                                                         ptr(m_power));
        kernel_valid_from<<<grid_for(nf), 256, 0, m_stream>>>(
            ptr(m_flagged), nf, 1, ptr(m_valid));
        cuda_utils::check_last_cuda_error("preprocess: zap_check power");
        local_mean_power(nf, o.zap_whiten_bins, ptr(m_valid));
        kernel_whiten_power<<<grid_for(nf), 256, 0, m_stream>>>(
            ptr(m_power), ptr(m_mean), nf);
        cuda_utils::check_last_cuda_error("preprocess: zap_check whiten");

        // Bins of each band, as the host's walk over k assigns them.
        const double t_obs   = static_cast<double>(n) * tsamp;
        const auto edges     = zap_band_edges(0.5 / tsamp);
        const SizeType nband = edges.size() - 1;
        const auto first_bin = [&](double edge) {
            auto k =
                static_cast<SizeType>(std::max(std::ceil(edge * t_obs), 1.0));
            while (k > 1 && static_cast<double>(k - 1) / t_obs >= edge) {
                --k;
            }
            while (k < nf && static_cast<double>(k) / t_obs < edge) {
                ++k;
            }
            return std::min(k, nf);
        };
        std::vector<SizeType> lo(nband);
        std::vector<SizeType> hi(nband);
        lo[0] = 1;
        for (SizeType i = 1; i < nband; ++i) {
            lo[i]     = std::max(first_bin(edges[i]), lo[i - 1]);
            hi[i - 1] = lo[i];
        }
        hi[nband - 1] = nf;
        DVec<SizeType> d_lo(nband);
        DVec<SizeType> d_hi(nband);
        DVec<double> d_sums(2 * nband);
        DVec<unsigned long long> d_counts(2 * nband);
        copy_up(lo.data(), ptr(d_lo), nband);
        copy_up(hi.data(), ptr(d_hi), nband);
        kernel_band_sums<<<static_cast<unsigned int>(nband), kThreads, 0,
                           m_stream>>>(ptr(m_power), ptr(m_flagged), ptr(d_lo),
                                       ptr(d_hi), ptr(d_sums), ptr(d_counts));
        cuda_utils::check_last_cuda_error("preprocess: zap_check bands");
        std::vector<double> sums(2 * nband);
        std::vector<unsigned long long> counts(2 * nband);
        copy_down(ptr(d_sums), sums.data(), 2 * nband);
        copy_down(ptr(d_counts), counts.data(), 2 * nband);
        release_zap();
        std::vector<PreprocessReport::ZapBand> out;
        for (SizeType i = 0; i < nband; ++i) {
            const double s_in  = sums[2 * i];
            const double s_out = sums[(2 * i) + 1];
            const auto c_in    = static_cast<SizeType>(counts[2 * i]);
            const auto c_out   = static_cast<SizeType>(counts[(2 * i) + 1]);
            if (c_in == 0 || c_out == 0 || !(s_out > 0.0)) {
                continue;
            }
            out.push_back(
                {.f_lo        = edges[i],
                 .f_hi        = edges[i + 1],
                 .n_zapped    = c_in,
                 .power_ratio = (s_in / static_cast<double>(c_in)) /
                                (s_out / static_cast<double>(c_out))});
        }
        return out;
    }

    void output(const BlockGrid& grid,
                std::span<const float> mu,
                std::span<const float> var,
                float cf,
                float clip2,
                bool mult,
                std::span<SizeType> n_clip,
                std::span<SizeType> n_mask) override {
        const SizeType nb = grid.nblocks;
        auto& f           = block_buffers(nb);
        copy_up(mu.data(), ptr(f.a), nb);
        copy_up(var.data(), ptr(f.b), nb);
        if (m_cnt.size() < 2 * nb) {
            m_cnt.resize(2 * nb);
        }
        kernel_output<<<grid_blocks(nb), kThreads, 0, m_stream>>>(
            ptr(m_x), ptr(m_z), ptr(m_mask), to_dev(grid), ptr(f.a), ptr(f.b),
            cf, clip2, mult, ptr(m_cnt), ptr(m_cnt) + nb);
        cuda_utils::check_last_cuda_error("preprocess: output");
        copy_down(ptr(m_cnt), n_clip.data(), nb);
        copy_down(ptr(m_cnt) + nb, n_mask.data(), nb);
        if (!m_resident) {
            copy_down(ptr(m_x), m_ts_e.data(), m_n);
            copy_down(ptr(m_z), m_ts_v.data(), m_n);
        }
    }

    /// The output (ts_e, ts_v) of a resident engine, moved out after
    /// run_robust().
    std::pair<DVec<float>, DVec<float>> take_output() {
        return {std::move(m_x), std::move(m_z)};
    }

    SizeType nsamps() const override { return m_n; }

    SizeType longest_zero_run() override {
        // On the device (ts_v may stay there): per-chunk runs, combined in
        // order on the host.
        constexpr SizeType kChunk = 4096;
        const SizeType nchunks    = (m_n + kChunk - 1) / kChunk;
        DVec<SizeType> d(3 * nchunks);
        kernel_zero_runs<<<grid_for(nchunks), 256, 0, m_stream>>>(
            ptr(m_z), m_n, kChunk, nchunks, ptr(d), ptr(d) + nchunks,
            ptr(d) + (2 * nchunks));
        cuda_utils::check_last_cuda_error("preprocess: zero runs");
        std::vector<SizeType> h(3 * nchunks);
        copy_down(ptr(d), h.data(), 3 * nchunks);
        SizeType best = 0;
        SizeType run  = 0;
        for (SizeType c = 0; c < nchunks; ++c) {
            const SizeType len  = std::min(kChunk, m_n - (c * kChunk));
            const SizeType lead = h[c];
            if (lead == len) {
                run += len;
                continue;
            }
            best = std::max({best, run + lead, h[(2 * nchunks) + c]});
            run  = h[nchunks + c];
        }
        return std::max(best, run);
    }

private:
    struct BlockBuffers {
        DVec<float> a;
        DVec<float> b;
        DVec<std::uint8_t> u;
        DVec<std::uint8_t> u2;
    };
    /// Launch shape of a per-row kernel.
    struct Rows {
        unsigned int grid;
        SizeType smem;
        float* scratch;
        SizeType cap;
    };

    SizeType m_n;
    int m_device;
    bool m_resident; ///< output stays on the device (take_output())
    std::span<float> m_ts_e;
    std::span<float> m_ts_v;
    DVec<float> m_x;
    DVec<float> m_z;
    DVec<std::uint8_t> m_mask;
    BlockBuffers m_blocks;
    DVec<float> m_scratch;
    DVec<SizeType> m_cnt;
    DVec<double> m_s1;
    DVec<double> m_s2;
    DVec<std::uint8_t> m_bad;
    cudaStream_t m_stream{nullptr};
    SizeType m_sms{1};

    // --- Spectral zap state: allocated by zap(), released by zap_check()
    // (or the destructor). flagged survives for the self-check.
    std::optional<math::LongRealFFT> m_fft;
    DVec<ComplexTypeCUDA> m_spec;
    DVec<float> m_power;
    DVec<float> m_mean;
    DVec<std::uint8_t> m_zapf;
    DVec<std::uint8_t> m_window;
    DVec<std::uint8_t> m_valid;
    DVec<std::uint8_t> m_flagged;
    DVec<float> m_lut;
    DVec<unsigned long long> m_count;

    /// Device bytes the zap needs on top of the series (docs/memory.md):
    /// per bin the spectrum, power, mean and four flag bytes; the block
    /// arrays of the local mean power (10 bytes per block, plus long-row
    /// scratch); the cuFFT work area and the noise table; 256 MB of slack
    /// for cuFFT's plan tables, temporaries and allocation granularity
    /// (measured: about 50 MB above the counted bytes at 2^26 and 2^27).
    [[nodiscard]] SizeType zap_bytes(SizeType whiten_bins) const {
        const SizeType nf  = (m_n / 2) + 1;
        const PowerGrid pg = power_grid(whiten_bins, nf);
        const SizeType nb  = (nf + pg.block - 1) / pg.block;
        const SizeType scratch =
            pg.block > kSmemCap ? 4 * m_sms * 2 * pg.block * sizeof(float) : 0;
        return (nf * (sizeof(ComplexTypeCUDA) + (2 * sizeof(float)) + 4)) +
               (nb * 10) + scratch + math::LongRealFFT::workspace_bytes(m_n) +
               kNormalLutBytes + (SizeType{1} << 28U);
    }
    static constexpr SizeType kNormalLutBytes =
        math::kNormalLutSize * sizeof(float);

    void allocate_zap(SizeType nf, SizeType whiten_bins) {
        const SizeType need = zap_bytes(whiten_bins);
        std::size_t free_b  = 0;
        std::size_t total_b = 0;
        cuda_utils::check_cuda_call(cudaMemGetInfo(&free_b, &total_b),
                                    "preprocess: cudaMemGetInfo failed");
        if (need > free_b) {
            throw std::runtime_error(fmt::format(
                "preprocess: the spectral zap of {} samples needs {:.2f} GB "
                "of device memory, but the device has {:.2f} GB free (CUDA "
                "searches zap on the GPU only)",
                m_n, static_cast<double>(need) * 1e-9,
                static_cast<double>(free_b) * 1e-9));
        }
        m_fft.emplace(m_n, m_device, m_stream);
        m_spec.resize(nf);
        m_power.resize(nf);
        m_mean.resize(nf);
        m_zapf.resize(nf);
        m_window.resize(nf);
        m_valid.resize(nf);
        m_flagged.resize(nf);
        m_count.resize(1);
        const auto lut = zap_noise_lut();
        m_lut.resize(lut.size());
        copy_up(lut.data(), ptr(m_lut), lut.size());
    }

    void release_zap() {
        m_fft.reset();
        free_dvec(m_spec);
        free_dvec(m_power);
        free_dvec(m_mean);
        free_dvec(m_zapf);
        free_dvec(m_window);
        free_dvec(m_valid);
        free_dvec(m_flagged);
        free_dvec(m_lut);
    }

    template <typename T> static void free_dvec(DVec<T>& v) {
        v.clear();
        v.shrink_to_fit();
    }

    /// data_seed() of x on the device: the chunk hashes, combined on the
    /// host.
    std::uint64_t data_seed() {
        const SizeType nchunks = (m_n + kZapSeedChunk - 1) / kZapSeedChunk;
        DVec<std::uint64_t> d_h(nchunks);
        kernel_chunk_hash<<<static_cast<unsigned int>((nchunks + 63) / 64), 64,
                            0, m_stream>>>(ptr(m_x), m_n, nchunks, ptr(d_h));
        cuda_utils::check_last_cuda_error("preprocess: chunk_hash");
        std::vector<std::uint64_t> h(nchunks);
        copy_down(ptr(d_h), h.data(), nchunks);
        return combine_chunk_hashes(m_n, h);
    }

    SizeType
    count_eq(const std::uint8_t* v, SizeType count, std::uint8_t value) {
        cuda_utils::check_cuda_call(cudaMemsetAsync(
            ptr(m_count), 0, sizeof(unsigned long long), m_stream));
        kernel_count_eq<<<grid_for(count), 256, 0, m_stream>>>(v, count, value,
                                                               ptr(m_count));
        cuda_utils::check_last_cuda_error("preprocess: count");
        unsigned long long h = 0;
        copy_down(ptr(m_count), &h, 1);
        return static_cast<SizeType>(h);
    }

    /// local_mean_power() of m_power into m_mean: block medians of the
    /// bins with @p valid != 0 (all when null), their running median on the
    /// device with the host's hole filling, then interpolation / ln 2.
    void local_mean_power(SizeType nf,
                          SizeType whiten_bins,
                          const std::uint8_t* valid) {
        const PowerGrid pg = power_grid(whiten_bins, nf);
        const BlockGrid fgrid(nf, pg.block);
        const SizeType nb = fgrid.nblocks;
        auto& f           = block_buffers(nb);
        const Rows rows   = rows_for(fgrid.block, nb, 1);
        kernel_block_medians<<<rows.grid, kThreads, rows.smem, m_stream>>>(
            ptr(m_power), valid, to_dev(fgrid), 0.0,
            std::min<SizeType>(fgrid.block, 3), ptr(f.a), ptr(f.u), ptr(f.b),
            rows.scratch, rows.cap);
        cuda_utils::check_last_cuda_error("preprocess: power medians");
        std::vector<float> med(nb);
        std::vector<std::uint8_t> usable(nb);
        copy_down(ptr(f.a), med.data(), nb);
        copy_down(ptr(f.u), usable.data(), nb);
        const auto fmean = running_median(med, usable, pg.k);
        copy_up(fmean.data(), ptr(f.a), nb);
        kernel_mean_interp<<<grid_blocks(nb), kThreads, 0, m_stream>>>(
            to_dev(fgrid), ptr(f.a), ptr(m_mean));
        cuda_utils::check_last_cuda_error("preprocess: mean_interp");
        // fmean is freed at scope end: finish its upload first.
        sync("preprocess: mean upload sync failed");
    }

    /// zap_spectrum() on the device, on m_z with the sample mask m_mask
    /// (1 = valid; others are gaps, zero in z). Fills m_flagged.
    SizeType
    zap_spectrum(double tsamp, const PreprocessOptions& o, std::uint64_t seed) {
        const SizeType n      = m_n;
        const SizeType nf     = (n / 2) + 1;
        const unsigned int gf = grid_for(nf);
        m_fft->rfft(ptr(m_z), ptr(m_spec));
        kernel_power<<<gf, 256, 0, m_stream>>>(ptr(m_spec), nf, ptr(m_power));
        const double t_obs = static_cast<double>(n) * tsamp;
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(ptr(m_window), 0, nf, m_stream));
        const auto ranges = birdie_ranges(o.birdies, t_obs, nf);
        if (!ranges.empty()) {
            std::vector<SizeType> flat;
            flat.reserve(2 * ranges.size());
            for (const auto& r : ranges) {
                flat.push_back(r.lo);
                flat.push_back(r.hi);
            }
            DVec<SizeType> d_r(flat.size());
            copy_up(flat.data(), ptr(d_r), flat.size());
            kernel_mark_ranges<<<grid_blocks(ranges.size()), 256, 0,
                                 m_stream>>>(ptr(d_r), ranges.size(),
                                             ptr(m_window));
            // d_r is freed at scope end: finish the launch first.
            cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream));
        }
        cuda_utils::check_last_cuda_error("preprocess: zap power/windows");
        const bool hard = o.zap_mode != ZapMode::kPhaseKept;
        // The hard modes leave the birdie windows out of the medians.
        if (hard) {
            kernel_valid_from<<<gf, 256, 0, m_stream>>>(ptr(m_window), nf, 0,
                                                        ptr(m_valid));
        }
        local_mean_power(nf, o.zap_whiten_bins, hard ? ptr(m_valid) : nullptr);
        kernel_zap_flags<<<gf, 256, 0, m_stream>>>(
            ptr(m_power), ptr(m_mean), ptr(m_window), nf, o.zap_periodic,
            exp_power_threshold(o.zap_sigma), hard, ptr(m_zapf),
            ptr(m_flagged));
        cuda_utils::check_last_cuda_error("preprocess: zap flags");

        if (!hard) {
            cuda_utils::check_cuda_call(cudaMemsetAsync(
                ptr(m_count), 0, sizeof(unsigned long long), m_stream));
            kernel_phase_kept<<<gf, 256, 0, m_stream>>>(
                ptr(m_spec), ptr(m_mean), ptr(m_zapf), nf, ptr(m_count));
            cuda_utils::check_last_cuda_error("preprocess: phase_kept");
            unsigned long long nz = 0;
            copy_down(ptr(m_count), &nz, 1);
            if (nz > 0) {
                m_fft->irfft(ptr(m_spec), ptr(m_z));
            }
            return static_cast<SizeType>(nz);
        }

        const SizeType nzap = count_eq(ptr(m_zapf), nf, 1U);
        if (nzap == 0) {
            return 0;
        }
        // Gaps put a fraction of a removed line back; repeat the removal
        // (see zap_spectrum() on the host).
        const SizeType n_gap  = n - count_eq(ptr(m_mask), n, 1U);
        const bool gaps       = static_cast<double>(n_gap) >=
                                kZapMinGapFraction * static_cast<double>(n);
        const SizeType passes = gaps ? std::max<SizeType>(kZapGapPasses, 1) : 1;
        for (SizeType pass = 0; pass + 1 < passes; ++pass) {
            kernel_zero_zapped<<<gf, 256, 0, m_stream>>>(ptr(m_spec),
                                                         ptr(m_zapf), nf);
            m_fft->irfft(ptr(m_spec), ptr(m_z));
            kernel_zero_gaps<<<grid_for(n), 256, 0, m_stream>>>(ptr(m_z),
                                                                ptr(m_mask), n);
            m_fft->rfft(ptr(m_z), ptr(m_spec));
        }
        cuda_utils::check_last_cuda_error("preprocess: zap gap passes");
        if (passes > 1) {
            kernel_power<<<gf, 256, 0, m_stream>>>(ptr(m_spec), nf,
                                                   ptr(m_power));
            kernel_valid_from<<<gf, 256, 0, m_stream>>>(ptr(m_zapf), nf, 0,
                                                        ptr(m_valid));
            local_mean_power(nf, o.zap_whiten_bins, ptr(m_valid));
        }
        if (gaps) {
            raise_fill_for_gaps(nf);
        }
        kernel_noise_fill<<<gf, 256, 0, m_stream>>>(
            ptr(m_spec), ptr(m_mean), ptr(m_zapf), nf,
            o.zap_mode == ZapMode::kNoise, n % 2 == 0, seed, ptr(m_lut));
        m_fft->irfft(ptr(m_spec), ptr(m_z));
        if (n_gap > 0) {
            kernel_zero_gaps<<<grid_for(n), 256, 0, m_stream>>>(ptr(m_z),
                                                                ptr(m_mask), n);
        }
        cuda_utils::check_last_cuda_error("preprocess: zap fill");
        return nzap;
    }

    /// Zeroing the gaps after the fill keeps only part of its in-window
    /// power (GapRetention); draws each flagged run that much stronger.
    void raise_fill_for_gaps(SizeType nf) {
        const SizeType block = gap_retention_block(m_n);
        const SizeType nd    = m_n / block;
        DVec<SizeType> d_cnt(nd);
        kernel_gap_counts<<<grid_blocks(nd), 256, 0, m_stream>>>(
            ptr(m_mask), nd, block, ptr(d_cnt));
        cuda_utils::check_last_cuda_error("preprocess: gap counts");
        std::vector<SizeType> cnt(nd);
        copy_down(ptr(d_cnt), cnt.data(), nd);
        std::vector<float> md(nd);
        for (SizeType j = 0; j < nd; ++j) {
            md[j] = static_cast<float>(cnt[j]) / static_cast<float>(block);
        }
        const GapRetention retention(md);
        // The runs of flagged bins (host scan of the flags).
        std::vector<std::uint8_t> zap(nf);
        copy_down(ptr(m_zapf), zap.data(), nf);
        std::vector<SizeType> runs;
        std::vector<float> gains;
        for (SizeType k = 1; k < nf;) {
            if (zap[k] == 0U) {
                ++k;
                continue;
            }
            SizeType end = k;
            while (end < nf && zap[end] != 0U) {
                ++end;
            }
            const double r = retention(end - k);
            runs.push_back(k);
            runs.push_back(end);
            gains.push_back(r > 0.0 ? static_cast<float>(1.0 / r) : 1.0F);
            k = end;
        }
        if (gains.empty()) {
            return;
        }
        DVec<SizeType> d_runs(runs.size());
        DVec<float> d_gains(gains.size());
        copy_up(runs.data(), ptr(d_runs), runs.size());
        copy_up(gains.data(), ptr(d_gains), gains.size());
        kernel_scale_runs<<<grid_blocks(gains.size()), 256, 0, m_stream>>>(
            ptr(m_mean), ptr(d_runs), ptr(d_gains), gains.size());
        cuda_utils::check_last_cuda_error("preprocess: scale runs");
        cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream));
    }

    BlockBuffers& block_buffers(SizeType nb) {
        if (m_blocks.a.size() < nb) {
            m_blocks.a.resize(nb);
            m_blocks.b.resize(nb);
            m_blocks.u.resize(nb);
            m_blocks.u2.resize(nb);
        }
        return m_blocks;
    }

    /// Rows of up to @p len values, @p arrays buffers each, for @p nrows
    /// rows: shared memory up to kSmemCap values, global scratch beyond.
    Rows rows_for(SizeType len, SizeType nrows, int arrays) {
        if (len <= kSmemCap) {
            return {.grid = grid_blocks(nrows),
                    .smem = static_cast<SizeType>(arrays) * len * sizeof(float),
                    .scratch = nullptr,
                    .cap     = len};
        }
        // Long rows: a few blocks per SM, each with its own scratch rows.
        const SizeType g = std::min<SizeType>(nrows, 4 * m_sms);
        if (m_scratch.size() < g * 2 * len) {
            m_scratch.resize(g * 2 * len);
        }
        return {.grid    = static_cast<unsigned int>(g),
                .smem    = 0,
                .scratch = ptr(m_scratch),
                .cap     = len};
    }

    static int current_device() {
        int dev = 0;
        cuda_utils::check_cuda_call(cudaGetDevice(&dev));
        return dev;
    }

    [[nodiscard]] static unsigned int grid_blocks(SizeType nrows) {
        return static_cast<unsigned int>(
            std::min<SizeType>(nrows, SizeType{1} << 30U));
    }

    [[nodiscard]] unsigned int grid_for(SizeType nthreads_total) const {
        const SizeType blocks = (nthreads_total + 255) / 256;
        return static_cast<unsigned int>(
            std::clamp<SizeType>(blocks, 1, 32 * m_sms));
    }

    template <typename T> void copy_up(const T* h, T* d, SizeType count) {
        cuda_utils::check_cuda_call(cudaMemcpyAsync(d, h, count * sizeof(T),
                                                    cudaMemcpyHostToDevice,
                                                    m_stream),
                                    "preprocess: upload failed");
    }

    template <typename T> void copy_down(const T* d, T* h, SizeType count) {
        cuda_utils::check_cuda_call(cudaMemcpyAsync(h, d, count * sizeof(T),
                                                    cudaMemcpyDeviceToHost,
                                                    m_stream),
                                    "preprocess: download failed");
        sync("preprocess: synchronize failed");
    }

    /// Waits for m_stream. The driver stages a pageable copy_up() source
    /// before the call returns; syncing before the source dies does not
    /// rely on that.
    void sync(const char* what) const {
        cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream), what);
    }
};

} // namespace

PreprocessReport preprocess_gpu(std::span<const float> raw,
                                double tsamp,
                                std::span<float> ts_e,
                                std::span<float> ts_v,
                                const PreprocessOptions& options,
                                int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    GpuRobustStages stages(raw, ts_e, ts_v);
    return run_robust(stages, tsamp, options);
}

std::pair<std::unique_ptr<SeriesDeviceStorage>, PreprocessReport>
preprocess_gpu_resident(std::span<const float> raw,
                        double tsamp,
                        const PreprocessOptions& options,
                        int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    GpuRobustStages stages(raw, {}, {});
    auto rep    = run_robust(stages, tsamp, options);
    auto [e, v] = stages.take_output();
    return {std::make_unique<CudaSeriesStorage>(std::move(e), std::move(v),
                                                device_id),
            std::move(rep)};
}

} // namespace loki::io::detail
