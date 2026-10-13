#include "loki/io/preprocess.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
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

__global__ void kernel_block_medians(const float* __restrict__ x,
                                     const uint8_t* __restrict__ mask,
                                     GridDev g,
                                     double min_good,
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
                ok = mask[i0 + j] != 0U;
                v  = x[i0 + j];
            },
            len, a, scan);
        const bool ok = cnt >= min_valid_dev(min_good, len);
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

    ~GpuRobustStages() override { cudaStreamDestroy(m_stream); }

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
            ptr(m_x), ptr(m_mask), to_dev(grid), min_good, ptr(f.a), ptr(f.u),
            ptr(f.b), rows.scratch, rows.cap);
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
    }

    SizeType zap(const BlockGrid& /*grid*/,
                 std::span<const float> /*mu*/,
                 std::span<const float> /*var*/,
                 double /*tsamp*/,
                 const PreprocessOptions& /*o*/) override {
        // preprocess() runs zap and birdie options on the CPU.
        throw std::logic_error("preprocess: the GPU backend does not zap");
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
        copy_down(ptr(m_x), m_ts_e.data(), m_n);
        copy_down(ptr(m_z), m_ts_v.data(), m_n);
    }

    std::span<const float> ts_v() override { return m_ts_v; }

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
        cuda_utils::check_cuda_call(cudaStreamSynchronize(m_stream),
                                    "preprocess: synchronize failed");
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

} // namespace loki::io::detail
