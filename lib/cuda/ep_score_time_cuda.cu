#include <algorithm>
#include <cstdint>

#include <cuda/std/limits>
#include <cuda_runtime.h>

#include "lib/algorithms/ep_memory.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ep_score_cuda.cuh"
#include "lib/cuda/ep_score_device.cuh"
#include "lib/cuda/ffa_score_device.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

using detail::EPBlock;
using detail::fold_norm_bin;
using detail::kWarpSize;
using detail::WidthConstsView;

constexpr uint32_t kThreads = 128;
// The thread scorer takes one leaf per thread of the first warp.
constexpr uint32_t kSerialProfiles = 32;
// Warp scorer: e and v rows of at most this many bytes per block, as the
// Fourier kernels' row budget, and at most kMaxWarpProfiles leaves.
constexpr uint32_t kMaxRowBytes      = 12U * 1024U;
constexpr uint32_t kMaxWarpProfiles  = 16;
constexpr uint32_t kMaxBlockProfiles = 32;
// The largest rows the warp scorer keeps on chip (one leaf per block):
// e and v of kEPCudaFusedMaxNbinsTime bins, 40 KiB.
static_assert(2 * (algorithms::detail::kEPCudaFusedMaxNbinsTime | 1U) *
                  sizeof(float) <=
              40U * 1024U);

// Same regime choice as score_and_filter_max_cuda_d.
bool serial_regime(SizeType nbins, SizeType n_leaves) {
    return nbins <= 64 && n_leaves >= (SizeType{1} << 16U);
}

// Kadane: one normalized row per leaf, at most kMaxBlockProfiles leaves.
uint32_t profiles_per_block(SizeType nbins, bool serial, bool kadane) {
    if (serial) {
        return kSerialProfiles;
    }
    if (kadane) {
        const SizeType fit = kMaxRowBytes / ((nbins | 1U) * sizeof(float));
        return static_cast<uint32_t>(
            std::clamp<SizeType>(fit, 1, kMaxBlockProfiles));
    }
    const SizeType fit = kMaxRowBytes / (2 * ((nbins | 1U)) * sizeof(float));
    return static_cast<uint32_t>(
        std::clamp<SizeType>(fit, 1, kMaxWarpProfiles));
}

// Shift of a leaf in whole bins, as kernel_shift_add_linear rounds it.
__device__ __forceinline__ uint32_t whole_shift(float phase_shift,
                                                uint32_t nbins) {
    uint32_t shift = __float2uint_rz(phase_shift + 0.5F);
    return shift == nbins ? 0U : shift;
}

/**
 * One pruning batch, fused, time domain: each block takes kpb leaves. The
 * threads form each leaf's e and v (tree + FFA rotated by the whole-bin
 * shift, kernel_shift_add_linear's arithmetic) into shared-memory rows, and
 * score them with the thread or warp scorer of score_and_filter_max_cuda_d,
 * or (kKadane) with the Kadane scorer on one normalized row per leaf.
 */
template <bool kSerial, bool kKadane, uint32_t NW>
__global__ __launch_bounds__(kThreads) void kernel_ep_shift_add_score_time(
    EPTimeScoreArgs a, uint32_t kpb) {
    extern __shared__ __align__(16) unsigned char smem_raw[]; // NOLINT
    __shared__ EPBlock<kMaxBlockProfiles> b;
    auto* rows = reinterpret_cast<float*>(smem_raw);
    // Width constants (any count) after the rows.
    const uint32_t row_floats =
        kpb * ((kSerial || kKadane) ? 1U : 2U) * (a.nbins | 1U);
    auto* wc_w = reinterpret_cast<int*>(rows + row_floats);
    auto* wc_h = reinterpret_cast<float*>(wc_w + a.nwidths);
    [[maybe_unused]] WidthConstsView s_wc{wc_w, wc_h, wc_h + a.nwidths};

    const uint32_t n        = a.nbins;
    const uint32_t stride   = 2 * n;
    const uint32_t pitch    = n | 1U;
    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const bool direct       = a.tree_folds != nullptr;
    uint32_t p0;
    uint32_t nprof;
    detail::ep_block_begin(b, a, kpb, tid, nthreads, p0, nprof);
    if constexpr (!kKadane) {
        detail::load_width_consts(a.widths, a.nwidths, static_cast<int>(n),
                                  s_wc, tid, nthreads);
    }
    __syncthreads();

    // e_j = tree_e[j] + ffa_e[(j - shift) mod n], and v likewise.
    const auto merge = [&](uint32_t q, uint32_t j, float& e, float& v) {
        const uint32_t shift   = whole_shift(b.shift[q], n);
        const uint32_t idx_add = (j < shift) ? (j + n - shift) : (j - shift);
        const float* __restrict__ tree =
            a.folds_tree + (static_cast<size_t>(b.tree[q]) * stride);
        const float* __restrict__ ffa =
            a.folds_ffa + (static_cast<size_t>(b.ffa[q]) * stride);
        e = tree[j] + ffa[idx_add];
        v = tree[n + j] + ffa[n + idx_add];
    };
    for (uint32_t idx = tid; idx < nprof * n; idx += nthreads) {
        const uint32_t q = idx / n;
        const uint32_t j = idx - (q * n);
        if (b.valid[q] == 0) {
            continue;
        }
        float e;
        float v;
        merge(q, j, e, v);
        if constexpr (kSerial) {
            rows[(q * pitch) + j] = fold_norm_bin(e, v);
        } else if constexpr (kKadane) {
            // As kernel_kadane_*: e * rsqrt(v), with no guard.
            rows[(q * pitch) + j] = e * rsqrtf(v);
        } else {
            rows[(((q * 2) + 0) * pitch) + j] = e;
            rows[(((q * 2) + 1) * pitch) + j] = v;
        }
        if (!direct) {
            float* __restrict__ out =
                a.folds_out + (static_cast<size_t>(p0 + q) * stride);
            out[j]     = e;
            out[n + j] = v;
        }
    }
    __syncthreads();

    uint32_t npass = 0;
    if constexpr (kKadane) {
        detail::ep_score_kadane_rows(b, a, rows, pitch, n, p0, nprof, tid,
                                     nthreads, npass);
    } else if constexpr (kSerial) {
        if (tid < nprof) {
            if (b.valid[tid] != 0) {
                float best = cuda::std::numeric_limits<float>::lowest();
                detail::score_row_serial_sink<NW>(
                    rows + (tid * pitch), static_cast<int>(n), s_wc, a.nwidths,
                    [&best](uint32_t /*iw*/, float snr) {
                        best = fmaxf(best, snr);
                    });
                detail::ep_block_record(b, a, tid, p0, best, npass);
            } else {
                a.filtered_mask[p0 + tid] = 0;
            }
        }
    } else {
        const uint32_t warp_id = tid / kWarpSize;
        const auto lane_id     = static_cast<int>(tid % kWarpSize);
        const uint32_t nwarps  = nthreads / kWarpSize;
        for (uint32_t p = warp_id; p < nprof; p += nwarps) {
            if (b.valid[p] == 0) {
                if (lane_id == 0) {
                    a.filtered_mask[p0 + p] = 0;
                }
                continue;
            }
            float best   = cuda::std::numeric_limits<float>::lowest();
            float* e_row = rows + ((p * 2) * pitch);
            detail::score_ev_warp_sink<NW>(
                e_row, e_row + pitch, 1.0F, static_cast<int>(n), s_wc,
                a.nwidths,
                [&best](uint32_t /*iw*/, float snr) {
                    best = fmaxf(best, snr);
                },
                lane_id);
            if (lane_id == 0) {
                detail::ep_block_record(b, a, p, p0, best, npass);
            }
        }
    }
    if (!detail::ep_block_finish(b, a, npass, p0, nprof, tid, nthreads)) {
        return;
    }
    // Direct mode: the survivors' folds, merged again from the L2-resident
    // rows with the same arithmetic.
    for (uint32_t idx = tid; idx < nprof * n; idx += nthreads) {
        const uint32_t q = idx / n;
        const uint32_t j = idx - (q * n);
        if (b.flag[q] == 0) {
            continue;
        }
        float e;
        float v;
        merge(q, j, e, v);
        float* __restrict__ out =
            a.tree_folds +
            (static_cast<size_t>(detail::ep_block_slot(b, a, q)) * stride);
        out[j]     = e;
        out[n + j] = v;
    }
}

template <bool kSerial, bool kKadane, uint32_t NW>
void launch_cfg(const EPTimeScoreArgs& args, cudaStream_t stream) {
    const uint32_t kpb = profiles_per_block(args.nbins, kSerial, kKadane);
    const SizeType smem =
        (static_cast<SizeType>(kpb) * ((kSerial || kKadane) ? 1 : 2) *
         (args.nbins | 1U) * sizeof(float)) +
        (kKadane ? 0 : static_cast<SizeType>(args.nwidths) * 3 * 4);
    const auto kernel = kernel_ep_shift_add_score_time<kSerial, kKadane, NW>;
    cuda_utils::set_kernel_dynamic_smem(kernel, smem);
    const dim3 grid((args.n_leaves + kpb - 1) / kpb);
    kernel<<<grid, kThreads, smem, stream>>>(args, kpb);
    cuda_utils::check_last_cuda_error(
        "kernel_ep_shift_add_score_time launch failed");
}

template <bool kSerial>
void launch_regime(const EPTimeScoreArgs& args, cudaStream_t stream) {
    launch_cfg<kSerial, false, detail::kEPWidthGroup>(args, stream);
}

} // namespace

template <>
bool ep_shift_add_score_supported<float>(SizeType nbins,
                                         SizeType nwidths) noexcept {
    return nbins >= 2 &&
           nbins <= algorithms::detail::kEPCudaFusedMaxNbinsTime &&
           nwidths >= 1;
}

template <>
SizeType ep_shift_add_score_tile_words<float>(SizeType nbins,
                                              SizeType n_leaves,
                                              bool kadane) {
    const SizeType kpb =
        kadane
            ? profiles_per_block(nbins, false, true)
            : profiles_per_block(nbins, serial_regime(nbins, n_leaves), false);
    return ((n_leaves + kpb - 1) / kpb) + 1;
}

template <>
void ep_shift_add_score_cuda<float>(const EPTimeScoreArgs& args,
                                    cudaStream_t stream) {
    if (args.n_leaves == 0) {
        return;
    }
    const bool kadane = args.kadane_biases != nullptr;
    error_check::check(
        ep_shift_add_score_supported<float>(args.nbins,
                                            kadane ? 1 : args.nwidths),
        "ep_shift_add_score_cuda: unsupported nbins or number of widths");
    if (kadane) {
        error_check::check(args.nbiases == 2 || args.nbiases == 3,
                           "ep_shift_add_score_cuda: Kadane takes 2 or 3 "
                           "biases");
        launch_cfg<false, true, detail::kEPWidthGroup>(args, stream);
        return;
    }
    if (serial_regime(args.nbins, args.n_leaves)) {
        launch_regime<true>(args, stream);
    } else {
        launch_regime<false>(args, stream);
    }
}

} // namespace loki::core
