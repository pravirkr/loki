#include "lib/cuda/ffa_score_cuda.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <cuda/std/limits>
#include <cuda_runtime.h>

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_score_device.cuh"
#include "lib/cuda/kernel_utils.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

using namespace detail; // NOLINT(google-build-using-namespace): device helpers

// Where the bins of one final-level profile come from. Depth 1: the final
// level is merged from its parent level (two segments). Depth 2: the last
// two levels are merged from the grandparent level (four segments), in the
// same float-add order as two kernel_ffa_iter levels:
// (G_tt + G_th) + (G_ht + G_hh). Rows are profile indices into the source
// level; shifts are the rounded integer bin shifts.
template <class FinalParents, class PrevParents, int kDepth>
struct FinalSource {
    static constexpr int kRows = 1 << kDepth;
    FinalParents final_parents; // coordinates of the final level
    PrevParents prev_parents;   // coordinates of the level below (depth 2)
    uint32_t ncoords_prev;      // coordinates of the level below the final one
    uint32_t ncoords_src;       // coordinates of the source level
    uint32_t bin_mask;          // nbins - 1 for a power-of-two nbins, else 0

    // Element offsets of the source rows (the source level has fewer than
    // 2^32 elements) and their total bin shifts: two nested rotations by s1
    // then s2 are one rotation by (s1 + s2) mod nbins.
    struct Rows {
        uint32_t off[kRows];
        uint32_t shift[kRows];
    };

    static constexpr int kParentRows = (kDepth == 2) ? 2 : 1;
    struct ParentRows {
        uint32_t off[kParentRows];
        uint32_t shift[kParentRows];
    };

    __device__ __forceinline__ static uint32_t
    add_shift(uint32_t a, uint32_t b, uint32_t nbins) {
        const uint32_t s = a + b;
        return (s >= nbins) ? s - nbins : s;
    }

    __device__ __forceinline__ Rows get_from_th(const TailHead& th,
                                                uint32_t nbins) const {
        Rows r{};
        const uint32_t stride = 2 * nbins;
        if constexpr (kDepth == 1) {
            r.off[0]   = th.i_tail * stride;
            r.off[1]   = (ncoords_src + th.i_head) * stride;
            r.shift[0] = th.shift_tail;
            r.shift[1] = th.shift_head;
        } else {
            const TailHead pt = prev_parents.get(th.i_tail, nbins);
            const TailHead ph = prev_parents.get(th.i_head, nbins);
            r.off[0]          = pt.i_tail * stride;
            r.off[1]          = (ncoords_src + pt.i_head) * stride;
            r.off[2]          = ((2 * ncoords_src) + ph.i_tail) * stride;
            r.off[3]          = ((3 * ncoords_src) + ph.i_head) * stride;
            r.shift[0]        = add_shift(th.shift_tail, pt.shift_tail, nbins);
            r.shift[1]        = add_shift(th.shift_tail, pt.shift_head, nbins);
            r.shift[2]        = add_shift(th.shift_head, ph.shift_tail, nbins);
            r.shift[3]        = add_shift(th.shift_head, ph.shift_head, nbins);
        }
        return r;
    }

    __device__ __forceinline__ Rows get(uint32_t icoord, uint32_t nbins) const {
        return get_from_th(final_parents.get(icoord, nbins), nbins);
    }

    __device__ __forceinline__ ParentRows get_parent(uint32_t seg,
                                                     uint32_t parent_icoord,
                                                     uint32_t nbins) const {
        ParentRows r{};
        const uint32_t stride = 2 * nbins;
        if constexpr (kDepth == 1) {
            r.off[0]   = ((seg * ncoords_src) + parent_icoord) * stride;
            r.shift[0] = 0U;
        } else {
            const TailHead p = prev_parents.get(parent_icoord, nbins);
            r.off[0]         = (((2 * seg) * ncoords_src) + p.i_tail) * stride;
            r.off[1]   = ((((2 * seg) + 1) * ncoords_src) + p.i_head) * stride;
            r.shift[0] = p.shift_tail;
            r.shift[1] = p.shift_head;
        }
        return r;
    }

    __device__ __forceinline__ static ParentRows
    bcast_parent(const ParentRows& mine, int src) {
        ParentRows r;
#pragma unroll
        for (int i = 0; i < kParentRows; ++i) {
            r.off[i]   = __shfl_sync(0xFFFFFFFF, mine.off[i], src);
            r.shift[i] = __shfl_sync(0xFFFFFFFF, mine.shift[i], src);
        }
        return r;
    }

    __device__ __forceinline__ static Rows bcast(const Rows& mine, int src) {
        Rows r;
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            r.off[i]   = __shfl_sync(0xFFFFFFFF, mine.off[i], src);
            r.shift[i] = __shfl_sync(0xFFFFFFFF, mine.shift[i], src);
        }
        return r;
    }

    // (n - shift) mod nbins.
    __device__ __forceinline__ static uint32_t
    rot(uint32_t n, uint32_t shift, uint32_t nbins, uint32_t mask) {
        if (mask != 0) {
            return (n - shift) & mask;
        }
        return (n < shift) ? (n + nbins - shift) : (n - shift);
    }

    // e and v of bin n, in the float-add order of the unfused levels.
    __device__ __forceinline__ void eval(const float* __restrict__ fold,
                                         const Rows& r,
                                         uint32_t n,
                                         uint32_t nbins,
                                         float& e,
                                         float& v) const {
        const float* a = fold + r.off[0] + rot(n, r.shift[0], nbins, bin_mask);
        const float* b = fold + r.off[1] + rot(n, r.shift[1], nbins, bin_mask);
        if constexpr (kDepth == 1) {
            e = a[0] + b[0];
            v = a[nbins] + b[nbins];
        } else {
            const float* c =
                fold + r.off[2] + rot(n, r.shift[2], nbins, bin_mask);
            const float* d =
                fold + r.off[3] + rot(n, r.shift[3], nbins, bin_mask);
            e = (a[0] + b[0]) + (c[0] + d[0]);
            v = (a[nbins] + b[nbins]) + (c[nbins] + d[nbins]);
        }
    }

    __device__ __forceinline__ void eval_parent(const float* __restrict__ fold,
                                                const ParentRows& r,
                                                uint32_t n,
                                                uint32_t nbins,
                                                float& e,
                                                float& v) const {
        const float* a = fold + r.off[0] + rot(n, r.shift[0], nbins, bin_mask);
        if constexpr (kDepth == 1) {
            e = a[0];
            v = a[nbins];
        } else {
            const float* b =
                fold + r.off[1] + rot(n, r.shift[1], nbins, bin_mask);
            e = a[0] + b[0];
            v = a[nbins] + b[nbins];
        }
    }
};

// Warp-regime variant: one warp per profile (blocks hold a few large profiles),
// lanes over bins, e and v rows.
template <class Src>
__device__ __forceinline__ void
merge_final_to_smem_warp(const float* __restrict__ fold,
                         const Src& src,
                         const FFAFinalScoreArgs& args,
                         uint32_t p0,
                         uint32_t nprof,
                         uint32_t nbins,
                         float* __restrict__ s_e,
                         float* __restrict__ s_v,
                         uint32_t pitch) {
    // The block's profiles resolve their rows in parallel (thread i takes
    // profile i), then each warp reads its profile's rows.
    constexpr uint32_t kMaxBlockProfiles = 8;
    __shared__ typename Src::Rows s_rows[kMaxBlockProfiles];
    if (threadIdx.x < nprof) {
        s_rows[threadIdx.x] =
            src.get(tile_order(p0 + threadIdx.x, args), nbins);
    }
    __syncthreads();
    const uint32_t warp_id = threadIdx.x / kWarpSize;
    const uint32_t lane_id = threadIdx.x % kWarpSize;
    const uint32_t nwarps  = blockDim.x / kWarpSize;
    for (uint32_t p = warp_id; p < nprof; p += nwarps) {
        const typename Src::Rows rows = s_rows[p];
        constexpr uint32_t kUnroll    = 4;
        for (uint32_t base = 0; base < nbins; base += kWarpSize * kUnroll) {
            float e[kUnroll];
            float v[kUnroll];
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u) {
                const uint32_t ibin = base + (u * kWarpSize) + lane_id;
                src.eval(fold, rows, min(ibin, nbins - 1), nbins, e[u], v[u]);
            }
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u) {
                const uint32_t ibin = base + (u * kWarpSize) + lane_id;
                if (ibin < nbins) {
                    s_e[(p * pitch) + ibin] = e[u];
                    s_v[(p * pitch) + ibin] = v[u];
                }
            }
        }
    }
}

// Thread-per-profile scoring (kernel_snr_boxcar_thread). That kernel sums
// the normalized bins serially per profile; the normalization compiles to a
// separate multiply (no FMA into the sum), so computing the bins in parallel
// first and then summing them serially gives the same bits. One thread then
// scores its own profile from a shared-memory row; the odd row pitch keeps
// the lanes of a warp on distinct banks.
constexpr uint32_t kSerialProfiles = 128;
constexpr uint32_t kSerialThreads  = kSerialProfiles;

template <class Src, uint32_t NW>
__global__ __launch_bounds__(kSerialThreads) void kernel_final_score_serial(
    const float* __restrict__ fold_in,
    Src src,
    uint32_t ncoords_cur,
    uint32_t nbins,
    FFAFinalScoreArgs args) {
    extern __shared__ float smem[]; // NOLINT
    __shared__ WidthConsts s_wc;
    const uint32_t pitch = nbins + 1;
    float* s_x           = smem;

    load_width_consts(args, static_cast<int>(nbins), s_wc);
    __syncthreads();

    const uint32_t p0    = args.profile_begin + (blockIdx.x * kSerialProfiles);
    const uint32_t nprof = min(kSerialProfiles, ncoords_cur - p0);
    const uint32_t warp_id = threadIdx.x / kWarpSize;
    const uint32_t lane_id = threadIdx.x % kWarpSize;
    const uint32_t base    = warp_id * kWarpSize;
    if (base >= nprof) {
        return;
    }
    const uint32_t n        = min(kWarpSize, nprof - base);
    float* s_warp           = s_x + (base * pitch);
    const uint32_t my_coord = (lane_id < n)
                                  ? tile_order(p0 + base + lane_id, args)
                                  : tile_order(p0 + base, args);
    TailHead th{};
    if (lane_id < n) {
        th = src.final_parents.get(my_coord, nbins);
    }
    const uint32_t t0 = __shfl_sync(0xFFFFFFFFU, th.i_tail, 0);
    const uint32_t h0 = __shfl_sync(0xFFFFFFFFU, th.i_head, 0);
    if (lane_id >= n) {
        th.i_tail = t0;
        th.i_head = h0;
    }

    uint32_t keep_mask = (n == kWarpSize) ? 0xFFFFFFFFU : ((1U << n) - 1U);
    if (args.threshold >= 4.0F) {
        const uint32_t match_t   = loki::utils::warp_match_any(th.i_tail);
        const uint32_t match_h   = loki::utils::warp_match_any(th.i_head);
        const uint32_t rep_t     = static_cast<uint32_t>(__ffs(match_t) - 1);
        const uint32_t rep_h     = static_cast<uint32_t>(__ffs(match_h) - 1);
        const uint32_t leaders_t = __ballot_sync(0xFFFFFFFFU, lane_id == rep_t);
        const uint32_t leaders_h = __ballot_sync(0xFFFFFFFFU, lane_id == rep_h);
        const auto nt            = static_cast<uint32_t>(__popc(leaders_t));
        const auto nh            = static_cast<uint32_t>(__popc(leaders_h));
        const uint32_t np        = nt + nh;
        if (np <= 16U && (np * 2U) <= n) {
            const auto my_tp =
                static_cast<uint32_t>(__popc(leaders_t & ((1U << rep_t) - 1U)));
            const auto my_hp =
                nt +
                static_cast<uint32_t>(__popc(leaders_h & ((1U << rep_h) - 1U)));
            auto* s_prows = reinterpret_cast<typename Src::ParentRows*>(s_warp);
            if (lane_id == rep_t) {
                s_prows[my_tp] = src.get_parent(0U, th.i_tail, nbins);
            }
            if (lane_id == rep_h) {
                s_prows[my_hp] = src.get_parent(1U, th.i_head, nbins);
            }
            __syncwarp();
            typename Src::ParentRows my_pr{};
            if (lane_id < np) {
                my_pr = s_prows[lane_id];
            }
            __syncwarp();

            typename Src::ParentRows prows[16];
#pragma unroll
            for (uint32_t u = 0; u < 16; ++u) {
                prows[u] =
                    Src::bcast_parent(my_pr, static_cast<int>(min(u, np - 1U)));
            }
            for (uint32_t ibin = lane_id; ibin < nbins; ibin += kWarpSize) {
                float ep[16];
                float vp[16];
#pragma unroll
                for (uint32_t u = 0; u < 16; ++u) {
                    if (u < np) {
                        src.eval_parent(fold_in, prows[u], ibin, nbins, ep[u],
                                        vp[u]);
                    }
                }
#pragma unroll
                for (uint32_t u = 0; u < 16; ++u) {
                    if (u < np) {
                        s_warp[(u * pitch) + ibin] = ep[u] * args.fold_scale;
                        s_warp[((u + 16) * pitch) + ibin] =
                            vp[u] * args.fold_scale;
                    }
                }
            }
            __syncwarp();

            float u_p    = 0.0F;
            float n_p    = 0.0F;
            float vmin_p = 0.0F;
            float vmax_p = 0.0F;
            if (lane_id < np) {
                float* ep       = s_warp + (lane_id * pitch);
                const float* vp = s_warp + ((lane_id + 16) * pitch);
                float running   = 0.0F;
                float abs_e     = 0.0F;
                vmin_p          = cuda::std::numeric_limits<float>::max();
                vmax_p          = 0.0F;
#pragma unroll 8
                for (int i = 0; i < static_cast<int>(nbins); ++i) {
                    const float e_val = ep[i];
                    const float v_val = vp[i];
                    running += e_val;
                    ep[i] = running;
                    abs_e += fabsf(e_val);
                    vmin_p = fminf(vmin_p, v_val);
                    vmax_p = fmaxf(vmax_p, v_val);
                }
                u_p       = cuda::std::numeric_limits<float>::lowest();
                float hbm = 0.0F;
                score_prefixed_row_serial<NW>(
                    ep, running, static_cast<int>(nbins), s_wc, args.nwidths,
                    [&](uint32_t iw, float snr) {
                        u_p = fmaxf(u_p, snr);
                        hbm = fmaxf(hbm, fmaxf(s_wc.h[iw], s_wc.b[iw]));
                    });
                n_p = hbm * abs_e;
            }
            const float u_t =
                __shfl_sync(0xFFFFFFFFU, u_p, static_cast<int>(my_tp));
            const float n_t =
                __shfl_sync(0xFFFFFFFFU, n_p, static_cast<int>(my_tp));
            const float vmin_t =
                __shfl_sync(0xFFFFFFFFU, vmin_p, static_cast<int>(my_tp));
            const float vmax_t =
                __shfl_sync(0xFFFFFFFFU, vmax_p, static_cast<int>(my_tp));
            const float u_h =
                __shfl_sync(0xFFFFFFFFU, u_p, static_cast<int>(my_hp));
            const float n_h =
                __shfl_sync(0xFFFFFFFFU, n_p, static_cast<int>(my_hp));
            const float vmin_h =
                __shfl_sync(0xFFFFFFFFU, vmin_p, static_cast<int>(my_hp));
            const float vmax_h =
                __shfl_sync(0xFFFFFFFFU, vmax_p, static_cast<int>(my_hp));
            const float tot_vmin = vmin_t + vmin_h;
            const float tot_vmax = vmax_t + vmax_h;
            bool keep            = (lane_id < n);
            if (tot_vmin > 0.0F && tot_vmax >= tot_vmin) {
                const float inv_vmin = rsqrtf(tot_vmin);
                const float inv_vmax = rsqrtf(tot_vmax);
                const float mu       = 0.5F * (inv_vmin + inv_vmax);
                const float eps      = 0.5F * (inv_vmin - inv_vmax);
                const float ub       = (mu * (u_t + u_h)) + (eps * (n_t + n_h));
                keep = keep && (ub >= (args.threshold - 1.0e-3F));
            }
            keep_mask = __ballot_sync(0xFFFFFFFFU, keep);
            if (keep_mask == 0U) {
                return;
            }
            __syncwarp();
        }
    }

    constexpr uint32_t kUnroll = 16;
    typename Src::Rows mine{};
    if (lane_id < n) {
        mine = src.get_from_th(th, nbins);
    }
    for (uint32_t q = 0; q < n; q += kUnroll) {
        if (((keep_mask >> q) & 0xFFFFU) == 0U) {
            continue;
        }
        typename Src::Rows rows[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            rows[u] = Src::bcast(mine, static_cast<int>(min(q + u, n - 1)));
        }
        for (uint32_t ibin = lane_id; ibin < nbins; ibin += kWarpSize) {
            float e[kUnroll];
            float v[kUnroll];
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u) {
                src.eval(fold_in, rows[u], ibin, nbins, e[u], v[u]);
            }
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u) {
                if (q + u < n) {
                    s_warp[((q + u) * pitch) + ibin] = fold_norm_bin(
                        e[u] * args.fold_scale, v[u] * args.fold_scale);
                }
            }
        }
    }
    __syncwarp();
    if (((keep_mask >> lane_id) & 1U) == 0U) {
        return;
    }

    score_row_serial<NW>(s_warp + (lane_id * pitch), static_cast<int>(nbins),
                         s_wc, my_coord * args.nwidths, args);
}

// Warp-per-profile scoring (kernel_snr_boxcar_warp): a warp scan in chunks of
// 32 bins, the prefix sums written over the e row.
template <class Src, uint32_t NW>
__global__ void kernel_final_score_warp(const float* __restrict__ fold_in,
                                        Src src,
                                        uint32_t ncoords_cur,
                                        uint32_t nbins,
                                        FFAFinalScoreArgs args) {
    extern __shared__ float smem[]; // NOLINT
    __shared__ WidthConsts s_wc;
    const uint32_t nwarps = blockDim.x / kWarpSize;
    float* s_e            = smem;
    float* s_v            = s_e + (nwarps * nbins);

    const uint32_t p0    = args.profile_begin + (blockIdx.x * nwarps);
    const uint32_t nprof = min(nwarps, ncoords_cur - p0);
    merge_final_to_smem_warp(fold_in, src, args, p0, nprof, nbins, s_e, s_v,
                             nbins);
    load_width_consts(args, static_cast<int>(nbins), s_wc);
    __syncthreads();

    const uint32_t warp_id = threadIdx.x / kWarpSize;
    const int lane_id      = static_cast<int>(threadIdx.x % kWarpSize);
    if (warp_id >= nprof) {
        return;
    }
    score_ev_warp<NW>(s_e + (warp_id * nbins), s_v + (warp_id * nbins),
                      args.fold_scale, static_cast<int>(nbins), s_wc,
                      tile_order(p0 + warp_id, args) * args.nwidths, args,
                      lane_id);
}

template <class Src, uint32_t NW>
void launch_final_score_nw(const float* fold_in,
                           const Src& src,
                           SizeType ncoords_cur,
                           SizeType nbins,
                           const FFAFinalScoreArgs& args,
                           cudaStream_t stream) {
    // Kernels see the end of the profile range as their level size.
    const SizeType end    = std::min<SizeType>(ncoords_cur, args.profile_end);
    const SizeType nrange = end - args.profile_begin;
    const auto ncur       = static_cast<uint32_t>(end);
    const auto nb         = static_cast<uint32_t>(nbins);
    // Same regime choice as score_and_filter_cuda_d.
    constexpr SizeType kThreadRegimeMaxBins     = 64;
    constexpr SizeType kThreadRegimeMinProfiles = 1U << 16U;
    if (nbins <= kThreadRegimeMaxBins &&
        ncoords_cur >= kThreadRegimeMinProfiles) {
        const SizeType smem = kSerialProfiles * (nbins + 1) * sizeof(float);
        const dim3 grid((nrange + kSerialProfiles - 1) / kSerialProfiles);
        const dim3 block(kSerialThreads);
        cuda_utils::check_kernel_launch_params(grid, block, smem);
        // Occupancy is bound by shared memory: prefer the largest carveout.
        cuda_utils::prefer_max_shared_carveout(
            kernel_final_score_serial<Src, NW>);
        kernel_final_score_serial<Src, NW>
            <<<grid, block, smem, stream>>>(fold_in, src, ncur, nb, args);
        cuda_utils::check_last_cuda_error(
            "kernel_final_score_serial launch failed");
        return;
    }
    // As many warps (profiles) per block as fit 48 KiB with the static
    // width table, at most 8.
    const SizeType per_profile = 2 * nbins * sizeof(float);
    const SizeType budget      = (48U * 1024U) - sizeof(WidthConsts);
    const SizeType nwarps = std::clamp<SizeType>(budget / per_profile, 1, 8);
    const SizeType smem   = nwarps * per_profile;
    const dim3 grid((nrange + nwarps - 1) / nwarps);
    const dim3 block(static_cast<uint32_t>(nwarps * kWarpSize));
    cuda_utils::check_kernel_launch_params(grid, block, smem);
    cuda_utils::prefer_max_shared_carveout(kernel_final_score_warp<Src, NW>);
    kernel_final_score_warp<Src, NW>
        <<<grid, block, smem, stream>>>(fold_in, src, ncur, nb, args);
    cuda_utils::check_last_cuda_error("kernel_final_score_warp launch failed");
}

template <class Src>
void launch_final_score(const float* fold_in,
                        const Src& src,
                        SizeType ncoords_cur,
                        SizeType nbins,
                        const FFAFinalScoreArgs& args,
                        cudaStream_t stream) {
    if (std::min<SizeType>(ncoords_cur, args.profile_end) <=
        args.profile_begin) {
        return;
    }
    error_check::check_less_equal(
        SizeType{args.nwidths}, SizeType{kMaxWidths},
        "ffa_final_score_cuda: too many boxcar widths");
    error_check::check_less_equal(
        ncoords_cur * args.nwidths, SizeType{UINT32_MAX},
        "ffa_final_score_cuda: score index exceeds 32 bits");
    // Width buckets: the per-width maxima live in registers, and slots past
    // nwidths are predicated off.
    if (args.nwidths <= 8) {
        launch_final_score_nw<Src, 8>(fold_in, src, ncoords_cur, nbins, args,
                                      stream);
    } else if (args.nwidths <= 12) {
        launch_final_score_nw<Src, 12>(fold_in, src, ncoords_cur, nbins, args,
                                       stream);
    } else if (args.nwidths <= 16) {
        launch_final_score_nw<Src, 16>(fold_in, src, ncoords_cur, nbins, args,
                                       stream);
    } else if (args.nwidths <= 24) {
        launch_final_score_nw<Src, 24>(fold_in, src, ncoords_cur, nbins, args,
                                       stream);
    } else {
        launch_final_score_nw<Src, kMaxWidths>(fold_in, src, ncoords_cur, nbins,
                                               args, stream);
    }
}

template <class FinalParents, class Parents, class CoordPtrs>
void dispatch_final_score(const float* fold_in,
                          FinalParents final_parents,
                          CoordPtrs coords_prev,
                          SizeType ncoords_cur,
                          SizeType ncoords_prev,
                          SizeType ncoords_src,
                          int depth,
                          SizeType nbins,
                          const FFAFinalScoreArgs& args,
                          cudaStream_t stream) {
    // Source-row element offsets are 32-bit.
    const SizeType nseg_src = depth == 2 ? 4 : 2;
    error_check::check_less_equal(
        nseg_src * ncoords_src * 2 * nbins, SizeType{UINT32_MAX},
        "ffa_final_score_cuda: source level exceeds 32-bit offsets");
    const auto bin_mask =
        static_cast<uint32_t>((nbins & (nbins - 1)) == 0 ? nbins - 1 : 0);
    if (depth == 1) {
        const FinalSource<FinalParents, Parents, 1> src{
            .final_parents = final_parents,
            .prev_parents  = Parents{coords_prev},
            .ncoords_prev  = static_cast<uint32_t>(ncoords_prev),
            .ncoords_src   = static_cast<uint32_t>(ncoords_prev),
            .bin_mask      = bin_mask};
        launch_final_score(fold_in, src, ncoords_cur, nbins, args, stream);
    } else if (depth == 2) {
        const FinalSource<FinalParents, Parents, 2> src{
            .final_parents = final_parents,
            .prev_parents  = Parents{coords_prev},
            .ncoords_prev  = static_cast<uint32_t>(ncoords_prev),
            .ncoords_src   = static_cast<uint32_t>(ncoords_src),
            .bin_mask      = bin_mask};
        launch_final_score(fold_in, src, ncoords_cur, nbins, args, stream);
    } else {
        throw std::invalid_argument("ffa_final_score_cuda: depth must be 1 "
                                    "or 2");
    }
}

} // namespace

void ffa_final_score_cuda(const float* __restrict__ fold_in,
                          coord::FFACoordDPtrs coords_final,
                          coord::FFACoordDPtrs coords_prev,
                          SizeType ncoords_cur,
                          SizeType ncoords_prev,
                          SizeType ncoords_src,
                          int depth,
                          SizeType nbins,
                          const FFAFinalScoreArgs& args,
                          cudaStream_t stream) {
    dispatch_final_score<detail::AccelParents, detail::AccelParents>(
        fold_in, detail::AccelParents{coords_final}, coords_prev, ncoords_cur,
        ncoords_prev, ncoords_src, depth, nbins, args, stream);
}

void ffa_final_score_cuda(const float* __restrict__ fold_in,
                          const FFAAccelResolveLevel& final_level,
                          coord::FFACoordDPtrs coords_prev,
                          SizeType ncoords_cur,
                          SizeType ncoords_prev,
                          SizeType ncoords_src,
                          int depth,
                          SizeType nbins,
                          const FFAFinalScoreArgs& args,
                          cudaStream_t stream) {
    dispatch_final_score<detail::AccelParentsComputed, detail::AccelParents>(
        fold_in, detail::AccelParentsComputed{final_level}, coords_prev,
        ncoords_cur, ncoords_prev, ncoords_src, depth, nbins, args, stream);
}

void ffa_freq_final_score_cuda(const float* __restrict__ fold_in,
                               coord::FFACoordFreqDPtrs coords_final,
                               coord::FFACoordFreqDPtrs coords_prev,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev,
                               SizeType ncoords_src,
                               int depth,
                               SizeType nbins,
                               const FFAFinalScoreArgs& args,
                               cudaStream_t stream) {
    dispatch_final_score<detail::FreqParents, detail::FreqParents>(
        fold_in, detail::FreqParents{coords_final}, coords_prev, ncoords_cur,
        ncoords_prev, ncoords_src, depth, nbins, args, stream);
}

} // namespace loki::core
