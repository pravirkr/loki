#pragma once

/**
 * @file ep_score_device.cuh
 * @brief Block-level pieces shared by the fused EP kernels
 * (ep_score_fourier_cuda.cu, ep_score_time_cuda.cu): the leaves of a block,
 * the per-leaf results and batch statistics, and the direct write of the
 * survivors into the world tree in leaf order. CUDA only.
 */

#include <cstdint>

#include <cuda_runtime.h>

#include "lib/cuda/ep_score_cuda.cuh"
#include "lib/cuda/ffa_score_device.cuh"

namespace loki::core::detail {

// Ordered compaction across blocks (decoupled look-back): tile t publishes
// its pass count, then its inclusive prefix once its predecessors' are
// known. Tiles are taken in ticket order, so every predecessor is running.
// One warp looks back over 32 predecessors per step.
constexpr unsigned long long kTileAggregate = 1ULL << 32U;
constexpr unsigned long long kTileInclusive = 2ULL << 32U;

// Called by a whole warp; returns the exclusive prefix on every lane.
__device__ inline uint32_t tile_exclusive_prefix(unsigned long long* state,
                                                 uint32_t tile,
                                                 uint32_t aggregate,
                                                 uint32_t lane) {
    constexpr unsigned int kFullMask = 0xFFFFFFFF;
    if (tile == 0) {
        if (lane == 0) {
            atomicExch(state, kTileInclusive | aggregate);
        }
        return 0;
    }
    if (lane == 0) {
        atomicExch(state + tile, kTileAggregate | aggregate);
    }
    uint32_t exclusive = 0;
    // Window of predecessors [base - 31, base], lane l reads base - l.
    auto base = static_cast<int64_t>(tile) - 1;
    while (true) {
        const int64_t j      = base - static_cast<int64_t>(lane);
        unsigned long long v = kTileInclusive; // before tile 0: prefix 0
        if (j >= 0) {
            do {
                v = *reinterpret_cast<volatile unsigned long long*>(state + j);
            } while ((v >> 32U) == 0);
        }
        // The nearest inclusive predecessor closes the window.
        const unsigned int inc = __ballot_sync(kFullMask, v >= kTileInclusive);
        const int stop = inc == 0 ? 32 : __ffs(static_cast<int>(inc)) - 1;
        uint32_t part  = (static_cast<int>(lane) <= stop && j >= 0)
                             ? static_cast<uint32_t>(v)
                             : 0U;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_xor_sync(kFullMask, part, off);
        }
        exclusive += part;
        if (inc != 0) {
            break;
        }
        base -= 32;
    }
    if (lane == 0) {
        atomicExch(state + tile, kTileInclusive | (exclusive + aggregate));
    }
    return exclusive;
}

/// Shared-memory state of a block of at most kMaxPB leaves.
template <uint32_t kMaxPB> struct EPBlock {
    uint32_t tree[kMaxPB]; // physical parent row
    uint32_t ffa[kMaxPB];  // FFA row
    float shift[kMaxPB];   // phase shift (bins)
    uint8_t valid[kMaxPB]; // validation mask
    uint8_t flag[kMaxPB];  // passes the threshold
    uint32_t rank[kMaxPB]; // rank among the block's survivors
    float best[kMaxPB];    // max S/N
    uint32_t pass;         // survivors in the block
    uint32_t key_max;      // ep_score_key of the max score
    uint32_t key_min_inv;  // ~ep_score_key of the min score
    uint32_t tile;         // the block's tile (ticket order when direct)
    uint32_t base;         // survivors in the tiles before it
};

/// Takes the block's tile and loads its first leaf p0 and nprof leaves.
template <uint32_t kMaxPB, class FoldT>
__device__ __forceinline__ void ep_block_begin(EPBlock<kMaxPB>& b,
                                               const EPScoreArgs<FoldT>& a,
                                               uint32_t kpb,
                                               uint32_t tid,
                                               uint32_t nthreads,
                                               uint32_t& p0,
                                               uint32_t& nprof) {
    if (tid == 0) {
        b.tile = a.tree_folds != nullptr ? static_cast<uint32_t>(atomicAdd(
                                               a.tile_state + gridDim.x, 1ULL))
                                         : blockIdx.x;
        b.pass = 0;
        b.key_max     = 0;
        b.key_min_inv = 0;
    }
    __syncthreads();
    p0    = b.tile * kpb;
    nprof = min(kpb, a.n_leaves - p0);
    for (uint32_t i = tid; i < nprof; i += nthreads) {
        const uint32_t leaf = p0 + i;
        b.flag[i]           = 0;
        if (a.indices_tree == nullptr) { // rows mode
            b.tree[i]  = leaf;
            b.ffa[i]   = 0;
            b.shift[i] = 0.0F;
            b.valid[i] = 1;
            continue;
        }
        uint32_t tree_idx = a.indices_tree[leaf] + a.physical_start_idx;
        if (tree_idx >= a.capacity) {
            tree_idx -= a.capacity;
        }
        b.tree[i]  = tree_idx;
        b.ffa[i]   = a.indices_ffa[leaf];
        b.shift[i] = a.phase_shift[leaf];
        b.valid[i] = a.validation_mask[leaf];
    }
}

/// Records the max S/N @p best of valid leaf q (one thread).
template <uint32_t kMaxPB, class FoldT>
__device__ __forceinline__ void ep_block_record(EPBlock<kMaxPB>& b,
                                                const EPScoreArgs<FoldT>& a,
                                                uint32_t q,
                                                uint32_t p0,
                                                float best,
                                                uint32_t& npass) {
    const bool pass  = best >= a.threshold;
    a.scores[p0 + q] = best;
    if (a.filtered_mask != nullptr) {
        a.filtered_mask[p0 + q] = static_cast<uint8_t>(pass);
    }
    b.flag[q] = static_cast<uint8_t>(pass);
    b.best[q] = best;
    npass += pass ? 1U : 0U;
    const uint32_t key = ep_score_key(__float_as_uint(best));
    atomicMax(&b.key_max, key);
    atomicMax(&b.key_min_inv, ~key);
}

/**
 * After every leaf is recorded: adds the block's statistics. In direct mode
 * it then ranks the survivors (decoupled look-back) and writes their leaves
 * and scores into the tree; returns true, and the caller writes their folds
 * at ep_block_slot(). Called by every thread.
 */
template <uint32_t kMaxPB, class FoldT>
__device__ __forceinline__ bool ep_block_finish(EPBlock<kMaxPB>& b,
                                                const EPScoreArgs<FoldT>& a,
                                                uint32_t npass,
                                                uint32_t p0,
                                                uint32_t nprof,
                                                uint32_t tid,
                                                uint32_t nthreads) {
    if (npass != 0) {
        atomicAdd(&b.pass, npass);
    }
    __syncthreads();
    if (tid == 0 && b.key_max != 0 && a.stats != nullptr) {
        if (b.pass != 0) {
            atomicAdd(a.stats, b.pass);
        }
        atomicMax(a.stats + 1, b.key_max);
        atomicMax(a.stats + 2, b.key_min_inv);
    }
    if (a.tree_folds == nullptr) {
        return false;
    }
    if (tid < kWarpSize) {
        uint32_t rank = 0;
        if (tid == 0) {
            for (uint32_t i = 0; i < nprof; ++i) {
                b.rank[i] = rank;
                rank += b.flag[i];
            }
        }
        rank = __shfl_sync(0xFFFFFFFF, rank, 0);
        const uint32_t base =
            tile_exclusive_prefix(a.tile_state, b.tile, rank, tid);
        if (tid == 0) {
            b.base = base;
        }
    }
    __syncthreads();
    for (uint32_t i = tid; i < nprof; i += nthreads) {
        if (b.flag[i] != 0) {
            uint32_t slot = a.tree_write_head + b.base + b.rank[i];
            if (slot >= a.capacity) {
                slot -= a.capacity;
            }
            const double* __restrict__ src =
                a.leaves_batch +
                (static_cast<size_t>(p0 + i) * a.leaves_stride);
            double* __restrict__ dst =
                a.tree_leaves + (static_cast<size_t>(slot) * a.leaves_stride);
            for (uint32_t j = 0; j < a.leaves_stride; ++j) {
                dst[j] = src[j];
            }
            a.tree_scores[slot] = b.best[i];
        }
    }
    return true;
}

/// Tree slot of survivor q after ep_block_finish.
template <uint32_t kMaxPB, class FoldT>
__device__ __forceinline__ uint32_t ep_block_slot(const EPBlock<kMaxPB>& b,
                                                  const EPScoreArgs<FoldT>& a,
                                                  uint32_t q) {
    uint32_t slot = a.tree_write_head + b.base + b.rank[q];
    if (slot >= a.capacity) {
        slot -= a.capacity;
    }
    return slot;
}

} // namespace loki::core::detail
