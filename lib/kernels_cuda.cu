#include "loki/kernels_cuda.cuh"

#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::kernels {

namespace {

__global__ void
kernel_shift_add_linear(const float* __restrict__ folds_tree,
                        const uint32_t* __restrict__ indices_tree,
                        const float* __restrict__ folds_ffa,
                        const uint32_t* __restrict__ indices_ffa,
                        const float* __restrict__ phase_shift,
                        float* __restrict__ folds_out,
                        uint32_t nbins,
                        uint32_t n_leaves) {

    constexpr uint32_t kWarpSize = 32;
    // 1D thread mapping with optimal work distribution
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = n_leaves * nbins;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, ibin)
    const uint32_t ibin  = tid % nbins;
    const uint32_t ileaf = tid / nbins;

    uint32_t shift = 0;
    // We can only use warp-level broadcast if each warp maps to exactly one
    // leaf. This is true iff nbins is a multiple of warpSize.
    if (nbins >= kWarpSize && (nbins & (kWarpSize - 1)) == 0) {
        // warp maps to a single leaf → safe to broadcast
        const uint32_t lane = threadIdx.x & (kWarpSize - 1);
        if (lane == 0) {
            shift = __float2uint_rn(phase_shift[ileaf]);
            if (shift >= nbins) {
                shift = 0;
            }
        }
        const uint32_t mask = __activemask();
        shift               = __shfl_sync(mask, shift, 0);
    } else {
        // warp spans multiple leaves, must compute per-thread
        shift = __float2uint_rn(phase_shift[ileaf]);
        if (shift >= nbins) {
            shift = 0;
        }
    }
    const uint32_t idx_add =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    // Calculate offsets
    const uint32_t total_size  = 2 * nbins;
    const uint32_t tree_offset = indices_tree[ileaf] * total_size;
    const uint32_t ffa_offset  = indices_ffa[ileaf] * total_size;
    const uint32_t out_offset  = ileaf * total_size;

    // Process both e and v components
    folds_out[out_offset + ibin] =
        folds_tree[tree_offset + ibin] + folds_ffa[ffa_offset + idx_add];
    folds_out[out_offset + ibin + nbins] =
        folds_tree[tree_offset + ibin + nbins] +
        folds_ffa[ffa_offset + idx_add + nbins];
}

__global__ void
kernel_shift_add_linear_complex(const ComplexTypeCUDA* __restrict__ folds_tree,
                                const uint32_t* __restrict__ indices_tree,
                                const ComplexTypeCUDA* __restrict__ folds_ffa,
                                const uint32_t* __restrict__ indices_ffa,
                                const float* __restrict__ phase_shift,
                                ComplexTypeCUDA* __restrict__ folds_out,
                                uint32_t nbins_f,
                                uint32_t nbins,
                                uint32_t n_leaves) {

    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = n_leaves * nbins_f;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, k)
    const uint32_t k     = tid % nbins_f;
    const uint32_t ileaf = tid / nbins_f;

    // Phase factor for head only: exp(-2πi * k * shift / nbins)
    const auto phase_factor =
        static_cast<float>(-2.0F * kPI * k * phase_shift[ileaf] / nbins);
    float cos_val, sin_val;
    __sincosf(phase_factor, &sin_val, &cos_val);

    // Calculate offsets
    const uint32_t total_size  = 2 * nbins_f;
    const uint32_t tree_offset = indices_tree[ileaf] * total_size;
    const uint32_t ffa_offset  = indices_ffa[ileaf] * total_size;
    const uint32_t out_offset  = ileaf * total_size;

    // Load complex values for both e and v components
    const ComplexTypeCUDA* __restrict__ ffa_e = folds_ffa + ffa_offset + k;
    const ComplexTypeCUDA* __restrict__ ffa_v =
        folds_ffa + ffa_offset + nbins_f + k;
    const ComplexTypeCUDA* __restrict__ tree_e = folds_tree + tree_offset + k;
    const ComplexTypeCUDA* __restrict__ tree_v =
        folds_tree + tree_offset + nbins_f + k;

    // OPTIMIZED complex multiplication using fmaf
    // ffa_shifted_e = ffa_e * exp(-2πi * k * shift / nbins)
    const float real_ffa_e =
        fmaf(ffa_e->real(), cos_val, -ffa_e->imag() * sin_val);
    const float imag_ffa_e =
        fmaf(ffa_e->real(), sin_val, ffa_e->imag() * cos_val);
    const float real_ffa_v =
        fmaf(ffa_v->real(), cos_val, -ffa_v->imag() * sin_val);
    const float imag_ffa_v =
        fmaf(ffa_v->real(), sin_val, ffa_v->imag() * cos_val);

    // Add tree (unshifted) + ffa (shifted)
    folds_out[out_offset + k] = ComplexTypeCUDA(tree_e->real() + real_ffa_e,
                                                tree_e->imag() + imag_ffa_e);
    folds_out[out_offset + k + nbins_f] = ComplexTypeCUDA(
        tree_v->real() + real_ffa_v, tree_v->imag() + imag_ffa_v);
}

// One thread per smallest work unit
__global__ void kernel_ffa_iter(const float* __restrict__ fold_in,
                                float* __restrict__ fold_out,
                                const plans::FFACoordDPtrs coords,
                                uint32_t ncoords_cur,
                                uint32_t ncoords_prev,
                                uint32_t nsegments,
                                uint32_t nbins) {

    constexpr uint32_t kWarpSize = 32;
    // 1D thread mapping with optimal work distribution
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = ncoords_cur * nsegments * nbins;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (icoord, iseg, ibin)
    // ibin - fastest varying (best for coalescing)
    const uint32_t ibin   = tid % nbins;
    const uint32_t temp   = tid / nbins;
    const uint32_t iseg   = temp % nsegments;
    const uint32_t icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_tail = coords.i_tail[icoord];
    const uint32_t coord_head = coords.i_head[icoord];

    uint32_t shift_tail = 0;
    uint32_t shift_head = 0;
    // We can only use warp-level broadcast if each warp maps to exactly one
    // leaf. This is true iff nbins is a multiple of warpSize.
    if (nbins >= kWarpSize && (nbins & (kWarpSize - 1)) == 0) {
        // warp maps to a single leaf → safe to broadcast
        const uint32_t lane = threadIdx.x & (kWarpSize - 1);
        if (lane == 0) {
            shift_tail = __float2uint_rn(coords.shift_tail[icoord]);
            shift_head = __float2uint_rn(coords.shift_head[icoord]);
            if (shift_tail >= nbins) {
                shift_tail = 0;
            }
            if (shift_head >= nbins) {
                shift_head = 0;
            }
        }
        const uint32_t mask = __activemask();
        shift_tail          = __shfl_sync(mask, shift_tail, 0);
        shift_head          = __shfl_sync(mask, shift_head, 0);
    } else {
        // warp spans multiple leaves, must compute per-thread
        shift_tail = __float2uint_rn(coords.shift_tail[icoord]);
        shift_head = __float2uint_rn(coords.shift_head[icoord]);
        if (shift_tail >= nbins) {
            shift_tail = 0;
        }
        if (shift_head >= nbins) {
            shift_head = 0;
        }
    }

    const uint32_t total_size = 2 * nbins;
    const uint32_t idx_tail =
        (ibin < shift_tail) ? (ibin + nbins - shift_tail) : (ibin - shift_tail);
    const uint32_t idx_head =
        (ibin < shift_head) ? (ibin + nbins - shift_head) : (ibin - shift_head);

    // Calculate offsets
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * total_size) + (coord_tail * total_size);
    const uint32_t head_offset = ((iseg * 2 + 1) * ncoords_prev * total_size) +
                                 (coord_head * total_size);
    const uint32_t out_offset =
        (iseg * ncoords_cur * total_size) + (icoord * total_size);

    // Process both e and v components (vectorized access)
    fold_out[out_offset + ibin] =
        fold_in[tail_offset + idx_tail] + fold_in[head_offset + idx_head];
    fold_out[out_offset + ibin + nbins] =
        fold_in[tail_offset + idx_tail + nbins] +
        fold_in[head_offset + idx_head + nbins];
}

__global__ void kernel_ffa_freq_iter(const float* __restrict__ fold_in,
                                     float* __restrict__ fold_out,
                                     const plans::FFACoordFreqDPtrs coords,
                                     uint32_t ncoords_cur,
                                     uint32_t ncoords_prev,
                                     uint32_t nsegments,
                                     uint32_t nbins) {
    constexpr uint32_t kWarpSize = 32;
    // 1D thread mapping with optimal work distribution
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = ncoords_cur * nsegments * nbins;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (icoord, iseg, ibin)
    // ibin - fastest varying (best for coalescing)
    const uint32_t ibin   = tid % nbins;
    const uint32_t temp   = tid / nbins;
    const uint32_t iseg   = temp % nsegments;
    const uint32_t icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_idx = coords.idx[icoord];
    uint32_t shift           = 0;
    // We can only use warp-level broadcast if each warp maps to exactly one
    // leaf. This is true iff nbins is a multiple of warpSize.
    if (nbins >= kWarpSize && (nbins & (kWarpSize - 1)) == 0) {
        // warp maps to a single leaf → safe to broadcast
        const uint32_t lane = threadIdx.x & (kWarpSize - 1);
        if (lane == 0) {
            shift = __float2uint_rn(coords.shift[icoord]);
            if (shift >= nbins) {
                shift = 0;
            }
        }
        const uint32_t mask = __activemask();
        shift               = __shfl_sync(mask, shift, 0);
    } else {
        // warp spans multiple leaves, must compute per-thread
        shift = __float2uint_rn(coords.shift[icoord]);
        if (shift >= nbins) {
            shift = 0;
        }
    }

    const uint32_t idx_add =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    // Calculate offsets
    const uint32_t total_size = 2 * nbins;
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * total_size) + (coord_idx * total_size);
    const uint32_t head_offset =
        ((iseg * 2 + 1) * ncoords_prev * total_size) + (coord_idx * total_size);
    const uint32_t out_offset =
        (iseg * ncoords_cur * total_size) + (icoord * total_size);

    // Process both e and v components (vectorized access)
    fold_out[out_offset + ibin] =
        fold_in[tail_offset + ibin] + fold_in[head_offset + idx_add];
    fold_out[out_offset + ibin + nbins] =
        fold_in[tail_offset + ibin + nbins] +
        fold_in[head_offset + idx_add + nbins];
}

__global__ void
kernel_ffa_freq_iter_shared(const float* __restrict__ fold_in,
                            float* __restrict__ fold_out,
                            const plans::FFACoordFreqDPtrs coords,
                            uint32_t ncoords_cur,
                            uint32_t ncoords_prev,
                            uint32_t nsegments,
                            uint32_t nbins) {
    // Strategy: Process one (icoord, iseg) pair per block
    const uint32_t iseg = blockIdx.x;
    const uint32_t icoord =
        blockIdx.y + blockIdx.z * gridDim.y; // Combine y and z
    const uint32_t tid = threadIdx.x;

    if (icoord >= ncoords_cur || iseg >= nsegments || tid >= nbins) {
        return;
    }

    // Shared memory: [head_e, head_v]
    extern __shared__ float s_mem[];
    float* s_head_ev = s_mem;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_idx = coords.idx[icoord];
    uint32_t shift           = __float2uint_rn(coords.shift[icoord]);
    if (shift >= nbins) {
        shift = 0;
    }

    // Calculate offsets
    const uint32_t total_size = 2 * nbins;
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * total_size) + (coord_idx * total_size);
    const uint32_t head_offset =
        ((iseg * 2 + 1) * ncoords_prev * total_size) + (coord_idx * total_size);
    const uint32_t out_offset =
        (iseg * ncoords_cur * total_size) + (icoord * total_size);

    // Load data from global memory with coalesced access
    for (uint32_t i = tid; i < nbins; i += blockDim.x) {
        uint32_t rot_idx = i + shift;
        if (rot_idx >= nbins) {
            rot_idx -= nbins;
        }
        s_head_ev[rot_idx]         = fold_in[head_offset + i];
        s_head_ev[rot_idx + nbins] = fold_in[head_offset + i + nbins];
    }
    __syncthreads();

    for (uint32_t i = tid; i < nbins; i += blockDim.x) {
        fold_out[out_offset + i] = fold_in[tail_offset + i] + s_head_ev[i];
        fold_out[out_offset + i + nbins] =
            fold_in[tail_offset + i + nbins] + s_head_ev[i + nbins];
    }
}

// For nbins <= 32 (warp-level communication)
// Could be optimal (theoretically) for nbins <= 32, but we are anyway hitting a
// memory wall, so not using it
__global__ void kernel_ffa_freq_iter_warp(const float* __restrict__ fold_in,
                                          float* __restrict__ fold_out,
                                          const plans::FFACoordFreqDPtrs coords,
                                          uint32_t ncoords_cur,
                                          uint32_t ncoords_prev,
                                          uint32_t nsegments,
                                          uint32_t nbins) {
    constexpr uint32_t kWarpSize = 32;
    // Calculate which warp and lane this thread belongs to
    const uint32_t global_warp_id =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const uint32_t lane_id = threadIdx.x % kWarpSize;

    // Each warp processes one (iseg, icoord) pair
    // Decode global_warp_id to (iseg, icoord)
    const uint32_t total_coords = ncoords_cur * nsegments;
    if (global_warp_id >= total_coords) {
        return;
    }

    const uint32_t icoord = global_warp_id / nsegments;
    const uint32_t iseg   = global_warp_id % nsegments;

    // Early exit for lanes beyond nbins
    if (lane_id >= nbins) {
        return;
    }

    // Warp-level shared memory (via shuffle)
    const uint32_t coord_idx = coords.idx[icoord];
    const uint32_t shift     = __float2uint_rn(coords.shift[icoord]) % nbins;

    // Calculate memory offsets
    const uint32_t total_size = 2 * nbins;
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * total_size) + (coord_idx * total_size);
    const uint32_t head_offset =
        ((iseg * 2 + 1) * ncoords_prev * total_size) + (coord_idx * total_size);
    const uint32_t out_offset =
        (iseg * ncoords_cur * total_size) + (icoord * total_size);

    // Load head data (coalesced within warp)
    const float head_e = fold_in[head_offset + lane_id];
    const float head_v = fold_in[head_offset + lane_id + nbins];

    // Apply rotation using warp shuffle
    // Each lane needs data from position (lane_id - shift) % nbins
    const uint32_t src_lane =
        (lane_id < shift) ? (lane_id + nbins - shift) : (lane_id - shift);

    // Shuffle to get unrotated values
    const float head_e_unrot = __shfl_sync(__activemask(), head_e, src_lane);
    const float head_v_unrot = __shfl_sync(__activemask(), head_v, src_lane);

    // Load tail (coalesced), add, and write (coalesced)
    fold_out[out_offset + lane_id] =
        fold_in[tail_offset + lane_id] + head_e_unrot;
    fold_out[out_offset + lane_id + nbins] =
        fold_in[tail_offset + lane_id + nbins] + head_v_unrot;
}

// OPTIMIZED: One thread per smallest work unit, optimized for memory coalescing
__global__ void
kernel_ffa_complex_iter(const ComplexTypeCUDA* __restrict__ fold_in,
                        ComplexTypeCUDA* __restrict__ fold_out,
                        const plans::FFACoordDPtrs coords,
                        uint32_t ncoords_cur,
                        uint32_t ncoords_prev,
                        uint32_t nsegments,
                        uint32_t nbins_f,
                        uint32_t nbins) {

    // 1D thread mapping with optimal work distribution
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = ncoords_cur * nsegments * nbins_f;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (icoord, iseg, k)
    // k - frequency bin (fastest varying)
    const uint32_t k      = tid % nbins_f;
    const uint32_t temp   = tid / nbins_f;
    const uint32_t iseg   = temp % nsegments;
    const uint32_t icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_tail = coords.i_tail[icoord];
    const uint32_t coord_head = coords.i_head[icoord];
    const float shift_tail    = coords.shift_tail[icoord];
    const float shift_head    = coords.shift_head[icoord];

    // Precompute phase factors: exp(-2πi * k * shift / nbins)
    const auto phase_factor_tail =
        static_cast<float>(-2.0F * kPI * k * shift_tail / nbins);
    const auto phase_factor_head =
        static_cast<float>(-2.0F * kPI * k * shift_head / nbins);
    // Fast sincos computation
    float cos_tail, sin_tail, cos_head, sin_head;
    __sincosf(phase_factor_tail, &sin_tail, &cos_tail);
    __sincosf(phase_factor_head, &sin_head, &cos_head);

    // Calculate memory offsets for e and v components
    const uint32_t tail_offset_e =
        ((iseg * 2) * ncoords_prev * 2 * nbins_f) + (coord_tail * 2 * nbins_f);
    const uint32_t tail_offset_v = tail_offset_e + nbins_f;
    const uint32_t head_offset_e =
        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
        (coord_head * 2 * nbins_f);
    const uint32_t head_offset_v = head_offset_e + nbins_f;

    const uint32_t out_offset_e =
        (iseg * ncoords_cur * 2 * nbins_f) + (icoord * 2 * nbins_f);
    const uint32_t out_offset_v = out_offset_e + nbins_f;

    // Load complex values for both e and v components
    const ComplexTypeCUDA* __restrict__ tail_e = fold_in + tail_offset_e + k;
    const ComplexTypeCUDA* __restrict__ tail_v = fold_in + tail_offset_v + k;
    const ComplexTypeCUDA* __restrict__ head_e = fold_in + head_offset_e + k;
    const ComplexTypeCUDA* __restrict__ head_v = fold_in + head_offset_v + k;

    // OPTIMIZED complex multiplication using fmaf
    // tail_shifted_e = tail_e * exp(-2πi * k * shift_tail / nbins)
    const float real_tail_e =
        fmaf(tail_e->real(), cos_tail, -tail_e->imag() * sin_tail);
    const float imag_tail_e =
        fmaf(tail_e->real(), sin_tail, tail_e->imag() * cos_tail);
    const float real_head_e =
        fmaf(head_e->real(), cos_head, -head_e->imag() * sin_head);
    const float imag_head_e =
        fmaf(head_e->real(), sin_head, head_e->imag() * cos_head);
    const float real_tail_v =
        fmaf(tail_v->real(), cos_tail, -tail_v->imag() * sin_tail);
    const float imag_tail_v =
        fmaf(tail_v->real(), sin_tail, tail_v->imag() * cos_tail);
    const float real_head_v =
        fmaf(head_v->real(), cos_head, -head_v->imag() * sin_head);
    const float imag_head_v =
        fmaf(head_v->real(), sin_head, head_v->imag() * cos_head);
    // Complex addition and store results
    fold_out[out_offset_e + k] =
        ComplexTypeCUDA(real_tail_e + real_head_e, imag_tail_e + imag_head_e);

    fold_out[out_offset_v + k] =
        ComplexTypeCUDA(real_tail_v + real_head_v, imag_tail_v + imag_head_v);
}

__global__ void
kernel_ffa_complex_freq_iter(const ComplexTypeCUDA* __restrict__ fold_in,
                             ComplexTypeCUDA* __restrict__ fold_out,
                             const plans::FFACoordFreqDPtrs coords,
                             uint32_t ncoords_cur,
                             uint32_t ncoords_prev,
                             uint32_t nsegments,
                             uint32_t nbins_f,
                             uint32_t nbins) {
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = ncoords_cur * nsegments * nbins_f;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (icoord, iseg, k)
    const uint32_t k      = tid % nbins_f;
    const uint32_t temp   = tid / nbins_f;
    const uint32_t iseg   = temp % nsegments;
    const uint32_t icoord = temp / nsegments;

    // Freq-only: tail has no shift, head has shift
    const uint32_t coord_idx = coords.idx[icoord];
    const float shift        = coords.shift[icoord];

    // Phase factor for head only: exp(-2πi * k * shift / nbins)
    const auto phase_factor =
        static_cast<float>(-2.0F * kPI * k * shift / nbins);
    float cos_val, sin_val;
    __sincosf(phase_factor, &sin_val, &cos_val);

    // Calculate memory offsets
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * 2 * nbins_f) + (coord_idx * 2 * nbins_f);
    const uint32_t head_offset = ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
                                 (coord_idx * 2 * nbins_f);
    const uint32_t out_offset =
        (iseg * ncoords_cur * 2 * nbins_f) + (icoord * 2 * nbins_f);

    // Load values - tail is unshifted, head gets phase shift
    const ComplexTypeCUDA* __restrict__ tail_e = fold_in + tail_offset + k;
    const ComplexTypeCUDA* __restrict__ tail_v =
        fold_in + tail_offset + nbins_f + k;
    const ComplexTypeCUDA* __restrict__ head_e = fold_in + head_offset + k;
    const ComplexTypeCUDA* __restrict__ head_v =
        fold_in + head_offset + nbins_f + k;

    // Apply phase shift to head only (tail stays as-is)
    const float real_head_e =
        fmaf(head_e->real(), cos_val, -head_e->imag() * sin_val);
    const float imag_head_e =
        fmaf(head_e->real(), sin_val, head_e->imag() * cos_val);
    const float real_head_v =
        fmaf(head_v->real(), cos_val, -head_v->imag() * sin_val);
    const float imag_head_v =
        fmaf(head_v->real(), sin_val, head_v->imag() * cos_val);

    // Add tail (unshifted) + head (shifted)
    fold_out[out_offset + k] = ComplexTypeCUDA(tail_e->real() + real_head_e,
                                               tail_e->imag() + imag_head_e);
    fold_out[out_offset + k + nbins_f] = ComplexTypeCUDA(
        tail_v->real() + real_head_v, tail_v->imag() + imag_head_v);
}

// CUDA kernel for folding operation with 1D block configuration
__global__ void kernel_fold_time_1d(const float* __restrict__ ts_e,
                                    const float* __restrict__ ts_v,
                                    const uint32_t* __restrict__ phase_map,
                                    float* __restrict__ fold,
                                    uint32_t nfreqs,
                                    uint32_t nsegments,
                                    uint32_t segment_len,
                                    uint32_t nbins) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Total (segment, sample) pairs
    const uint32_t total_work = nsegments * segment_len;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (segment, sample)
    const uint32_t iseg  = tid / segment_len;
    const uint32_t isamp = tid - (iseg * segment_len);

    // Process all frequencies for this (segment, sample) pair
    for (uint32_t ifreq = 0; ifreq < nfreqs; ++ifreq) {
        const uint32_t phase_idx = (ifreq * segment_len) + isamp;
        const uint32_t phase_bin = phase_map[phase_idx];
        const uint32_t ts_idx    = (iseg * segment_len) + isamp;
        const uint32_t fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + (ifreq * 2 * nbins);

        // Atomic add (but much less contention now!)
        atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
        atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
    }
}

// CUDA kernel for folding operation with 2D block configuration
__global__ void kernel_fold_time_2d(const float* __restrict__ ts_e,
                                    const float* __restrict__ ts_v,
                                    const uint32_t* __restrict__ phase_map,
                                    float* __restrict__ fold,
                                    uint32_t nfreqs,
                                    uint32_t nsegments,
                                    uint32_t segment_len,
                                    uint32_t nbins) {
    const uint32_t isamp = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t ifreq = blockIdx.y;

    if (isamp >= segment_len || ifreq >= nfreqs) {
        return;
    }
    const uint32_t phase_idx   = (ifreq * segment_len) + isamp;
    const uint32_t phase_bin   = phase_map[phase_idx];
    const uint32_t freq_offset = ifreq * 2 * nbins;
    for (uint32_t iseg = 0; iseg < nsegments; ++iseg) {
        const uint32_t ts_idx = (iseg * segment_len) + isamp;
        const uint32_t fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + freq_offset;

        atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
        atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
    }
}

// Alternative: Use shared memory for even better performance
__global__ void kernel_fold_time_shmem(const float* __restrict__ ts_e,
                                       const float* __restrict__ ts_v,
                                       const uint32_t* __restrict__ phase_map,
                                       float* __restrict__ fold,
                                       uint32_t nfreqs,
                                       uint32_t nsegments,
                                       uint32_t segment_len,
                                       uint32_t nbins) {
    // One block per frequency, each block processes all samples for that
    // frequency
    extern __shared__ float shared_bins[];
    float* shared_e = shared_bins;
    float* shared_v = shared_bins + nbins;

    const uint32_t tid               = threadIdx.x;
    const uint32_t ifreq             = blockIdx.y;
    const uint32_t threads_per_block = blockDim.x;

    for (uint32_t iseg = 0; iseg < nsegments; ++iseg) {
        // Initialize shared memory for this segment
        for (uint32_t bin = tid; bin < nbins; bin += threads_per_block) {
            shared_e[bin] = 0.0F;
            shared_v[bin] = 0.0F;
        }
        __syncthreads();

        // Process samples for this frequency and segment
        for (uint32_t isamp = tid; isamp < segment_len;
             isamp += threads_per_block) {
            const uint32_t phase_idx = (ifreq * segment_len) + isamp;
            const uint32_t phase_bin = phase_map[phase_idx];
            const uint32_t ts_idx    = (iseg * segment_len) + isamp;

            // Accumulate in shared memory
            atomicAdd(&shared_e[phase_bin], ts_e[ts_idx]);
            atomicAdd(&shared_v[phase_bin], ts_v[ts_idx]);
        }
        __syncthreads();

        // Write shared memory results to global memory for this segment
        const uint32_t fold_base_idx =
            (iseg * nfreqs * 2 * nbins) + (ifreq * 2 * nbins);
        for (uint32_t bin = tid; bin < nbins; bin += threads_per_block) {
            fold[fold_base_idx + bin]         = shared_e[bin];
            fold[fold_base_idx + nbins + bin] = shared_v[bin];
        }
        __syncthreads();
    }
}

// =============================================================================
// Optimized kernel for Complex BruteFold with small number of harmonics
// (num_harms <= blockDim.x) Each thread handles exactly one harmonic, better
// occupancy
// =============================================================================
__global__ void
kernel_fold_complex_one_harmonic_per_thread(const float* __restrict__ ts_e,
                                            const float* __restrict__ ts_v,
                                            ComplexTypeCUDA* __restrict__ fold,
                                            const double* __restrict__ freqs,
                                            uint32_t nfreqs,
                                            uint32_t nsegments,
                                            uint32_t segment_len,
                                            uint32_t nbins_f,
                                            double tsamp,
                                            double t_ref) {
    const uint32_t iseg      = blockIdx.x;
    const uint32_t ifreq     = blockIdx.y;
    const uint32_t tid       = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    if (iseg >= nsegments || ifreq >= nfreqs || tid >= nbins_f) {
        return;
    }
    extern __shared__ float sh[];
    float* sh_e = sh;
    float* sh_v = sh + segment_len;

    // Cooperative load - all threads participate
    const uint32_t start_idx = iseg * segment_len;
    for (uint32_t i = tid; i < segment_len; i += block_dim) {
        sh_e[i] = ts_e[start_idx + i];
        sh_v[i] = ts_v[start_idx + i];
    }
    __syncthreads();

    const auto base_offset =
        (iseg * nfreqs * 2 * nbins_f) + (ifreq * 2 * nbins_f);

    // Thread 0 handles DC via reduction
    if (tid == 0) {
        float sum_e = 0.0F, sum_v = 0.0F;
        for (uint32_t k = 0; k < segment_len; ++k) {
            sum_e += sh_e[k];
            sum_v += sh_v[k];
        }
        fold[base_offset]           = {sum_e, 0.0F};
        fold[base_offset + nbins_f] = {sum_v, 0.0F};
    }

    // Threads 1..nbins_f-1 handle AC harmonics
    if (tid >= 1) {
        // Compute AC for this harmonic
        const double phase_factor =
            2.0 * kPI * freqs[ifreq] * static_cast<double>(tid);
        const double init_phase  = phase_factor * t_ref;
        const double delta_phase = -phase_factor * tsamp;
        // Fast sincos computation
        float ph_r, ph_i, step_r, step_i;
        __sincosf(static_cast<float>(init_phase), &ph_i, &ph_r);
        __sincosf(static_cast<float>(delta_phase), &step_i, &step_r);
        float acc_e_r = 0.0F, acc_e_i = 0.0F;
        float acc_v_r = 0.0F, acc_v_i = 0.0F;

        for (uint32_t k = 0; k < segment_len; ++k) {
            acc_e_r = fmaf(sh_e[k], ph_r, acc_e_r);
            acc_e_i = fmaf(sh_e[k], ph_i, acc_e_i);
            acc_v_r = fmaf(sh_v[k], ph_r, acc_v_r);
            acc_v_i = fmaf(sh_v[k], ph_i, acc_v_i);

            const float new_r = (ph_r * step_r) - (ph_i * step_i);
            const float new_i = (ph_r * step_i) + (ph_i * step_r);
            ph_r              = new_r;
            ph_i              = new_i;
        }

        fold[base_offset + tid]           = {acc_e_r, acc_e_i};
        fold[base_offset + nbins_f + tid] = {acc_v_r, acc_v_i};
    }
}

template <bool UseShared>
__global__ void kernel_fold_complex_unified(const float* __restrict__ ts_e,
                                            const float* __restrict__ ts_v,
                                            ComplexTypeCUDA* __restrict__ fold,
                                            const double* __restrict__ freqs,
                                            uint32_t nfreqs,
                                            uint32_t nsegments,
                                            uint32_t segment_len,
                                            uint32_t nbins_f,
                                            double tsamp,
                                            double t_ref) {
    const uint32_t iseg      = blockIdx.x;
    const uint32_t ifreq     = blockIdx.y;
    const uint32_t tid       = threadIdx.x;
    const uint32_t block_dim = blockDim.x;
    if (iseg >= nsegments || ifreq >= nfreqs || tid >= nbins_f) {
        return;
    }

    const float* ts_e_seg;
    const float* ts_v_seg;
    const uint32_t start_idx = iseg * segment_len;
    if constexpr (UseShared) {
        extern __shared__ float sh[];
        float* sh_e = sh;
        float* sh_v = sh + segment_len;

        // Cooperative load - all threads participate
        for (uint32_t i = tid; i < segment_len; i += block_dim) {
            sh_e[i] = ts_e[start_idx + i];
            sh_v[i] = ts_v[start_idx + i];
        }
        __syncthreads();
        ts_e_seg = sh_e;
        ts_v_seg = sh_v;
    } else {
        ts_e_seg = ts_e + start_idx;
        ts_v_seg = ts_v + start_idx;
    }

    const auto base_offset =
        (iseg * nfreqs * 2 * nbins_f) + (ifreq * 2 * nbins_f);

    // DC Component: parallel reduction from global memory
    float sum_e = 0.0F, sum_v = 0.0F;
    for (uint32_t i = tid; i < segment_len; i += block_dim) {
        sum_e += ts_e_seg[i];
        sum_v += ts_v_seg[i];
    }

    // Warp reduction
    for (uint32_t off = 16; off > 0; off >>= 1U) {
        sum_e += __shfl_down_sync(0xffffffff, sum_e, off);
        sum_v += __shfl_down_sync(0xffffffff, sum_v, off);
    }

    __shared__ float warp_e[32];
    __shared__ float warp_v[32];

    const uint32_t warp_id   = tid >> 5U;
    const uint32_t lane_id   = tid & 31U;
    const uint32_t num_warps = (block_dim + 31U) >> 5U;

    if (lane_id == 0) {
        warp_e[warp_id] = sum_e;
        warp_v[warp_id] = sum_v;
    }
    __syncthreads();

    if (tid < 32) {
        float e = (tid < num_warps) ? warp_e[tid] : 0.0F;
        float v = (tid < num_warps) ? warp_v[tid] : 0.0F;

        for (uint32_t off = 16; off > 0; off >>= 1U) {
            e += __shfl_down_sync(0xffffffff, e, off);
            v += __shfl_down_sync(0xffffffff, v, off);
        }

        if (tid == 0) {
            fold[base_offset]           = {e, 0.0F};
            fold[base_offset + nbins_f] = {v, 0.0F};
        }
    }
    __syncthreads();

    // AC Components
    const double phase_factor = 2.0 * kPI * freqs[ifreq];
    for (uint32_t m = tid + 1; m < nbins_f; m += block_dim) {
        float ph_r, ph_i, step_r, step_i;
        __sincosf(static_cast<float>(phase_factor * m * t_ref), &ph_i, &ph_r);
        __sincosf(static_cast<float>(-phase_factor * m * tsamp), &step_i,
                  &step_r);
        float acc_e_r = 0.0F, acc_e_i = 0.0F;
        float acc_v_r = 0.0F, acc_v_i = 0.0F;

        for (uint32_t k = 0; k < segment_len; ++k) {
            acc_e_r = fmaf(ts_e_seg[k], ph_r, acc_e_r);
            acc_e_i = fmaf(ts_e_seg[k], ph_i, acc_e_i);
            acc_v_r = fmaf(ts_v_seg[k], ph_r, acc_v_r);
            acc_v_i = fmaf(ts_v_seg[k], ph_i, acc_v_i);

            const float new_r = (ph_r * step_r) - (ph_i * step_i);
            const float new_i = (ph_r * step_i) + (ph_i * step_r);
            ph_r              = new_r;
            ph_i              = new_i;
        }

        fold[base_offset + m]           = {acc_e_r, acc_e_i};
        fold[base_offset + nbins_f + m] = {acc_v_r, acc_v_i};
    }
}

} // namespace

void brute_fold_ts_cuda(const float* __restrict__ ts_e,
                        const float* __restrict__ ts_v,
                        float* __restrict__ fold,
                        const uint32_t* __restrict__ phase_map,
                        SizeType nsegments,
                        SizeType nfreqs,
                        SizeType segment_len,
                        SizeType nbins,
                        cudaStream_t stream) {
    // Use 1D block configuration for small nfreqs
    if (nfreqs <= 64) {
        const auto total_work               = nsegments * segment_len;
        constexpr SizeType kThreadsPerBlock = 512;
        const auto blocks_per_grid =
            (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        kernel_fold_time_1d<<<grid_dim, block_dim, 0, stream>>>(
            ts_e, ts_v, phase_map, fold, nfreqs, nsegments, segment_len, nbins);
    } else if (nbins <= 512 && nfreqs <= 65535) {
        // Use shared memory for small bin counts
        constexpr SizeType kThreadsPerBlock = 256;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(1, nfreqs);
        const auto shmem_size = 2 * nbins * sizeof(float);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);
        kernel_fold_time_shmem<<<grid_dim, block_dim, shmem_size, stream>>>(
            ts_e, ts_v, phase_map, fold, nfreqs, nsegments, segment_len, nbins);
    } else {
        // Use 2D block configuration for large nfreqs
        constexpr SizeType kThreadsPerBlock = 256;
        const auto blocks_per_grid_x =
            (segment_len + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid_x, nfreqs, 1);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        kernel_fold_time_2d<<<grid_dim, block_dim, 0, stream>>>(
            ts_e, ts_v, phase_map, fold, nfreqs, nsegments, segment_len, nbins);
    }
    cuda_utils::check_last_cuda_error("kernel_fold launch failed");
}

void brute_fold_ts_complex_cuda(const float* __restrict__ ts_e,
                                const float* __restrict__ ts_v,
                                ComplexTypeCUDA* __restrict__ fold,
                                const double* __restrict__ freqs,
                                SizeType nfreqs,
                                SizeType nsegments,
                                SizeType segment_len,
                                SizeType nbins_f,
                                double tsamp,
                                double t_ref,
                                cudaStream_t stream) {
    const auto max_shmem  = cuda_utils::get_max_shared_memory();
    const auto shmem_size = 2 * segment_len * sizeof(float);
    // Strategy selection based on problem size and hardware constraints
    if (shmem_size <= max_shmem) {
        if (nbins_f <= 1024) {
            const auto threads_per_block = nbins_f;
            const dim3 block_dim(threads_per_block);
            const dim3 grid_dim(nsegments, nfreqs);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                                   shmem_size);
            kernel_fold_complex_one_harmonic_per_thread<<<grid_dim, block_dim,
                                                          shmem_size, stream>>>(
                ts_e, ts_v, fold, freqs, nfreqs, nsegments, segment_len,
                nbins_f, tsamp, t_ref);
        } else {
            // Strided approach for larger number of harmonics
            constexpr SizeType kThreadsPerBlock = 256;
            const dim3 block_dim(kThreadsPerBlock);
            const dim3 grid_dim(nsegments, nfreqs);
            cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                                   shmem_size);
            kernel_fold_complex_unified<true>
                <<<grid_dim, block_dim, shmem_size, stream>>>(
                    ts_e, ts_v, fold, freqs, nfreqs, nsegments, segment_len,
                    nbins_f, tsamp, t_ref);
        }
    } else {
        // Fallback: segment too large for shared memory
        constexpr SizeType kThreadsPerBlock = 256;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(nsegments, nfreqs);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        kernel_fold_complex_unified<false><<<grid_dim, block_dim, 0, stream>>>(
            ts_e, ts_v, fold, freqs, nfreqs, nsegments, segment_len, nbins_f,
            tsamp, t_ref);
    }

    cuda_utils::check_last_cuda_error("execute_device_complex failed");
}

void ffa_iter_cuda(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   plans::FFACoordDPtrs coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   cudaStream_t stream) {
    const auto total_work        = ncoords_cur * nsegments * nbins;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments, nbins);
    cuda_utils::check_last_cuda_error("FFA iter kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        plans::FFACoordFreqDPtrs coords,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        SizeType nsegments,
                        SizeType nbins,
                        cudaStream_t stream) {
    const auto total_work        = ncoords_cur * nsegments * nbins;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_freq_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments, nbins);
    cuda_utils::check_last_cuda_error("FFA freq iter kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

/*
void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        plans::FFACoordFreqDPtrs coords,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        SizeType nsegments,
                        SizeType nbins,
                        cudaStream_t stream) {
    // Strategy selection based on nbins
    constexpr uint32_t kWarpSize           = 32;
    constexpr uint32_t kSharedMemThreshold = 128;
    constexpr uint32_t kMaxGridDim         = 65535;

    const SizeType shmem_bytes = 2 * nbins * sizeof(float);
    const SizeType max_shmem   = cuda_utils::get_max_shared_memory();

    // Warp-shuffle for nbins <= 32
    if (nbins <= kWarpSize) {
        constexpr uint32_t kThreadsPerBlock = 256; // 8 warps per block
        const uint32_t kWarpsPerBlock       = kThreadsPerBlock / kWarpSize;
        // Total work: one warp per (iseg, icoord) pair
        const SizeType total_warps = ncoords_cur * nsegments;
        const SizeType total_blocks =
            (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(total_blocks);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        kernel_ffa_freq_iter_warp<<<grid_dim, block_dim, 0, stream>>>(
            fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
            nbins);
        cuda_utils::check_last_cuda_error(
            "FFA freq iter (warp) kernel launch failed");
    }
    // Shared memory for nbins >= 128
    else if (nbins >= kSharedMemThreshold && shmem_bytes <= max_shmem) {
        constexpr uint32_t kThreadsPerBlock = 256;
        uint32_t grid_y, grid_z;
        if (ncoords_cur <= kMaxGridDim) {
            grid_y = ncoords_cur;
            grid_z = 1;
        } else {
            // Split across y and z dimensions
            grid_y = kMaxGridDim;
            grid_z = (ncoords_cur + kMaxGridDim - 1) / kMaxGridDim;

            if (grid_z > kMaxGridDim) {
                throw std::runtime_error(std::format(
                    "ncoords_cur={} too large: exceeds 3D grid capacity ({})",
                    ncoords_cur, kMaxGridDim * kMaxGridDim));
            }
        }
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(nsegments, grid_y, grid_z);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim,
                                               shmem_bytes);
        kernel_ffa_freq_iter_shared<<<grid_dim, block_dim, shmem_bytes,
                                      stream>>>(fold_in, fold_out, coords,
                                                ncoords_cur, ncoords_prev,
                                                nsegments, nbins);
        cuda_utils::check_last_cuda_error(
            "FFA freq iter (shared) kernel launch failed");
    } else { // Fallback: shared memory too large or nbins not enough
        const auto total_work        = ncoords_cur * nsegments * nbins;
        const auto threads_per_block = (total_work < 65536) ? 256 : 512;
        const auto blocks_per_grid =
            (total_work + threads_per_block - 1) / threads_per_block;
        const dim3 block_dim(threads_per_block);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        kernel_ffa_freq_iter<<<grid_dim, block_dim, 0, stream>>>(
            fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
            nbins);
        cuda_utils::check_last_cuda_error("FFA freq iter kernel launch failed");
    }
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}
*/

void ffa_complex_iter_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                           ComplexTypeCUDA* __restrict__ fold_out,
                           plans::FFACoordDPtrs coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           cudaStream_t stream) {
    const auto total_work        = ncoords_cur * nsegments * nbins_f;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_complex_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
        nbins_f, nbins);
    cuda_utils::check_last_cuda_error("FFA complex iter kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

void ffa_complex_iter_freq_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                ComplexTypeCUDA* __restrict__ fold_out,
                                plans::FFACoordFreqDPtrs coords,
                                SizeType ncoords_cur,
                                SizeType ncoords_prev,
                                SizeType nsegments,
                                SizeType nbins_f,
                                SizeType nbins,
                                cudaStream_t stream) {
    const auto total_work        = ncoords_cur * nsegments * nbins_f;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_complex_freq_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
        nbins_f, nbins);
    cuda_utils::check_last_cuda_error(
        "FFA complex freq iter kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

void shift_add_linear_batch_cuda(const float* __restrict__ folds_tree,
                                 const uint32_t* __restrict__ indices_tree,
                                 const float* __restrict__ folds_ffa,
                                 const uint32_t* __restrict__ indices_ffa,
                                 const float* __restrict__ phase_shift,
                                 float* __restrict__ folds_out,
                                 SizeType nbins,
                                 SizeType n_leaves,
                                 cudaStream_t stream) {
    const auto total_work        = n_leaves * nbins;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_shift_add_linear<<<grid_dim, block_dim, 0, stream>>>(
        folds_tree, indices_tree, folds_ffa, indices_ffa, phase_shift,
        folds_out, nbins, n_leaves);
    cuda_utils::check_last_cuda_error("kernel_shift_add_linear launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

void shift_add_linear_complex_batch_cuda(
    const ComplexTypeCUDA* __restrict__ folds_tree,
    const uint32_t* __restrict__ indices_tree,
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_out,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_leaves,
    cudaStream_t stream) {
    const auto total_work        = n_leaves * nbins_f;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_shift_add_linear_complex<<<grid_dim, block_dim, 0, stream>>>(
        folds_tree, indices_tree, folds_ffa, indices_ffa, phase_shift,
        folds_out, nbins_f, nbins, n_leaves);
    cuda_utils::check_last_cuda_error(
        "kernel_shift_add_linear_complex launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

} // namespace loki::kernels