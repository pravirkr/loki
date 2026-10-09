#include "lib/cuda/kernels_cuda.cuh"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cub_helpers.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

// x / nbins in double. A double divide is slow on GPUs with little FP64
// throughput; for a power-of-two nbins, multiplying by the exact reciprocal
// gives the same result.
__device__ __forceinline__ double div_by_nbins(double x, uint32_t nbins) {
    if ((nbins & (nbins - 1U)) == 0U) {
        const int log2_nbins = __ffs(static_cast<int>(nbins)) - 1;
        return x * __longlong_as_double(
                       static_cast<long long>(1023 - log2_nbins) << 52);
    }
    return x / static_cast<double>(nbins);
}

// Fourier phase -2*pi*k*shift/nbins, formed in double, rounded to float.
__device__ __forceinline__ float
fourier_phase(uint32_t k, float shift, uint32_t nbins) {
    return static_cast<float>(
        div_by_nbins(-2.0F * cub_helpers::kPI * k * shift, nbins));
}

// Profile offset (segment, coord) * stride. The factors fit in 32 bits, but
// the product does not once a level is larger than 4 GiB (frequency-only
// FFA at nsamps 2^25). A 64-bit product matches the 32-bit one when the
// 32-bit one does not wrap.
__device__ __forceinline__ size_t profile_offset(uint32_t segment,
                                                 uint32_t ncoords,
                                                 uint32_t icoord,
                                                 uint32_t stride) {
    return (static_cast<size_t>(segment) * ncoords + icoord) * stride;
}

// indices_tree are logical indices, convert to physical indices
__global__ __launch_bounds__(256, 4) void kernel_shift_add_linear(
    const float* __restrict__ folds_tree,
    const uint32_t* __restrict__ indices_tree,
    const uint8_t* __restrict__ validation_mask,
    const float* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    float* __restrict__ folds_out,
    uint32_t nbins,
    uint32_t n_leaves,
    uint32_t physical_start_idx,
    uint32_t capacity) {
    // 1D thread mapping with optimal work distribution
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = n_leaves * nbins;
    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, ibin)
    const uint32_t ileaf  = tid / nbins;
    const uint32_t ibin   = tid % nbins;
    const uint32_t stride = 2 * nbins;

    if (validation_mask[ileaf] == 0) {
        return;
    }

    uint32_t shift = __float2uint_rz(phase_shift[ileaf] + 0.5F);
    if (shift == nbins) {
        shift = 0;
    }
    const uint32_t idx_add =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    uint32_t tree_idx = indices_tree[ileaf] + physical_start_idx;
    if (tree_idx >= capacity) {
        tree_idx -= capacity;
    }
    const uint32_t ffa_idx = indices_ffa[ileaf];

    const SizeType tree_base = static_cast<SizeType>(tree_idx) * stride;
    const SizeType ffa_base  = static_cast<SizeType>(ffa_idx) * stride;
    const SizeType out_base  = static_cast<SizeType>(ileaf) * stride;

    const float* __restrict__ tree_e = folds_tree + tree_base;
    const float* __restrict__ tree_v = tree_e + nbins;
    const float* __restrict__ ffa_e  = folds_ffa + ffa_base;
    const float* __restrict__ ffa_v  = ffa_e + nbins;

    // Process both e and v components
    folds_out[out_base + ibin]         = tree_e[ibin] + ffa_e[idx_add];
    folds_out[out_base + ibin + nbins] = tree_v[ibin] + ffa_v[idx_add];
}

__global__ __launch_bounds__(256, 4) void kernel_shift_add_ascend_linear(
    const float* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    float* __restrict__ folds_tree,
    uint32_t nbins,
    uint32_t n_coords_init,
    uint32_t n_leaves,
    uint32_t n_segments) {

    // Each thread handles one (ileaf, ibin) pair
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = n_leaves * nbins;
    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, ibin)
    const uint32_t ileaf  = tid / nbins;
    const uint32_t ibin   = tid % nbins;
    const uint32_t stride = 2 * nbins;

    // Base pointer for this leaf's output (e and v components)
    float* __restrict__ out_e =
        folds_tree + (static_cast<SizeType>(ileaf) * stride);
    float* __restrict__ out_v = out_e + nbins;

    // Zero-initialize: each thread zeroes its own two elements (e and v bins)
    // This replaces the memset in the CPU code, done per-thread so no race
    // condition
    out_e[ibin] = 0.0F;
    out_v[ibin] = 0.0F;

    // Accumulate over all segments (loop-carried dependency on out, but
    // each thread owns its (ileaf, ibin) exclusively — no races)
    for (uint32_t iseg = 0; iseg < n_segments; ++iseg) {
        // phase_shift and indices_ffa are laid out [iseg * n_leaves + ileaf]
        const uint32_t leaf_seg_idx = iseg * n_leaves + ileaf;
        const uint32_t ffa_idx_seg  = indices_ffa[leaf_seg_idx];
        const uint32_t segment_idx  = indices_segment[iseg];

        // Compute circular shift amount (same rounding as CPU)
        uint32_t shift = __float2uint_rz(phase_shift[leaf_seg_idx] + 0.5F);
        if (shift == nbins) {
            shift = 0;
        }

        // Circular shift: CPU does a right-rotate of data_head by `shift`.
        // After rotation, output[ibin] = data_head[(ibin - shift + nbins) %
        // nbins] i.e. we read from idx_add in the *original* (unrotated) array.
        const uint32_t idx_add =
            (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

        // folds_ffa layout: [segment_idx * n_coords_init * stride + ffa_idx_seg
        // * stride]
        const SizeType ffa_base =
            static_cast<SizeType>(segment_idx) * n_coords_init * stride +
            static_cast<SizeType>(ffa_idx_seg) * stride;

        const float* __restrict__ ffa_e = folds_ffa + ffa_base;
        const float* __restrict__ ffa_v = ffa_e + nbins;

        // Accumulate shifted ffa into output (+=, not =)
        out_e[ibin] += ffa_e[idx_add];
        out_v[ibin] += ffa_v[idx_add];
    }
}

__global__ __launch_bounds__(256, 4) void kernel_shift_add_linear_complex(
    const ComplexTypeCUDA* __restrict__ folds_tree,
    const uint32_t* __restrict__ indices_tree,
    const uint8_t* __restrict__ validation_mask,
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_out,
    uint32_t nbins_f,
    uint32_t nbins,
    uint32_t n_leaves,
    uint32_t physical_start_idx,
    uint32_t capacity) {
    // 1D thread mapping with optimal work distribution
    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = n_leaves * nbins_f;
    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, k)
    const uint32_t ileaf  = tid / nbins_f;
    const uint32_t k      = tid % nbins_f;
    const uint32_t stride = 2 * nbins_f;

    if (validation_mask[ileaf] == 0) {
        return;
    }

    // Phase factor for head only: exp(-2πi * k * shift / nbins)
    const auto phase = fourier_phase(k, phase_shift[ileaf], nbins);
    float cosv, sinv;
    __sincosf(phase, &sinv, &cosv);

    // Calculate offsets (in circular tree)
    uint32_t tree_idx = indices_tree[ileaf] + physical_start_idx;
    if (tree_idx >= capacity) {
        tree_idx -= capacity;
    }
    const uint32_t ffa_idx = indices_ffa[ileaf];

    const SizeType tree_base = static_cast<SizeType>(tree_idx) * stride;
    const SizeType ffa_base  = static_cast<SizeType>(ffa_idx) * stride;
    const SizeType out_base  = static_cast<SizeType>(ileaf) * stride;

    const ComplexTypeCUDA* __restrict__ tree_e = folds_tree + tree_base + k;
    const ComplexTypeCUDA* __restrict__ tree_v = tree_e + nbins_f;
    const ComplexTypeCUDA* __restrict__ ffa_e  = folds_ffa + ffa_base + k;
    const ComplexTypeCUDA* __restrict__ ffa_v  = ffa_e + nbins_f;

    // OPTIMIZED complex multiplication using fmaf
    // ffa_shifted_e = ffa_e * exp(-2πi * k * shift / nbins)
    const float re_ffa_e = fmaf(ffa_e->real(), cosv, -ffa_e->imag() * sinv);
    const float im_ffa_e = fmaf(ffa_e->real(), sinv, ffa_e->imag() * cosv);
    const float re_ffa_v = fmaf(ffa_v->real(), cosv, -ffa_v->imag() * sinv);
    const float im_ffa_v = fmaf(ffa_v->real(), sinv, ffa_v->imag() * cosv);

    // Add tree (unshifted) + ffa (shifted)
    folds_out[out_base + k] =
        ComplexTypeCUDA(tree_e->real() + re_ffa_e, tree_e->imag() + im_ffa_e);
    folds_out[out_base + k + nbins_f] =
        ComplexTypeCUDA(tree_v->real() + re_ffa_v, tree_v->imag() + im_ffa_v);
}

__global__
__launch_bounds__(256, 4) void kernel_shift_add_ascend_linear_complex(
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_tree,
    uint32_t nbins_f,
    uint32_t nbins,
    uint32_t n_coords_init,
    uint32_t n_leaves,
    uint32_t n_segments) {

    const uint32_t tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t total_work = n_leaves * nbins_f;
    if (tid >= total_work) {
        return;
    }

    const uint32_t ileaf  = tid / nbins_f;
    const uint32_t k      = tid % nbins_f;
    const uint32_t stride = 2 * nbins_f;

    // Pointers to this leaf's output (e and v components)
    // Each thread owns exactly these two elements — no races
    ComplexTypeCUDA* __restrict__ out_e =
        folds_tree + static_cast<SizeType>(ileaf) * stride;
    ComplexTypeCUDA* __restrict__ out_v = out_e + nbins_f;

    // Zero-initialize: replaces the CPU memset, race-free per thread
    out_e[k] = ComplexTypeCUDA(0.0F, 0.0F);
    out_v[k] = ComplexTypeCUDA(0.0F, 0.0F);

    // Accumulate over all segments
    // Each segment contributes: ffa[k] * exp(-2πi * k * phase_shift / nbins)
    for (uint32_t iseg = 0; iseg < n_segments; ++iseg) {
        const uint32_t leaf_seg_idx = iseg * n_leaves + ileaf;
        const uint32_t ffa_idx_seg  = indices_ffa[leaf_seg_idx];
        const uint32_t segment_idx  = indices_segment[iseg];

        // Phase factor: same formula as old kernel, but phase_shift is now
        // the fractional bin shift (not an index), matching CPU semantics
        // exp(-2πi * k * phase_shift / nbins)
        const float phase = fourier_phase(k, phase_shift[leaf_seg_idx], nbins);
        float cosv, sinv;
        __sincosf(phase, &sinv, &cosv);

        // folds_ffa layout: segment_idx * n_coords_init * stride + ffa_idx_seg
        // * stride
        const SizeType ffa_base =
            static_cast<SizeType>(segment_idx) * n_coords_init * stride +
            static_cast<SizeType>(ffa_idx_seg) * stride;

        const ComplexTypeCUDA* __restrict__ ffa_e = folds_ffa + ffa_base + k;
        const ComplexTypeCUDA* __restrict__ ffa_v = ffa_e + nbins_f;

        // Complex multiply-accumulate using fmaf for FMA fusion:
        // out += ffa * (cosv + i*sinv)
        // Re(ffa * phasor) = Re(ffa)*cosv - Im(ffa)*sinv
        // Im(ffa * phasor) = Re(ffa)*sinv + Im(ffa)*cosv
        const float re_e = fmaf(ffa_e->real(), cosv, -ffa_e->imag() * sinv);
        const float im_e = fmaf(ffa_e->real(), sinv, ffa_e->imag() * cosv);
        const float re_v = fmaf(ffa_v->real(), cosv, -ffa_v->imag() * sinv);
        const float im_v = fmaf(ffa_v->real(), sinv, ffa_v->imag() * cosv);

        out_e[k] =
            ComplexTypeCUDA(out_e[k].real() + re_e, out_e[k].imag() + im_e);
        out_v[k] =
            ComplexTypeCUDA(out_v[k].real() + re_v, out_v[k].imag() + im_v);
    }
}

// One thread per smallest work unit
__global__ void kernel_ffa_iter(const float* __restrict__ fold_in,
                                float* __restrict__ fold_out,
                                const coord::FFACoordDPtrs coords,
                                uint32_t ncoords_cur,
                                uint32_t ncoords_prev,
                                uint32_t nsegments,
                                uint32_t nbins) {
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

    uint32_t shift_tail = __float2uint_rz(coords.shift_tail[icoord] + 0.5F);
    uint32_t shift_head = __float2uint_rz(coords.shift_head[icoord] + 0.5F);
    if (shift_tail == nbins) {
        shift_tail = 0;
    }
    if (shift_head == nbins) {
        shift_head = 0;
    }

    const uint32_t idx_tail =
        (ibin < shift_tail) ? (ibin + nbins - shift_tail) : (ibin - shift_tail);
    const uint32_t idx_head =
        (ibin < shift_head) ? (ibin + nbins - shift_head) : (ibin - shift_head);

    // Calculate offsets
    const uint32_t total_size = 2 * nbins;
    const size_t tail_offset =
        profile_offset(iseg * 2, ncoords_prev, coord_tail, total_size);
    const size_t head_offset =
        profile_offset(iseg * 2 + 1, ncoords_prev, coord_head, total_size);
    const size_t out_offset =
        profile_offset(iseg, ncoords_cur, icoord, total_size);

    // Process both e and v components (vectorized access)
    fold_out[out_offset + ibin] =
        fold_in[tail_offset + idx_tail] + fold_in[head_offset + idx_head];
    fold_out[out_offset + ibin + nbins] =
        fold_in[tail_offset + idx_tail + nbins] +
        fold_in[head_offset + idx_head + nbins];
}

__global__ void kernel_ffa_freq_iter(const float* __restrict__ fold_in,
                                     float* __restrict__ fold_out,
                                     const coord::FFACoordFreqDPtrs coords,
                                     uint32_t ncoords_cur,
                                     uint32_t ncoords_prev,
                                     uint32_t nsegments,
                                     uint32_t nbins) {
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

    uint32_t shift = __float2uint_rz(coords.shift[icoord] + 0.5F);
    if (shift == nbins) {
        shift = 0;
    }

    const uint32_t idx_add =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    // Calculate offsets
    const uint32_t total_size = 2 * nbins;
    const size_t tail_offset =
        profile_offset(iseg * 2, ncoords_prev, coord_idx, total_size);
    const size_t head_offset =
        profile_offset(iseg * 2 + 1, ncoords_prev, coord_idx, total_size);
    const size_t out_offset =
        profile_offset(iseg, ncoords_cur, icoord, total_size);

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
                            const coord::FFACoordFreqDPtrs coords,
                            uint32_t ncoords_cur,
                            uint32_t ncoords_prev,
                            uint32_t nsegments,
                            uint32_t nbins) {
    // Strategy: Process one (icoord, iseg) pair per block
    const uint32_t iseg = blockIdx.x;
    // Combine y and z
    const uint32_t icoord = blockIdx.y + (blockIdx.z * gridDim.y);
    const uint32_t tid    = threadIdx.x;

    if (icoord >= ncoords_cur || iseg >= nsegments || tid >= nbins) {
        return;
    }

    // Shared memory: [head_e, head_v]
    extern __shared__ float s_mem[];
    float* s_head_ev = s_mem;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_idx = coords.idx[icoord];
    uint32_t shift           = __float2uint_rz(coords.shift[icoord] + 0.5F);
    if (shift == nbins) {
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
                                          const coord::FFACoordFreqDPtrs coords,
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
    uint32_t shift           = __float2uint_rz(coords.shift[icoord] + 0.5F);
    if (shift == nbins) {
        shift = 0;
    }

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
    const uint32_t mask = __activemask();
    const float head_e_unrot =
        __shfl_sync(mask, head_e, static_cast<int>(src_lane));
    const float head_v_unrot =
        __shfl_sync(mask, head_v, static_cast<int>(src_lane));

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
                        const coord::FFACoordDPtrs coords,
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
    const auto phase_factor_tail = fourier_phase(k, shift_tail, nbins);
    const auto phase_factor_head = fourier_phase(k, shift_head, nbins);
    // Fast sincos computation
    float cos_tail, sin_tail, cos_head, sin_head;
    __sincosf(phase_factor_tail, &sin_tail, &cos_tail);
    __sincosf(phase_factor_head, &sin_head, &cos_head);

    // Calculate memory offsets for e and v components
    const uint32_t stride = 2 * nbins_f;
    const size_t tail_offset_e =
        profile_offset(iseg * 2, ncoords_prev, coord_tail, stride);
    const size_t tail_offset_v = tail_offset_e + nbins_f;
    const size_t head_offset_e =
        profile_offset(iseg * 2 + 1, ncoords_prev, coord_head, stride);
    const size_t head_offset_v = head_offset_e + nbins_f;

    const size_t out_offset_e =
        profile_offset(iseg, ncoords_cur, icoord, stride);
    const size_t out_offset_v = out_offset_e + nbins_f;

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
                             const coord::FFACoordFreqDPtrs coords,
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
    const auto phase_factor = fourier_phase(k, shift, nbins);
    float cos_val, sin_val;
    __sincosf(phase_factor, &sin_val, &cos_val);

    // Calculate memory offsets
    const uint32_t stride = 2 * nbins_f;
    const size_t tail_offset =
        profile_offset(iseg * 2, ncoords_prev, coord_idx, stride);
    const size_t head_offset =
        profile_offset(iseg * 2 + 1, ncoords_prev, coord_idx, stride);
    const size_t out_offset = profile_offset(iseg, ncoords_cur, icoord, stride);

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
        const size_t phase_idx =
            (static_cast<size_t>(ifreq) * segment_len) + isamp;
        const uint32_t phase_bin = phase_map[phase_idx];
        const size_t ts_idx = (static_cast<size_t>(iseg) * segment_len) + isamp;
        const size_t fold_base_idx =
            (static_cast<size_t>(iseg) * nfreqs + ifreq) * (2U * nbins);

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
    if (isamp >= segment_len) {
        return;
    }
    // gridDim.y is capped at 65535, so frequencies stride over it.
    for (uint32_t ifreq = blockIdx.y; ifreq < nfreqs; ifreq += gridDim.y) {
        const size_t phase_idx =
            (static_cast<size_t>(ifreq) * segment_len) + isamp;
        const uint32_t phase_bin = phase_map[phase_idx];
        for (uint32_t iseg = 0; iseg < nsegments; ++iseg) {
            const size_t ts_idx =
                (static_cast<size_t>(iseg) * segment_len) + isamp;
            const size_t fold_base_idx =
                (static_cast<size_t>(iseg) * nfreqs + ifreq) * (2U * nbins);

            atomicAdd(&fold[fold_base_idx + phase_bin], ts_e[ts_idx]);
            atomicAdd(&fold[fold_base_idx + nbins + phase_bin], ts_v[ts_idx]);
        }
    }
}

// One block per (segment, frequency). Block size stays 256, so each sample
// is atomic-added by the same thread as the old serial-over-segments loop.
// Segments do not share bins.
__global__ void kernel_fold_time_shmem(const float* __restrict__ ts_e,
                                       const float* __restrict__ ts_v,
                                       const uint32_t* __restrict__ phase_map,
                                       float* __restrict__ fold,
                                       uint32_t nfreqs,
                                       uint32_t nsegments,
                                       uint32_t segment_len,
                                       uint32_t nbins) {
    extern __shared__ float shared_bins[];
    float* shared_e = shared_bins;
    float* shared_v = shared_bins + nbins;

    const uint32_t tid               = threadIdx.x;
    const uint32_t ifreq             = blockIdx.y;
    const uint32_t iseg              = blockIdx.x;
    const uint32_t threads_per_block = blockDim.x;
    if (ifreq >= nfreqs || iseg >= nsegments) {
        return;
    }

    for (uint32_t bin = tid; bin < nbins; bin += threads_per_block) {
        shared_e[bin] = 0.0F;
        shared_v[bin] = 0.0F;
    }
    __syncthreads();

    for (uint32_t isamp = tid; isamp < segment_len;
         isamp += threads_per_block) {
        const size_t phase_idx =
            (static_cast<size_t>(ifreq) * segment_len) + isamp;
        const uint32_t phase_bin = phase_map[phase_idx];
        const size_t ts_idx = (static_cast<size_t>(iseg) * segment_len) + isamp;
        atomicAdd(&shared_e[phase_bin], ts_e[ts_idx]);
        atomicAdd(&shared_v[phase_bin], ts_v[ts_idx]);
    }
    __syncthreads();

    const size_t fold_base_idx =
        (static_cast<size_t>(iseg) * nfreqs + ifreq) * (2U * nbins);
    for (uint32_t bin = tid; bin < nbins; bin += threads_per_block) {
        fold[fold_base_idx + bin]         = shared_e[bin];
        fold[fold_base_idx + nbins + bin] = shared_v[bin];
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
    const size_t start_idx = static_cast<size_t>(iseg) * segment_len;
    for (uint32_t i = tid; i < segment_len; i += block_dim) {
        sh_e[i] = ts_e[start_idx + i];
        sh_v[i] = ts_v[start_idx + i];
    }
    __syncthreads();

    const size_t base_offset =
        (static_cast<size_t>(iseg) * nfreqs + ifreq) * (2U * nbins_f);

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
            2.0 * cub_helpers::kPI * freqs[ifreq] * static_cast<double>(tid);
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
    const size_t start_idx = static_cast<size_t>(iseg) * segment_len;
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

    const size_t base_offset =
        (static_cast<size_t>(iseg) * nfreqs + ifreq) * (2U * nbins_f);

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
    const double phase_factor = 2.0 * cub_helpers::kPI * freqs[ifreq];
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

struct FreqLvl {
    const uint32_t* idx;
    const float* shift;
    uint32_t ncoords;
};

template <int K> struct FreqLvls {
    FreqLvl level[K];
};

__device__ uint32_t lower_bound_u32(const uint32_t* data,
                                    uint32_t n,
                                    uint32_t key) {
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1U);
        if (data[mid] < key) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// One block: one level-0 coordinate across 2^K segments. The host picks K
// only for levels where every parent has at most 2 children and the parent
// indices are sorted (ffa_freq_fuse_check_levels_cuda), so the descendants
// of one ancestor stay within 2^K profiles. A coordinate with no children is
// legal: its block writes nothing.
template <int K>
__global__ void kernel_ffa_freq_fuse(const float* __restrict__ fold_in,
                                     float* __restrict__ fold_out,
                                     FreqLvls<K> levels,
                                     uint32_t ncoords_in,
                                     uint32_t ncoords_out,
                                     uint32_t nbins,
                                     bool vec4) {
    constexpr int kCap      = 1 << K;
    constexpr int kThreads  = 256;
    const uint32_t stride   = 2U * nbins;
    const uint32_t tid      = threadIdx.x;
    const size_t block_id   = blockIdx.x;
    const uint32_t ancestor = static_cast<uint32_t>(block_id % ncoords_in);
    const uint32_t tile     = static_cast<uint32_t>(block_id / ncoords_in);
    const uint32_t seg0     = tile << K;
    // 128-wide profiles: 8 profiles in flight, each thread one float4, when
    // the host found both buffers 16-byte aligned. Other cases stay scalar.
    // Same values, wider loads.
    const bool use_vec4 = vec4 && stride == 128U;

    extern __shared__ __align__(16) float fuse_smem[];
    float* bufs[2] = {fuse_smem, fuse_smem + (kCap * stride)};

    float* cur = bufs[0];
    if (use_vec4) {
        const float4* in4            = reinterpret_cast<const float4*>(fold_in);
        float4* cur4                 = reinterpret_cast<float4*>(cur);
        const uint32_t prof_in_batch = tid >> 5U;
        const uint32_t vec           = tid & 31U;
#pragma unroll
        for (int base = 0; base < kCap; base += 8) {
            const uint32_t prof = static_cast<uint32_t>(base) + prof_in_batch;
            const size_t src =
                profile_offset(seg0 + prof, ncoords_in, ancestor, 128U);
            cur4[(static_cast<size_t>(prof) << 5U) + vec] =
                in4[(src >> 2U) + vec];
        }
    } else {
        for (int prof = 0; prof < kCap; ++prof) {
            const size_t src =
                profile_offset(seg0 + static_cast<uint32_t>(prof), ncoords_in,
                               ancestor, stride);
            for (uint32_t elem = tid; elem < stride; elem += kThreads) {
                cur[static_cast<size_t>(prof) * stride + elem] =
                    fold_in[src + elem];
            }
        }
    }
    __syncthreads();

    __shared__ uint32_t s_lo;
    __shared__ uint32_t s_nout;
    __shared__ uint32_t s_shift[kCap];
    __shared__ uint32_t s_local[kCap];

    uint32_t parent_lo = ancestor;
    uint32_t nparents  = 1;
    uint32_t nseg      = static_cast<uint32_t>(kCap);
    int cur_buf        = 0;

    for (int step = 0; step < K; ++step) {
        if (tid == 0) {
            const FreqLvl lvl = levels.level[step];
            const uint32_t lo =
                lower_bound_u32(lvl.idx, lvl.ncoords, parent_lo);
            const uint32_t hi =
                lower_bound_u32(lvl.idx, lvl.ncoords, parent_lo + nparents);
            s_lo   = lo;
            s_nout = hi - lo;
        }
        __syncthreads();

        const uint32_t lo   = s_lo;
        const uint32_t nout = s_nout;
        for (uint32_t child = tid; child < nout; child += kThreads) {
            const float phase = levels.level[step].shift[lo + child];
            uint32_t shift    = __float2uint_rz(phase + 0.5F);
            if (shift == nbins) {
                shift = 0;
            }
            s_shift[child] = shift;
            s_local[child] = levels.level[step].idx[lo + child] - parent_lo;
        }
        __syncthreads();

        const uint32_t nseg_out = nseg >> 1U;
        float* nxt              = bufs[1 - cur_buf];
        const float* src        = bufs[cur_buf];
        for (uint32_t seg = 0; seg < nseg_out; ++seg) {
            for (uint32_t child = 0; child < nout; ++child) {
                const uint32_t local = s_local[child];
                const uint32_t shift = s_shift[child];
                const size_t tail_at =
                    (static_cast<size_t>(seg * 2U) * nparents + local) * stride;
                const size_t head_at =
                    (static_cast<size_t>(seg * 2U + 1U) * nparents + local) *
                    stride;
                const size_t dst_at =
                    (static_cast<size_t>(seg) * nout + child) * stride;
                for (uint32_t elem = tid; elem < stride; elem += kThreads) {
                    const uint32_t bin = (elem < nbins) ? elem : elem - nbins;
                    const uint32_t rot =
                        (bin < shift) ? (bin + nbins - shift) : (bin - shift);
                    const uint32_t head_elem =
                        (elem < nbins) ? rot : rot + nbins;
                    nxt[dst_at + elem] =
                        src[tail_at + elem] + src[head_at + head_elem];
                }
            }
        }
        __syncthreads();

        parent_lo = lo;
        nparents  = nout;
        nseg      = nseg_out;
        cur_buf ^= 1;
    }

    const float* src = bufs[cur_buf];
    if (use_vec4) {
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* out4       = reinterpret_cast<float4*>(fold_out);
        const uint32_t vec = tid & 31U;
        for (uint32_t child = tid >> 5U; child < nparents; child += 8U) {
            const size_t dst =
                profile_offset(tile, ncoords_out, parent_lo + child, 128U);
            out4[(dst >> 2U) + vec] =
                src4[(static_cast<size_t>(child) << 5U) + vec];
        }
    } else {
        for (uint32_t child = 0; child < nparents; ++child) {
            const size_t dst =
                profile_offset(tile, ncoords_out, parent_lo + child, stride);
            const float* from = src + (static_cast<size_t>(child) * stride);
            for (uint32_t elem = tid; elem < stride; elem += kThreads) {
                fold_out[dst + elem] = from[elem];
            }
        }
    }
}

// Counts, per level, the coordinates that break the fusion contract: parent
// indices not sorted, a parent with more than 2 children, or a parent index
// outside the previous level. One thread per coordinate of levels >= 1.
__global__ void kernel_ffa_freq_fuse_check(const uint32_t* __restrict__ idx,
                                           const uint32_t* __restrict__ offsets,
                                           const uint32_t* __restrict__ ncoords,
                                           uint32_t n_levels,
                                           uint32_t* __restrict__ bad) {
    const size_t total = offsets[n_levels - 1] + ncoords[n_levels - 1];
    const size_t first = offsets[1];
    const size_t gid =
        first + (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (gid >= total) {
        return;
    }
    uint32_t level = 1;
    while (level + 1 < n_levels && gid >= offsets[level + 1]) {
        ++level;
    }
    const size_t local   = gid - offsets[level];
    const uint32_t value = idx[gid];
    bool ok              = value < ncoords[level - 1];
    if (local >= 1) {
        ok = ok && idx[gid - 1] <= value;
    }
    if (local >= 2) {
        ok = ok && idx[gid - 2] != value;
    }
    if (!ok) {
        atomicAdd(&bad[level], 1U);
    }
}

} // namespace

namespace {

// Kernels that decode a 32-bit thread index (tid / nbins, tid % nbins, ...)
// need every launched thread index to fit in 32 bits, or the index wraps
// and part of the output is silently left unwritten.
void check_u32_threads(SizeType blocks,
                       SizeType threads_per_block,
                       std::string_view kernel) {
    error_check::check_less_equal(
        blocks * threads_per_block, SizeType{1} << 32U,
        std::format("{}: work exceeds the 32-bit thread index", kernel));
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
    // The atomic kernels accumulate into the fold, so they need it zeroed.
    // The shared-memory kernel writes every element.
    const auto zero_fold = [&] {
        cuda_utils::check_cuda_call(
            cudaMemsetAsync(fold, 0,
                            nsegments * nfreqs * 2 * nbins * sizeof(float),
                            stream),
            "brute_fold_ts_cuda: memset fold failed");
    };
    // Use 1D block configuration for small nfreqs
    if (nfreqs <= 64) {
        const auto total_work               = nsegments * segment_len;
        constexpr SizeType kThreadsPerBlock = 512;
        const auto blocks_per_grid =
            (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
        check_u32_threads(blocks_per_grid, kThreadsPerBlock,
                          "kernel_fold_time_1d");
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        zero_fold();
        kernel_fold_time_1d<<<grid_dim, block_dim, 0, stream>>>(
            ts_e, ts_v, phase_map, fold, nfreqs, nsegments, segment_len, nbins);
    } else if (nbins <= 512 && nfreqs <= 65535) {
        // Use shared memory for small bin counts
        constexpr SizeType kThreadsPerBlock = 256;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(static_cast<unsigned>(nsegments),
                            static_cast<unsigned>(nfreqs));
        const auto shmem_size = 2 * nbins * sizeof(float);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim, shmem_size);
        kernel_fold_time_shmem<<<grid_dim, block_dim, shmem_size, stream>>>(
            ts_e, ts_v, phase_map, fold, nfreqs, nsegments, segment_len, nbins);
    } else {
        // Use 2D block configuration for large nfreqs
        constexpr SizeType kThreadsPerBlock = 256;
        const auto blocks_per_grid_x =
            (segment_len + kThreadsPerBlock - 1) / kThreadsPerBlock;
        constexpr SizeType kMaxGridY = 65535;
        const dim3 block_dim(kThreadsPerBlock);
        const dim3 grid_dim(blocks_per_grid_x, std::min(nfreqs, kMaxGridY), 1);
        cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
        zero_fold();
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
                   coord::FFACoordDPtrs coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   cudaStream_t stream) {
    const auto total_work        = ncoords_cur * nsegments * nbins;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    check_u32_threads(blocks_per_grid, threads_per_block, "kernel_ffa_iter");
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments, nbins);
    cuda_utils::check_last_cuda_error("FFA iter kernel launch failed");
}

void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        coord::FFACoordFreqDPtrs coords,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        SizeType nsegments,
                        SizeType nbins,
                        cudaStream_t stream) {
    const auto total_work        = ncoords_cur * nsegments * nbins;
    const auto threads_per_block = (total_work < 65536) ? 256 : 512;
    const auto blocks_per_grid =
        (total_work + threads_per_block - 1) / threads_per_block;
    check_u32_threads(blocks_per_grid, threads_per_block,
                      "kernel_ffa_freq_iter");
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_freq_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments, nbins);
    cuda_utils::check_last_cuda_error("FFA freq iter kernel launch failed");
}

template <int K> SizeType ffa_freq_fuse_static_smem() {
    // Static __shared__ arrays count against the same per-block limit as
    // the dynamic tile. Queried once per K (the size is per kernel, not per
    // device).
    static const SizeType bytes = [] {
        cudaFuncAttributes attr{};
        cuda_utils::check_cuda_call(
            cudaFuncGetAttributes(&attr, kernel_ffa_freq_fuse<K>),
            "cudaFuncGetAttributes kernel_ffa_freq_fuse failed");
        return static_cast<SizeType>(attr.sharedSizeBytes);
    }();
    return bytes;
}

SizeType ffa_freq_fuse_dynamic_smem(int k, SizeType nbins) {
    return 2ULL * (SizeType{1} << k) * 2ULL * nbins * sizeof(float);
}

template <int K>
void launch_ffa_freq_fuse(const float* fold_in,
                          float* fold_out,
                          const uint32_t* const* level_idx,
                          const float* const* level_shift,
                          const uint32_t* level_ncoords,
                          uint32_t ncoords_in,
                          uint32_t ncoords_out,
                          uint32_t nsegments_in,
                          uint32_t nbins,
                          cudaStream_t stream) {
    FreqLvls<K> levels{};
    for (int step = 0; step < K; ++step) {
        levels.level[step] = FreqLvl{.idx     = level_idx[step],
                                     .shift   = level_shift[step],
                                     .ncoords = level_ncoords[step]};
    }
    const uint32_t ntiles = nsegments_in >> K;
    const size_t nblocks  = static_cast<size_t>(ntiles) * ncoords_in;
    const size_t smem     = ffa_freq_fuse_dynamic_smem(K, nbins);
    const bool vec4 = (reinterpret_cast<std::uintptr_t>(fold_in) % 16U == 0) &&
                      (reinterpret_cast<std::uintptr_t>(fold_out) % 16U == 0);
    cuda_utils::check_kernel_launch_params(
        dim3(static_cast<unsigned>(nblocks)), dim3(256),
        smem + ffa_freq_fuse_static_smem<K>());
    kernel_ffa_freq_fuse<K>
        <<<static_cast<unsigned>(nblocks), 256, smem, stream>>>(
            fold_in, fold_out, levels, ncoords_in, ncoords_out, nbins, vec4);
    cuda_utils::check_last_cuda_error(
        "FFA fused freq iter kernel launch failed");
}

bool ffa_freq_fuse_fits_smem(int k, SizeType nbins) {
    SizeType static_bytes = 0;
    if (k == 5) {
        static_bytes = ffa_freq_fuse_static_smem<5>();
    } else if (k == 4) {
        static_bytes = ffa_freq_fuse_static_smem<4>();
    } else if (k == 3) {
        static_bytes = ffa_freq_fuse_static_smem<3>();
    } else {
        return false;
    }
    return ffa_freq_fuse_dynamic_smem(k, nbins) + static_bytes <=
           cuda_utils::get_max_shared_memory();
}

void ffa_iter_freq_fused_cuda(const float* __restrict__ fold_in,
                              float* __restrict__ fold_out,
                              const uint32_t* const* level_idx,
                              const float* const* level_shift,
                              const uint32_t* level_ncoords,
                              uint32_t ncoords_in,
                              uint32_t ncoords_out,
                              uint32_t nsegments_in,
                              uint32_t nbins,
                              int k,
                              cudaStream_t stream) {
    if (k == 5) {
        launch_ffa_freq_fuse<5>(fold_in, fold_out, level_idx, level_shift,
                                level_ncoords, ncoords_in, ncoords_out,
                                nsegments_in, nbins, stream);
    } else if (k == 4) {
        launch_ffa_freq_fuse<4>(fold_in, fold_out, level_idx, level_shift,
                                level_ncoords, ncoords_in, ncoords_out,
                                nsegments_in, nbins, stream);
    } else if (k == 3) {
        launch_ffa_freq_fuse<3>(fold_in, fold_out, level_idx, level_shift,
                                level_ncoords, ncoords_in, ncoords_out,
                                nsegments_in, nbins, stream);
    } else {
        throw std::invalid_argument(
            "ffa_iter_freq_fused_cuda: k must be 3, 4, or 5");
    }
}

std::vector<uint32_t>
ffa_freq_fuse_check_levels_cuda(const uint32_t* idx,
                                std::span<const uint32_t> ncoords_offsets,
                                std::span<const SizeType> ncoords,
                                uint32_t* bad_d,
                                cudaStream_t stream) {
    const auto n_levels = ncoords.size();
    error_check::check_equal(ncoords_offsets.size(), n_levels,
                             "ffa_freq_fuse_check_levels_cuda: offsets and "
                             "ncoords must have the same length");
    std::vector<uint32_t> bad(n_levels, 0U);
    if (n_levels < 2) {
        return bad;
    }
    // bad_d holds n_levels counters, then offsets and ncoords (u32 each).
    std::vector<uint32_t> meta(2 * n_levels);
    for (SizeType i = 0; i < n_levels; ++i) {
        meta[i] = ncoords_offsets[i];
        error_check::check_less_equal(
            ncoords[i], SizeType{UINT32_MAX},
            "ffa_freq_fuse_check_levels_cuda: ncoords exceeds 32 bits");
        meta[n_levels + i] = static_cast<uint32_t>(ncoords[i]);
    }
    uint32_t* offsets_d = bad_d + n_levels;
    uint32_t* ncoords_d = offsets_d + n_levels;
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(bad_d, 0, n_levels * sizeof(uint32_t), stream),
        "ffa_freq_fuse_check_levels_cuda: memset failed");
    cuda_utils::check_cuda_call(cudaMemcpyAsync(offsets_d, meta.data(),
                                                meta.size() * sizeof(uint32_t),
                                                cudaMemcpyHostToDevice, stream),
                                "ffa_freq_fuse_check_levels_cuda: H2D failed");
    const SizeType total =
        ncoords_offsets[n_levels - 1] + ncoords[n_levels - 1];
    const SizeType work = total - ncoords_offsets[1];
    if (work > 0) {
        constexpr SizeType kThreads = 256;
        const dim3 grid((work + kThreads - 1) / kThreads);
        cuda_utils::check_kernel_launch_params(grid, dim3(kThreads));
        kernel_ffa_freq_fuse_check<<<grid, kThreads, 0, stream>>>(
            idx, offsets_d, ncoords_d, static_cast<uint32_t>(n_levels), bad_d);
        cuda_utils::check_last_cuda_error(
            "kernel_ffa_freq_fuse_check launch failed");
    }
    cuda_utils::check_cuda_call(cudaMemcpyAsync(bad.data(), bad_d,
                                                n_levels * sizeof(uint32_t),
                                                cudaMemcpyDeviceToHost, stream),
                                "ffa_freq_fuse_check_levels_cuda: D2H failed");
    // `meta` and `bad` are host stack data used by the async copies.
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "ffa_freq_fuse_check_levels_cuda: sync failed");
    return bad;
}

std::vector<int> plan_ffa_freq_fuse_groups(std::span<const SizeType> ncoords,
                                           std::span<const SizeType> nsegments,
                                           std::span<const uint32_t> level_bad,
                                           std::span<const bool> k_fits_smem) {
    const SizeType levels = ncoords.size();
    error_check::check(nsegments.size() == levels && level_bad.size() == levels,
                       "plan_ffa_freq_fuse_groups: per-level inputs must "
                       "have the same length");
    const auto fits = [&](int k, SizeType level) -> bool {
        const auto ku = static_cast<SizeType>(k);
        if (k < 3 || static_cast<SizeType>(k) >= k_fits_smem.size() ||
            !k_fits_smem[ku] || ku > levels - level) {
            return false;
        }
        const SizeType nseg   = nsegments[level - 1];
        const SizeType ncoord = ncoords[level - 1];
        if ((nseg % (SizeType{1} << ku)) != 0) {
            return false;
        }
        for (SizeType step = 0; step < ku; ++step) {
            if (level_bad[level + step] != 0) {
                return false;
            }
        }
        const SizeType nblocks = (nseg >> ku) * ncoord;
        return nblocks != 0 && nblocks <= 0x7fffffffULL &&
               ncoord <= 0xffffffffULL &&
               ncoords[level + ku - 1] <= 0xffffffffULL;
    };

    std::vector<int> groups;
    SizeType level = 1;
    // A prefix of k = 4 groups, then k = 5, then k = 3, then single levels.
    bool k4_prefix = true;
    while (level < levels) {
        int k = 0;
        if (k4_prefix && fits(4, level)) {
            k = 4;
        } else {
            k4_prefix = false;
            if (fits(5, level)) {
                k = 5;
            } else if (fits(3, level)) {
                k = 3;
            } else {
                k = 1;
            }
        }
        groups.push_back(k);
        level += static_cast<SizeType>(k);
    }
    return groups;
}

/*
void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        coord::FFACoordFreqDPtrs coords,
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
                           coord::FFACoordDPtrs coords,
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
    check_u32_threads(blocks_per_grid, threads_per_block,
                      "kernel_ffa_complex_iter");
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_complex_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
        nbins_f, nbins);
    cuda_utils::check_last_cuda_error("FFA complex iter kernel launch failed");
}

void ffa_complex_iter_freq_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                ComplexTypeCUDA* __restrict__ fold_out,
                                coord::FFACoordFreqDPtrs coords,
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
    check_u32_threads(blocks_per_grid, threads_per_block,
                      "kernel_ffa_complex_freq_iter");
    const dim3 block_dim(threads_per_block);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_complex_freq_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords, ncoords_cur, ncoords_prev, nsegments,
        nbins_f, nbins);
    cuda_utils::check_last_cuda_error(
        "FFA complex freq iter kernel launch failed");
}

void shift_add_linear_batch_cuda(const float* __restrict__ folds_tree,
                                 const uint32_t* __restrict__ indices_tree,
                                 const uint8_t* __restrict__ validation_mask,
                                 const float* __restrict__ folds_ffa,
                                 const uint32_t* __restrict__ indices_ffa,
                                 const float* __restrict__ phase_shift,
                                 float* __restrict__ folds_out,
                                 SizeType nbins,
                                 SizeType n_leaves,
                                 SizeType physical_start_idx,
                                 SizeType capacity,
                                 cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;

    const SizeType total_work = n_leaves * nbins;
    const SizeType blocks_per_grid =
        (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
    check_u32_threads(blocks_per_grid, kThreadsPerBlock,
                      "kernel_shift_add_linear");
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_shift_add_linear<<<grid_dim, block_dim, 0, stream>>>(
        folds_tree, indices_tree, validation_mask, folds_ffa, indices_ffa,
        phase_shift, folds_out, nbins, n_leaves, physical_start_idx, capacity);
    cuda_utils::check_last_cuda_error("kernel_shift_add_linear launch failed");
    // No need to sync, the next kernel will do it
}

void shift_add_linear_complex_batch_cuda(
    const ComplexTypeCUDA* __restrict__ folds_tree,
    const uint32_t* __restrict__ indices_tree,
    const uint8_t* __restrict__ validation_mask,
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_out,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_leaves,
    SizeType physical_start_idx,
    SizeType capacity,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;

    const SizeType total_work = n_leaves * nbins_f;
    const SizeType blocks_per_grid =
        (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
    check_u32_threads(blocks_per_grid, kThreadsPerBlock,
                      "kernel_shift_add_linear_complex");
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_shift_add_linear_complex<<<grid_dim, block_dim, 0, stream>>>(
        folds_tree, indices_tree, validation_mask, folds_ffa, indices_ffa,
        phase_shift, folds_out, nbins_f, nbins, n_leaves, physical_start_idx,
        capacity);
    cuda_utils::check_last_cuda_error(
        "kernel_shift_add_linear_complex launch failed");
    // No need to sync, the next kernel will do it
}

void shift_add_ascend_linear_batch_cuda(
    const float* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    float* __restrict__ folds_tree,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;

    const SizeType total_work = n_leaves * nbins;
    const SizeType blocks_per_grid =
        (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
    check_u32_threads(blocks_per_grid, kThreadsPerBlock,
                      "kernel_shift_add_ascend_linear");
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_shift_add_ascend_linear<<<grid_dim, block_dim, 0, stream>>>(
        folds_ffa, indices_segment, indices_ffa, phase_shift, folds_tree, nbins,
        n_coords_init, n_leaves, n_segments);
    cuda_utils::check_last_cuda_error(
        "kernel_shift_add_ascend_linear launch failed");
    // No need to sync, the next kernel will do it
}

void shift_add_ascend_linear_complex_batch_cuda(
    const ComplexTypeCUDA* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const uint32_t* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexTypeCUDA* __restrict__ folds_tree,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;

    const SizeType total_work = n_leaves * nbins_f;
    const SizeType blocks_per_grid =
        (total_work + kThreadsPerBlock - 1) / kThreadsPerBlock;
    check_u32_threads(blocks_per_grid, kThreadsPerBlock,
                      "kernel_shift_add_ascend_linear_complex");
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_shift_add_ascend_linear_complex<<<grid_dim, block_dim, 0, stream>>>(
        folds_ffa, indices_segment, indices_ffa, phase_shift, folds_tree,
        nbins_f, nbins, n_coords_init, n_leaves, n_segments);
    cuda_utils::check_last_cuda_error(
        "kernel_shift_add_ascend_linear_complex launch failed");
    // No need to sync, the next kernel will do it
}

} // namespace loki::core
