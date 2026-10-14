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
#include "lib/cuda/ep_fourier_device.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

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

    // Calculate offsets (in circular tree)
    uint32_t tree_idx = indices_tree[ileaf] + physical_start_idx;
    if (tree_idx >= capacity) {
        tree_idx -= capacity;
    }
    const uint32_t ffa_idx = indices_ffa[ileaf];

    const SizeType tree_base = static_cast<SizeType>(tree_idx) * stride;
    const SizeType ffa_base  = static_cast<SizeType>(ffa_idx) * stride;
    const SizeType out_base  = static_cast<SizeType>(ileaf) * stride;

    // Tree (unshifted) + FFA rotated by exp(-2 pi i k shift / nbins)
    detail::ep_shift_add_harmonic(
        folds_tree[tree_base + k], folds_tree[tree_base + k + nbins_f],
        folds_ffa[ffa_base + k], folds_ffa[ffa_base + k + nbins_f], k,
        phase_shift[ileaf], nbins, folds_out[out_base + k],
        folds_out[out_base + k + nbins_f]);
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

        // exp(-2πi * k * phase_shift / nbins); phase_shift is the fractional
        // bin shift (not an index), matching CPU semantics
        float cosv;
        float sinv;
        detail::ep_phasor(k, phase_shift[leaf_seg_idx], nbins, cosv, sinv);

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
// Fourier brute fold: one block per (segment, frequency), one thread per
// harmonic (looping over groups when nbins_f > blockDim). The segment
// streams through shared memory in tiles, so any segment length works. The
// phasor of sample k, exp(2 pi i f m (t_ref - k tsamp)), comes from the
// double phase reduced to a fraction of a cycle and sincospif, re-seeded
// every kFoldReseed samples; in between it advances by an exact step
// phasor. Sums are two-level (per run of kFoldReseed samples, then total).
// =============================================================================
constexpr uint32_t kFoldTile   = 2048;
constexpr uint32_t kFoldReseed = 64;

__device__ __forceinline__ float2 cycles_phasor(double cycles) {
    const double frac = cycles - rint(cycles);
    float2 p;
    sincospif(2.0F * static_cast<float>(frac), &p.y, &p.x);
    return p;
}

__global__ void kernel_fold_complex_tiled(const float* __restrict__ ts_e,
                                          const float* __restrict__ ts_v,
                                          ComplexTypeCUDA* __restrict__ fold,
                                          const double* __restrict__ freqs,
                                          uint32_t nfreqs,
                                          SizeType npairs,
                                          SizeType segment_len,
                                          uint32_t nbins_f,
                                          double tsamp,
                                          double t_ref) {
    // Tiles of tile_len <= kFoldTile samples (dynamic: short segments keep
    // shared memory small and occupancy high).
    extern __shared__ float sh[];
    const auto tile_len =
        static_cast<uint32_t>(min(SizeType{kFoldTile}, segment_len));
    float* sh_e = sh;
    float* sh_v = sh + tile_len;
    for (SizeType pair = blockIdx.x; pair < npairs; pair += gridDim.x) {
        const SizeType iseg  = pair / nfreqs;
        const SizeType ifreq = pair % nfreqs;
        const float* e_seg   = ts_e + (iseg * segment_len);
        const float* v_seg   = ts_v + (iseg * segment_len);
        ComplexTypeCUDA* out = fold + (pair * 2 * nbins_f);
        const double f       = freqs[ifreq];
        for (uint32_t m0 = 0; m0 < nbins_f; m0 += blockDim.x) {
            const uint32_t m  = m0 + threadIdx.x;
            const bool active = m < nbins_f;
            const double fm   = f * static_cast<double>(m);
            const float2 step = cycles_phasor(-fm * tsamp);
            // Cycles at sample k: fm t_ref - k (fm tsamp), one FMA per run.
            const double c_ref  = fm * t_ref;
            const double c_step = -fm * tsamp;
            float2 acc_e{0.0F, 0.0F};
            float2 acc_v{0.0F, 0.0F};
            for (SizeType t0 = 0; t0 < segment_len; t0 += tile_len) {
                const auto len = static_cast<uint32_t>(
                    min(SizeType{tile_len}, segment_len - t0));
                __syncthreads(); // the previous tile is consumed
                for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
                    sh_e[i] = e_seg[t0 + i];
                    sh_v[i] = v_seg[t0 + i];
                }
                __syncthreads();
                if (!active) {
                    continue;
                }
                // Two-level sums: each run of kFoldReseed samples is summed
                // on its own, then added to the total (error growth
                // ~ (kFoldReseed + n / kFoldReseed) eps instead of n eps).
                for (uint32_t k0 = 0; k0 < len; k0 += kFoldReseed) {
                    const uint32_t k_end = min(k0 + kFoldReseed, len);
                    float2 pe{0.0F, 0.0F};
                    float2 pv{0.0F, 0.0F};
                    if (m == 0) {
                        for (uint32_t k = k0; k < k_end; ++k) {
                            pe.x += sh_e[k];
                            pv.x += sh_v[k];
                        }
                    } else {
                        float2 ph = cycles_phasor(
                            fma(static_cast<double>(t0 + k0), c_step, c_ref));
                        for (uint32_t k = k0; k < k_end; ++k) {
                            const float xe = sh_e[k];
                            const float xv = sh_v[k];
                            pe.x           = fmaf(xe, ph.x, pe.x);
                            pe.y           = fmaf(xe, ph.y, pe.y);
                            pv.x           = fmaf(xv, ph.x, pv.x);
                            pv.y           = fmaf(xv, ph.y, pv.y);
                            ph = {fmaf(ph.x, step.x, -ph.y * step.y),
                                  fmaf(ph.x, step.y, ph.y * step.x)};
                        }
                    }
                    acc_e = {acc_e.x + pe.x, acc_e.y + pe.y};
                    acc_v = {acc_v.x + pv.x, acc_v.y + pv.y};
                }
            }
            if (active) {
                out[m]           = {acc_e.x, acc_e.y};
                out[nbins_f + m] = {acc_v.x, acc_v.y};
            }
        }
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
            if (prof < static_cast<uint32_t>(kCap)) {
                const size_t src =
                    profile_offset(seg0 + prof, ncoords_in, ancestor, 128U);
                cur4[(static_cast<size_t>(prof) << 5U) + vec] =
                    in4[(src >> 2U) + vec];
            }
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
    const SizeType npairs = nsegments * nfreqs;
    if (npairs == 0) {
        return;
    }
    // A warp-multiple of threads, one harmonic each, at most 256 per group.
    const SizeType threads =
        std::clamp<SizeType>(((nbins_f + 31) / 32) * 32, 32, 256);
    const SizeType blocks = std::min<SizeType>(npairs, SizeType{1} << 30U);
    const dim3 block_dim(static_cast<uint32_t>(threads));
    const dim3 grid_dim(static_cast<uint32_t>(blocks));
    const SizeType smem =
        2 * std::min<SizeType>(segment_len, kFoldTile) * sizeof(float);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim, smem);
    kernel_fold_complex_tiled<<<grid_dim, block_dim, smem, stream>>>(
        ts_e, ts_v, fold, freqs, static_cast<uint32_t>(nfreqs), npairs,
        segment_len, static_cast<uint32_t>(nbins_f), tsamp, t_ref);
    cuda_utils::check_last_cuda_error("kernel_fold_complex_tiled failed");
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
    const SizeType total_smem = smem + ffa_freq_fuse_static_smem<K>();
    error_check::check_less_equal(
        total_smem, cuda_utils::get_max_shared_memory_optin(),
        "launch_ffa_freq_fuse: tile exceeds the shared memory per block");
    if (total_smem > cuda_utils::get_max_shared_memory()) {
        cuda_utils::check_cuda_call(
            cudaFuncSetAttribute(kernel_ffa_freq_fuse<K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)),
            "cudaFuncSetAttribute kernel_ffa_freq_fuse failed");
    }
    cuda_utils::check_kernel_launch_params(dim3(static_cast<unsigned>(nblocks)),
                                           dim3(256));
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
    } else if (k == 2) {
        static_bytes = ffa_freq_fuse_static_smem<2>();
    } else {
        return false;
    }
    // Large tiles opt in to the device's per-block maximum (the launcher
    // raises the kernel's dynamic shared-memory limit).
    return ffa_freq_fuse_dynamic_smem(k, nbins) + static_bytes <=
           cuda_utils::get_max_shared_memory_optin();
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
    } else if (k == 2) {
        launch_ffa_freq_fuse<2>(fold_in, fold_out, level_idx, level_shift,
                                level_ncoords, ncoords_in, ncoords_out,
                                nsegments_in, nbins, stream);
    } else {
        throw std::invalid_argument(
            "ffa_iter_freq_fused_cuda: k must be 2, 3, 4, or 5");
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
        if (k < 2 || static_cast<SizeType>(k) >= k_fits_smem.size() ||
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
            } else if (fits(2, level)) {
                k = 2;
            } else {
                k = 1;
            }
        }
        groups.push_back(k);
        level += static_cast<SizeType>(k);
    }
    return groups;
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
