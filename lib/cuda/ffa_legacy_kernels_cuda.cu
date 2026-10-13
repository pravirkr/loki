#include "lib/cuda/ffa_legacy_kernels_cuda.cuh"

/**
 * @file ffa_legacy_kernels_cuda.cu
 * @brief Legacy FFA kernels, kept for reference and comparison. Nothing on
 * the library's paths calls them (2026-10-13).
 *
 * | Kernel | Replaced by |
 * | --- | --- |
 * | kernel_ffa_complex_iter, kernel_ffa_complex_freq_iter |
 * kernel_ffa_complex_iter_sweep (FP32 phasors: exact fmaf split, reduction to a
 * cycle, sincospif, instead of __sincosf of large arguments), the complex fused
 * frequency groups | | kernel_ffa_freq_iter_shared, kernel_ffa_freq_iter_warp |
 * kernel_ffa_freq_iter and kernel_ffa_freq_fuse<K> | |
 * kernel_fold_complex_one_harmonic_per_thread, kernel_fold_complex_unified |
 * kernel_fold_complex_tiled (kernels_cuda.cu) |
 *
 * Known defects, not fixed here:
 * - kernel_fold_complex_unified returns for threadIdx.x >= nbins_f before
 *   its block-wide DC reduction (wrong DC, and __syncthreads / full-mask
 *   shuffles after exits); its harmonics drift with a float recurrence over
 *   the whole segment.
 * - The Fourier kernels take __sincosf of float-cast phases of hundreds of
 *   radians.
 * - 32-bit thread indices (total_work); the host checks the launch size.
 */

#include <cstdint>
#include <format>
#include <string_view>

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cub_helpers.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ep_fourier_device.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

using detail::div_by_nbins;
using detail::fourier_phase;

__device__ __forceinline__ size_t profile_offset(uint32_t segment,
                                                 uint32_t ncoords,
                                                 uint32_t icoord,
                                                 uint32_t stride) {
    return (static_cast<size_t>(segment) * ncoords + icoord) * stride;
}

void check_u32_threads(SizeType blocks,
                       SizeType threads_per_block,
                       std::string_view kernel) {
    error_check::check_less_equal(
        blocks * threads_per_block, SizeType{1} << 32U,
        std::format("{}: work exceeds the 32-bit thread index", kernel));
}

} // namespace

// External linkage: the kernels are compiled although nothing launches
// them.
namespace legacy {

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

// Both variants of the templated kernel are compiled.
template __global__ void
kernel_fold_complex_unified<true>(const float* __restrict__,
                                  const float* __restrict__,
                                  ComplexTypeCUDA* __restrict__,
                                  const double* __restrict__,
                                  uint32_t,
                                  uint32_t,
                                  uint32_t,
                                  uint32_t,
                                  double,
                                  double);
template __global__ void
kernel_fold_complex_unified<false>(const float* __restrict__,
                                   const float* __restrict__,
                                   ComplexTypeCUDA* __restrict__,
                                   const double* __restrict__,
                                   uint32_t,
                                   uint32_t,
                                   uint32_t,
                                   uint32_t,
                                   double,
                                   double);

} // namespace legacy

using legacy::kernel_ffa_complex_freq_iter;
using legacy::kernel_ffa_complex_iter;

void legacy_ffa_complex_iter_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
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

void legacy_ffa_complex_iter_freq_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
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

/*
// Former shared-memory and warp dispatch of the frequency-only float
// level (superseded by kernel_ffa_freq_iter and the fused groups).
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

} // namespace loki::core
