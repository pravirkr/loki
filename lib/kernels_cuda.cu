#include "loki/kernels.hpp"

#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::kernels {

namespace {

__global__ void shift_add_kernel(const float* __restrict__ tree_folds,
                                 const SizeType* __restrict__ tree_indices,
                                 const float* __restrict__ ffa_fold_segment,
                                 const SizeType* __restrict__ param_idx,
                                 const float* __restrict__ phase_shift,
                                 float* __restrict__ out_folds,
                                 uint32_t n_leaves,
                                 uint32_t nbins) {

    constexpr uint32_t kWarpSize = 32;

    // 1D thread mapping with optimal work distribution
    const uint32_t tid    = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto total_work = n_leaves * nbins;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, ibin)
    // Fastest varying (best for coalescing)
    const auto ibin  = tid % nbins;
    const auto ileaf = tid / nbins;

    uint32_t shift = 0;
    // We can only use warp-level broadcast if each warp maps to exactly one
    // leaf. This is true iff nbins is a multiple of warpSize.
    if (nbins >= kWarpSize && (nbins & (kWarpSize - 1)) == 0) {
        // warp maps to a single leaf → safe to broadcast
        const auto lane = threadIdx.x & (kWarpSize - 1);
        if (lane == 0) {
            shift = __float2int_rn(phase_shift[ileaf]) % nbins;
        }
        const auto mask = __activemask();
        shift           = __shfl_sync(mask, shift, 0);
    } else {
        // warp spans multiple leaves, must compute per-thread
        shift = __float2int_rn(phase_shift[ileaf]) % nbins;
    }
    const auto idx_add =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    // Calculate offsets
    const auto total_size  = 2 * nbins;
    const auto tree_offset = tree_indices[ileaf] * total_size;
    const auto ffa_offset  = param_idx[ileaf] * total_size;
    const auto out_offset  = ileaf * total_size;

    // Process both e and v components
    out_folds[out_offset + ibin] =
        tree_folds[tree_offset + ibin] + ffa_fold_segment[ffa_offset + idx_add];
    out_folds[out_offset + ibin + nbins] =
        tree_folds[tree_offset + ibin + nbins] +
        ffa_fold_segment[ffa_offset + idx_add + nbins];
}

__global__ void
shift_add_complex_kernel(const ComplexTypeCUDA* __restrict__ tree_folds,
                         const SizeType* __restrict__ tree_indices,
                         const ComplexTypeCUDA* __restrict__ ffa_fold_segment,
                         const SizeType* __restrict__ param_idx,
                         const float* __restrict__ phase_shift,
                         ComplexTypeCUDA* __restrict__ out_folds,
                         uint32_t n_leaves,
                         uint32_t nbins,
                         uint32_t nbins_f) {

    // 1D thread mapping with optimal work distribution
    const uint32_t tid    = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto total_work = n_leaves * nbins_f;

    if (tid >= total_work) {
        return;
    }

    // Decode thread ID to (ileaf, k)
    const auto k     = tid % nbins_f;
    const auto ileaf = tid / nbins_f;

    // Precompute phase factors: exp(-2πi * k * shift / nbins)
    const auto phase_factor =
        static_cast<float>(-2.0F * M_PI * k * phase_shift[ileaf] / nbins);
    // Fast sincos computation
    float cos_phase, sin_phase;
    __sincosf(phase_factor, &sin_phase, &cos_phase);

    // Calculate offsets
    const auto total_size  = 2 * nbins_f;
    const auto tree_offset = tree_indices[ileaf] * total_size;
    const auto ffa_offset  = param_idx[ileaf] * total_size;
    const auto out_offset  = ileaf * total_size;

    // Load complex values for both e and v components
    const ComplexTypeCUDA ffa_e  = ffa_fold_segment[ffa_offset + k];
    const ComplexTypeCUDA ffa_v  = ffa_fold_segment[ffa_offset + k + nbins_f];
    const ComplexTypeCUDA tree_e = tree_folds[tree_offset + k];
    const ComplexTypeCUDA tree_v = tree_folds[tree_offset + k + nbins_f];

    // OPTIMIZED complex multiplication using fmaf
    // ffa_shifted_e = ffa_e * exp(-2πi * k * shift / nbins)
    const float real_ffa_e =
        fmaf(ffa_e.real(), cos_phase, -ffa_e.imag() * sin_phase);
    const float imag_ffa_e =
        fmaf(ffa_e.real(), sin_phase, ffa_e.imag() * cos_phase);
    const float real_ffa_v =
        fmaf(ffa_v.real(), cos_phase, -ffa_v.imag() * sin_phase);
    const float imag_ffa_v =
        fmaf(ffa_v.real(), sin_phase, ffa_v.imag() * cos_phase);

    // Process both e and v components
    out_folds[out_offset + k] =
        ComplexTypeCUDA(tree_e.real() + real_ffa_e, tree_e.imag() + imag_ffa_e);
    out_folds[out_offset + k + nbins_f] =
        ComplexTypeCUDA(tree_v.real() + real_ffa_v, tree_v.imag() + imag_ffa_v);
}

// OPTIMIZED: One thread per smallest work unit, optimized for memory coalescing
__global__ void kernel_ffa_iter(const float* __restrict__ fold_in,
                                float* __restrict__ fold_out,
                                const plans::FFACoordDPtrs* __restrict__ coords,
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

    // Decode thread ID to (icoord, iseg, ibin) - OPTIMIZED ORDER for coalescing
    const uint32_t ibin = tid % nbins; // Fastest varying (best for coalescing)
    const uint32_t temp = tid / nbins;
    const uint32_t iseg = temp % nsegments;
    const uint32_t icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_tail = coords->i_tail[icoord];
    const uint32_t coord_head = coords->i_head[icoord];
    const uint32_t shift_tail =
        __float2int_rn(coords->shift_tail[icoord]) % nbins;
    const uint32_t shift_head =
        __float2int_rn(coords->shift_head[icoord]) % nbins;

    const uint32_t idx_tail =
        (ibin < shift_tail) ? (ibin + nbins - shift_tail) : (ibin - shift_tail);
    const uint32_t idx_head =
        (ibin < shift_head) ? (ibin + nbins - shift_head) : (ibin - shift_head);

    // Calculate offsets
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * 2 * nbins) + (coord_tail * 2 * nbins);
    const uint32_t head_offset =
        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) + (coord_head * 2 * nbins);
    const uint32_t out_offset =
        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

    // Process both e and v components (vectorized access)
    fold_out[out_offset + ibin] =
        fold_in[tail_offset + idx_tail] + fold_in[head_offset + idx_head];
    fold_out[out_offset + ibin + nbins] =
        fold_in[tail_offset + idx_tail + nbins] +
        fold_in[head_offset + idx_head + nbins];
}

__global__ void
kernel_ffa_freq_iter(const float* __restrict__ fold_in,
                     float* __restrict__ fold_out,
                     const plans::FFACoordFreqDPtrs* __restrict__ coords,
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

    // Decode thread ID to (icoord, iseg, ibin) - OPTIMIZED ORDER for coalescing
    const uint32_t ibin = tid % nbins; // Fastest varying (best for coalescing)
    const uint32_t temp = tid / nbins;
    const uint32_t iseg = temp % nsegments;
    const uint32_t icoord = temp / nsegments;

    // Precompute coordinate data (avoid repeated access)
    const uint32_t coord_idx = coords->idx[icoord];
    const uint32_t shift     = __float2int_rn(coords->shift[icoord]) % nbins;

    const uint32_t idx =
        (ibin < shift) ? (ibin + nbins - shift) : (ibin - shift);

    // Calculate offsets
    const uint32_t tail_offset =
        ((iseg * 2) * ncoords_prev * 2 * nbins) + (coord_idx * 2 * nbins);
    const uint32_t head_offset =
        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins) + (coord_idx * 2 * nbins);
    const uint32_t out_offset =
        (iseg * ncoords_cur * 2 * nbins) + (icoord * 2 * nbins);

    // Process both e and v components (vectorized access)
    fold_out[out_offset + ibin] =
        fold_in[tail_offset + ibin] + fold_in[head_offset + idx];
    fold_out[out_offset + ibin + nbins] = fold_in[tail_offset + ibin + nbins] +
                                          fold_in[head_offset + idx + nbins];
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

    // Decode thread ID to (icoord, iseg, k) - OPTIMIZED ORDER for coalescing
    const uint32_t k      = tid % nbins_f; // Frequency bin (fastest varying)
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
    const ComplexTypeCUDA data_tail_e = fold_in[tail_offset_e + k];
    const ComplexTypeCUDA data_tail_v = fold_in[tail_offset_v + k];
    const ComplexTypeCUDA data_head_e = fold_in[head_offset_e + k];
    const ComplexTypeCUDA data_head_v = fold_in[head_offset_v + k];

    // OPTIMIZED complex multiplication using fmaf
    // tail_shifted_e = data_tail_e * exp(-2πi * k * shift_tail / nbins)
    const float real_tail_e =
        fmaf(data_tail_e.real(), cos_tail, -data_tail_e.imag() * sin_tail);
    const float imag_tail_e =
        fmaf(data_tail_e.real(), sin_tail, data_tail_e.imag() * cos_tail);
    const float real_head_e =
        fmaf(data_head_e.real(), cos_head, -data_head_e.imag() * sin_head);
    const float imag_head_e =
        fmaf(data_head_e.real(), sin_head, data_head_e.imag() * cos_head);
    const float real_tail_v =
        fmaf(data_tail_v.real(), cos_tail, -data_tail_v.imag() * sin_tail);
    const float imag_tail_v =
        fmaf(data_tail_v.real(), sin_tail, data_tail_v.imag() * cos_tail);
    const float real_head_v =
        fmaf(data_head_v.real(), cos_head, -data_head_v.imag() * sin_head);
    const float imag_head_v =
        fmaf(data_head_v.real(), sin_head, data_head_v.imag() * cos_head);
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

    // Decode thread ID to (icoord, iseg, k) - OPTIMIZED ORDER for coalescing
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
    const uint32_t tail_offset_e =
        ((iseg * 2) * ncoords_prev * 2 * nbins_f) + (coord_idx * 2 * nbins_f);
    const uint32_t tail_offset_v = tail_offset_e + nbins_f;
    const uint32_t head_offset_e =
        ((iseg * 2 + 1) * ncoords_prev * 2 * nbins_f) +
        (coord_idx * 2 * nbins_f);
    const uint32_t head_offset_v = head_offset_e + nbins_f;
    const uint32_t out_offset_e =
        (iseg * ncoords_cur * 2 * nbins_f) + (icoord * 2 * nbins_f);
    const uint32_t out_offset_v = out_offset_e + nbins_f;

    // Load values - tail is unshifted, head gets phase shift
    const ComplexTypeCUDA tail_e = fold_in[tail_offset_e + k];
    const ComplexTypeCUDA tail_v = fold_in[tail_offset_v + k];
    const ComplexTypeCUDA head_e = fold_in[head_offset_e + k];
    const ComplexTypeCUDA head_v = fold_in[head_offset_v + k];

    // Apply phase shift to head only (tail stays as-is)
    const float real_head_e =
        fmaf(head_e.real(), cos_val, -head_e.imag() * sin_val);
    const float imag_head_e =
        fmaf(head_e.real(), sin_val, head_e.imag() * cos_val);
    const float real_head_v =
        fmaf(head_v.real(), cos_val, -head_v.imag() * sin_val);
    const float imag_head_v =
        fmaf(head_v.real(), sin_val, head_v.imag() * cos_val);

    // Add tail (unshifted) + head (shifted)
    fold_out[out_offset_e + k] = ComplexTypeCUDA(tail_e.real() + real_head_e,
                                                 tail_e.imag() + imag_head_e);
    fold_out[out_offset_v + k] = ComplexTypeCUDA(tail_v.real() + real_head_v,
                                                 tail_v.imag() + imag_head_v);
}

} // namespace

void ffa_iter_cuda(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const plans::FFACoordDPtrs* __restrict__ coords_cur,
                   SizeType nsegments,
                   SizeType nbins,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   cudaStream_t stream) {
    const auto total_work = ncoords_cur * nsegments * nbins;
    const auto block_size = (total_work < 65536) ? 256 : 512;
    const auto grid_size  = (total_work + block_size - 1) / block_size;
    const dim3 block_dim(block_size);
    const dim3 grid_dim(grid_size);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords_cur, ncoords_cur, ncoords_prev, nsegments,
        nbins);
    cuda_utils::check_last_cuda_error("FFA iter kernel launch failed");
}

void ffa_iter_freq_cuda(const float* __restrict__ fold_in,
                        float* __restrict__ fold_out,
                        const plans::FFACoordFreqDPtrs* __restrict__ coords_cur,
                        SizeType nsegments,
                        SizeType nbins,
                        SizeType ncoords_cur,
                        SizeType ncoords_prev,
                        cudaStream_t stream) {
    const auto total_work = ncoords_cur * nsegments * nbins;
    const auto block_size = (total_work < 65536) ? 256 : 512;
    const auto grid_size  = (total_work + block_size - 1) / block_size;
    const dim3 block_dim(block_size);
    const dim3 grid_dim(grid_size);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_ffa_freq_iter<<<grid_dim, block_dim, 0, stream>>>(
        fold_in, fold_out, coords_cur, ncoords_cur, ncoords_prev, nsegments,
        nbins);
    cuda_utils::check_last_cuda_error("FFA freq iter kernel launch failed");
}
} // namespace loki::kernels