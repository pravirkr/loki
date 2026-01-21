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
} // namespace
} // namespace loki::kernels