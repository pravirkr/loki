#pragma once

/**
 * @file ep_fourier_device.cuh
 * @brief Fourier-domain shift-add of the EP pruning (one harmonic), shared
 * by kernel_shift_add_linear_complex (kernels_cuda.cu) and the fused
 * shift-add + score kernel (ep_score_fourier_cuda.cu), so both store the
 * same folds bit for bit. CUDA only.
 */

#include <cstdint>

#include <cuda_runtime.h>

#include "lib/cuda/cub_helpers.cuh"
#include "lib/cuda/ffa_score_device.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core::detail {

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
// Only the legacy kernels use it: rounding a phase of up to pi*nbins radians
// to float, then __sincosf of it, loses accuracy (B9). Use ep_phasor.
__device__ __forceinline__ float
fourier_phase(uint32_t k, float shift, uint32_t nbins) {
    return static_cast<float>(
        div_by_nbins(-2.0F * cub_helpers::kPI * k * shift, nbins));
}

// exp(-2 pi i k shift / nbins), accurate for any argument and any nbins
// (fourier_phasor, shared with the FFA merges).
__device__ __forceinline__ void
ep_phasor(uint32_t k, float shift, uint32_t nbins, float& cosv, float& sinv) {
    const auto nf   = static_cast<float>(nbins);
    const Phasor ph = fourier_phasor(k, shift, nf, __frcp_rn(nf));
    cosv            = ph.c;
    sinv            = ph.s;
}

// Harmonic k of a pruning leaf: the parent (unshifted) plus the FFA segment
// rotated by exp(-2 pi i k shift / nbins), for the e and v components.
__device__ __forceinline__ void
ep_shift_add_harmonic(const ComplexTypeCUDA& tree_e,
                      const ComplexTypeCUDA& tree_v,
                      const ComplexTypeCUDA& ffa_e,
                      const ComplexTypeCUDA& ffa_v,
                      uint32_t k,
                      float shift,
                      uint32_t nbins,
                      ComplexTypeCUDA& out_e,
                      ComplexTypeCUDA& out_v) {
    float cosv;
    float sinv;
    ep_phasor(k, shift, nbins, cosv, sinv);
    const float re_ffa_e = fmaf(ffa_e.real(), cosv, -ffa_e.imag() * sinv);
    const float im_ffa_e = fmaf(ffa_e.real(), sinv, ffa_e.imag() * cosv);
    const float re_ffa_v = fmaf(ffa_v.real(), cosv, -ffa_v.imag() * sinv);
    const float im_ffa_v = fmaf(ffa_v.real(), sinv, ffa_v.imag() * cosv);
    out_e = ComplexTypeCUDA(tree_e.real() + re_ffa_e, tree_e.imag() + im_ffa_e);
    out_v = ComplexTypeCUDA(tree_v.real() + re_ffa_v, tree_v.imag() + im_ffa_v);
}

} // namespace loki::core::detail
