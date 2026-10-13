#pragma once

/**
 * @file ffa_score_cuda.cuh
 * @brief The last FFA merge level fused with the boxcar scorer (CUDA only).
 *
 * The frequency sweep only needs the scores of the final fold, so the final
 * level is merged into shared memory and scored there; the final fold is
 * never written to global memory. The merge arithmetic matches
 * kernel_ffa_iter / kernel_ffa_freq_iter, and the scoring arithmetic matches
 * score_and_filter_cuda_d (thread kernel for nbins <= 64 with at least 2^16
 * profiles, warp kernel otherwise), so the scores are bit-identical to the
 * unfused path. Passing (profile, width) pairs are compacted with an atomic
 * slot claim, as in score_and_filter_cuda_d.
 */

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "loki/common/types.hpp"

#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/taylor_ffa_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

struct FFAFinalScoreArgs {
    const uint32_t* widths{nullptr};
    uint32_t nwidths{0};
    float threshold{0.0F};
    // Multiplies every e and v value before scoring (deferred IRFFT 1/nbins).
    float fold_scale{1.0F};
    float* scores{nullptr};
    uint32_t* indices{nullptr};
    uint32_t* counter{nullptr};
    // Slots in scores / indices. The counter still counts every passing
    // pair, so a count above capacity tells the caller to score again in
    // smaller profile ranges.
    uint32_t capacity{UINT32_MAX};
    // Final-level profiles [profile_begin, profile_end) to score (clamped
    // to the level size). Indices are always global.
    uint32_t profile_begin{0};
    uint32_t profile_end{UINT32_MAX};
    // Traversal order of the final level for an (accel, freq) grid of
    // grid_rows x grid_cols coordinates (row-major): tile_rows consecutive
    // accel rows are visited together, freq-major within a tile, so final
    // coordinates that share ancestors run together and their source rows
    // stay in L2. grid_rows = 1 (or tile_rows <= 1) is the identity. The
    // emitted indices are the true coordinates either way.
    uint32_t grid_rows{1};
    uint32_t grid_cols{0};
    uint32_t tile_rows{1};
    // Fourier scorers whose nbins is not a power of two in [32, 1024]: the
    // Bluestein table of ffa_fourier_bluestein_table(nbins) on the device.
    const ComplexTypeCUDA* bluestein{nullptr};
};

/**
 * @brief Final level(s) of a time-domain FFA fused with the scorer.
 *
 * depth 1: @p fold_in is the level below the final one (two segments,
 * ncoords_prev coordinates); @p coords_prev is unused and ncoords_src must
 * equal ncoords_prev. depth 2: @p fold_in is two levels below the final one
 * (four segments, ncoords_src coordinates); the level in between
 * (@p coords_prev, ncoords_prev coordinates) is merged on chip too.
 */
void ffa_final_score_cuda(const float* __restrict__ fold_in,
                          coord::FFACoordDPtrs coords_final,
                          coord::FFACoordDPtrs coords_prev,
                          SizeType ncoords_cur,
                          SizeType ncoords_prev,
                          SizeType ncoords_src,
                          int depth,
                          SizeType nbins,
                          const FFAFinalScoreArgs& args,
                          cudaStream_t stream);

/// Accel variant of ffa_final_score_cuda whose final-level coordinates are
/// resolved on the fly from @p final_level (they are never stored).
void ffa_final_score_cuda(const float* __restrict__ fold_in,
                          const FFAAccelResolveLevel& final_level,
                          coord::FFACoordDPtrs coords_prev,
                          SizeType ncoords_cur,
                          SizeType ncoords_prev,
                          SizeType ncoords_src,
                          int depth,
                          SizeType nbins,
                          const FFAFinalScoreArgs& args,
                          cudaStream_t stream);

/// Frequency-only variant of ffa_final_score_cuda.
void ffa_freq_final_score_cuda(const float* __restrict__ fold_in,
                               coord::FFACoordFreqDPtrs coords_final,
                               coord::FFACoordFreqDPtrs coords_prev,
                               SizeType ncoords_cur,
                               SizeType ncoords_prev,
                               SizeType ncoords_src,
                               int depth,
                               SizeType nbins,
                               const FFAFinalScoreArgs& args,
                               cudaStream_t stream);

/// Whether the fused Fourier final level (on-chip inverse FFT, portable
/// block FFT, every architecture) handles @p nbins: any nbins in [2, 1024].
[[nodiscard]] bool ffa_final_score_fourier_supported(SizeType nbins) noexcept;

/// Whether the fused Fourier scorer needs a Bluestein table for @p nbins
/// (every nbins except a power of two in [32, 1024], which has a direct
/// on-chip C2R).
[[nodiscard]] bool ffa_fourier_needs_bluestein(SizeType nbins) noexcept;

/**
 * @brief Bluestein table of the fused Fourier scorer and of the batched GPU
 * FFTs (fft_cuda.cuh) for @p nbins = N in [2, math::kGpuFFTMaxN].
 *
 * The inverse DFT of length N runs on chip as a circular convolution of
 * power-of-two length M = max(32, bit_ceil(2N - 1)). Layout: the chirp
 * c[j] = exp(i pi j^2 / N) for j < N, then the M-point forward DFT of
 * conj(c[|j|]) (wrapped circularly) divided by M. Built in double.
 */
[[nodiscard]] std::vector<ComplexTypeCUDA>
ffa_fourier_bluestein_table(SizeType nbins);

/**
 * @brief Final level(s) of a Fourier-domain FFA (complex folds), inverse
 * real FFT on chip (unnormalized; set args.fold_scale = 1 / nbins), then
 * scored. Depth and level counts as in ffa_final_score_cuda.
 */
void ffa_final_score_fourier_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                  coord::FFACoordDPtrs coords_final,
                                  coord::FFACoordDPtrs coords_prev,
                                  SizeType ncoords_cur,
                                  SizeType ncoords_prev,
                                  SizeType ncoords_src,
                                  int depth,
                                  SizeType nbins,
                                  const FFAFinalScoreArgs& args,
                                  cudaStream_t stream);

/// Accel variant of ffa_final_score_fourier_cuda with the final-level
/// coordinates resolved on the fly.
void ffa_final_score_fourier_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                  const FFAAccelResolveLevel& final_level,
                                  coord::FFACoordDPtrs coords_prev,
                                  SizeType ncoords_cur,
                                  SizeType ncoords_prev,
                                  SizeType ncoords_src,
                                  int depth,
                                  SizeType nbins,
                                  const FFAFinalScoreArgs& args,
                                  cudaStream_t stream);

/// Frequency-only variant of ffa_final_score_fourier_cuda.
void ffa_freq_final_score_fourier_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
    coord::FFACoordFreqDPtrs coords_final,
    coord::FFACoordFreqDPtrs coords_prev,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType ncoords_src,
    int depth,
    SizeType nbins,
    const FFAFinalScoreArgs& args,
    cudaStream_t stream);

/// One complex merge level for the frequency sweep: FP32 phasors computed
/// once per (coordinate, harmonic) and reused across segments. Same layout
/// as ffa_complex_iter_cuda; the phase rounding differs at float level.
void ffa_complex_iter_sweep_cuda(const ComplexTypeCUDA* __restrict__ fold_in,
                                 ComplexTypeCUDA* __restrict__ fold_out,
                                 coord::FFACoordDPtrs coords,
                                 SizeType ncoords_cur,
                                 SizeType ncoords_prev,
                                 SizeType nsegments,
                                 SizeType nbins_f,
                                 SizeType nbins,
                                 cudaStream_t stream);

/// Frequency-only variant of ffa_complex_iter_sweep_cuda.
void ffa_complex_iter_freq_sweep_cuda(
    const ComplexTypeCUDA* __restrict__ fold_in,
    ComplexTypeCUDA* __restrict__ fold_out,
    coord::FFACoordFreqDPtrs coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nsegments,
    SizeType nbins_f,
    SizeType nbins,
    cudaStream_t stream);

/// Phase-bin map of the time-domain brute fold ([freq][sample]), the same
/// values BruteFoldCudaEngine computes.
void brute_fold_phase_map_cuda(const double* freqs,
                               SizeType nfreqs,
                               SizeType segment_len,
                               SizeType nbins,
                               double tsamp,
                               double t_ref,
                               uint32_t* phase_map,
                               cudaStream_t stream);

/// Time-domain brute fold, one block per segment with all frequencies'
/// profiles in shared memory. Writes every element of the fold.
void brute_fold_segment_cuda(const float* ts_e,
                             const float* ts_v,
                             const uint32_t* phase_map,
                             float* fold,
                             SizeType nsegments,
                             SizeType nfreqs,
                             SizeType segment_len,
                             SizeType nbins,
                             cudaStream_t stream);

/// Whether a k-level complex frequency-only fused tile fits the per-block
/// (opt-in) shared memory.
[[nodiscard]] bool ffa_complex_freq_fuse_fits_smem(int k, SizeType nbins_f);

/**
 * @brief k (2 to 5) complex frequency-only merge levels in one launch,
 * intermediates in shared memory; valid where ffa_freq_fuse_check_levels_cuda
 * reports clean levels (plan with plan_ffa_freq_fuse_groups). Merges as
 * ffa_complex_iter_freq_sweep_cuda does.
 */
void ffa_complex_freq_fused_sweep_cuda(const ComplexTypeCUDA* fold_in,
                                       ComplexTypeCUDA* fold_out,
                                       const uint32_t* const* level_idx,
                                       const float* const* level_shift,
                                       const uint32_t* level_ncoords,
                                       uint32_t ncoords_in,
                                       uint32_t ncoords_out,
                                       uint32_t nsegments_in,
                                       uint32_t nbins_f,
                                       uint32_t nbins,
                                       int k,
                                       cudaStream_t stream);

} // namespace loki::core
