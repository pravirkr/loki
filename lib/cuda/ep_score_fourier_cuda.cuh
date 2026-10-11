#pragma once

/**
 * @file ep_score_fourier_cuda.cuh
 * @brief Fused Fourier-domain pruning step of the EP (CUDA only): shift-add
 * of each branched leaf, inverse real FFT on chip and boxcar score with the
 * threshold, in one kernel. See docs/fft.md.
 */

#include <cstdint>

#include <cuda_runtime.h>

#include "loki/common/types.hpp"

#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

/// Arguments of ep_shift_add_score_fourier_cuda. Folds are complex with
/// 2 * nbins_f harmonics per leaf (e, then v), nbins_f = nbins / 2 + 1.
struct EPFourierScoreArgs {
    // Parent folds in the circular world tree, indexed by the logical
    // indices_tree + physical_start_idx (mod capacity).
    const ComplexTypeCUDA* folds_tree{nullptr};
    const uint32_t* indices_tree{nullptr};
    const uint8_t* validation_mask{nullptr};
    // FFA segment folds, indexed by indices_ffa; rotated by phase_shift bins.
    const ComplexTypeCUDA* folds_ffa{nullptr};
    const uint32_t* indices_ffa{nullptr};
    const float* phase_shift{nullptr};
    // Merged folds of the valid leaves (bit-identical to
    // kernel_shift_add_linear_complex), max S/N over the widths, and
    // max S/N >= threshold (0 for invalid leaves).
    ComplexTypeCUDA* folds_out{nullptr};
    float* scores{nullptr};
    uint8_t* filtered_mask{nullptr};
    // Batch statistics, zeroed by the caller: [0] the number of passing
    // leaves, [1] the ordered key (ep_score_key) of the largest score of a
    // valid leaf, [2] the complement of the key of the smallest.
    uint32_t* stats{nullptr};
    const uint32_t* widths{nullptr};
    uint32_t nwidths{0};
    float threshold{0.0F};
    uint32_t n_leaves{0};
    uint32_t physical_start_idx{0};
    uint32_t capacity{0};
    uint32_t nbins{0};
    // ffa_fourier_bluestein_table(nbins) on the device when
    // ffa_fourier_needs_bluestein(nbins).
    const ComplexTypeCUDA* bluestein{nullptr};
    // Direct mode (tree_folds != nullptr): the passing leaves go straight
    // into the world tree's write region, in leaf order (the order of a
    // stable compaction of filtered_mask): leaf r gets physical slot
    // (tree_write_head + r) % capacity, with its row of leaves_batch, its
    // merged fold and its score. folds_out is not written. The caller
    // guarantees n_leaves free slots.
    const double* leaves_batch{nullptr};
    uint32_t leaves_stride{0};
    double* tree_leaves{nullptr};
    ComplexTypeCUDA* tree_folds{nullptr};
    float* tree_scores{nullptr};
    uint32_t tree_write_head{0};
    // Direct mode: ep_shift_add_score_fourier_tile_words() words, zeroed by
    // the caller (tile states of the ordered compaction, then a ticket).
    unsigned long long* tile_state{nullptr};
};

/// Order-preserving map of a float to an unsigned key: a < b iff
/// key(a) < key(b), and no float maps to 0 (the empty value).
__host__ __device__ inline uint32_t ep_score_key(uint32_t bits) {
    return (bits & 0x80000000U) != 0 ? ~bits : (bits | 0x80000000U);
}
/// Inverse of ep_score_key, as the float's bits.
__host__ __device__ inline uint32_t ep_score_key_bits(uint32_t key) {
    return (key & 0x80000000U) != 0 ? (key & 0x7FFFFFFFU) : ~key;
}

/// Whether the fused step handles @p nbins (any nbins in [2, 1024]).
[[nodiscard]] bool
ep_shift_add_score_fourier_supported(SizeType nbins, SizeType nwidths) noexcept;

/// Words of EPFourierScoreArgs::tile_state for a launch of @p n_leaves
/// leaves (zero them before each direct-mode launch).
[[nodiscard]] SizeType ep_shift_add_score_fourier_tile_words(SizeType nbins,
                                                             SizeType n_leaves);

/**
 * @brief Shift-add, on-chip inverse real FFT and boxcar score of the
 * branched leaves of one pruning batch. Equivalent to
 * shift_add_linear_complex_batch_cuda, an unnormalized C2R and
 * score_and_filter_max_cuda_d with fold_scale 1 / nbins, up to the rounding
 * of the inverse FFT. Does not synchronize.
 */
void ep_shift_add_score_fourier_cuda(const EPFourierScoreArgs& args,
                                     cudaStream_t stream);

} // namespace loki::core
