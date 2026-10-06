#pragma once

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::core {

using PhaseRun = coord::PhaseRun;

/**
 * @brief Optional per-tile score hook for the top of a cone band.
 *
 * Called from inside the band's OpenMP region, once per output-frequency
 * tile of one output segment. `profiles` holds `nprofiles` packed
 * energy/variance folds starting at global coordinate `coord_begin`.
 * Implementations must be thread-safe (typically thread-local output).
 */
using ConeScoreFn = void (*)(const float* profiles,
                             SizeType coord_begin,
                             SizeType nprofiles,
                             SizeType nbins,
                             void* ctx);

/**
 * @brief Sum of per-thread seconds inside one cone band.
 *
 * Divide by the band's wall time only after scaling by the fraction of
 * thread-time each bucket represents; the fields themselves are not wall
 * times.
 */
struct ConeBandThreadSeconds {
    double brute{0};       ///< Sum of per-thread brute-fold seconds.
    double merge{0};       ///< Sum of per-thread merge seconds.
    double score{0};       ///< Sum of per-thread score-hook seconds.
    double prefix_wall{0}; ///< Wall-clock seconds of a shared prefix build.
};

/**
 * @brief Brute force fold a time series of data.
 *
 *
 * @param ts_e  The time series of data to fold (size: nsamps)
 * @param ts_v  The time series of data to fold (size: nsamps)
 * @param fold  The output array (size: nsegments * nfreqs * 2 * nbins)
 * @param bucket_indices  The bucket indices (size: nfreqs * segment_len)
 * @param offsets  Prefix sum of the bucket indices (size: nfreqs * nbins + 1)
 * @param nsegments  The number of segments
 * @param nfreqs  The number of frequencies
 * @param nbins  The number of bins in the output array
 * @param nthreads  The number of threads to use
 */
void brute_fold_ts(const float* __restrict__ ts_e,
                   const float* __restrict__ ts_v,
                   float* __restrict__ fold,
                   const PhaseRun* __restrict__ runs,
                   const SizeType* __restrict__ run_offsets,
                   SizeType nsegments,
                   SizeType nfreqs,
                   SizeType segment_len,
                   SizeType nbins,
                   int nthreads) noexcept;

/**
 * @brief Fused time-domain brute fold + first `nlevels` frequency-only FFA
 * merge levels, processed tile by tile so the intermediate levels stay in
 * cache. Bit-exact with brute_fold_ts followed by `nlevels` calls of
 * ffa_iter_freq.
 *
 * A tile is 2^nlevels adjacent brute-fold segments. The result is the FFA
 * level-`nlevels` fold (nsegments >> nlevels segments, ncoords[nlevels]
 * profiles each).
 *
 * @param fold_out  Level-`nlevels` output (size: (nsegments >> nlevels) *
 *                  ncoords[nlevels] * 2 * nbins)
 * @param runs  Phase runs for every frequency (same table for every segment)
 * @param run_offsets  `nfreqs + 1` offsets into `runs`
 * @param coords_levels  coords_levels[j] points to the level-j coordinates,
 *                       j = 1..nlevels (index 0 is unused)
 * @param ncoords  Profiles per segment at levels 0..nlevels
 * @param nsegments  Brute-fold segments; must be a multiple of 2^nlevels
 */
void brute_fold_ffa_fused_freq(const float* __restrict__ ts_e,
                               const float* __restrict__ ts_v,
                               float* __restrict__ fold_out,
                               const PhaseRun* __restrict__ runs,
                               const SizeType* __restrict__ run_offsets,
                               const coord::FFACoordFreq* const* coords_levels,
                               const SizeType* ncoords,
                               SizeType nsegments,
                               SizeType nfreqs,
                               SizeType segment_len,
                               SizeType nbins,
                               SizeType nlevels,
                               int nthreads) noexcept;

/**
 * @brief Working-set size, in floats, of one frequency tile of a cone band.
 *
 * A cone band folds `k_levels` merge steps for a tile of `tile_coords`
 * output coordinates entirely in scratch. Because `coords[i].idx` is
 * monotone, the profiles a tile needs at every lower level are a contiguous
 * index range, read off the first and last coordinate of the tile. The
 * scratch holds the widest of those ranges (two ping-pong copies are paid
 * by the caller). Returns 0 when any level in the band is not monotone, in
 * which case the band must not run.
 *
 * Halo: a window of W consecutive coordinates maps to
 * `idx(last) - idx(first) + 1` profiles, which is about `W / 2` when the
 * frequency grid doubles each level, plus a one-bin rounding fringe that
 * grows like `2^(k-j)` at band-local level j. For `tile_coords = 512` and
 * `k_levels <= 6` that fringe is a few percent of the tile.
 *
 * @param coords_levels  `coords_levels[j]` is band-local level j, j = 1..k
 *                       (index 0 is unused)
 * @param ncoords        Profile counts at band-local levels 0..k
 */
[[nodiscard]] SizeType
cone_band_working_floats(const coord::FFACoordFreq* const* coords_levels,
                         const SizeType* ncoords,
                         SizeType k_levels,
                         SizeType tile_coords,
                         SizeType nbins) noexcept;

/**
 * @brief Execute one frequency-tiled cone band of a frequency-only FFA.
 *
 * The band covers `k_levels` merges. Input is either a materialized fold
 * level (`level_in`, layout `[nsegments_in, ncoords[0], 2, nbins]`) or, when
 * `level_in` is null, the time series folded with `runs` (the bottom band).
 * Output tiles are packed back into `level_out` with the same layout at
 * `nsegments_in >> k_levels` segments. Tiles partition the output coordinates,
 * so writes do not overlap. Each tile's scratch is fixed (see
 * cone_band_working_floats), which keeps memory predictable and maps a tile
 * onto one GPU thread block.
 *
 * The bottom band walks segment-groups first (`2^k_levels` input segments)
 * so the group's prefix sums stay in cache across every frequency tile of
 * that group. Level 0 of the band is never written to DRAM.
 *
 * `score_fn`, when set, is invoked on each output tile while it is still in
 * scratch. Pass a null `level_out` to skip materialising the band output
 * (the top band of a scored search).
 *
 * @param thread_seconds  Optional accumulator of per-thread brute, merge and
 *                        score seconds. Not wall-clock time.
 */
void ffa_cone_band_freq(const float* level_in,
                        const float* ts_e,
                        const float* ts_v,
                        const PhaseRun* runs,
                        const SizeType* run_offsets,
                        SizeType segment_len,
                        float* level_out,
                        const coord::FFACoordFreq* const* coords_levels,
                        const SizeType* ncoords,
                        SizeType nsegments_in,
                        SizeType nbins,
                        SizeType k_levels,
                        SizeType tile_coords,
                        ConeScoreFn score_fn,
                        void* score_ctx,
                        ConeBandThreadSeconds* thread_seconds,
                        int nthreads);

// Fallback implementation
void brute_fold_ts_complex_xsimd(const float* __restrict__ ts_e,
                                 const float* __restrict__ ts_v,
                                 ComplexType* __restrict__ fold,
                                 const double* __restrict__ freqs,
                                 SizeType nfreqs,
                                 SizeType nsegments,
                                 SizeType segment_len,
                                 SizeType nbins,
                                 double tsamp,
                                 double t_ref,
                                 int nthreads) noexcept;

void brute_fold_ts_complex(const float* __restrict__ ts_e,
                           const float* __restrict__ ts_v,
                           ComplexType* __restrict__ fold,
                           const double* __restrict__ freqs,
                           SizeType nfreqs,
                           SizeType nsegments,
                           SizeType segment_len,
                           SizeType nbins,
                           double tsamp,
                           double t_ref,
                           int nthreads) noexcept;

void ffa_iter(const float* __restrict__ fold_in,
              float* __restrict__ fold_out,
              const coord::FFACoord* __restrict__ coords,
              SizeType ncoords_cur,
              SizeType ncoords_prev,
              SizeType nsegments,
              SizeType nbins,
              int nthreads) noexcept;

void ffa_iter_freq(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const coord::FFACoordFreq* __restrict__ coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   int nthreads) noexcept;

void ffa_complex_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      const coord::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nsegments,
                      SizeType nbins_f,
                      SizeType nbins,
                      int nthreads) noexcept;

void ffa_complex_iter_freq(const ComplexType* __restrict__ fold_in,
                           ComplexType* __restrict__ fold_out,
                           const coord::FFACoordFreq* __restrict__ coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           int nthreads) noexcept;

/**
 * @brief Shift ffa folds and add it to the tree folds for each batch.
 *
 * @param folds_tree  The tree folds data (size: n_leaves * 2 * nbins)
 * @param indices_tree  Indices to access the leaf folds (size: n_leaves)
 * @param folds_ffa  The precomputed ffa folds (size: n_coords * 2 * nbins)
 * @param indices_ffa  Indices to access the ffa folds (size: n_leaves)
 * @param phase_shift  Phase shifts to apply to the ffa folds (size: n_leaves)
 * @param folds_out  The output array (size: n_leaves * 2 * nbins)
 * @param temp_buffer  Pre-allocated buffer of size 2 * nbins
 * @param nbins  The number of bins in the input/output arrays (time-domain)
 * @param n_leaves  The number of valid leaves in the tree
 */
void shift_add_linear_batch(const float* __restrict__ folds_tree,
                            const SizeType* __restrict__ indices_tree,
                            const float* __restrict__ folds_ffa,
                            const SizeType* __restrict__ indices_ffa,
                            const float* __restrict__ phase_shift,
                            float* __restrict__ folds_out,
                            float* __restrict__ temp_buffer,
                            SizeType nbins,
                            SizeType n_leaves,
                            SizeType physical_start_idx,
                            SizeType capacity) noexcept;

/**
 * @brief Shift complex ffa folds and add it to the complex tree folds for each
 * leaf.
 *
 * @param folds_tree  The tree folds data (size: n_leaves * 2 * nbins_f)
 * @param indices_tree  Indices to access the leaf folds (size: n_leaves)
 * @param folds_ffa  The precomputed ffa folds (size: n_coords * 2 * nbins_f)
 * @param indices_ffa  Indices to access the ffa folds (size: n_coords)
 * @param phase_shift  Phase shifts to apply to the ffa folds (size: n_leaves)
 * @param folds_out  The output array (size: n_leaves * 2 * nbins_f)
 * @param nbins_f  The number of frequency bins (FFT size)
 * @param nbins  The number of time-domain bins (original fold size)
 * @param n_leaves  The number of valid leaves in the tree
 *
 * @note Uses recurrence relation for phase calculation for optimal performance.
 */
void shift_add_linear_complex_batch(const ComplexType* __restrict__ folds_tree,
                                    const SizeType* __restrict__ indices_tree,
                                    const ComplexType* __restrict__ folds_ffa,
                                    const SizeType* __restrict__ indices_ffa,
                                    const float* __restrict__ phase_shift,
                                    ComplexType* __restrict__ folds_out,
                                    SizeType nbins_f,
                                    SizeType nbins,
                                    SizeType n_leaves,
                                    SizeType physical_start_idx,
                                    SizeType capacity) noexcept;

void shift_add_ascend_linear_batch(const float* __restrict__ folds_ffa,
                                   const SizeType* __restrict__ indices_segment,
                                   const SizeType* __restrict__ indices_ffa,
                                   const float* __restrict__ phase_shift,
                                   float* __restrict__ folds_tree,
                                   float* __restrict__ temp_buffer,
                                   SizeType nbins,
                                   SizeType n_coords_init,
                                   SizeType n_leaves,
                                   SizeType n_segments) noexcept;

void shift_add_ascend_linear_complex_batch(
    const ComplexType* __restrict__ folds_ffa,
    const SizeType* __restrict__ indices_segment,
    const SizeType* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexType* __restrict__ folds_tree,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments) noexcept;

} // namespace loki::core