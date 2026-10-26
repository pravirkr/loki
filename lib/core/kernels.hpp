#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::core {

using coord::PhaseRun;

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
 * @brief Called after each batch of top coordinates of a cone band, by one
 * thread while the others wait (e.g. to hand over the batch's hits). Must
 * not throw.
 */
using ConeBatchFn = void (*)(void* ctx);

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
                   int nthreads);

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
                               int nthreads);

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
 * The top coordinates run in batches of whole tiles, at most
 * `batch_coords` of them (0: one batch); `batch_fn`, when set, runs after
 * each batch.
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
                        SizeType batch_coords,
                        ConeBatchFn batch_fn,
                        void* batch_ctx,
                        int nthreads);

/**
 * @brief Final FFA level (time domain, accel) merged straight into a scored
 * tile: the final fold is never stored.
 *
 * `fold_in` is the level before the final one (layout `[2, ncoords_prev, 2,
 * nbins]`). Final coordinates are a grid of rows of `row_len` frequencies
 * (`row_len_prev` at the level before). Rows that read the same parent rows
 * are tiled together, `tile_coords` frequencies at a time, so the parents
 * stay in cache across those rows (0 for either length: one row). Each row of
 * a tile is merged into per-thread scratch with the same arithmetic as
 * `ffa_iter` and passed to `score_fn` with its first coordinate index. Tiles
 * are not visited in coordinate order: sort the outputs by index.
 */
void ffa_final_scored(const float* __restrict__ fold_in,
                      const coord::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nbins,
                      SizeType row_len,
                      SizeType row_len_prev,
                      SizeType tile_coords,
                      ConeScoreFn score_fn,
                      void* score_ctx,
                      int nthreads);

/// Called on each tile of final Fourier-domain profiles (layout `[count, 2,
/// nbins_f]`) in per-thread scratch; must be thread-safe.
using ComplexTileFn = void (*)(ComplexType* tile,
                               SizeType coord_begin,
                               SizeType count,
                               void* ctx);

/**
 * @brief Final FFA level (Fourier domain) merged tile by tile into scratch.
 *
 * `fold_in` is the level before the final one (`[2, ncoords_prev, 2,
 * nbins_f]`). Same merge arithmetic and phase steps as `ffa_complex_iter` /
 * `ffa_complex_iter_freq`; `tile_coords` must be a multiple of 32. The accel
 * variant tiles rows that share parent rows as `ffa_final_scored` does and
 * calls `tile_fn` once per row of a tile (sort the outputs by index); the
 * frequency-only variant splits contiguous tiles over threads in static
 * blocks (coordinate order).
 */
void ffa_complex_final_tiles(const ComplexType* __restrict__ fold_in,
                             const coord::FFACoord* __restrict__ coords,
                             SizeType ncoords_cur,
                             SizeType ncoords_prev,
                             SizeType nbins_f,
                             SizeType nbins,
                             SizeType row_len,
                             SizeType row_len_prev,
                             SizeType tile_coords,
                             ComplexTileFn tile_fn,
                             void* tile_ctx,
                             int nthreads);

/**
 * @brief The last two FFA levels (accel, time domain) merged straight into a
 * scored tile: neither the final level nor the one before it is stored.
 *
 * `fold_in` is level L-2 (layout `[4, ncoords_prev2, 2, nbins]`),
 * `coords_prev` the level-(L-1) coordinates, `coords` the final ones. Tiles
 * are those of `ffa_final_scored`; for each, the level-(L-1) rows it reads
 * are merged from level L-2 into per-thread scratch (same arithmetic as
 * `ffa_iter`), then the final profiles from scratch. Bit-identical to
 * storing both levels. Sort the outputs by index.
 */
void ffa_final2_scored(const float* __restrict__ fold_in,
                       const coord::FFACoord* __restrict__ coords_prev,
                       const coord::FFACoord* __restrict__ coords,
                       SizeType ncoords_cur,
                       SizeType ncoords_prev2,
                       SizeType nbins,
                       SizeType row_len,
                       SizeType row_len_prev,
                       SizeType tile_coords,
                       ConeScoreFn score_fn,
                       void* score_ctx,
                       int nthreads);

/**
 * @brief Fourier-domain counterpart of `ffa_final2_scored`: level L-1 rows
 * are merged into scratch in the phase-step blocks of the stored level, so
 * the result is bit-identical to storing both levels.
 */
void ffa_complex_final2_tiles(const ComplexType* __restrict__ fold_in,
                              const coord::FFACoord* __restrict__ coords_prev,
                              const coord::FFACoord* __restrict__ coords,
                              SizeType ncoords_cur,
                              SizeType ncoords_prev,
                              SizeType ncoords_prev2,
                              SizeType nbins_f,
                              SizeType nbins,
                              SizeType row_len,
                              SizeType row_len_prev,
                              SizeType tile_coords,
                              ComplexTileFn tile_fn,
                              void* tile_ctx,
                              int nthreads);

void ffa_complex_final_tiles_freq(
    const ComplexType* __restrict__ fold_in,
    const coord::FFACoordFreq* __restrict__ coords,
    SizeType ncoords_cur,
    SizeType ncoords_prev,
    SizeType nbins_f,
    SizeType nbins,
    SizeType tile_coords,
    ComplexTileFn tile_fn,
    void* tile_ctx,
    int nthreads);

/**
 * @brief Fourier-domain cone band of a frequency-only FFA: `k_levels` merges
 * from the stored level `level_in` (`[nsegments_in, ncoords[0], 2,
 * nbins_f]`), tile by tile in per-thread scratch (temporal tiling, as
 * `ffa_cone_band_freq`).
 *
 * Each tile of `tile_coords` top coordinates of each group of 2^k input
 * segments is merged in cache with the arithmetic and phase steps of
 * `ffa_complex_iter_freq` (bit-identical), then copied to `level_out` (if
 * set) and passed to `tile_fn` (if set; the top level must then be one
 * segment). Tiles are split over threads in contiguous, static blocks.
 * `cone_band_working_floats(coords_levels, ncoords, k_levels, tile_coords,
 * nbins_f)` complex elements of scratch per buffer, two buffers per thread.
 *
 * The phase steps go to `step_table` (64-byte aligned): the top coordinates
 * run in batches of whole tiles whose steps fit it, and of at most
 * `batch_coords` coordinates when that is not 0; `batch_fn`, when set, runs
 * after each batch. A table smaller than `cone_band_tile_step_bytes` throws.
 */
void ffa_complex_cone_band_freq(const ComplexType* level_in,
                                ComplexType* level_out,
                                const coord::FFACoordFreq* const* coords_levels,
                                const SizeType* ncoords,
                                SizeType nsegments_in,
                                SizeType nbins_f,
                                SizeType nbins,
                                SizeType k_levels,
                                SizeType tile_coords,
                                ComplexTileFn tile_fn,
                                void* tile_ctx,
                                std::span<std::byte> step_table,
                                SizeType batch_coords,
                                ConeBatchFn batch_fn,
                                void* batch_ctx,
                                int nthreads);

/// Bytes of one phase step of the complex merges (depends on the SIMD
/// width of the build).
[[nodiscard]] SizeType phase_step_bytes() noexcept;

/**
 * @brief Phase-step table bytes a Fourier cone band needs at least: the
 * steps of one tile of `tile_coords` top coordinates (each level's range
 * rounded out to step blocks). 0 when the band cannot run.
 */
[[nodiscard]] SizeType
cone_band_tile_step_bytes(const coord::FFACoordFreq* const* coords_levels,
                          const SizeType* ncoords,
                          SizeType k_levels,
                          SizeType tile_coords) noexcept;

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
                           int nthreads);

void ffa_iter(const float* __restrict__ fold_in,
              float* __restrict__ fold_out,
              const coord::FFACoord* __restrict__ coords,
              SizeType ncoords_cur,
              SizeType ncoords_prev,
              SizeType nsegments,
              SizeType nbins,
              int nthreads);

void ffa_iter_freq(const float* __restrict__ fold_in,
                   float* __restrict__ fold_out,
                   const coord::FFACoordFreq* __restrict__ coords,
                   SizeType ncoords_cur,
                   SizeType ncoords_prev,
                   SizeType nsegments,
                   SizeType nbins,
                   int nthreads);

/**
 * @brief One complex (Fourier) merge level. `step_table` (64-byte aligned,
 * at least two step blocks) holds the phase steps of a batch of
 * coordinates where a level is merged segment by segment.
 */
void ffa_complex_iter(const ComplexType* __restrict__ fold_in,
                      ComplexType* __restrict__ fold_out,
                      const coord::FFACoord* __restrict__ coords,
                      SizeType ncoords_cur,
                      SizeType ncoords_prev,
                      SizeType nsegments,
                      SizeType nbins_f,
                      SizeType nbins,
                      std::span<std::byte> step_table,
                      int nthreads);

/// Frequency-only counterpart of ffa_complex_iter.
void ffa_complex_iter_freq(const ComplexType* __restrict__ fold_in,
                           ComplexType* __restrict__ fold_out,
                           const coord::FFACoordFreq* __restrict__ coords,
                           SizeType ncoords_cur,
                           SizeType ncoords_prev,
                           SizeType nsegments,
                           SizeType nbins_f,
                           SizeType nbins,
                           std::span<std::byte> step_table,
                           int nthreads);

/**
 * @brief Shift ffa folds and add it to the tree folds for each batch.
 *
 * @param folds_tree  The tree folds data (size: n_leaves * 2 * nbins)
 * @param indices_tree  Indices to access the leaf folds (size: n_leaves)
 * @param folds_ffa  The precomputed ffa folds (size: n_coords * 2 * nbins)
 * @param indices_ffa  Indices to access the ffa folds (size: n_leaves)
 * @param phase_shift  Phase shifts to apply to the ffa folds (size: n_leaves)
 * @param folds_out  The output array (size: n_leaves * 2 * nbins)
 * @param nbins  The number of bins in the input/output arrays (time-domain)
 * @param n_leaves  The number of valid leaves in the tree
 */
void shift_add_linear_batch(const float* __restrict__ folds_tree,
                            const SizeType* __restrict__ indices_tree,
                            const float* __restrict__ folds_ffa,
                            const SizeType* __restrict__ indices_ffa,
                            const float* __restrict__ phase_shift,
                            float* __restrict__ folds_out,
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
                                   const uint32_t* __restrict__ indices_segment,
                                   const SizeType* __restrict__ indices_ffa,
                                   const float* __restrict__ phase_shift,
                                   float* __restrict__ folds_tree,
                                   SizeType nbins,
                                   SizeType n_coords_init,
                                   SizeType n_leaves,
                                   SizeType n_segments) noexcept;

void shift_add_ascend_linear_complex_batch(
    const ComplexType* __restrict__ folds_ffa,
    const uint32_t* __restrict__ indices_segment,
    const SizeType* __restrict__ indices_ffa,
    const float* __restrict__ phase_shift,
    ComplexType* __restrict__ folds_tree,
    SizeType nbins_f,
    SizeType nbins,
    SizeType n_coords_init,
    SizeType n_leaves,
    SizeType n_segments) noexcept;

} // namespace loki::core
