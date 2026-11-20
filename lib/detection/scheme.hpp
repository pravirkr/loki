#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detection/thresholds_engine.hpp"

namespace loki::detection::detail {

std::vector<float>
compute_thresholds(float snr_start, float snr_final, SizeType nthresholds);

std::vector<float> compute_probs(SizeType nprobs, float prob_min = 0.05F);

/// Most probability bins below the target of mode "shared": a bounded
/// fallback, not another fine grid at high targets.
inline constexpr SizeType kSharedMaxBelow = 16;

/** Probability grid of mode "shared", anchored at @p min_pd: @p n_above
 * log-spaced bins from min_pd to 1 (min_pd is a bin edge), extended below
 * min_pd to min_pd / 2 with the same step, then thinned to at most
 * @p max_below bins below min_pd. The lowest edge stays min_pd / 2 (rounded
 * up to the step), so the "window" beam is unchanged. Built edge by edge,
 * never the full grid. Throws unless min_pd is in (0, 1).
 */
std::vector<float> shared_probability_grid(
    float min_pd, SizeType n_above, SizeType max_below = kSharedMaxBelow);

std::vector<float> bound_scheme(SizeType nstages, float snr_bound);

std::vector<float> trials_scheme(std::span<const float> branching_pattern,
                                 SizeType trials_start = 1,
                                 float min_trials      = 1E10F);

std::vector<float> guess_scheme(SizeType nstages,
                                float snr_bound,
                                std::span<const float> branching_pattern,
                                SizeType trials_start = 1,
                                float min_trials      = 1E10F);

/// Beam guess modes of DynamicThresholdScheme.
enum class BeamGuess : uint8_t { kAnalytic, kPilot, kWindow };

/// "analytic", "pilot" or "window". Throws on an unknown name.
BeamGuess beam_guess_from_string(std::string_view guess);

/// Trials of the pilot simulation that centres the "pilot" beam guess.
inline constexpr SizeType kPilotGuessTrials = 4096;

/// A seed for an auxiliary simulation, derived from @p seed and @p salt.
uint64_t derive_seed(uint64_t seed, uint64_t salt) noexcept;

/** "pilot" beam guess: per stage, the lower of the median unpruned signal
 * score (h1_scores, stage-major nstages x ntrials) and the trials-aware S/N
 * of trials_scheme.
 */
std::vector<float> pilot_guess_scheme(std::span<const float> h1_scores,
                                      SizeType ntrials,
                                      std::span<const float> branching_pattern,
                                      SizeType trials_start = 1,
                                      float min_trials      = 1E10F);

/// Trials of the signal simulation that sets the "window" beam.
inline constexpr SizeType kWindowTrials = SizeType{1} << 15U;
/// Largest relative Pd the "window" beam's lower edges may cost a path at
/// prob_min.
inline constexpr float kWindowEps = 0.02F;

/// Per-stage threshold beam [lo, hi] (S/N).
struct BeamBounds {
    std::vector<float> lo;
    std::vector<float> hi;
};

/// Beam of half-width @p beam_width around @p guess_path, clipped to
/// [0, snr_max].
BeamBounds centred_beam(std::span<const float> guess_path,
                        float beam_width,
                        float snr_max);

/** "window" beam from unpruned signal scores (h1_scores, stage-major
 * nstages x ntrials). Per stage s:
 *   hi_s = Q_s(1 - prob_min / 2): a threshold above it keeps less than
 *          prob_min / 2 of the signal on its own, so no path that reaches
 *          the probability grid uses it (the factor 2 covers the sampling
 *          error of the quantile);
 *   lo_s = Q_s(eps prob_min / nstages): raising every threshold below it to
 *          lo_s costs a path at most eps prob_min of Pd and never adds noise.
 * Quantiles beyond the sample are clamped to its extremes.
 */
BeamBounds window_scheme(std::span<const float> h1_scores,
                         SizeType ntrials,
                         SizeType nstages,
                         float prob_min,
                         float eps = kWindowEps);

/// Constructor inputs of a searching DynamicThresholdScheme engine.
struct SchemeInputs {
    std::span<const float> branching_pattern;
    float ref_ducy;
    float ducy_max;
    float wtsp;
    float beam_width;
    float prob_min;
    float snr_final;
    SizeType nbins;
    SizeType ntrials;
    SizeType nprobs;
    SizeType nthresholds;
    SizeType batch_size;
};

/// Checks the inputs every backend's search needs; throws
/// std::invalid_argument naming @p who.
void validate_scheme_inputs(const SchemeInputs& in, std::string_view who);

/// Logs the first stage that ended with no state (nothing survived into it).
void warn_empty_stages(std::span<const State> states,
                       SizeType nstages,
                       SizeType nthresholds,
                       SizeType nprobs,
                       std::string_view who);

/// Statistics of the unpruned signal scores that set the guided beam.
struct PilotStats {
    BeamBounds window;        ///< window_scheme
    std::vector<float> guess; ///< pilot_guess_scheme, clamped into window
    std::vector<float> sigma; ///< score_sigma
};

/** window_scheme, pilot_guess_scheme and score_sigma in one pass over
 * @p sorted_scores (stage-major nstages x ntrials, every row ascending).
 * Order statistics equal those of the unsorted functions.
 */
PilotStats pilot_stats(std::span<const float> sorted_scores,
                       SizeType ntrials,
                       std::span<const float> branching_pattern,
                       SizeType trials_start,
                       float prob_min,
                       float eps = kWindowEps);

/// Standard deviation per stage of the unpruned signal scores (h1_scores,
/// stage-major nstages x ntrials).
std::vector<float> score_sigma(std::span<const float> h1_scores,
                               SizeType ntrials,
                               SizeType nstages);

/** Beam of half-width @p width sigma_s around @p guess_path, inside the
 * feasibility @p window. A stage whose guess lies outside its window gets
 * the window edge nearest to the guess.
 */
BeamBounds guided_beam(const BeamBounds& window,
                       std::span<const float> guess_path,
                       std::span<const float> sigma,
                       float width);

/** Largest number of grid thresholds a guided_beam of half-width @p width
 * can hold at any stage, wherever its guess lies: the pool size that fits
 * every recentred beam.
 */
SizeType guided_beam_capacity(std::span<const float> thresholds,
                              const BeamBounds& window,
                              std::span<const float> sigma,
                              float width);

/** A transition of the guided beam from threshold @p prev (guess
 * @p guess_prev) to @p cur (guess @p guess_cur) changes the offset from the
 * guess by at most @p reach standard deviations @p sigma_cur.
 */
inline bool within_reach(float cur,
                         float guess_cur,
                         float prev,
                         float guess_prev,
                         float sigma_cur,
                         float reach) {
    const float d = (cur - guess_cur) - (prev - guess_prev);
    return std::abs(d) <= reach * sigma_cur;
}

/** Guess learnt from a searched path: g + smoothed (path - g), the offset
 * averaged over 3 consecutive stages. @p guess_path if @p path is empty.
 */
std::vector<float> recentre_guess(std::span<const float> guess_path,
                                  std::span<const float> path);

/// Indices of @p thresholds inside [lo, hi].
std::vector<SizeType>
beam_indices(std::span<const float> thresholds, float lo, float hi);

/** Best-path thresholds (one per stage) after a dynamic threshold run.
 *
 * Among final-stage states with cumulative H1 survival >= \p min_pd, picks
 * minimum cost and backtracks using stored prev links. Layout of \p states must
 * match `(nstages × nthresholds × nprobs)` row-major flattening.
 *
 * Returns an empty vector when no final state meets \p min_pd.
 */
std::vector<float> get_best_path_thresholds(std::span<const State> states,
                                            std::span<const float> thresholds,
                                            std::span<const float> probs,
                                            SizeType nstages,
                                            SizeType nthresholds,
                                            SizeType nprobs,
                                            float min_pd = 0.1F);

/** States of the path ending at terminal cell @p terminal_cell, one per
 * stage, by the stored prev links.
 */
std::vector<State> backtrack_states(std::span<const State> states,
                                    std::span<const float> thresholds,
                                    std::span<const float> probs,
                                    SizeType nstages,
                                    SizeType nthresholds,
                                    SizeType nprobs,
                                    SizeType terminal_cell);

/** Terminal ranking of mode "shared": C / Pd with C = complexity_cumul + w *
 * complexity. With @p w_terminal == 0 the stored cost (complexity_cumul / Pd,
 * the historical definition).
 */
double terminal_rank(const State& s, double w_terminal);

/** Stores terminal_rank (with @p w_terminal) in the `cost` of every non-empty
 * cell of the last stage, so saved states rank terminal cells as the solver
 * does. A no-op for @p w_terminal == 0.
 */
void store_terminal_cost(std::span<State> states,
                         SizeType nstages,
                         SizeType nthresholds,
                         SizeType nprobs,
                         double w_terminal);

/** The terminal cell (ithres * nprobs + iprob of the last stage) of lowest
 * terminal_rank among non-empty cells with success_h1_cumul >= @p min_pd, or
 * the maximum SizeType when none qualifies.
 */
SizeType best_terminal_cell(std::span<const State> states,
                            SizeType nstages,
                            SizeType nthresholds,
                            SizeType nprobs,
                            float min_pd,
                            double w_terminal);

/** Expected future cost per surviving noise leaf, per stage, from a finished
 * run (the second pass of modes improved with the "window" beam and robust
 * ranks cell candidates by cost so far plus h[s] * complexity). Paths: the
 * cheapest terminal cell (terminal_rank) of every probability bin below
 * pd_max_factor * probs[0]. Per stage, h of a path is (C_final - C_s) / n_s;
 * the result is the median over paths, then a moving average over up to 2 *
 * half_window + 1 stages, both in log space and over all stages but the last,
 * which gets @p w_terminal (0: no future and no weight). Empty when no such
 * terminal cell exists.
 */
std::vector<float> cost_to_go_scheme(std::span<const State> states,
                                     std::span<const float> thresholds,
                                     std::span<const float> probs,
                                     SizeType nstages,
                                     SizeType nthresholds,
                                     SizeType nprobs,
                                     float pd_max_factor  = 8.0F,
                                     SizeType half_window = 4,
                                     double w_terminal    = 0.0);

/** Thresholds of the path ending at terminal cell @p terminal_cell
 * (ithres * nprobs + iprob of the last stage), by the stored prev links.
 */
std::vector<float> backtrack_path(std::span<const State> states,
                                  std::span<const float> thresholds,
                                  std::span<const float> probs,
                                  SizeType nstages,
                                  SizeType nthresholds,
                                  SizeType nprobs,
                                  SizeType terminal_cell);

/** Number of trials that survive every stage of @p thresholds, given the
 * unpruned H1 scores (stage-major, nstages x ntrials). A trial survives a
 * stage when its score is above that stage's threshold.
 */
SizeType count_survivors(std::span<const float> h1_scores,
                         std::span<const float> thresholds,
                         SizeType ntrials);

/** The path ending at terminal cell @p terminal_cell with its in-run Pd and
 * total cost (with @p w_terminal). Empty thresholds when @p terminal_cell is
 * the maximum SizeType (no cell).
 */
SearchPath search_path(std::span<const State> states,
                       std::span<const float> thresholds,
                       std::span<const float> probs,
                       SizeType nstages,
                       SizeType nthresholds,
                       SizeType nprobs,
                       SizeType terminal_cell,
                       double w_terminal);

/// Trials of @p bank that survive every stage of @p thresholds.
SizeType count_survivors(const SignalBank& bank,
                         std::span<const float> thresholds);

/** Lowers every threshold by one common shift, only if needed, so that at
 * least ceil(target * n) of the n trials of @p bank survive every stage.
 * Never raises. Returns the shift (<= 0).
 */
float relax_schedule(std::vector<float>& thresholds,
                     const SignalBank& bank,
                     double target);

/** Raises each threshold to just below the lowest score of the trials of
 * @p bank that survive the whole schedule, so none of them is lost. A no-op
 * when no trial survives. Returns the number of raised stages and the number
 * of surviving trials.
 */
std::pair<SizeType, SizeType> tighten_schedule(std::vector<float>& thresholds,
                                               const SignalBank& bank);

/** Cut points (stage indices, first 0, last nstages) of up to four blocks of
 * equal total log-branching. */
std::vector<SizeType> refinement_blocks(std::span<const float> branching);

/** DynamicThresholdScheme::solve on a mode "shared" or "robust" engine after
 * run(). @p robust adds the shortlist and the block refinement.
 */
SchemeSolution solve_scheme(ThresholdsEngine& engine,
                            float min_pd,
                            double w_terminal,
                            bool robust);

/** Re-fits a schedule found at a smaller nbins to the folds of @p engine (an
 * evaluation-only engine at the true nbins; nothing else of it is used).
 *
 * The maximum over boxcar templates has a heavier noise tail at a larger
 * nbins (about nbins times more independent phase positions), so a schedule
 * searched at few bins is too permissive there, mostly by one common offset.
 * The refit scans a common upward shift of the schedule on fresh samples at
 * the true nbins and keeps the shift of lowest C / Pd among those that reach
 * the target Pd, then tightens, repairs and reports exactly as solve() does,
 * on further fresh samples. The result is honest at the true nbins; it is not
 * the schedule a native search would find (cost per detection typically
 * 1.0-1.1 times its, measured), and it never fails for lack of memory:
 * @p budget bounds the samples. @p searched is the solution at the small
 * nbins (kept in the diagnostics).
 */
SchemeSolution refit_scheme(const ThresholdsEngine& engine,
                            const SchemeSolution& searched,
                            float min_pd,
                            double w_terminal,
                            uint64_t seed,
                            const SolveBudget& budget);

/** DynamicThresholdScheme::calibrate on any engine (after run()). */
CalibratedPath calibrate_path(const ThresholdsEngine& engine,
                              float min_pd,
                              const CalibrationOptions& opts);

} // namespace loki::detection::detail
