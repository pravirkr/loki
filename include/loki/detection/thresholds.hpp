#pragma once

#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

namespace loki::detection {

// kLegacy: one Monte Carlo simulation per transition. Reference simulation.
// kImproved: one simulation per parent cell, shared by every child threshold.
// kShared: noise (H0) as kImproved; the signal (H1) is one unpruned simulation
//   of shared trials, and a cell keeps the set of shared trials still above its
//   thresholds. Every path's Pd is counted on the same trials (common random
//   numbers), so the forward pass no longer selects paths whose Pd estimate
//   was high by chance. The mode is a complete pipeline, see solve():
//   - search: a cell (stage, threshold, Pd bin) keeps the candidate of lowest
//     (cost so far + cost to go) / Pd, ranked with a floor of minimum support
//     on its noise survivors (a regularised search model, not a population
//     forecast); a surviving noise trial keeps its slot, failed slots are
//     refilled stratified over the survivors; the Pd grid is anchored at the
//     target min_pd with at most 16 bins below it;
//   - selection and refinement on fresh samples (RefineLevel), then
//     tightening, repair and an independent report of Pd and cost of exactly
//     the returned thresholds.
//   The cost C counts every scored node and w_terminal times the survivors of
//   the last stage (the next processing step), C = 1 + sum_s n_{s-1} B_s +
//   w n_{M-1}; w_terminal = 0 recovers the historical definition. legacy and
//   improved always use w = 0.
//
// Three workflows. They are not interchangeable.
//
// Validated (mode "shared"; the EP planner on both backends). After run(),
// solve() returns the schedule to use, with the Pd and cost of exactly those
// thresholds measured on samples that chose nothing:
//   1. scheme(..., mode="shared", min_pd, w_terminal, refine); run().
//   2. sol = solve(): candidates are compared, refined and repaired on fresh
//      samples (RefineLevel), and sol.pd is a direct count on one more fresh
//      signal sample, sol.c an evaluation on fresh noise. get_best_path_
//      thresholds() and the in-run states stay the search's own (optimistic)
//      selection, and save() writes them together with the solution.
//   The Pd is honest within its binomial error; the cost is an independent
//   Monte Carlo estimate, unbiased by the selection but noisy on very large
//   irregular trees. Neither forecasts the populations of another branching
//   pattern: evaluate() the thresholds on it (the EP planner does, per chunk).
//
// Operational (live search, modes legacy and improved). The path is generated
// on the fly and used at once:
//   1. run(thres_neigh) with mode="improved", a fixed seed, ntrials=1024.
//   2. path = get_best_path_thresholds(min_pd) and prune real data with it.
//   3. The in-run terminal cost and success_h1_cumul are internal Monte Carlo
//      estimates. The forward pass keeps a minimum of noisy scores, so those
//      estimates are optimistic (winner's curse). They are not a measurement
//      of detection performance on data.
//
// Independent evaluation (reporting and EP population forecasts). evaluate()
// never changes the selected path or applies min_pd as a second filter:
//   1. The same run() and path as above.
//   2. ev = evaluate(path, ntrials_eval, seed_eval) with seed_eval != seed.
//   3. Report ev.back().cost and ev.back().success_h1_cumul; a lower evaluated
//      Pd does not invalidate the selection. EP planners use evaluated
//      complexity on actual chunk branching patterns to estimate memory.
// EP planners accept the best available simulated path with a warning when no
// path meets min_pd; the low-level selector still returns empty in that case.
// The configured cutoff is never modified. Raising ntrials inside run() is
// what changes the path. evaluate() does not.
//
// Why the in-run Pd of legacy and improved is optimistic: every cell keeps the
// cheapest of many candidates whose Pd fell in its bin, and the cheapest is
// the one whose noisy Pd estimate came out high. This selection acts at every
// stage, so the bias grows with the number of stages and falls roughly as
// 1/ntrials. The shared mode removes it (every path's Pd is counted on the
// same signal trials), and solve() adds the independent report.
//
// Calibrated workflow (modes legacy and improved). calibrate(min_pd) runs
// after run() and does not change the search:
//   1. Candidates: the cheapest terminal cells whose in-run Pd is near min_pd,
//      always including the get_best_path_thresholds(min_pd) path.
//   2. One unpruned signal simulation of ntrials_cal trials scores every
//      stage. Pd does not depend on the branching pattern, so it serves every
//      candidate.
//   3. Each candidate is shifted by one constant delta (in S/N) so that its
//      Pd on that sample is min_pd: a closed-form order statistic. A
//      candidate that already exceeds min_pd is also kept unshifted.
//   4. evaluate() on fresh noise measures each variant's cost; the cheapest
//      C / Pd wins. Noise and signal streams are independent, so this choice
//      does not bias Pd.
// The returned Pd is at least min_pd, to within
// sqrt(min_pd (1 - min_pd) / ntrials_cal); a path is moved only when it
// misses min_pd, or when moving it lowers C / Pd.
enum class DynamicThresholdMode : uint8_t { kLegacy, kImproved, kShared };

inline DynamicThresholdMode
dynamic_threshold_mode_from_string(std::string_view s) {
    if (s == "legacy") {
        return DynamicThresholdMode::kLegacy;
    }
    if (s == "improved") {
        return DynamicThresholdMode::kImproved;
    }
    if (s == "shared") {
        return DynamicThresholdMode::kShared;
    }
    throw std::invalid_argument(std::format("unknown mode: {}", s));
}

inline std::string mode_to_string(DynamicThresholdMode m) {
    switch (m) {
    case DynamicThresholdMode::kLegacy:
        return "legacy";
    case DynamicThresholdMode::kImproved:
        return "improved";
    case DynamicThresholdMode::kShared:
        return "shared";
    }
    return "unknown";
}

struct State {
    float success_h0{0.0F};
    float success_h1{0.0F};
    float complexity{0.0F};
    float complexity_cumul{std::numeric_limits<float>::max()};
    float success_h1_cumul{0.0F};
    float nbranches{0.0F};
    float threshold{-1.0F};
    float cost{std::numeric_limits<float>::max()};
    float threshold_prev{-1.0F};
    float success_h1_cumul_prev{0.0F};
    bool is_empty{true};

    LOKI_HD State() = default;

    LOKI_HD State gen_next(float threshold,
                           float success_h0,
                           float success_h1,
                           float nbranches) const noexcept {
#ifdef __CUDA_ARCH__
        // Explicitly rounded ops: no FMA contraction or fast-math division,
        // so every kernel that evaluates this gets the same bits.
        const auto nleaves_next = __fmul_rn(this->complexity, nbranches);
        const auto nleaves_surv = __fmul_rn(nleaves_next, success_h0);
        const auto complexity_cumul_next =
            __fadd_rn(this->complexity_cumul, nleaves_next);
        const auto success_h1_cumul_next =
            __fmul_rn(this->success_h1_cumul, success_h1);
        const auto cost_next =
            __fdiv_rn(complexity_cumul_next, success_h1_cumul_next);
#else
        const auto nleaves_next = this->complexity * nbranches;
        const auto nleaves_surv = nleaves_next * success_h0;
        const auto complexity_cumul_next =
            this->complexity_cumul + nleaves_next;
        const auto success_h1_cumul_next = this->success_h1_cumul * success_h1;
        const auto cost_next = complexity_cumul_next / success_h1_cumul_next;
#endif

        // Create a new state struct
        State state_next;
        state_next.success_h0       = success_h0;
        state_next.success_h1       = success_h1;
        state_next.complexity       = nleaves_surv;
        state_next.complexity_cumul = complexity_cumul_next;
        state_next.success_h1_cumul = success_h1_cumul_next;
        state_next.nbranches        = nbranches;
        state_next.threshold        = threshold;
        state_next.cost             = cost_next;
        state_next.is_empty         = false;
        // For backtracking
        state_next.threshold_prev        = this->threshold;
        state_next.success_h1_cumul_prev = this->success_h1_cumul;
        return state_next;
    }

    static LOKI_HD State initial() noexcept {
        State s;
        s.complexity       = 1.0F;
        s.complexity_cumul = 1.0F;
        s.success_h1_cumul = 1.0F;
        s.is_empty         = false;
        return s;
    }
};

/// Total cost of a path ending in state @p s: every scored node plus
/// @p w_terminal times the survivors of the last stage.
inline double total_cost(const State& s, double w_terminal) noexcept {
    return static_cast<double>(s.complexity_cumul) +
           (w_terminal * static_cast<double>(s.complexity));
}

/// What solve() does after the search (mode "shared"). Every level ends with
/// the same tightening, repair and independent report.
///   kNone:      the search's best path only.
///   kShortlist: up to four candidate paths (second pass best, first pass
///               best, diverse terminal cells); the one with the lowest
///               C / Pd on a fresh noise sample is kept.
///   kFull:      kShortlist, then a local search over four blocks of stages
///               on fresh samples.
enum class RefineLevel : uint8_t { kNone, kShortlist, kFull };

inline RefineLevel refine_level_from_string(std::string_view s) {
    if (s == "none") {
        return RefineLevel::kNone;
    }
    if (s == "shortlist") {
        return RefineLevel::kShortlist;
    }
    if (s == "full") {
        return RefineLevel::kFull;
    }
    throw std::invalid_argument(std::format("unknown refine level: {}", s));
}

inline std::string refine_to_string(RefineLevel r) {
    switch (r) {
    case RefineLevel::kNone:
        return "none";
    case RefineLevel::kShortlist:
        return "shortlist";
    case RefineLevel::kFull:
        return "full";
    }
    return "unknown";
}

/// Settings of mode "shared"; the other modes ignore them.
struct SharedOptions {
    /// Target Pd. Anchors the probability grid: @p nprobs bins from min_pd to
    /// 1, plus at most 16 below it down to min_pd / 2 (the constructor's
    /// prob_min is unused). solve() and get_best_path_thresholds() default to
    /// it.
    float min_pd = 0.1F;
    /// Weight of the last stage's survivors in the cost (see above).
    float w_terminal   = 1.0F;
    RefineLevel refine = RefineLevel::kFull;
};

/// How the Pd of a returned schedule compares with the target.
enum class Delivery : uint8_t {
    kNoPath,          ///< no terminal path: thresholds are empty
    kMet,             ///< pd >= min_pd
    kWithinTolerance, ///< 0.9 min_pd <= pd < min_pd
    kShort,           ///< pd < 0.9 min_pd
};

inline std::string delivery_to_string(Delivery d) {
    switch (d) {
    case Delivery::kNoPath:
        return "no_path";
    case Delivery::kMet:
        return "met";
    case Delivery::kWithinTolerance:
        return "within_tolerance";
    case Delivery::kShort:
        return "short";
    }
    return "unknown";
}

/// Result of DynamicThresholdScheme::solve().
struct SchemeSolution {
    /// The schedule to use: one threshold per stage.
    std::vector<float> thresholds;
    /// Pd of exactly these thresholds: a direct count on a fresh sample that
    /// never influenced any choice, with its binomial standard error.
    double pd{0.0};
    double pd_se{0.0};
    SizeType pd_trials{0};
    /// Total cost C (with w_terminal) of these thresholds on a fresh noise
    /// sample (independent of the selection, but a Monte Carlo estimate: on
    /// very large irregular trees it can be off by a large factor), and
    /// C / pd.
    double c{0.0};
    double cost{0.0};
    /// Stage states of that noise evaluation of the region's branching
    /// pattern (the signal side is 1). Not a forecast for another pattern:
    /// evaluate() those.
    std::vector<State> eval_states;
    /// The candidate paths of the shortlist, after tightening (empty at
    /// RefineLevel::kNone).
    std::vector<std::vector<float>> shortlist;
    /// Raw search selection before tightening, repair and refinement, with
    /// its in-run Pd and C. Both are optimistic.
    std::vector<float> search_thresholds;
    double search_pd{0.0};
    double search_c{0.0};
    /// Named counters and timings of the pipeline (shift applied, tightened
    /// stages, block moves, shortlist sizes, floor events, ...).
    std::vector<std::pair<std::string, double>> diagnostics;
    Delivery delivery{Delivery::kNoPath};
    float min_pd{0.0F};
    float w_terminal{0.0F};
    RefineLevel refine{RefineLevel::kNone};
};

struct CalibrationOptions {
    /// Cheapest terminal cells to calibrate, on top of the
    /// get_best_path_thresholds(min_pd) path. More did not lower the cost in
    /// tests: their cost estimates are noisy, so picking among them adds
    /// little.
    SizeType ncandidates = 1;
    /// Candidates need an in-run Pd of at least candidate_frac * min_pd.
    float candidate_frac = 0.5F;
    /// Trials of the unpruned signal simulation that sets the shift.
    SizeType ntrials_cal = SizeType{1} << 15U;
    /// Trials of the evaluate() that measures each candidate's cost.
    SizeType ntrials_cost = 4096;
    /// Calibration seed. Unset draws a random one. The cost evaluation uses a
    /// seed derived from it.
    std::optional<uint64_t> seed = std::nullopt;
    /// Largest |delta| (S/N). A shift beyond it is clamped, with a warning.
    float delta_max = 1.5F;
    /// Allow positive shifts (raise thresholds of a path whose Pd overshoots).
    bool allow_raise = true;
};

struct CalibratedPath {
    /// Shifted thresholds, one per stage. Empty when there is no candidate.
    std::vector<float> thresholds;
    /// Shift added to the source path's thresholds.
    float delta{0.0F};
    /// Pd on the calibration sample, and its binomial standard error.
    float pd_certified{0.0F};
    float pd_sigma{0.0F};
    /// Pd of the cost evaluation (independent check).
    float pd_check{0.0F};
    /// Cumulative complexity of the cost evaluation, and complexity /
    /// pd_certified.
    float complexity_cumul{0.0F};
    float cost{0.0F};
    /// In-run Pd and cost of the source terminal cell.
    float pd_inrun{0.0F};
    float cost_inrun{0.0F};
    /// Candidates calibrated.
    SizeType ncandidates{0};
    /// Stage states of the cost evaluation.
    std::vector<State> states;
};

class DynamicThresholdScheme {
public:
    /// @p batch_size is used by the GPU backends only; @p exec.nthreads by
    /// the CPU backend only.
    ///
    /// Probability grid: @p nprobs log-spaced bins of cumulative Pd from
    /// @p prob_min to 1, i.e. (nprobs - 1) / log10(1 / prob_min) bins per
    /// decade. A cell keeps one path per (threshold, bin), so the bin width
    /// is the resolution at which the search trades Pd against cost. It is
    /// the setting that matters most for the cost of the selected path.
    /// Mode "shared" builds its grid from SharedOptions::min_pd instead: nprobs
    /// bins from min_pd to 1 (min_pd is an edge) and at most 16 below it;
    /// prob_min is unused. Run time and run() memory grow linearly with
    /// nprobs. Measured with the 127-stage production pattern (nbins 64,
    /// ntrials 1024, the search of the shared mode before its cost-per-
    /// detection ranking, grid anchored at min_pd 0.1, CUDA on an L40S, 16
    /// seeds), cost L of the selected path rescored on 2^20 trials:
    ///
    ///   bins per decade |  19 |  29 |  39 |  49 |  59 |  79
    ///   L / L(79)       | 1.53| 1.34| 1.20| 1.13| 1.05| 1.00
    ///   scatter of L    | 0.23| 0.20| 0.15| 0.14| 0.12| 0.16
    ///   run time (s)    | 0.09| 0.13| 0.16| 0.20| 0.24| 0.32
    ///   memory (GiB)    | 1.3 | 2.0 | 2.7 | 3.3 | 4.0 | 5.3
    ///
    /// The default (40 bins over the default prob_min 0.05, 31 per decade)
    /// is the compromise; about 60 per decade buys another ~10 % in L at
    /// 1.5x the time and memory. More thresholds do not help (nthresholds
    /// 50-200 were equivalent).
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           float ref_ducy,
                           SizeType nbins               = 64,
                           SizeType ntrials             = 1024,
                           SizeType nprobs              = 40,
                           float prob_min               = 0.05F,
                           float snr_final              = 8.0F,
                           SizeType nthresholds         = 100,
                           float ducy_max               = 0.3F,
                           float wtsp                   = 1.0F,
                           float beam_width             = 0.7F,
                           SizeType trials_start        = 1,
                           std::string_view mode        = "legacy",
                           std::optional<uint64_t> seed = std::nullopt,
                           SizeType batch_size          = 256,
                           Exec exec                    = {});
    /// As above, with the beam guess: "analytic" (the default above: the
    /// lower of the expected signal S/N sqrt(m/nseg) snr_final and the
    /// trials-aware S/N) or "pilot" (the expected signal term is replaced by
    /// the median score of an unpruned signal simulation, so the beam follows
    /// the score the scheme thresholds: boxcar match and the maximum over
    /// templates included) or "window" (recommended; no guess and no beam
    /// constants):
    ///   - stage s searches every grid threshold in
    ///     [Q_s(eps prob_min / nstages), Q_s(1 - prob_min / 2)], with Q_s the
    ///     quantile of an unpruned signal simulation and eps 0.02: every
    ///     threshold a path on the probability grid can use;
    ///   - run() takes every transition between consecutive beams
    ///     (thres_neigh and beam_width are unused);
    ///   - in the improved and shared modes run() makes two passes. The
    ///     first estimates, from its paths ending below 8 prob_min, the
    ///     future cost per surviving noise leaf at each stage; the second
    ///     keeps, per cell, the candidate of lowest cost so far plus that
    ///     cost to go of its survivors. Ranking by cost so far alone keeps
    ///     cheap pasts with expensive futures once the beam is wide.
    ///   Cost (two passes, a wider beam): at equal nprobs, 2.4x the run time
    ///   of the analytic beam at the production pattern (0.36 s against
    ///   0.15 s on an L40S), 6-8x on 8-32 segment patterns (still under
    ///   0.1 s), and 1.2-4x its run memory. Measured
    ///   (CUDA, 8-16 seeds, rescored): equal cost L at the 127-stage
    ///   production pattern with half its scatter, and 30-55 % lower L on
    ///   the 4-32 segment patterns of issue #15, where the analytic beam
    ///   misses the cheap paths.
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           float ref_ducy,
                           SizeType nbins,
                           SizeType ntrials,
                           SizeType nprobs,
                           float prob_min,
                           float snr_final,
                           SizeType nthresholds,
                           float ducy_max,
                           float wtsp,
                           float beam_width,
                           SizeType trials_start,
                           std::string_view mode,
                           std::optional<uint64_t> seed,
                           SizeType batch_size,
                           std::string_view guess,
                           Exec exec = {});
    /// As above, with the settings of mode "shared".
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           float ref_ducy,
                           SizeType nbins,
                           SizeType ntrials,
                           SizeType nprobs,
                           float prob_min,
                           float snr_final,
                           SizeType nthresholds,
                           float ducy_max,
                           float wtsp,
                           float beam_width,
                           SizeType trials_start,
                           std::string_view mode,
                           std::optional<uint64_t> seed,
                           SizeType batch_size,
                           std::string_view guess,
                           const SharedOptions& shared,
                           Exec exec = {});
    ~DynamicThresholdScheme();
    DynamicThresholdScheme(DynamicThresholdScheme&&) noexcept;
    DynamicThresholdScheme& operator=(DynamicThresholdScheme&&) noexcept;
    DynamicThresholdScheme(const DynamicThresholdScheme&)            = delete;
    DynamicThresholdScheme& operator=(const DynamicThresholdScheme&) = delete;

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const;
    std::vector<float> get_branching_pattern() const;
    std::vector<float> get_profile() const;
    std::vector<float> get_thresholds() const;
    std::vector<float> get_probs() const;
    SizeType get_nstages() const;
    SizeType get_nthresholds() const;
    SizeType get_nprobs() const;
    std::vector<SizeType> get_box_score_widths() const;
    std::vector<State> get_states() const;
    /// Search. Deterministic per (seed, mode, toolchain), and on the GPU also
    /// per `batch_size`. On the CPU every simulated work item (stage, parent
    /// cell, target threshold, H0/H1) draws from its own stream, so results
    /// do not depend on the thread count, on scheduling or on earlier runs in
    /// the process. CPU and CUDA use different generators and agree only
    /// statistically.
    /// In-run cost and success_h1_cumul are optimistic Monte Carlo estimates.
    void run(SizeType thres_neigh = 10);
    /// Fresh Monte Carlo of one fixed threshold per stage, for reporting or
    /// EP population forecasts. Never modifies selection or re-filters Pd.
    /// `ntrials` may
    /// differ from the search. Unset `seed` draws a random one; pass a seed
    /// different from the search seed. Deterministic per (seed, thresholds,
    /// ntrials). Each trial has its own stream.
    std::vector<State>
    evaluate(std::span<const float> thresholds,
             SizeType ntrials,
             std::optional<uint64_t> seed = std::nullopt) const;
    std::string save(const std::string& outdir = "./") const;
    /// Select minimum-cost terminal states whose in-run cumulative Pd >=
    /// min_pd. Returns an empty path when none qualify; never relaxes min_pd.
    /// Independent evaluate() estimates may differ and do not change this path.
    std::vector<float> get_best_path_thresholds(float min_pd = 0.1F) const;
    /// Certify and calibrate after run(): see the workflow above. Does not
    /// change the search states or get_best_path_thresholds().
    CalibratedPath calibrate(float min_pd                   = 0.1F,
                             const CalibrationOptions& opts = {}) const;
    /// Mode "shared" only, after run(): the schedule to use, with the Pd and
    /// cost of exactly those thresholds measured on fresh samples. Selects
    /// among candidates, refines (RefineLevel) and repairs on independent
    /// samples, then reports; nothing reported was used to choose. Frees the
    /// search's device memory. Deterministic per (seed, toolchain), and on the
    /// GPU per `batch_size`. Also kept for save().
    SchemeSolution solve();
    /// The last solve(), or nullptr.
    const SchemeSolution* solution() const;
    /// Unpruned signal (H1) scores of every stage: `ntrials` trials,
    /// stage-major (nstages x ntrials). Uses evaluate()'s streams for `seed`.
    std::vector<float> simulate_h1_scores(SizeType ntrials,
                                          uint64_t seed) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

std::vector<State> evaluate_scheme(std::span<const float> thresholds,
                                   std::span<const float> branching_pattern,
                                   float ref_ducy,
                                   SizeType nbins   = 64,
                                   SizeType ntrials = 1024,
                                   float snr_final  = 8.0F,
                                   float ducy_max   = 0.3F,
                                   float wtsp       = 1.0F);

std::vector<State> determine_scheme(std::span<const float> survive_probs,
                                    std::span<const float> branching_pattern,
                                    float ref_ducy,
                                    SizeType nbins   = 64,
                                    SizeType ntrials = 1024,
                                    float snr_final  = 8.0F,
                                    float ducy_max   = 0.3F,
                                    float wtsp       = 1.0F);

} // namespace loki::detection
