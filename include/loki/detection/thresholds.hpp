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

// Dynamic threshold scheme: one pruning threshold per EP stage. The problem,
// the modes, the default pipeline and its measured behaviour are described in
// docs/dts.md; the comments here cover the interface only.
//
// kLegacy: one Monte Carlo simulation per transition (reference).
// kImproved: one simulation per parent cell (reference). After run(),
//   get_best_path_thresholds(min_pd) is the search's selection; its in-run
//   cost and Pd are optimistic. calibrate() and evaluate() measure it on
//   independent samples.
// kShared (default): shared signal trials and the guided beam
//   (SharedOptions), one search pass; after run(), solve() returns the
//   schedule to use with its Pd and cost measured on fresh samples.
// kRobust: kShared with a second pass and a refinement in solve(); slower,
//   lower cost per detection.
// The cost C of the shared modes includes w_terminal times the survivors of
// the last stage; legacy and improved use w = 0.
enum class DynamicThresholdMode : uint8_t {
    kLegacy,
    kImproved,
    kShared,
    kRobust
};

/// Modes with shared signal trials, the guided beam and solve().
inline bool is_shared_family(DynamicThresholdMode m) {
    return m == DynamicThresholdMode::kShared ||
           m == DynamicThresholdMode::kRobust;
}

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
    if (s == "robust") {
        return DynamicThresholdMode::kRobust;
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
    case DynamicThresholdMode::kRobust:
        return "robust";
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

/// Settings of modes "shared" and "robust"; the other modes ignore them.
struct SharedOptions {
    /// Target Pd. Anchors the probability grid: @p nprobs bins from min_pd to
    /// 1, plus at most 16 below it down to min_pd / 2 (the constructor's
    /// prob_min is unused). solve() aims at it.
    float min_pd = 0.1F;
    /// Weight of the last stage's survivors in the cost (see above).
    float w_terminal = 1.0F;
    /// Beam half-width around the guess, in standard deviations of the
    /// unpruned signal score of each stage. The searched paths lie within
    /// 2.5 of the guess on the patterns tested.
    float beam_sigma = 3.0F;
    /// Largest change of a path's offset from the guess between two
    /// consecutive stages, in the same unit. The searched paths need up to
    /// 0.9; 0.5 loses cost on short patterns.
    float reach_sigma = 1.0F;
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
    /// The candidate paths of the shortlist, after tightening (mode robust
    /// only).
    std::vector<std::vector<float>> shortlist;
    /// Raw search selection before tightening, repair and refinement, with
    /// its in-run Pd and C. Both are optimistic.
    std::vector<float> search_thresholds;
    double search_pd{0.0};
    double search_c{0.0};
    /// Named counters and timings of the pipeline (shift applied, tightened
    /// stages, block moves, shortlist sizes, ...).
    std::vector<std::pair<std::string, double>> diagnostics;
    Delivery delivery{Delivery::kNoPath};
    float min_pd{0.0F};
    float w_terminal{0.0F};
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
    /// @p prob_min to 1; a cell keeps one path per (threshold, bin). Modes
    /// shared and robust build it from SharedOptions::min_pd instead: nprobs
    /// bins from min_pd to 1 and at most 16 below it (prob_min is unused).
    /// Run time and run() memory grow linearly with nprobs (docs/dts.md).
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
                           std::string_view mode        = "shared",
                           std::optional<uint64_t> seed = std::nullopt,
                           SizeType batch_size          = 256,
                           Exec exec                    = {});
    /// As above, with the beam guess of modes legacy and improved:
    /// "analytic" (the default above: the lower of the expected signal S/N
    /// and the trials-aware S/N), "pilot" (the median score of an unpruned
    /// signal simulation instead of the expected S/N) or "window" (every grid
    /// threshold in a window of signal quantiles and every transition, two
    /// passes ranked with a cost to go; thres_neigh and beam_width unused).
    /// Modes shared and robust ignore it and use the guided beam.
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
    /// As above, with the settings of modes "shared" and "robust".
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
    /// Modes "shared" and "robust" only, after run(): the schedule to use,
    /// with the Pd and cost of exactly those thresholds measured on fresh
    /// samples. Tightens (robust: also selects among candidates and refines)
    /// and repairs on independent samples, then reports; nothing reported
    /// was used to choose. Frees the
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
