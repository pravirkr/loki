#pragma once

#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::detection {

// kLegacy: one Monte Carlo simulation per transition. Reference simulation.
// kImproved: one simulation per parent cell, shared by every child threshold.
//   Operational default. A cell keeps the path with the lowest cumulative
//   complexity. The finished path is then chosen by minimum cost
//   (complexity / P_d) among states with P_d >= min_pd.
//
// Two workflows. They are not interchangeable.
//
// Operational (live search). The path is generated on the fly and used at once:
//   1. run(thres_neigh) with mode="improved", a fixed seed, ntrials=1024.
//   2. path = get_best_path_thresholds(min_pd) and prune real data with it.
//   3. The in-run terminal cost and success_h1_cumul are internal Monte Carlo
//      estimates. The forward pass keeps a minimum of noisy scores, so those
//      estimates are optimistic (winner's curse). They are not a measurement
//      of detection performance on data.
//
// Reporting only (papers, validation, offline tables). evaluate() does not
// change the path and is not part of the on-the-fly pipeline:
//   1. The same run() and path as above.
//   2. ev = evaluate(path, ntrials_eval, seed_eval) with seed_eval != seed.
//   3. Quote ev.back().cost and ev.back().success_h1_cumul.
// Raising ntrials inside run() is what changes the path. evaluate() does not.
enum class DynamicThresholdMode : uint8_t { kLegacy, kImproved };

inline DynamicThresholdMode
dynamic_threshold_mode_from_string(std::string_view s) {
    if (s == "legacy") {
        return DynamicThresholdMode::kLegacy;
    }
    if (s == "improved") {
        return DynamicThresholdMode::kImproved;
    }
    throw std::invalid_argument(std::format("unknown mode: {}", s));
}

inline std::string mode_to_string(DynamicThresholdMode m) {
    switch (m) {
    case DynamicThresholdMode::kLegacy:
        return "legacy";
    case DynamicThresholdMode::kImproved:
        return "improved";
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

class DynamicThresholdScheme {
public:
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           float ref_ducy,
                           SizeType nbins        = 64,
                           SizeType ntrials      = 1024,
                           SizeType nprobs       = 10,
                           float prob_min        = 0.05F,
                           float snr_final       = 8.0F,
                           SizeType nthresholds  = 100,
                           float ducy_max        = 0.3F,
                           float wtsp            = 1.0F,
                           float beam_width      = 0.7F,
                           SizeType trials_start = 1,
                           std::string_view mode = "legacy",
                           int nthreads          = 1,
                           std::optional<uint64_t> seed = std::nullopt);
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
    /// Search. Deterministic per (seed, mode, nthreads, toolchain). Thread
    /// count changes the draws, so two thread counts are not bit-identical.
    /// In-run cost and success_h1_cumul are optimistic Monte Carlo estimates.
    void run(SizeType thres_neigh = 10);
    /// Reporting only. Fresh Monte Carlo of one threshold per stage. Does not
    /// modify run() and is not called before a live search. `ntrials` may
    /// differ from the search. Unset `seed` draws a random one; pass a seed
    /// different from the search seed. Deterministic per (seed, thresholds,
    /// ntrials). Each trial has its own stream.
    std::vector<State> evaluate(std::span<const float> thresholds,
                                SizeType ntrials,
                                std::optional<uint64_t> seed = std::nullopt) const;
    std::string save(const std::string& outdir = "./") const;
    std::vector<float> get_best_path_thresholds(float min_pd = 0.1F) const;

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

#ifdef LOKI_ENABLE_CUDA

class DynamicThresholdSchemeCUDA {
public:
    DynamicThresholdSchemeCUDA(std::span<const float> branching_pattern,
                               float ref_ducy,
                               SizeType nbins        = 64,
                               SizeType ntrials      = 1024,
                               SizeType nprobs       = 10,
                               float prob_min        = 0.05F,
                               float snr_final       = 8.0F,
                               SizeType nthresholds  = 100,
                               float ducy_max        = 0.3F,
                               float wtsp            = 1.0F,
                               float beam_width      = 0.7F,
                               SizeType trials_start = 1,
                               std::string_view mode = "legacy",
                               SizeType batch_size   = 256,
                               int device_id         = 0,
                               std::optional<uint64_t> seed = std::nullopt);
    ~DynamicThresholdSchemeCUDA();
    DynamicThresholdSchemeCUDA(DynamicThresholdSchemeCUDA&&) noexcept;
    DynamicThresholdSchemeCUDA&
    operator=(DynamicThresholdSchemeCUDA&&) noexcept;
    DynamicThresholdSchemeCUDA(const DynamicThresholdSchemeCUDA&) = delete;
    DynamicThresholdSchemeCUDA&
    operator=(const DynamicThresholdSchemeCUDA&) = delete;

    /// Search. Deterministic per (seed, mode, batch_size, toolchain), including
    /// across GPU architectures with that toolchain. batch_size affects how
    /// cumulative Philox stream indices are assigned per stage. In-run cost
    /// and success_h1_cumul are optimistic Monte Carlo estimates (winner's
    /// curse). The returned path is what live pruning uses; it is not
    /// re-scored first.
    void run(SizeType thres_neigh = 10);
    /// Reporting only. Fresh Monte Carlo of one threshold per stage, using
    /// this object's profile and boxcar widths. Does not modify the grid from
    /// run() and is not part of the on-the-fly pipeline. `ntrials` may differ
    /// from the search. Unset `seed` draws a random one; pass a seed different
    /// from the search seed. Deterministic per (seed, thresholds, ntrials).
    std::vector<State> evaluate(std::span<const float> thresholds,
                                SizeType ntrials,
                                std::optional<uint64_t> seed = std::nullopt) const;
    std::string save(const std::string& outdir = "./") const;
    std::vector<float> get_best_path_thresholds(float min_pd = 0.1F) const;
    /// State grid of the last run, [nstages x nthresholds x nprobs].
    std::vector<State> get_states() const;
    std::vector<float> get_thresholds() const;
    std::vector<float> get_probs() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // LOKI_ENABLE_CUDA

} // namespace loki::detection