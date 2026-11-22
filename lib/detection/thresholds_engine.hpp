#pragma once

/**
 * @file thresholds_engine.hpp
 * @brief Backend engine interface for DynamicThresholdScheme. Internal.
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

namespace loki::detection::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

/// Sizes of the fresh samples of solve(): the signal bank of the selection,
/// repair and report steps (each uses its own, independent bank), and the
/// noise sample of every cost comparison. Resource knobs of the backend, not
/// of the algorithm's design: results depend on them.
struct SolveBudget {
    SizeType nbank{SizeType{1} << 17U};
    SizeType nnoise{SizeType{1} << 15U};
};

/// A terminal path of the search with its in-run estimates.
struct SearchPath {
    std::vector<float> thresholds;
    double pd{0.0}; ///< in-run cumulative Pd
    double c{0.0};  ///< in-run total cost (with the terminal weight)
};

/** One noise evaluation of a schedule with its ancestry: per stage, every
 * trial's score and the slot of the previous stage it continues. Mode
 * "descent" counts the noise survival of any schedule above the simulated one
 * along the stored ancestry.
 */
struct NoiseGenealogy {
    SizeType ntrials{0};
    /// Stage-major (nstages x ntrials): the score of slot t at stage s.
    std::vector<float> scores;
    /// Stage-major: parent[s][t] is the slot of stage s that slot t of stage
    /// s + 1 continues (the last row is unused).
    std::vector<uint32_t> parent;
    /// Fraction of the trials above the simulated schedule, per stage.
    std::vector<double> pass_rate;
};

/** A fresh unpruned signal sample of solve() (simulate_h1_scores), kept
 * where the engine simulated it: the steps of solve() need only per-trial
 * margins and per-stage minima, not the stage x trial matrix.
 */
class SignalBank {
public:
    SignalBank()                             = default;
    virtual ~SignalBank()                    = default;
    SignalBank(const SignalBank&)            = delete;
    SignalBank& operator=(const SignalBank&) = delete;
    SignalBank(SignalBank&&)                 = delete;
    SignalBank& operator=(SignalBank&&)      = delete;
    virtual SizeType size() const            = 0;
    /// Per trial, min over stages of (score - threshold): the trial
    /// survives @p thresholds iff its margin is > 0.
    virtual std::vector<float>
    margins(std::span<const float> thresholds) const = 0;
    /// Per stage, the lowest score of the trials that survive
    /// @p thresholds (the maximum float when none does).
    virtual std::vector<float>
    survivor_minima(std::span<const float> thresholds) const = 0;
};

/// SignalBank of a host score matrix (stage-major nstages x n).
class HostSignalBank final : public SignalBank {
public:
    HostSignalBank(std::vector<float> scores, SizeType n);
    SizeType size() const override { return m_n; }
    std::vector<float>
    margins(std::span<const float> thresholds) const override;
    std::vector<float>
    survivor_minima(std::span<const float> thresholds) const override;

private:
    std::vector<float> m_scores;
    SizeType m_n;
};

class ThresholdsEngine {
protected:
    ThresholdsEngine() = default;

public:
    virtual ~ThresholdsEngine() = default;

    virtual std::vector<SizeType>
    get_current_thresholds_idx(SizeType istage) const          = 0;
    virtual std::vector<float> get_branching_pattern() const   = 0;
    virtual std::vector<float> get_profile() const             = 0;
    virtual std::vector<float> get_thresholds() const          = 0;
    virtual std::vector<float> get_probs() const               = 0;
    virtual SizeType get_nstages() const                       = 0;
    virtual SizeType get_nthresholds() const                   = 0;
    virtual SizeType get_nprobs() const                        = 0;
    virtual std::vector<SizeType> get_box_score_widths() const = 0;
    virtual std::vector<State> get_states() const              = 0;
    /// Memory of the simulation pools: the device pools a GPU engine
    /// allocates on its first run(), or the host pools a simulating CPU
    /// engine allocates at construction (an evaluation-only CPU engine
    /// reports what a simulating one would need, without allocating).
    virtual SizeType run_memory_bytes() const { return 0; }
    virtual void run(SizeType thres_neigh, bool show_progress) = 0;
    /// @p h0_only skips the signal branch: success_h1 is 1 in the states and
    /// the noise states are those of the full evaluation (its branches use
    /// independent streams).
    virtual std::vector<State> evaluate(std::span<const float> thresholds,
                                        SizeType ntrials,
                                        std::optional<uint64_t> seed,
                                        bool h0_only) const   = 0;
    virtual std::string save(const std::string& outdir) const = 0;
    /// Seed of the search (the seed of every derived stream of solve()).
    virtual uint64_t get_seed() const = 0;
    /// Mode "shared": the signal trials the search counted its Pd
    /// on, stage-major (nstages x nh1), after run().
    virtual std::vector<float> shared_h1_bank() const {
        throw std::logic_error(
            "shared_h1_bank: the engine is not in mode 'shared'");
    }
    /// Samples of solve() (the defaults of SolveBudget on every backend).
    virtual SolveBudget solve_budget() const { return {}; }
    /// Frees the simulation pools. States and getters stay valid; run() may
    /// be called again and allocates them anew.
    virtual void release_search_memory() {}
    /// evaluate() with every threshold at lowest(): no trial is pruned or
    /// resampled. Returns the H1 scores, stage-major (nstages x ntrials).
    virtual std::vector<float> simulate_h1_scores(SizeType ntrials,
                                                  uint64_t seed) const = 0;
    /// evaluate()'s noise branch for @p thresholds, keeping the scores and
    /// the ancestry of every trial (mode "descent").
    virtual NoiseGenealogy
    simulate_h0_genealogy(std::span<const float> /*thresholds*/,
                          SizeType /*ntrials*/,
                          uint64_t /*seed*/) const {
        throw std::logic_error(
            "simulate_h0_genealogy: not implemented on this backend");
    }
    /// A signal bank of simulate_h1_scores(n, seed); the default keeps the
    /// matrix on the host.
    virtual std::unique_ptr<SignalBank> make_signal_bank(SizeType n,
                                                         uint64_t seed) const {
        return std::make_unique<HostSignalBank>(simulate_h1_scores(n, seed), n);
    }
    /// Margins of simulate_h1_scores(n, seed) against @p thresholds for every
    /// trial whose running margin stays above @p min_margin; trials that drop
    /// to or below @p min_margin at an earlier stage stop there and hold a
    /// value <= @p min_margin. Exact for every trial with margin > @p
    /// min_margin.
    virtual std::vector<float>
    h1_margins_above(std::span<const float> thresholds,
                     float min_margin,
                     SizeType n,
                     uint64_t seed) const {
        (void)min_margin;
        return make_signal_bank(n, seed)->margins(thresholds);
    }
    /// Trials of simulate_h1_scores(n, seed) that survive every stage of
    /// @p thresholds (margin > 0).
    virtual SizeType count_h1_survivors(std::span<const float> thresholds,
                                        SizeType n,
                                        uint64_t seed) const {
        const auto margins = h1_margins_above(thresholds, 0.0F, n, seed);
        SizeType count     = 0;
        for (const float m : margins) {
            count += m > 0.0F ? 1 : 0;
        }
        return count;
    }

    ThresholdsEngine(const ThresholdsEngine&)            = delete;
    ThresholdsEngine& operator=(const ThresholdsEngine&) = delete;
    ThresholdsEngine(ThresholdsEngine&&)                 = delete;
    ThresholdsEngine& operator=(ThresholdsEngine&&)      = delete;
    virtual std::vector<float> get_best_path_thresholds(float min_pd) const = 0;
};

/// @p simulate = false builds an engine for evaluate() only: no simulation
/// pools and no stage-0 states; run(), save() and the state getters throw.
std::unique_ptr<ThresholdsEngine>
make_thresholds_cpu(std::span<const float> branching_pattern,
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
                    int nthreads,
                    std::optional<uint64_t> seed,
                    bool simulate          = true,
                    std::string_view guess = "analytic",
                    float min_pd           = 0.1F,
                    float w_terminal       = 0.0F,
                    float beam_sigma       = 3.0F,
                    float reach_sigma      = 1.0F);

/// Mode "shared" on the CPU (lib/cpu/thresholds_shared_cpu.cpp):
/// @p nprobs bins from @p min_pd to 1, the guided beam.
std::unique_ptr<ThresholdsEngine>
make_thresholds_shared_cpu(std::span<const float> branching_pattern,
                           float ref_ducy,
                           SizeType nbins,
                           SizeType ntrials,
                           SizeType nprobs,
                           float snr_final,
                           SizeType nthresholds,
                           float ducy_max,
                           float wtsp,
                           float beam_width,
                           SizeType trials_start,
                           int nthreads,
                           std::optional<uint64_t> seed,
                           std::string_view guess,
                           std::string_view mode,
                           float min_pd,
                           float w_terminal,
                           float beam_sigma,
                           float reach_sigma);

std::unique_ptr<ThresholdsEngine>
make_thresholds_gpu(std::span<const float> branching_pattern,
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
                    SizeType batch_size,
                    int device_id,
                    std::optional<uint64_t> seed,
                    std::string_view guess = "analytic",
                    float min_pd           = 0.1F,
                    float w_terminal       = 0.0F,
                    float beam_sigma       = 3.0F,
                    float reach_sigma      = 1.0F);

// The facade's backend dispatch, for internal callers that need the engine
// itself (the EP planner checks run_memory_bytes() against its budget).
std::unique_ptr<ThresholdsEngine>
make_thresholds_engine(std::span<const float> branching_pattern,
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
                       Exec exec);

} // namespace loki::detection::detail
