#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <span>
#include <vector>

struct FoldsType {
    // ntrials x nbins
    std::vector<float> data;
    float variance;
    size_t ntrials;
    size_t nbins;

    FoldsType(size_t ntrials, size_t nbins, float variance = 0.0F)
        : data(ntrials * nbins),
          variance(variance),
          ntrials(ntrials),
          nbins(nbins) {}
};

struct StateInfo {
    float success_h0;
    float success_h1;
    float complexity;
    float complexity_cumul;
    float success_h1_cumul;
    std::vector<std::array<float, 2>> backtrack;
    float cost() const { return complexity_cumul / success_h1_cumul; }
};

class State {
public:
    FoldsType m_folds_h0;
    FoldsType m_folds_h1;
    StateInfo m_state_info;
    bool is_empty() const;

    State gen_next(float threshold,
                   float bias_snr,
                   std::span<const float> profile,
                   size_t nbranches,
                   std::mt19937& rng,
                   size_t ntrials,
                   float ducy_max = 0.3F) const;

    static State init(float threshold,
                      float bias_snr,
                      std::span<const float> profile,
                      size_t nbranches,
                      std::mt19937& rng,
                      size_t ntrials,
                      float ducy_max = 0.3F,
                      float var_init = 2.0F);
};

std::vector<size_t> neighbouring_indices(std::span<const float> arr,
                                         float target,
                                         size_t left_size,
                                         size_t right_size);

/**
 * @brief Simulate folded profiles by adding signal + noise to the template.
 *
 * @param folds_in Folded data with shape (ntrials, nbins).
 * @param profile Normalized template profile.
 * @param rng Random number generator.
 * @param bias_snr Bias signal-to-noise ratio.
 * @param var_add Variance to add to the noise.
 * @param ntrials Number of trials to simulate.
 * @return FoldsType Folded data with shape (ntrials, nbins).
 */
FoldsType simulate_folds(const FoldsType& folds_in,
                         std::span<const float> profile,
                         std::mt19937& rng,
                         float bias_snr = 0.0F,
                         float var_add  = 1.0F,
                         size_t ntrials = 1024);

std::vector<float> compute_scores(const FoldsType& folds,
                                  float ducy_max = 0.3F);

FoldsType prune_folds(const FoldsType& folds_in,
                      std::span<const float> scores,
                      float threshold);

std::pair<FoldsType, State>
gen_next_using_threshold(const FoldsType& folds_in_h0,
                         const FoldsType& folds_in_h1,
                         float threshold,
                         float bias_snr,
                         std::span<const float> profile,
                         size_t nbranches,
                         std::mt19937& rng,
                         size_t ntrials,
                         float ducy_max = 0.3F);

class DynamicThresholdScheme {
public:
    std::vector<size_t> m_branching_pattern;
    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    size_t m_ntrials;

    size_t m_nbins;
    size_t m_nsegments;
    size_t m_nthresholds;
    size_t m_nprobs;
    float m_bias_snr;
    std::vector<std::optional<State>> m_states_in;
    std::vector<std::optional<State>> m_states_out;
    std::vector<std::optional<StateInfo>> m_states_info;

    DynamicThresholdScheme(const std::vector<size_t>& branching_pattern,
                           const std::vector<float>& profile,
                           float snr_final = 8.0F,
                           float snr_step  = 0.1F,
                           size_t ntrials  = 1024,
                           size_t nprobs   = 10);
    void run(size_t thres_neigh = 10);

    State get_best_state() const;

private:
    std::mt19937 m_rng;

    static std::vector<float>
    get_norm_profile(const std::vector<float>& profile);
    static std::vector<float> get_thresholds(float snr_final, float snr_step);
    static std::vector<float> get_probs(size_t nprobs);
    void init_states();
    void run_segment(size_t isegment, size_t thres_neigh = 10);
};