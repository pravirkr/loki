#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <vector>

struct FoldsType {
    // shape: (nthresholds, nprobs, ntrials, nbins)
    std::vector<float> data;
    float variance{};
    size_t nthresholds;
    size_t nprobs;
    size_t ntrials;
    size_t nbins;

    FoldsType(size_t nthresholds, size_t nprobs, size_t ntrials, size_t nbins)
        : data(nthresholds * nprobs * ntrials * nbins, 0.0F),
          nthresholds(nthresholds),
          nprobs(nprobs),
          ntrials(ntrials),
          nbins(nbins) {}

    size_t get_trial_pos(size_t ithreshold, size_t iprob) const {
        return ithreshold * nprobs * ntrials * nbins + iprob * ntrials * nbins;
    }

    std::vector<size_t> get_non_nan_trial_indices(size_t ithreshold,
                                                  size_t iprob) const {
        std::vector<size_t> indices;
        const auto trials_pos = get_trial_pos(ithreshold, iprob);
        for (size_t i = 0; i < ntrials; ++i) {
            if (!std::isnan(data[trials_pos + i * nbins])) {
                indices.push_back(i);
            }
        }
        return indices;
    }
};

class State {
public:
    float m_success_h0;
    float m_success_h1;
    float m_complexity;
    float m_complexity_cumul;
    float m_success_h1_cumul;
    std::vector<std::array<float, 2>> m_backtrack;

    float cost() const;
    bool is_empty() const;
    State gen_next_using_threshold(
        const FoldsType& folds_in_h0, FoldsType& folds_out_h0,
        const FoldsType& folds_in_h1, FoldsType& folds_out_h1, float threshold,
        float bias_snr, const std::vector<float>& profile, size_t nbranches,
        std::mt19937& gen, size_t ithreshold, size_t iprob) const;

    static init_stat

private:
    static size_t pick_random_element(const std::vector<size_t>& vec,
                                      std::mt19937& gen);

    void simulate_folds(const FoldsType& folds_in, FoldsType& folds_out,
                        const std::vector<float>& profile, std::mt19937& gen,
                        float bias_snr, float var_add, size_t ithreshold,
                        size_t iprob) const;

    static std::vector<size_t> measure_success(const FoldsType& folds,
                                               float snr_threshold,
                                               size_t ithreshold, size_t iprob);
};

class DynamicThresholdScheme {
public:
    std::vector<size_t> m_branching_pattern;
    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    size_t m_ntrials;
    float m_bias_snr;

    FoldsType m_folds_in_h0;
    FoldsType m_folds_in_h1;
    FoldsType m_folds_out_h0;
    FoldsType m_folds_out_h1;

    DynamicThresholdScheme(const std::vector<size_t>& branching_pattern,
                           const std::vector<float>& profile,
                           float snr_final = 8.0F, float snr_step = 0.1F,
                           size_t ntrials = 1024, size_t nprobs = 10);
    void run(size_t nsteps);

    State get_best_state() const;

private:
    static std::vector<float>
    get_norm_profile(const std::vector<float>& profile);
    static std::vector<float> get_thresholds(float snr_final, float snr_step);
    static std::vector<float> get_probs(size_t nprobs);
    void init_states();
};