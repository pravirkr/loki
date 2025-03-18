#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <omp.h>

#include <loki/loki_types.hpp>

class SimpleRNG {
private:
    std::mt19937 m_rng;

public:
    explicit SimpleRNG(unsigned int seed) : m_rng(seed) {}

    unsigned int operator()() { return m_rng(); }

    template <class Distribution>
    typename Distribution::result_type generate(Distribution& dist) {
        return dist(m_rng);
    }
};

class ThreadSafeRNG {
private:
    std::vector<std::mt19937> m_rngs;

public:
    explicit ThreadSafeRNG(unsigned int seed) {
        int max_threads = omp_get_max_threads();
        m_rngs.reserve(max_threads);
        for (int i = 0; i < max_threads; ++i) {
            m_rngs.emplace_back(seed + i); // Each thread gets a different seed
        }
    }

    unsigned int operator()() {
        int thread_num = omp_get_thread_num();
        return m_rngs[thread_num]();
    }

    template <class Distribution>
    typename Distribution::result_type generate(Distribution& dist) {
        int thread_num = omp_get_thread_num();
        return dist(m_rngs[thread_num]);
    }
};

struct FoldVector {
    std::vector<float> data;
    float variance;
    SizeType ntrials;
    SizeType nbins;

    FoldVector(const std::vector<float>& data,
               float variance,
               SizeType ntrials,
               SizeType nbins);

    FoldVector(SizeType ntrials, SizeType nbins, float variance = 0.0F);

    std::vector<float> get_norm() const;
};

struct FoldsType {
    FoldVector folds_h0;
    FoldVector folds_h1;

    FoldsType(SizeType ntrials, SizeType nbins, float variance = 0.0F);
    template <typename FoldVector0, typename FoldVector1>
    FoldsType(FoldVector0&& folds_h0, FoldVector1&& folds_h1)
        : folds_h0(std::forward<FoldVector0>(folds_h0)),
          folds_h1(std::forward<FoldVector1>(folds_h1)) {}

    bool is_empty() const;
};

struct SaveState {
    float success_h0;
    float success_h1;
    float complexity;
    float complexity_cumul;
    float success_h1_cumul;
    float nbranches;
};

class State {
public:
    float success_h0;
    float success_h1;
    float complexity;
    float complexity_cumul;
    float success_h1_cumul;
    float nbranches;
    std::vector<std::array<float, 2>> backtrack;

    explicit State(float success_h0                            = 0.0,
                   float success_h1                            = 0.0,
                   float complexity                            = 1.0,
                   float complexity_cumul                      = 1.0,
                   float success_h1_cumul                      = 1.0,
                   float nbranches                             = 1.0,
                   std::vector<std::array<float, 2>> backtrack = {});

    float cost() const;
    State gen_next_state(float threshold,
                         float success_h0,
                         float success_h1,
                         float nbranches) const;

    std::tuple<State, FoldsType>
    gen_next_using_thresh(const FoldsType& fold_state,
                          float threshold,
                          float nbranches,
                          float bias_snr,
                          std::span<const float> profile,
                          ThreadSafeRNG& rng,
                          float var_add    = 1.0F,
                          SizeType ntrials = 1024,
                          float ducy_max   = 0.3F) const;

    std::tuple<State, FoldsType>
    gen_next_using_surv_prob(const FoldsType& fold_state,
                             float surv_prob_h0,
                             float nbranches,
                             float bias_snr,
                             std::span<const float> profile,
                             ThreadSafeRNG& rng,
                             float var_add    = 1.0F,
                             SizeType ntrials = 1024,
                             float ducy_max   = 0.3F) const;
};

class DynamicThresholdScheme {
public:
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           std::span<const float> profile,
                           float snr_final      = 8.0F,
                           SizeType nthresholds = 100,
                           SizeType ntrials     = 1024,
                           SizeType nprobs      = 10,
                           float prob_min       = 0.05F,
                           float ducy_max       = 0.3F,
                           float beam_width     = 0.7F);

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const;
    std::vector<float> get_branching_pattern() const;
    std::vector<float> get_profile() const;
    std::vector<float> get_thresholds() const;
    std::vector<float> get_probs() const;
    SizeType get_nstages() const;
    SizeType get_nthresholds() const;
    SizeType get_nprobs() const;
    std::vector<std::optional<State>> get_states() const;
    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

private:
    std::vector<float> m_branching_pattern;
    std::vector<float> m_profile;
    SizeType m_nthresholds;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_beam_width;

    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nprobs;
    float m_bias_snr;
    std::vector<float> m_guess_path;
    std::vector<std::optional<FoldsType>> m_folds_in;
    std::vector<std::optional<FoldsType>> m_folds_out;
    std::vector<std::optional<State>> m_states;
    ThreadSafeRNG m_rng;

    void run_segment(SizeType istage, SizeType thres_neigh = 10);
    void init_states();
    static std::vector<float>
    compute_thresholds(float snr_start, float snr_final, SizeType nthresholds);
    static std::vector<float> compute_probs(SizeType nprobs,
                                            float prob_min = 0.05F);
    static std::vector<float> compute_probs_linear(SizeType nprobs,
                                                float prob_min = 0.05F);
    static std::vector<float> bound_scheme(SizeType nstages, float snr_bound);
    static std::vector<float>
    trials_scheme(std::span<const float> branching_pattern,
                  SizeType trials_start = 1,
                  float min_trials      = 1E10F);
    static std::vector<float>
    guess_scheme(SizeType nstages,
                 float snr_bound,
                 std::span<const float> branching_pattern,
                 SizeType trials_start = 1,
                 float min_trials      = 1E10F);
};

std::vector<float> compute_norm_profile(std::span<const float> profile);

FoldVector simulate_folds(const FoldVector& folds_in,
                          std::span<const float> profile,
                          ThreadSafeRNG& rng,
                          float bias_snr       = 0.0F,
                          float var_add        = 1.0F,
                          SizeType ntrials_min = 1024);

std::vector<float> compute_scores(const FoldVector& folds,
                                  float ducy_max = 0.3F);

float compute_threshold_survival(std::span<const float> scores,
                                 float survive_prob);

FoldVector prune_folds(const FoldVector& folds_in,
                       std::span<const float> scores,
                       float threshold);

IndexType find_bin_index(std::span<const float> bins, float value);

std::vector<std::optional<State>>
evaluate_threshold_scheme(std::span<const float> thresholds,
                          std::span<const float> branching_pattern,
                          std::span<const float> profile,
                          SizeType ntrials = 1024,
                          float snr_final  = 8.0F,
                          float ducy_max   = 0.3F);

std::vector<std::optional<State>>
determine_threshold_scheme(std::span<const float> survive_probs,
                           std::span<const float> branching_pattern,
                           std::span<const float> profile,
                           SizeType ntrials = 1024,
                           float snr_final  = 8.0F,
                           float ducy_max   = 0.3F);