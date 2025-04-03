#pragma once

#include <cstddef>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <omp.h>

#include "loki/loki_types.hpp"

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
    explicit ThreadSafeRNG(unsigned int base_seed = std::random_device{}()) {
        const int max_threads = omp_get_max_threads();
        if (max_threads <= 0) {
            throw std::runtime_error("OpenMP: Invalid thread count");
        }
        m_rngs.reserve(max_threads);
        std::seed_seq seed_seq{base_seed};
        std::vector<std::uint32_t> seeds(max_threads);
        seed_seq.generate(seeds.begin(), seeds.end());
        for (int i = 0; i < max_threads; ++i) {
            // Each thread gets a unique seed by offsetting the base seed.
            m_rngs.emplace_back(seeds[i]);
        }
    }
    ThreadSafeRNG(const ThreadSafeRNG&)            = delete;
    ThreadSafeRNG& operator=(const ThreadSafeRNG&) = delete;
    ThreadSafeRNG(ThreadSafeRNG&&)                 = delete;
    ThreadSafeRNG& operator=(ThreadSafeRNG&&)      = delete;
    ~ThreadSafeRNG()                               = default;

    unsigned int operator()() {
        const int tid = omp_get_thread_num();
        assert(tid < static_cast<int>(m_rngs.size()));
        return m_rngs[tid]();
    }

    std::mt19937& get_engine_for_current_thread() {
        const int tid = omp_get_thread_num();
        if (tid < 0 || tid >= static_cast<int>(m_rngs.size())) {
            throw std::out_of_range("Invalid OpenMP thread id");
        }
        return m_rngs[tid];
    }

    template <class Distribution>
    typename Distribution::result_type generate(Distribution& dist) noexcept {
        return dist(get_engine_for_current_thread());
    }

    // Fills a range with random numbers using a provided distribution.
    template <class Distribution, typename T>
    void generate_range(Distribution& dist, std::span<T> range) {
        std::generate(range.begin(), range.end(),
                      [&]() { return dist(get_engine_for_current_thread()); });
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

struct State {
    float success_h0{1.0F};
    float success_h1{1.0F};
    float complexity{1.0F};
    float complexity_cumul{1.0F};
    float success_h1_cumul{1.0F};
    float nbranches{1.0F};
    float threshold{-1.0F};
    float cost{1.0F};
    float threshold_prev{-1.0F};
    float success_h1_cumul_prev{1.0F};
    bool is_empty{true};

    State() = default;
};

class DynamicThresholdScheme {
public:
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           std::span<const float> profile,
                           SizeType ntrials     = 1024,
                           SizeType nprobs      = 10,
                           float prob_min       = 0.05F,
                           float snr_final      = 8.0F,
                           SizeType nthresholds = 100,
                           float ducy_max       = 0.3F,
                           float wtsp           = 1.0F,
                           float beam_width     = 0.7F,
                           SizeType nthreads    = 1);

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const;
    std::vector<float> get_branching_pattern() const;
    std::vector<float> get_profile() const;
    std::vector<float> get_thresholds() const;
    std::vector<float> get_probs() const;
    SizeType get_nstages() const;
    SizeType get_nthresholds() const;
    SizeType get_nprobs() const;
    std::vector<State> get_states() const;
    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

private:
    std::vector<float> m_branching_pattern;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    ThreadSafeRNG m_rng;
    SizeType m_nthreads;

    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    float m_bias_snr;
    std::vector<float> m_guess_path;
    std::vector<std::optional<FoldsType>> m_folds_in;
    std::vector<std::optional<FoldsType>> m_folds_out;
    std::vector<State> m_states;

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

std::vector<State> evaluate_scheme(std::span<const float> thresholds,
                                   std::span<const float> branching_pattern,
                                   std::span<const float> profile,
                                   SizeType ntrials = 1024,
                                   float snr_final  = 8.0F,
                                   float ducy_max   = 0.3F,
                                   float wtsp       = 1.0F);

std::vector<State> determine_scheme(std::span<const float> survive_probs,
                                    std::span<const float> branching_pattern,
                                    std::span<const float> profile,
                                    SizeType ntrials = 1024,
                                    float snr_final  = 8.0F,
                                    float ducy_max   = 0.3F,
                                    float wtsp       = 1.0F);