#pragma once

#include <span>
#include <string>
#include <vector>

#include "loki/loki_types.hpp"

namespace loki::thresholds {

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
                           float ref_ducy,
                           SizeType nbins       = 64,
                           SizeType ntrials     = 1024,
                           SizeType nprobs      = 10,
                           float prob_min       = 0.05F,
                           float snr_final      = 8.0F,
                           SizeType nthresholds = 100,
                           float ducy_max       = 0.3F,
                           float wtsp           = 1.0F,
                           float beam_width     = 0.7F,
                           int nthreads         = 1);
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
    std::vector<State> get_states() const;
    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

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

} // namespace loki::thresholds