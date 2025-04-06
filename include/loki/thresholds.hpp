#pragma once

#include <concepts>
#include <span>
#include <string>
#include <vector>

#include "loki/loki_types.hpp"

namespace loki {

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

template <backend::ExecutionBackend Backend = backend::CPU>
class DynamicThresholdScheme {
public:
    // Specialized constructors for different execution policies
    template <std::same_as<backend::CPU> P = Backend>
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
#ifdef LOKI_ENABLE_CUDA
    template <std::same_as<backend::CUDA> P = Backend>
    DynamicThresholdScheme(std::span<const float> branching_pattern,
                           std::span<const float> profile,
                           SizeType ntrials,
                           SizeType nprobs,
                           float prob_min,
                           float snr_final,
                           SizeType nthresholds,
                           float ducy_max,
                           float wtsp,
                           float beam_width,
                           int device_id);
#endif
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

// Type aliases for convenience
using DynamicThresholdSchemeCPU = DynamicThresholdScheme<backend::CPU>;
#ifdef LOKI_ENABLE_CUDA
using DynamicThresholdSchemeCUDA = DynamicThresholdScheme<backend::CUDA>;
#endif

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
} // namespace loki