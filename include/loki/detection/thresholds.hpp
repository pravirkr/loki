#pragma once

#include <span>
#include <string>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::detection {

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

    State gen_next(float threshold,
                   float success_h0,
                   float success_h1,
                   float nbranches) const noexcept;
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
                           bool use_lut_rng      = false,
                           int nthreads          = 1);
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

#ifdef LOKI_ENABLE_CUDA

struct StateD {
    float success_h0{1.0F};
    float success_h1{1.0F};
    float complexity{1.0F};
    float complexity_cumul{std::numeric_limits<float>::infinity()};
    float success_h1_cumul{1.0F};
    float nbranches{1.0F};
    float threshold{-1.0F};
    float cost{1.0F};
    float threshold_prev{-1.0F};
    float success_h1_cumul_prev{1.0F};
    bool is_empty{true};

    __host__ __device__ StateD() = default;

    __host__ __device__ StateD gen_next(float threshold,
                                        float success_h0,
                                        float success_h1,
                                        float nbranches) const noexcept;
    __host__ __device__ State to_state() const;
};

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
                               int device_id         = 0);
    ~DynamicThresholdSchemeCUDA();
    DynamicThresholdSchemeCUDA(DynamicThresholdSchemeCUDA&&) noexcept;
    DynamicThresholdSchemeCUDA&
    operator=(DynamicThresholdSchemeCUDA&&) noexcept;
    DynamicThresholdSchemeCUDA(const DynamicThresholdSchemeCUDA&) = delete;
    DynamicThresholdSchemeCUDA&
    operator=(const DynamicThresholdSchemeCUDA&) = delete;

    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // LOKI_ENABLE_CUDA

} // namespace loki::detection