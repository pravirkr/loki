#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::detection {

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
        const auto nleaves_next = this->complexity * nbranches;
        const auto nleaves_surv = nleaves_next * success_h0;
        const auto complexity_cumul_next =
            this->complexity_cumul + nleaves_next;
        const auto success_h1_cumul_next = this->success_h1_cumul * success_h1;
        const auto cost_next = complexity_cumul_next / success_h1_cumul_next;

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
                               SizeType batch_size   = 256,
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