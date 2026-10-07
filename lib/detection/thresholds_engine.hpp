#pragma once

/**
 * @file thresholds_engine.hpp
 * @brief Backend engine interface for DynamicThresholdScheme. Internal.
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

namespace loki::detection::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

class ThresholdsEngine {
protected:
    ThresholdsEngine() = default;

public:
    virtual ~ThresholdsEngine() = default;

    virtual std::vector<SizeType>
    get_current_thresholds_idx(SizeType istage) const                       = 0;
    virtual std::vector<float> get_branching_pattern() const                = 0;
    virtual std::vector<float> get_profile() const                          = 0;
    virtual std::vector<float> get_thresholds() const                       = 0;
    virtual std::vector<float> get_probs() const                            = 0;
    virtual SizeType get_nstages() const                                    = 0;
    virtual SizeType get_nthresholds() const                                = 0;
    virtual SizeType get_nprobs() const                                     = 0;
    virtual std::vector<SizeType> get_box_score_widths() const              = 0;
    virtual std::vector<State> get_states() const                           = 0;
    virtual void run(SizeType thres_neigh)                                  = 0;
    virtual std::vector<State> evaluate(std::span<const float> thresholds,
                                        SizeType ntrials,
                                        std::optional<uint64_t> seed) const = 0;
    virtual std::string save(const std::string& outdir) const               = 0;

    ThresholdsEngine(const ThresholdsEngine&)            = delete;
    ThresholdsEngine& operator=(const ThresholdsEngine&) = delete;
    ThresholdsEngine(ThresholdsEngine&&)                 = delete;
    ThresholdsEngine& operator=(ThresholdsEngine&&)      = delete;
    virtual std::vector<float> get_best_path_thresholds(float min_pd) const = 0;
};

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
                    std::optional<uint64_t> seed);

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
                    std::optional<uint64_t> seed);

} // namespace loki::detection::detail
