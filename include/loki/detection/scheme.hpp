#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::detection {

std::vector<float>
compute_thresholds(float snr_start, float snr_final, SizeType nthresholds);

std::vector<float> compute_probs(SizeType nprobs, float prob_min = 0.05F);

std::vector<float> compute_probs_linear(SizeType nprobs,
                                        float prob_min = 0.05F);

std::vector<float> bound_scheme(SizeType nstages, float snr_bound);

std::vector<float> trials_scheme(std::span<const float> branching_pattern,
                                 SizeType trials_start = 1,
                                 float min_trials      = 1E10F);

std::vector<float> guess_scheme(SizeType nstages,
                                float snr_bound,
                                std::span<const float> branching_pattern,
                                SizeType trials_start = 1,
                                float min_trials      = 1E10F);

} // namespace loki::detection