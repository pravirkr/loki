#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

namespace loki::detection::detail {

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

/** Best-path thresholds (one per stage) after a dynamic threshold run.
 *
 * Among final-stage states with cumulative H1 survival >= \p min_pd, picks
 * minimum cost and backtracks using stored prev links. Layout of \p states must
 * match `(nstages × nthresholds × nprobs)` row-major flattening.
 *
 * Returns an empty vector when no final state meets \p min_pd.
 */
std::vector<float> get_best_path_thresholds(std::span<const State> states,
                                            std::span<const float> thresholds,
                                            std::span<const float> probs,
                                            SizeType nstages,
                                            SizeType nthresholds,
                                            SizeType nprobs,
                                            float min_pd = 0.1F);

} // namespace loki::detection::detail