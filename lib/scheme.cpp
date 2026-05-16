#include "loki/scheme.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "loki/detection/thresholds.hpp"
#include "loki/exceptions.hpp"
#include "loki/math.hpp"
#include "loki/utils.hpp"

namespace loki::detection::detail {

std::vector<float>
compute_thresholds(float snr_start, float snr_final, SizeType nthresholds) {
    std::vector<float> thresholds(nthresholds);
    const auto snr_step =
        (snr_final - snr_start) / static_cast<float>(nthresholds - 1);
    for (SizeType i = 0; i < nthresholds; ++i) {
        thresholds[i] = (static_cast<float>(i) * snr_step) + snr_start;
    }
    return thresholds;
}

std::vector<float> compute_probs(SizeType nprobs, float prob_min) {
    error_check::check_greater(nprobs, 1,
                               "Number of probabilities must be > 1");
    error_check::check_less(prob_min, 1.0F, "Probability must be < 1");
    error_check::check_greater(prob_min, 0.0F, "Probability must be > 0");
    std::vector<float> probs(nprobs);
    const float log_prob_min = std::log10(prob_min);
    const float step = (0.0F - log_prob_min) / static_cast<float>(nprobs - 1);
    for (SizeType i = 0; i < nprobs; ++i) {
        probs[i] =
            std::pow(10.0F, log_prob_min + (step * static_cast<float>(i)));
    }
    return probs;
}

std::vector<float> compute_probs_linear(SizeType nprobs, float prob_min) {
    error_check::check_greater(nprobs, 1,
                               "Number of probabilities must be > 1");
    error_check::check_less(prob_min, 1.0F, "Probability must be < 1");
    error_check::check_greater(prob_min, 0.0F, "Probability must be > 0");
    std::vector<float> probs(nprobs);
    float step = (1.0F - prob_min) / static_cast<float>(nprobs - 1);

    for (SizeType i = 0; i < nprobs; ++i) {
        probs[i] = prob_min + (step * static_cast<float>(i));
    }

    return probs;
}

std::vector<float> bound_scheme(SizeType nstages, float snr_bound) {
    const auto nsegments = nstages + 1;
    std::vector<float> thresholds(nstages);
    for (SizeType i = 0; i < nstages; ++i) {
        thresholds[i] = std::sqrt(static_cast<float>((i + 2)) * snr_bound *
                                  snr_bound / static_cast<float>(nsegments));
    }
    return thresholds;
}

std::vector<float> trials_scheme(std::span<const float> branching_pattern,
                                 SizeType trials_start,
                                 float min_trials) {
    const auto nstages = branching_pattern.size();
    std::vector<float> result(nstages);
    // trials = np.cumprod(branching_pattern) * trials_start
    auto log2_trials = std::log2(static_cast<float>(trials_start));
    for (SizeType i = 0; i < nstages; ++i) {
        log2_trials += std::log2(branching_pattern[i]);
        const auto trials           = std::exp2(log2_trials);
        const auto effective_trials = std::max(trials, min_trials);
        result[i] = loki::math::norm_isf(1 / effective_trials);
    }
    return result;
}

std::vector<float> guess_scheme(SizeType nstages,
                                float snr_bound,
                                std::span<const float> branching_pattern,
                                SizeType trials_start,
                                float min_trials) {
    const auto thresholds_bound = bound_scheme(nstages, snr_bound);
    const auto thresholds_trials =
        trials_scheme(branching_pattern, trials_start, min_trials);
    std::vector<float> result(nstages);
    std::ranges::transform(
        thresholds_bound, thresholds_trials, result.begin(),
        [](float bound, float trials) { return std::min(bound, trials); });
    return result;
}

std::vector<float> get_best_path_thresholds(std::span<const State> states,
                                            std::span<const float> thresholds,
                                            std::span<const float> probs,
                                            SizeType nstages,
                                            SizeType nthresholds,
                                            SizeType nprobs,
                                            float min_pd) {
    error_check::check_greater(nstages, 0, "nstages must be positive");
    if (thresholds.empty() || probs.empty()) {
        throw std::invalid_argument(
            "get_best_path_thresholds: thresholds and probs must be "
            "non-empty");
    }
    error_check::check_equal(
        states.size(), nstages * nthresholds * nprobs,
        "states span size does not match nstages × nthresholds × nprobs");

    const SizeType stage_last = (nstages - 1) * nthresholds * nprobs;
    const State* best         = nullptr;
    float best_cost           = std::numeric_limits<float>::max();

    for (SizeType i = 0; i < nthresholds * nprobs; ++i) {
        const State& s = states[stage_last + i];
        if (s.is_empty) {
            continue;
        }
        if (s.success_h1_cumul < min_pd) {
            continue;
        }
        if (s.cost < best_cost) {
            best_cost = s.cost;
            best      = &s;
        }
    }

    if (best == nullptr) {
        return {};
    }

    std::vector<const State*> backward;
    backward.reserve(nstages);
    backward.push_back(best);

    float prev_threshold        = best->threshold_prev;
    float prev_success_h1_cumul = best->success_h1_cumul_prev;

    for (SizeType k = 0; k + 1 < nstages; ++k) {
        const SizeType istage = nstages - 2 - k;
        const SizeType ithres =
            utils::find_nearest_index(thresholds, prev_threshold);
        const IndexType iprob =
            utils::find_lower_bin_index(probs, prev_success_h1_cumul);
        error_check::check_range(iprob, nprobs,
                                 "probability bin index out of range");
        const auto flat_idx = (istage * nthresholds * nprobs) +
                              (ithres * nprobs) + static_cast<SizeType>(iprob);
        const State& prev_state = states[flat_idx];
        if (prev_state.is_empty) {
            throw std::runtime_error(
                "get_best_path_thresholds: backtracking failed (empty state) "
                "at stage " +
                std::to_string(istage));
        }
        backward.push_back(&prev_state);
        prev_threshold        = prev_state.threshold_prev;
        prev_success_h1_cumul = prev_state.success_h1_cumul_prev;
    }

    std::vector<float> out;
    out.reserve(nstages);
    for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
        out.push_back((*it)->threshold);
    }
    return out;
}

} // namespace loki::detection::detail