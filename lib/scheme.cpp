#include "loki/detection/scheme.hpp"

#include "loki/math.hpp"

namespace loki::detection {

std::vector<float>
compute_thresholds(float snr_start, float snr_final, SizeType nthresholds) {
    std::vector<float> thresholds(nthresholds);
    const auto snr_step =
        (snr_final - snr_start) / static_cast<float>(nthresholds - 1);
    for (SizeType i = 0; i < nthresholds; ++i) {
        thresholds[i] = static_cast<float>(i) * snr_step + snr_start;
    }
    return thresholds;
}

std::vector<float> compute_probs(SizeType nprobs, float prob_min) {
    if (nprobs <= 1) {
        throw std::invalid_argument("Number of probabilities must be > 1");
    }
    if (prob_min <= 0.0F || prob_min >= 1.0F) {
        throw std::invalid_argument("Probability must be in the range (0, 1)");
    }
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
    if (nprobs <= 1) {
        throw std::invalid_argument("Number of probabilities must be > 1");
    }
    std::vector<float> probs(nprobs);
    float step = (1.0F - prob_min) / static_cast<float>(nprobs - 1);

    for (SizeType i = 0; i < nprobs; ++i) {
        probs[i] = prob_min + step * static_cast<float>(i);
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

} // namespace loki::detection