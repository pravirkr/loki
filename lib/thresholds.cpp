#include <cmath>
#include <cstddef>
#include <span>

#include <loki/score.hpp>
#include <loki/thresholds.hpp>

void State::simulate_folds(const FoldsType& folds_in, FoldsType& folds_out,
                           const std::vector<float>& profile, std::mt19937& gen,
                           float bias_snr, float var_add, size_t ithreshold,
                           size_t iprob) const {
    const auto trials_pos_in   = folds_in.get_trial_pos(ithreshold, iprob);
    const auto trials_pos_out  = folds_out.get_trial_pos(ithreshold, iprob);
    const auto trials_size_in  = folds_in.ntrials * folds_in.nbins;
    const auto trials_size_out = folds_out.ntrials * folds_out.nbins;
    const auto& folds =
        std::span(folds_in.data).subspan(trials_pos_in, trials_size_in);
    auto folds_sim =
        std::span(folds_out.data).subspan(trials_pos_out, trials_size_out);

    const size_t nbins = folds_in.nbins;
    const auto non_nan_trial_indices =
        folds_in.get_non_nan_trial_indices(ithreshold, iprob);
    for (size_t i_trial = 0; i_trial < folds_out.ntrials; ++i_trial) {
        size_t i_trial_in =
            std::isnan(folds[i_trial * nbins])
                ? pick_random_element(non_nan_trial_indices, gen)
                : i_trial;
        std::copy_n(folds.begin() + static_cast<int>(i_trial_in * nbins), nbins,
                    folds_sim.begin() + static_cast<int>(i_trial * nbins));
    }
    std::normal_distribution<float> dist(0.0, std::sqrt(var_add));
    for (size_t i = 0; i < folds_out.ntrials; ++i) {
        for (size_t j = 0; j < nbins; ++j) {
            folds_sim[i * nbins + j] += dist(gen) + bias_snr * profile[j];
        }
    }
    folds_out.variance = folds_in.variance + var_add;
}

std::vector<size_t> State::measure_success(const FoldsType& folds,
                                           float snr_threshold,
                                           size_t ithreshold, size_t iprob) {
    const auto trials_pos_in  = folds.get_trial_pos(ithreshold, iprob);
    const auto trials_size_in = folds.ntrials * folds.nbins;
    const auto& folds_h =
        std::span(folds.data).subspan(trials_pos_in, trials_size_in);
    std::vector<size_t> template_widths(folds.nbins / 2);
    std::iota(template_widths.begin(), template_widths.end(), 1);
    const size_t ntemplates = template_widths.size();
    std::vector<float> folds_snr(folds.ntrials * ntemplates);
    loki::snr_2d(folds_h, folds.ntrials, std::span(template_widths),
                 std::sqrt(folds.variance), std::span(folds_snr));
    std::vector<float> scores_arr(folds.ntrials);
    for (size_t i = 0; i < folds.ntrials; ++i) {
        const auto snr_idx = folds_snr.begin() + i * ntemplates;
        scores_arr[i]      = *std::max_element(snr_idx, snr_idx + ntemplates);
    }
    std::vector<size_t> good_scores_idx;
    for (size_t i = 0; i < scores_arr.size(); ++i) {
        if (scores_arr[i] > snr_threshold) {
            good_scores_idx.push_back(i);
        }
    }
    return good_scores_idx;
}

size_t State::pick_random_element(const std::vector<size_t>& vec,
                                  std::mt19937& gen) {
    if (vec.empty()) {
        throw std::invalid_argument("Vector is empty");
    }
    std::uniform_int_distribution<size_t> dist(0, vec.size() - 1);
    size_t random_index = dist(gen);
    // Return the element at the random index
    return vec[random_index];
}

State State::gen_next_using_threshold(
    const FoldsType& folds_in_h0, FoldsType& folds_out_h0,
    const FoldsType& folds_in_h1, FoldsType& folds_out_h1, float threshold,
    float bias_snr, const std::vector<float>& profile, size_t nbranches,
    std::mt19937& gen, size_t ithreshold, size_t iprob) const {
    if (is_empty()) {
        throw std::invalid_argument("State is empty");
    }
    simulate_folds(folds_in_h0, folds_out_h0, profile, gen, bias_snr, 1.0F,
                   ithreshold, iprob);
    simulate_folds(folds_in_h1, folds_out_h1, profile, gen, bias_snr, 1.0F,
                   ithreshold, iprob);
    const auto good_scores_idx_h0 =
        measure_success(folds_out_h0, threshold, 0, iprob);
    const auto good_scores_idx_h1 =
        measure_success(folds_out_h1, threshold, 1, iprob);
    const auto success_h0 = static_cast<float>(good_scores_idx_h0.size()) /
                            static_cast<float>(folds_in_h0.ntrials);
    const auto success_h1 = static_cast<float>(good_scores_idx_h1.size()) /
                            static_cast<float>(folds_in_h1.ntrials);
    const auto complexity =
        m_complexity * static_cast<float>(nbranches) * success_h0;
    float complexity_cumul = m_complexity_cumul + complexity;
    float success_h1_cumul = m_success_h1_cumul * success_h1;
    auto backtrack         = m_backtrack;
    backtrack.push_back({threshold, success_h1_cumul});
    return State{success_h0,       success_h1,       complexity,
                 complexity_cumul, success_h1_cumul, backtrack};
}

DynamicThresholdScheme::DynamicThresholdScheme(
    const std::vector<size_t>& branching_pattern,
    const std::vector<float>& profile, float snr_final, float snr_step,
    size_t ntrials, size_t nprobs)
    : m_branching_pattern(branching_pattern),
      m_profile(get_norm_profile(profile)),
      m_thresholds(get_thresholds(snr_final, snr_step)),
      m_probs(get_probs(nprobs)),
      m_ntrials(ntrials),
      m_bias_snr(snr_final /
                 static_cast<float>(std::sqrt(m_branching_pattern.size()))),
      m_folds_in_h0(
          FoldsType(m_thresholds.size(), nprobs, ntrials, m_profile.size())),
      m_folds_in_h1(
          FoldsType(m_thresholds.size(), nprobs, ntrials, m_profile.size())),
      m_folds_out_h0(
          FoldsType(m_thresholds.size(), nprobs, ntrials, m_profile.size())),
      m_folds_out_h1(
          FoldsType(m_thresholds.size(), nprobs, ntrials, m_profile.size())) {
    init_states();
}

void DynamicThresholdScheme::init_states() {
    m_states.clear();
    m_states.reserve(m_branching_pattern.size());
    m_states.push_back(State{0.0F, 0.0F, 1.0F, 0.0F, 1.0F, {}});
    for (size_t i = 1; i < m_branching_pattern.size(); ++i) {
        m_states.push_back(m_states[i - 1]);
    }
}

std::vector<float>
DynamicThresholdScheme::get_norm_profile(const std::vector<float>& profile) {
    std::vector<float> profile_norm = profile;
    float sum_of_squares = std::inner_product(profile.begin(), profile.end(),
                                              profile.begin(), 0.0F);
    float norm           = std::sqrt(sum_of_squares);
    for (float& elem : profile_norm) {
        elem /= norm;
    }
    return profile_norm;
}

std::vector<float> DynamicThresholdScheme::get_thresholds(float snr_final,
                                                          float snr_step) {
    int num_thresholds = static_cast<int>(snr_final / snr_step);
    std::vector<float> thresholds(num_thresholds);
    for (int i = 0; i < num_thresholds; ++i) {
        thresholds[i] = static_cast<float>(i) * snr_step + snr_step;
    }
    return thresholds;
}

std::vector<float> DynamicThresholdScheme::get_probs(size_t nprobs) {
    std::vector<float> probs(nprobs);
    for (size_t i = 0; i < nprobs; ++i) {
        float value = std::exp(-3 + (3.0F * static_cast<float>(i)) /
                                        (static_cast<float>(nprobs) - 1));
        probs[i]    = 1 - value;
    }
    std::reverse(probs.begin(), probs.end());
    return probs;
}