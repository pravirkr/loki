#include <algorithm> 
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

#include <loki/score.hpp>
#include <loki/thresholds.hpp>

std::vector<size_t> neighbouring_indices(std::span<const float> arr,
                                         float target,
                                         size_t left_size,
                                         size_t right_size) {
    auto it = std::lower_bound(arr.begin(), arr.end(), target);
    if (it == arr.end() || *it != target) {
        return {};
    }
    size_t index = std::distance(arr.begin(), it);
    size_t left  = index >= left_size ? index - left_size : 0;
    size_t right = std::min(arr.size() - 1, index + right_size);

    std::vector<size_t> indices(right - left + 1);
    std::iota(indices.begin(), indices.end(), left);
    return indices;
}

FoldsType simulate_fold(const FoldsType& folds_in,
                        std::span<const float> profile,
                        std::mt19937& rng,
                        float bias_snr,
                        float var_add,
                        size_t ntrials) {
    const auto ntrials_in = folds_in.ntrials;
    const auto nbins      = folds_in.nbins;
    FoldsType folds_out(ntrials, nbins, folds_in.variance + var_add);
    std::normal_distribution<float> dist(0.0, std::sqrt(var_add));

    const auto& fold_in = folds_in.data;
    auto& fold_out      = folds_out.data;
    for (size_t i_trial = 0; i_trial < ntrials; ++i_trial) {
        const auto i_trial_mod = i_trial % ntrials_in;
        const auto offset_in   = i_trial_mod * nbins;
        const auto offset_out  = i_trial * nbins;
        for (size_t j = 0; j < nbins; ++j) {
            fold_out[offset_out + j] =
                fold_in[offset_in + j] + dist(rng) + bias_snr * profile[j];
        }
    }
    return folds_out;
}

std::vector<float> compute_scores(const FoldsType& folds, float ducy_max) {
    const auto widths_max = static_cast<size_t>(
        std::ceil(static_cast<float>(folds.nbins) * ducy_max));
    std::vector<size_t> widths(widths_max);
    std::iota(widths.begin(), widths.end(), 1);

    std::vector<float> folds_snr(folds.ntrials * widths.size());
    loki::snr_2d(std::span(folds.data), folds.ntrials, std::span(widths),
                 std::sqrt(folds.variance), std::span(folds_snr));
    std::vector<float> scores(folds.ntrials);
    for (size_t i = 0; i < folds.ntrials; ++i) {
        const auto snr_idx =
            folds_snr.begin() + static_cast<int>(i * widths.size());
        scores[i] = *std::max_element(
            snr_idx, snr_idx + static_cast<int>(widths.size()));
    }
    return scores;
}

FoldsType prune_folds(const FoldsType& folds_in,
                      std::span<const float> scores,
                      float threshold) {
    const auto ntrials = folds_in.ntrials;
    const auto nbins   = folds_in.nbins;
    if (scores.size() != ntrials) {
        throw std::invalid_argument("Scores size does not match");
    }
    std::vector<size_t> good_scores_idx;
    for (size_t i_trial = 0; i_trial < ntrials; ++i_trial) {
        if (scores[i_trial] > threshold) {
            good_scores_idx.push_back(i_trial);
        }
    }
    const auto ntrials_success = good_scores_idx.size();
    FoldsType folds_out(ntrials_success, folds_in.nbins, folds_in.variance);
    const auto& fold_in = folds_in.data;
    auto& fold_out      = folds_out.data;
    for (size_t i_trial = 0; i_trial < ntrials_success; ++i_trial) {
        const auto offset_in  = good_scores_idx[i_trial] * nbins;
        const auto offset_out = i_trial * nbins;
        std::copy_n(fold_in.begin() + static_cast<int>(offset_in), nbins,
                    fold_out.begin() + static_cast<int>(offset_out));
    }
    return folds_out;
}

bool State::is_empty() const {
    return m_folds_h0.ntrials == 0 || m_folds_h1.ntrials == 0;
}

State State::init(float threshold,
                  float bias_snr,
                  std::span<const float> profile,
                  size_t nbranches,
                  std::mt19937& rng,
                  size_t ntrials,
                  float ducy_max,
                  float var_init) {
    const float var_add = 1.0F;
    const size_t nbins  = profile.size();
    const FoldsType folds_init(ntrials, nbins);
    // Simulate the initial folds (pruning level = 0)
    const auto folds_h0_init =
        simulate_fold(folds_init, profile, rng, 0.0F, var_init, ntrials);
    const auto folds_h1_init =
        simulate_fold(folds_init, profile, rng, bias_snr, var_init, ntrials);
    // Add the next folds to the initial folds (pruning level = 1)
    const auto folds_h0 =
        simulate_fold(folds_h0_init, profile, rng, 0.0F, var_add, ntrials);
    const auto folds_h1 =
        simulate_fold(folds_h1_init, profile, rng, bias_snr, var_add, ntrials);
    const auto scores_h0 = compute_scores(folds_h0, ducy_max);
    const auto folds_h0_pruned =
        prune_folds(folds_h0, std::span(scores_h0), threshold);
    const auto success_h0 = static_cast<float>(folds_h0_pruned.ntrials) /
                            static_cast<float>(folds_h0.ntrials);
    const auto scores_h1 = compute_scores(folds_h1, ducy_max);
    const auto folds_h1_pruned =
        prune_folds(folds_h1, std::span(scores_h1), threshold);
    const auto success_h1 = static_cast<float>(folds_h1_pruned.ntrials) /
                            static_cast<float>(folds_h1.ntrials);
    const auto complexity = static_cast<float>(nbranches) * success_h0;
    const auto backtrack  = std::array{threshold, success_h1};
    const auto state_info = StateInfo{success_h0,     success_h1, complexity,
                                      complexity + 1, success_h1, {backtrack}};
    return State{folds_h0_pruned, folds_h1_pruned, state_info};
}

State State::gen_next(float threshold,
                      float bias_snr,
                      std::span<const float> profile,
                      size_t nbranches,
                      std::mt19937& rng,
                      size_t ntrials,
                      float ducy_max) const {
    const auto folds_h0 =
        simulate_fold(m_folds_h0, profile, rng, 0.0F, 1.0F, ntrials);
    const auto scores_h0 = compute_scores(folds_h0, ducy_max);
    const auto folds_h0_pruned =
        prune_folds(folds_h0, std::span(scores_h0), threshold);
    const auto success_h0 = static_cast<float>(folds_h0_pruned.ntrials) /
                            static_cast<float>(folds_h0.ntrials);
    const auto folds_h1 =
        simulate_fold(m_folds_h1, profile, rng, bias_snr, 1.0F, ntrials);
    const auto scores_h1 = compute_scores(folds_h1, ducy_max);
    const auto folds_h1_pruned =
        prune_folds(folds_h1, std::span(scores_h1), threshold);
    const auto success_h1 = static_cast<float>(folds_h1_pruned.ntrials) /
                            static_cast<float>(folds_h1.ntrials);
    const auto complexity =
        m_state_info.complexity * static_cast<float>(nbranches) * success_h0;
    const auto complexity_cumul = m_state_info.complexity_cumul + complexity;
    const auto success_h1_cumul = m_state_info.success_h1_cumul * success_h1;
    auto backtrack              = m_state_info.backtrack;
    backtrack.push_back({threshold, success_h1_cumul});
    const auto state_info =
        StateInfo{success_h0,       success_h1,       complexity,
                  complexity_cumul, success_h1_cumul, std::move(backtrack)};
    return State{folds_h0_pruned, folds_h1_pruned, state_info};
}

DynamicThresholdScheme::DynamicThresholdScheme(
    const std::vector<size_t>& branching_pattern,
    const std::vector<float>& profile,
    float snr_final,
    float snr_step,
    size_t ntrials,
    size_t nprobs)
    : m_branching_pattern(branching_pattern),
      m_ntrials(ntrials) {
    if (m_branching_pattern.empty()) {
        throw std::invalid_argument("Branching pattern is empty");
    }
    if (m_profile.empty()) {
        throw std::invalid_argument("Profile is empty");
    }
    m_profile     = get_norm_profile(profile);
    m_thresholds  = get_thresholds(snr_final, snr_step);
    m_probs       = get_probs(nprobs);
    m_nbins       = m_profile.size();
    m_nsegments   = m_branching_pattern.size();
    m_nthresholds = m_thresholds.size();
    m_nprobs      = m_probs.size();
    m_bias_snr    = snr_final / static_cast<float>(std::sqrt(m_nsegments));
    m_rng         = std::mt19937(std::random_device{}());

    m_states_in.resize(m_nthresholds * m_nprobs);
    m_states_out.resize(m_nthresholds * m_nprobs);
    m_states_info.resize(m_nsegments * m_nthresholds * m_nprobs);
    init_states();
}

void DynamicThresholdScheme::run(size_t thres_neigh) {
    for (size_t isegment = 1; isegment < m_nsegments; ++isegment) {
        run_segment(isegment, thres_neigh);
        // swap folds
        std::swap(m_states_in, m_states_out);
        std::fill(m_states_out.begin(), m_states_out.end(), std::nullopt);
    }
}

void DynamicThresholdScheme::run_segment(size_t isegment, size_t thres_neigh) {
    std::vector<State> tmp_states;
    tmp_states.reserve(thres_neigh * 2 * m_nprobs);

    const auto segment_offset_prev = (isegment - 1) * m_nthresholds * m_nprobs;
    const auto segment_offset_cur  = isegment * m_nthresholds * m_nprobs;

    for (size_t ithres = 0; ithres < m_nthresholds; ++ithres) {
        tmp_states.clear();
        const auto neighbour_thres_ids = neighbouring_indices(
            m_thresholds, m_thresholds[ithres], thres_neigh, thres_neigh);
        for (size_t jthresh : neighbour_thres_ids) {
            for (size_t iprob = 0; iprob < m_nprobs; ++iprob) {
                const auto& prev_state =
                    m_states_in[segment_offset_prev + jthresh * m_nprobs +
                                iprob];
                if (prev_state.has_value() && !prev_state->is_empty()) {
                    const auto state_next = prev_state->gen_next(
                        m_thresholds[ithres], m_bias_snr, std::span(m_profile),
                        m_branching_pattern[isegment], m_rng, m_ntrials, 0.3F);
                    tmp_states.emplace_back(state_next);
                }
            }
        }
        if (tmp_states.empty()) {
            continue;
        }
        std::vector<const State*> min_complexity_states(m_nprobs, nullptr);
        for (const auto& state : tmp_states) {
            const size_t iprob =
                std::distance(
                    m_probs.begin(),
                    std::upper_bound(m_probs.begin(), m_probs.end(),
                                     state.m_state_info.success_h1_cumul)) -
                1;
            auto& min_state = min_complexity_states[iprob];
            if ((min_state == nullptr) ||
                state.m_state_info.complexity_cumul <
                    min_state->m_state_info.complexity_cumul) {
                min_state = &state;
            }
        }
        const auto thresh_offset_cur = segment_offset_cur + ithres * m_nprobs;
        for (size_t iprob = 0; iprob < m_nprobs; ++iprob) {
            if (min_complexity_states[iprob] != nullptr) {
                m_states_out[thresh_offset_cur + iprob] =
                    *min_complexity_states[iprob];
                m_states_info[thresh_offset_cur + iprob] =
                    min_complexity_states[iprob]->m_state_info;
            }
        }
    }
}

void DynamicThresholdScheme::init_states() {
    for (size_t ithres = 0; ithres < m_nthresholds; ++ithres) {
        const auto threshold = m_thresholds[ithres];
        const auto state =
            State::init(threshold, m_bias_snr, std::span(m_profile),
                        m_branching_pattern[0], m_rng, m_ntrials, 0.3F, 2.0F);
        const auto iprob =
            std::distance(
                m_probs.begin(),
                std::upper_bound(m_probs.begin(), m_probs.end(),
                                 state.m_state_info.success_h1_cumul)) -
            1;
        m_states_in[ithres * m_nprobs + iprob]   = state;
        m_states_info[ithres * m_nprobs + iprob] = state.m_state_info;
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