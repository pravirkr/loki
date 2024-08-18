#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <random>
#include <stdexcept>
#include <vector>

#include <highfive/highfive.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <loki/loki_types.hpp>
#include <loki/math.hpp>
#include <loki/score.hpp>
#include <loki/thresholds.hpp>
#include <loki/utils.hpp>

FoldVector::FoldVector(SizeType ntrials, SizeType nbins, float variance)
    : data(ntrials * nbins),
      variance(variance),
      ntrials(ntrials),
      nbins(nbins) {}

FoldVector::FoldVector(const std::vector<float>& data,
                       float variance,
                       SizeType ntrials,
                       SizeType nbins)
    : data(data),
      variance(variance),
      ntrials(ntrials),
      nbins(nbins) {}

std::vector<float> FoldVector::get_norm() const {
    std::vector<float> norm_data(data.size());
    const auto std = std::sqrt(variance);
    for (SizeType i = 0; i < ntrials; ++i) {
        for (SizeType j = 0; j < nbins; ++j) {
            norm_data[i * nbins + j] = data[i * nbins + j] / std;
        }
    }
    return norm_data;
}

FoldsType::FoldsType(SizeType ntrials, SizeType nbins, float variance)
    : folds_h0(ntrials, nbins, variance),
      folds_h1(ntrials, nbins, variance) {}

bool FoldsType::is_empty() const {
    return folds_h0.data.empty() || folds_h1.data.empty();
}

State::State(float success_h0,
             float success_h1,
             float complexity,
             float complexity_cumul,
             float success_h1_cumul,
             float nbranches,
             std::vector<std::array<float, 2>> backtrack)
    : success_h0(success_h0),
      success_h1(success_h1),
      complexity(complexity),
      complexity_cumul(complexity_cumul),
      success_h1_cumul(success_h1_cumul),
      nbranches(nbranches),
      backtrack(std::move(backtrack)) {}

float State::cost() const { return complexity_cumul / success_h1_cumul; }

State State::gen_next_state(float threshold,
                            float success_h0,
                            float success_h1,
                            float nbranches) const {
    const auto complexity       = this->complexity * nbranches * success_h0;
    const auto complexity_cumul = this->complexity_cumul + complexity;
    const auto success_h1_cumul = this->success_h1_cumul * success_h1;
    auto backtrack              = this->backtrack;
    backtrack.push_back({threshold, success_h1_cumul});
    return State{success_h0,          success_h1,       complexity,
                 complexity_cumul,    success_h1_cumul, nbranches,
                 std::move(backtrack)};
}

std::tuple<State, FoldsType>
State::gen_next_using_thresh(const FoldsType& fold_state,
                             float threshold,
                             float nbranches,
                             float bias_snr,
                             std::span<const float> profile,
                             ThreadSafeRNG& rng,
                             float var_add,
                             SizeType ntrials,
                             float ducy_max) const {
    const auto folds_h0  = simulate_folds(fold_state.folds_h0, profile, rng,
                                          0.0F, var_add, ntrials);
    const auto scores_h0 = compute_scores(folds_h0, ducy_max);
    const auto folds_h0_pruned = prune_folds(folds_h0, scores_h0, threshold);
    const auto success_h0      = static_cast<float>(folds_h0_pruned.ntrials) /
                            static_cast<float>(folds_h0.ntrials);
    const auto folds_h1  = simulate_folds(fold_state.folds_h1, profile, rng,
                                          bias_snr, var_add, ntrials);
    const auto scores_h1 = compute_scores(folds_h1, ducy_max);
    const auto folds_h1_pruned = prune_folds(folds_h1, scores_h1, threshold);
    const auto success_h1      = static_cast<float>(folds_h1_pruned.ntrials) /
                            static_cast<float>(folds_h1.ntrials);
    const auto next_state =
        gen_next_state(threshold, success_h0, success_h1, nbranches);
    return {next_state, FoldsType{folds_h0_pruned, folds_h1_pruned}};
}

std::tuple<State, FoldsType>
State::gen_next_using_surv_prob(const FoldsType& fold_state,
                                float surv_prob_h0,
                                float nbranches,
                                float bias_snr,
                                std::span<const float> profile,
                                ThreadSafeRNG& rng,
                                float var_add,
                                SizeType ntrials,
                                float ducy_max) const {
    const auto folds_h0  = simulate_folds(fold_state.folds_h0, profile, rng,
                                          0.0F, var_add, ntrials);
    const auto scores_h0 = compute_scores(folds_h0, ducy_max);
    const auto threshold_h0 =
        compute_threshold_survival(scores_h0, surv_prob_h0);
    const auto folds_h0_pruned = prune_folds(folds_h0, scores_h0, threshold_h0);
    const auto success_h0      = static_cast<float>(folds_h0_pruned.ntrials) /
                            static_cast<float>(folds_h0.ntrials);
    const auto folds_h1  = simulate_folds(fold_state.folds_h1, profile, rng,
                                          bias_snr, var_add, ntrials);
    const auto scores_h1 = compute_scores(folds_h1, ducy_max);
    const auto folds_h1_pruned = prune_folds(folds_h1, scores_h1, threshold_h0);
    const auto success_h1      = static_cast<float>(folds_h1_pruned.ntrials) /
                            static_cast<float>(folds_h1.ntrials);
    const auto next_state =
        gen_next_state(threshold_h0, success_h0, success_h1, nbranches);
    return {next_state, FoldsType{folds_h0_pruned, folds_h1_pruned}};
}

DynamicThresholdScheme::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    std::span<const float> profile,
    SizeType nparams,
    float snr_final,
    SizeType nthresholds,
    SizeType ntrials,
    SizeType nprobs,
    float ducy_max,
    float beam_width)
    : m_branching_pattern(branching_pattern.begin(), branching_pattern.end()),
      m_nparams(nparams),
      m_ntrials(ntrials),
      m_ducy_max(ducy_max),
      m_beam_width(beam_width),
      m_rng(std::random_device{}()) {
    if (m_branching_pattern.empty()) {
        throw std::invalid_argument("Branching pattern is empty");
    }
    m_profile     = compute_norm_profile(profile);
    m_thresholds  = compute_thresholds(0.1F, snr_final, nthresholds);
    m_probs       = compute_probs(nprobs);
    m_nbins       = m_profile.size();
    m_nstages     = m_branching_pattern.size();
    m_nthresholds = m_thresholds.size();
    m_nprobs      = m_probs.size();
    m_bias_snr    = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
    m_guess_path  = guess_scheme(m_nstages, snr_final, m_nparams);
    m_folds_in.resize(m_nthresholds * m_nprobs);
    m_folds_out.resize(m_nthresholds * m_nprobs);
    m_states.resize(m_nstages * m_nthresholds * m_nprobs);
    init_states();
}

std::vector<SizeType>
DynamicThresholdScheme::get_current_thresholds_idx(SizeType istage) const {
    const auto guess       = m_guess_path[istage];
    const auto half_extent = m_beam_width;
    const auto lower_bound = std::max(0.0F, guess - half_extent);
    const auto upper_bound = std::min(m_thresholds.back(), guess + half_extent);

    std::vector<SizeType> result;
    for (SizeType i = 0; i < m_thresholds.size(); ++i) {
        if (m_thresholds[i] >= lower_bound && m_thresholds[i] <= upper_bound) {
            result.push_back(i);
        }
    }

    return result;
}

std::vector<float> DynamicThresholdScheme::get_branching_pattern() const {
    return m_branching_pattern;
}
std::vector<float> DynamicThresholdScheme::get_profile() const {
    return m_profile;
}
std::vector<float> DynamicThresholdScheme::get_thresholds() const {
    return m_thresholds;
}
std::vector<float> DynamicThresholdScheme::get_probs() const { return m_probs; }
SizeType DynamicThresholdScheme::get_nstages() const { return m_nstages; }
SizeType DynamicThresholdScheme::get_nthresholds() const {
    return m_nthresholds;
}
SizeType DynamicThresholdScheme::get_nprobs() const { return m_nprobs; }
std::vector<std::optional<State>> DynamicThresholdScheme::get_states() const {
    return m_states;
}

void DynamicThresholdScheme::run(SizeType thres_neigh) {
    spdlog::info("Running dynamic threshold scheme");
    indicators::show_console_cursor(false);
    indicators::ProgressBar bar{
        indicators::option::PrefixText{"Computing"},
        indicators::option::ShowPercentage(true),
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
    };
    for (SizeType istage = 1; istage < m_nstages; ++istage) {
        run_segment(istage, thres_neigh);
        // swap folds
        std::swap(m_folds_in, m_folds_out);
        std::fill(m_folds_out.begin(), m_folds_out.end(), std::nullopt);
        const auto progress = static_cast<float>(istage) /
                              static_cast<float>(m_nstages - 1) * 100.0F;
        bar.set_progress(static_cast<SizeType>(progress));
    }
    indicators::show_console_cursor(true);
}

// Create a compound type for State
HighFive::CompoundType create_compound_state() {
    return {{"success_h0", HighFive::create_datatype<float>()},
            {"success_h1", HighFive::create_datatype<float>()},
            {"complexity", HighFive::create_datatype<float>()},
            {"complexity_cumul", HighFive::create_datatype<float>()},
            {"success_h1_cumul", HighFive::create_datatype<float>()},
            {"nbranches", HighFive::create_datatype<float>()}};
}
HIGHFIVE_REGISTER_TYPE(SaveState, create_compound_state)

std::string DynamicThresholdScheme::save(const std::string& outdir) const {
    const std::filesystem::path filebase =
        std::format("dynscheme_nstages_{}_nthresh_{}_nprobs_{}_"
                    "ntrials_{}_snr_{:.1f}_beam_{:.1f}.h5",
                    m_nstages, m_nthresholds, m_nprobs, m_ntrials,
                    m_thresholds.back(), m_beam_width);
    const std::filesystem::path filepath =
        std::filesystem::path(outdir) / filebase;
    HighFive::File file(filepath, HighFive::File::Overwrite);
    // Save simple attributes
    file.createAttribute("ntrials", m_ntrials);
    file.createAttribute("snr_final", m_thresholds.back());
    file.createAttribute("ducy_max", m_ducy_max);
    file.createAttribute("beam_width", m_beam_width);

    // Create dataset creation property list and enable compression
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking(std::vector<hsize_t>{1024}));
    props.add(HighFive::Deflate(9));

    // Save arrays
    file.createDataSet("branching_pattern", m_branching_pattern);
    file.createDataSet("profile", m_profile);
    file.createDataSet("thresholds", m_thresholds);
    file.createDataSet("probs", m_probs);
    file.createDataSet("guess_path", m_guess_path);

    std::vector<SaveState> state_data;
    std::vector<std::array<float, 2>> backtrack_data;
    std::vector<SizeType> backtrack_lengths;
    std::vector<std::array<SizeType, 3>> grid_indices;

    for (SizeType istage = 0; istage < m_nstages; ++istage) {
        for (SizeType ithres = 0; ithres < m_nthresholds; ++ithres) {
            for (SizeType iprob = 0; iprob < m_nprobs; ++iprob) {
                const auto idx = istage * m_nthresholds * m_nprobs +
                                 ithres * m_nprobs + iprob;
                const auto& state = m_states[idx];
                if (state.has_value()) {
                    state_data.push_back(
                        {state->success_h0, state->success_h1,
                         state->complexity, state->complexity_cumul,
                         state->success_h1_cumul, state->nbranches});
                    backtrack_data.insert(backtrack_data.end(),
                                          state->backtrack.begin(),
                                          state->backtrack.end());
                    backtrack_lengths.push_back(state->backtrack.size());
                    grid_indices.push_back({istage, ithres, iprob});
                }
            }
        }
    }
    file.createDataSet("states", state_data, props);
    file.createDataSet("backtrack", backtrack_data);
    file.createDataSet("backtrack_lengths", backtrack_lengths);
    file.createDataSet("grid_indices", grid_indices);
    return filepath.string();
}

void DynamicThresholdScheme::run_segment(SizeType istage,
                                         SizeType thres_neigh) {
    const auto beam_idx_cur      = get_current_thresholds_idx(istage);
    const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
    const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
    const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;

#pragma omp parallel
    {
        std::vector<std::tuple<State, FoldsType>> candidates;
        candidates.reserve(beam_idx_prev.size() * m_nprobs);

#pragma omp for schedule(dynamic)
        for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
            const auto ithres = beam_idx_cur[i];
            // Find nearest neighbors in the previous beam
            const auto neighbour_beam_indices = loki::find_neighbouring_indices(
                beam_idx_prev, ithres, thres_neigh);

            SizeType icand = 0;
            for (SizeType jthresh : neighbour_beam_indices) {
                for (SizeType kprob = 0; kprob < m_nprobs; ++kprob) {
                    const auto prev_idx = jthresh * m_nprobs + kprob;
                    const auto prev_state =
                        m_states[stage_offset_prev + prev_idx];
                    const auto prev_fold_state = m_folds_in[prev_idx];
                    if (prev_fold_state.has_value() &&
                        !prev_fold_state->is_empty()) {
                        const auto [cur_state, cur_fold_state] =
                            prev_state->gen_next_using_thresh(
                                *prev_fold_state, m_thresholds[ithres],
                                m_branching_pattern[istage], m_bias_snr,
                                m_profile, m_rng, 1.0F, m_ntrials, m_ducy_max);
                        if (icand < candidates.size()) {
                            candidates[icand] = {cur_state, cur_fold_state};
                        } else {
                            candidates.emplace_back(cur_state, cur_fold_state);
                        }
                        ++icand;
                    }
                }
            }
            // Find the state with the minimum complexity for each probability
            for (SizeType j = 0; j < icand; ++j) {
                const auto& state      = std::get<0>(candidates[j]);
                const auto& fold_state = std::get<1>(candidates[j]);
                const auto iprob =
                    find_bin_index(m_probs, state.success_h1_cumul);
                const auto cur_idx       = ithres * m_nprobs + iprob;
                const auto cur_state_idx = stage_offset_cur + cur_idx;
                auto& existing_state     = m_states[cur_state_idx];
                if (!existing_state.has_value() ||
                    state.complexity_cumul < existing_state->complexity_cumul) {
                    existing_state       = state;
                    m_folds_out[cur_idx] = fold_state;
                }
            }
        }
    }
#pragma omp barrier
}

void DynamicThresholdScheme::init_states() {
    const float var_init = 2.0F;
    const FoldVector folds_init(m_ntrials, m_nbins);
    // Simulate the initial folds (pruning level = 0)
    const auto folds_h0 =
        simulate_folds(folds_init, m_profile, m_rng, 0.0F, var_init, m_ntrials);
    const auto folds_h1 = simulate_folds(folds_init, m_profile, m_rng,
                                         m_bias_snr, var_init, m_ntrials);
    const State state   = State();
    const FoldsType fold_state{folds_h0, folds_h1};
    const auto thresholds_idx = get_current_thresholds_idx(0);
    for (SizeType ithres : thresholds_idx) {
        const auto [cur_state, cur_fold_state] = state.gen_next_using_thresh(
            fold_state, m_thresholds[ithres], m_branching_pattern[0],
            m_bias_snr, m_profile, m_rng, 1.0F, m_ntrials, m_ducy_max);
        const auto iprob = find_bin_index(m_probs, cur_state.success_h1_cumul);
        m_states[ithres * m_nprobs + iprob]   = cur_state;
        m_folds_in[ithres * m_nprobs + iprob] = cur_fold_state;
    }
}

std::vector<float> DynamicThresholdScheme::compute_thresholds(
    float snr_start, float snr_final, SizeType nthresholds) {
    std::vector<float> thresholds(nthresholds);
    const auto snr_step =
        (snr_final - snr_start) / static_cast<float>(nthresholds - 1);
    for (SizeType i = 0; i < nthresholds; ++i) {
        thresholds[i] = static_cast<float>(i) * snr_step + snr_start;
    }
    return thresholds;
}

std::vector<float> DynamicThresholdScheme::compute_probs(SizeType nprobs) {
    std::vector<float> probs(nprobs);
    for (SizeType i = 0; i < nprobs; ++i) {
        float value = std::exp(-3 + (3.0F * static_cast<float>(i)) /
                                        (static_cast<float>(nprobs) - 1));
        probs[i]    = 1 - value;
    }
    std::reverse(probs.begin(), probs.end());
    return probs;
}

std::vector<float> DynamicThresholdScheme::bound_scheme(SizeType nstages,
                                                        float snr_bound) {
    const auto nsegments = nstages + 1;
    std::vector<float> thresholds(nstages);
    for (SizeType i = 0; i < nstages; ++i) {
        thresholds[i] = std::sqrt(static_cast<float>((i + 2)) * snr_bound *
                                  snr_bound / static_cast<float>(nsegments));
    }
    return thresholds;
}

std::vector<float> DynamicThresholdScheme::trials_scheme(
    SizeType nstages, SizeType nparams, SizeType trials_start) {
    SizeType complexity_scaling = 0;
    for (SizeType i = 1; i <= nparams; ++i) {
        complexity_scaling += i;
    }
    std::vector<float> result(nstages);
    for (SizeType i = 0; i < nstages; ++i) {
        const auto trials =
            static_cast<float>(trials_start) *
            static_cast<float>(std::pow(i + 2, complexity_scaling));
        result[i] = loki::norm_isf(1 / trials);
    }
    return result;
}

std::vector<float> DynamicThresholdScheme::guess_scheme(SizeType nstages,
                                                        float snr_bound,
                                                        SizeType nparams,
                                                        SizeType trials_start) {
    const auto thresholds_bound = bound_scheme(nstages, snr_bound);
    const auto thresholds_trials =
        trials_scheme(nstages, nparams, trials_start);
    std::vector<float> result(nstages);
    std::transform(
        thresholds_bound.begin(), thresholds_bound.end(),
        thresholds_trials.begin(), result.begin(),
        [](float bound, float trials) { return std::min(bound, trials); });
    return result;
}

std::vector<float> compute_norm_profile(std::span<const float> profile) {
    std::vector<float> profile_norm(profile.begin(), profile.end());
    float sum_of_squares = std::inner_product(profile.begin(), profile.end(),
                                              profile.begin(), 0.0F);
    float norm           = std::sqrt(sum_of_squares);
    for (float& elem : profile_norm) {
        elem /= norm;
    }
    return profile_norm;
}

FoldVector simulate_folds(const FoldVector& folds_in,
                          std::span<const float> profile,
                          ThreadSafeRNG& rng,
                          float bias_snr,
                          float var_add,
                          SizeType ntrials_min) {
    const auto ntrials_in = folds_in.ntrials;
    const auto nbins      = folds_in.nbins;

    // Create even folds by copying the input folds
    const auto repeat_factor = static_cast<SizeType>(std::ceil(
        static_cast<float>(ntrials_min) / static_cast<float>(ntrials_in)));
    const auto ntrials       = repeat_factor * ntrials_in;
    FoldVector folds_out(ntrials, nbins, folds_in.variance + var_add);
    for (SizeType i = 0; i < repeat_factor; ++i) {
        std::copy_n(folds_in.data.begin(), nbins * ntrials_in,
                    folds_out.data.begin() +
                        static_cast<int>(i * nbins * ntrials_in));
    }
    std::normal_distribution<float> dist(0.0F, std::sqrt(var_add));
    for (SizeType i = 0; i < ntrials * nbins; ++i) {
        folds_out.data[i] += rng.generate(dist) + bias_snr * profile[i % nbins];
    }
    return folds_out;
}

std::vector<float> compute_scores(const FoldVector& folds, float ducy_max) {
    const auto widths     = loki::generate_width_trials(static_cast<SizeType>(
        std::ceil(static_cast<float>(folds.nbins) * ducy_max)));
    const auto folds_norm = folds.get_norm();
    std::vector<float> folds_snr(folds.ntrials * widths.size());
    loki::snr_2d(folds_norm, folds.ntrials, widths, folds_snr);
    std::vector<float> scores(folds.ntrials);
    for (SizeType i = 0; i < folds.ntrials; ++i) {
        const auto snr_idx =
            folds_snr.begin() + static_cast<int>(i * widths.size());
        scores[i] = *std::max_element(
            snr_idx, snr_idx + static_cast<int>(widths.size()));
    }
    return scores;
}

float compute_threshold_survival(std::span<const float> scores,
                                 float survive_prob) {
    if (scores.empty()) {
        throw std::invalid_argument("Scores array is empty");
    }
    auto n_surviving =
        static_cast<SizeType>(survive_prob * static_cast<float>(scores.size()));
    n_surviving = std::max(static_cast<SizeType>(1),
                           std::min(n_surviving, scores.size()));

    std::vector<float> top_scores(n_surviving);
    std::partial_sort_copy(scores.begin(), scores.end(), top_scores.begin(),
                           top_scores.end(), std::greater<>());

    return top_scores.back();
}

FoldVector prune_folds(const FoldVector& folds_in,
                       std::span<const float> scores,
                       float threshold) {
    const auto ntrials = folds_in.ntrials;
    const auto nbins   = folds_in.nbins;
    if (scores.size() != ntrials) {
        throw std::invalid_argument("Scores size does not match");
    }
    std::vector<SizeType> good_scores_idx;
    for (SizeType i_trial = 0; i_trial < ntrials; ++i_trial) {
        if (scores[i_trial] > threshold) {
            good_scores_idx.push_back(i_trial);
        }
    }
    const auto ntrials_success = good_scores_idx.size();
    FoldVector folds_out(ntrials_success, folds_in.nbins, folds_in.variance);
    const auto& fold_in = folds_in.data;
    auto& fold_out      = folds_out.data;
    for (SizeType i_trial = 0; i_trial < ntrials_success; ++i_trial) {
        const auto offset_in  = good_scores_idx[i_trial] * nbins;
        const auto offset_out = i_trial * nbins;
        std::copy_n(fold_in.begin() + static_cast<int>(offset_in), nbins,
                    fold_out.begin() + static_cast<int>(offset_out));
    }
    return folds_out;
}

SizeType find_bin_index(std::span<const float> bins, float value) {
    auto it = std::upper_bound(bins.begin(), bins.end(), value);
    return std::distance(bins.begin(), it) - 1;
}

std::vector<std::optional<State>>
evaluate_threshold_scheme(std::span<const float> thresholds,
                          std::span<const float> branching_pattern,
                          std::span<const float> profile,
                          SizeType ntrials_min,
                          float snr_final,
                          float ducy_max) {
    if (thresholds.empty() || branching_pattern.empty() || profile.empty()) {
        throw std::invalid_argument("Input arrays cannot be empty");
    }
    if (thresholds.size() != branching_pattern.size()) {
        throw std::invalid_argument(
            "Number of thresholds must match the number of stages");
    }

    const auto nstages      = branching_pattern.size();
    const auto nbins        = profile.size();
    const auto profile_norm = compute_norm_profile(profile);
    const auto bias_snr =
        snr_final / std::sqrt(static_cast<float>(nstages + 1));
    const float var_init = 2.0F;
    std::vector<std::optional<State>> states(nstages);
    std::vector<std::optional<FoldsType>> fold_states(nstages);

    ThreadSafeRNG rng(std::random_device{}());
    const FoldVector folds_init(ntrials_min, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 = simulate_folds(folds_init, profile_norm, rng, 0.0F,
                                   var_init, ntrials_min);
    auto folds_h1 = simulate_folds(folds_init, profile_norm, rng, bias_snr,
                                   var_init, ntrials_min);
    const State initial_state;
    FoldsType initial_fold_state{std::move(folds_h0), std::move(folds_h1)};
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto& prev_state =
            (istage == 0) ? initial_state : *states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : *fold_states[istage - 1];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info("Path not viable, stopping at stage {}", istage);
            break;
        }
        auto [cur_state, cur_fold_state] = prev_state.gen_next_using_thresh(
            prev_fold_state, thresholds[istage], branching_pattern[istage],
            bias_snr, profile_norm, rng, 1.0F, ntrials_min, ducy_max);
        states[istage]      = std::move(cur_state);
        fold_states[istage] = std::move(cur_fold_state);
    }
    return states;
}

std::vector<std::optional<State>>
determine_threshold_scheme(std::span<const float> survive_probs,
                           std::span<const float> branching_pattern,
                           std::span<const float> profile,
                           SizeType ntrials_min,
                           float snr_final,
                           float ducy_max) {
    if (survive_probs.empty() || branching_pattern.empty() || profile.empty()) {
        throw std::invalid_argument("Input arrays cannot be empty");
    }
    if (survive_probs.size() != branching_pattern.size()) {
        throw std::invalid_argument(
            "Number of survival probabilities must match the number of stages");
    }

    const auto nstages      = branching_pattern.size();
    const auto nbins        = profile.size();
    const auto profile_norm = compute_norm_profile(profile);
    const auto bias_snr =
        snr_final / std::sqrt(static_cast<float>(nstages + 1));
    const float var_init = 2.0F;
    std::vector<std::optional<State>> states(nstages);
    std::vector<std::optional<FoldsType>> fold_states(nstages);

    ThreadSafeRNG rng(std::random_device{}());
    const FoldVector folds_init(ntrials_min, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 = simulate_folds(folds_init, profile_norm, rng, 0.0F,
                                   var_init, ntrials_min);
    auto folds_h1 = simulate_folds(folds_init, profile_norm, rng, bias_snr,
                                   var_init, ntrials_min);
    const State initial_state;
    const FoldsType initial_fold_state{std::move(folds_h0),
                                       std::move(folds_h1)};
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto& prev_state =
            (istage == 0) ? initial_state : *states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : *fold_states[istage - 1];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info("Path not viable, stopping at stage {}", istage);
            break;
        }
        const auto [cur_state, cur_fold_state] =
            prev_state.gen_next_using_surv_prob(
                prev_fold_state, survive_probs[istage],
                branching_pattern[istage], bias_snr, profile_norm, rng, 1.0F,
                ntrials_min, ducy_max);
        states[istage]      = cur_state;
        fold_states[istage] = cur_fold_state;
    }
    return states;
}