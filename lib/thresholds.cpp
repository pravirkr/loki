#include "loki/thresholds.hpp"

#include <algorithm>
#include <cmath>
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

#include "loki/loki_types.hpp"
#include "loki/math.hpp"
#include "loki/score.hpp"
#include "loki/utils.hpp"

namespace loki {
namespace {
struct FoldVector {
    std::vector<float> data;
    float variance;
    SizeType ntrials;
    SizeType nbins;

    FoldVector(const std::vector<float>& data,
               float variance,
               SizeType ntrials,
               SizeType nbins)
        : data(data),
          variance(variance),
          ntrials(ntrials),
          nbins(nbins) {}

    FoldVector(SizeType ntrials, SizeType nbins, float variance = 0.0F)
        : data(ntrials * nbins),
          variance(variance),
          ntrials(ntrials),
          nbins(nbins) {}

    std::vector<float> get_norm() const {
        std::vector<float> norm_data(data.size());
        const auto std = std::sqrt(variance);
        for (SizeType i = 0; i < ntrials; ++i) {
            for (SizeType j = 0; j < nbins; ++j) {
                const auto idx = (i * nbins) + j;
                norm_data[idx] = data[idx] / std;
            }
        }
        return norm_data;
    }
};

struct FoldsType {
    FoldVector folds_h0;
    FoldVector folds_h1;

    FoldsType(SizeType ntrials, SizeType nbins, float variance = 0.0F)
        : folds_h0(ntrials, nbins, variance),
          folds_h1(ntrials, nbins, variance) {}

    template <typename FoldVector0, typename FoldVector1>
    FoldsType(FoldVector0&& folds_h0, FoldVector1&& folds_h1)
        : folds_h0(std::forward<FoldVector0>(folds_h0)),
          folds_h1(std::forward<FoldVector1>(folds_h1)) {}

    bool is_empty() const {
        return folds_h0.data.empty() || folds_h1.data.empty();
    }
};

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

IndexType find_bin_index(std::span<const float> bins, float value) {
    auto it = std::ranges::upper_bound(bins, value);
    return std::distance(bins.begin(), it) - 1;
}

FoldVector simulate_folds(const FoldVector& folds_in,
                          std::span<const float> profile,
                          utils::ThreadSafeRNG& rng,
                          float bias_snr       = 0.0F,
                          float var_add        = 1.0F,
                          SizeType ntrials_min = 1024) {
    const auto ntrials_in = folds_in.ntrials;
    const auto nbins      = folds_in.nbins;
    if (ntrials_in == 0) {
        throw std::invalid_argument("No trials in the input folds");
    }
    // Create even folds by copying the input folds
    const auto repeat_factor = static_cast<SizeType>(std::ceil(
        static_cast<float>(ntrials_min) / static_cast<float>(ntrials_in)));
    const auto ntrials       = repeat_factor * ntrials_in;
    FoldVector folds_out(ntrials, nbins, folds_in.variance + var_add);

    std::vector<float> template_scaled(nbins);
    std::ranges::transform(profile, template_scaled.begin(),
                           [bias_snr](float x) { return x * bias_snr; });

    std::vector<float> noise(ntrials * nbins);
    std::normal_distribution<float> dist(0.0F, std::sqrt(var_add));
    rng.generate_range(dist, std::span<float>(noise));

    for (SizeType i = 0; i < ntrials; ++i) {
        const auto orig_trial   = i % ntrials_in;
        const auto trial_offset = i * nbins;
        const auto orig_offset  = orig_trial * nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            folds_out.data[trial_offset + j] = folds_in.data[orig_offset + j] +
                                               noise[trial_offset + j] +
                                               template_scaled[j];
        }
    }
    return folds_out;
}

std::vector<float>
compute_scores(const FoldVector& folds, float ducy_max, float wtsp) {
    const auto max_width = static_cast<SizeType>(
        std::ceil(static_cast<float>(folds.nbins) * ducy_max));
    const auto widths     = score::generate_width_trials(max_width, wtsp);
    const auto folds_norm = folds.get_norm();
    std::vector<float> folds_snr(folds.ntrials * widths.size());
    score::snr_2d(folds_norm, folds.ntrials, widths, folds_snr);
    std::vector<float> scores(folds.ntrials);
    for (SizeType i = 0; i < folds.ntrials; ++i) {
        const auto offset = i * widths.size();
        auto* const begin = folds_snr.data() + offset;
        scores[i]         = *std::max_element(begin, begin + widths.size());
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
    std::ranges::partial_sort_copy(scores, top_scores, std::greater<>());
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

State gen_next_state(const State& state_cur,
                     float threshold,
                     float success_h0,
                     float success_h1,
                     float nbranches) {
    const auto nleaves_next     = state_cur.complexity * nbranches;
    const auto nleaves_surv     = nleaves_next * success_h0;
    const auto complexity_cumul = state_cur.complexity_cumul + nleaves_next;
    const auto success_h1_cumul = state_cur.success_h1_cumul * success_h1;

    // Create a new state struct
    State state_next;
    state_next.success_h0       = success_h0;
    state_next.success_h1       = success_h1;
    state_next.complexity       = nleaves_surv;
    state_next.complexity_cumul = complexity_cumul;
    state_next.success_h1_cumul = success_h1_cumul;
    state_next.nbranches        = nbranches;
    state_next.threshold        = threshold;
    state_next.cost             = complexity_cumul / success_h1_cumul;
    state_next.is_empty         = false;
    // For backtracking
    state_next.threshold_prev        = state_cur.threshold;
    state_next.success_h1_cumul_prev = state_cur.success_h1_cumul;
    return state_next;
}

std::tuple<State, FoldsType>
gen_next_using_thresh(const State& state_cur,
                      const FoldsType& folds_cur,
                      float threshold,
                      float nbranches,
                      float bias_snr,
                      std::span<const float> profile,
                      utils::ThreadSafeRNG& rng,
                      float var_add    = 1.0F,
                      SizeType ntrials = 1024,
                      float ducy_max   = 0.3F,
                      float wtsp       = 1.0F) {
    const auto folds_h0 = simulate_folds(folds_cur.folds_h0, profile, rng, 0.0F,
                                         var_add, ntrials);
    const auto scores_h0       = compute_scores(folds_h0, ducy_max, wtsp);
    const auto folds_h0_pruned = prune_folds(folds_h0, scores_h0, threshold);
    const auto success_h0      = static_cast<float>(folds_h0_pruned.ntrials) /
                            static_cast<float>(folds_h0.ntrials);
    const auto folds_h1  = simulate_folds(folds_cur.folds_h1, profile, rng,
                                          bias_snr, var_add, ntrials);
    const auto scores_h1 = compute_scores(folds_h1, ducy_max, wtsp);
    const auto folds_h1_pruned = prune_folds(folds_h1, scores_h1, threshold);
    const auto success_h1      = static_cast<float>(folds_h1_pruned.ntrials) /
                            static_cast<float>(folds_h1.ntrials);
    const auto state_next =
        gen_next_state(state_cur, threshold, success_h0, success_h1, nbranches);
    return {state_next, FoldsType{folds_h0_pruned, folds_h1_pruned}};
}

std::tuple<State, FoldsType>
gen_next_using_surv_prob(const State& state_cur,
                         const FoldsType& folds_cur,
                         float surv_prob_h0,
                         float nbranches,
                         float bias_snr,
                         std::span<const float> profile,
                         utils::ThreadSafeRNG& rng,
                         float var_add    = 1.0F,
                         SizeType ntrials = 1024,
                         float ducy_max   = 0.3F,
                         float wtsp       = 1.0F) {
    const auto folds_h0 = simulate_folds(folds_cur.folds_h0, profile, rng, 0.0F,
                                         var_add, ntrials);
    const auto scores_h0 = compute_scores(folds_h0, ducy_max, wtsp);
    const auto threshold_h0 =
        compute_threshold_survival(scores_h0, surv_prob_h0);
    const auto folds_h0_pruned = prune_folds(folds_h0, scores_h0, threshold_h0);
    const auto success_h0      = static_cast<float>(folds_h0_pruned.ntrials) /
                            static_cast<float>(folds_h0.ntrials);
    const auto folds_h1  = simulate_folds(folds_cur.folds_h1, profile, rng,
                                          bias_snr, var_add, ntrials);
    const auto scores_h1 = compute_scores(folds_h1, ducy_max, wtsp);
    const auto folds_h1_pruned = prune_folds(folds_h1, scores_h1, threshold_h0);
    const auto success_h1      = static_cast<float>(folds_h1_pruned.ntrials) /
                            static_cast<float>(folds_h1.ntrials);
    const auto state_next = gen_next_state(state_cur, threshold_h0, success_h0,
                                           success_h1, nbranches);
    return {state_next, FoldsType{folds_h0_pruned, folds_h1_pruned}};
}

// Create a compound type for State
HighFive::CompoundType create_compound_state() {
    return {{"success_h0", HighFive::create_datatype<float>()},
            {"success_h1", HighFive::create_datatype<float>()},
            {"complexity", HighFive::create_datatype<float>()},
            {"complexity_cumul", HighFive::create_datatype<float>()},
            {"success_h1_cumul", HighFive::create_datatype<float>()},
            {"nbranches", HighFive::create_datatype<float>()},
            {"threshold", HighFive::create_datatype<float>()},
            {"cost", HighFive::create_datatype<float>()},
            {"threshold_prev", HighFive::create_datatype<float>()},
            {"success_h1_cumul_prev", HighFive::create_datatype<float>()},
            {"is_empty", HighFive::create_datatype<bool>()}};
}

} // namespace

// CPU-specific implementation
template <> class DynamicThresholdScheme<backend::CPU>::Impl {
public:
    Impl(std::span<const float> branching_pattern,
         std::span<const float> profile,
         SizeType ntrials,
         SizeType nprobs,
         float prob_min,
         float snr_final,
         SizeType nthresholds,
         float ducy_max,
         float wtsp,
         float beam_width,
         SizeType nthreads);

    std::vector<SizeType> get_current_thresholds_idx(SizeType istage) const;
    std::vector<float> get_branching_pattern() const {
        return m_branching_pattern;
    }
    std::vector<float> get_profile() const { return m_profile; }
    std::vector<float> get_thresholds() const { return m_thresholds; }
    std::vector<float> get_probs() const { return m_probs; }
    SizeType get_nstages() const { return m_nstages; }
    SizeType get_nthresholds() const { return m_nthresholds; }
    SizeType get_nprobs() const { return m_nprobs; }
    std::vector<State> get_states() const { return m_states; }
    void run(SizeType thres_neigh = 10);
    std::string save(const std::string& outdir = "./") const;

private:
    std::vector<float> m_branching_pattern;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    utils::ThreadSafeRNG m_rng;
    SizeType m_nthreads;

    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    SizeType m_nprobs;
    SizeType m_nbins;
    SizeType m_nstages;
    SizeType m_nthresholds;
    float m_bias_snr;
    std::vector<float> m_guess_path;
    std::vector<std::optional<FoldsType>> m_folds_in;
    std::vector<std::optional<FoldsType>> m_folds_out;
    std::vector<State> m_states;

    void run_segment(SizeType istage, SizeType thres_neigh = 10);
    void init_states();
    static std::vector<float>
    compute_thresholds(float snr_start, float snr_final, SizeType nthresholds);
    static std::vector<float> compute_probs(SizeType nprobs,
                                            float prob_min = 0.05F);
    static std::vector<float> compute_probs_linear(SizeType nprobs,
                                                   float prob_min = 0.05F);
    static std::vector<float> bound_scheme(SizeType nstages, float snr_bound);
    static std::vector<float>
    trials_scheme(std::span<const float> branching_pattern,
                  SizeType trials_start = 1,
                  float min_trials      = 1E10F);
    static std::vector<float>
    guess_scheme(SizeType nstages,
                 float snr_bound,
                 std::span<const float> branching_pattern,
                 SizeType trials_start = 1,
                 float min_trials      = 1E10F);
};

DynamicThresholdScheme<backend::CPU>::Impl::Impl(
    std::span<const float> branching_pattern,
    std::span<const float> profile,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType nthreads)
    : m_branching_pattern(branching_pattern.begin(), branching_pattern.end()),
      m_ntrials(ntrials),
      m_ducy_max(ducy_max),
      m_wtsp(wtsp),
      m_beam_width(beam_width),
      m_rng(std::random_device{}()),
      m_nthreads(nthreads) {
    if (m_branching_pattern.empty()) {
        throw std::invalid_argument("Branching pattern is empty");
    }
    m_profile     = compute_norm_profile(profile);
    m_thresholds  = compute_thresholds(0.1F, snr_final, nthresholds);
    m_probs       = compute_probs(nprobs, prob_min);
    m_nprobs      = m_probs.size();
    m_nbins       = m_profile.size();
    m_nstages     = m_branching_pattern.size();
    m_nthresholds = m_thresholds.size();

    m_nthreads = std::max<SizeType>(m_nthreads, 1);
    m_nthreads =
        std::min<SizeType>(m_nthreads, std::thread::hardware_concurrency());

    m_bias_snr   = snr_final / static_cast<float>(std::sqrt(m_nstages + 1));
    m_guess_path = guess_scheme(m_nstages, snr_final, m_branching_pattern);
    m_folds_in.resize(m_nthresholds * m_nprobs);
    m_folds_out.resize(m_nthresholds * m_nprobs);
    State initial_state;
    m_states.resize(m_nstages * m_nthresholds * m_nprobs, initial_state);
    init_states();
}

std::vector<SizeType>
DynamicThresholdScheme<backend::CPU>::Impl::get_current_thresholds_idx(
    SizeType istage) const {
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

void DynamicThresholdScheme<backend::CPU>::Impl::run(SizeType thres_neigh) {
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
        std::ranges::fill(m_folds_out, std::nullopt);
        const auto progress = static_cast<float>(istage) /
                              static_cast<float>(m_nstages - 1) * 100.0F;
        bar.set_progress(static_cast<SizeType>(progress));
    }
    indicators::show_console_cursor(true);
}

std::string DynamicThresholdScheme<backend::CPU>::Impl::save(
    const std::string& outdir) const {
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
    file.createAttribute("wtsp", m_wtsp);
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
    // Define the 3D dataspace for states
    std::vector<SizeType> dims = {m_nstages, m_nthresholds, m_nprobs};
    HighFive::DataSetCreateProps props_states;
    std::vector<hsize_t> chunk_dims(dims.begin(), dims.end());
    props_states.add(HighFive::Chunking(chunk_dims));
    auto dataset = file.createDataSet("states", HighFive::DataSpace(dims),
                                      create_compound_state(), props_states);
    dataset.write_raw(m_states.data());
    spdlog::info("Saved dynamic threshold scheme to {}", filepath.string());
    return filepath.string();
}

void DynamicThresholdScheme<backend::CPU>::Impl::run_segment(
    SizeType istage, SizeType thres_neigh) {
    const auto beam_idx_cur      = get_current_thresholds_idx(istage);
    const auto beam_idx_prev     = get_current_thresholds_idx(istage - 1);
    const auto stage_offset_prev = (istage - 1) * m_nthresholds * m_nprobs;
    const auto stage_offset_cur  = istage * m_nthresholds * m_nprobs;

#pragma omp parallel for num_threads(m_nthreads)
    for (SizeType i = 0; i < beam_idx_cur.size(); ++i) {
        const auto ithres = beam_idx_cur[i];
        // Find nearest neighbors in the previous beam
        const auto neighbour_beam_indices = utils::find_neighbouring_indices(
            beam_idx_prev, ithres, thres_neigh);
        for (SizeType jthresh : neighbour_beam_indices) {
            for (SizeType kprob = 0; kprob < m_nprobs; ++kprob) {
                const auto prev_fold_idx = (jthresh * m_nprobs) + kprob;
                const auto prev_state =
                    m_states[stage_offset_prev + prev_fold_idx];
                if (prev_state.is_empty) {
                    continue;
                }
                const auto prev_fold_state = m_folds_in[prev_fold_idx];
                if (prev_fold_state.has_value() &&
                    !prev_fold_state->is_empty()) {
                    const auto [cur_state, cur_fold_state] =
                        gen_next_using_thresh(
                            prev_state, *prev_fold_state, m_thresholds[ithres],
                            m_branching_pattern[istage], m_bias_snr, m_profile,
                            m_rng, 1.0F, m_ntrials, m_ducy_max, m_wtsp);
                    const auto iprob =
                        find_bin_index(m_probs, cur_state.success_h1_cumul);
                    if (iprob < 0 ||
                        iprob >= static_cast<IndexType>(m_nprobs)) {
                        continue;
                    }
                    const auto cur_idx       = (ithres * m_nprobs) + iprob;
                    const auto cur_state_idx = stage_offset_cur + cur_idx;
                    auto& existing_state     = m_states[cur_state_idx];
                    if (existing_state.is_empty ||
                        cur_state.complexity_cumul <
                            existing_state.complexity_cumul) {
                        existing_state       = cur_state;
                        m_folds_out[cur_idx] = cur_fold_state;
                    }
                }
            }
        }
    }
}

void DynamicThresholdScheme<backend::CPU>::Impl::init_states() {
    const float var_init = 1.0F;
    const FoldVector folds_init(m_ntrials, m_nbins);
    // Simulate the initial folds (pruning level = 0)
    const auto folds_h0 =
        simulate_folds(folds_init, m_profile, m_rng, 0.0F, var_init, m_ntrials);
    const auto folds_h1 = simulate_folds(folds_init, m_profile, m_rng,
                                         m_bias_snr, var_init, m_ntrials);
    State initial_state;
    const FoldsType fold_state{folds_h0, folds_h1};
    const auto thresholds_idx = get_current_thresholds_idx(0);
    for (SizeType ithres : thresholds_idx) {
        const auto [cur_state, cur_fold_state] = gen_next_using_thresh(
            initial_state, fold_state, m_thresholds[ithres],
            m_branching_pattern[0], m_bias_snr, m_profile, m_rng, 1.0F,
            m_ntrials, m_ducy_max, m_wtsp);
        const auto iprob = find_bin_index(m_probs, cur_state.success_h1_cumul);
        if (iprob < 0 || iprob >= static_cast<IndexType>(m_nprobs)) {
            continue;
        }
        m_states[(ithres * m_nprobs) + iprob]   = cur_state;
        m_folds_in[(ithres * m_nprobs) + iprob] = cur_fold_state;
    }
}

std::vector<float>
DynamicThresholdScheme<backend::CPU>::Impl::compute_thresholds(
    float snr_start, float snr_final, SizeType nthresholds) {
    std::vector<float> thresholds(nthresholds);
    const auto snr_step =
        (snr_final - snr_start) / static_cast<float>(nthresholds - 1);
    for (SizeType i = 0; i < nthresholds; ++i) {
        thresholds[i] = static_cast<float>(i) * snr_step + snr_start;
    }
    return thresholds;
}

std::vector<float>
DynamicThresholdScheme<backend::CPU>::Impl::compute_probs(SizeType nprobs,
                                                          float prob_min) {
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

std::vector<float>
DynamicThresholdScheme<backend::CPU>::Impl::compute_probs_linear(
    SizeType nprobs, float prob_min) {
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

std::vector<float>
DynamicThresholdScheme<backend::CPU>::Impl::bound_scheme(SizeType nstages,
                                                         float snr_bound) {
    const auto nsegments = nstages + 1;
    std::vector<float> thresholds(nstages);
    for (SizeType i = 0; i < nstages; ++i) {
        thresholds[i] = std::sqrt(static_cast<float>((i + 2)) * snr_bound *
                                  snr_bound / static_cast<float>(nsegments));
    }
    return thresholds;
}

std::vector<float> DynamicThresholdScheme<backend::CPU>::Impl::trials_scheme(
    std::span<const float> branching_pattern,
    SizeType trials_start,
    float min_trials) {
    const auto nstages = branching_pattern.size();
    std::vector<float> result(nstages);
    // trials = np.cumprod(branching_pattern) * trials_start
    auto trials = static_cast<float>(trials_start);
    for (SizeType i = 0; i < nstages; ++i) {
        trials *= branching_pattern[i];
        const auto effective_trials = std::max(trials, min_trials);
        result[i] = loki::math::norm_isf(1 / effective_trials);
    }
    return result;
}

std::vector<float> DynamicThresholdScheme<backend::CPU>::Impl::guess_scheme(
    SizeType nstages,
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

// CPU-specific constructor implementation
template <>
template <std::same_as<backend::CPU> P>
DynamicThresholdScheme<backend::CPU>::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    std::span<const float> profile,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType nthreads)
    : m_impl(std::make_unique<Impl>(branching_pattern,
                                    profile,
                                    ntrials,
                                    nprobs,
                                    prob_min,
                                    snr_final,
                                    nthresholds,
                                    ducy_max,
                                    wtsp,
                                    beam_width,
                                    nthreads)) {}

// Method implementations - these forward to the Impl
template <>
std::vector<float>
DynamicThresholdScheme<backend::CPU>::get_branching_pattern() const {
    return m_impl->get_branching_pattern();
}
template <>
std::vector<float> DynamicThresholdScheme<backend::CPU>::get_profile() const {
    return m_impl->get_profile();
}
template <>
std::vector<float>
DynamicThresholdScheme<backend::CPU>::get_thresholds() const {
    return m_impl->get_thresholds();
}
template <>
std::vector<float> DynamicThresholdScheme<backend::CPU>::get_probs() const {
    return m_impl->get_probs();
}
template <> SizeType DynamicThresholdScheme<backend::CPU>::get_nstages() const {
    return m_impl->get_nstages();
}
template <>
SizeType DynamicThresholdScheme<backend::CPU>::get_nthresholds() const {
    return m_impl->get_nthresholds();
}
template <> SizeType DynamicThresholdScheme<backend::CPU>::get_nprobs() const {
    return m_impl->get_nprobs();
}
template <>
std::vector<State> DynamicThresholdScheme<backend::CPU>::get_states() const {
    return m_impl->get_states();
}
template <>
void DynamicThresholdScheme<backend::CPU>::run(SizeType thres_neigh) {
    m_impl->run(thres_neigh);
}
template <>
std::string
DynamicThresholdScheme<backend::CPU>::save(const std::string& outdir) const {
    return m_impl->save(outdir);
}

std::vector<State> evaluate_scheme(std::span<const float> thresholds,
                                   std::span<const float> branching_pattern,
                                   std::span<const float> profile,
                                   SizeType ntrials,
                                   float snr_final,
                                   float ducy_max,
                                   float wtsp) {
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
    const float var_init = 1.0F;
    State initial_state;
    std::vector<State> states(nstages, initial_state);
    std::vector<std::optional<FoldsType>> fold_states(nstages);

    utils::ThreadSafeRNG rng(std::random_device{}());
    const FoldVector folds_init(ntrials, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 =
        simulate_folds(folds_init, profile_norm, rng, 0.0F, var_init, ntrials);
    auto folds_h1 = simulate_folds(folds_init, profile_norm, rng, bias_snr,
                                   var_init, ntrials);
    FoldsType initial_fold_state{std::move(folds_h0), std::move(folds_h1)};
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto prev_state =
            (istage == 0) ? initial_state : states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : *fold_states[istage - 1];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info(
                "Path not viable, No trials survived, stopping at stage {}",
                istage);
            break;
        }
        auto [cur_state, cur_fold_state] = gen_next_using_thresh(
            prev_state, prev_fold_state, thresholds[istage],
            branching_pattern[istage], bias_snr, profile_norm, rng, 1.0F,
            ntrials, ducy_max, wtsp);
        states[istage]      = cur_state;
        fold_states[istage] = std::move(cur_fold_state);
    }
    return states;
}

std::vector<State> determine_scheme(std::span<const float> survive_probs,
                                    std::span<const float> branching_pattern,
                                    std::span<const float> profile,
                                    SizeType ntrials,
                                    float snr_final,
                                    float ducy_max,
                                    float wtsp) {
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
    const float var_init = 1.0F;
    State initial_state;
    std::vector<State> states(nstages, initial_state);
    std::vector<std::optional<FoldsType>> fold_states(nstages);

    utils::ThreadSafeRNG rng(std::random_device{}());
    const FoldVector folds_init(ntrials, nbins);
    // Simulate the initial folds (pruning level = 0)
    auto folds_h0 =
        simulate_folds(folds_init, profile_norm, rng, 0.0F, var_init, ntrials);
    auto folds_h1 = simulate_folds(folds_init, profile_norm, rng, bias_snr,
                                   var_init, ntrials);
    const FoldsType initial_fold_state{std::move(folds_h0),
                                       std::move(folds_h1)};
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto prev_state =
            (istage == 0) ? initial_state : states[istage - 1];
        const auto& prev_fold_state =
            (istage == 0) ? initial_fold_state : *fold_states[istage - 1];
        if (istage > 0 && prev_fold_state.is_empty()) {
            spdlog::info("Path not viable, stopping at stage {}", istage);
            break;
        }
        auto [cur_state, cur_fold_state] = gen_next_using_surv_prob(
            prev_state, prev_fold_state, survive_probs[istage],
            branching_pattern[istage], bias_snr, profile_norm, rng, 1.0F,
            ntrials, ducy_max, wtsp);
        states[istage]      = cur_state;
        fold_states[istage] = cur_fold_state;
    }
    return states;
}
} // namespace loki

HIGHFIVE_REGISTER_TYPE(loki::State, loki::create_compound_state)