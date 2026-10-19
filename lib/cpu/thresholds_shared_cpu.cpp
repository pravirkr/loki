#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <highfive/highfive.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/detection/thresholds.hpp"
#include "loki/simulation/simulation.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/math.hpp"
#include "lib/detail/progress.hpp"
#include "lib/detail/timing.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"

// CPU engine of mode "shared": the algorithm of the CUDA engine, written for
// the host. The signal (H1) is one unpruned bank of shared trials and a cell
// keeps a bitmask of the trials still above its thresholds; the noise (H0)
// of every parent cell is resampled to ntrials each stage, a surviving trial
// keeping its slot, with a survivor index map instead of copied folds.
// Evaluation and the signal simulation are those of the evaluation-only
// ThresholdsCpuEngine this engine owns, so they agree with every other CPU
// mode. Every draw comes from a stream keyed on (seed, purpose, stage,
// trial), and every reduction is ordered, so results do not depend on the
// thread count.

namespace loki::detection {

namespace {

// make_keyed_pcg32 purposes (0-3: evaluate(); 4, 5: the improved mode).
constexpr uint32_t kPurposeInit      = 6; // initial noise folds
constexpr uint32_t kPurposeNoise     = 7; // noise of one stage
constexpr uint32_t kPurposeSelect    = 8; // refill draw of a failed slot
constexpr uint64_t kPilotSalt        = 0x9110CA11ULL;
constexpr uint64_t kSharedSalt       = 0x5BA7ED11ULL;
constexpr SizeType kSharedH1PerTrial = 16;
constexpr unsigned long long kEmptyKey =
    std::numeric_limits<unsigned long long>::max();

struct ThresholdPair {
    uint32_t ithres;
    uint32_t islot;
    uint32_t jthres;
    uint32_t jslot;
};

struct Parent {
    uint32_t jslot;
    uint32_t jthres;
    uint32_t kprob;
};

float unit_interval(math::PCG32& rng) noexcept {
    return static_cast<float>(rng() >> 8U) * (1.0F / 16777216.0F);
}

// Same layout as the compound written by the other engines.
HighFive::CompoundType state_compound() {
    const auto f = HighFive::create_datatype<float>();
    using detection::State;
    return HighFive::CompoundType(
        {{"success_h0", f, offsetof(State, success_h0)},
         {"success_h1", f, offsetof(State, success_h1)},
         {"complexity", f, offsetof(State, complexity)},
         {"complexity_cumul", f, offsetof(State, complexity_cumul)},
         {"success_h1_cumul", f, offsetof(State, success_h1_cumul)},
         {"nbranches", f, offsetof(State, nbranches)},
         {"threshold", f, offsetof(State, threshold)},
         {"cost", f, offsetof(State, cost)},
         {"threshold_prev", f, offsetof(State, threshold_prev)},
         {"success_h1_cumul_prev", f, offsetof(State, success_h1_cumul_prev)},
         {"is_empty", HighFive::create_datatype<bool>(),
          offsetof(State, is_empty)}},
        sizeof(State));
}

class ThresholdsSharedCpuEngine final : public detail::ThresholdsEngine {
public:
    ThresholdsSharedCpuEngine(std::span<const float> branching_pattern,
                              float ref_ducy,
                              SizeType nbins,
                              SizeType ntrials,
                              SizeType nprobs,
                              float snr_final,
                              SizeType nthresholds,
                              float ducy_max,
                              float wtsp,
                              float beam_width,
                              SizeType trials_start,
                              int nthreads,
                              std::optional<uint64_t> seed,
                              std::string_view guess,
                              DynamicThresholdMode mode,
                              float min_pd,
                              float w_terminal,
                              float beam_sigma,
                              float reach_sigma)
        : m_branching_pattern(branching_pattern.begin(),
                              branching_pattern.end()),
          m_ref_ducy(ref_ducy),
          m_ntrials(ntrials),
          m_ducy_max(ducy_max),
          m_wtsp(wtsp),
          m_beam_width(beam_width),
          m_nbins(nbins),
          m_nthreads(std::max(nthreads, 1)),
          m_seed(seed.value_or(std::random_device{}())),
          m_min_pd(min_pd),
          m_w(static_cast<double>(w_terminal)),
          m_guess(guess),
          m_mode(mode),
          m_guide_width(beam_sigma),
          m_guide_reach(reach_sigma) {
        if (!is_shared_family(mode)) {
            throw std::invalid_argument(
                "ThresholdsSharedCpuEngine: modes 'shared' and 'robust' only");
        }
        // The guess applies to legacy and improved only, but must be valid.
        (void)detail::beam_guess_from_string(guess);
        // The checks of every backend's search (prob_min is unused here).
        detail::validate_scheme_inputs({.branching_pattern = branching_pattern,
                                        .ref_ducy          = ref_ducy,
                                        .ducy_max          = ducy_max,
                                        .wtsp              = wtsp,
                                        .beam_width        = beam_width,
                                        .prob_min          = 0.5F,
                                        .snr_final         = snr_final,
                                        .nbins             = nbins,
                                        .ntrials           = ntrials,
                                        .nprobs            = nprobs,
                                        .nthresholds       = nthresholds,
                                        .batch_size        = 1},
                                       "ThresholdsSharedCpu");
        if (!(beam_sigma > 0.0F) || !(reach_sigma > 0.0F)) {
            throw std::invalid_argument(
                "DynamicThresholdScheme: beam_sigma and reach_sigma must be "
                "positive");
        }
        if (!(min_pd > 0.0F && min_pd < 1.0F)) {
            throw std::invalid_argument(
                std::format("DynamicThresholdScheme: min_pd={} must be in "
                            "(0, 1)",
                            min_pd));
        }
        if (nprobs < 2) {
            throw std::invalid_argument(
                "DynamicThresholdScheme: shared mode needs nprobs >= 2");
        }
        if (!(w_terminal >= 0.0F) || !utils::is_finite(w_terminal)) {
            throw std::invalid_argument(
                "DynamicThresholdScheme: w_terminal must be finite and >= 0");
        }
        if (ntrials == 0 ||
            ntrials > std::numeric_limits<uint32_t>::max() / 2) {
            throw std::invalid_argument(
                "DynamicThresholdScheme: ntrials out of range");
        }
        // Evaluation and the signal simulation: the evaluation-only engine
        // shares the physics of every other CPU mode. Its grid is unused.
        m_eval = detail::make_thresholds_cpu(
            branching_pattern, ref_ducy, nbins, ntrials, 2, 0.5F, snr_final,
            nthresholds, ducy_max, wtsp, beam_width, trials_start, "improved",
            m_nthreads, m_seed, /*simulate=*/false, "analytic");
        m_profile    = simulation::generate_folded_profile(nbins, ref_ducy);
        m_thresholds = detail::compute_thresholds(0.1F, snr_final, nthresholds);
        m_probs      = detail::shared_probability_grid(min_pd, nprobs);
        m_nprobs     = m_probs.size();
        m_nbins      = m_profile.size();
        m_nstages    = m_branching_pattern.size();
        m_nthresholds = m_thresholds.size();
        m_box_score_widths =
            detection::generate_box_width_trials(m_nbins, m_ducy_max, m_wtsp);
        if (*std::ranges::max_element(m_box_score_widths) >= m_nbins) {
            throw std::invalid_argument(
                std::format("ducy_max={} gives a box width >= nbins={}",
                            m_ducy_max, m_nbins));
        }
        m_nh1      = std::min<SizeType>(kSharedH1PerTrial * m_ntrials,
                                        SizeType{1} << 20U);
        m_h1_words = (m_nh1 + 31) / 32;
        {
            auto pilot = m_eval->simulate_h1_scores(
                detail::kWindowTrials, detail::derive_seed(m_seed, kPilotSalt));
            for (SizeType s = 0; s < m_nstages; ++s) {
                const auto row = std::span(pilot).subspan(
                    s * detail::kWindowTrials, detail::kWindowTrials);
                std::ranges::sort(row);
            }
            auto stats = detail::pilot_stats(pilot, detail::kWindowTrials,
                                             m_branching_pattern, trials_start,
                                             m_probs.front());
            m_window   = std::move(stats.window);
            m_guess_pilot = std::move(stats.guess);
            m_score_sigma = std::move(stats.sigma);
        }
        m_guess_path = m_guess_pilot;
        m_beam = detail::guided_beam(m_window, m_guess_path, m_score_sigma,
                                     m_guide_width);
        for (SizeType s = 0; s < m_nstages; ++s) {
            const auto n = get_current_thresholds_idx(s).size();
            if (n == 0) {
                throw std::invalid_argument(
                    std::format("Threshold beam of stage {} is empty", s));
            }
            m_max_beam = std::max(m_max_beam, n);
        }
        // Any recentred beam (mode robust's second pass) fits the pools.
        m_max_beam = std::max(m_max_beam, detail::guided_beam_capacity(
                                              m_thresholds, m_window,
                                              m_score_sigma, m_guide_width));
        const SizeType cells = m_max_beam * m_nprobs;
        if (cells * 2 > std::numeric_limits<uint32_t>::max() ||
            m_nthresholds * m_nprobs > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument(
                "DynamicThresholdScheme: problem size exceeds 32-bit cell "
                "indexing");
        }
        m_states.resize(m_nstages * m_nthresholds * m_nprobs, State{});
        m_box_cache = std::make_unique<detection::BoxcarWidthsCache>(
            m_box_score_widths, m_nbins);
    }

    // ---- getters ----
    std::vector<float> get_branching_pattern() const override {
        return m_branching_pattern;
    }
    std::vector<float> get_profile() const override { return m_profile; }
    std::vector<float> get_thresholds() const override { return m_thresholds; }
    std::vector<float> get_probs() const override { return m_probs; }
    SizeType get_nstages() const override { return m_nstages; }
    SizeType get_nthresholds() const override { return m_nthresholds; }
    SizeType get_nprobs() const override { return m_nprobs; }
    std::vector<SizeType> get_box_score_widths() const override {
        return m_box_score_widths;
    }
    std::vector<State> get_states() const override { return m_states; }
    uint64_t get_seed() const override { return m_seed; }
    std::vector<SizeType>
    get_current_thresholds_idx(SizeType istage) const override {
        return detail::beam_indices(m_thresholds, m_beam.lo[istage],
                                    m_beam.hi[istage]);
    }

    SizeType run_memory_bytes() const override {
        const SizeType pool_rows = pool_rows_count();
        const SizeType cells     = m_max_beam * m_nprobs;
        return (2 * pool_rows * m_nbins * sizeof(float)) +
               (2 * pool_rows * sizeof(uint32_t)) +
               (pool_rows * sizeof(float)) +
               (2 * cells * 3 * sizeof(uint32_t)) +
               (((2 * cells) + m_max_beam) * m_h1_words * sizeof(uint32_t)) +
               (m_nstages * m_nh1 * sizeof(float)) +
               (m_states.size() * sizeof(State));
    }

    // ---- selection ----
    detail::SearchPath selected_path(std::span<const State> states,
                                     float min_pd) const {
        return detail::search_path(
            states, m_thresholds, m_probs, m_nstages, m_nthresholds, m_nprobs,
            detail::best_terminal_cell(states, m_nstages, m_nthresholds,
                                       m_nprobs, min_pd, m_w),
            m_w);
    }

    std::vector<float> get_best_path_thresholds(float min_pd) const override {
        auto path = selected_path(m_states, min_pd);
        if (!path.thresholds.empty()) {
            return std::move(path.thresholds);
        }
        return detail::get_best_path_thresholds(
            std::span<const State>(m_states.data(), m_states.size()),
            m_thresholds, m_probs, m_nstages, m_nthresholds, m_nprobs, min_pd);
    }

    detail::SearchPath first_pass_path(float min_pd) const override {
        if (m_first_states.empty()) {
            throw std::logic_error(
                "first_pass_path: needs run() in mode 'robust'");
        }
        return selected_path(m_first_states, min_pd);
    }

    std::vector<float> shared_h1_bank() const override {
        if (m_h1_bank.empty()) {
            throw std::logic_error("shared_h1_bank: run() has not been called");
        }
        return m_h1_bank;
    }

    void release_search_memory() override {
        const auto drop = [](auto& v) {
            std::remove_reference_t<decltype(v)>().swap(v);
        };
        for (auto& v : m_folds) {
            drop(v);
        }
        for (auto& v : m_ntrials_desc) {
            drop(v);
        }
        for (auto& v : m_src) {
            drop(v);
        }
        for (auto& v : m_idx) {
            drop(v);
        }
        for (auto& v : m_h1_mask) {
            drop(v);
        }
        drop(m_scores);
        drop(m_h1_pass);
    }

    // ---- evaluation: the evaluation-only engine ----
    std::vector<State> evaluate(std::span<const float> thresholds,
                                SizeType ntrials,
                                std::optional<uint64_t> seed,
                                bool h0_only) const override {
        return m_eval->evaluate(thresholds, ntrials, seed, h0_only);
    }
    std::vector<float> simulate_h1_scores(SizeType ntrials,
                                          uint64_t seed) const override {
        return m_eval->simulate_h1_scores(ntrials, seed);
    }

    // ---- search ----
    void run(SizeType thres_neigh) override {
        // thres_neigh is unused (the guided beam sets the reach); positive as
        // on every backend.
        if (thres_neigh == 0) {
            throw std::invalid_argument("thres_neigh must be positive");
        }
        spdlog::info("Running dynamic threshold scheme on the CPU ({} mode)",
                     mode_to_string(m_mode));
        timing::SimpleTimer total_timer;
        total_timer.start();
        allocate_pools();
        // One signal bank per run(): both passes count on the same trials.
        m_h1_bank = m_eval->simulate_h1_scores(
            m_nh1, detail::derive_seed(m_seed, kSharedSalt));
        m_cost_to_go.clear();
        m_first_states.clear();
        m_guess_path = m_guess_pilot;
        m_beam = detail::guided_beam(m_window, m_guess_path, m_score_sigma,
                                     m_guide_width);
        run_pass();
        if (m_mode == DynamicThresholdMode::kRobust) {
            // Second pass: ranked with the first pass's cost to go, around
            // the guess recentred on its path.
            m_first_states = m_states;
            m_cost_to_go   = detail::cost_to_go_scheme(
                m_states, m_thresholds, m_probs, m_nstages, m_nthresholds,
                m_nprobs, 8.0F, 4, m_w);
            if (!m_cost_to_go.empty()) {
                m_guess_path = detail::recentre_guess(
                    m_guess_pilot, get_best_path_thresholds(m_min_pd));
                m_beam = detail::guided_beam(m_window, m_guess_path,
                                             m_score_sigma, m_guide_width);
                run_pass();
            }
        }
        detail::store_terminal_cost(m_states, m_nstages, m_nthresholds,
                                    m_nprobs, m_w);
        detail::warn_empty_stages(m_states, m_nstages, m_nthresholds, m_nprobs,
                                  "ThresholdsSharedCpu");
        spdlog::info("Dynamic threshold scheme ({}) took {:.2f} s",
                     mode_to_string(m_mode), total_timer.stop());
    }

    std::string save(const std::string& outdir) const override {
        const std::filesystem::path filebase = std::format(
            "dynscheme_{}_nstages_{:03d}_nthresh_{:03d}_nprobs_{:03d}_"
            "ntrials_{:04d}_snr_{:04.1f}_ducy_{:04.2f}_beam_{:03.1f}.h5",
            mode_to_string(m_mode), m_nstages, m_nthresholds, m_nprobs,
            m_ntrials, m_thresholds.back(), m_ref_ducy, m_beam_width);
        std::filesystem::create_directories(outdir);
        const std::filesystem::path filepath =
            std::filesystem::path(outdir) / filebase;
        HighFive::File file(filepath, HighFive::File::Overwrite);
        file.createAttribute("ntrials", m_ntrials);
        file.createAttribute("snr_final", m_thresholds.back());
        file.createAttribute("ref_ducy", m_ref_ducy);
        file.createAttribute("ducy_max", m_ducy_max);
        file.createAttribute("wtsp", m_wtsp);
        file.createAttribute("beam_width", m_beam_width);
        file.createAttribute("mode", mode_to_string(m_mode));
        file.createAttribute("guess", m_guess);
        file.createAttribute("seed", m_seed);
        file.createAttribute("nthreads", m_nthreads);
        file.createAttribute("min_pd", m_min_pd);
        file.createAttribute("w_terminal", m_w);
        file.createAttribute("beam_sigma", m_guide_width);
        file.createAttribute("reach_sigma", m_guide_reach);
        file.createDataSet("branching_pattern", m_branching_pattern);
        file.createDataSet("profile", m_profile);
        file.createDataSet("thresholds", m_thresholds);
        file.createDataSet("probs", m_probs);
        file.createDataSet("guess_path", m_guess_path);
        file.createDataSet("beam_lo", m_beam.lo);
        file.createDataSet("beam_hi", m_beam.hi);
        file.createDataSet("guess_path_pilot", m_guess_pilot);
        const std::vector<SizeType> dims = {m_nstages, m_nthresholds, m_nprobs};
        HighFive::DataSetCreateProps props_states;
        props_states.add(
            HighFive::Chunking(std::vector<hsize_t>(dims.begin(), dims.end())));
        const auto compound = state_compound();
        auto dataset = file.createDataSet("states", HighFive::DataSpace(dims),
                                          compound, props_states);
        dataset.write_raw(m_states.data(), compound);
        spdlog::info("Saved dynamic threshold scheme to {}", filepath.string());
        return filepath.string();
    }

private:
    SizeType pool_rows_count() const {
        return std::max<SizeType>(m_max_beam * m_nprobs, 2) * m_ntrials;
    }

    void allocate_pools() {
        if (!m_scores.empty()) {
            return;
        }
        const SizeType cells     = m_max_beam * m_nprobs;
        const SizeType pool_rows = pool_rows_count();
        for (auto& v : m_folds) {
            v.assign(pool_rows * m_nbins, 0.0F);
        }
        for (auto& v : m_ntrials_desc) {
            v.assign(cells * 2, 0U);
        }
        for (auto& v : m_src) {
            v.assign(std::max<SizeType>(cells, 1), 0U);
        }
        for (auto& v : m_idx) {
            v.assign(pool_rows, 0U);
        }
        m_scores.assign(pool_rows, 0.0F);
        for (auto& v : m_h1_mask) {
            v.assign(std::max<SizeType>(cells, 1) * m_h1_words, 0U);
        }
        m_h1_pass.assign(m_max_beam * m_h1_words, 0U);
    }

    /// Noise survivors a transition must show to count at full weight in the
    /// ranking surrogate: 8 of 1024 trials, the same fraction at any ntrials.
    uint32_t min_support() const {
        const auto n = static_cast<uint32_t>(m_ntrials);
        return std::clamp<uint32_t>((n + 64U) / 128U, 1U, n);
    }

    float rank_weight(SizeType istage) const {
        if (istage + 1 == m_nstages) {
            return static_cast<float>(m_w);
        }
        return m_cost_to_go.empty() ? 0.0F : m_cost_to_go[istage];
    }

    static int find_bin_index(std::span<const float> probs, float v) {
        if (v < probs[0]) {
            return -1;
        }
        for (SizeType i = 1; i < probs.size(); ++i) {
            if (v < probs[i]) {
                return static_cast<int>(i - 1);
            }
        }
        return static_cast<int>(probs.size() - 1);
    }

    struct Plan {
        std::vector<SizeType> beam_cur;
        std::vector<ThresholdPair> pairs;
    };

    /// Every pair (previous beam threshold, current beam threshold) within
    /// reach of the guess is a transition. Stage 0 reads a virtual
    /// one-threshold beam.
    std::vector<Plan> build_plans() const {
        std::vector<Plan> plans(m_nstages);
        std::vector<SizeType> beam_prev = {0};
        for (SizeType s = 0; s < m_nstages; ++s) {
            auto& plan    = plans[s];
            plan.beam_cur = get_current_thresholds_idx(s);
            for (SizeType islot = 0; islot < plan.beam_cur.size(); ++islot) {
                if (s == 0) {
                    plan.pairs.push_back(
                        {static_cast<uint32_t>(plan.beam_cur[islot]),
                         static_cast<uint32_t>(islot), 0U, 0U});
                    continue;
                }
                for (SizeType jslot = 0; jslot < beam_prev.size(); ++jslot) {
                    if (!detail::within_reach(
                            m_thresholds[plan.beam_cur[islot]], m_guess_path[s],
                            m_thresholds[beam_prev[jslot]], m_guess_path[s - 1],
                            m_score_sigma[s], m_guide_reach)) {
                        continue;
                    }
                    plan.pairs.push_back(
                        {static_cast<uint32_t>(plan.beam_cur[islot]),
                         static_cast<uint32_t>(islot),
                         static_cast<uint32_t>(beam_prev[jslot]),
                         static_cast<uint32_t>(jslot)});
                }
            }
            beam_prev = plan.beam_cur;
        }
        return plans;
    }

    // One forward pass into m_states, from a fresh grid.
    void run_pass() {
        const auto ntrials         = static_cast<uint32_t>(m_ntrials);
        const auto nprobs          = static_cast<uint32_t>(m_nprobs);
        const SizeType grid_stride = m_nthresholds * m_nprobs;
        const auto nh1             = static_cast<uint32_t>(m_nh1);
        const auto h1_words        = static_cast<uint32_t>(m_h1_words);
        const float ntrials_f      = static_cast<float>(ntrials);
        const uint32_t support     = min_support();
        std::ranges::fill(m_states, State{});
        for (auto& v : m_ntrials_desc) {
            std::ranges::fill(v, 0U);
        }
        const auto plans = build_plans();

        // Initial parent: cell 0 of the odd buffers, all trials alive, with an
        // identity survivor map and its noise folds.
        std::vector<State> init_states(m_nprobs, State{});
        init_states[0] = State::initial();
        m_src[1][0]    = 0;
        for (uint32_t i = 0; i < ntrials; ++i) {
            m_idx[1][i] = i;
        }
        // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(m_nthreads)
        for (std::int64_t t = 0; t < static_cast<std::int64_t>(ntrials); ++t) {
            auto rng = math::make_keyed_pcg32(m_seed, kPurposeInit, 0U, 0U, 0U,
                                              static_cast<uint32_t>(t));
            math::NormalSampler::generate(
                rng,
                std::span<float>(m_folds[1].data() +
                                     (static_cast<SizeType>(t) * m_nbins),
                                 m_nbins),
                0.0F, 1.0F);
        }
        m_ntrials_desc[1][0] = ntrials;
        m_ntrials_desc[1][1] = nh1;
        // The virtual parent keeps every signal trial.
        std::fill_n(m_h1_mask[1].begin(), h1_words, 0xFFFFFFFFU);
        if (const uint32_t tail = nh1 % 32U; tail != 0) {
            m_h1_mask[1][h1_words - 1] = (1U << tail) - 1U;
        }
        std::vector<float> noise(m_ntrials * m_nbins);
        std::vector<detection::BoxcarWidthsCache> caches(
            static_cast<SizeType>(m_nthreads), *m_box_cache);

        progress::ProgressGuard const progress_guard(true);
        auto bar =
            progress::make_standard_bar("Computing scheme (shared)", m_nstages);
        float var_in = 1.0F;
        for (SizeType istage = 0; istage < m_nstages; ++istage) {
            const auto& plan      = plans[istage];
            const uint32_t cur    = static_cast<uint32_t>(istage & 1U);
            const uint32_t prev   = cur ^ 1U;
            const float nbranches = m_branching_pattern[istage];
            const State* prev_states =
                istage == 0 ? init_states.data()
                            : m_states.data() + ((istage - 1) * grid_stride);
            State* cur_states = m_states.data() + (istage * grid_stride);
            const auto n_beam_prev =
                istage == 0 ? SizeType{1} : plans[istage - 1].beam_cur.size();

            // Valid parent cells of the previous stage, compacted.
            std::vector<Parent> parents;
            parents.reserve(n_beam_prev * nprobs);
            for (SizeType jslot = 0; jslot < n_beam_prev; ++jslot) {
                const uint32_t jthres =
                    istage == 0 ? 0U
                                : static_cast<uint32_t>(
                                      plans[istage - 1].beam_cur[jslot]);
                for (uint32_t kprob = 0; kprob < nprobs; ++kprob) {
                    const uint32_t cell =
                        static_cast<uint32_t>(jslot * nprobs) + kprob;
                    if (!prev_states[(jthres * nprobs) + kprob].is_empty &&
                        m_ntrials_desc[prev][2 * cell] != 0 &&
                        m_ntrials_desc[prev][(2 * cell) + 1] != 0) {
                        parents.push_back(
                            {static_cast<uint32_t>(jslot), jthres, kprob});
                    }
                }
            }

            // Signal trials above each current threshold, as bitmasks.
            const float* bank = m_h1_bank.data() + (istage * m_nh1);
            // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(m_nthreads)
            for (std::int64_t i = 0;
                 i < static_cast<std::int64_t>(plan.beam_cur.size() * h1_words);
                 ++i) {
                const auto islot = static_cast<SizeType>(i) / h1_words;
                const auto w =
                    static_cast<uint32_t>(static_cast<SizeType>(i) % h1_words);
                const float threshold = m_thresholds[plan.beam_cur[islot]];
                uint32_t bits         = 0;
                for (uint32_t b = 0; b < 32; ++b) {
                    const uint32_t t = (w * 32) + b;
                    if (t < nh1 && bank[t] > threshold) {
                        bits |= 1U << b;
                    }
                }
                m_h1_pass[i] = bits;
            }

            // Noise of this stage: the same for every parent (common random
            // numbers).
            // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(m_nthreads)
            for (std::int64_t t = 0; t < static_cast<std::int64_t>(ntrials);
                 ++t) {
                auto rng = math::make_keyed_pcg32(
                    m_seed, kPurposeNoise, static_cast<uint32_t>(istage), 0U,
                    0U, static_cast<uint32_t>(t));
                math::NormalSampler::generate(
                    rng,
                    std::span<float>(noise.data() +
                                         (static_cast<SizeType>(t) * m_nbins),
                                     m_nbins),
                    0.0F, 1.0F);
            }

            // Every parent's population, one stage on, and its scores.
            const float stdnoise = std::sqrt(var_in + 1.0F);
            const auto n_sim     = static_cast<std::int64_t>(parents.size()) *
                                   static_cast<std::int64_t>(ntrials);
            const auto istage_u  = static_cast<uint32_t>(istage);
            // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
            for (std::int64_t item = 0; item < n_sim; ++item) {
                const auto ip    = static_cast<SizeType>(item) / ntrials;
                const auto trial = static_cast<uint32_t>(
                    static_cast<SizeType>(item) % ntrials);
                const Parent par     = parents[ip];
                const uint32_t cell  = (par.jslot * nprobs) + par.kprob;
                const uint32_t count = m_ntrials_desc[prev][2 * cell];
                const uint32_t* survivors =
                    m_idx[prev].data() +
                    (static_cast<SizeType>(cell) * ntrials);
                const uint32_t* lo =
                    std::lower_bound(survivors, survivors + count, trial);
                const auto nlo     = static_cast<uint32_t>(lo - survivors);
                uint32_t src_trial = nlo;
                if (!(nlo < count && survivors[nlo] == trial)) {
                    // A failed slot: its rank among the failed slots draws a
                    // stratified quantile of the survivors.
                    auto rng = math::make_keyed_pcg32(m_seed, kPurposeSelect,
                                                      istage_u, 0U, 0U, trial);
                    const float u        = unit_interval(rng);
                    const uint32_t nfail = ntrials - count;
                    const float quantile =
                        (static_cast<float>(trial - nlo) + u) /
                        static_cast<float>(nfail);
                    src_trial =
                        std::min(static_cast<uint32_t>(
                                     quantile * static_cast<float>(count)),
                                 count - 1U);
                }
                const SizeType src_row =
                    (static_cast<SizeType>(m_src[prev][cell]) * ntrials) +
                    survivors[src_trial];
                const SizeType out_row =
                    (static_cast<SizeType>(cell) * ntrials) + trial;
                const float* in = m_folds[prev].data() + (src_row * m_nbins);
                const float* nz =
                    noise.data() + (static_cast<SizeType>(trial) * m_nbins);
                float* out = m_folds[cur].data() + (out_row * m_nbins);
                for (SizeType j = 0; j < m_nbins; ++j) {
                    out[j] = in[j] + nz[j];
                }
                m_scores[out_row] = detection::snr_boxcar_max_with_cache(
                    std::span<const float>(out, m_nbins), m_nbins,
                    caches[static_cast<SizeType>(omp_get_thread_num())],
                    stdnoise);
            }

            // Every transition: counts, candidate state and its rank.
            const SizeType n_items = plan.pairs.size() * nprobs;
            std::vector<uint2_t> counts(n_items);
            std::vector<State> cand(n_items);
            std::vector<unsigned long long> keys(n_items, kEmptyKey);
            std::vector<uint32_t> key_idx(n_items, 0U);
            const float weight = rank_weight(istage);
            // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
            for (std::int64_t it = 0; it < static_cast<std::int64_t>(n_items);
                 ++it) {
                const auto item          = static_cast<uint32_t>(it);
                const ThresholdPair pair = plan.pairs[item / nprobs];
                const uint32_t kprob     = item % nprobs;
                const uint32_t cell      = (pair.jslot * nprobs) + kprob;
                if (prev_states[(pair.jthres * nprobs) + kprob].is_empty ||
                    m_ntrials_desc[prev][2 * cell] == 0 ||
                    m_ntrials_desc[prev][(2 * cell) + 1] == 0) {
                    continue;
                }
                const float threshold = m_thresholds[pair.ithres];
                const float* scores =
                    m_scores.data() + (static_cast<SizeType>(cell) * ntrials);
                uint32_t count_h0 = 0;
                for (uint32_t t = 0; t < ntrials; ++t) {
                    count_h0 += scores[t] > threshold ? 1U : 0U;
                }
                const uint32_t* mask = m_h1_mask[prev].data() +
                                       (static_cast<SizeType>(cell) * h1_words);
                const uint32_t* pass =
                    m_h1_pass.data() +
                    (static_cast<SizeType>(pair.islot) * h1_words);
                uint32_t count_h1 = 0;
                for (uint32_t w = 0; w < h1_words; ++w) {
                    count_h1 +=
                        static_cast<uint32_t>(std::popcount(mask[w] & pass[w]));
                }
                // No noise survivor: the best trial stands in, counted as one.
                count_h0     = std::max(count_h0, 1U);
                counts[item] = {count_h0, count_h1};
                const float ntrials_h1 =
                    static_cast<float>(m_ntrials_desc[prev][(2 * cell) + 1]);
                const State next =
                    prev_states[(pair.jthres * nprobs) + kprob].gen_next(
                        threshold,
                        static_cast<float>(std::max(count_h0, support)) /
                            ntrials_f,
                        static_cast<float>(count_h1) / ntrials_h1, nbranches);
                cand[item] = next;
                const int iprob =
                    find_bin_index(m_probs, next.success_h1_cumul);
                if (iprob >= 0 && iprob < static_cast<int>(nprobs)) {
                    const float predicted =
                        weight > 0.0F
                            ? next.complexity_cumul + (weight * next.complexity)
                            : next.complexity_cumul;
                    const float rank = predicted / next.success_h1_cumul;
                    key_idx[item] =
                        (pair.ithres * nprobs) + static_cast<uint32_t>(iprob);
                    keys[item] = (static_cast<unsigned long long>(
                                      std::bit_cast<uint32_t>(rank))
                                  << 32U) |
                                 item;
                }
            }
            // The lowest key of every output cell wins (ties: lowest item).
            std::vector<unsigned long long> winner(m_nthresholds * nprobs,
                                                   kEmptyKey);
            for (SizeType item = 0; item < n_items; ++item) {
                if (keys[item] != kEmptyKey) {
                    winner[key_idx[item]] =
                        std::min(winner[key_idx[item]], keys[item]);
                }
            }

            // Each output cell takes its winner: state, counts, survivor map
            // into the parent's simulated folds, and the signal bitmask.
            const SizeType n_out = plan.beam_cur.size() * nprobs;
            // NOLINTNEXTLINE(openmp-use-default-none): clauses cannot name members
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
            for (std::int64_t oc = 0; oc < static_cast<std::int64_t>(n_out);
                 ++oc) {
                const auto out_cell = static_cast<uint32_t>(oc);
                const uint32_t ithres =
                    static_cast<uint32_t>(plan.beam_cur[out_cell / nprobs]);
                const uint32_t kidx = (ithres * nprobs) + (out_cell % nprobs);
                const unsigned long long key = winner[kidx];
                if (key == kEmptyKey) {
                    m_ntrials_desc[cur][2 * out_cell]       = 0;
                    m_ntrials_desc[cur][(2 * out_cell) + 1] = 0;
                    continue;
                }
                const auto item = static_cast<uint32_t>(key & 0xFFFFFFFFULL);
                const ThresholdPair pair = plan.pairs[item / nprobs];
                const uint32_t par_cell =
                    (pair.jslot * nprobs) + (item % nprobs);
                const float threshold                   = m_thresholds[ithres];
                cur_states[kidx]                        = cand[item];
                m_ntrials_desc[cur][2 * out_cell]       = counts[item].x;
                m_ntrials_desc[cur][(2 * out_cell) + 1] = counts[item].y;
                m_src[cur][out_cell]                    = par_cell;
                const uint32_t* mask =
                    m_h1_mask[prev].data() +
                    (static_cast<SizeType>(par_cell) * h1_words);
                const uint32_t* pass =
                    m_h1_pass.data() +
                    (static_cast<SizeType>(out_cell / nprobs) * h1_words);
                uint32_t* mout = m_h1_mask[cur].data() +
                                 (static_cast<SizeType>(out_cell) * h1_words);
                for (uint32_t w = 0; w < h1_words; ++w) {
                    mout[w] = mask[w] & pass[w];
                }
                const float* scores =
                    m_scores.data() +
                    (static_cast<SizeType>(par_cell) * ntrials);
                uint32_t* idx_out = m_idx[cur].data() +
                                    (static_cast<SizeType>(out_cell) * ntrials);
                uint32_t n        = 0;
                for (uint32_t t = 0; t < ntrials; ++t) {
                    if (scores[t] > threshold) {
                        idx_out[n++] = t;
                    }
                }
                if (n == 0) {
                    idx_out[0] = static_cast<uint32_t>(
                        std::max_element(scores, scores + ntrials) - scores);
                }
            }
            var_in += 1.0F;
            bar.set_progress(istage + 1);
        }
        bar.mark_as_completed();
    }

    struct uint2_t {
        uint32_t x;
        uint32_t y;
    };

    std::vector<float> m_branching_pattern;
    float m_ref_ducy;
    SizeType m_ntrials;
    float m_ducy_max;
    float m_wtsp;
    float m_beam_width;
    SizeType m_nbins;
    int m_nthreads;
    uint64_t m_seed;
    float m_min_pd;
    double m_w;
    std::string m_guess;
    DynamicThresholdMode m_mode;
    float m_guide_width;
    float m_guide_reach;

    std::unique_ptr<detail::ThresholdsEngine> m_eval;
    std::unique_ptr<detection::BoxcarWidthsCache> m_box_cache;
    std::vector<float> m_profile;
    std::vector<float> m_thresholds;
    std::vector<float> m_probs;
    std::vector<SizeType> m_box_score_widths;
    std::vector<float> m_guess_path;
    detail::BeamBounds m_beam;
    // Guided beam: feasibility window, pilot guess, signal score std per
    // stage, beam half-width and reach (both in sigma).
    detail::BeamBounds m_window;
    std::vector<float> m_guess_pilot;
    std::vector<float> m_score_sigma;
    SizeType m_nprobs{};
    SizeType m_nstages{};
    SizeType m_nthresholds{};
    SizeType m_nh1{};
    SizeType m_h1_words{};
    SizeType m_max_beam{};
    std::vector<float> m_cost_to_go;
    std::vector<State> m_states;
    std::vector<State> m_first_states;
    std::vector<float> m_h1_bank;

    // Ping-pong pools, index (s & 1) written at stage s: simulated noise
    // folds of the stage's parent cells, per-cell [H0, H1] counts, survivor
    // maps (src: parent cell; idx: [cell][trial]) and the signal bitmasks.
    std::array<std::vector<float>, 2> m_folds;
    std::array<std::vector<uint32_t>, 2> m_ntrials_desc;
    std::array<std::vector<uint32_t>, 2> m_src;
    std::array<std::vector<uint32_t>, 2> m_idx;
    std::vector<float> m_scores;
    std::array<std::vector<uint32_t>, 2> m_h1_mask;
    std::vector<uint32_t> m_h1_pass;
};

} // namespace

namespace detail {

std::unique_ptr<ThresholdsEngine>
make_thresholds_shared_cpu(std::span<const float> branching_pattern,
                           float ref_ducy,
                           SizeType nbins,
                           SizeType ntrials,
                           SizeType nprobs,
                           float snr_final,
                           SizeType nthresholds,
                           float ducy_max,
                           float wtsp,
                           float beam_width,
                           SizeType trials_start,
                           int nthreads,
                           std::optional<uint64_t> seed,
                           std::string_view guess,
                           std::string_view mode,
                           float min_pd,
                           float w_terminal,
                           float beam_sigma,
                           float reach_sigma) {
    return std::make_unique<ThresholdsSharedCpuEngine>(
        branching_pattern, ref_ducy, nbins, ntrials, nprobs, snr_final,
        nthresholds, ducy_max, wtsp, beam_width, trials_start, nthreads, seed,
        guess, dynamic_threshold_mode_from_string(mode), min_pd, w_terminal,
        beam_sigma, reach_sigma);
}

} // namespace detail

} // namespace loki::detection
