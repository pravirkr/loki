#include "lib/detection/scheme.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/math.hpp"
#include "lib/detail/utils.hpp"
#include "lib/detection/thresholds_engine.hpp"

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

std::vector<float>
shared_probability_grid(float min_pd, SizeType n_above, SizeType max_below) {
    error_check::check_greater(n_above, 1,
                               "Number of probabilities must be > 1");
    error_check::check(min_pd > 0.0F && min_pd < 1.0F,
                       "shared mode: min_pd must be in (0, 1)");
    error_check::check_greater(max_below, 0,
                               "shared mode: max_below must be positive");
    const double step = -std::log10(static_cast<double>(min_pd)) /
                        static_cast<double>(n_above - 1);
    const auto below = static_cast<SizeType>(std::ceil(std::log10(2.0) / step));
    const auto prob_min =
        static_cast<float>(static_cast<double>(min_pd) *
                           std::pow(10.0, -static_cast<double>(below) * step));
    // Edge k of the full grid, as compute_probs(n_above + below, prob_min).
    const SizeType nfull     = n_above + below;
    const float log_prob_min = std::log10(prob_min);
    const float fstep = (0.0F - log_prob_min) / static_cast<float>(nfull - 1);
    const auto edge   = [&](SizeType k) {
        return std::pow(10.0F, log_prob_min + (fstep * static_cast<float>(k)));
    };
    std::vector<float> probs;
    if (below > max_below) {
        probs.reserve(max_below + n_above);
        for (SizeType i = 0; i < max_below; ++i) {
            probs.push_back(edge(i * below / max_below));
        }
        for (SizeType k = below; k < nfull; ++k) {
            probs.push_back(edge(k));
        }
    } else {
        probs.reserve(nfull);
        for (SizeType k = 0; k < nfull; ++k) {
            probs.push_back(edge(k));
        }
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

BeamGuess beam_guess_from_string(std::string_view guess) {
    if (guess == "analytic") {
        return BeamGuess::kAnalytic;
    }
    if (guess == "pilot") {
        return BeamGuess::kPilot;
    }
    if (guess == "window") {
        return BeamGuess::kWindow;
    }
    throw std::invalid_argument(
        std::format("unknown beam guess '{}' (expected 'analytic', 'pilot' "
                    "or 'window')",
                    guess));
}

uint64_t derive_seed(uint64_t seed, uint64_t salt) noexcept {
    uint64_t x = seed ^ salt;
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

std::vector<float> pilot_guess_scheme(std::span<const float> h1_scores,
                                      SizeType ntrials,
                                      std::span<const float> branching_pattern,
                                      SizeType trials_start,
                                      float min_trials) {
    const auto nstages = branching_pattern.size();
    error_check::check_equal(h1_scores.size(), nstages * ntrials,
                             "h1_scores size does not match nstages x ntrials");
    const auto thresholds_trials =
        trials_scheme(branching_pattern, trials_start, min_trials);
    std::vector<float> result(nstages);
    std::vector<float> row(ntrials);
    for (SizeType s = 0; s < nstages; ++s) {
        std::copy_n(h1_scores.begin() +
                        static_cast<std::ptrdiff_t>(s * ntrials),
                    ntrials, row.begin());
        const auto mid = row.begin() + static_cast<std::ptrdiff_t>(ntrials / 2);
        std::ranges::nth_element(row, mid);
        result[s] = std::min(*mid, thresholds_trials[s]);
    }
    return result;
}

BeamBounds centred_beam(std::span<const float> guess_path,
                        float beam_width,
                        float snr_max) {
    BeamBounds beam;
    beam.lo.reserve(guess_path.size());
    beam.hi.reserve(guess_path.size());
    for (const float guess : guess_path) {
        beam.lo.push_back(std::max(0.0F, guess - beam_width));
        beam.hi.push_back(std::min(snr_max, guess + beam_width));
    }
    return beam;
}

BeamBounds window_scheme(std::span<const float> h1_scores,
                         SizeType ntrials,
                         SizeType nstages,
                         float prob_min,
                         float eps) {
    error_check::check_equal(h1_scores.size(), nstages * ntrials,
                             "h1_scores size does not match nstages x ntrials");
    error_check::check(ntrials > 0, "window_scheme: ntrials must be positive");
    const auto n = static_cast<double>(ntrials);
    const auto q_lo =
        static_cast<double>(eps * prob_min) / static_cast<double>(nstages);
    const auto q_hi = 1.0 - (0.5 * static_cast<double>(prob_min));
    const auto rank = [&](double q) {
        const auto r = static_cast<SizeType>(std::floor(q * n));
        return std::min(r, ntrials - 1);
    };
    const auto r_lo = rank(q_lo);
    const auto r_hi = rank(q_hi);
    BeamBounds beam{.lo = std::vector<float>(nstages),
                    .hi = std::vector<float>(nstages)};
    std::vector<float> row(ntrials);
    for (SizeType s = 0; s < nstages; ++s) {
        std::copy_n(h1_scores.begin() +
                        static_cast<std::ptrdiff_t>(s * ntrials),
                    ntrials, row.begin());
        // Two order statistics: the same values as a full sort.
        const auto high = row.begin() + static_cast<std::ptrdiff_t>(r_hi);
        std::ranges::nth_element(row, high);
        beam.hi[s] = *high;
        if (r_lo < r_hi) {
            std::ranges::nth_element(
                row.begin(), row.begin() + static_cast<std::ptrdiff_t>(r_lo),
                high);
        }
        beam.lo[s] = row[r_lo];
    }
    return beam;
}

std::vector<float> score_sigma(std::span<const float> h1_scores,
                               SizeType ntrials,
                               SizeType nstages) {
    error_check::check_equal(h1_scores.size(), nstages * ntrials,
                             "h1_scores size does not match nstages x ntrials");
    error_check::check(ntrials > 1, "score_sigma: ntrials must be > 1");
    std::vector<float> sigma(nstages);
    for (SizeType s = 0; s < nstages; ++s) {
        const auto row = h1_scores.subspan(s * ntrials, ntrials);
        double sum     = 0.0;
        double sum2    = 0.0;
        for (const float x : row) {
            sum += x;
            sum2 += static_cast<double>(x) * x;
        }
        const auto n    = static_cast<double>(ntrials);
        const auto mean = sum / n;
        sigma[s]        = static_cast<float>(
            std::sqrt(std::max(0.0, (sum2 - (n * mean * mean)) / (n - 1.0))));
    }
    return sigma;
}

void validate_scheme_inputs(const SchemeInputs& in, std::string_view who) {
    const auto fail = [who](const std::string& msg) {
        throw std::invalid_argument(std::format("{}: {}", who, msg));
    };
    for (const float x : {in.ref_ducy, in.ducy_max, in.wtsp, in.beam_width,
                          in.prob_min, in.snr_final}) {
        if (!utils::is_finite(x)) {
            fail("float parameters must be finite");
        }
    }
    if (in.branching_pattern.size() < 2) {
        fail("branching pattern needs at least 2 stages");
    }
    for (SizeType i = 0; i < in.branching_pattern.size(); ++i) {
        const float b = in.branching_pattern[i];
        if (!utils::is_finite(b) || b <= 0.0F) {
            fail(std::format("branching_pattern[{}] = {} must be finite and "
                             "positive",
                             i, b));
        }
    }
    if (in.nbins < 4 || in.nbins > 1024) {
        fail(std::format("nbins={} must be in [4, 1024]", in.nbins));
    }
    if (in.ntrials == 0 || in.nprobs == 0 || in.batch_size == 0) {
        fail("ntrials, nprobs and batch_size must be positive");
    }
    if (in.nthresholds < 2 || !(in.snr_final > 0.1F)) {
        fail("need nthresholds >= 2 and a finite snr_final > 0.1");
    }
    if (!(in.prob_min > 0.0F && in.prob_min < 1.0F)) {
        fail(std::format("prob_min={} must be in (0, 1)", in.prob_min));
    }
    if (!(in.ref_ducy > 0.0F && in.ref_ducy < 1.0F) ||
        !(in.ducy_max > 0.0F && in.ducy_max < 1.0F)) {
        fail("ref_ducy and ducy_max must be in (0, 1)");
    }
    if (!(in.wtsp > 0.0F) || !(in.beam_width > 0.0F)) {
        fail("wtsp and beam_width must be positive");
    }
}

void warn_empty_stages(std::span<const State> states,
                       SizeType nstages,
                       SizeType nthresholds,
                       SizeType nprobs,
                       std::string_view who) {
    const auto stride = nthresholds * nprobs;
    for (SizeType istage = 0; istage < nstages; ++istage) {
        const auto stage = states.subspan(istage * stride, stride);
        if (std::ranges::all_of(stage,
                                [](const State& st) { return st.is_empty; })) {
            spdlog::warn("{}: stage {} (and all later stages) has no "
                         "surviving state",
                         who, istage);
            return;
        }
    }
}

PilotStats pilot_stats(std::span<const float> sorted_scores,
                       SizeType ntrials,
                       std::span<const float> branching_pattern,
                       SizeType trials_start,
                       float prob_min,
                       float eps) {
    const auto nstages = branching_pattern.size();
    error_check::check_equal(sorted_scores.size(), nstages * ntrials,
                             "sorted_scores size does not match nstages x "
                             "ntrials");
    error_check::check(ntrials > 1, "pilot_stats: ntrials must be > 1");
    // The ranks of window_scheme and pilot_guess_scheme.
    const auto n = static_cast<double>(ntrials);
    const auto q_lo =
        static_cast<double>(eps * prob_min) / static_cast<double>(nstages);
    const auto q_hi = 1.0 - (0.5 * static_cast<double>(prob_min));
    const auto rank = [&](double q) {
        return std::min(static_cast<SizeType>(std::floor(q * n)), ntrials - 1);
    };
    const auto r_lo   = rank(q_lo);
    const auto r_hi   = rank(q_hi);
    const auto trials = trials_scheme(branching_pattern, trials_start, 1E10F);
    PilotStats out{.window = {.lo = std::vector<float>(nstages),
                              .hi = std::vector<float>(nstages)},
                   .guess  = std::vector<float>(nstages),
                   .sigma  = score_sigma(sorted_scores, ntrials, nstages)};
    for (SizeType s = 0; s < nstages; ++s) {
        const auto row   = sorted_scores.subspan(s * ntrials, ntrials);
        out.window.lo[s] = row[r_lo];
        out.window.hi[s] = row[r_hi];
        out.guess[s]     = std::min(row[ntrials / 2], trials[s]);
    }
    return out;
}

BeamBounds guided_beam(const BeamBounds& window,
                       std::span<const float> guess_path,
                       std::span<const float> sigma,
                       float width) {
    const auto nstages = guess_path.size();
    error_check::check(window.lo.size() == nstages &&
                           window.hi.size() == nstages &&
                           sigma.size() == nstages,
                       "guided_beam: size mismatch");
    BeamBounds beam{.lo = std::vector<float>(nstages),
                    .hi = std::vector<float>(nstages)};
    for (SizeType s = 0; s < nstages; ++s) {
        const float g = std::clamp(guess_path[s], window.lo[s], window.hi[s]);
        beam.lo[s]    = std::max(window.lo[s], g - (width * sigma[s]));
        beam.hi[s]    = std::min(window.hi[s], g + (width * sigma[s]));
    }
    return beam;
}

SizeType guided_beam_capacity(std::span<const float> thresholds,
                              const BeamBounds& window,
                              std::span<const float> sigma,
                              float width) {
    error_check::check(thresholds.size() > 1,
                       "guided_beam_capacity: needs a threshold grid");
    const float step  = thresholds[1] - thresholds[0];
    SizeType capacity = 0;
    for (SizeType s = 0; s < sigma.size(); ++s) {
        const auto band =
            static_cast<SizeType>(std::ceil(2.0F * width * sigma[s] / step)) +
            2;
        const auto in_window =
            beam_indices(thresholds, window.lo[s], window.hi[s]).size();
        capacity = std::max(capacity, std::min(band, in_window));
    }
    return capacity;
}

std::vector<float> recentre_guess(std::span<const float> guess_path,
                                  std::span<const float> path) {
    std::vector<float> guess(guess_path.begin(), guess_path.end());
    if (path.size() != guess.size()) {
        return guess;
    }
    const auto n = guess.size();
    for (SizeType s = 0; s < n; ++s) {
        const SizeType a = s == 0 ? 0 : s - 1;
        const SizeType b = std::min(s + 1, n - 1);
        float offset     = 0.0F;
        for (SizeType k = a; k <= b; ++k) {
            offset += path[k] - guess_path[k];
        }
        guess[s] += offset / static_cast<float>(b - a + 1);
    }
    return guess;
}

std::vector<SizeType>
beam_indices(std::span<const float> thresholds, float lo, float hi) {
    std::vector<SizeType> result;
    for (SizeType i = 0; i < thresholds.size(); ++i) {
        if (thresholds[i] >= lo && thresholds[i] <= hi) {
            result.push_back(i);
        }
    }
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
        SizeType last_live   = 0;
        float best_pd        = 0.0F;
        bool any_live        = false;
        const SizeType cells = nthresholds * nprobs;
        for (SizeType stage = 0; stage < nstages; ++stage) {
            const SizeType base = stage * cells;
            float stage_best    = 0.0F;
            bool stage_live     = false;
            for (SizeType i = 0; i < cells; ++i) {
                const State& s = states[base + i];
                if (s.is_empty) {
                    continue;
                }
                stage_live = true;
                stage_best = std::max(stage_best, s.success_h1_cumul);
            }
            if (stage_live) {
                any_live  = true;
                last_live = stage;
                best_pd   = stage_best;
            }
        }
        if (!any_live) {
            spdlog::warn(
                "get_best_path_thresholds: no path for min_pd={:.4g}; every "
                "stage is empty (extinction at stage 0)",
                min_pd);
        } else if (last_live + 1 < nstages) {
            spdlog::warn(
                "get_best_path_thresholds: no path for min_pd={:.4g}; "
                "extinction at stage {} (last surviving stage {} had best "
                "cumulative Pd {:.4g})",
                min_pd, last_live + 1, last_live, best_pd);
        } else {
            spdlog::warn(
                "get_best_path_thresholds: no path for min_pd={:.4g}; best "
                "cumulative Pd on the final stage is {:.4g}",
                min_pd, best_pd);
        }
        return {};
    }

    const auto terminal_cell =
        static_cast<SizeType>(best - (states.data() + stage_last));
    return backtrack_path(states, thresholds, probs, nstages, nthresholds,
                          nprobs, terminal_cell);
}

std::vector<State> backtrack_states(std::span<const State> states,
                                    std::span<const float> thresholds,
                                    std::span<const float> probs,
                                    SizeType nstages,
                                    SizeType nthresholds,
                                    SizeType nprobs,
                                    SizeType terminal_cell) {
    error_check::check_equal(
        states.size(), nstages * nthresholds * nprobs,
        "states span size does not match nstages × nthresholds × nprobs");
    error_check::check_less(terminal_cell, nthresholds * nprobs,
                            "terminal cell out of range");
    const SizeType stage_last = (nstages - 1) * nthresholds * nprobs;
    const State* best         = &states[stage_last + terminal_cell];
    if (best->is_empty) {
        throw std::invalid_argument("backtrack_path: terminal cell is empty");
    }
    std::vector<State> path(nstages);
    path[nstages - 1] = *best;

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
        path[istage]          = prev_state;
        prev_threshold        = prev_state.threshold_prev;
        prev_success_h1_cumul = prev_state.success_h1_cumul_prev;
    }
    return path;
}

std::vector<float> backtrack_path(std::span<const State> states,
                                  std::span<const float> thresholds,
                                  std::span<const float> probs,
                                  SizeType nstages,
                                  SizeType nthresholds,
                                  SizeType nprobs,
                                  SizeType terminal_cell) {
    const auto path = backtrack_states(states, thresholds, probs, nstages,
                                       nthresholds, nprobs, terminal_cell);
    std::vector<float> out(nstages);
    std::ranges::transform(path, out.begin(),
                           [](const State& s) { return s.threshold; });
    return out;
}

double terminal_rank(const State& s, double w_terminal) {
    if (w_terminal == 0.0) {
        return static_cast<double>(s.cost);
    }
    return total_cost(s, w_terminal) / static_cast<double>(s.success_h1_cumul);
}

void store_terminal_cost(std::span<State> states,
                         SizeType nstages,
                         SizeType nthresholds,
                         SizeType nprobs,
                         double w_terminal) {
    if (w_terminal == 0.0 || nstages == 0) {
        return;
    }
    const auto last = states.subspan((nstages - 1) * nthresholds * nprobs,
                                     nthresholds * nprobs);
    for (auto& state : last) {
        if (!state.is_empty && state.success_h1_cumul > 0.0F) {
            state.cost = static_cast<float>(terminal_rank(state, w_terminal));
        }
    }
}

SizeType best_terminal_cell(std::span<const State> states,
                            SizeType nstages,
                            SizeType nthresholds,
                            SizeType nprobs,
                            float min_pd,
                            double w_terminal) {
    const SizeType stage_last = (nstages - 1) * nthresholds * nprobs;
    SizeType best             = std::numeric_limits<SizeType>::max();
    double best_cost          = std::numeric_limits<double>::max();
    for (SizeType i = 0; i < nthresholds * nprobs; ++i) {
        const State& s = states[stage_last + i];
        if (s.is_empty || s.success_h1_cumul < min_pd) {
            continue;
        }
        const double c = terminal_rank(s, w_terminal);
        if (c < best_cost) {
            best_cost = c;
            best      = i;
        }
    }
    return best;
}

std::vector<float> cost_to_go_scheme(std::span<const State> states,
                                     std::span<const float> thresholds,
                                     std::span<const float> probs,
                                     SizeType nstages,
                                     SizeType nthresholds,
                                     SizeType nprobs,
                                     float pd_max_factor,
                                     SizeType half_window,
                                     double w_terminal) {
    const SizeType stage_last = (nstages - 1) * nthresholds * nprobs;
    const float pd_hi         = probs.front() * pd_max_factor;
    // log h per stage of every target path, stage-major
    std::vector<std::vector<double>> log_h(nstages);
    for (SizeType iprob = 0; iprob < nprobs; ++iprob) {
        if (probs[iprob] >= pd_hi) {
            break;
        }
        // The cheapest terminal cell of this probability bin
        SizeType best_cell = nthresholds * nprobs;
        double best_cost   = std::numeric_limits<double>::max();
        for (SizeType ithres = 0; ithres < nthresholds; ++ithres) {
            const auto cell   = (ithres * nprobs) + iprob;
            const State& term = states[stage_last + cell];
            if (!term.is_empty && terminal_rank(term, w_terminal) < best_cost) {
                best_cost = terminal_rank(term, w_terminal);
                best_cell = cell;
            }
        }
        if (best_cell == nthresholds * nprobs) {
            continue;
        }
        const auto path = backtrack_states(states, thresholds, probs, nstages,
                                           nthresholds, nprobs, best_cell);
        const auto c_final =
            w_terminal == 0.0
                ? static_cast<double>(path.back().complexity_cumul)
                : total_cost(path.back(), w_terminal);
        for (SizeType s = 0; s + 1 < nstages; ++s) {
            const auto n = std::max(static_cast<double>(path[s].complexity),
                                    std::numeric_limits<double>::min());
            const auto h =
                (c_final - static_cast<double>(path[s].complexity_cumul)) / n;
            log_h[s].push_back(std::log(std::max(h, 1e-3)));
        }
    }
    if (nstages < 2 || log_h.front().empty()) {
        return {};
    }
    // Median over paths, then a moving average over stages (in log space).
    // The last stage has no future and stays out of both.
    const SizeType last = nstages - 1;
    std::vector<double> med(last);
    for (SizeType s = 0; s < last; ++s) {
        auto& v        = log_h[s];
        const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
        std::ranges::nth_element(v, mid);
        med[s] = *mid;
    }
    std::vector<float> out(nstages, 0.0F);
    for (SizeType s = 0; s < last; ++s) {
        const SizeType lo = s > half_window ? s - half_window : 0;
        const SizeType hi = std::min(last - 1, s + half_window);
        double sum        = 0.0;
        for (SizeType j = lo; j <= hi; ++j) {
            sum += med[j];
        }
        out[s] = static_cast<float>(
            std::exp(sum / static_cast<double>(hi - lo + 1)));
    }
    out[last] = static_cast<float>(w_terminal); // no future: only the weight
    return out;
}

SearchPath search_path(std::span<const State> states,
                       std::span<const float> thresholds,
                       std::span<const float> probs,
                       SizeType nstages,
                       SizeType nthresholds,
                       SizeType nprobs,
                       SizeType terminal_cell,
                       double w_terminal) {
    SearchPath out;
    if (terminal_cell == std::numeric_limits<SizeType>::max()) {
        return out;
    }
    const auto path = backtrack_states(states, thresholds, probs, nstages,
                                       nthresholds, nprobs, terminal_cell);
    out.thresholds.reserve(nstages);
    for (const auto& s : path) {
        out.thresholds.push_back(s.threshold);
    }
    out.pd = static_cast<double>(path.back().success_h1_cumul);
    out.c  = total_cost(path.back(), w_terminal);
    return out;
}

SizeType count_survivors(std::span<const float> h1_scores,
                         std::span<const float> thresholds,
                         SizeType ntrials) {
    const SizeType nstages = thresholds.size();
    error_check::check_equal(h1_scores.size(), nstages * ntrials,
                             "h1_scores size does not match nstages x ntrials");
    std::vector<char> alive(ntrials, 1);
    for (SizeType s = 0; s < nstages; ++s) {
        const float* row = h1_scores.data() + (s * ntrials);
        const float t    = thresholds[s];
        for (SizeType i = 0; i < ntrials; ++i) {
            alive[i] = static_cast<char>(alive[i] != 0 && row[i] > t);
        }
    }
    return static_cast<SizeType>(std::ranges::count(alive, 1));
}

namespace {

std::vector<float> shifted(std::span<const float> path, float delta) {
    std::vector<float> out(path.size());
    std::ranges::transform(path, out.begin(),
                           [delta](float t) { return t + delta; });
    return out;
}

struct Shift {
    float delta;
    SizeType count;
    bool clamped;
};

/// Largest shift delta in [-delta_max, hi_max] that keeps at least `target`
/// trials, counted exactly on the shifted float thresholds. The closed form
/// (midpoint of the target-th and next margins) is exact up to rounding;
/// a short bisection covers ties and rounding.
Shift solve_shift(std::span<const float> h1_scores,
                  std::span<const float> path,
                  SizeType ntrials,
                  SizeType target,
                  float delta_max,
                  bool allow_raise) {
    const SizeType nstages = path.size();
    // margin_i = min_s (S_is - t_s): trial i survives t + delta iff
    // margin_i > delta
    std::vector<float> margin(ntrials, std::numeric_limits<float>::max());
    for (SizeType s = 0; s < nstages; ++s) {
        const float* row = h1_scores.data() + (s * ntrials);
        for (SizeType i = 0; i < ntrials; ++i) {
            margin[i] = std::min(margin[i], row[i] - path[s]);
        }
    }
    std::ranges::sort(margin, std::greater<>());
    const float upper = margin[target - 1];
    const float lower =
        target < ntrials ? margin[target] : upper - 1.0F; // next margin
    const float lo_bound = -delta_max;
    const float hi_bound = allow_raise ? delta_max : 0.0F;
    float delta          = 0.5F * (upper + lower);
    bool clamped         = false;
    if (delta < lo_bound || delta > hi_bound) {
        delta   = std::clamp(delta, lo_bound, hi_bound);
        clamped = true;
    }
    auto count_at = [&](float d) {
        return count_survivors(h1_scores, shifted(path, d), ntrials);
    };
    SizeType count = count_at(delta);
    if (!clamped && count < target) {
        // Rounding moved a survivor across: bisect towards lower shifts.
        float hi = delta;
        float lo = std::max(lo_bound, lower - 1.0F);
        if (count_at(lo) >= target) {
            for (int it = 0; it < 48 && hi - lo > 0.0F; ++it) {
                const float mid = 0.5F * (lo + hi);
                if (mid <= lo || mid >= hi) {
                    break;
                }
                if (count_at(mid) >= target) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            delta = lo;
            count = count_at(delta);
        }
    }
    return {delta, count, clamped};
}

} // namespace

CalibratedPath calibrate_path(const ThresholdsEngine& engine,
                              float min_pd,
                              const CalibrationOptions& opts) {
    if (!(min_pd > 0.0F && min_pd < 1.0F)) {
        throw std::invalid_argument("calibrate: min_pd must be in (0, 1)");
    }
    const SizeType kMaxTrials = SizeType{1} << 20U;
    if (opts.ntrials_cal == 0 || opts.ntrials_cal > kMaxTrials ||
        opts.ntrials_cost == 0 || opts.ntrials_cost > kMaxTrials) {
        throw std::invalid_argument(
            "calibrate: ntrials_cal and ntrials_cost must be in [1, 2^20]");
    }
    if (opts.ncandidates == 0) {
        throw std::invalid_argument("calibrate: ncandidates must be positive");
    }
    if (!(opts.delta_max >= 0.0F)) {
        throw std::invalid_argument("calibrate: delta_max must be >= 0");
    }
    const auto states      = engine.get_states();
    const auto thresholds  = engine.get_thresholds();
    const auto probs       = engine.get_probs();
    const SizeType nstages = engine.get_nstages();
    const SizeType nthr    = engine.get_nthresholds();
    const SizeType nprobs  = engine.get_nprobs();
    const SizeType cells   = nthr * nprobs;
    const State* terminal  = states.data() + ((nstages - 1) * cells);
    const auto by_cost     = [&](SizeType a, SizeType b) {
        return terminal[a].cost < terminal[b].cost ||
               (terminal[a].cost == terminal[b].cost && a < b);
    };

    // Candidates: cheapest cells near min_pd, always with the selection of
    // get_best_path_thresholds(min_pd).
    std::vector<SizeType> live;
    std::vector<SizeType> near;
    for (SizeType i = 0; i < cells; ++i) {
        if (terminal[i].is_empty) {
            continue;
        }
        live.push_back(i);
        if (terminal[i].success_h1_cumul >= opts.candidate_frac * min_pd) {
            near.push_back(i);
        }
    }
    if (live.empty()) {
        spdlog::warn("calibrate: no terminal path to calibrate");
        return {};
    }
    std::vector<SizeType> cand;
    if (!near.empty()) {
        std::ranges::sort(near, by_cost);
        cand.assign(near.begin(),
                    near.begin() + static_cast<std::ptrdiff_t>(std::min(
                                       opts.ncandidates, near.size())));
        const auto it = std::ranges::find_if(near, [&](SizeType i) {
            return terminal[i].success_h1_cumul >= min_pd;
        });
        if (it != near.end() && std::ranges::find(cand, *it) == cand.end()) {
            cand.push_back(*it);
        }
    } else {
        // Nothing near the target: the highest in-run Pd paths
        std::ranges::sort(live, [&](SizeType a, SizeType b) {
            return terminal[a].success_h1_cumul > terminal[b].success_h1_cumul;
        });
        cand.assign(live.begin(),
                    live.begin() + static_cast<std::ptrdiff_t>(std::min(
                                       opts.ncandidates, live.size())));
    }

    const uint64_t seed_cal  = opts.seed.value_or(std::random_device{}());
    const uint64_t seed_cost = derive_seed(seed_cal, 0xC0575EEDULL);
    const SizeType ntrials   = opts.ntrials_cal;
    const auto h1_scores     = engine.simulate_h1_scores(ntrials, seed_cal);
    const auto target        = std::clamp<SizeType>(
        static_cast<SizeType>(std::ceil(static_cast<double>(min_pd) *
                                        static_cast<double>(ntrials))),
        1, ntrials);

    CalibratedPath best;
    double best_cost = std::numeric_limits<double>::max();
    bool any_clamped = false;
    // One shifted path, scored on the cost sample. All variants share the
    // cost seed, so their costs are compared on the same noise.
    const auto consider = [&](SizeType cell, std::span<const float> path,
                              float delta, SizeType count) {
        if (count == 0) {
            return;
        }
        auto thresh = shifted(path, delta);
        auto ev = engine.evaluate(thresh, opts.ntrials_cost, seed_cost, false);
        const State& fin = ev.back();
        if (fin.is_empty) {
            return; // the cost evaluation lost every signal trial
        }
        const double pd_cert =
            static_cast<double>(count) / static_cast<double>(ntrials);
        const double cost = static_cast<double>(fin.complexity_cumul) / pd_cert;
        if (cost < best_cost) {
            best_cost             = cost;
            best.thresholds       = std::move(thresh);
            best.delta            = delta;
            best.pd_certified     = static_cast<float>(pd_cert);
            best.pd_sigma         = static_cast<float>(std::sqrt(
                pd_cert * (1.0 - pd_cert) / static_cast<double>(ntrials)));
            best.pd_check         = fin.success_h1_cumul;
            best.complexity_cumul = fin.complexity_cumul;
            best.cost             = static_cast<float>(cost);
            best.pd_inrun         = terminal[cell].success_h1_cumul;
            best.cost_inrun       = terminal[cell].cost;
            best.states           = std::move(ev);
        }
    };
    for (const SizeType cell : cand) {
        const auto path  = backtrack_path(states, thresholds, probs, nstages,
                                          nthr, nprobs, cell);
        const auto shift = solve_shift(h1_scores, path, ntrials, target,
                                       opts.delta_max, opts.allow_raise);
        any_clamped      = any_clamped || shift.clamped;
        consider(cell, path, shift.delta, shift.count);
        // A path that already meets min_pd is also kept as it is: raising
        // its thresholds to min_pd exactly can cost more per unit Pd than it
        // saves. The lower cost C / Pd wins.
        if (shift.delta > 0.0F) {
            consider(cell, path, 0.0F,
                     count_survivors(h1_scores, path, ntrials));
        }
    }
    best.ncandidates = cand.size();
    if (best.thresholds.empty()) {
        spdlog::warn("calibrate: no candidate survived calibration");
        return best;
    }
    if (any_clamped) {
        spdlog::warn("calibrate: a shift was clamped at |delta| = {:.3g}; "
                     "its Pd is not min_pd",
                     opts.delta_max);
    }
    spdlog::info("calibrate: Pd {:.4f} +/- {:.4f} (target {:.4g}, check "
                 "{:.4f}), delta {:+.3f}, cost {:.4g} (in-run Pd {:.4f}, "
                 "cost {:.4g}), {} candidates",
                 best.pd_certified, best.pd_sigma, min_pd, best.pd_check,
                 best.delta, best.cost, best.pd_inrun, best.cost_inrun,
                 best.ncandidates);
    return best;
}

} // namespace loki::detection::detail
