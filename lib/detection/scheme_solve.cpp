#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/detection/thresholds.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"

namespace loki::detection::detail {

// Salts of the independent samples of solve(); every one is derived from the
// search seed. Selection, repair and report never share a sample (a sample
// that chose something cannot measure it).
namespace {

constexpr uint64_t kBankSelect = 0x5A9E0001ULL; // relax + tighten
constexpr uint64_t kBankRepair = 0x5A9E0002ULL; // final repair
constexpr uint64_t kBankReport =
    0x5A9E0003ULL; // reported Pd, unused by any choice
constexpr uint64_t kNoiseReport = 0x5A9E0004ULL; // reported cost

struct Candidate {
    std::vector<float> thresholds;
    double pd{0.0};
    double c{0.0};
};

} // namespace

HostSignalBank::HostSignalBank(std::vector<float> scores, SizeType n)
    : m_scores(std::move(scores)),
      m_n(n) {
    error_check::check(n > 0 && m_scores.size() % n == 0,
                       "HostSignalBank: scores must be nstages x n");
}

std::vector<float>
HostSignalBank::margins(std::span<const float> thresholds) const {
    error_check::check_equal(thresholds.size() * m_n, m_scores.size(),
                             "HostSignalBank: one threshold per stage");
    std::vector<float> margins(m_n, std::numeric_limits<float>::max());
    for (SizeType s = 0; s < thresholds.size(); ++s) {
        for (SizeType t = 0; t < m_n; ++t) {
            margins[t] =
                std::min(margins[t], m_scores[(s * m_n) + t] - thresholds[s]);
        }
    }
    return margins;
}

std::vector<float>
HostSignalBank::survivor_minima(std::span<const float> thresholds) const {
    const auto margin = margins(thresholds);
    std::vector<float> minima(thresholds.size(),
                              std::numeric_limits<float>::max());
    for (SizeType s = 0; s < thresholds.size(); ++s) {
        for (SizeType t = 0; t < m_n; ++t) {
            if (margin[t] > 0.0F) {
                minima[s] = std::min(minima[s], m_scores[(s * m_n) + t]);
            }
        }
    }
    return minima;
}

SizeType count_survivors(const SignalBank& bank,
                         std::span<const float> thresholds) {
    const auto margins = bank.margins(thresholds);
    return static_cast<SizeType>(
        std::ranges::count_if(margins, [](float m) { return m > 0.0F; }));
}

namespace {

float relax_from_margins(std::vector<float>& thresholds,
                         std::vector<float>& margins,
                         double target) {
    const SizeType n        = margins.size();
    const SizeType required = std::min(
        n, static_cast<SizeType>(std::ceil(target * static_cast<double>(n))));
    const auto rank =
        margins.begin() + static_cast<std::ptrdiff_t>(n - required);
    std::ranges::nth_element(margins, rank);
    // The margin is a difference of floats and the shifted threshold a float
    // sum, so the survivor count can be one trial short at the rounding
    // boundary: the count is not a guarantee, and the independent report
    // measures what the thresholds deliver.
    const float shift = std::min(
        0.0F, std::nextafter(*rank, std::numeric_limits<float>::lowest()));
    if (shift < 0.0F) {
        for (auto& threshold : thresholds) {
            threshold += shift;
        }
    }
    return shift;
}

} // namespace

float relax_schedule(std::vector<float>& thresholds,
                     const SignalBank& bank,
                     double target) {
    auto margins = bank.margins(thresholds);
    return relax_from_margins(thresholds, margins, target);
}

std::pair<SizeType, SizeType> tighten_schedule(std::vector<float>& thresholds,
                                               const SignalBank& bank) {
    const SizeType count = count_survivors(bank, thresholds);
    SizeType tightened   = 0;
    if (count == 0) {
        return {tightened, count};
    }
    const auto minima = bank.survivor_minima(thresholds);
    for (SizeType s = 0; s < thresholds.size(); ++s) {
        const float bound =
            std::nextafter(minima[s], std::numeric_limits<float>::lowest());
        if (bound > thresholds[s]) {
            thresholds[s] = bound;
            ++tightened;
        }
    }
    return {tightened, count};
}

SchemeSolution solve_scheme(ThresholdsEngine& engine, float min_pd, double w) {
    error_check::check(min_pd > 0.0F && min_pd < 1.0F,
                       "solve: min_pd must be in (0, 1)");
    SchemeSolution out;
    out.min_pd             = min_pd;
    out.w_terminal         = static_cast<float>(w);
    const SizeType nstages = engine.get_nstages();
    const uint64_t seed    = engine.get_seed();
    const auto budget      = engine.solve_budget();
    const SizeType n       = budget.nbank;
    const double p         = static_cast<double>(min_pd);
    // The repair aims 2 binomial standard errors above the target, so the
    // reported Pd lands near it with margin on either side.
    const double target = std::min(
        1.0, p + (2.0 * std::sqrt(p * (1.0 - p) / static_cast<double>(n))));

    // The search's selection.
    Candidate best;
    {
        const auto states     = engine.get_states();
        const auto thresholds = engine.get_thresholds();
        const auto probs      = engine.get_probs();
        const SizeType nthr   = engine.get_nthresholds();
        const SizeType nprobs = engine.get_nprobs();
        SizeType cell =
            best_terminal_cell(states, nstages, nthr, nprobs, min_pd, w);
        if (cell == std::numeric_limits<SizeType>::max()) {
            // No path reaches the target in the search: take the one that
            // gets closest (highest Pd) and let the repair try to close the
            // gap. The reported Pd says what it delivers.
            float best_pd = 0.0F;
            for (SizeType i = 0; i < nthr * nprobs; ++i) {
                const auto& state = states[((nstages - 1) * nthr * nprobs) + i];
                if (!state.is_empty && state.success_h1_cumul > best_pd) {
                    best_pd = state.success_h1_cumul;
                    cell    = i;
                }
            }
            out.diagnostics.emplace_back("search_missed_target", 1.0);
        }
        const auto selected   = search_path(states, thresholds, probs, nstages,
                                            nthr, nprobs, cell, w);
        out.search_thresholds = selected.thresholds;
        out.search_pd         = selected.pd;
        out.search_c          = selected.c;
        if (selected.thresholds.empty()) {
            out.delivery = Delivery::kNoPath;
            engine.release_search_memory();
            return out;
        }
        best = {selected.thresholds, selected.pd, selected.c};
    }
    engine.release_search_memory();

    // Tighten (and relax if short) the selection on a fresh signal bank.
    const auto bank =
        engine.make_signal_bank(n, derive_seed(seed, kBankSelect));
    const auto evaluate_noise = [&](std::span<const float> thresholds,
                                    uint64_t salt) {
        return engine.evaluate(thresholds, budget.nnoise,
                               derive_seed(seed, salt), /*h0_only=*/true);
    };
    {
        const float shift    = relax_schedule(best.thresholds, *bank, target);
        const auto tightened = tighten_schedule(best.thresholds, *bank);
        out.diagnostics.emplace_back("initial_shift",
                                     static_cast<double>(shift));
        out.diagnostics.emplace_back("tightened_stages",
                                     static_cast<double>(tightened.first));
    }

    // Repair on another bank: lowers every threshold, only if the Pd of the
    // chosen schedule fell short of the target there.
    float repair_shift = 0.0F;
    {
        constexpr float kRepairCut = -0.5F;
        const uint64_t repair_seed = derive_seed(seed, kBankRepair);
        auto margins = engine.h1_margins_above(best.thresholds, kRepairCut, n,
                                               repair_seed);
        const SizeType required = std::min(
            n,
            static_cast<SizeType>(std::ceil(target * static_cast<double>(n))));
        const auto rank =
            margins.begin() + static_cast<std::ptrdiff_t>(n - required);
        std::ranges::nth_element(margins, rank);
        if (*rank > kRepairCut) {
            const float shift = std::min(
                0.0F,
                std::nextafter(*rank, std::numeric_limits<float>::lowest()));
            if (shift < 0.0F) {
                for (auto& threshold : best.thresholds) {
                    threshold += shift;
                }
            }
            repair_shift = shift;
        } else {
            const auto repair = engine.make_signal_bank(n, repair_seed);
            repair_shift = relax_schedule(best.thresholds, *repair, target);
        }
    }
    // Report on a third bank that no choice above has seen.
    {
        const SizeType count = engine.count_h1_survivors(
            best.thresholds, n, derive_seed(seed, kBankReport));
        out.pd    = static_cast<double>(count) / static_cast<double>(n);
        out.pd_se = std::sqrt(out.pd * (1.0 - out.pd) / static_cast<double>(n));
        out.pd_trials = n;
    }
    out.eval_states = evaluate_noise(best.thresholds, kNoiseReport);
    out.c           = total_cost(out.eval_states.back(), w);
    out.cost =
        out.pd > 0.0 ? out.c / out.pd : std::numeric_limits<double>::max();
    out.thresholds = std::move(best.thresholds);
    out.diagnostics.emplace_back("repair_shift",
                                 static_cast<double>(repair_shift));
    if (out.pd >= p) {
        out.delivery = Delivery::kMet;
    } else if (out.pd >= 0.9 * p) {
        out.delivery = Delivery::kWithinTolerance;
    } else {
        out.delivery = Delivery::kShort;
    }
    return out;
}

} // namespace loki::detection::detail
