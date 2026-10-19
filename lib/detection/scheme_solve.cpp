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
// search seed. Selection, confirmation, repair and report never share a
// sample (a sample that chose something cannot measure it).
namespace {

constexpr uint64_t kBankSelect = 0x5A9E0001ULL; // tighten + candidate Pd
constexpr uint64_t kBankRepair = 0x5A9E0002ULL; // final repair
constexpr uint64_t kBankReport =
    0x5A9E0003ULL; // reported Pd, unused by any choice
constexpr uint64_t kNoiseReport  = 0x5A9E0004ULL; // reported cost
constexpr uint64_t kNoiseSelect  = 0xC1800020ULL; // candidate / proposal cost
constexpr uint64_t kNoiseConfirm = 0xC1800021ULL; // confirmation of a move

constexpr SizeType kShortlist    = 4;
constexpr SizeType kRefineStarts = 2;
constexpr SizeType kRefineBlocks = 4;
// A refinement move needs this relative cost gain on the selection noise
// sample and on the confirmation one (heuristic margins, not guarantees).
constexpr double kMoveGain    = 0.98;
constexpr double kConfirmGain = 0.995;
constexpr float kMoveSteps    = 2.0F;

struct Candidate {
    std::vector<float> thresholds;
    double pd{0.0};
    double c{0.0};
};

double rank_of(const State& last, double pd, double w) {
    return total_cost(last, w) / pd;
}

} // namespace

float relax_schedule(std::vector<float>& thresholds,
                     std::span<const float> scores,
                     SizeType n,
                     double target) {
    std::vector<float> margins(n, std::numeric_limits<float>::max());
    for (SizeType s = 0; s < thresholds.size(); ++s) {
        for (SizeType t = 0; t < n; ++t) {
            margins[t] =
                std::min(margins[t], scores[(s * n) + t] - thresholds[s]);
        }
    }
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

std::pair<SizeType, SizeType> tighten_schedule(std::vector<float>& thresholds,
                                               std::span<const float> scores,
                                               SizeType n) {
    std::vector<uint8_t> alive(n, 1);
    for (SizeType s = 0; s < thresholds.size(); ++s) {
        for (SizeType t = 0; t < n; ++t) {
            alive[t] &=
                static_cast<uint8_t>(scores[(s * n) + t] > thresholds[s]);
        }
    }
    std::vector<uint32_t> survivors;
    survivors.reserve(n);
    for (SizeType t = 0; t < n; ++t) {
        if (alive[t] != 0) {
            survivors.push_back(static_cast<uint32_t>(t));
        }
    }
    const SizeType count = survivors.size();
    SizeType tightened   = 0;
    if (count == 0) {
        return {tightened, count};
    }
    for (SizeType s = 0; s < thresholds.size(); ++s) {
        float bound = std::numeric_limits<float>::max();
        for (const auto t : survivors) {
            bound = std::min(bound, scores[(s * n) + t]);
        }
        bound = std::nextafter(bound, std::numeric_limits<float>::lowest());
        if (bound > thresholds[s]) {
            thresholds[s] = bound;
            ++tightened;
        }
    }
    return {tightened, count};
}

std::vector<SizeType> refinement_blocks(std::span<const float> branching) {
    const SizeType m       = branching.size();
    const SizeType nblocks = std::min<SizeType>(kRefineBlocks, m);
    std::vector<double> prefix(m);
    double total = 0.0;
    for (SizeType s = 0; s < m; ++s) {
        total += std::log(std::max(1.0, static_cast<double>(branching[s])));
        prefix[s] = total;
    }
    std::vector<SizeType> cuts{0};
    for (SizeType k = 1; k < nblocks; ++k) {
        const SizeType cut =
            total > 0.0 ? static_cast<SizeType>(
                              std::ranges::lower_bound(
                                  prefix, total * static_cast<double>(k) /
                                              static_cast<double>(nblocks)) -
                              prefix.begin())
                        : k * m / nblocks;
        if (cut > cuts.back() && cut < m) {
            cuts.push_back(cut);
        }
    }
    cuts.push_back(m);
    return cuts;
}

namespace {

struct BlockRefinement {
    Candidate best;
    double rank{0.0};
    SizeType moves{0};
    SizeType proposals{0};
    SizeType confirmations{0};
};

/// One sweep of +-2 grid steps over every block, each move kept only if the
/// cost per detection drops on the selection noise sample and again on an
/// independent confirmation sample.
BlockRefinement refine_blocks(const ThresholdsEngine& engine,
                              Candidate start,
                              double initial_pd,
                              double initial_rank,
                              std::span<const float> scores,
                              SizeType n,
                              double target,
                              std::span<const SizeType> cuts,
                              double w,
                              SizeType nnoise,
                              uint64_t seed) {
    BlockRefinement out{std::move(start), initial_rank};
    const auto evaluate_rank = [&](std::span<const float> thresholds, double pd,
                                   uint64_t salt) {
        const auto states = engine.evaluate(
            thresholds, nnoise, derive_seed(seed, salt), /*h0_only=*/true);
        return rank_of(states.back(), pd, w);
    };
    double confirm_rank =
        evaluate_rank(out.best.thresholds, initial_pd, kNoiseConfirm);
    ++out.confirmations;
    const auto grid   = engine.get_thresholds();
    const float delta = kMoveSteps * (grid[1] - grid[0]);
    for (SizeType b = 0; b + 1 < cuts.size(); ++b) {
        std::vector<float> best_thresholds;
        double best_rank    = out.rank;
        double best_confirm = confirm_rank;
        for (const float direction : {-1.0F, 1.0F}) {
            auto thresholds = out.best.thresholds;
            for (SizeType s = cuts[b]; s < cuts[b + 1]; ++s) {
                thresholds[s] += direction * delta;
            }
            relax_schedule(thresholds, scores, n, target);
            const auto tightened = tighten_schedule(thresholds, scores, n);
            if (thresholds == out.best.thresholds || tightened.second == 0) {
                continue;
            }
            const double pd =
                static_cast<double>(tightened.second) / static_cast<double>(n);
            const double rank = evaluate_rank(thresholds, pd, kNoiseSelect);
            ++out.proposals;
            if (rank > kMoveGain * out.rank || rank >= best_rank) {
                continue;
            }
            const double check = evaluate_rank(thresholds, pd, kNoiseConfirm);
            ++out.confirmations;
            if (check > kConfirmGain * confirm_rank) {
                continue;
            }
            best_thresholds = std::move(thresholds);
            best_rank       = rank;
            best_confirm    = check;
        }
        if (!best_thresholds.empty()) {
            out.best.thresholds = std::move(best_thresholds);
            out.rank            = best_rank;
            confirm_rank        = best_confirm;
            ++out.moves;
        }
    }
    return out;
}

/// Up to kShortlist distinct candidate paths: the final selection, the first
/// pass's selection, then the cheapest terminal cells that differ from every
/// earlier one by at least half a grid step rms. Each is replayed on the
/// search's own signal bank: its terminal survivor count must match the
/// search's Pd.
std::vector<Candidate>
shortlist_candidates(const ThresholdsEngine& engine,
                     std::span<const State> states,
                     const SearchPath& selected,
                     float min_pd,
                     double w,
                     std::vector<std::pair<std::string, double>>& diag) {
    const auto thresholds  = engine.get_thresholds();
    const auto probs       = engine.get_probs();
    const SizeType nstages = engine.get_nstages();
    const SizeType nthr    = engine.get_nthresholds();
    const SizeType nprobs  = engine.get_nprobs();
    const double step      = static_cast<double>(thresholds[1] - thresholds[0]);
    const double min_distance =
        0.25 * step * step * static_cast<double>(nstages);
    std::vector<Candidate> out;
    const auto append = [&](const SearchPath& path, bool diverse) {
        if (path.thresholds.empty()) {
            return;
        }
        for (const auto& old : out) {
            if (path.thresholds == old.thresholds) {
                return;
            }
            double distance = 0.0;
            for (SizeType s = 0; s < nstages; ++s) {
                const double d = static_cast<double>(path.thresholds[s]) -
                                 static_cast<double>(old.thresholds[s]);
                distance += d * d;
            }
            if (diverse && distance < min_distance) {
                return;
            }
        }
        out.push_back({path.thresholds, path.pd, path.c});
    };
    append(selected, false);
    append(engine.first_pass_path(min_pd), false);
    const SizeType offset = (nstages - 1) * nthr * nprobs;
    std::vector<std::pair<double, SizeType>> order;
    for (SizeType cell = 0; cell < nthr * nprobs; ++cell) {
        const auto& state = states[offset + cell];
        if (!state.is_empty && state.success_h1_cumul >= min_pd) {
            order.emplace_back(terminal_rank(state, w), cell);
        }
    }
    std::ranges::sort(order);
    for (const auto& entry : order) {
        if (out.size() == kShortlist) {
            break;
        }
        append(search_path(states, thresholds, probs, nstages, nthr, nprobs,
                           entry.second, w),
               true);
    }
    const auto bank     = engine.shared_h1_bank();
    const SizeType nh1  = bank.size() / nstages;
    SizeType mismatches = 0;
    for (const auto& candidate : out) {
        SizeType count = 0;
        for (SizeType t = 0; t < nh1; ++t) {
            bool alive = true;
            for (SizeType s = 0; s < nstages && alive; ++s) {
                alive = bank[(s * nh1) + t] > candidate.thresholds[s];
            }
            count += alive ? 1 : 0;
        }
        // Float products may round slightly; the survivor count must agree.
        if (static_cast<SizeType>(std::llround(
                candidate.pd * static_cast<double>(nh1))) != count) {
            ++mismatches;
        }
    }
    if (mismatches > 0) {
        spdlog::warn("solve: {} shortlist path(s) do not replay the search's "
                     "signal count",
                     mismatches);
    }
    diag.emplace_back("shortlist_replay_mismatches",
                      static_cast<double>(mismatches));
    return out;
}

} // namespace

SchemeSolution solve_scheme(ThresholdsEngine& engine,
                            float min_pd,
                            double w,
                            RefineLevel refine) {
    error_check::check(min_pd > 0.0F && min_pd < 1.0F,
                       "solve: min_pd must be in (0, 1)");
    SchemeSolution out;
    out.min_pd             = min_pd;
    out.w_terminal         = static_cast<float>(w);
    out.refine             = refine;
    const SizeType nstages = engine.get_nstages();
    const uint64_t seed    = engine.get_seed();
    const auto budget      = engine.solve_budget();
    const SizeType n       = budget.nbank;
    const double p         = static_cast<double>(min_pd);
    // The repair aims 2 binomial standard errors above the target, so the
    // reported Pd lands near it with margin on either side.
    const double target = std::min(
        1.0, p + (2.0 * std::sqrt(p * (1.0 - p) / static_cast<double>(n))));

    // The search's selection and (for the shortlist) its alternatives.
    std::vector<Candidate> candidates;
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
        if (refine == RefineLevel::kNone) {
            candidates.push_back(
                {selected.thresholds, selected.pd, selected.c});
        } else {
            candidates = shortlist_candidates(engine, states, selected, min_pd,
                                              w, out.diagnostics);
        }
    }
    engine.release_search_memory();

    // Tighten (and relax if short) every candidate on a fresh signal bank and
    // pick by C / Pd on a fresh noise sample.
    const auto scores =
        engine.simulate_h1_scores(n, derive_seed(seed, kBankSelect));
    Candidate best;
    const auto evaluate_noise = [&](std::span<const float> thresholds,
                                    uint64_t salt) {
        return engine.evaluate(thresholds, budget.nnoise,
                               derive_seed(seed, salt), /*h0_only=*/true);
    };
    if (refine == RefineLevel::kNone) {
        best              = candidates.front();
        const float shift = relax_schedule(best.thresholds, scores, n, target);
        const auto tightened = tighten_schedule(best.thresholds, scores, n);
        out.diagnostics.emplace_back("initial_shift",
                                     static_cast<double>(shift));
        out.diagnostics.emplace_back("tightened_stages",
                                     static_cast<double>(tightened.first));
    } else {
        double best_rank    = std::numeric_limits<double>::max();
        double base_rank    = 0.0;
        SizeType best_index = 0;
        std::vector<double> bank_pd(candidates.size());
        std::vector<std::pair<double, SizeType>> order;
        for (SizeType i = 0; i < candidates.size(); ++i) {
            auto& candidate = candidates[i];
            relax_schedule(candidate.thresholds, scores, n, target);
            const auto tightened =
                tighten_schedule(candidate.thresholds, scores, n);
            const auto check =
                evaluate_noise(candidate.thresholds, kNoiseSelect);
            const double pd =
                static_cast<double>(tightened.second) / static_cast<double>(n);
            const double rank = rank_of(check.back(), pd, w);
            bank_pd[i]        = pd;
            order.emplace_back(rank, i);
            if (i == 0) {
                base_rank = rank;
            }
            if (rank < best_rank) {
                best_rank  = rank;
                best_index = i;
            }
        }
        best                          = candidates[best_index];
        const double initializer_rank = best_rank;
        SizeType starts               = 0;
        SizeType proposals            = 0;
        SizeType confirmations        = 0;
        SizeType winning_moves        = 0;
        const auto cuts = refinement_blocks(engine.get_branching_pattern());
        if (refine == RefineLevel::kFull) {
            std::ranges::sort(order);
            std::vector<std::vector<float>> used_starts;
            for (const auto& entry : order) {
                if (starts == kRefineStarts) {
                    break;
                }
                const auto idx = entry.second;
                if (std::ranges::find(used_starts,
                                      candidates[idx].thresholds) !=
                    used_starts.end()) {
                    continue;
                }
                used_starts.push_back(candidates[idx].thresholds);
                ++starts;
                auto refined = refine_blocks(
                    engine, candidates[idx], bank_pd[idx], entry.first, scores,
                    n, target, cuts, w, budget.nnoise, seed);
                proposals += refined.proposals;
                confirmations += refined.confirmations;
                if (refined.rank < best_rank) {
                    best_rank     = refined.rank;
                    best_index    = idx;
                    winning_moves = refined.moves;
                    best          = std::move(refined.best);
                }
            }
            out.diagnostics.emplace_back("block_count",
                                         static_cast<double>(cuts.size() - 1));
            out.diagnostics.emplace_back("block_starts",
                                         static_cast<double>(starts));
            out.diagnostics.emplace_back("block_proposals",
                                         static_cast<double>(proposals));
            out.diagnostics.emplace_back("block_confirmations",
                                         static_cast<double>(confirmations));
            out.diagnostics.emplace_back("block_moves",
                                         static_cast<double>(winning_moves));
        }
        out.diagnostics.emplace_back("shortlist_initializer_L",
                                     initializer_rank);
        out.diagnostics.emplace_back("shortlist_count",
                                     static_cast<double>(candidates.size()));
        out.diagnostics.emplace_back("shortlist_selected",
                                     static_cast<double>(best_index));
        out.diagnostics.emplace_back("shortlist_bank_L", best_rank);
        out.diagnostics.emplace_back("shortlist_base_bank_L", base_rank);
        for (const auto& candidate : candidates) {
            out.shortlist.push_back(candidate.thresholds);
        }
    }

    // Repair on another bank: lowers every threshold, only if the Pd of the
    // chosen schedule fell short of the target there.
    float repair_shift = 0.0F;
    {
        const auto repair_scores =
            engine.simulate_h1_scores(n, derive_seed(seed, kBankRepair));
        repair_shift =
            relax_schedule(best.thresholds, repair_scores, n, target);
    }
    // Report on a third bank that no choice above has seen.
    {
        const auto report_scores =
            engine.simulate_h1_scores(n, derive_seed(seed, kBankReport));
        const SizeType count =
            count_survivors(report_scores, best.thresholds, n);
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

SchemeSolution refit_scheme(const ThresholdsEngine& engine,
                            const SchemeSolution& searched,
                            float min_pd,
                            double w,
                            uint64_t seed,
                            const SolveBudget& budget) {
    error_check::check(!searched.thresholds.empty(), "refit: nothing to refit");
    constexpr uint64_t kRefitSelect = 0x5A9E0011ULL;
    constexpr uint64_t kRefitRepair = 0x5A9E0012ULL;
    constexpr uint64_t kRefitReport = 0x5A9E0013ULL;
    constexpr uint64_t kRefitNoiseS = 0xC1800030ULL;
    constexpr uint64_t kRefitNoiseR = 0x5A9E0014ULL;
    constexpr float kShiftStep      = 0.2F;
    constexpr float kShiftFine      = 0.1F;
    constexpr float kShiftMax       = 3.0F;
    const SizeType n                = budget.nbank;
    const double p                  = static_cast<double>(min_pd);
    const double target             = std::min(
        1.0, p + (2.0 * std::sqrt(p * (1.0 - p) / static_cast<double>(n))));
    SchemeSolution out    = searched;
    out.search_thresholds = searched.thresholds;
    out.search_pd         = searched.pd;
    out.search_c          = searched.c;
    out.shortlist.clear();

    // Common upward shift of lowest C / Pd among those reaching the target,
    // on one signal bank and one noise sample.
    const auto scores =
        engine.simulate_h1_scores(n, derive_seed(seed, kRefitSelect));
    const auto& base   = searched.thresholds;
    const auto shifted = [&](float shift) {
        std::vector<float> t(base.size());
        std::ranges::transform(base, t.begin(),
                               [shift](float x) { return x + shift; });
        return t;
    };
    struct Point {
        float shift{0.0F};
        double pd{0.0};
        double rank{std::numeric_limits<double>::max()};
    };
    SizeType nevaluations = 0;
    const auto probe      = [&](float shift) {
        Point pt{.shift = shift};
        const auto t = shifted(shift);
        pt.pd        = static_cast<double>(count_survivors(scores, t, n)) /
                       static_cast<double>(n);
        if (pt.pd >= target) {
            const auto states = engine.evaluate(t, budget.nnoise,
                                                derive_seed(seed, kRefitNoiseS),
                                                /*h0_only=*/true);
            ++nevaluations;
            pt.rank = rank_of(states.back(), pt.pd, w);
        }
        return pt;
    };
    Point best = probe(0.0F);
    for (float shift = kShiftStep; shift <= kShiftMax; shift += kShiftStep) {
        const Point pt = probe(shift);
        if (pt.pd < target) {
            break; // the Pd only falls with the shift
        }
        if (pt.rank < best.rank) {
            best = pt;
        }
    }
    // One finer step on either side of the coarse optimum
    for (const float shift :
         {best.shift - kShiftFine, best.shift + kShiftFine}) {
        if (shift >= 0.0F && shift <= kShiftMax) {
            const Point pt = probe(shift);
            if (pt.rank < best.rank) {
                best = pt;
            }
        }
    }
    auto thresholds      = shifted(best.shift);
    const auto tightened = tighten_schedule(thresholds, scores, n);
    // Repair on another bank, report on a third.
    float repair_shift = 0.0F;
    {
        const auto repair_scores =
            engine.simulate_h1_scores(n, derive_seed(seed, kRefitRepair));
        repair_shift = relax_schedule(thresholds, repair_scores, n, target);
    }
    {
        const auto report_scores =
            engine.simulate_h1_scores(n, derive_seed(seed, kRefitReport));
        const SizeType count = count_survivors(report_scores, thresholds, n);
        out.pd    = static_cast<double>(count) / static_cast<double>(n);
        out.pd_se = std::sqrt(out.pd * (1.0 - out.pd) / static_cast<double>(n));
        out.pd_trials = n;
    }
    out.eval_states = engine.evaluate(thresholds, budget.nnoise,
                                      derive_seed(seed, kRefitNoiseR), true);
    out.c           = total_cost(out.eval_states.back(), w);
    out.cost =
        out.pd > 0.0 ? out.c / out.pd : std::numeric_limits<double>::max();
    out.thresholds = std::move(thresholds);
    out.diagnostics.emplace_back("refit_shift",
                                 static_cast<double>(best.shift));
    out.diagnostics.emplace_back("refit_repair_shift",
                                 static_cast<double>(repair_shift));
    out.diagnostics.emplace_back("refit_tightened_stages",
                                 static_cast<double>(tightened.first));
    out.diagnostics.emplace_back("refit_evaluations",
                                 static_cast<double>(nevaluations));
    out.diagnostics.emplace_back("refit_bank", static_cast<double>(n));
    out.diagnostics.emplace_back("refit_noise",
                                 static_cast<double>(budget.nnoise));
    out.diagnostics.emplace_back("searched_cost", searched.cost);
    out.delivery = out.pd >= p         ? Delivery::kMet
                   : out.pd >= 0.9 * p ? Delivery::kWithinTolerance
                                       : Delivery::kShort;
    return out;
}

} // namespace loki::detection::detail
