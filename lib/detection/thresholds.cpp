#include "loki/detection/thresholds.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <highfive/highfive.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/detection/scheme.hpp"
#include "lib/detection/thresholds_engine.hpp"

namespace loki::detection {

namespace detail {

// Shared with the EP planner (thresholds_engine.hpp).
std::unique_ptr<ThresholdsEngine>
make_thresholds_engine(std::span<const float> branching_pattern,
                       float ref_ducy,
                       SizeType nbins,
                       SizeType ntrials,
                       SizeType nprobs,
                       float prob_min,
                       float snr_final,
                       SizeType nthresholds,
                       float ducy_max,
                       float wtsp,
                       float beam_width,
                       SizeType trials_start,
                       std::string_view mode,
                       std::optional<uint64_t> seed,
                       SizeType batch_size,
                       std::string_view guess,
                       float min_pd,
                       float w_terminal,
                       Exec exec) {
    if (exec.backend == Backend::kCPU) {
        return detail::make_thresholds_cpu(
            branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
            snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start,
            mode, exec.nthreads, seed, /*simulate=*/true, guess, min_pd,
            w_terminal);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        return detail::make_thresholds_gpu(
            branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
            snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start,
            mode, batch_size, exec.device, seed, guess, min_pd, w_terminal);
    }
#endif
    (void)batch_size;
    loki::detail::throw_unavailable("DynamicThresholdScheme", exec.backend);
}

} // namespace detail

namespace {

/// Appends the solve() result to a file written by the engine's save(): one
/// group "solution". Existing datasets are untouched.
void append_solution(const std::string& path, const SchemeSolution& sol) {
    HighFive::File file(path, HighFive::File::ReadWrite);
    auto grp = file.createGroup("solution");
    grp.createDataSet("thresholds", sol.thresholds);
    grp.createDataSet("search_thresholds", sol.search_thresholds);
    const auto field = [&](const char* name, auto get) {
        std::vector<float> v(sol.eval_states.size());
        for (SizeType i = 0; i < v.size(); ++i) {
            v[i] = get(sol.eval_states[i]);
        }
        grp.createDataSet(name, v);
    };
    field("eval_success_h0", [](const State& s) { return s.success_h0; });
    field("eval_complexity", [](const State& s) { return s.complexity; });
    field("eval_complexity_cumul",
          [](const State& s) { return s.complexity_cumul; });
    field("eval_threshold", [](const State& s) { return s.threshold; });
    grp.createAttribute("pd", sol.pd);
    grp.createAttribute("pd_se", sol.pd_se);
    grp.createAttribute("pd_trials", sol.pd_trials);
    grp.createAttribute("c", sol.c);
    grp.createAttribute("cost", sol.cost);
    grp.createAttribute("search_pd", sol.search_pd);
    grp.createAttribute("search_c", sol.search_c);
    grp.createAttribute("min_pd", sol.min_pd);
    grp.createAttribute("w_terminal", sol.w_terminal);
    grp.createAttribute("refine", refine_to_string(sol.refine));
    grp.createAttribute("delivery", delivery_to_string(sol.delivery));
    for (const auto& [name, value] : sol.diagnostics) {
        grp.createAttribute("diag_" + name, value);
    }
}

} // namespace

class DynamicThresholdScheme::Impl {
public:
    Impl(std::unique_ptr<detail::ThresholdsEngine> engine,
         DynamicThresholdMode mode,
         const SharedOptions& shared)
        : m_engine(std::move(engine)),
          m_mode(mode),
          m_shared(shared) {}

    std::unique_ptr<detail::ThresholdsEngine> m_engine;
    DynamicThresholdMode m_mode;
    SharedOptions m_shared;
    std::optional<SchemeSolution> m_solution;
};

DynamicThresholdScheme::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    float ref_ducy,
    SizeType nbins,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType trials_start,
    std::string_view mode,
    std::optional<uint64_t> seed,
    SizeType batch_size,
    Exec exec)
    : DynamicThresholdScheme(branching_pattern,
                             ref_ducy,
                             nbins,
                             ntrials,
                             nprobs,
                             prob_min,
                             snr_final,
                             nthresholds,
                             ducy_max,
                             wtsp,
                             beam_width,
                             trials_start,
                             mode,
                             seed,
                             batch_size,
                             "analytic",
                             SharedOptions{},
                             exec) {}

DynamicThresholdScheme::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    float ref_ducy,
    SizeType nbins,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType trials_start,
    std::string_view mode,
    std::optional<uint64_t> seed,
    SizeType batch_size,
    std::string_view guess,
    Exec exec)
    : DynamicThresholdScheme(branching_pattern,
                             ref_ducy,
                             nbins,
                             ntrials,
                             nprobs,
                             prob_min,
                             snr_final,
                             nthresholds,
                             ducy_max,
                             wtsp,
                             beam_width,
                             trials_start,
                             mode,
                             seed,
                             batch_size,
                             guess,
                             SharedOptions{},
                             exec) {}

DynamicThresholdScheme::DynamicThresholdScheme(
    std::span<const float> branching_pattern,
    float ref_ducy,
    SizeType nbins,
    SizeType ntrials,
    SizeType nprobs,
    float prob_min,
    float snr_final,
    SizeType nthresholds,
    float ducy_max,
    float wtsp,
    float beam_width,
    SizeType trials_start,
    std::string_view mode,
    std::optional<uint64_t> seed,
    SizeType batch_size,
    std::string_view guess,
    const SharedOptions& shared,
    Exec exec)
    : m_impl(std::make_unique<Impl>(
          detail::make_thresholds_engine(branching_pattern,
                                         ref_ducy,
                                         nbins,
                                         ntrials,
                                         nprobs,
                                         prob_min,
                                         snr_final,
                                         nthresholds,
                                         ducy_max,
                                         wtsp,
                                         beam_width,
                                         trials_start,
                                         mode,
                                         seed,
                                         batch_size,
                                         guess,
                                         shared.min_pd,
                                         shared.w_terminal,
                                         exec),
          dynamic_threshold_mode_from_string(mode),
          shared)) {}

DynamicThresholdScheme::~DynamicThresholdScheme() = default;
DynamicThresholdScheme::DynamicThresholdScheme(
    DynamicThresholdScheme&&) noexcept = default;
DynamicThresholdScheme&
DynamicThresholdScheme::operator=(DynamicThresholdScheme&&) noexcept = default;

std::vector<SizeType>
DynamicThresholdScheme::get_current_thresholds_idx(SizeType istage) const {
    return m_impl->m_engine->get_current_thresholds_idx(istage);
}
std::vector<float> DynamicThresholdScheme::get_branching_pattern() const {
    return m_impl->m_engine->get_branching_pattern();
}
std::vector<float> DynamicThresholdScheme::get_profile() const {
    return m_impl->m_engine->get_profile();
}
std::vector<float> DynamicThresholdScheme::get_thresholds() const {
    return m_impl->m_engine->get_thresholds();
}
std::vector<float> DynamicThresholdScheme::get_probs() const {
    return m_impl->m_engine->get_probs();
}
SizeType DynamicThresholdScheme::get_nstages() const {
    return m_impl->m_engine->get_nstages();
}
SizeType DynamicThresholdScheme::get_nthresholds() const {
    return m_impl->m_engine->get_nthresholds();
}
SizeType DynamicThresholdScheme::get_nprobs() const {
    return m_impl->m_engine->get_nprobs();
}
std::vector<SizeType> DynamicThresholdScheme::get_box_score_widths() const {
    return m_impl->m_engine->get_box_score_widths();
}
std::vector<State> DynamicThresholdScheme::get_states() const {
    return m_impl->m_engine->get_states();
}

std::vector<float>
DynamicThresholdScheme::get_best_path_thresholds(float min_pd) const {
    return m_impl->m_engine->get_best_path_thresholds(min_pd);
}

void DynamicThresholdScheme::run(SizeType thres_neigh) {
    m_impl->m_engine->run(thres_neigh);
}
std::vector<State>
DynamicThresholdScheme::evaluate(std::span<const float> thresholds,
                                 SizeType ntrials,
                                 std::optional<uint64_t> seed) const {
    return m_impl->m_engine->evaluate(thresholds, ntrials, seed, false);
}
CalibratedPath
DynamicThresholdScheme::calibrate(float min_pd,
                                  const CalibrationOptions& opts) const {
    return detail::calibrate_path(*m_impl->m_engine, min_pd, opts);
}
std::vector<float>
DynamicThresholdScheme::simulate_h1_scores(SizeType ntrials,
                                           uint64_t seed) const {
    if (ntrials == 0 || ntrials > (SizeType{1} << 20U)) {
        throw std::invalid_argument(
            "DynamicThresholdScheme::simulate_h1_scores: ntrials must be in "
            "[1, 2^20]");
    }
    return m_impl->m_engine->simulate_h1_scores(ntrials, seed);
}
SchemeSolution DynamicThresholdScheme::solve() {
    if (m_impl->m_mode != DynamicThresholdMode::kShared) {
        throw std::invalid_argument(
            "DynamicThresholdScheme::solve: only mode 'shared'");
    }
    m_impl->m_solution = detail::solve_scheme(
        *m_impl->m_engine, m_impl->m_shared.min_pd, m_impl->m_shared.w_terminal,
        m_impl->m_shared.refine);
    return *m_impl->m_solution;
}
const SchemeSolution* DynamicThresholdScheme::solution() const {
    return m_impl->m_solution ? &*m_impl->m_solution : nullptr;
}
std::string DynamicThresholdScheme::save(const std::string& outdir) const {
    const auto path = m_impl->m_engine->save(outdir);
    if (m_impl->m_solution) {
        append_solution(path, *m_impl->m_solution);
    }
    return path;
}

} // namespace loki::detection
