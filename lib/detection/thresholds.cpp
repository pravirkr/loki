#include "loki/detection/thresholds.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/detection/thresholds_engine.hpp"

namespace loki::detection {

namespace {

std::unique_ptr<detail::ThresholdsEngine>
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
                       Exec exec) {
    if (exec.backend == Backend::kCPU) {
        return detail::make_thresholds_cpu(
            branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
            snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start,
            mode, exec.nthreads, seed);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        return detail::make_thresholds_gpu(
            branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
            snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start,
            mode, batch_size, exec.device, seed);
    }
#endif
    (void)batch_size;
    loki::detail::throw_unavailable("DynamicThresholdScheme", exec.backend);
}

} // namespace

class DynamicThresholdScheme::Impl {
public:
    explicit Impl(std::unique_ptr<detail::ThresholdsEngine> engine)
        : m_engine(std::move(engine)) {}

    std::unique_ptr<detail::ThresholdsEngine> m_engine;
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
    : m_impl(std::make_unique<Impl>(make_thresholds_engine(
          branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
          snr_final, nthresholds, ducy_max, wtsp, beam_width, trials_start,
          mode, seed, batch_size, exec))) {}

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
    return m_impl->m_engine->evaluate(thresholds, ntrials, seed);
}
std::string DynamicThresholdScheme::save(const std::string& outdir) const {
    return m_impl->m_engine->save(outdir);
}

} // namespace loki::detection
