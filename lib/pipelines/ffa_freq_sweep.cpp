#include "loki/pipelines/ffa_freq_sweep.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "loki/common/backend.hpp"
#include "loki/search/configs.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/io/candidate_results.hpp"
#include "lib/pipelines/ffa_freq_sweep_engine.hpp"

namespace loki::pipelines {

namespace {

std::unique_ptr<detail::FFAFreqSweepEngine> make_ffa_freq_sweep_engine(
    const search::FFASearchConfig& cfg, bool show_progress, Exec exec) {
    loki::detail::warn_ignored_nthreads(exec, "FFAFreqSweep");
    if (exec.backend == Backend::kCPU) {
        return detail::make_ffa_freq_sweep_cpu(cfg, show_progress);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        return detail::make_ffa_freq_sweep_gpu(cfg, exec.device, show_progress);
    }
#endif
    loki::detail::throw_unavailable("FFAFreqSweep", exec.backend);
}

} // namespace

class FFAFreqSweep::Impl {
public:
    explicit Impl(std::unique_ptr<detail::FFAFreqSweepEngine> engine)
        : m_engine(std::move(engine)) {}
    std::unique_ptr<detail::FFAFreqSweepEngine> m_engine;
    search::CandidateOptions candidates;
};

FFAFreqSweep::FFAFreqSweep(const search::FFASearchConfig& cfg,
                           bool show_progress,
                           Exec exec)
    : FFAFreqSweep(cfg, search::CandidateOptions{}, show_progress, exec) {}

FFAFreqSweep::FFAFreqSweep(const search::FFASearchConfig& cfg,
                           const search::CandidateOptions& candidates,
                           bool show_progress,
                           Exec exec) {
    candidates.validate();
    // On CPU both the persistent engine workspace and postprocessing are
    // resident. Reserve postprocessing before fitting the search chunks.
    auto search_cfg = cfg;
    if (exec.backend == Backend::kCPU) {
        const auto remaining =
            cfg.get_max_process_memory_gb() - candidates.memory_gb;
        if (remaining <= 0) {
            throw std::invalid_argument(
                "candidate host memory budget must be smaller than total CPU "
                "search memory budget");
        }
        search_cfg.set_max_process_memory_gb(remaining);
    }
    m_impl = std::make_unique<Impl>(
        make_ffa_freq_sweep_engine(search_cfg, show_progress, exec));
    m_impl->candidates = candidates;
}

FFAFreqSweep::~FFAFreqSweep()                                        = default;
FFAFreqSweep::FFAFreqSweep(FFAFreqSweep&& other) noexcept            = default;
FFAFreqSweep& FFAFreqSweep::operator=(FFAFreqSweep&& other) noexcept = default;

void FFAFreqSweep::execute(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           const std::filesystem::path& outdir,
                           std::string_view file_prefix,
                           std::string_view config_toml,
                           const io::ObservationMetadata& observation) {
    observation.validate();
    std::filesystem::create_directories(outdir);
    std::filesystem::path staging;
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        staging =
            outdir /
            (".loki-ffa-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "-" + std::to_string(attempt));
        if (std::filesystem::create_directory(staging)) {
            break;
        }
        staging.clear();
    }
    if (staging.empty()) {
        throw std::runtime_error("cannot create FFA staging directory");
    }
    try {
        m_impl->m_engine->execute(ts_e, ts_v, staging, "spool", config_toml);
        const auto destination =
            outdir / (std::string(file_prefix) + "_ffa_results.h5");
        io::detail::write_candidate_results(staging / "spool_ffa_results.h5",
                                            destination, observation,
                                            m_impl->candidates);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove_all(staging, error);
        throw;
    }
    std::error_code error;
    std::filesystem::remove_all(staging, error);
}
} // namespace loki::pipelines
