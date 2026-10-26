#include "loki/pipelines/ffa_freq_sweep.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <spdlog/spdlog.h>

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
    const auto discard_staging = [&staging] {
        std::error_code error;
        std::filesystem::remove_all(staging, error);
    };
    try {
        m_impl->m_engine->execute(ts_e, ts_v, staging, "spool", config_toml);
    } catch (...) {
        // An incomplete search leaves nothing worth keeping.
        discard_staging();
        throw;
    }
    const auto spool = staging / "spool_ffa_results.h5";
    try {
        io::detail::write_candidate_results(
            spool, outdir / (std::string(file_prefix) + "_ffa_results.h5"),
            observation, m_impl->candidates);
    } catch (...) {
        // The search finished: keep its detections so post-processing can be
        // retried without searching again.
        const auto kept = outdir / (std::string(file_prefix) + "_ffa_spool.h5");
        std::error_code error;
        std::filesystem::rename(spool, kept, error);
        if (error) {
            spdlog::error("FFA post-processing failed and the search spool "
                          "could not be kept ({}); it remains in '{}'",
                          error.message(), staging.string());
        } else {
            discard_staging();
            spdlog::error(
                "FFA post-processing failed; the search's detections are kept "
                "in '{}'. Retry with FFAFreqSweep::publish_candidates (Python: "
                "libloki.ffa.FFAFreqSweep.publish_candidates).",
                kept.string());
        }
        throw;
    }
    discard_staging();
}

void FFAFreqSweep::publish_candidates(
    const std::filesystem::path& spool,
    const std::filesystem::path& output,
    const search::CandidateOptions& candidates,
    const io::ObservationMetadata& observation) {
    candidates.validate();
    observation.validate();
    if (!std::filesystem::exists(spool)) {
        throw std::invalid_argument(
            std::format("FFA spool '{}' does not exist", spool.string()));
    }
    io::detail::write_candidate_results(spool, output, observation, candidates);
}
} // namespace loki::pipelines
