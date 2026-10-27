#include "loki/pipelines/ep_freq_sweep.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/io/candidate_results.hpp"
#include "lib/pipelines/ep_freq_sweep_engine.hpp"
#include "lib/search/ep_support.hpp"

namespace loki::pipelines {

namespace {

std::unique_ptr<detail::EPFreqSweepEngine> make_ep_freq_sweep_engine(
    const search::PulsarSearchConfig& cfg,
    bool show_progress,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const algorithms::PruneRFIConfig& rfi_config,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    const std::optional<std::vector<SizeType>>& ref_segs,
    Exec exec) {
    search::detail::check_ep_supported(cfg, "EPFreqSweep");
    loki::detail::warn_ignored_nthreads(exec, "EPFreqSweep");
    if (exec.backend == Backend::kCPU) {
        return detail::make_ep_freq_sweep_cpu(
            cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
            plan_cache_file, n_runs, ref_segs);
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        if (rfi_config.is_active()) {
            loki::detail::throw_unimplemented("EPFreqSweep (rfi_config)",
                                              exec.backend);
        }
        return detail::make_ep_freq_sweep_gpu(cfg, min_pd, poly_basis, ref_ducy,
                                              plan_cache_file, n_runs, ref_segs,
                                              exec.device);
    }
#endif
    loki::detail::throw_unavailable("EPFreqSweep", exec.backend);
}

} // namespace

class EPFreqSweep::Impl {
public:
    explicit Impl(std::unique_ptr<detail::EPFreqSweepEngine> engine)
        : m_engine(std::move(engine)) {}
    std::unique_ptr<detail::EPFreqSweepEngine> m_engine;
    std::optional<search::CandidateOptions> candidates;
};

EPFreqSweep::EPFreqSweep(
    const search::PulsarSearchConfig& cfg,
    bool show_progress,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const algorithms::PruneRFIConfig& rfi_config,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param): public signature
    std::optional<std::vector<SizeType>> ref_segs,
    Exec exec)
    : m_impl(std::make_unique<Impl>(make_ep_freq_sweep_engine(cfg,
                                                              show_progress,
                                                              min_pd,
                                                              poly_basis,
                                                              ref_ducy,
                                                              rfi_config,
                                                              plan_cache_file,
                                                              n_runs,
                                                              ref_segs,
                                                              exec))) {}

EPFreqSweep::EPFreqSweep(
    const search::PulsarSearchConfig& cfg,
    const search::CandidateOptions& candidates,
    bool show_progress,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const algorithms::PruneRFIConfig& rfi_config,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    // NOLINTNEXTLINE(performance-unnecessary-value-param): public signature
    std::optional<std::vector<SizeType>> ref_segs,
    Exec exec) {
    candidates.validate();
    if (!candidates.enabled) {
        // Run diagnostics only, as the constructor without candidates.
        m_impl = std::make_unique<Impl>(make_ep_freq_sweep_engine(
            cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
            plan_cache_file, n_runs, std::move(ref_segs), exec));
        return;
    }
    if (poly_basis != "taylor" || cfg.get_nparams() < 2 ||
        cfg.get_nparams() > 4 || rfi_config.is_active()) {
        throw std::invalid_argument("EP candidates require polynomial Taylor "
                                    "and an inactive RFI config");
    }
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
    m_impl             = std::make_unique<Impl>(make_ep_freq_sweep_engine(
        search_cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
        plan_cache_file, n_runs, ref_segs, exec));
    m_impl->candidates = candidates;
}

EPFreqSweep::~EPFreqSweep()                                       = default;
EPFreqSweep::EPFreqSweep(EPFreqSweep&& other) noexcept            = default;
EPFreqSweep& EPFreqSweep::operator=(EPFreqSweep&& other) noexcept = default;

void EPFreqSweep::execute(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          const std::filesystem::path& outdir,
                          std::string_view file_prefix,
                          const io::ObservationMetadata& observation) {
    observation.validate();
    if (!m_impl->candidates) {
        m_impl->m_engine->execute(ts_e, ts_v, outdir, file_prefix);
        return;
    }
    std::filesystem::create_directories(outdir);
    std::filesystem::path staging;
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        staging =
            outdir /
            (".loki-ep-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "-" + std::to_string(attempt));
        if (std::filesystem::create_directory(staging)) {
            break;
        }
        staging.clear();
    }
    if (staging.empty()) {
        throw std::runtime_error("cannot create EP staging directory");
    }
    const auto cleanup = [&] {
        std::error_code error;
        std::filesystem::remove_all(staging, error);
    };
    try {
        m_impl->m_engine->execute(ts_e, ts_v, staging, "spool");
    } catch (...) {
        cleanup();
        throw;
    }
    const auto spool = staging / "spool_ep_results.h5";
    try {
        publish_candidates(
            spool, outdir / (std::string(file_prefix) + "_ep_results.h5"),
            *m_impl->candidates, observation);
    } catch (...) {
        const auto kept = outdir / (std::string(file_prefix) + "_ep_spool.h5");
        std::error_code error;
        std::filesystem::rename(spool, kept, error);
        if (!error) {
            cleanup();
            spdlog::error(
                "EP candidate postprocessing failed; completed runs are kept "
                "in '{}'. Retry with EPFreqSweep::publish_candidates.",
                kept.string());
        } else {
            spdlog::error("EP candidate postprocessing failed; completed runs "
                          "remain in '{}' ({})",
                          staging.string(), error.message());
        }
        throw;
    }
    cleanup();
}

void EPFreqSweep::publish_candidates(const std::filesystem::path& spool,
                                     const std::filesystem::path& output,
                                     const search::CandidateOptions& candidates,
                                     const io::ObservationMetadata& observation,
                                     std::optional<double> snr_min) {
    candidates.validate();
    observation.validate();
    if (!candidates.enabled) {
        throw std::invalid_argument(
            "publish_candidates needs candidate processing enabled");
    }
    if (!std::filesystem::exists(spool)) {
        throw std::invalid_argument("EP spool does not exist: " +
                                    spool.string());
    }
    io::detail::write_ep_candidate_results(spool, output, observation,
                                           candidates, snr_min);
}

} // namespace loki::pipelines
