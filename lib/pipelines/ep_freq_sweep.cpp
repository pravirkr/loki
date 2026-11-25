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
#include "loki/common/interrupt.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/common/dispatch.hpp"
#include "lib/io/candidate_results.hpp"
#include "lib/pipelines/ep_freq_sweep_engine.hpp"
#include "lib/pipelines/sweep_input.hpp"
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
        if (rfi_config.impulsive_veto) {
            loki::detail::throw_unimplemented(
                "EPFreqSweep (rfi_config.impulsive_veto)", exec.backend);
        }
        return detail::make_ep_freq_sweep_gpu(cfg, min_pd, poly_basis, ref_ducy,
                                              plan_cache_file, n_runs, ref_segs,
                                              exec.device, rfi_config);
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
    Exec exec;
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
                                                              exec))) {
    m_impl->exec = exec;
}

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
    const bool taylor = poly_basis == "taylor" && cfg.get_nparams() >= 2 &&
                        cfg.get_nparams() <= kParamNames.size();
    if (candidates.enabled && !taylor) {
        throw std::invalid_argument("EP candidates require polynomial Taylor");
    }
    // The search reduces each run as it writes it, exactly as publication
    // would (CandidateOptions::ep_reduces_runs): the spool, and without
    // candidates the raw result, then hold only the retained leaves, and
    // `loki publish` of that raw result gives the search's product.
    const bool reduce = taylor && candidates.ep_reduces_runs();
    auto search_cfg   = cfg;
    if (exec.backend == Backend::kCPU && taylor &&
        candidates.ep_reserves_memory()) {
        const auto remaining =
            cfg.get_max_process_memory_gb() - candidates.memory_gb;
        if (remaining <= 0) {
            throw std::invalid_argument(
                "candidate host memory budget must be smaller than total CPU "
                "search memory budget");
        }
        search_cfg.set_max_process_memory_gb(remaining);
    }
    m_impl       = std::make_unique<Impl>(make_ep_freq_sweep_engine(
        search_cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
        plan_cache_file, n_runs, std::move(ref_segs), exec));
    m_impl->exec = exec;
    if (candidates.enabled) {
        m_impl->candidates = candidates;
    }
    if (reduce) {
        m_impl->m_engine->set_leaf_reduction(search::detail::LeafReductionSpec{
            .options = candidates,
            .snr_min = cfg.get_snr_min(),
            .tobs    = cfg.get_tobs(),
        });
    }
}

EPFreqSweep::~EPFreqSweep()                                       = default;
EPFreqSweep::EPFreqSweep(EPFreqSweep&& other) noexcept            = default;
EPFreqSweep& EPFreqSweep::operator=(EPFreqSweep&& other) noexcept = default;

namespace {

/// Searches @p input; with candidates, through a staging spool that is
/// published, or kept for publish_candidates() when that fails.
void run_ep(detail::EPFreqSweepEngine& engine,
            const std::optional<search::CandidateOptions>& candidates,
            const detail::SweepInput& input,
            const std::filesystem::path& outdir,
            std::string_view file_prefix,
            const io::ObservationMetadata& observation) {
    observation.validate();
    if (!candidates) {
        engine.execute(input, outdir, file_prefix);
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
        engine.execute(input, staging, "spool");
    } catch (...) {
        cleanup();
        throw;
    }
    const auto spool = staging / "spool_ep_results.h5";
    // The search finished: keep its runs so postprocessing can be retried
    // without searching again, and remove the staging directory.
    const auto keep_spool = [&](std::string_view reason) {
        const auto kept = outdir / (std::string(file_prefix) + "_ep_spool.h5");
        std::error_code error;
        std::filesystem::rename(spool, kept, error);
        if (!error) {
            cleanup();
            std::error_code size_error;
            const auto bytes = std::filesystem::file_size(kept, size_error);
            spdlog::error(
                "EP candidate postprocessing {}; the completed search is kept "
                "in '{}' ({:.1f} MB). Publish it with `loki publish '{}'` "
                "(Python: libloki.prune.EPFreqSweep.publish_candidates)",
                reason, kept.string(),
                size_error ? 0.0 : static_cast<double>(bytes) / 1.0e6,
                kept.string());
        } else {
            spdlog::error("EP candidate postprocessing {}; completed runs "
                          "remain in '{}' ({})",
                          reason, staging.string(), error.message());
        }
    };
    try {
        EPFreqSweep::publish_candidates(
            spool, outdir / (std::string(file_prefix) + "_ep_results.h5"),
            *candidates, observation);
    } catch (const loki::Interrupted&) {
        keep_spool("was interrupted");
        throw;
    } catch (...) {
        keep_spool("failed");
        throw;
    }
    cleanup();
}

} // namespace

void EPFreqSweep::execute(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          const std::filesystem::path& outdir,
                          std::string_view file_prefix,
                          const io::ObservationMetadata& observation) {
    run_ep(*m_impl->m_engine, m_impl->candidates,
           detail::SweepInput::host(ts_e, ts_v), outdir, file_prefix,
           observation);
}

void EPFreqSweep::execute(const io::SearchSeries& series,
                          const std::filesystem::path& outdir,
                          std::string_view file_prefix,
                          const io::ObservationMetadata& observation) {
    run_ep(*m_impl->m_engine, m_impl->candidates,
           detail::sweep_input(series, m_impl->exec, "EPFreqSweep::execute"),
           outdir, file_prefix, observation);
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
