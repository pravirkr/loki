#include "loki/pipelines/ep_freq_sweep.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "lib/common/dispatch.hpp"
#include "lib/pipelines/ep_freq_sweep_engine.hpp"

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
    std::optional<std::vector<SizeType>> ref_segs,
    Exec exec) {
    loki::detail::warn_ignored_nthreads(exec, "EPFreqSweep");
    if (exec.backend == Backend::kCPU) {
        return detail::make_ep_freq_sweep_cpu(
            cfg, show_progress, min_pd, poly_basis, ref_ducy, rfi_config,
            plan_cache_file, n_runs, std::move(ref_segs));
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        loki::detail::throw_unimplemented("EPFreqSweep", exec.backend);
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
};

EPFreqSweep::EPFreqSweep(
    const search::PulsarSearchConfig& cfg,
    bool show_progress,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    algorithms::PruneRFIConfig rfi_config,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    Exec exec)
    : m_impl(
          std::make_unique<Impl>(make_ep_freq_sweep_engine(cfg,
                                                           show_progress,
                                                           min_pd,
                                                           poly_basis,
                                                           ref_ducy,
                                                           rfi_config,
                                                           plan_cache_file,
                                                           n_runs,
                                                           std::move(ref_segs),
                                                           exec))) {}

EPFreqSweep::~EPFreqSweep()                                       = default;
EPFreqSweep::EPFreqSweep(EPFreqSweep&& other) noexcept            = default;
EPFreqSweep& EPFreqSweep::operator=(EPFreqSweep&& other) noexcept = default;

void EPFreqSweep::execute(std::span<const float> ts_e,
                          std::span<const float> ts_v,
                          const std::filesystem::path& outdir,
                          std::string_view file_prefix) {
    m_impl->m_engine->execute(ts_e, ts_v, outdir, file_prefix);
}

} // namespace loki::pipelines
