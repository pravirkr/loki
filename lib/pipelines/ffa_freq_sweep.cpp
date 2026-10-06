#include "loki/pipelines/ffa_freq_sweep.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "common/dispatch.hpp"
#include "loki/common/backend.hpp"
#include "loki/search/configs.hpp"
#include "pipelines/ffa_freq_sweep_engine.hpp"

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
};

FFAFreqSweep::FFAFreqSweep(const search::FFASearchConfig& cfg,
                           bool show_progress,
                           Exec exec)
    : m_impl(std::make_unique<Impl>(
          make_ffa_freq_sweep_engine(cfg, show_progress, exec))) {}

FFAFreqSweep::~FFAFreqSweep()                                        = default;
FFAFreqSweep::FFAFreqSweep(FFAFreqSweep&& other) noexcept            = default;
FFAFreqSweep& FFAFreqSweep::operator=(FFAFreqSweep&& other) noexcept = default;

void FFAFreqSweep::execute(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           const std::filesystem::path& outdir,
                           std::string_view file_prefix,
                           std::string_view config_toml) {
    m_impl->m_engine->execute(ts_e, ts_v, outdir, file_prefix, config_toml);
}
} // namespace loki::pipelines
