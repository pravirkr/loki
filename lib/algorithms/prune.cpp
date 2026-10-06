#include "loki/algorithms/prune.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "algorithms/prune_engine.hpp"
#include "common/dispatch.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/workspace.hpp"
#include "utils/workspace_impl.hpp"

namespace loki::algorithms {

namespace {

/// The GPU engine has no RFI controls yet; refuse rather than ignore them.
[[maybe_unused]] void check_gpu_supports(const PruneRFIConfig& rfi_config) {
    if (rfi_config.is_active()) {
        loki::detail::throw_unimplemented("EPMultiPass (rfi_config)",
                                          Backend::kCUDA);
    }
}

template <SupportedFoldType FoldType>
std::unique_ptr<detail::EPMultiPassEngine<FoldType>>
make_ep_engine(search::PulsarSearchConfig cfg,
               std::span<const float> threshold_scheme,
               std::optional<SizeType> n_runs,
               std::optional<std::vector<SizeType>> ref_segs,
               std::span<const SizeType> ascend_levels,
               SizeType max_sugg,
               SizeType batch_size,
               std::string_view poly_basis,
               bool show_progress,
               PruneRFIConfig rfi_config,
               Exec exec) {
    loki::detail::warn_ignored_nthreads(exec, "EPMultiPass");
    if (exec.backend == Backend::kCPU) {
        return detail::make_ep_cpu<FoldType>(
            std::move(cfg), threshold_scheme, n_runs, std::move(ref_segs),
            ascend_levels, max_sugg, batch_size, poly_basis, show_progress,
            std::move(rfi_config));
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        check_gpu_supports(rfi_config);
        return detail::make_ep_gpu<FoldType>(
            std::move(cfg), threshold_scheme, n_runs, std::move(ref_segs),
            ascend_levels, max_sugg, batch_size, poly_basis, exec.device);
    }
#endif
    loki::detail::throw_unavailable("EPMultiPass", exec.backend);
}

template <SupportedFoldType FoldType>
std::unique_ptr<detail::EPMultiPassEngine<FoldType>>
make_ep_engine(std::span<memory::EPWorkspace<FoldType>> workspaces,
               search::PulsarSearchConfig cfg,
               std::span<const float> threshold_scheme,
               std::optional<SizeType> n_runs,
               std::optional<std::vector<SizeType>> ref_segs,
               std::span<const SizeType> ascend_levels,
               SizeType max_sugg,
               SizeType batch_size,
               std::string_view poly_basis,
               bool show_progress,
               PruneRFIConfig rfi_config,
               Exec exec) {
    loki::detail::warn_ignored_nthreads(exec, "EPMultiPass");
    for (auto& ws : workspaces) {
        loki::detail::check_same_exec(ws.exec(), exec,
                                      "EPMultiPass (workspaces)");
    }
    if (exec.backend == Backend::kCPU) {
        std::vector<memory::EPWorkspaceCPU<FoldType>*> cpu_workspaces;
        cpu_workspaces.reserve(workspaces.size());
        for (auto& ws : workspaces) {
            cpu_workspaces.push_back(
                &memory::detail::cpu_workspace(ws, "EPMultiPass"));
        }
        return detail::make_ep_cpu<FoldType>(
            std::span(cpu_workspaces), std::move(cfg), threshold_scheme, n_runs,
            std::move(ref_segs), ascend_levels, max_sugg, batch_size,
            poly_basis, show_progress, std::move(rfi_config));
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        check_gpu_supports(rfi_config);
        if (workspaces.size() != 1) {
            throw std::invalid_argument(
                "EPMultiPass: the GPU backend runs on a single stream and "
                "takes exactly one workspace");
        }
        return detail::make_ep_gpu<FoldType>(
            workspaces.front(), std::move(cfg), threshold_scheme, n_runs,
            std::move(ref_segs), ascend_levels, max_sugg, batch_size,
            poly_basis, exec.device);
    }
#endif
    loki::detail::throw_unavailable("EPMultiPass", exec.backend);
}

} // namespace

template <SupportedFoldType FoldType> class EPMultiPass<FoldType>::Impl {
public:
    explicit Impl(std::unique_ptr<detail::EPMultiPassEngine<FoldType>> engine)
        : m_engine(std::move(engine)) {}

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir,
                 std::string_view file_prefix) {
        m_engine->execute(ts_e, ts_v, outdir, file_prefix);
    }

private:
    std::unique_ptr<detail::EPMultiPassEngine<FoldType>> m_engine;
};

template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    bool show_progress,
    PruneRFIConfig rfi_config,
    Exec exec)
    : m_impl(
          std::make_unique<Impl>(make_ep_engine<FoldType>(std::move(cfg),
                                                          threshold_scheme,
                                                          n_runs,
                                                          std::move(ref_segs),
                                                          ascend_levels,
                                                          max_sugg,
                                                          batch_size,
                                                          poly_basis,
                                                          show_progress,
                                                          std::move(rfi_config),
                                                          exec))) {}

template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(
    std::span<memory::EPWorkspace<FoldType>> workspaces,
    search::PulsarSearchConfig cfg,
    std::span<const float> threshold_scheme,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs,
    std::span<const SizeType> ascend_levels,
    SizeType max_sugg,
    SizeType batch_size,
    std::string_view poly_basis,
    bool show_progress,
    PruneRFIConfig rfi_config,
    Exec exec)
    : m_impl(
          std::make_unique<Impl>(make_ep_engine<FoldType>(workspaces,
                                                          std::move(cfg),
                                                          threshold_scheme,
                                                          n_runs,
                                                          std::move(ref_segs),
                                                          ascend_levels,
                                                          max_sugg,
                                                          batch_size,
                                                          poly_basis,
                                                          show_progress,
                                                          std::move(rfi_config),
                                                          exec))) {}

template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::~EPMultiPass() = default;

template <SupportedFoldType FoldType>
EPMultiPass<FoldType>::EPMultiPass(EPMultiPass&& other) noexcept = default;

template <SupportedFoldType FoldType>
EPMultiPass<FoldType>&
EPMultiPass<FoldType>::operator=(EPMultiPass&& other) noexcept = default;

template <SupportedFoldType FoldType>
void EPMultiPass<FoldType>::execute(std::span<const float> ts_e,
                                    std::span<const float> ts_v,
                                    const std::filesystem::path& outdir,
                                    std::string_view file_prefix) {
    m_impl->execute(ts_e, ts_v, outdir, file_prefix);
}

// Explicit instantiation
template class EPMultiPass<float>;
template class EPMultiPass<ComplexType>;

} // namespace loki::algorithms