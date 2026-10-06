#include "loki/utils/workspace.hpp"

#include <format>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "common/dispatch.hpp"
#include "utils/workspace_impl.hpp"

namespace loki::memory {

namespace {

template <typename Handle>
void require_sized(const Handle& ws, std::string_view what) {
    if (ws.empty()) {
        throw std::invalid_argument(
            std::format("{}: workspace handle is empty", what));
    }
}

} // namespace

// --- FFAWorkspace ---

template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace() noexcept = default;

template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan,
                                     Exec exec)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->exec = exec;
    if (exec.backend == Backend::kCPU) {
        m_impl->cpu = std::make_unique<FFAWorkspaceCPU<FoldType>>(ffa_plan);
        return;
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        m_impl->device =
            detail::make_ffa_workspace_gpu<FoldType>(ffa_plan, exec.device);
        return;
    }
#endif
    loki::detail::throw_unavailable("FFAWorkspace", exec.backend);
}

template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(SizeType buffer_size,
                                     SizeType coord_size,
                                     SizeType n_levels,
                                     SizeType n_params,
                                     Exec exec)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->exec = exec;
    if (exec.backend == Backend::kCPU) {
        m_impl->cpu = std::make_unique<FFAWorkspaceCPU<FoldType>>(
            buffer_size, coord_size, n_params);
        return;
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        m_impl->device = detail::make_ffa_workspace_gpu<FoldType>(
            buffer_size, coord_size, n_levels, n_params, exec.device);
        return;
    }
#endif
    (void)n_levels;
    loki::detail::throw_unavailable("FFAWorkspace", exec.backend);
}

template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::~FFAWorkspace() = default;
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::FFAWorkspace(FFAWorkspace&&) noexcept = default;
template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>&
FFAWorkspace<FoldType>::operator=(FFAWorkspace&&) noexcept = default;

template <SupportedFoldType FoldType>
Exec FFAWorkspace<FoldType>::exec() const {
    require_sized(*this, "FFAWorkspace::exec");
    return m_impl->exec;
}

template <SupportedFoldType FoldType>
FFAWorkspace<FoldType>::Impl& FFAWorkspace<FoldType>::impl() {
    require_sized(*this, "FFAWorkspace");
    return *m_impl;
}

// --- EPWorkspace ---

template <SupportedFoldType FoldType>
EPWorkspace<FoldType>::EPWorkspace() noexcept = default;

template <SupportedFoldType FoldType>
EPWorkspace<FoldType>::EPWorkspace(SizeType batch_size,
                                   SizeType branch_max,
                                   SizeType max_sugg,
                                   SizeType ncoords_ffa,
                                   SizeType nparams,
                                   SizeType nbins,
                                   SizeType nsegments,
                                   Exec exec)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->exec = exec;
    if (exec.backend == Backend::kCPU) {
        m_impl->cpu = std::make_unique<EPWorkspaceCPU<FoldType>>(
            batch_size, branch_max, max_sugg, ncoords_ffa, nparams, nbins,
            nsegments);
        return;
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        m_impl->device = detail::make_ep_workspace_gpu<FoldType>(
            batch_size, branch_max, max_sugg, ncoords_ffa, nparams, nbins,
            nsegments, exec.device);
        return;
    }
#endif
    loki::detail::throw_unavailable("EPWorkspace", exec.backend);
}

template <SupportedFoldType FoldType>
EPWorkspace<FoldType>::~EPWorkspace() = default;
template <SupportedFoldType FoldType>
EPWorkspace<FoldType>::EPWorkspace(EPWorkspace&&) noexcept = default;
template <SupportedFoldType FoldType>
EPWorkspace<FoldType>&
EPWorkspace<FoldType>::operator=(EPWorkspace&&) noexcept = default;

template <SupportedFoldType FoldType> Exec EPWorkspace<FoldType>::exec() const {
    require_sized(*this, "EPWorkspace::exec");
    return m_impl->exec;
}

template <SupportedFoldType FoldType>
float EPWorkspace<FoldType>::get_memory_usage_gib() const {
    require_sized(*this, "EPWorkspace::get_memory_usage_gib");
    return m_impl->cpu ? m_impl->cpu->get_memory_usage_gib()
                       : m_impl->device->get_memory_usage_gib();
}

template <SupportedFoldType FoldType>
EPWorkspace<FoldType>::Impl& EPWorkspace<FoldType>::impl() {
    require_sized(*this, "EPWorkspace");
    return *m_impl;
}

// --- Internal accessors ---

namespace detail {

template <SupportedFoldType FoldType>
FFAWorkspaceCPU<FoldType>& cpu_workspace(FFAWorkspace<FoldType>& ws,
                                         std::string_view what) {
    auto& impl = ws.impl();
    if (!impl.cpu) {
        throw std::invalid_argument(
            std::format("{}: expected a CPU FFAWorkspace", what));
    }
    return *impl.cpu;
}

template <SupportedFoldType FoldType>
EPWorkspaceCPU<FoldType>& cpu_workspace(EPWorkspace<FoldType>& ws,
                                        std::string_view what) {
    auto& impl = ws.impl();
    if (!impl.cpu) {
        throw std::invalid_argument(
            std::format("{}: expected a CPU EPWorkspace", what));
    }
    return *impl.cpu;
}

template FFAWorkspaceCPU<float>& cpu_workspace(FFAWorkspace<float>&,
                                               std::string_view);
template FFAWorkspaceCPU<ComplexType>& cpu_workspace(FFAWorkspace<ComplexType>&,
                                                     std::string_view);
template EPWorkspaceCPU<float>& cpu_workspace(EPWorkspace<float>&,
                                              std::string_view);
template EPWorkspaceCPU<ComplexType>& cpu_workspace(EPWorkspace<ComplexType>&,
                                                    std::string_view);

} // namespace detail

template class FFAWorkspace<float>;
template class FFAWorkspace<ComplexType>;
template class EPWorkspace<float>;
template class EPWorkspace<ComplexType>;

} // namespace loki::memory
