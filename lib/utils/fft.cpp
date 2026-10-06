#include "loki/utils/fft.hpp"

#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

#include "common/dispatch.hpp"
#include "utils/fft_impl.hpp"

namespace loki::math {

FFTManager::FFTManager() noexcept = default;

FFTManager::FFTManager(Exec exec) : m_impl(std::make_unique<Impl>()) {
    m_impl->exec = exec;
    if (exec.backend == Backend::kCPU) {
        m_impl->cpu = std::make_unique<FFTWManager>();
        return;
    }
#ifdef LOKI_ENABLE_GPU
    if (exec.backend == loki::detail::kGPUBackend) {
        m_impl->device = detail::make_fft_manager_gpu(exec.device);
        return;
    }
#endif
    loki::detail::throw_unavailable("FFTManager", exec.backend);
}

FFTManager::~FFTManager()                                = default;
FFTManager::FFTManager(FFTManager&&) noexcept            = default;
FFTManager& FFTManager::operator=(FFTManager&&) noexcept = default;

FFTManager::Impl& FFTManager::impl() {
    if (m_impl == nullptr) {
        throw std::invalid_argument("FFTManager: handle is empty");
    }
    return *m_impl;
}

Exec FFTManager::exec() const {
    if (m_impl == nullptr) {
        throw std::invalid_argument("FFTManager::exec: handle is empty");
    }
    return m_impl->exec;
}

void FFTManager::prepare_plans(std::span<const SizeType> n_reals) {
    auto& impl = this->impl();
    if (impl.cpu) {
        impl.cpu->prepare_plans(n_reals);
        return;
    }
    impl.device->prepare_plans(n_reals);
}

void FFTManager::prepare_exact_plans(std::span<const SizeType> n_reals) {
    auto& impl = this->impl();
    if (impl.cpu) {
        impl.cpu->prepare_exact_plans(n_reals);
        return;
    }
    impl.device->prepare_exact_plans(n_reals);
}

bool FFTManager::has_prepared(SizeType n_real) const {
    if (m_impl == nullptr) {
        return false;
    }
    return m_impl->cpu ? m_impl->cpu->has_prepared(n_real)
                       : m_impl->device->has_prepared(n_real);
}

SizeType FFTManager::n_cached_plans() const {
    if (m_impl == nullptr) {
        return 0;
    }
    return m_impl->cpu ? m_impl->cpu->n_cached_plans()
                       : m_impl->device->n_cached_plans();
}

namespace detail {

FFTWManager& cpu_fft(FFTManager& fft, std::string_view what) {
    auto& impl = loki::detail::HandleAccess::impl(fft);
    if (!impl.cpu) {
        throw std::invalid_argument(
            std::format("{}: expected a CPU FFTManager", what));
    }
    return *impl.cpu;
}

} // namespace detail

} // namespace loki::math
