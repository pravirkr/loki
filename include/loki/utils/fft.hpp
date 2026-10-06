#pragma once

/**
 * @file fft.hpp
 * @brief Backend FFT plan cache, shareable across FFA / EPMultiPass instances.
 */

#include <memory>
#include <span>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

namespace loki::math {

/**
 * @brief Owns batched real-FFT plans (FFTW on CPU, cuFFT on CUDA).
 *
 * Opaque handle. Fourier-domain searches create and destroy plans on every
 * call unless a manager with prepared plans is passed in. Build one manager
 * per backend and device, prepare the transform lengths it will see, and
 * share it between instances to amortise planning.
 */
class FFTManager {
public:
    /// Empty handle. Assign a manager before passing it on.
    FFTManager() noexcept;
    explicit FFTManager(Exec exec);

    ~FFTManager();
    FFTManager(FFTManager&&) noexcept;
    FFTManager& operator=(FFTManager&&) noexcept;
    FFTManager(const FFTManager&)            = delete;
    FFTManager& operator=(const FFTManager&) = delete;

    /**
     * @brief Pre-build plans for each real length in @p n_reals, for every
     * power-of-two batch size up to the backend maximum. Repeated calls with
     * the same lengths are no-ops.
     */
    void prepare_plans(std::span<const SizeType> n_reals);

    /**
     * @brief Register @p n_reals for lazily cached exact-batch inverse plans
     * (suited to fixed lengths with varying batch sizes, as in EP pruning).
     */
    void prepare_exact_plans(std::span<const SizeType> n_reals);

    [[nodiscard]] bool has_prepared(SizeType n_real) const;
    [[nodiscard]] SizeType n_cached_plans() const;

    /// Backend and device the plans are built for.
    [[nodiscard]] Exec exec() const;
    [[nodiscard]] bool empty() const noexcept { return m_impl == nullptr; }

    /// Backend storage. Defined inside the library only.
    class Impl;
    [[nodiscard]] Impl& impl();

private:
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::math
