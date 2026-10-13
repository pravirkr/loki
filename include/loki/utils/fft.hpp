#pragma once

/**
 * @file fft.hpp
 * @brief Backend FFT plan cache, shareable across FFA / EPMultiPass instances.
 */

#include <memory>
#include <span>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

namespace loki::detail {
/// Library-internal accessor to a handle's backend storage.
struct HandleAccess;
} // namespace loki::detail

namespace loki::math {

/**
 * @brief Owns the tables of the batched real FFTs (on CPU, loki's host FFT;
 * on CUDA, the portable block FFT).
 *
 * Opaque handle. Build one manager per backend and device, prepare the
 * transform lengths it will see, and share it between instances to amortise
 * building the tables.
 */
class FFTManager {
public:
    /// Empty handle. Assign a manager before passing it on.
    FFTManager() noexcept;
    /// Manager for `exec`'s backend and device. `exec` has no default on
    /// purpose: `FFTManager()` is the empty handle, not a CPU manager. Write
    /// `FFTManager(Exec::cpu())` for one.
    explicit FFTManager(Exec exec);

    ~FFTManager();
    FFTManager(FFTManager&&) noexcept;
    FFTManager& operator=(FFTManager&&) noexcept;
    FFTManager(const FFTManager&)            = delete;
    FFTManager& operator=(const FFTManager&) = delete;

    /**
     * @brief Pre-build the transform tables for each real length in
     * @p n_reals. Repeated calls with the same lengths are no-ops.
     */
    void prepare_plans(std::span<const SizeType> n_reals);

    /**
     * @brief Register @p n_reals; their tables are built on first use
     * (suited to fixed lengths with varying batch sizes, as in EP pruning).
     */
    void prepare_exact_plans(std::span<const SizeType> n_reals);

    [[nodiscard]] bool has_prepared(SizeType n_real) const;
    [[nodiscard]] SizeType n_cached_plans() const;

    /// Backend and device the plans are built for.
    [[nodiscard]] Exec exec() const;
    [[nodiscard]] bool empty() const noexcept { return m_impl == nullptr; }

private:
    friend struct loki::detail::HandleAccess;

    /// Backend storage. Defined inside the library only.
    class Impl;
    [[nodiscard]] Impl& impl();

    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::math
