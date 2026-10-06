#pragma once

/**
 * @file workspace.hpp
 * @brief Reusable buffers for FFA and EP pruning, on any backend.
 *
 * Both workspaces are opaque handles. The buffers live on the backend named
 * by the Exec the handle was built with: host vectors for CPU, device memory
 * for CUDA. Allocate a workspace once, sized for the largest search it will
 * serve, and pass it to several FFA / EPMultiPass instances on the same
 * backend and device to avoid repeated allocation.
 */

#include <memory>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"

namespace loki::memory {

/**
 * @brief FFA fold and coordinate buffers, shareable across FFA instances.
 *
 * @tparam FoldType float (time domain) or ComplexType (Fourier domain).
 */
template <SupportedFoldType FoldType> class FFAWorkspace {
public:
    /// Empty handle. Assign a sized workspace before passing it to an FFA.
    FFAWorkspace() noexcept;
    /// Buffers sized for @p ffa_plan.
    explicit FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan,
                          Exec exec = {});
    /// Buffers sized explicitly, e.g. for the largest of several plans.
    /// @p n_levels is used by the GPU backends only.
    FFAWorkspace(SizeType buffer_size,
                 SizeType coord_size,
                 SizeType n_levels,
                 SizeType n_params,
                 Exec exec = {});

    ~FFAWorkspace();
    FFAWorkspace(FFAWorkspace&&) noexcept;
    FFAWorkspace& operator=(FFAWorkspace&&) noexcept;
    FFAWorkspace(const FFAWorkspace&)            = delete;
    FFAWorkspace& operator=(const FFAWorkspace&) = delete;

    /// Backend and device the buffers live on.
    [[nodiscard]] Exec exec() const;
    [[nodiscard]] bool empty() const noexcept { return m_impl == nullptr; }

    /// Backend storage. Defined inside the library only.
    class Impl;
    [[nodiscard]] Impl& impl();

private:
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Per-worker EP pruning buffers (beam, branching and seed storage).
 *
 * @tparam FoldType float (time domain) or ComplexType (Fourier domain).
 */
template <SupportedFoldType FoldType> class EPWorkspace {
public:
    /// Empty handle. Assign a sized workspace before passing it on.
    EPWorkspace() noexcept;
    EPWorkspace(SizeType batch_size,
                SizeType branch_max,
                SizeType max_sugg,
                SizeType ncoords_ffa,
                SizeType nparams,
                SizeType nbins,
                SizeType nsegments,
                Exec exec = {});

    ~EPWorkspace();
    EPWorkspace(EPWorkspace&&) noexcept;
    EPWorkspace& operator=(EPWorkspace&&) noexcept;
    EPWorkspace(const EPWorkspace&)            = delete;
    EPWorkspace& operator=(const EPWorkspace&) = delete;

    /// Backend and device the buffers live on.
    [[nodiscard]] Exec exec() const;
    [[nodiscard]] bool empty() const noexcept { return m_impl == nullptr; }
    /// Total allocation of this workspace, in GiB.
    [[nodiscard]] float get_memory_usage_gib() const;

    /// Backend storage. Defined inside the library only.
    class Impl;
    [[nodiscard]] Impl& impl();

private:
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::memory
