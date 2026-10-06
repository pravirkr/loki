#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::algorithms {

/**
 * @brief Hierarchial EP (Extreme Pruning) algorithm for Pulsar Search
 *
 * @tparam FoldType The type of fold to use (float for time domain, ComplexType
 * for Fourier domain)
 */
template <SupportedFoldType FoldType> class EPMultiPass {
public:
    /**
     * @brief Owns its workspaces (one per CPU thread).
     *
     * The CPU thread count comes from @p cfg; @p exec selects the backend
     * and device. The GPU backend does not support @p rfi_config yet and
     * throws if it is active.
     */
    EPMultiPass(search::PulsarSearchConfig cfg,
                std::span<const float> threshold_scheme,
                std::optional<SizeType> n_runs                = std::nullopt,
                std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
                std::span<const SizeType> ascend_levels       = {},
                SizeType max_sugg                             = 1U << 18U,
                SizeType batch_size                           = 1024U,
                std::string_view poly_basis                   = "taylor",
                bool show_progress                            = true,
                PruneRFIConfig rfi_config                     = {},
                Exec exec                                     = {});

    /**
     * @brief Runs on caller-owned workspaces, so several runs (e.g. one per
     * frequency chunk) can share one allocation.
     *
     * Every workspace must be built for the same backend and device as
     * @p exec and sized for @p cfg. The CPU backend needs at least one
     * workspace per thread; the GPU backend takes exactly one.
     */
    EPMultiPass(std::span<memory::EPWorkspace<FoldType>> workspaces,
                search::PulsarSearchConfig cfg,
                std::span<const float> threshold_scheme,
                std::optional<SizeType> n_runs                = std::nullopt,
                std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
                std::span<const SizeType> ascend_levels       = {},
                SizeType max_sugg                             = 1U << 18U,
                SizeType batch_size                           = 1024U,
                std::string_view poly_basis                   = "taylor",
                bool show_progress                            = true,
                PruneRFIConfig rfi_config                     = {},
                Exec exec                                     = {});

    // --- Rule of five: PIMPL ---
    ~EPMultiPass();
    EPMultiPass(EPMultiPass&&) noexcept;
    EPMultiPass& operator=(EPMultiPass&&) noexcept;
    EPMultiPass(const EPMultiPass&)            = delete;
    EPMultiPass& operator=(const EPMultiPass&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using EPMultiPassTime    = EPMultiPass<float>;
using EPMultiPassFourier = EPMultiPass<ComplexType>;

} // namespace loki::algorithms
