#pragma once

/**
 * @file prune_engine.hpp
 * @brief Backend engine interface for EPMultiPass. Internal.
 */

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/workspace.hpp"

#include "lib/utils/fft_impl.hpp"
#include "lib/utils/workspace_impl.hpp"

namespace loki::algorithms::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

template <SupportedFoldType FoldType> class EPMultiPassEngine {
protected:
    EPMultiPassEngine() = default;

public:
    virtual ~EPMultiPassEngine() = default;

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix) = 0;

    EPMultiPassEngine(const EPMultiPassEngine&)            = delete;
    EPMultiPassEngine& operator=(const EPMultiPassEngine&) = delete;
    EPMultiPassEngine(EPMultiPassEngine&&)                 = delete;
    EPMultiPassEngine& operator=(EPMultiPassEngine&&)      = delete;
};

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_cpu(search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            bool show_progress,
            PruneRFIConfig rfi_config);

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_cpu(std::span<memory::EPWorkspaceCPU<FoldType>* const> workspaces,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            bool show_progress,
            PruneRFIConfig rfi_config);

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_cpu(std::span<memory::EPWorkspaceCPU<FoldType>* const> workspaces,
            memory::FFAWorkspaceCPU<FoldType>& ffa_workspace,
            math::FFTWManager& fft_manager,
            std::span<FoldType> ffa_fold,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            bool show_progress,
            PruneRFIConfig rfi_config);

template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_gpu(search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            int device_id);

/// Runs on the caller's GPU workspace (a GPU handle on @p device_id; the
/// facade checks). The GPU engine is single-stream: one workspace per run.
template <SupportedFoldType FoldType>
std::unique_ptr<EPMultiPassEngine<FoldType>>
make_ep_gpu(memory::EPWorkspace<FoldType>& workspace,
            search::PulsarSearchConfig cfg,
            std::span<const float> threshold_scheme,
            std::optional<SizeType> n_runs,
            std::optional<std::vector<SizeType>> ref_segs,
            std::span<const SizeType> ascend_levels,
            SizeType max_sugg,
            SizeType batch_size,
            std::string_view poly_basis,
            int device_id);

} // namespace loki::algorithms::detail
