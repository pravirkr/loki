#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/workspace.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

/**
 * @brief Hierarchial EP (Extreme Pruning) algorithm for Pulsar Search
 *
 * @tparam FoldType The type of fold to use (float for time domain, ComplexType
 * for Fourier domain)
 */
template <SupportedFoldType FoldType> class EPMultiPass {
public:
    // Chunked EP constructor (owns workspace)
    EPMultiPass(search::PulsarSearchConfig cfg,
                std::span<const float> threshold_scheme,
                std::optional<SizeType> n_runs                = std::nullopt,
                std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
                SizeType max_sugg                             = 1U << 18U,
                SizeType batch_size                           = 1024U,
                std::string_view poly_basis                   = "taylor",
                bool show_progress                            = true);

    // Pipeline-based EP constructor uses external workspace
    EPMultiPass(std::span<memory::EPWorkspace<FoldType>> workspaces,
                search::PulsarSearchConfig cfg,
                std::span<const float> threshold_scheme,
                std::optional<SizeType> n_runs                = std::nullopt,
                std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
                SizeType max_sugg                             = 1U << 18U,
                SizeType batch_size                           = 1024U,
                std::string_view poly_basis                   = "taylor",
                bool show_progress                            = true);

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

#ifdef LOKI_ENABLE_CUDA

template <SupportedFoldTypeCUDA FoldTypeCUDA> class EPMultiPassCUDA {
public:
    // Chunked EP constructor (owns workspace)
    EPMultiPassCUDA(
        search::PulsarSearchConfig cfg,
        std::span<const float> threshold_scheme,
        std::optional<SizeType> n_runs                = std::nullopt,
        std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
        SizeType max_sugg                             = 1U << 20U,
        SizeType batch_size                           = 4096U,
        std::string_view poly_basis                   = "taylor",
        int device_id                                 = 0);

    // Pipeline-based EP constructor uses external workspace
    EPMultiPassCUDA(
        memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
        search::PulsarSearchConfig cfg,
        std::span<const float> threshold_scheme,
        std::optional<SizeType> n_runs                = std::nullopt,
        std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
        SizeType max_sugg                             = 1U << 20U,
        SizeType batch_size                           = 4096U,
        std::string_view poly_basis                   = "taylor",
        int device_id                                 = 0);

    ~EPMultiPassCUDA();
    EPMultiPassCUDA(EPMultiPassCUDA&&) noexcept;
    EPMultiPassCUDA& operator=(EPMultiPassCUDA&&) noexcept;
    EPMultiPassCUDA(const EPMultiPassCUDA&)            = delete;
    EPMultiPassCUDA& operator=(const EPMultiPassCUDA&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using EPMultiPassTimeCUDA    = EPMultiPassCUDA<float>;
using EPMultiPassFourierCUDA = EPMultiPassCUDA<ComplexTypeCUDA>;

#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms