#pragma once

/**
 * @file prune_cuda.cuh
 * @brief Internal CUDA extreme-pruning implementation. Not part of the
 * installed API. The public type is loki::algorithms::EPMultiPass<FoldType>.
 */

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/workspace_cuda.cuh"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

template <SupportedFoldTypeCUDA FoldTypeCUDA> class EPMultiPassCudaCore {
public:
    EPMultiPassCudaCore(
        search::PulsarSearchConfig cfg,
        std::span<const float> threshold_scheme,
        std::optional<SizeType> n_runs                = std::nullopt,
        std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
        std::span<const SizeType> ascend_levels       = {},
        SizeType max_sugg                             = 1U << 20U,
        SizeType batch_size                           = 4096U,
        std::string_view poly_basis                   = "taylor",
        int device_id                                 = 0);

    /// Upstream owns the workspace. @p execution_stream must be the stream
    /// used to construct that workspace so scratch alloc/free matches kernels.
    EPMultiPassCudaCore(
        memory::EPWorkspaceCUDA<FoldTypeCUDA>& workspace,
        cudaStream_t execution_stream,
        search::PulsarSearchConfig cfg,
        std::span<const float> threshold_scheme,
        std::optional<SizeType> n_runs                = std::nullopt,
        std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
        std::span<const SizeType> ascend_levels       = {},
        SizeType max_sugg                             = 1U << 20U,
        SizeType batch_size                           = 4096U,
        std::string_view poly_basis                   = "taylor",
        int device_id                                 = 0);

    ~EPMultiPassCudaCore();
    EPMultiPassCudaCore(EPMultiPassCudaCore&&) noexcept;
    EPMultiPassCudaCore& operator=(EPMultiPassCudaCore&&) noexcept;
    EPMultiPassCudaCore(const EPMultiPassCudaCore&)            = delete;
    EPMultiPassCudaCore& operator=(const EPMultiPassCudaCore&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms
