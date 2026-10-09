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

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::algorithms {

/**
 * @brief What a sweep shares between its chunks (EPFreqSweep on CUDA).
 *
 * The sweep owns every buffer; the chunk's EPMultiPassCudaCore only uses
 * them. All pointers and spans must outlive the core.
 */
template <SupportedFoldTypeCUDA FoldTypeCUDA> struct EPCudaSharedPipeline {
    /// EP workspace of the chunk group, sized from the group's maxima.
    memory::EPWorkspaceCUDA<FoldTypeCUDA>* ep_workspace{nullptr};
    memory::FFAWorkspaceCUDA<FoldTypeCUDA>* ffa_workspace{nullptr};
    math::CUFFTManager* fft_manager{nullptr};
    /// FFT manager of the pruning functors, prepared by the sweep for every
    /// chunk's nbins. Null when the functors own theirs.
    math::CUFFTManager* prune_fft_manager{nullptr};
    /// Output fold buffer (at least the chunk's FFA buffer size).
    cuda::std::span<DeviceFoldType<FoldTypeCUDA>> fold_d;
    /// Input time series on the device (nsamps each).
    cuda::std::span<const float> ts_e_d;
    cuda::std::span<const float> ts_v_d;
    /// Non-null stream the FFA and the pruning run on.
    cudaStream_t stream{nullptr};
    /// The shape @p ep_workspace was built with.
    SizeType ws_max_sugg{0};
    SizeType ws_branch_max{0};
    SizeType ws_ncoords{0};
    SizeType ws_nsegments{0};
};

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

    /// Upstream owns the workspace. @p execution_stream must be non-null and
    /// ordered after the stream the workspace was allocated on: the same
    /// stream, or a blocking stream when the workspace was allocated on the
    /// legacy default stream (as public EPWorkspace handles are).
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

    /// Shared-pipeline form: runs on the sweep's buffers (see
    /// EPCudaSharedPipeline). @p max_sugg is this chunk's, for the result file;
    /// the workspace capacity is @p pipeline.ws_max_sugg. Use execute_device().
    EPMultiPassCudaCore(
        const EPCudaSharedPipeline<FoldTypeCUDA>& pipeline,
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

    /// Runs the FFA of the chunk on the shared device inputs and prunes it.
    /// Only for the shared-pipeline constructor.
    void execute_device(const std::filesystem::path& outdir,
                        std::string_view file_prefix);

private:
    // The facade engine embeds Impl directly (no extra indirection).
    template <SupportedFoldType> friend class EPMultiPassCudaEngine;

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms
