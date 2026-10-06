#pragma once

/**
 * @file ffa_cuda.cuh
 * @brief Internal CUDA FFA implementation. Not part of the installed API.
 *
 * Included only from lib/cuda translation units. The public type is
 * loki::algorithms::FFA<FoldType>, which dispatches here through FFAEngine.
 */

#include <memory>
#include <span>
#include <tuple>
#include <vector>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cuda/fft_cuda.cuh"
#include "cuda/workspace_cuda.cuh"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

template <SupportedFoldTypeCUDA FoldTypeCUDA> class FFACudaCore {
public:
    using HostFoldT   = HostFoldType<FoldTypeCUDA>;
    using DeviceFoldT = DeviceFoldType<FoldTypeCUDA>;

    explicit FFACudaCore(const search::FFASearchConfig& cfg, int device_id = 0);

    explicit FFACudaCore(memory::FFAWorkspaceCUDA<FoldTypeCUDA>& workspace,
                         const search::FFASearchConfig& cfg,
                         int device_id = 0);

    explicit FFACudaCore(memory::FFAWorkspaceCUDA<FoldTypeCUDA>& workspace,
                         math::CUFFTManager& fft_manager,
                         const search::FFASearchConfig& cfg,
                         int device_id = 0);

    ~FFACudaCore();
    FFACudaCore(FFACudaCore&&) noexcept;
    FFACudaCore& operator=(FFACudaCore&&) noexcept;
    FFACudaCore(const FFACudaCore&)            = delete;
    FFACudaCore& operator=(const FFACudaCore&) = delete;

    const plans::FFAPlan<HostFoldT>& get_plan() const noexcept;
    [[nodiscard]] plans::FFAPlan<HostFoldT> extract_plan() && noexcept;
    float get_brute_fold_timing() const noexcept;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<HostFoldT> fold);

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 cuda::std::span<DeviceFoldT> fold_d);

    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<DeviceFoldT> fold,
                 cudaStream_t stream);

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold)
        requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>);

    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<float> fold,
                 cudaStream_t stream = nullptr)
        requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::tuple<std::vector<HostFoldType<FoldTypeCUDA>>,
           plans::FFAPlan<HostFoldType<FoldTypeCUDA>>>
compute_ffa_cuda(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const search::FFASearchConfig& cfg,
                 int device_id,
                 bool quiet = false);

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::tuple<thrust::device_vector<FoldTypeCUDA>,
           plans::FFAPlan<HostFoldType<FoldTypeCUDA>>>
compute_ffa_cuda_device(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        const search::FFASearchConfig& cfg,
                        int device_id);

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_fourier_return_to_time_cuda(std::span<const float> ts_e,
                                        std::span<const float> ts_v,
                                        const search::FFASearchConfig& cfg,
                                        int device_id,
                                        bool quiet = false);

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores_cuda(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        const search::FFASearchConfig& cfg,
                        int device_id,
                        bool quiet = false);

} // namespace loki::algorithms
