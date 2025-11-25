#pragma once

#include <memory>
#include <span>
#include <vector>

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

/**
 * @brief Workspace for FFA buffers (can be reused across multiple FFA
 * instances).
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> class FFAWorkspace {
public:
    FFAWorkspace();
    explicit FFAWorkspace(const plans::FFAPlan<FoldType>& ffa_plan);
    FFAWorkspace(SizeType buffer_size, SizeType coord_size, SizeType n_params);

    // --- Rule of five: PIMPL ---
    ~FFAWorkspace();
    FFAWorkspace(FFAWorkspace&&) noexcept;
    FFAWorkspace& operator=(FFAWorkspace&&) noexcept;
    FFAWorkspace(const FFAWorkspace&)            = delete;
    FFAWorkspace& operator=(const FFAWorkspace&) = delete;

    // --- Getters ---
    [[nodiscard]] std::vector<FoldType>& get_fold_buffer() noexcept;
    [[nodiscard]] std::vector<plans::FFACoord>& get_coords() noexcept;
    [[nodiscard]] std::vector<plans::FFACoordFreq>& get_coords_freq() noexcept;

    // --- Methods ---
    void validate(const plans::FFAPlan<FoldType>& ffa_plan) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Hierarchial P-FFA folding algorithm for Pulsar Search
 *
 * @tparam FoldType The type of fold to use (float for time domain, ComplexType
 * for Fourier domain)
 */
template <SupportedFoldType FoldType> class FFA {
public:
    // Chunked FFA constructor (owns workspace)
    explicit FFA(const search::PulsarSearchConfig& cfg,
                 bool show_progress = true);

    // Pipeline-based FFA constructor uses external workspace
    explicit FFA(const search::PulsarSearchConfig& cfg,
                 FFAWorkspace<FoldType>& workspace,
                 bool show_progress = true);

    // --- Rule of five: PIMPL ---
    ~FFA();
    FFA(FFA&&) noexcept;
    FFA& operator=(FFA&&) noexcept;
    FFA(const FFA&)            = delete;
    FFA& operator=(const FFA&) = delete;

    const plans::FFAPlan<FoldType>& get_plan() const noexcept;
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<FoldType> fold);

    // This overload is ONLY enabled when FoldType is ComplexType
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold)
        requires(std::is_same_v<FoldType, ComplexType>);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using FFATime    = FFA<float>;
using FFAFourier = FFA<ComplexType>;

// Convenience function to fold time series using P-FFA (both time and Fourier
// domains)
template <SupportedFoldType FoldType>
std::tuple<std::vector<FoldType>, plans::FFAPlan<FoldType>>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::PulsarSearchConfig& cfg,
            bool quiet         = false,
            bool show_progress = false);

// Convenience function to fold time series using P-FFA in the Fourier domain
// and return the result in the time domain (floats)
std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_fourier_return_to_time(std::span<const float> ts_e,
                                   std::span<const float> ts_v,
                                   const search::PulsarSearchConfig& cfg,
                                   bool quiet         = false,
                                   bool show_progress = false);

std::tuple<std::vector<float>, plans::FFAPlan<float>>
compute_ffa_scores(std::span<const float> ts_e,
                   std::span<const float> ts_v,
                   const search::PulsarSearchConfig& cfg,
                   bool quiet         = false,
                   bool show_progress = false);

#ifdef LOKI_ENABLE_CUDA

class FFACUDA {
public:
    FFACUDA(const search::PulsarSearchConfig& cfg, int device_id);

    ~FFACUDA();
    FFACUDA(FFACUDA&&) noexcept;
    FFACUDA& operator=(FFACUDA&&) noexcept;
    FFACUDA(const FFACUDA&)            = delete;
    FFACUDA& operator=(const FFACUDA&) = delete;

    const plans::FFAPlan& get_plan() const noexcept;
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);
    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<float> fold,
                 cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

class FFACOMPLEXCUDA {
public:
    FFACOMPLEXCUDA(const search::PulsarSearchConfig& cfg, int device_id);

    ~FFACOMPLEXCUDA();
    FFACOMPLEXCUDA(FFACOMPLEXCUDA&&) noexcept;
    FFACOMPLEXCUDA& operator=(FFACOMPLEXCUDA&&) noexcept;
    FFACOMPLEXCUDA(const FFACOMPLEXCUDA&)            = delete;
    FFACOMPLEXCUDA& operator=(const FFACOMPLEXCUDA&) = delete;

    const plans::FFAPlan& get_plan() const noexcept;
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);
    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<float> fold,
                 cudaStream_t stream);
    void execute(cuda::std::span<const float> ts_e,
                 cuda::std::span<const float> ts_v,
                 cuda::std::span<ComplexTypeCUDA> fold_complex,
                 cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_cuda(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const search::PulsarSearchConfig& cfg,
                 int device_id,
                 bool quiet = false);

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_complex_cuda(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const search::PulsarSearchConfig& cfg,
                         int device_id,
                         bool quiet = false);

std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_scores_cuda(std::span<const float> ts_e,
                        std::span<const float> ts_v,
                        const search::PulsarSearchConfig& cfg,
                        int device_id,
                        bool quiet = false);
#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms