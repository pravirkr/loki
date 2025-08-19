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

class FFA {
public:
    explicit FFA(const search::PulsarSearchConfig& cfg,
                 bool show_progress = true);

    ~FFA();
    FFA(FFA&&) noexcept;
    FFA& operator=(FFA&&) noexcept;
    FFA(const FFA&)            = delete;
    FFA& operator=(const FFA&) = delete;

    const plans::FFAPlan& get_plan() const noexcept;
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

class FFACOMPLEX {
public:
    explicit FFACOMPLEX(const search::PulsarSearchConfig& cfg,
                        bool show_progress = true);

    ~FFACOMPLEX();
    FFACOMPLEX(FFACOMPLEX&&) noexcept;
    FFACOMPLEX& operator=(FFACOMPLEX&&) noexcept;
    FFACOMPLEX(const FFACOMPLEX&)            = delete;
    FFACOMPLEX& operator=(const FFACOMPLEX&) = delete;

    const plans::FFAPlan& get_plan() const noexcept;
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<ComplexType> fold_complex);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Convenience function to fold time series using FFA method
std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa(std::span<const float> ts_e,
            std::span<const float> ts_v,
            const search::PulsarSearchConfig& cfg,
            bool quiet         = false,
            bool show_progress = false);

// Convenience function to fold time series using FFA method with Fourier shifts
std::tuple<std::vector<float>, plans::FFAPlan>
compute_ffa_complex(std::span<const float> ts_e,
                    std::span<const float> ts_v,
                    const search::PulsarSearchConfig& cfg,
                    bool quiet         = false,
                    bool show_progress = false);

// Convenience function to fold time series using FFA method with Fourier shifts
// and return the result in the complex domain
std::tuple<std::vector<ComplexType>, plans::FFAPlan>
compute_ffa_complex_domain(std::span<const float> ts_e,
                           std::span<const float> ts_v,
                           const search::PulsarSearchConfig& cfg,
                           bool quiet         = false,
                           bool show_progress = false);

std::tuple<std::vector<float>, plans::FFAPlan>
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