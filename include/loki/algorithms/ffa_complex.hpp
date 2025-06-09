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

class FFACOMPLEX {
public:
    explicit FFACOMPLEX(const search::PulsarSearchConfig& cfg);

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
                 std::span<ComplexType> fold);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Convenience function to fold time series using FFA method with Fourier shifts
std::vector<float> compute_ffa_complex(std::span<const float> ts_e,
                                       std::span<const float> ts_v,
                                       const search::PulsarSearchConfig& cfg);

#ifdef LOKI_ENABLE_CUDA

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
                 cuda::std::span<ComplexTypeCUDA> fold,
                 cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

std::vector<float>
compute_ffa_complex_cuda(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const search::PulsarSearchConfig& cfg,
                         int device_id);
#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms