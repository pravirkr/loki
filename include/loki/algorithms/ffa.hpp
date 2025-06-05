#pragma once

#include <memory>
#include <span>
#include <vector>

#include "loki/algorithms/plans.hpp"
#include "loki/search/configs.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

class FFA {
public:
    explicit FFA(const search::PulsarSearchConfig& cfg);

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

// Convenience function to fold time series using FFA method
std::vector<float> compute_ffa(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               const search::PulsarSearchConfig& cfg);

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

std::vector<float> compute_ffa_cuda(std::span<const float> ts_e,
                                    std::span<const float> ts_v,
                                    const search::PulsarSearchConfig& cfg,
                                    int device_id);
#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms