#pragma once

#include <filesystem>
#include <memory>
#include <span>

#include "loki/search/configs.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

class EPFreqSweep {
public:
    explicit EPFreqSweep(const search::PulsarSearchConfig& cfg,
                         bool show_progress = true);
    ~EPFreqSweep();
    EPFreqSweep(EPFreqSweep&&) noexcept;
    EPFreqSweep& operator=(EPFreqSweep&&) noexcept;
    EPFreqSweep(const EPFreqSweep&)      = delete;
    EPFreqSweep& operator=(EPFreqSweep&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

    // Opaque handle to the implementation
    class BaseImpl;

private:
    std::unique_ptr<BaseImpl> m_impl;
};

#ifdef LOKI_ENABLE_CUDA

class EPFreqSweepCUDA {
public:
    explicit EPFreqSweepCUDA(const search::PulsarSearchConfig& cfg,
                             int device_id = 0);
    ~EPFreqSweepCUDA();
    EPFreqSweepCUDA(EPFreqSweepCUDA&&) noexcept;
    EPFreqSweepCUDA& operator=(EPFreqSweepCUDA&&) noexcept;
    EPFreqSweepCUDA(const EPFreqSweepCUDA&)      = delete;
    EPFreqSweepCUDA& operator=(EPFreqSweepCUDA&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

    // Opaque handle to the implementation
    class BaseImpl;

private:
    std::unique_ptr<BaseImpl> m_impl;
};

#endif // LOKI_ENABLE_CUDA

} // namespace loki::algorithms