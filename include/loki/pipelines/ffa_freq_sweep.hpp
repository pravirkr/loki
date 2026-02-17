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

class FFAFreqSweep {
public:
    explicit FFAFreqSweep(const search::PulsarSearchConfig& cfg,
                          bool show_progress = true);
    ~FFAFreqSweep();
    FFAFreqSweep(FFAFreqSweep&&) noexcept;
    FFAFreqSweep& operator=(FFAFreqSweep&&) noexcept;
    FFAFreqSweep(const FFAFreqSweep&)            = delete;
    FFAFreqSweep& operator=(const FFAFreqSweep&) = delete;

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

class FFAFreqSweepCUDA {
public:
    explicit FFAFreqSweepCUDA(const search::PulsarSearchConfig& cfg,
                              int device_id = 0);
    ~FFAFreqSweepCUDA();
    FFAFreqSweepCUDA(FFAFreqSweepCUDA&&) noexcept;
    FFAFreqSweepCUDA& operator=(FFAFreqSweepCUDA&&) noexcept;
    FFAFreqSweepCUDA(const FFAFreqSweepCUDA&)            = delete;
    FFAFreqSweepCUDA& operator=(const FFAFreqSweepCUDA&) = delete;

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