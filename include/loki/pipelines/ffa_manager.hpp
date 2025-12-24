#pragma once

#include <filesystem>
#include <memory>
#include <span>

#include "loki/search/configs.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::algorithms {

class FFAManager {
public:
    explicit FFAManager(const search::PulsarSearchConfig& cfg,
                        bool show_progress = true);
    ~FFAManager();
    FFAManager(FFAManager&&) noexcept;
    FFAManager& operator=(FFAManager&&) noexcept;
    FFAManager(const FFAManager&)            = delete;
    FFAManager& operator=(const FFAManager&) = delete;

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

class FFAManagerCUDA {
public:
    explicit FFAManagerCUDA(const search::PulsarSearchConfig& cfg,
                            int device_id = 0);
    ~FFAManagerCUDA();
    FFAManagerCUDA(FFAManagerCUDA&&) noexcept;
    FFAManagerCUDA& operator=(FFAManagerCUDA&&) noexcept;
    FFAManagerCUDA(const FFAManagerCUDA&)            = delete;
    FFAManagerCUDA& operator=(const FFAManagerCUDA&) = delete;

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