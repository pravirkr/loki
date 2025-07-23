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

class FFAPipeline {
public:
    explicit FFAPipeline(const search::PulsarSearchConfig& cfg,
                         float max_memory_gb,
                         bool show_progress = true);

    ~FFAPipeline();
    FFAPipeline(FFAPipeline&&) noexcept;
    FFAPipeline& operator=(FFAPipeline&&) noexcept;
    FFAPipeline(const FFAPipeline&)            = delete;
    FFAPipeline& operator=(const FFAPipeline&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms