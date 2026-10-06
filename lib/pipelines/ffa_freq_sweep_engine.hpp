#pragma once

/**
 * @file ffa_freq_sweep_engine.hpp
 * @brief Backend engine interface for FFAFreqSweep. Internal.
 */

#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include "loki/search/configs.hpp"

namespace loki::pipelines::detail {

// make_*_cpu is defined in lib/cpu/, make_*_gpu in lib/cuda/ (GPU builds only).

class FFAFreqSweepEngine {
public:
    virtual ~FFAFreqSweepEngine() = default;

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix,
                         std::string_view config_toml) = 0;
};

std::unique_ptr<FFAFreqSweepEngine>
make_ffa_freq_sweep_cpu(const search::FFASearchConfig& cfg, bool show_progress);

std::unique_ptr<FFAFreqSweepEngine> make_ffa_freq_sweep_gpu(
    const search::FFASearchConfig& cfg, int device_id, bool show_progress);

} // namespace loki::pipelines::detail
