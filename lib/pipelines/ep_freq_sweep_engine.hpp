#pragma once

/**
 * @file ep_freq_sweep_engine.hpp
 * @brief Backend engine interface for EPFreqSweep. Internal.
 */

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms::detail {

// make_ep_freq_sweep_cpu is defined in lib/cpu/. There is no GPU engine yet.

class EPFreqSweepEngine {
public:
    virtual ~EPFreqSweepEngine() = default;

    virtual void execute(std::span<const float> ts_e,
                         std::span<const float> ts_v,
                         const std::filesystem::path& outdir,
                         std::string_view file_prefix) = 0;
};

std::unique_ptr<EPFreqSweepEngine> make_ep_freq_sweep_cpu(
    const search::PulsarSearchConfig& cfg,
    bool show_progress,
    float min_pd,
    std::string_view poly_basis,
    float ref_ducy,
    const PruneRFIConfig& rfi_config,
    const std::optional<std::filesystem::path>& plan_cache_file,
    std::optional<SizeType> n_runs,
    std::optional<std::vector<SizeType>> ref_segs);

} // namespace loki::algorithms::detail
