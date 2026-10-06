#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

/**
 * @brief Frequency sweep pipeline for Extreme Pruning (EP) search.
 *
 * Divides the requested frequency range into optimal chunks using
 * EPRegionPlanner (which runs DynamicThresholdScheme simulation once per coarse
 * band and bisects chunk widths to obey max_process_memory_gb).
 * Executes EPMultiPass on each chunk and merges all chunk candidates into a
 * single unified HDF5 result file: {file_prefix}_ep_results.h5.
 */
class EPFreqSweep {
public:
    /// The CPU thread count comes from @p cfg; @p exec selects the backend.
    /// Only the CPU backend is implemented.
    explicit EPFreqSweep(
        const search::PulsarSearchConfig& cfg,
        bool show_progress          = true,
        float min_pd                = 0.1F,
        std::string_view poly_basis = "taylor",
        float ref_ducy              = 0.1F,
        PruneRFIConfig rfi_config   = {},
        const std::optional<std::filesystem::path>& plan_cache_file =
            std::nullopt,
        std::optional<SizeType> n_runs                = std::nullopt,
        std::optional<std::vector<SizeType>> ref_segs = std::nullopt,
        Exec exec                                     = {});

    ~EPFreqSweep();
    EPFreqSweep(EPFreqSweep&&) noexcept;
    EPFreqSweep& operator=(EPFreqSweep&&) noexcept;
    EPFreqSweep(const EPFreqSweep&)            = delete;
    EPFreqSweep& operator=(const EPFreqSweep&) = delete;

    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 const std::filesystem::path& outdir = "./",
                 std::string_view file_prefix        = "test");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms
