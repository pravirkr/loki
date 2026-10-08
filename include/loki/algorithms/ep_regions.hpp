#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

/**
 * @brief Configuration and threshold scheme for a single EP chunk.
 */
struct EPChunkConfig {
    search::PulsarSearchConfig cfg;
    std::vector<float> threshold_scheme;
    std::vector<float> branching_pattern;
    SizeType max_sugg{1U << 18U};
    /// Branching capacity of this chunk's own plan (not the region's).
    SizeType branch_max{32U};
    double nominal_f_start{0.0};
    double nominal_f_end{0.0};
    double actual_f_start{0.0};
    double actual_f_end{0.0};
    double peak_complexity{1.0};
    double chunk_memory_gb{0.0};
    SizeType nsegments{0};
    SizeType ncoords{0};
    SizeType buffer_size{0};
    SizeType coord_size{0};
    SizeType fold_size{0};
};

/**
 * @brief Statistics for a single planned EP chunk.
 */
struct EPChunkStats {
    SizeType chunk_id{0};
    double nominal_f_start{0.0};
    double nominal_f_end{0.0};
    double actual_f_start{0.0};
    double actual_f_end{0.0};
    double nominal_width{0.0};
    double actual_width{0.0};
    SizeType nbins{0};
    double eta{0.0};
    SizeType ncoords{0};
    SizeType max_sugg{0};
    SizeType branch_max{0};
    double peak_complexity{0.0};
    double memory_gb{0.0};
    double overlap_fraction{0.0};
};

/**
 * @brief Aggregate statistics for EP regions planned across a frequency sweep.
 */
class EPRegionStats {
public:
    EPRegionStats() = default;
    EPRegionStats(SizeType max_sugg,
                  SizeType max_ncoords,
                  SizeType max_branch_max,
                  float max_memory_gb,
                  std::vector<EPChunkStats> chunk_stats)
        : m_max_sugg(max_sugg),
          m_max_ncoords(max_ncoords),
          m_max_branch_max(max_branch_max),
          m_max_memory_gb(max_memory_gb),
          m_chunk_stats(std::move(chunk_stats)) {}

    EPRegionStats(SizeType max_sugg,
                  SizeType max_ncoords,
                  SizeType max_branch_max,
                  float max_memory_gb,
                  SizeType max_buffer_size,
                  SizeType max_coord_size,
                  SizeType max_fold_size,
                  std::vector<EPChunkStats> chunk_stats)
        : m_max_sugg(max_sugg),
          m_max_ncoords(max_ncoords),
          m_max_branch_max(max_branch_max),
          m_max_memory_gb(max_memory_gb),
          m_max_buffer_size(max_buffer_size),
          m_max_coord_size(max_coord_size),
          m_max_fold_size(max_fold_size),
          m_chunk_stats(std::move(chunk_stats)) {}

    [[nodiscard]] SizeType get_max_sugg() const noexcept { return m_max_sugg; }
    [[nodiscard]] SizeType get_max_ncoords() const noexcept {
        return m_max_ncoords;
    }
    [[nodiscard]] SizeType get_max_branch_max() const noexcept {
        return m_max_branch_max;
    }
    [[nodiscard]] float get_max_memory_gb() const noexcept {
        return m_max_memory_gb;
    }
    [[nodiscard]] SizeType get_max_buffer_size() const noexcept {
        return m_max_buffer_size;
    }
    [[nodiscard]] SizeType get_max_coord_size() const noexcept {
        return m_max_coord_size;
    }
    [[nodiscard]] SizeType get_max_fold_size() const noexcept {
        return m_max_fold_size;
    }
    [[nodiscard]] SizeType get_nchunks() const noexcept {
        return m_chunk_stats.size();
    }
    [[nodiscard]] const std::vector<EPChunkStats>&
    get_chunk_stats() const noexcept {
        return m_chunk_stats;
    }

private:
    SizeType m_max_sugg{0};
    SizeType m_max_ncoords{0};
    SizeType m_max_branch_max{0};
    float m_max_memory_gb{0.0F};
    SizeType m_max_buffer_size{0};
    SizeType m_max_coord_size{0};
    SizeType m_max_fold_size{0};
    std::vector<EPChunkStats> m_chunk_stats;
};

/**
 * @brief A planner for EP (Extreme Pruning) regions across a frequency sweep.
 *
 * Subdivides a frequency search range into optimal memory-bounded chunks.
 * Uses generate_ffa_regions() for coarse bands (nbins, eta), runs
 * DynamicThresholdScheme simulation once per coarse band, and bisects chunk
 * widths analytically to respect max_process_memory_gb.
 * The threshold scheme and max_sugg are designed per coarse band; branch_max
 * is derived per chunk from the chunk's own plan.
 *
 * Supports saving and loading planned chunk configurations to/from HDF5 cache
 * files with strict validation of all search parameters.
 *
 * @tparam FoldType float for time domain, ComplexType for Fourier domain.
 */
template <SupportedFoldType FoldType> class EPRegionPlanner {
public:
    explicit EPRegionPlanner(const search::PulsarSearchConfig& cfg,
                             float min_pd                = 0.1F,
                             std::string_view poly_basis = "taylor",
                             float ref_ducy              = 0.1F,
                             const std::optional<std::filesystem::path>&
                                 plan_cache_file = std::nullopt);

    ~EPRegionPlanner();
    EPRegionPlanner(EPRegionPlanner&&) noexcept;
    EPRegionPlanner& operator=(EPRegionPlanner&&) noexcept;
    EPRegionPlanner(const EPRegionPlanner&)            = delete;
    EPRegionPlanner& operator=(const EPRegionPlanner&) = delete;

    [[nodiscard]] const std::vector<EPChunkConfig>&
    get_chunk_cfgs() const noexcept;
    [[nodiscard]] SizeType get_nchunks() const noexcept;
    [[nodiscard]] const EPRegionStats& get_stats() const noexcept;

    void save_cache(const std::filesystem::path& filepath) const;
    void load_cache(const std::filesystem::path& filepath);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::algorithms
