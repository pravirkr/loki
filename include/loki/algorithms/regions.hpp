#pragma once

#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::regions {

inline constexpr SizeType kFFAManagerWriteBatchSize = 1U << 16U;

/**
 * @brief Generates frequency regions for an efficient FFA search.
 *
 * This function divides a wide frequency search range into smaller, contiguous
 * chunks. The core principle is to maintain a nearly constant physical time
 * resolution per folding bin, ensuring consistent sensitivity to a given pulse
 * duty cycle across the search. The number of bins grows with the frequency
 * until an optional maximum is reached.
 *
 * @param p_min The minimum period of the entire search range (in seconds).
 * @param p_max The maximum period of the entire search range (in seconds).
 * @param tsamp The sampling interval of the input data (in seconds).
 * @param nbins_min The minimum number of folding bins to use for the shortest
 * period in the search range.
 * @param eta_min The minimum tolerance in bins for the shortest period in the
 * search range. Must be positive.
 * @param octave_scale multiplicative factor between successive FFA search
bands. An octave_scale = 2.0 gives true octave spacing (each band doubles in
period and bin count). Values < 2.0 create pseudo-octaves for smoother
duty-cycle resolution.
 * @param nbins_max The maximum number of bins to cap memory usage for
 * long periods in the search range. Must be >= nbins_min.
 * @return A std::vector of FFARegion structs, each defining a single FFA search
 * frequency region.
 * @throws std::invalid_argument if input parameters are illogical.
 * @throws std::runtime_error if nbins_max < nbins_min.
 */
std::vector<coord::FFARegion> generate_ffa_regions(double p_min,
                                                   double p_max,
                                                   double tsamp,
                                                   SizeType nbins_min,
                                                   double eta_min,
                                                   double octave_scale = 2.0,
                                                   SizeType nbins_max  = 1024);

/**
 * @brief A class to store the size stats for FFA regions (Time or Fourier
 * domain).
 * @details
 * This class stores the size stats for FFA regions for a given search
 * configuration.
 */
template <SupportedFoldType FoldType> class FFARegionStats {
public:
    /**
     * @brief Constructs the FFA region stats from a search configuration.
     * @param cfg The pulsar search configuration object.
     */
    FFARegionStats(SizeType max_buffer_size,
                   SizeType max_coord_size,
                   SizeType max_ncoords,
                   SizeType max_ffa_levels,
                   SizeType max_total_params_flat_count,
                   SizeType n_widths,
                   SizeType n_params,
                   SizeType n_samps,
                   SizeType max_passing_candidates,
                   bool use_gpu);

    // --- Rule of five: PIMPL ---
    ~FFARegionStats();
    FFARegionStats(FFARegionStats&&) noexcept;
    FFARegionStats& operator=(FFARegionStats&&) noexcept;
    FFARegionStats(const FFARegionStats&);
    FFARegionStats& operator=(const FFARegionStats&);

    // --- Getters ---
    /// @brief Get the maximum size of the FFA workspace buffer.
    SizeType get_max_buffer_size() const noexcept;
    /// @brief Get the maximum size of the coordinate storage.
    SizeType get_max_coord_size() const noexcept;
    /// @brief Get the maximum number of coordinates in the last level.
    SizeType get_max_ncoords() const noexcept;
    /// @brief Get the maximum number of FFA levels.
    SizeType get_max_ffa_levels() const noexcept;
    /// @brief Get the maximum total number of parameters in the flat array.
    SizeType get_max_total_params_flat_count() const noexcept;
    /// @brief Get the maximum size of the FFA workspace buffer (time domain).
    SizeType get_max_buffer_size_time() const noexcept;
    /// @brief Get the maximum size of the scores storage.
    SizeType get_max_scores_size() const noexcept;
    /// @brief Get the write parameter sets storage.
    SizeType get_write_param_sets_size() const noexcept;
    /// @brief Get the memory usage of the buffer storage (in GB).
    float get_buffer_memory_usage() const noexcept;
    /// @brief Get the memory usage of the coordinate storage (in GB).
    float get_coord_memory_usage() const noexcept;
    /// @brief Get the memory usage of the scores + param sets storage (in GB).
    float get_extra_memory_usage() const noexcept;
    /// @brief Get the memory usage of the FFA manager for this region (in GB).
    float get_manager_memory_usage() const noexcept;
    /// @brief Get the chunk stats for the planner.
    [[nodiscard]] std::vector<coord::FFAChunkStats>
    get_chunk_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief A planner for FFA regions (Time or Fourier domain).
 * @details
 * This class plans the FFA regions for a given search configuration.
 */
template <SupportedFoldType FoldType> class FFARegionPlanner {
public:
    /**
     * @brief Constructs the FFA region planner from a search configuration.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFARegionPlanner(const search::PulsarSearchConfig& cfg,
                              bool use_gpu = false);

    // --- Rule of five: PIMPL ---
    ~FFARegionPlanner();
    FFARegionPlanner(FFARegionPlanner&&) noexcept;
    FFARegionPlanner& operator=(FFARegionPlanner&&) noexcept;
    FFARegionPlanner(const FFARegionPlanner&)            = delete;
    FFARegionPlanner& operator=(const FFARegionPlanner&) = delete;

    // --- Getters ---
    /// @brief Get the search configurations for each region.
    [[nodiscard]] const std::vector<search::PulsarSearchConfig>&
    get_cfgs() const noexcept;
    /// @brief Get the number of regions.
    SizeType get_nregions() const noexcept;
    /// @brief Get the stats for the planner.
    [[nodiscard]] const FFARegionStats<FoldType>& get_stats() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::regions