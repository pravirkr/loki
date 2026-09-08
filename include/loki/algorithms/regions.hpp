#pragma once

#include <memory>
#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::regions {

inline constexpr SizeType kFFAFreqSweepWriteBatchSize = 1U << 16U;

/**
 * @brief Generates frequency regions for an efficient FFA search.
 *
 * Divides a period range into contiguous bands that keep a nearly constant
 * physical time resolution per folding bin. Bin count grows with period until
 * `nbins_max`. Requested `nbins_min` must fit in `p_min / tsamp` or the
 * function throws.
 *
 * @param p_min Minimum period (seconds). Must exceed 2*tsamp (Nyquist) and
 * satisfy p_min >= nbins_min * tsamp.
 * @param p_max Maximum period (seconds). Must be > p_min.
 * @param tsamp Sampling interval (seconds).
 * @param nbins_min Folding bins at the shortest period. Must be >= 2.
 * @param eta_min Tolerance in bins at the shortest period. Must be positive.
 * @param octave_scale multiplicative factor between successive FFA search
 * bands. An octave_scale = 2.0 gives true octave spacing (each band doubles in
 * period and bin count). Values in (1.0, 2.0) create pseudo-octaves for
 * smoother duty-cycle resolution. Must be strictly > 1.0.
 * @param nbins_max Cap on folding bins for long periods. Must be >= nbins_min.
 * @return Contiguous FFARegion bands covering [1/p_max, 1/p_min] Hz.
 * @throws std::runtime_error if the inputs are illogical.
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
class FFARegionStats {
public:
    /**
     * @brief Constructs the FFA region stats from a search configuration.
     * @param cfg The pulsar search configuration object.
     */
    FFARegionStats(SizeType max_buffer_size,
                   SizeType max_coord_size,
                   SizeType max_ncoords,
                   SizeType max_ffa_levels,
                   SizeType max_scores_scratch,
                   SizeType n_params,
                   SizeType n_samps,
                   SizeType max_passing_candidates,
                   bool use_fourier,
                   bool use_gpu);

    ~FFARegionStats()                                    = default;
    FFARegionStats(FFARegionStats&&) noexcept            = default;
    FFARegionStats& operator=(FFARegionStats&&) noexcept = default;
    FFARegionStats(const FFARegionStats&)                = default;
    FFARegionStats& operator=(const FFARegionStats&)     = default;

    // --- Getters ---
    /// @brief Get the maximum size of the FFA workspace buffer.
    SizeType get_max_buffer_size() const noexcept { return m_max_buffer_size; }
    /// @brief Get the maximum size of the coordinate storage.
    SizeType get_max_coord_size() const noexcept { return m_max_coord_size; }
    /// @brief Get the maximum number of coordinates in the last level.
    SizeType get_max_ncoords() const noexcept { return m_max_ncoords; }
    /// @brief Get the maximum number of FFA levels.
    SizeType get_max_ffa_levels() const noexcept { return m_max_ffa_levels; }
    /// @brief Get the maximum size of the FFA workspace buffer (time domain).
    SizeType get_max_buffer_size_time() const noexcept;
    /// @brief Get the per-chunk raw score scratch size.
    /// @details max over chunks of ncoords * n_widths(nbins). The boxcar
    /// width count is derived from nbins, which varies between regions, so
    /// this is not simply max_ncoords times the base width count.
    SizeType get_max_scores_scratch_size() const noexcept;
    /// @brief Get the capacity of the surviving-candidate accumulator.
    SizeType get_max_candidates() const noexcept;
    /// @brief Get the write parameter sets storage.
    SizeType get_write_param_sets_size() const noexcept;
    /// @brief Get the memory usage of the buffer storage (in GB).
    float get_buffer_memory_usage() const noexcept;
    /// @brief Get the memory usage of the coordinate storage (in GB).
    float get_coord_memory_usage() const noexcept;
    /// @brief Get the host memory usage of the score scratch, candidate
    /// accumulator and write staging (in GB).
    float get_extra_memory_usage() const noexcept;
    /// @brief Get the device memory usage of the timeseries and per-chunk
    /// score scratch (in GB). Excludes the host-side candidate accumulator.
    float get_device_extra_memory_usage() const noexcept;
    float get_cpu_memory_usage() const noexcept;
    float get_device_memory_usage() const noexcept;
    /// @brief Get the memory usage of the FFA freq sweep for this region (in
    /// GB).
    float get_freq_sweep_memory_usage() const noexcept;
    /// @brief Get the chunk stats for the planner.
    [[nodiscard]] std::vector<coord::FFAChunkStats>
    get_chunk_stats() const noexcept;

private:
    SizeType m_max_buffer_size;
    SizeType m_max_coord_size;
    SizeType m_max_ncoords; // maximum number of coordinates in the last level
    SizeType m_max_ffa_levels;
    SizeType m_max_scores_scratch; // max over chunks of ncoords * n_widths
    SizeType m_n_params;
    SizeType m_n_samps; // ts_e.size()
    SizeType m_max_passing_candidates;
    bool m_use_fourier;
    bool m_use_gpu;
};

/**
 * @brief A planner for FFA regions (Time or Fourier domain).
 * @details
 * This class plans the FFA regions for a given search configuration. Each
 * frequency searchregion produced by generate_ffa_regions() is
 * subdivided into memory-bounded chunks by bisecting the region's frequency
 * width; nbins/eta and every other search parameter (e.g. acceleration,
 * jerk) are held fixed within a region and are never altered by this
 * subdivision.
 *
 * The planner also accounts for Doppler/drift expansion of each chunk's
 * frequency window when nparams > 1 (derived from the acceleration/jerk
 * entries of param_limits and tobs).
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
    [[nodiscard]] const FFARegionStats& get_stats() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace loki::regions