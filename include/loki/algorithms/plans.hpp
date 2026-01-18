#pragma once

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::plans {

constexpr SizeType kFFAManagerWriteBatchSize = 1U << 16U;

// FFA Coordinate plan for a single param coordinate in a single iteration
struct FFACoord {
    uint32_t i_tail;  // Tail coordinate index in the previous iteration
    float shift_tail; // Phase bin shift in the tail coordinate
    uint32_t i_head;  // Head coordinate index in the previous iteration
    float shift_head; // Phase bin shift in the head coordinate
};

struct FFACoordFreq {
    uint32_t idx; // Phase bin index in the previous iteration
    float shift;  // Phase bin shift
};

// A structure to hold the parameters for a single FFA search region.
struct FFARegion {
    double f_start; // Hz, inclusive (lower frequency)
    double f_end;   // Hz, inclusive (upper frequency)
    SizeType nbins; // fixed within region
    double eta;     // tolerance in bins for this region
};

// A structure to hold the stats for a single FFA search chunk.
struct FFAChunkStats {
    double nominal_f_start;
    double nominal_f_end;
    double actual_f_start;
    double actual_f_end;
    double nominal_width;
    double actual_width;
    double total_memory_gb;
    double overlap_fraction; // fraction of actual range that's overlap
};

/**
 * @brief Base class for an FFA search plan.
 * @details
 * This class holds all type-invariant (non-template) data and logic
 * for an FFA plan. This includes parameter grids, coordinate mappings,
 * and segment lengths.
 */
class FFAPlanBase {
public:
    /**
     * @brief Constructs the FFA plan from a search configuration.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFAPlanBase(const search::PulsarSearchConfig& cfg);

    // --- Rule of five: PIMPL ---
    virtual ~FFAPlanBase();
    FFAPlanBase(FFAPlanBase&&) noexcept;
    FFAPlanBase& operator=(FFAPlanBase&&) noexcept;
    FFAPlanBase(const FFAPlanBase&);
    FFAPlanBase& operator=(const FFAPlanBase&);

    // --- Getters ---
    /// @brief Number of parameters to search over (..., a, f).
    SizeType get_n_params() const noexcept;
    /// @brief Number of FFA merge levels.
    SizeType get_n_levels() const noexcept;
    /// @brief Segment length for each level.
    [[nodiscard]] const std::vector<SizeType>&
    get_segment_lens() const noexcept;
    /// @brief Number of segments for each level.
    [[nodiscard]] const std::vector<SizeType>& get_nsegments() const noexcept;
    /// @brief Segment lengths in seconds.
    [[nodiscard]] const std::vector<double>& get_tsegments() const noexcept;
    /// @brief Number of coordinates for each level.
    [[nodiscard]] const std::vector<SizeType>& get_ncoords() const noexcept;
    /// @brief Log2 of number of coordinates for each level.
    [[nodiscard]] const std::vector<float>& get_ncoords_lb() const noexcept;
    /// @brief Offset number of coordinates for each level (cumulative sum).
    [[nodiscard]] const std::vector<SizeType>&
    get_ncoords_offsets() const noexcept;
    /// @brief Parameter grid for each level.
    [[nodiscard]] const std::vector<std::vector<std::vector<double>>>&
    get_params() const noexcept;
    /// @brief Cartesian strides for each parameter in the parameter grid.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_param_cart_strides() const noexcept;
    /// @brief Grid step sizes for each parameter.
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_dparams() const noexcept;
    /// @brief Grid step size (limited) for each parameter.
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_dparams_lim() const noexcept;
    /// @brief Get the search configuration object used to build the plan.
    [[nodiscard]] const search::PulsarSearchConfig& get_config() const noexcept;

    // --- Getters ---
    /// @brief Get the total number of coordinates across all levels.
    SizeType get_coord_size() const noexcept;
    /// @brief Get the memory usage of the coordinate storage (in GB).
    float get_coord_memory_usage() const noexcept;
    /// @brief Get a dictionary of parameters for the last level of the plan.
    [[nodiscard]] std::map<std::string, std::vector<double>>
    get_params_dict() const;

    // --- Methods ---
    /**
     * @brief Resolve the coordinates for the plan.
     * @param coords A span of FFACoord objects.
     */
    void resolve_coordinates(std::span<FFACoord> coords);

    /**
     * @brief Resolve the coordinates for the plan.
     * @return A vector of vectors of FFACoord for each level.
     */
    std::vector<std::vector<FFACoord>> resolve_coordinates();

    /**
     * @brief Resolve the coordinates for the plan (frequency-only coordinates).
     * @param coords_freq A span of FFACoordFreq objects.
     */
    void resolve_coordinates_freq(std::span<FFACoordFreq> coords_freq);

    /**
     * @brief Resolve the coordinates for the plan (frequency-only coordinates).
     * @return A vector of vectors of FFACoordFreq for each level.
     */
    std::vector<std::vector<FFACoordFreq>> resolve_coordinates_freq();

    /**
     * @brief Get the approximate branching pattern for the plan.
     * @param poly_basis The polynomial basis for the branching pattern (e.g.
     * "taylor").
     * @param ref_seg The reference segment for the branching pattern.
     * @param isuggest The index of the leaf to use for the branching pattern.
     * @return A vector of branching pattern values.
     */
    std::vector<double>
    get_branching_pattern_approx(std::string_view poly_basis = "taylor",
                                 SizeType ref_seg            = 0,
                                 IndexType isuggest          = 0) const;

    /**
     * @brief Get the exact branching pattern for the plan.
     * @param poly_basis The polynomial basis for the branching pattern (e.g.
     * "taylor").
     * @param ref_seg The reference segment for the branching pattern.
     * @return A vector of branching pattern values.
     */
    std::vector<double>
    get_branching_pattern(std::string_view poly_basis = "taylor",
                          SizeType ref_seg            = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief A type-aware FFA plan (Time or Fourier domain).
 * @details
 * This class inherits all common plan logic from FFAPlanBase and adds
 * type-specific data.
 *
 * @tparam FoldType float or ComplexType.
 */
template <SupportedFoldType FoldType> class FFAPlan final : public FFAPlanBase {
public:
    /**
     * @brief Constructs the full, type-aware plan.
     * @param cfg The pulsar search configuration object.
     */
    explicit FFAPlan(const search::PulsarSearchConfig& cfg);

    // --- Rule of five: PIMPL ---
    ~FFAPlan() override;
    FFAPlan(FFAPlan&&) noexcept;
    FFAPlan& operator=(FFAPlan&&) noexcept;
    FFAPlan(const FFAPlan&);
    FFAPlan& operator=(const FFAPlan&);

    // --- Getters ---
    /// @brief Get the fold shapes for each level.
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_fold_shapes() const noexcept;
    /// @brief Get the fold shapes for each level (time domain).
    [[nodiscard]] const std::vector<std::vector<SizeType>>&
    get_fold_shapes_time() const noexcept;
    /// @brief Get the size of the brute fold buffer.
    SizeType get_brute_fold_size() const noexcept;
    /// @brief Get the size of the fold buffer.
    SizeType get_fold_size() const noexcept;
    /// @brief Get the size of the fold buffer (time domain).
    SizeType get_fold_size_time() const noexcept;
    /// @brief Get the size of the FFA workspace buffer.
    SizeType get_buffer_size() const noexcept;
    /// @brief Get the size of the FFA workspace buffer (time domain).
    SizeType get_buffer_size_time() const noexcept;
    /// @brief Get the memory usage of the FFA workspace buffer (in GB).
    float get_buffer_memory_usage() const noexcept;
    /// @brief Get the compute FLOPS for the FFA plan (in GFLOPS).
    float get_gflops(bool return_in_time) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

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
std::vector<FFARegion> generate_ffa_regions(double p_min,
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
                   SizeType n_widths,
                   SizeType n_params,
                   SizeType n_samps,
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
    /// @brief Get the maximum size of the FFA workspace buffer (time domain).
    SizeType get_max_buffer_size_time() const noexcept;
    /// @brief Get the maximum size of the scores storage.
    SizeType get_max_scores_size() const noexcept;
    /// @brief Get the write parameter sets storage.
    SizeType get_write_param_sets_size() const noexcept;
    /// @brief Get the write scores storage.
    SizeType get_write_scores_size() const noexcept;
    /// @brief Get the memory usage of the buffer storage (in GB).
    float get_buffer_memory_usage() const noexcept;
    /// @brief Get the memory usage of the coordinate storage (in GB).
    float get_coord_memory_usage() const noexcept;
    /// @brief Get the memory usage of the scores + param sets storage (in GB).
    float get_extra_memory_usage() const noexcept;
    /// @brief Get the memory usage of the FFA manager for this region (in GB).
    float get_manager_memory_usage() const noexcept;
    /// @brief Get the chunk stats for the planner.
    [[nodiscard]] std::vector<FFAChunkStats> get_chunk_stats() const noexcept;

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

} // namespace loki::plans