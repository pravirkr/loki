#pragma once

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::plans {

// FFA Coordinate plan for a single param coordinate in a single iteration
struct FFACoord {
    uint32_t i_tail;  // Tail coordinate index in the previous iteration
    float shift_tail; // Shift in the tail coordinate
    uint32_t i_head;  // Head coordinate index in the previous iteration
    float shift_head; // Shift in the head coordinate
};

struct FFACoordFreq {
    uint32_t idx;
    float shift;
};

struct FFAPlan {
    SizeType n_params{};
    SizeType n_levels{};
    std::vector<SizeType> segment_lens;
    std::vector<SizeType> nsegments;
    std::vector<double> tsegments;
    std::vector<SizeType> ncoords;
    std::vector<float> ncoords_lb;
    std::vector<std::vector<std::vector<double>>> params;
    std::vector<std::vector<SizeType>> param_cart_strides;
    std::vector<std::vector<double>> dparams;
    std::vector<std::vector<double>> dparams_lim;
    std::vector<std::vector<SizeType>> fold_shapes;
    std::vector<std::vector<SizeType>> fold_shapes_complex;

    FFAPlan() = delete;
    explicit FFAPlan(search::PulsarSearchConfig cfg);

    SizeType get_buffer_size() const noexcept;
    SizeType get_buffer_size_complex() const noexcept;
    SizeType get_brute_fold_size() const noexcept;
    SizeType get_fold_size() const noexcept;
    SizeType get_fold_size_complex() const noexcept;
    SizeType get_coord_size() const noexcept;
    float get_buffer_memory_usage() const noexcept;
    float get_coord_memory_usage() const noexcept;
    // Get a dictionary of parameters for the last level of the plan
    std::map<std::string, std::vector<double>> get_params_dict() const;

    void resolve_coordinates(std::span<std::vector<FFACoord>> coordinates);
    std::vector<std::vector<FFACoord>> resolve_coordinates();

    // Specialized functions for frequency coordinates
    void
    resolve_coordinates_freq(std::span<std::vector<FFACoordFreq>> coordinates);
    std::vector<std::vector<FFACoordFreq>> resolve_coordinates_freq();

    // Generate a branching pattern for the pruning Taylor search.
    std::vector<double>
    get_branching_pattern(std::string_view kind = "taylor") const;

private:
    search::PulsarSearchConfig m_cfg;
    void configure_plan();
    void validate_plan() const;
    static std::vector<SizeType>
    calculate_strides(std::span<const std::vector<double>> p_arr);
};

// Generate an approximate branching pattern for the pruning search.
std::vector<double> generate_branching_pattern_approx(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams_lim,
    const std::vector<ParamLimitType>& param_limits,
    double tseg_ffa,
    SizeType nsegments,
    SizeType fold_bins,
    double tol_bins,
    bool use_conservative_errors = false,
    std::string_view kind        = "taylor");

// Generate an exact branching pattern for the pruning search.
std::vector<double>
generate_branching_pattern(std::span<const std::vector<double>> param_arr,
                           std::span<const double> dparams_lim,
                           const std::vector<ParamLimitType>& param_limits,
                           double tseg_ffa,
                           SizeType nsegments,
                           SizeType fold_bins,
                           double tol_bins,
                           bool use_conservative_errors = false,
                           std::string_view kind        = "taylor");

// A structure to hold the parameters for a single FFA search region.
struct FFARegion {
    double f_start;
    double f_end;
    SizeType nbins;
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
 * periods.
 * @param growth_factor The factor by which the number of bins grows for each
 * region. A value of 2.0 creates octave regions.
 * @param nbins_max An optional maximum number of bins to cap memory usage for
 * long periods.
 * @return A std::vector of FFARegion structs, each defining a single FFA search
 * region.
 * @throws std::invalid_argument if input parameters are illogical.
 */
std::vector<FFARegion>
generate_ffa_regions(double p_min,
                     double p_max,
                     double tsamp,
                     SizeType nbins_min,
                     double growth_factor              = 2.0,
                     std::optional<SizeType> nbins_max = std::nullopt);

} // namespace loki::plans