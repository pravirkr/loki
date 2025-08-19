#pragma once

#include <map>
#include <span>
#include <string>
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
    std::vector<double> generate_branching_pattern() const;

private:
    search::PulsarSearchConfig m_cfg;
    void configure_plan();
    void validate_plan() const;
    static std::vector<SizeType>
    calculate_strides(std::span<const std::vector<double>> p_arr);
};

} // namespace loki::plans