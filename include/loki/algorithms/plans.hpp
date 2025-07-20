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
    SizeType i_tail;   // Tail coordinate index in the previous iteration
    double shift_tail; // Shift in the tail coordinate
    SizeType i_head;   // Head coordinate index in the previous iteration
    double shift_head; // Shift in the head coordinate
};

struct FFAPlan {
    std::vector<SizeType> segment_lens;
    std::vector<SizeType> nsegments;
    std::vector<double> tsegments;
    std::vector<SizeType> ncoords;
    std::vector<float> ncoords_lb;
    std::vector<std::vector<std::vector<double>>> params;
    std::vector<std::vector<double>> dparams;
    std::vector<std::vector<SizeType>> fold_shapes;
    std::vector<std::vector<SizeType>> fold_shapes_complex;
    std::vector<std::vector<FFACoord>> coordinates;

    FFAPlan() = delete;
    explicit FFAPlan(search::PulsarSearchConfig cfg);

    SizeType get_buffer_size() const noexcept;
    SizeType get_buffer_size_complex() const noexcept;
    SizeType get_brute_fold_size() const noexcept;
    SizeType get_fold_size() const noexcept;
    SizeType get_fold_size_complex() const noexcept;
    SizeType get_memory_usage() const noexcept;
    // Get a dictionary of parameters for the last level of the plan
    std::map<std::string, std::vector<double>> get_params_dict() const;

private:
    search::PulsarSearchConfig m_cfg;
    void configure_plan();
    void validate_plan() const;
    static std::vector<SizeType>
    calculate_strides(std::span<const std::vector<double>> p_arr);
};

} // namespace loki::plans