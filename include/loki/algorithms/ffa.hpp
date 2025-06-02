#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

namespace loki::algorithms {

// FFA Coordinate plan for a single param coordinate in a single iteration
struct FFACoord {
    SizeType i_tail;     // Tail coordinate index in the previous iteration
    SizeType shift_tail; // Shift in the tail coordinate
    SizeType i_head;     // Head coordinate index in the previous iteration
    SizeType shift_head; // Shift in the head coordinate
};

struct FFAPlan {
    std::vector<SizeType> segment_lens;
    std::vector<double> tsegments;
    std::vector<std::vector<std::vector<double>>> params;
    std::vector<std::vector<double>> dparams;
    std::vector<std::vector<SizeType>> fold_shapes;
    std::vector<std::vector<FFACoord>> coordinates;

    FFAPlan() = delete;
    explicit FFAPlan(search::PulsarSearchConfig cfg);

    SizeType get_memory_usage() const noexcept;
    SizeType get_buffer_size() const noexcept;
    SizeType get_fold_size() const noexcept;

private:
    search::PulsarSearchConfig m_cfg;
    void configure_plan();
    static std::vector<SizeType>
    calculate_strides(std::span<const std::vector<double>> p_arr);
};

class FFA {
public:
    explicit FFA(search::PulsarSearchConfig cfg);
    FFA(const FFA&)            = delete;
    FFA& operator=(const FFA&) = delete;
    FFA(FFA&&)                 = delete;
    FFA& operator=(FFA&&)      = delete;
    ~FFA()                     = default;

    const FFAPlan& get_plan() const;
    void execute(std::span<const float> ts_e,
                 std::span<const float> ts_v,
                 std::span<float> fold);

private:
    search::PulsarSearchConfig m_cfg;
    FFAPlan m_ffa_plan;
    int m_nthreads;

    // Buffers for the FFA plan
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;

    void initialize(std::span<const float> ts_e, std::span<const float> ts_v);
    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_level);
};

// Convenience function to fold time series using FFA method
std::vector<float> compute_ffa(std::span<const float> ts_e,
                               std::span<const float> ts_v,
                               search::PulsarSearchConfig cfg);

} // namespace loki::algorithms