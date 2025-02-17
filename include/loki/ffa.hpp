#pragma once

#include <cstddef>
#include <span>

#include <loki/configs.hpp>
#include <loki/loki_types.hpp>

// FFA Coordinate plan for a single param for a single iteration
struct FFACoord {
    SizeType i_coord_tail; // Tail coordinate index in the previous iteration
    SizeType shift_tail;   // Shift in the tail coordinate
    SizeType i_coord_head; // Head coordinate index in the previous iteration
    SizeType shift_head;   // Shift in the head coordinate
};

struct FFAPlan {
    std::vector<SizeType> segment_lens;
    std::vector<float> tsegments;
    std::vector<std::vector<std::vector<float>>> params;
    std::vector<std::vector<float>> dparams;
    std::vector<std::vector<SizeType>> fold_shapes;
    std::vector<std::vector<FFACoord>> coordinates;

    FFAPlan() = delete;
    explicit FFAPlan(PulsarSearchConfig cfg);

    SizeType get_memory_usage() const noexcept;
    SizeType get_buffer_size() const noexcept;

private:
    PulsarSearchConfig m_cfg;
    void configure_plan();
    static std::vector<SizeType>
    calculate_strides(std::span<const std::vector<float>> p_arr);
};

class FFA {
public:
    explicit FFA(PulsarSearchConfig cfg);
    FFA(const FFA&)            = delete;
    FFA& operator=(const FFA&) = delete;
    FFA(FFA&&)                 = delete;
    FFA& operator=(FFA&&)      = delete;
    ~FFA();

    const FFAPlan& get_plan() const;
    void execute(std::span<const float> ts, std::span<float> fold);
    void initialize(std::span<const float> ts, std::span<float> fold);

private:
    PulsarSearchConfig m_cfg;
    FFAPlan m_ffa_plan;

    // Buffers for the FFA plan
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;

    void execute_iter(const float* __restrict__ fold_in,
                      float* __restrict__ fold_out,
                      SizeType i_iter);
};