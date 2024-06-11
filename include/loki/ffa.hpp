#pragma once

#include <array>
#include <cstddef>
#include <span>

using SizeType       = std::size_t;
using ParamLimitType = std::array<float, 2>;

struct SearchConfig {
    // Number of samples in time series
    SizeType nsamps;
    // Time resolution of time series
    float tsamp;
    // Number of phase bins in the folded profile
    SizeType nbins;
    // Tolerance (in bins) for the FFA search
    float tolerance;
    // Parameter limits for the FFA search
    std::vector<ParamLimitType> param_limits;

    SizeType nparams{};
    float tobs{};
    float f_max{};
    float f_min{};

    SearchConfig(SizeType nsamps,
                 float tsamp,
                 SizeType nbins,
                 float tolerance,
                 const std::vector<ParamLimitType>& param_limits);

    std::vector<float> ffa_step(float tsegment_cur) const;

private:
    void validate_config() const;
};

struct FFAPlan {
    std::vector<float> tsegments;
    std::vector<std::vector<std::vector<float>>> params;
    std::vector<std::vector<float>> dparams;
    std::vector<std::vector<SizeType>> fold_shapes;
    SizeType buffer_size;
};

class FFA {
public:
    explicit FFA(SearchConfig cfg,
                 int segment_len_init  = -1,
                 int segment_len_final = -1);
    FFA(const FFA&)            = delete;
    FFA& operator=(const FFA&) = delete;
    FFA(FFA&&)                 = delete;
    FFA& operator=(FFA&&)      = delete;
    ~FFA();

    const FFAPlan& get_plan() const;
    void execute(std::span<const float> ts, std::span<float> fold);
    void initialize(std::span<const float> ts, std::span<float> fold);

private:
    SearchConfig m_cfg;
    SizeType m_segment_len_init;
    SizeType m_segment_len_final;

    SizeType m_niters;
    FFAPlan m_ffa_plan;

    // Buffers for the FFA plan
    std::vector<float> m_fold_in;
    std::vector<float> m_fold_out;

    void execute_iter(std::span<const float> fold_in,
                      std::span<float> fold_out);
    void configure_plan();
    SizeType check_segment_len_init(int segment_len_init) const;
    SizeType check_segment_len_final(int segment_len_final) const;
    SizeType calculate_niters() const;
};