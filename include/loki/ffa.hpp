#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include <loki/fold.hpp>
#include <loki/loki_types.hpp>

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

    std::vector<float> get_ffa_step(float tsegment_cur) const;
    SizeType get_segment_len_init_default() const;
    SizeType get_segment_len_final_default() const;

private:
    void validate_config() const;
};

struct FFAPlan {
    std::vector<SizeType> segment_lens;
    std::vector<float> tsegments;
    std::vector<std::vector<std::vector<float>>> params;
    std::vector<std::vector<float>> dparams;
    std::vector<std::vector<SizeType>> fold_shapes;
    SizeType buffer_size;
};

class FFADPFunctions {
public:
    explicit FFADPFunctions(SearchConfig cfg, FFAPlan& ffa_plan);
    FFADPFunctions(const FFADPFunctions&)            = delete;
    FFADPFunctions& operator=(const FFADPFunctions&) = delete;
    FFADPFunctions(FFADPFunctions&&)                 = delete;
    FFADPFunctions& operator=(FFADPFunctions&&)      = delete;
    ~FFADPFunctions();

    void init(std::span<const float> ts, std::span<float> fold);
    void resolve(std::span<const float> pset_cur,
                 const std::vector<std::vector<float>>& parr_prev,
                 int ffa_level,
                 int latter);
    void add(std::span<const float> ts_a,
             std::span<const float> ts_b,
             std::span<float> ts_c);
    void shift(std::span<const float> pset_cur, float t_ref_prev);
    void pack(std::span<float> fold_out) const;

private:
    SearchConfig m_cfg;
    FFAPlan& m_ffa_plan;
    std::unique_ptr<BruteFold> m_brute_fold;
};

class FFA {
public:
    explicit FFA(SearchConfig cfg,
                 std::optional<SizeType> segment_len_init  = std::nullopt,
                 std::optional<SizeType> segment_len_final = std::nullopt);
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
    void validate_params() const;
    SizeType calculate_niters() const;
};