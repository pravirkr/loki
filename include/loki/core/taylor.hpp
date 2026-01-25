#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::core {

// Old method (not used anymore)
std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve(std::span<const double> pset_cur,
                   std::span<const std::vector<double>> param_arr,
                   SizeType ffa_level,
                   SizeType latter,
                   double tseg_brute,
                   SizeType nbins);

void ffa_taylor_resolve_freq_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_accel_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_jerk_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_snap_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

void ffa_taylor_resolve_crackle_batch(
    std::span<const std::vector<double>> param_arr_cur,
    std::span<const std::vector<double>> param_arr_prev,
    std::span<uint32_t> pindex_prev_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins);

SizeType poly_taylor_seed(std::span<const std::vector<double>> param_arr,
                          std::span<const double> dparams,
                          SizeType poly_order,
                          std::pair<double, double> coord_init,
                          std::span<double> seed_leaves);

std::vector<SizeType>
poly_taylor_branch_batch(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_cur,
                         std::span<double> leaves_branch_batch,
                         SizeType n_batch,
                         SizeType n_params,
                         SizeType nbins,
                         double eta,
                         const std::vector<ParamLimitType>& param_limits,
                         SizeType branch_max);

SizeType
poly_taylor_branch_accel_batch(std::span<const double> leaves_tree,
                               std::pair<double, double> coord_cur,
                               std::span<double> leaves_branch,
                               std::span<SizeType> leaves_origins,
                               SizeType n_leaves,
                               SizeType nbins,
                               double eta,
                               const std::vector<ParamLimitType>& param_limits,
                               SizeType branch_max,
                               std::span<double> scratch_params,
                               std::span<double> scratch_dparams,
                               std::span<SizeType> scratch_counts);

SizeType
poly_taylor_branch_jerk_batch(std::span<const double> leaves_tree,
                              std::pair<double, double> coord_cur,
                              std::span<double> leaves_branch,
                              std::span<SizeType> leaves_origins,
                              SizeType n_leaves,
                              SizeType nbins,
                              double eta,
                              const std::vector<ParamLimitType>& param_limits,
                              SizeType branch_max,
                              std::span<double> scratch_params,
                              std::span<double> scratch_dparams,
                              std::span<SizeType> scratch_counts);

SizeType
poly_taylor_branch_snap_batch(std::span<const double> leaves_tree,
                              std::pair<double, double> coord_cur,
                              std::span<double> leaves_branch,
                              std::span<SizeType> leaves_origins,
                              SizeType n_leaves,
                              SizeType nbins,
                              double eta,
                              const std::vector<ParamLimitType>& param_limits,
                              SizeType branch_max,
                              std::span<double> scratch_params,
                              std::span<double> scratch_dparams,
                              std::span<SizeType> scratch_counts);

void poly_taylor_resolve_accel_batch(
    std::span<const double> leaves_tree,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    SizeType nbins,
    SizeType n_leaves);

void poly_taylor_resolve_jerk_batch(
    std::span<const double> leaves_tree,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    SizeType nbins,
    SizeType n_leaves);

void poly_taylor_resolve_snap_batch(
    std::span<const double> leaves_tree,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<const std::vector<double>> param_arr,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    SizeType nbins,
    SizeType n_leaves);

void poly_taylor_transform_accel_batch(std::span<double> leaves_tree,
                                       std::pair<double, double> coord_next,
                                       std::pair<double, double> coord_cur,
                                       SizeType n_leaves,
                                       bool use_conservative_tile);

void poly_taylor_transform_jerk_batch(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      bool use_conservative_tile);

void poly_taylor_transform_snap_batch(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_next,
                                      std::pair<double, double> coord_cur,
                                      SizeType n_leaves,
                                      bool use_conservative_tile);

void report_leaves_taylor_batch(std::span<double> leaves_tree,
                                std::pair<double, double> coord_report,
                                SizeType n_leaves,
                                SizeType n_params);

std::vector<double>
poly_taylor_branch(std::span<const double> leaf,
                   std::pair<double, double> coord_cur,
                   SizeType n_params,
                   SizeType nbins,
                   double eta,
                   const std::vector<ParamLimitType>& param_limits);

// Generate an approximate branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_poly_taylor_approx(std::span<const std::vector<double>> param_arr,
                               std::span<const double> dparams,
                               const std::vector<ParamLimitType>& param_limits,
                               double tseg_ffa,
                               SizeType nsegments,
                               SizeType nbins,
                               double eta,
                               SizeType ref_seg,
                               IndexType isuggest         = 0,
                               bool use_conservative_tile = false);

// Generate an exact branching pattern for the pruning Taylor search.
std::vector<double>
generate_bp_poly_taylor(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        const std::vector<ParamLimitType>& param_limits,
                        double tseg_ffa,
                        SizeType nsegments,
                        SizeType nbins,
                        double eta,
                        SizeType ref_seg,
                        bool use_conservative_tile = false);

#ifdef LOKI_ENABLE_CUDA

std::tuple<SizeType, SizeType> poly_taylor_branch_and_validate_cuda(
    cuda::std::span<const double> leaves_tree,
    std::pair<double, double> coord_cur,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<SizeType> leaves_origins,
    SizeType n_leaves,
    SizeType n_params,
    SizeType nbins,
    double eta,
    const std::vector<ParamLimitType>& param_limits,
    SizeType branch_max,
    cuda::std::span<double> scratch_params,
    cuda::std::span<double> scratch_dparams,
    cuda::std::span<SizeType> scratch_counts,
    cudaStream_t stream);

void poly_taylor_resolve_cuda(cuda::std::span<const double> leaves_branch,
                              cuda::std::span<const float> accel_grid,
                              cuda::std::span<const float> freq_grid,
                              cuda::std::span<SizeType> param_indices,
                              cuda::std::span<float> phase_shift,
                              std::pair<double, double> coord_add,
                              std::pair<double, double> coord_cur,
                              std::pair<double, double> coord_init,
                              SizeType nbins,
                              SizeType n_leaves,
                              SizeType n_params,
                              cudaStream_t stream);

void poly_taylor_transform_cuda(cuda::std::span<double> leaves_tree,
                                std::pair<double, double> coord_next,
                                std::pair<double, double> coord_cur,
                                SizeType n_leaves,
                                SizeType n_params,
                                bool use_conservative_tile,
                                cudaStream_t stream);

#endif // LOKI_ENABLE_CUDA

} // namespace loki::core
