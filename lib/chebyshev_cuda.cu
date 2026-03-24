#include "loki/core/chebyshev.hpp"

#include <cstdint>
#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/kernel_utils.cuh"
#include "loki/utils.hpp"

namespace loki::core {

namespace {

__device__ __forceinline__ void
cheby_to_taylor_accel(double* __restrict__ leaves_tree,
                      uint32_t n_leaves,
                      cuda::std::pair<double, double> coord_report) {
    constexpr SizeType kLeavesStride = 8;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double ts      = coord_report.second;
    const double inv_ts  = 1.0 / ts;
    const double inv_ts2 = inv_ts * inv_ts;
    const double w2_2    = 4.0 * inv_ts2;
    const double w1_1    = inv_ts;

    const uint32_t lo      = tid * kLeavesStride;
    const auto alpha_2_val = leaves_tree[lo + 0];
    const auto alpha_2_err = leaves_tree[lo + 1];
    const auto alpha_1_val = leaves_tree[lo + 2];
    const auto alpha_1_err = leaves_tree[lo + 3];
    const auto alpha_0_val = leaves_tree[lo + 4];
    const auto alpha_0_err = leaves_tree[lo + 5];

    // Write Values
    leaves_tree[lo + 0] = alpha_2_val * w2_2;
    leaves_tree[lo + 2] = alpha_1_val * w1_1;
    leaves_tree[lo + 4] = alpha_0_val - alpha_2_val;

    // Write Errors
    leaves_tree[lo + 1] = alpha_2_err * w2_2;
    leaves_tree[lo + 3] = alpha_1_err * w1_1;
    leaves_tree[lo + 5] =
        sqrt((alpha_0_err * alpha_0_err) + (alpha_2_err * alpha_2_err));
}

__device__ __forceinline__ void
taylor_to_cheby_accel(double* __restrict__ leaves_tree,
                      uint32_t n_leaves,
                      cuda::std::pair<double, double> coord_init) {
    constexpr SizeType kLeavesStride = 8;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double ts   = coord_init.second;
    const double ts2  = ts * ts;
    const double w2_2 = (ts2 / 2.0) * 0.5;

    const uint32_t lo = tid * kLeavesStride;
    const auto d2_val = leaves_tree[lo + 0];
    const auto d2_err = leaves_tree[lo + 1];
    const auto d1_val = leaves_tree[lo + 2];
    const auto d1_err = leaves_tree[lo + 3];
    const auto d0_val = leaves_tree[lo + 4];
    const auto d0_err = leaves_tree[lo + 5];

    // Write Values
    leaves_tree[lo + 0] = d2_val * w2_2;
    leaves_tree[lo + 2] = d1_val * ts;
    leaves_tree[lo + 4] = d0_val + (d2_val * w2_2);

    const double e2    = d2_err * w2_2;
    const double e1_ts = d1_err * ts;

    // Write Errors
    leaves_tree[lo + 1] = e2;
    leaves_tree[lo + 3] = e1_ts;
    leaves_tree[lo + 5] = sqrt((d0_err * d0_err) + (e2 * e2));
}

__device__ __forceinline__ void
cheby_to_taylor_jerk(double* __restrict__ leaves_tree,
                     uint32_t n_leaves,
                     cuda::std::pair<double, double> coord_report) {
    constexpr SizeType kLeavesStride = 10;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double ts      = coord_report.second;
    const double inv_ts  = 1.0 / ts;
    const double inv_ts2 = inv_ts * inv_ts;
    const double inv_ts3 = inv_ts2 * inv_ts;
    const double w3_3    = 24.0 * inv_ts3;
    const double w2_2    = 4.0 * inv_ts2;
    const double w1_1    = inv_ts;
    const double w3_1    = 3.0 * inv_ts;

    const uint32_t lo      = tid * kLeavesStride;
    const auto alpha_3_val = leaves_tree[lo + 0];
    const auto alpha_3_err = leaves_tree[lo + 1];
    const auto alpha_2_val = leaves_tree[lo + 2];
    const auto alpha_2_err = leaves_tree[lo + 3];
    const auto alpha_1_val = leaves_tree[lo + 4];
    const auto alpha_1_err = leaves_tree[lo + 5];
    const auto alpha_0_val = leaves_tree[lo + 6];
    const auto alpha_0_err = leaves_tree[lo + 7];

    // Write Values
    leaves_tree[lo + 0] = alpha_3_val * w3_3;
    leaves_tree[lo + 2] = alpha_2_val * w2_2;
    leaves_tree[lo + 4] = (alpha_1_val * w1_1) - (alpha_3_val * w3_1);
    leaves_tree[lo + 6] = alpha_0_val - alpha_2_val;

    // Pre-scale errors for variances
    const double e3_3 = alpha_3_err * w3_3;
    const double e2_2 = alpha_2_err * w2_2;
    const double e1_1 = alpha_1_err * w1_1;
    const double e3_1 = alpha_3_err * w3_1;

    // Write Errors
    leaves_tree[lo + 1] = e3_3;
    leaves_tree[lo + 3] = e2_2;
    leaves_tree[lo + 5] = sqrt((e1_1 * e1_1) + (e3_1 * e3_1));
    leaves_tree[lo + 7] =
        sqrt((alpha_0_err * alpha_0_err) + (alpha_2_err * alpha_2_err));
}

__device__ __forceinline__ void
taylor_to_cheby_jerk(double* __restrict__ leaves_tree,
                     uint32_t n_leaves,
                     cuda::std::pair<double, double> coord_init) {
    constexpr SizeType kLeavesStride = 10;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double ts   = coord_init.second;
    const double ts2  = ts * ts;
    const double ts3  = ts2 * ts;
    const double w3_3 = (ts3 / 6.0) * 0.25;
    const double w2_2 = (ts2 / 2.0) * 0.5;
    const double w3_1 = (ts3 / 6.0) * 0.75;

    const uint32_t lo = tid * kLeavesStride;
    const auto d3_val = leaves_tree[lo + 0];
    const auto d3_err = leaves_tree[lo + 1];
    const auto d2_val = leaves_tree[lo + 2];
    const auto d2_err = leaves_tree[lo + 3];
    const auto d1_val = leaves_tree[lo + 4];
    const auto d1_err = leaves_tree[lo + 5];
    const auto d0_val = leaves_tree[lo + 6];
    const auto d0_err = leaves_tree[lo + 7];

    // Write Values
    leaves_tree[lo + 0] = d3_val * w3_3;
    leaves_tree[lo + 2] = d2_val * w2_2;
    leaves_tree[lo + 4] = (d1_val * ts) + (d3_val * w3_1);
    leaves_tree[lo + 6] = d0_val + (d2_val * w2_2);

    const double e3_3  = d3_err * w3_3;
    const double e2_2  = d2_err * w2_2;
    const double e3_1  = d3_err * w3_1;
    const double e1_ts = d1_err * ts;

    // Write Errors
    leaves_tree[lo + 1] = e3_3;
    leaves_tree[lo + 3] = e2_2;
    leaves_tree[lo + 5] = sqrt((e1_ts * e1_ts) + (e3_1 * e3_1));
    leaves_tree[lo + 7] = sqrt((d0_err * d0_err) + (e2_2 * e2_2));
}

__device__ __forceinline__ void
cheby_to_taylor_snap(double* __restrict__ leaves_tree,
                     uint32_t n_leaves,
                     cuda::std::pair<double, double> coord_report) {
    constexpr SizeType kLeavesStride = 12;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double ts      = coord_report.second;
    const double inv_ts  = 1.0 / ts;
    const double inv_ts2 = inv_ts * inv_ts;
    const double inv_ts3 = inv_ts2 * inv_ts;
    const double inv_ts4 = inv_ts2 * inv_ts2;
    const double w4_4    = 192.0 * inv_ts4;
    const double w3_3    = 24.0 * inv_ts3;
    const double w2_2    = 4.0 * inv_ts2;
    const double w4_2    = 16.0 * inv_ts2;
    const double w1_1    = inv_ts;
    const double w3_1    = 3.0 * inv_ts;

    const uint32_t lo      = tid * kLeavesStride;
    const auto alpha_4_val = leaves_tree[lo + 0];
    const auto alpha_4_err = leaves_tree[lo + 1];
    const auto alpha_3_val = leaves_tree[lo + 2];
    const auto alpha_3_err = leaves_tree[lo + 3];
    const auto alpha_2_val = leaves_tree[lo + 4];
    const auto alpha_2_err = leaves_tree[lo + 5];
    const auto alpha_1_val = leaves_tree[lo + 6];
    const auto alpha_1_err = leaves_tree[lo + 7];
    const auto alpha_0_val = leaves_tree[lo + 8];
    const auto alpha_0_err = leaves_tree[lo + 9];

    // CSE for Values
    const double term4_2_val = alpha_4_val * w4_2;
    const double term3_1_val = alpha_3_val * w3_1;

    // Write Values
    leaves_tree[lo + 0] = alpha_4_val * w4_4;
    leaves_tree[lo + 2] = alpha_3_val * w3_3;
    leaves_tree[lo + 4] = (alpha_2_val * w2_2) - term4_2_val;
    leaves_tree[lo + 6] = (alpha_1_val * w1_1) - term3_1_val;
    leaves_tree[lo + 8] = alpha_0_val - alpha_2_val + alpha_4_val;

    // Pre-scale errors for variances
    const double e4_4 = alpha_4_err * w4_4;
    const double e3_3 = alpha_3_err * w3_3;
    const double e2_2 = alpha_2_err * w2_2;
    const double e4_2 = alpha_4_err * w4_2;
    const double e1_1 = alpha_1_err * w1_1;
    const double e3_1 = alpha_3_err * w3_1;

    // Write Errors
    leaves_tree[lo + 1] = e4_4;
    leaves_tree[lo + 3] = e3_3;
    leaves_tree[lo + 5] = sqrt((e2_2 * e2_2) + (e4_2 * e4_2));
    leaves_tree[lo + 7] = sqrt((e1_1 * e1_1) + (e3_1 * e3_1));
    leaves_tree[lo + 9] =
        sqrt((alpha_0_err * alpha_0_err) + (alpha_2_err * alpha_2_err) +
             (alpha_4_err * alpha_4_err));
}

__device__ __forceinline__ void
taylor_to_cheby_snap(double* __restrict__ leaves_tree,
                     uint32_t n_leaves,
                     cuda::std::pair<double, double> coord_init) {
    constexpr SizeType kLeavesStride = 12;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }

    const double ts   = coord_init.second;
    const double ts2  = ts * ts;
    const double ts3  = ts2 * ts;
    const double ts4  = ts2 * ts2;
    const double w2_2 = (ts2 / 2.0) * 0.5;
    const double w4_4 = (ts4 / 24.0) * 0.125;
    const double w3_3 = (ts3 / 6.0) * 0.25;
    const double w4_2 = (ts4 / 24.0) * 0.5;
    const double w3_1 = (ts3 / 6.0) * 0.75;
    const double w4_0 = (ts4 / 24.0) * 0.375;

    const uint32_t lo = tid * kLeavesStride;
    const auto d4_val = leaves_tree[lo + 0];
    const auto d4_err = leaves_tree[lo + 1];
    const auto d3_val = leaves_tree[lo + 2];
    const auto d3_err = leaves_tree[lo + 3];
    const auto d2_val = leaves_tree[lo + 4];
    const auto d2_err = leaves_tree[lo + 5];
    const auto d1_val = leaves_tree[lo + 6];
    const auto d1_err = leaves_tree[lo + 7];
    const auto d0_val = leaves_tree[lo + 8];
    const auto d0_err = leaves_tree[lo + 9];

    const double term2_2_val = d2_val * w2_2;
    const double term4_2_val = d4_val * w4_2;

    // Write Values
    leaves_tree[lo + 0] = d4_val * w4_4;
    leaves_tree[lo + 2] = d3_val * w3_3;
    leaves_tree[lo + 4] = term2_2_val + term4_2_val;
    leaves_tree[lo + 6] = (d1_val * ts) + (d3_val * w3_1);
    leaves_tree[lo + 8] = d0_val + term2_2_val + (d4_val * w4_0);

    const double e4_4  = d4_err * w4_4;
    const double e3_3  = d3_err * w3_3;
    const double e4_2  = d4_err * w4_2;
    const double e2_2  = d2_err * w2_2;
    const double e3_1  = d3_err * w3_1;
    const double e4_0  = d4_err * w4_0;
    const double e1_ts = d1_err * ts;

    // Write Errors
    leaves_tree[lo + 1] = e4_4;
    leaves_tree[lo + 3] = e3_3;
    leaves_tree[lo + 5] = sqrt((e2_2 * e2_2) + (e4_2 * e4_2));
    leaves_tree[lo + 7] = sqrt((e1_ts * e1_ts) + (e3_1 * e3_1));
    leaves_tree[lo + 9] =
        sqrt((d0_err * d0_err) + (e2_2 * e2_2) + (e4_0 * e4_0));
}

__global__ void
kernel_poly_taylor_seed(const SizeType* __restrict__ param_grid_count_init,
                        const double* __restrict__ dparams_init,
                        const ParamLimit* __restrict__ param_limits,
                        double* __restrict__ seed_leaves,
                        uint32_t n_leaves,
                        uint32_t n_params) {
    constexpr uint32_t kParamStride = 2;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const auto n_accel_init  = param_grid_count_init[n_params - 2];
    const auto n_freq_init   = param_grid_count_init[n_params - 1];
    const auto d_freq_cur    = dparams_init[n_params - 1];
    const uint32_t accel_idx = ileaf / n_freq_init;
    const uint32_t freq_idx  = ileaf % n_freq_init;

    const double a_cur = utils::get_param_val_at_idx_device(
        param_limits[n_params - 2].min, param_limits[n_params - 2].max,
        n_accel_init, accel_idx);
    const double f_cur = utils::get_param_val_at_idx_device(
        param_limits[n_params - 1].min, param_limits[n_params - 1].max,
        n_freq_init, freq_idx);

    const uint32_t lo = (n_params + 2) * kParamStride * ileaf;
    // Copy till d2 (acceleration)
    for (uint32_t j = 0; j < n_params - 1; ++j) {
        seed_leaves[lo + (j * kParamStride) + 0] = 0.0;
        seed_leaves[lo + (j * kParamStride) + 1] = dparams_init[j];
    }
    seed_leaves[lo + ((n_params - 2) * kParamStride) + 0] = a_cur;
    // Update frequency to velocity
    // f = f0(1 - v / C) => dv = -(C/f0) * df
    seed_leaves[lo + ((n_params - 1) * kParamStride) + 0] = 0;
    seed_leaves[lo + ((n_params - 1) * kParamStride) + 1] =
        d_freq_cur * (utils::kCval / f_cur);
    // intialize d0 (measure from t=t_init) and store f0
    seed_leaves[lo + ((n_params + 0) * kParamStride) + 0] = 0;
    seed_leaves[lo + ((n_params + 0) * kParamStride) + 1] = 0;
    seed_leaves[lo + ((n_params + 1) * kParamStride) + 0] = f_cur;
    // Store basis flag (0: Polynomial, 1: Physical)
    seed_leaves[lo + ((n_params + 1) * kParamStride) + 1] = 0;
}

__global__ void
kernel_analyze_and_branch_accel(const double* __restrict__ leaves_tree,
                                uint32_t n_leaves,
                                double dt,
                                double nbins,
                                double eta,
                                const ParamLimit* __restrict__ param_limits,
                                uint32_t branch_max,
                                utils::BranchingWorkspaceCUDAView branch_ws) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t lo = ileaf * kLeavesStride;
    const uint32_t fb = ileaf * kParams;

    const double dt2     = dt * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double dphi    = eta / nbins;

    const double d2_cur     = leaves_tree[lo + 0];
    const double d2_sig_cur = leaves_tree[lo + 1];
    const double d1_cur     = leaves_tree[lo + 2];
    const double d1_sig_cur = leaves_tree[lo + 3];
    const double f0         = leaves_tree[lo + 6];

    const double dfactor = utils::kCval / f0;
    double d2_sig_new    = dphi * dfactor * 4.0 * inv_dt2;
    double d1_sig_new    = dphi * dfactor * 1.0 * inv_dt;

    const double d2_range = param_limits[0].max - param_limits[0].min;
    const double d1_range =
        dfactor * (param_limits[1].max - param_limits[1].min);

    d2_sig_new = cuda::std::min(d2_sig_new, d2_range);
    d1_sig_new = cuda::std::min(d1_sig_new, d1_range);

    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins / (1.0 * dfactor);

    utils::branch_one_param_padded_device(
        0, d2_cur, d2_sig_cur, d2_sig_new, eta, shift_d2,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        1, d1_cur, d1_sig_cur, d1_sig_new, eta, shift_d1,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);

    const uint32_t c0 = branch_ws.scratch_counts[fb + 0];
    const uint32_t c1 = branch_ws.scratch_counts[fb + 1];

    branch_ws.leaf_branch_count[ileaf] = c0 * c1;
}

__global__ void
kernel_analyze_and_branch_jerk(const double* __restrict__ leaves_tree,
                               uint32_t n_leaves,
                               double dt,
                               double nbins,
                               double eta,
                               const ParamLimit* __restrict__ param_limits,
                               uint32_t branch_max,
                               utils::BranchingWorkspaceCUDAView branch_ws) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t lo = ileaf * kLeavesStride;
    const uint32_t fb = ileaf * kParams;

    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const double dphi    = eta / nbins;

    const double d3_cur     = leaves_tree[lo + 0];
    const double d3_sig_cur = leaves_tree[lo + 1];
    const double d2_cur     = leaves_tree[lo + 2];
    const double d2_sig_cur = leaves_tree[lo + 3];
    const double d1_cur     = leaves_tree[lo + 4];
    const double d1_sig_cur = leaves_tree[lo + 5];
    const double f0         = leaves_tree[lo + 8];

    const double dfactor = utils::kCval / f0;
    double d3_sig_new    = dphi * dfactor * 24.0 * inv_dt3;
    double d2_sig_new    = dphi * dfactor * 4.0 * inv_dt2;
    double d1_sig_new    = dphi * dfactor * 1.0 * inv_dt;

    const double d3_range = param_limits[0].max - param_limits[0].min;
    const double d2_range = param_limits[1].max - param_limits[1].min;
    const double d1_range =
        dfactor * (param_limits[2].max - param_limits[2].min);

    d3_sig_new = cuda::std::min(d3_sig_new, d3_range);
    d2_sig_new = cuda::std::min(d2_sig_new, d2_range);
    d1_sig_new = cuda::std::min(d1_sig_new, d1_range);

    const double shift_d3 =
        (d3_sig_cur - d3_sig_new) * dt3 * nbins / (24.0 * dfactor);
    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins / (1.0 * dfactor);

    utils::branch_one_param_padded_device(
        0, d3_cur, d3_sig_cur, d3_sig_new, eta, shift_d3,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        1, d2_cur, d2_sig_cur, d2_sig_new, eta, shift_d2,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        2, d1_cur, d1_sig_cur, d1_sig_new, eta, shift_d1,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);

    const uint32_t c0 = branch_ws.scratch_counts[fb + 0];
    const uint32_t c1 = branch_ws.scratch_counts[fb + 1];
    const uint32_t c2 = branch_ws.scratch_counts[fb + 2];

    branch_ws.leaf_branch_count[ileaf] = c0 * c1 * c2;
}

__global__ void
kernel_analyze_and_branch_snap(const double* __restrict__ leaves_tree,
                               uint32_t n_leaves,
                               double dt,
                               double nbins,
                               double eta,
                               const ParamLimit* __restrict__ param_limits,
                               uint32_t branch_max,
                               utils::BranchingWorkspaceCUDAView branch_ws) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    const uint32_t ileaf = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ileaf >= n_leaves) {
        return;
    }

    const uint32_t lo = ileaf * kLeavesStride;
    const uint32_t fb = ileaf * kParams;

    const double dt2     = dt * dt;
    const double dt3     = dt2 * dt;
    const double dt4     = dt2 * dt2;
    const double inv_dt  = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double inv_dt3 = inv_dt2 * inv_dt;
    const double inv_dt4 = inv_dt2 * inv_dt2;
    const double dphi    = eta / nbins;

    const double d4_cur     = leaves_tree[lo + 0];
    const double d4_sig_cur = leaves_tree[lo + 1];
    const double d3_cur     = leaves_tree[lo + 2];
    const double d3_sig_cur = leaves_tree[lo + 3];
    const double d2_cur     = leaves_tree[lo + 4];
    const double d2_sig_cur = leaves_tree[lo + 5];
    const double d1_cur     = leaves_tree[lo + 6];
    const double d1_sig_cur = leaves_tree[lo + 7];
    const double f0         = leaves_tree[lo + 10];

    const double dfactor = utils::kCval / f0;
    double d4_sig_new    = dphi * dfactor * 192.0 * inv_dt4;
    double d3_sig_new    = dphi * dfactor * 24.0 * inv_dt3;
    double d2_sig_new    = dphi * dfactor * 4.0 * inv_dt2;
    double d1_sig_new    = dphi * dfactor * 1.0 * inv_dt;

    const double d4_range = param_limits[0].max - param_limits[0].min;
    const double d3_range = param_limits[1].max - param_limits[1].min;
    const double d2_range = param_limits[2].max - param_limits[2].min;
    const double d1_range =
        dfactor * (param_limits[3].max - param_limits[3].min);

    d4_sig_new = cuda::std::min(d4_sig_new, d4_range);
    d3_sig_new = cuda::std::min(d3_sig_new, d3_range);
    d2_sig_new = cuda::std::min(d2_sig_new, d2_range);
    d1_sig_new = cuda::std::min(d1_sig_new, d1_range);

    const double shift_d4 =
        (d4_sig_cur - d4_sig_new) * dt4 * nbins / (192.0 * dfactor);
    const double shift_d3 =
        (d3_sig_cur - d3_sig_new) * dt3 * nbins / (24.0 * dfactor);
    const double shift_d2 =
        (d2_sig_cur - d2_sig_new) * dt2 * nbins / (4.0 * dfactor);
    const double shift_d1 =
        (d1_sig_cur - d1_sig_new) * dt * nbins / (1.0 * dfactor);

    utils::branch_one_param_padded_device(
        0, d4_cur, d4_sig_cur, d4_sig_new, eta, shift_d4,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        1, d3_cur, d3_sig_cur, d3_sig_new, eta, shift_d3,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        2, d2_cur, d2_sig_cur, d2_sig_new, eta, shift_d2,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);
    utils::branch_one_param_padded_device(
        3, d1_cur, d1_sig_cur, d1_sig_new, eta, shift_d1,
        branch_ws.scratch_params, branch_ws.scratch_dparams,
        branch_ws.scratch_counts, fb, branch_max);

    const uint32_t c0 = branch_ws.scratch_counts[fb + 0];
    const uint32_t c1 = branch_ws.scratch_counts[fb + 1];
    const uint32_t c2 = branch_ws.scratch_counts[fb + 2];
    const uint32_t c3 = branch_ws.scratch_counts[fb + 3];

    branch_ws.leaf_branch_count[ileaf] = c0 * c1 * c2 * c3;
}

template <uint32_t KThreadsPerBlock>
__global__ void
kernel_materialize_branches_accel(const double* __restrict__ leaves_tree,
                                  double* __restrict__ leaves_branch,
                                  uint32_t* __restrict__ leaves_origins,
                                  uint32_t n_leaves,
                                  uint32_t n_leaves_branched,
                                  uint32_t branch_max,
                                  utils::BranchingWorkspaceCUDAView branch_ws) {
    constexpr uint32_t kParams       = 2;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    // Strategy: Block-Cooperative Search
    __shared__ uint32_t smem_offsets[KThreadsPerBlock + 2];
    __shared__ uint32_t base_leaf_idx;

    utils::load_block_cooperative_map_device<KThreadsPerBlock>(
        smem_offsets, base_leaf_idx, branch_ws.leaf_output_offset, n_leaves);

    const uint32_t tid          = threadIdx.x;
    const uint32_t branch_ileaf = (blockIdx.x * blockDim.x) + tid;

    if (branch_ileaf >= n_leaves_branched) {
        return;
    }

    // Phase 3: Shared Memory Binary Search
    const uint32_t search_len =
        min(KThreadsPerBlock + 1, n_leaves - base_leaf_idx);
    const uint32_t local_leaf_idx =
        utils::binary_search_cartesian(smem_offsets, search_len, branch_ileaf);
    // Resolve actual leaf index
    const uint32_t tree_ileaf        = base_leaf_idx + local_leaf_idx;
    const uint32_t leaf_start_offset = smem_offsets[local_leaf_idx];
    const uint32_t flat_base         = tree_ileaf * kParams;

    const uint32_t n_d1 = branch_ws.scratch_counts[flat_base + 1];

    // Inverse mapping (Div/Mod)
    const uint32_t rem = branch_ileaf - leaf_start_offset;
    const uint32_t b   = rem % n_d1;
    const uint32_t a   = rem / n_d1;

    const uint32_t bo = branch_ileaf * kLeavesStride;
    const uint32_t lo = tree_ileaf * kLeavesStride;

    leaves_branch[bo + 0] =
        branch_ws.scratch_params[((flat_base + 0) * branch_max) + a];
    leaves_branch[bo + 1] = branch_ws.scratch_dparams[flat_base + 0];
    leaves_branch[bo + 2] =
        branch_ws.scratch_params[((flat_base + 1) * branch_max) + b];
    leaves_branch[bo + 3] = branch_ws.scratch_dparams[flat_base + 1];
// copy d0 + f0 + flag (4 doubles) from parent tree
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        leaves_branch[bo + 4 + k] = leaves_tree[lo + 4 + k];
    }
    leaves_origins[branch_ileaf] = tree_ileaf;
}

template <uint32_t KThreadsPerBlock>
__global__ void
kernel_materialize_branches_jerk(const double* __restrict__ leaves_tree,
                                 double* __restrict__ leaves_branch,
                                 uint32_t* __restrict__ leaves_origins,
                                 uint32_t n_leaves,
                                 uint32_t n_leaves_branched,
                                 uint32_t branch_max,
                                 utils::BranchingWorkspaceCUDAView branch_ws) {
    constexpr uint32_t kParams       = 3;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    // Strategy: Block-Cooperative Search
    __shared__ uint32_t smem_offsets[KThreadsPerBlock + 2];
    __shared__ uint32_t base_leaf_idx;

    utils::load_block_cooperative_map_device<KThreadsPerBlock>(
        smem_offsets, base_leaf_idx, branch_ws.leaf_output_offset, n_leaves);

    const uint32_t tid          = threadIdx.x;
    const uint32_t branch_ileaf = (blockIdx.x * blockDim.x) + tid;

    if (branch_ileaf >= n_leaves_branched) {
        return;
    }

    // Phase 3: SMEM Search
    const uint32_t search_len =
        min(KThreadsPerBlock + 1, n_leaves - base_leaf_idx);
    const uint32_t local_leaf_idx =
        utils::binary_search_cartesian(smem_offsets, search_len, branch_ileaf);

    // Resolve actual leaf index
    const uint32_t tree_ileaf        = base_leaf_idx + local_leaf_idx;
    const uint32_t leaf_start_offset = smem_offsets[local_leaf_idx];
    const uint32_t flat_base         = tree_ileaf * kParams;

    const uint32_t n_d2 = branch_ws.scratch_counts[flat_base + 1];
    const uint32_t n_d1 = branch_ws.scratch_counts[flat_base + 2];

    // Inverse Mapping (Div/Mod)
    uint32_t rem     = branch_ileaf - leaf_start_offset;
    const uint32_t c = rem % n_d1;
    rem /= n_d1;
    const uint32_t b = rem % n_d2;
    const uint32_t a = rem / n_d2; // remaining part is 'a'

    const uint32_t bo = branch_ileaf * kLeavesStride;
    const uint32_t lo = tree_ileaf * kLeavesStride;

    leaves_branch[bo + 0] =
        branch_ws.scratch_params[((flat_base + 0) * branch_max) + a];
    leaves_branch[bo + 1] = branch_ws.scratch_dparams[flat_base + 0];
    leaves_branch[bo + 2] =
        branch_ws.scratch_params[((flat_base + 1) * branch_max) + b];
    leaves_branch[bo + 3] = branch_ws.scratch_dparams[flat_base + 1];
    leaves_branch[bo + 4] =
        branch_ws.scratch_params[((flat_base + 2) * branch_max) + c];
    leaves_branch[bo + 5] = branch_ws.scratch_dparams[flat_base + 2];
// copy d0 + f0 + flag (4 doubles) from parent tree
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        leaves_branch[bo + 6 + k] = leaves_tree[lo + 6 + k];
    }
    leaves_origins[branch_ileaf] = tree_ileaf;
}

template <uint32_t KThreadsPerBlock>
__global__ void
kernel_materialize_branches_snap(const double* __restrict__ leaves_tree,
                                 double* __restrict__ leaves_branch,
                                 uint32_t* __restrict__ leaves_origins,
                                 uint32_t n_leaves,
                                 uint32_t n_leaves_branched,
                                 uint32_t branch_max,
                                 utils::BranchingWorkspaceCUDAView branch_ws) {
    constexpr uint32_t kParams       = 4;
    constexpr uint32_t kParamStride  = 2;
    constexpr uint32_t kLeavesStride = (kParams + 2) * kParamStride;

    // Strategy: Block-Cooperative Search
    __shared__ uint32_t smem_offsets[KThreadsPerBlock + 2];
    __shared__ uint32_t base_leaf_idx;

    utils::load_block_cooperative_map_device<KThreadsPerBlock>(
        smem_offsets, base_leaf_idx, branch_ws.leaf_output_offset, n_leaves);

    const uint32_t tid          = threadIdx.x;
    const uint32_t branch_ileaf = (blockIdx.x * blockDim.x) + tid;

    if (branch_ileaf >= n_leaves_branched) {
        return;
    }

    // Phase 3: SMEM Search
    const uint32_t search_len =
        min(KThreadsPerBlock + 1, n_leaves - base_leaf_idx);
    const uint32_t local_leaf_idx =
        utils::binary_search_cartesian(smem_offsets, search_len, branch_ileaf);

    // Resolve actual leaf index
    const uint32_t tree_ileaf        = base_leaf_idx + local_leaf_idx;
    const uint32_t leaf_start_offset = smem_offsets[local_leaf_idx];
    const uint32_t flat_base         = tree_ileaf * kParams;

    const uint32_t n_d3 = branch_ws.scratch_counts[flat_base + 1];
    const uint32_t n_d2 = branch_ws.scratch_counts[flat_base + 2];
    const uint32_t n_d1 = branch_ws.scratch_counts[flat_base + 3];

    // Inverse Mapping (Div/Mod)
    uint32_t rem     = branch_ileaf - leaf_start_offset;
    const uint32_t d = rem % n_d1;
    rem /= n_d1;
    const uint32_t c = rem % n_d2;
    rem /= n_d2;
    const uint32_t b = rem % n_d3;
    const uint32_t a = rem / n_d3; // remaining part is 'a'

    const uint32_t bo = branch_ileaf * kLeavesStride;
    const uint32_t lo = tree_ileaf * kLeavesStride;

    leaves_branch[bo + 0] =
        branch_ws.scratch_params[((flat_base + 0) * branch_max) + a];
    leaves_branch[bo + 1] = branch_ws.scratch_dparams[flat_base + 0];
    leaves_branch[bo + 2] =
        branch_ws.scratch_params[((flat_base + 1) * branch_max) + b];
    leaves_branch[bo + 3] = branch_ws.scratch_dparams[flat_base + 1];
    leaves_branch[bo + 4] =
        branch_ws.scratch_params[((flat_base + 2) * branch_max) + c];
    leaves_branch[bo + 5] = branch_ws.scratch_dparams[flat_base + 2];
    leaves_branch[bo + 6] =
        branch_ws.scratch_params[((flat_base + 3) * branch_max) + d];
    leaves_branch[bo + 7] = branch_ws.scratch_dparams[flat_base + 3];

// copy d0 + f0 + flag (4 doubles) from parent tree
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        leaves_branch[bo + 8 + k] = leaves_tree[lo + 8 + k];
    }
    leaves_origins[branch_ileaf] = tree_ileaf;
}

__device__ __forceinline__ void
resolve_chebyshev_accel(const double* __restrict__ leaves_tree,
                        const uint8_t* __restrict__ validation_mask,
                        uint32_t* __restrict__ param_indices,
                        float* __restrict__ phase_shift,
                        const ParamLimit* __restrict__ param_limits,
                        cuda::std::pair<double, double> coord_add,
                        cuda::std::pair<double, double> coord_cur,
                        cuda::std::pair<double, double> coord_init,
                        uint32_t n_accel_init,
                        uint32_t n_freq_init,
                        uint32_t nbins,
                        uint32_t n_leaves) {
    constexpr uint32_t kLeavesStride = 8;

    // Compute locally
    const double scale_cur = coord_cur.second;
    const double dt_add    = coord_add.first - coord_cur.first;
    const double dt_init   = coord_init.first - coord_cur.first;
    const double dt        = dt_add - dt_init;
    const double dt2       = (dt_add * dt_add) - (dt_init * dt_init);
    const double inv_ts    = 1.0 / scale_cur;
    const double inv_ts2   = inv_ts * inv_ts;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const uint32_t lo    = tid * kLeavesStride;
    const double alpha_2 = leaves_tree[lo + 0];
    const double alpha_1 = leaves_tree[lo + 2];
    const double f0      = leaves_tree[lo + 6];

    const double a_new   = 2.0 * alpha_2 * inv_ts2;
    const double delta_v = 4.0 * alpha_2 * inv_ts2 * dt;
    const double delta_d =
        (alpha_1 * inv_ts * dt) + (2.0 * alpha_2 * inv_ts2 * dt2);
    const double f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
    const double delay_rel = delta_d * utils::kInvCval;

    const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
        a_new, param_limits[0].min, param_limits[0].max, n_accel_init);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_new, param_limits[1].min, param_limits[1].max, n_freq_init);
    param_indices[tid] = (idx_a * n_freq_init) + idx_f;
    phase_shift[tid]   = utils::get_phase_idx_device(dt, f0, nbins, delay_rel);
}

__device__ __forceinline__ void
resolve_chebyshev_jerk(const double* __restrict__ leaves_tree,
                       const uint8_t* __restrict__ validation_mask,
                       uint32_t* __restrict__ param_indices,
                       float* __restrict__ phase_shift,
                       const ParamLimit* __restrict__ param_limits,
                       cuda::std::pair<double, double> coord_add,
                       cuda::std::pair<double, double> coord_cur,
                       cuda::std::pair<double, double> coord_init,
                       uint32_t n_accel_init,
                       uint32_t n_freq_init,
                       uint32_t nbins,
                       uint32_t n_leaves) {
    constexpr uint32_t kLeavesStride = 10;

    // Compute locally
    const double scale_cur = coord_cur.second;
    const double dt_add    = coord_add.first - coord_cur.first;
    const double dt_init   = coord_init.first - coord_cur.first;
    const double dt2_add   = dt_add * dt_add;
    const double dt2_init  = dt_init * dt_init;
    const double dt        = dt_add - dt_init;
    const double dt2       = dt2_add - dt2_init;
    const double dt3       = (dt2_add * dt_add) - (dt2_init * dt_init);
    const double inv_ts    = 1.0 / scale_cur;
    const double inv_ts2   = inv_ts * inv_ts;
    const double inv_ts3   = inv_ts2 * inv_ts;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const uint32_t lo    = tid * kLeavesStride;
    const double alpha_3 = leaves_tree[lo + 0];
    const double alpha_2 = leaves_tree[lo + 2];
    const double alpha_1 = leaves_tree[lo + 4];
    const double f0      = leaves_tree[lo + 8];

    const double a_new =
        ((2.0 * alpha_2) + (12.0 * alpha_3 * (dt_add * inv_ts))) * inv_ts2;
    const double delta_v =
        ((4.0 * alpha_2 * inv_ts2) * dt) + ((12.0 * alpha_3 * inv_ts3) * dt2);
    const double delta_d   = ((alpha_1 - (3.0 * alpha_3)) * inv_ts * dt) +
                             ((2.0 * alpha_2 * inv_ts2) * dt2) +
                             ((4.0 * alpha_3 * inv_ts3) * dt3);
    const double f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
    const double delay_rel = delta_d * utils::kInvCval;

    const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
        a_new, param_limits[1].min, param_limits[1].max, n_accel_init);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_new, param_limits[2].min, param_limits[2].max, n_freq_init);
    param_indices[tid] = (idx_a * n_freq_init) + idx_f;
    phase_shift[tid]   = utils::get_phase_idx_device(dt, f0, nbins, delay_rel);
}

__device__ __forceinline__ void
resolve_chebyshev_snap(const double* __restrict__ leaves_tree,
                       const uint8_t* __restrict__ validation_mask,
                       uint32_t* __restrict__ param_indices,
                       float* __restrict__ phase_shift,
                       const ParamLimit* __restrict__ param_limits,
                       cuda::std::pair<double, double> coord_add,
                       cuda::std::pair<double, double> coord_cur,
                       cuda::std::pair<double, double> coord_init,
                       uint32_t n_accel_init,
                       uint32_t n_freq_init,
                       uint32_t nbins,
                       uint32_t n_leaves) {
    constexpr uint32_t kLeavesStride = 12;

    // Compute locally
    const double scale_cur = coord_cur.second;
    const double dt_add    = coord_add.first - coord_cur.first;
    const double dt_init   = coord_init.first - coord_cur.first;
    const double dt2_add   = dt_add * dt_add;
    const double dt2_init  = dt_init * dt_init;
    const double dt        = dt_add - dt_init;
    const double dt2       = dt2_add - dt2_init;
    const double dt3       = (dt2_add * dt_add) - (dt2_init * dt_init);
    const double dt4       = (dt2_add * dt2_add) - (dt2_init * dt2_init);
    const double inv_ts    = 1.0 / scale_cur;
    const double inv_ts2   = inv_ts * inv_ts;
    const double inv_ts3   = inv_ts2 * inv_ts;
    const double inv_ts4   = inv_ts3 * inv_ts;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const uint32_t lo    = tid * kLeavesStride;
    const double alpha_4 = leaves_tree[lo + 0];
    const double alpha_3 = leaves_tree[lo + 2];
    const double alpha_2 = leaves_tree[lo + 4];
    const double alpha_1 = leaves_tree[lo + 6];
    const double f0      = leaves_tree[lo + 10];

    const auto a_new =
        ((2.0 * alpha_2) + (12.0 * alpha_3 * (dt_add * inv_ts)) +
         (48.0 * alpha_4 * (dt_add * inv_ts) * (dt_add * inv_ts)) -
         (8.0 * alpha_4)) *
        inv_ts2;
    const auto delta_v = (((4.0 * alpha_2) - (16.0 * alpha_4)) * inv_ts2 * dt) +
                         ((12.0 * alpha_3 * inv_ts3) * dt2) +
                         ((32.0 * alpha_4 * inv_ts4) * dt3);

    const auto delta_d = ((alpha_1 - (3.0 * alpha_3)) * inv_ts * dt) +
                         (((2.0 * alpha_2) - (8.0 * alpha_4)) * inv_ts2 * dt2) +
                         ((4.0 * alpha_3 * inv_ts3) * dt3) +
                         ((8.0 * alpha_4 * inv_ts4) * dt4);
    const double f_new = f0 * (1.0 - (delta_v * utils::kInvCval));
    const double delay_rel = delta_d * utils::kInvCval;

    const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
        a_new, param_limits[2].min, param_limits[2].max, n_accel_init);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_new, param_limits[3].min, param_limits[3].max, n_freq_init);
    param_indices[tid] = (idx_a * n_freq_init) + idx_f;
    phase_shift[tid]   = utils::get_phase_idx_device(dt, f0, nbins, delay_rel);
}

__device__ __forceinline__ void
transform_chebyshev_accel(double* __restrict__ leaves_tree,
                          const uint8_t* __restrict__ validation_mask,
                          uint32_t n_leaves,
                          cuda::std::pair<double, double> coord_next,
                          cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 8;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const double scale_cur  = coord_cur.second;
    const double scale_next = coord_next.second;
    const double dt         = coord_next.first - coord_cur.first;
    const double p          = scale_next / scale_cur;
    const double q          = dt / scale_cur;
    const double p2         = p * p;
    const double q2         = q * q;
    const double pq         = p * q;

    const uint32_t lo          = tid * kLeavesStride;
    const double alpha_2_val_i = leaves_tree[lo + 0];
    const double alpha_2_err_i = leaves_tree[lo + 1];
    const double alpha_1_val_i = leaves_tree[lo + 2];
    const double alpha_1_err_i = leaves_tree[lo + 3];
    const double alpha_0_val_i = leaves_tree[lo + 4];

    leaves_tree[lo + 0] = p2 * alpha_2_val_i;
    leaves_tree[lo + 2] = (p * alpha_1_val_i) + (4.0 * pq * alpha_2_val_i);
    leaves_tree[lo + 4] = alpha_0_val_i + (q * alpha_1_val_i) +
                          ((p2 + (2.0 * q2) - 1.0) * alpha_2_val_i);

    // Non-conservative: errors * |diag(T)|
    leaves_tree[lo + 1] = p2 * alpha_2_err_i;
    leaves_tree[lo + 3] = p * alpha_1_err_i;
}

__device__ __forceinline__ void
transform_chebyshev_jerk(double* __restrict__ leaves_tree,
                         const uint8_t* __restrict__ validation_mask,
                         uint32_t n_leaves,
                         cuda::std::pair<double, double> coord_next,
                         cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 10;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const double scale_cur  = coord_cur.second;
    const double scale_next = coord_next.second;
    const double dt         = coord_next.first - coord_cur.first;
    const double p          = scale_next / scale_cur;
    const double q          = dt / scale_cur;
    const double p2         = p * p;
    const double p3         = p2 * p;
    const double q2         = q * q;

    const uint32_t lo          = tid * kLeavesStride;
    const double alpha_3_val_i = leaves_tree[lo + 0];
    const double alpha_3_err_i = leaves_tree[lo + 1];
    const double alpha_2_val_i = leaves_tree[lo + 2];
    const double alpha_2_err_i = leaves_tree[lo + 3];
    const double alpha_1_val_i = leaves_tree[lo + 4];
    const double alpha_1_err_i = leaves_tree[lo + 5];
    const double alpha_0_val_i = leaves_tree[lo + 6];

    leaves_tree[lo + 0] = p3 * alpha_3_val_i;
    leaves_tree[lo + 2] = (6.0 * p2 * q * alpha_3_val_i) + (p2 * alpha_2_val_i);
    leaves_tree[lo + 4] = (3.0 * p * (p2 + (4.0 * q2) - 1.0) * alpha_3_val_i) +
                          (4.0 * p * q * alpha_2_val_i) + (p * alpha_1_val_i);
    leaves_tree[lo + 6] =
        (q * ((6.0 * p2) + (4.0 * q2) - 3.0) * alpha_3_val_i) +
        ((p2 + (2.0 * q2) - 1.0) * alpha_2_val_i) + (q * alpha_1_val_i) +
        alpha_0_val_i;

    // Non-conservative: errors * |diag(T)|
    leaves_tree[lo + 1] = p3 * alpha_3_err_i;
    leaves_tree[lo + 3] = p2 * alpha_2_err_i;
    leaves_tree[lo + 5] = p * alpha_1_err_i;
}

template <bool UseConservativeTile>
__device__ __forceinline__ void
transform_chebyshev_snap(double* __restrict__ leaves_tree,
                         const uint8_t* __restrict__ validation_mask,
                         uint32_t n_leaves,
                         cuda::std::pair<double, double> coord_next,
                         cuda::std::pair<double, double> coord_cur) {
    constexpr SizeType kLeavesStride = 12;

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    if (validation_mask[tid] == 0) {
        return;
    }

    const double scale_cur  = coord_cur.second;
    const double scale_next = coord_next.second;
    const double dt         = coord_next.first - coord_cur.first;
    const double p          = scale_next / scale_cur;
    const double q          = dt / scale_cur;
    const double p2         = p * p;
    const double p3         = p2 * p;
    const double p4         = p2 * p2;
    const double q2         = q * q;
    const double q4         = q2 * q2;

    const uint32_t lo          = tid * kLeavesStride;
    const double alpha_4_val_i = leaves_tree[lo + 0];
    const double alpha_4_err_i = leaves_tree[lo + 1];
    const double alpha_3_val_i = leaves_tree[lo + 2];
    const double alpha_3_err_i = leaves_tree[lo + 3];
    const double alpha_2_val_i = leaves_tree[lo + 4];
    const double alpha_2_err_i = leaves_tree[lo + 5];
    const double alpha_1_val_i = leaves_tree[lo + 6];
    const double alpha_1_err_i = leaves_tree[lo + 7];
    const double alpha_0_val_i = leaves_tree[lo + 8];

    leaves_tree[lo + 0] = p4 * alpha_4_val_i;
    leaves_tree[lo + 2] =
        ((8.0 * p3 * q) * alpha_4_val_i) + (p3 * alpha_3_val_i);
    leaves_tree[lo + 4] =
        (((4.0 * p4) + (24.0 * p2 * q2) - (4.0 * p2)) * alpha_4_val_i) +
        (6.0 * p2 * q * alpha_3_val_i) + (p2 * alpha_2_val_i);
    leaves_tree[lo + 6] =
        (8.0 * p * q * ((3.0 * p2) + (4.0 * q2) - 2.0) * alpha_4_val_i) +
        (3.0 * p * (p2 + (4.0 * q2) - 1.0) * alpha_3_val_i) +
        (4.0 * p * q * alpha_2_val_i) + (p * alpha_1_val_i);
    leaves_tree[lo + 8] =
        (((3.0 * p4) + (24.0 * p2 * q2) - (4.0 * p2) + (8.0 * q4) - (8.0 * q2) +
          1.0) *
         alpha_4_val_i) +
        (q * ((6.0 * p2) + (4.0 * q2) - 3.0) * alpha_3_val_i) +
        ((p2 + (2.0 * q2) - 1.0) * alpha_2_val_i) + (q * alpha_1_val_i) +
        alpha_0_val_i;

    // Non-conservative: errors * |diag(T)|
    leaves_tree[lo + 1] = p4 * alpha_4_err_i;
    leaves_tree[lo + 3] = p3 * alpha_3_err_i;
    leaves_tree[lo + 5] = p2 * alpha_2_err_i;
    leaves_tree[lo + 7] = p * alpha_1_err_i;
}

// Resolve Kernel
template <int NPARAMS>
__global__ void
kernel_poly_chebyshev_resolve_batch(const double* __restrict__ leaves_tree,
                                    const uint8_t* __restrict__ validation_mask,
                                    uint32_t* __restrict__ param_indices,
                                    float* __restrict__ phase_shift,
                                    const ParamLimit* __restrict__ param_limits,
                                    cuda::std::pair<double, double> coord_add,
                                    cuda::std::pair<double, double> coord_cur,
                                    cuda::std::pair<double, double> coord_init,
                                    SizeType n_accel_init,
                                    SizeType n_freq_init,
                                    SizeType nbins,
                                    uint32_t n_leaves) {
    if constexpr (NPARAMS == 2) {
        resolve_chebyshev_accel(leaves_tree, validation_mask, param_indices,
                                phase_shift, param_limits, coord_add, coord_cur,
                                coord_init, n_accel_init, n_freq_init, nbins,
                                n_leaves);
    } else if constexpr (NPARAMS == 3) {
        resolve_chebyshev_jerk(leaves_tree, validation_mask, param_indices,
                               phase_shift, param_limits, coord_add, coord_cur,
                               coord_init, n_accel_init, n_freq_init, nbins,
                               n_leaves);
    } else if constexpr (NPARAMS == 4) {
        resolve_chebyshev_snap(leaves_tree, validation_mask, param_indices,
                               phase_shift, param_limits, coord_add, coord_cur,
                               coord_init, n_accel_init, n_freq_init, nbins,
                               n_leaves);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Taylor order");
    }
}

// Transform Kernel
template <int NPARAMS>
__global__ void kernel_poly_chebyshev_transform_batch(
    double* __restrict__ leaves_tree,
    const uint8_t* __restrict__ validation_mask,
    uint32_t n_leaves,
    cuda::std::pair<double, double> coord_next,
    cuda::std::pair<double, double> coord_cur) {
    if constexpr (NPARAMS == 2) {
        transform_chebyshev_accel(leaves_tree, validation_mask, n_leaves,
                                  coord_next, coord_cur);
    } else if constexpr (NPARAMS == 3) {
        transform_chebyshev_jerk(leaves_tree, validation_mask, n_leaves,
                                 coord_next, coord_cur);
    } else if constexpr (NPARAMS == 4) {
        transform_chebyshev_snap(leaves_tree, validation_mask, n_leaves,
                                 coord_next, coord_cur);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Chebyshev order");
    }
}

__global__ void kernel_poly_taylor_report_batch(
    double* __restrict__ leaves_tree, uint32_t n_leaves, uint32_t n_params) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= n_leaves) {
        return;
    }
    const uint32_t leaves_stride = n_params * 2;
    const uint32_t leaf_offset   = tid * leaves_stride;
    const double v_final  = leaves_tree[leaf_offset + ((n_params - 1) * 2) + 0];
    const double dv_final = leaves_tree[leaf_offset + ((n_params - 1) * 2) + 1];
    const double f0_batch = leaves_tree[leaf_offset + ((n_params + 1) * 2) + 0];
    const double s_factor = 1.0 - (v_final * utils::kInvCval);
    // Gauge transform + error propagation
    for (uint32_t j = 0; j < n_params - 1; ++j) {
        const uint32_t param_offset   = leaf_offset + (j * 2);
        const double param_val        = leaves_tree[param_offset + 0];
        const double param_err        = leaves_tree[param_offset + 1];
        leaves_tree[param_offset + 0] = param_val / s_factor;
        leaves_tree[param_offset + 1] =
            sqrt(((param_err / s_factor) * (param_err / s_factor)) +
                 ((param_val * utils::kInvCval / (s_factor * s_factor)) *
                  (param_val * utils::kInvCval / (s_factor * s_factor)) *
                  (dv_final * dv_final)));
    }
    leaves_tree[leaf_offset + ((n_params - 1) * 2) + 0] = f0_batch * s_factor;
    leaves_tree[leaf_offset + ((n_params - 1) * 2) + 1] =
        f0_batch * dv_final * utils::kInvCval;
}

// Chebyshev to Taylor Transform Kernel
template <int NPARAMS>
__global__ void kernel_poly_cheby_to_taylor_batch(
    double* __restrict__ leaves_tree,
    uint32_t n_leaves,
    cuda::std::pair<double, double> coord_report) {
    if constexpr (NPARAMS == 2) {
        cheby_to_taylor_accel(leaves_tree, n_leaves, coord_report);
    } else if constexpr (NPARAMS == 3) {
        cheby_to_taylor_jerk(leaves_tree, n_leaves, coord_report);
    } else if constexpr (NPARAMS == 4) {
        cheby_to_taylor_snap(leaves_tree, n_leaves, coord_report);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Chebyshev order");
    }
}

// Taylor to Chebyshev Transform Kernel
template <int NPARAMS>
__global__ void
kernel_poly_taylor_to_cheby_batch(double* __restrict__ leaves_tree,
                                  uint32_t n_leaves,
                                  cuda::std::pair<double, double> coord_init) {
    if constexpr (NPARAMS == 2) {
        taylor_to_cheby_accel(leaves_tree, n_leaves, coord_init);
    } else if constexpr (NPARAMS == 3) {
        taylor_to_cheby_jerk(leaves_tree, n_leaves, coord_init);
    } else if constexpr (NPARAMS == 4) {
        taylor_to_cheby_snap(leaves_tree, n_leaves, coord_init);
    } else {
        static_assert(NPARAMS <= 4, "Unsupported Chebyshev order");
    }
}

template <SizeType NPARAMS>
SizeType
poly_taylor_branch_impl_cuda(cuda::std::span<const double> leaves_tree,
                             cuda::std::span<double> leaves_branch,
                             cuda::std::span<uint32_t> leaves_origins,
                             cuda::std::span<uint8_t> validation_mask,
                             std::pair<double, double> coord_cur,
                             SizeType nbins,
                             double eta,
                             cuda::std::span<const ParamLimit> param_limits,
                             SizeType branch_max,
                             SizeType n_leaves,
                             utils::BranchingWorkspaceCUDAView branch_ws,
                             utils::CUBScratchArena& scratch_ws,
                             cudaStream_t stream) {
    static_assert(NPARAMS >= 2 && NPARAMS <= 4);
    constexpr SizeType kLeavesStride = (NPARAMS + 2) * 2;

    // ---- Kernel 1: analyze + branch enumeration ----
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    if constexpr (NPARAMS == 2) {
        kernel_analyze_and_branch_accel<<<grid_dim, block_dim, 0, stream>>>(
            leaves_tree.data(), n_leaves, coord_cur.second,
            static_cast<double>(nbins), eta, param_limits.data(), branch_max,
            branch_ws);
    } else if constexpr (NPARAMS == 3) {
        kernel_analyze_and_branch_jerk<<<grid_dim, block_dim, 0, stream>>>(
            leaves_tree.data(), n_leaves, coord_cur.second,
            static_cast<double>(nbins), eta, param_limits.data(), branch_max,
            branch_ws);
    } else {
        kernel_analyze_and_branch_snap<<<grid_dim, block_dim, 0, stream>>>(
            leaves_tree.data(), n_leaves, coord_cur.second,
            static_cast<double>(nbins), eta, param_limits.data(), branch_max,
            branch_ws);
    }
    cuda_utils::check_last_cuda_error("Kernel 1 launch failed");

    // compute output size and offsets (leaf_output_offset)
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(
            scratch_ws.cub_temp_storage, scratch_ws.cub_temp_bytes,
            branch_ws.leaf_branch_count, branch_ws.leaf_output_offset, n_leaves,
            stream),
        "cub::DeviceScan::ExclusiveSum failed");
    uint32_t last_offset, last_count;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_offset,
                        branch_ws.leaf_output_offset + (n_leaves - 1),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(&last_count,
                        branch_ws.leaf_branch_count + (n_leaves - 1),
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync failed");
    // Unavoidable sync
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize branch kernel failed");
    const SizeType n_leaves_branched = last_offset + last_count;
    if (n_leaves_branched == 0) {
        return n_leaves_branched;
    }

    // Set validation mask for produced outputs
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(validation_mask.data(), 1,
                        n_leaves_branched * sizeof(uint8_t), stream),
        "cudaMemsetAsync failed");

    if (n_leaves_branched == n_leaves) {
        // FAST PATH: no branching
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(leaves_branch.data(), leaves_tree.data(),
                            n_leaves * kLeavesStride * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync failed");
        thrust::sequence(thrust::cuda::par.on(stream), leaves_origins.data(),
                         leaves_origins.data() + n_leaves,
                         static_cast<uint32_t>(0));

        return n_leaves_branched;
    }

    // ---- Kernel 2: materialize ----
    const auto blocks_per_grid_out =
        (n_leaves_branched + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 grid_dim_out(blocks_per_grid_out);
    cuda_utils::check_kernel_launch_params(grid_dim_out, block_dim);
    if constexpr (NPARAMS == 2) {

        kernel_materialize_branches_accel<kThreadsPerBlock>
            <<<grid_dim_out, block_dim, 0, stream>>>(
                leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
                n_leaves, n_leaves_branched, branch_max, branch_ws);
    } else if constexpr (NPARAMS == 3) {
        kernel_materialize_branches_jerk<kThreadsPerBlock>
            <<<grid_dim_out, block_dim, 0, stream>>>(
                leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
                n_leaves, n_leaves_branched, branch_max, branch_ws);
    } else {
        kernel_materialize_branches_snap<kThreadsPerBlock>
            <<<grid_dim_out, block_dim, 0, stream>>>(
                leaves_tree.data(), leaves_branch.data(), leaves_origins.data(),
                n_leaves, n_leaves_branched, branch_max, branch_ws);
    }
    cuda_utils::check_last_cuda_error("Kernel 2 launch failed");
    // No need to sync, the next kernel will do it
    return n_leaves_branched;
}

} // namespace

void poly_chebyshev_seed_cuda(
    cuda::std::span<const SizeType> param_grid_count_init,
    cuda::std::span<const double> dparams_init,
    cuda::std::span<const ParamLimit> param_limits,
    cuda::std::span<double> seed_leaves,
    std::pair<double, double> coord_init,
    SizeType n_leaves,
    SizeType n_params,
    cudaStream_t stream) {
    constexpr SizeType kThreadPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadPerBlock - 1) / kThreadPerBlock;

    const dim3 block_dim(kThreadPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    kernel_poly_chebyshev_seed<<<grid_dim, block_dim, 0, stream>>>(
        param_grid_count_init.data(), dparams_init.data(), param_limits.data(),
        seed_leaves.data(), coord_init, n_leaves, n_params);
    cuda_utils::check_last_cuda_error("Chebyshev seed kernel launch failed");

    // Convert to Chebyshev basis
    auto dispatch = [&]<int N>() {
        kernel_poly_taylor_to_cheby_batch<N>
            <<<grid_dim, block_dim, 0, stream>>>(seed_leaves.data(), n_leaves,
                                                 coord_init);
    };

    switch (n_params) {
    case 2:
        dispatch.template operator()<2>();
        break;
    case 3:
        dispatch.template operator()<3>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported n_params");
    }

    cuda_utils::check_last_cuda_error(
        "kernel_poly_taylor_to_cheby_batch kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize seed kernel failed");
}

SizeType
poly_chebyshev_branch_batch_cuda(cuda::std::span<double> leaves_tree,
                                 cuda::std::span<double> leaves_branch,
                                 cuda::std::span<uint32_t> leaves_origins,
                                 cuda::std::span<uint8_t> validation_mask,
                                 std::pair<double, double> coord_cur,
                                 std::pair<double, double> coord_prev,
                                 SizeType nbins,
                                 double eta,
                                 cuda::std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 SizeType n_params,
                                 utils::BranchingWorkspaceCUDAView branch_ws,
                                 utils::CUBScratchArena& scratch_ws,
                                 cudaStream_t stream) {
    auto dispatch = [&]<SizeType N>() {
        return poly_taylor_branch_impl_cuda<N>(
            leaves_tree, leaves_branch, leaves_origins, validation_mask,
            coord_cur, nbins, eta, param_limits, branch_max, n_leaves,
            branch_ws, scratch_ws, stream);
    };
    switch (n_params) {
    case 2:
        return dispatch.template operator()<2>();
    case 3:
        return dispatch.template operator()<3>();
    case 4:
        return dispatch.template operator()<4>();
    default:
        throw std::invalid_argument("Unsupported n_params");
    }
}

void poly_chebyshev_resolve_batch_cuda(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    cuda::std::span<const ParamLimit> param_limits,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves,
    SizeType n_params,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;
    const auto blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<int N>() {
        kernel_poly_chebyshev_resolve_batch<N>
            <<<grid_dim, block_dim, 0, stream>>>(
                leaves_branch.data(), validation_mask.data(),
                param_indices.data(), phase_shift.data(), param_limits.data(),
                coord_add, coord_cur, coord_init, n_accel_init, n_freq_init,
                nbins, n_leaves);
    };

    // Fully specialized dispatch{
    switch (n_params) {
    case 2:
        dispatch.template operator()<2>();
        break;
    case 3:
        dispatch.template operator()<3>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported n_params");
    }

    cuda_utils::check_last_cuda_error("Chebyshev resolve kernel launch failed");
    // No need to sync, the next kernel will do it
}

void poly_chebyshev_transform_batch_cuda(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    SizeType n_params,
    cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256;
    const SizeType blocks_per_grid =
        (n_leaves + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);

    auto dispatch = [&]<int N>() {
        kernel_poly_chebyshev_transform_batch<N>
            <<<grid_dim, block_dim, 0, stream>>>(
                leaves_tree.data(), validation_mask.data(), n_leaves,
                coord_next, coord_cur);
    };

    // Fully specialized dispatch{
    switch (n_params) {
    case 2:
        dispatch.template operator()<2>();
        break;
    case 3:
        dispatch.template operator()<3>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported n_params");
    }
    cuda_utils::check_last_cuda_error(
        "Chebyshev transform kernel launch failed");
    // No need to sync, the next kernel will do it
}

void poly_taylor_report_batch_cuda(cuda::std::span<double> leaves_tree,
                                   std::pair<double, double> /*coord_report*/,
                                   SizeType n_leaves,
                                   SizeType n_params,
                                   cudaStream_t stream) {
    // Better occupancy than 512 for many kernels
    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((n_leaves + kBlockSize - 1) / kBlockSize);
    cuda_utils::check_kernel_launch_params(grid, block);

    // Convert to Taylor basis
    auto dispatch = [&]<int N>() {
        kernel_poly_cheby_to_taylor_batch<N><<<grid, block, 0, stream>>>(
            leaves_tree.data(), n_leaves, coord_report);
    };

    switch (n_params) {
    case 2:
        dispatch.template operator()<2>();
        break;
    case 3:
        dispatch.template operator()<3>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported n_params");
    }

    cuda_utils::check_last_cuda_error(
        "kernel_poly_cheby_to_taylor_batch kernel launch failed");

    kernel_poly_taylor_report_batch<<<grid, block, 0, stream>>>(
        leaves_tree.data(), n_leaves, n_params);
    cuda_utils::check_last_cuda_error("Taylor report kernel launch failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize report kernel failed");
}

} // namespace loki::core