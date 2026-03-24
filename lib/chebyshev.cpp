#include "loki/core/chebyshev.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>

#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"
#include "loki/transforms.hpp"
#include "loki/utils.hpp"
#include "loki/utils/workspace.hpp"

namespace loki::core {

namespace {

void poly_cheby_to_taylor_accel_batch(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_report,
                                      SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double ts      = coord_report.second;
    const double inv_ts  = 1.0 / ts;
    const double inv_ts2 = inv_ts * inv_ts;

    // Precompute invariant matrix weights.
    const double w2_2 = 4.0 * inv_ts2;
    const double w1_1 = inv_ts;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo          = i * kLeavesStride;
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
        leaves_tree[lo + 5] = std::sqrt((alpha_0_err * alpha_0_err) +
                                        (alpha_2_err * alpha_2_err));
    }
}

void poly_taylor_to_cheby_accel_batch(std::span<double> leaves_tree,
                                      std::pair<double, double> coord_init,
                                      SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double ts  = coord_init.second;
    const double ts2 = ts * ts;

    // Precompute all invariant matrix weights.
    const double w2_2 = (ts2 / 2.0) * 0.5;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo     = i * kLeavesStride;
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
        leaves_tree[lo + 5] = std::sqrt((d0_err * d0_err) + (e2 * e2));
    }
}

void poly_cheby_to_taylor_jerk_batch(std::span<double> leaves_tree,
                                     std::pair<double, double> coord_report,
                                     SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double ts      = coord_report.second;
    const double inv_ts  = 1.0 / ts;
    const double inv_ts2 = inv_ts * inv_ts;
    const double inv_ts3 = inv_ts2 * inv_ts;

    // Precompute invariant matrix weights.
    const double w3_3 = 24.0 * inv_ts3;
    const double w2_2 = 4.0 * inv_ts2;
    const double w1_1 = inv_ts;
    const double w3_1 = 3.0 * inv_ts;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo          = i * kLeavesStride;
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
        leaves_tree[lo + 5] = std::sqrt((e1_1 * e1_1) + (e3_1 * e3_1));
        leaves_tree[lo + 7] = std::sqrt((alpha_0_err * alpha_0_err) +
                                        (alpha_2_err * alpha_2_err));
    }
}

void poly_taylor_to_cheby_jerk_batch(std::span<double> leaves_tree,
                                     std::pair<double, double> coord_init,
                                     SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double ts  = coord_init.second;
    const double ts2 = ts * ts;
    const double ts3 = ts2 * ts;

    // Precompute all invariant matrix weights.
    const double w3_3 = (ts3 / 6.0) * 0.25;
    const double w2_2 = (ts2 / 2.0) * 0.5;
    const double w3_1 = (ts3 / 6.0) * 0.75;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo     = i * kLeavesStride;
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
        leaves_tree[lo + 5] = std::sqrt((e1_ts * e1_ts) + (e3_1 * e3_1));
        leaves_tree[lo + 7] = std::sqrt((d0_err * d0_err) + (e2_2 * e2_2));
    }
}

void poly_cheby_to_taylor_snap_batch(std::span<double> leaves_tree,
                                     std::pair<double, double> coord_report,
                                     SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double ts      = coord_report.second;
    const double inv_ts  = 1.0 / ts;
    const double inv_ts2 = inv_ts * inv_ts;
    const double inv_ts3 = inv_ts2 * inv_ts;
    const double inv_ts4 = inv_ts2 * inv_ts2;

    // Precompute invariant matrix weights.
    const double w4_4 = 192.0 * inv_ts4;
    const double w3_3 = 24.0 * inv_ts3;
    const double w2_2 = 4.0 * inv_ts2;
    const double w4_2 = 16.0 * inv_ts2;
    const double w1_1 = inv_ts;
    const double w3_1 = 3.0 * inv_ts;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo          = i * kLeavesStride;
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
        leaves_tree[lo + 5] = std::sqrt((e2_2 * e2_2) + (e4_2 * e4_2));
        leaves_tree[lo + 7] = std::sqrt((e1_1 * e1_1) + (e3_1 * e3_1));
        leaves_tree[lo + 9] = std::sqrt((alpha_0_err * alpha_0_err) +
                                        (alpha_2_err * alpha_2_err) +
                                        (alpha_4_err * alpha_4_err));
    }
}

void poly_taylor_to_cheby_snap_batch(std::span<double> leaves_tree,
                                     std::pair<double, double> coord_init,
                                     SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");

    const double ts  = coord_init.second;
    const double ts2 = ts * ts;
    const double ts3 = ts2 * ts;
    const double ts4 = ts2 * ts2;

    // Precompute all invariant matrix weights.
    const double w2_2 = (ts2 / 2.0) * 0.5;
    const double w4_4 = (ts4 / 24.0) * 0.125;
    const double w3_3 = (ts3 / 6.0) * 0.25;
    const double w4_2 = (ts4 / 24.0) * 0.5;
    const double w3_1 = (ts3 / 6.0) * 0.75;
    const double w4_0 = (ts4 / 24.0) * 0.375;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo     = i * kLeavesStride;
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
        leaves_tree[lo + 5] = std::sqrt((e2_2 * e2_2) + (e4_2 * e4_2));
        leaves_tree[lo + 7] = std::sqrt((e1_ts * e1_ts) + (e3_1 * e3_1));
        leaves_tree[lo + 9] =
            std::sqrt((d0_err * d0_err) + (e2_2 * e2_2) + (e4_0 * e4_0));
    }
}

void poly_chebyshev_transform_accel_batch(std::span<double> leaves_tree,
                                          std::span<SizeType> indices_tree,
                                          std::pair<double, double> coord_next,
                                          std::pair<double, double> coord_cur,
                                          SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto dt = t0_next - t0_cur;
    const auto p  = scale_next / scale_cur;
    const auto q  = dt / scale_cur;
    const auto p2 = p * p;
    const auto q2 = q * q;
    const auto pq = p * q;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo            = indices_tree[i] * kLeavesStride;
        const auto alpha_2_val_i = leaves_tree[lo + 0];
        const auto alpha_2_err_i = leaves_tree[lo + 1];
        const auto alpha_1_val_i = leaves_tree[lo + 2];
        const auto alpha_1_err_i = leaves_tree[lo + 3];
        const auto alpha_0_val_i = leaves_tree[lo + 4];

        leaves_tree[lo + 0] = p2 * alpha_2_val_i;
        leaves_tree[lo + 2] = (p * alpha_1_val_i) + (4.0 * pq * alpha_2_val_i);
        leaves_tree[lo + 4] = alpha_0_val_i + (q * alpha_1_val_i) +
                              ((p2 + (2.0 * q2) - 1.0) * alpha_2_val_i);

        // Non-conservative: errors * |diag(T)|
        leaves_tree[lo + 1] = p2 * alpha_2_err_i;
        leaves_tree[lo + 3] = p * alpha_1_err_i;
    }
}

void poly_chebyshev_transform_jerk_batch(std::span<double> leaves_tree,
                                         std::span<SizeType> indices_tree,
                                         std::pair<double, double> coord_next,
                                         std::pair<double, double> coord_cur,
                                         SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto dt = t0_next - t0_cur;
    const auto p  = scale_next / scale_cur;
    const auto q  = dt / scale_cur;
    const auto p2 = p * p;
    const auto p3 = p2 * p;
    const auto q2 = q * q;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo            = indices_tree[i] * kLeavesStride;
        const auto alpha_3_val_i = leaves_tree[lo + 0];
        const auto alpha_3_err_i = leaves_tree[lo + 1];
        const auto alpha_2_val_i = leaves_tree[lo + 2];
        const auto alpha_2_err_i = leaves_tree[lo + 3];
        const auto alpha_1_val_i = leaves_tree[lo + 4];
        const auto alpha_1_err_i = leaves_tree[lo + 5];
        const auto alpha_0_val_i = leaves_tree[lo + 6];

        leaves_tree[lo + 0] = p3 * alpha_3_val_i;
        leaves_tree[lo + 2] =
            (6.0 * p2 * q * alpha_3_val_i) + (p2 * alpha_2_val_i);
        leaves_tree[lo + 4] =
            (3.0 * p * (p2 + (4.0 * q2) - 1.0) * alpha_3_val_i) +
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
}

void poly_chebyshev_transform_snap_batch(std::span<double> leaves_tree,
                                         std::span<SizeType> indices_tree,
                                         std::pair<double, double> coord_next,
                                         std::pair<double, double> coord_cur,
                                         SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "batch_leaves size mismatch");
    error_check::check_greater_equal(indices_tree.size(), n_leaves,
                                     "indices_tree size mismatch");

    const auto [t0_next, scale_next] = coord_next;
    const auto [t0_cur, scale_cur]   = coord_cur;
    // Pre-compute constants to avoid repeated calculations
    const auto dt = t0_next - t0_cur;
    const auto p  = scale_next / scale_cur;
    const auto q  = dt / scale_cur;
    const auto p2 = p * p;
    const auto p3 = p2 * p;
    const auto p4 = p2 * p2;
    const auto q2 = q * q;
    const auto q4 = q2 * q2;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo            = indices_tree[i] * kLeavesStride;
        const auto alpha_4_val_i = leaves_tree[lo + 0];
        const auto alpha_4_err_i = leaves_tree[lo + 1];
        const auto alpha_3_val_i = leaves_tree[lo + 2];
        const auto alpha_3_err_i = leaves_tree[lo + 3];
        const auto alpha_2_val_i = leaves_tree[lo + 4];
        const auto alpha_2_err_i = leaves_tree[lo + 5];
        const auto alpha_1_val_i = leaves_tree[lo + 6];
        const auto alpha_1_err_i = leaves_tree[lo + 7];
        const auto alpha_0_val_i = leaves_tree[lo + 8];

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
            (((3.0 * p4) + (24.0 * p2 * q2) - (4.0 * p2) + (8.0 * q4) -
              (8.0 * q2) + 1.0) *
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
}

SizeType
poly_chebyshev_branch_accel_batch(std::span<double> leaves_tree,
                                  std::span<double> leaves_branch,
                                  std::span<SizeType> leaves_origins,
                                  std::pair<double, double> coord_cur,
                                  std::pair<double, double> coord_prev,
                                  SizeType nbins,
                                  double eta,
                                  std::span<const ParamLimit> param_limits,
                                  SizeType branch_max,
                                  SizeType n_leaves,
                                  utils::BranchingWorkspaceView branch_ws) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");
    const auto [_, ts] = coord_cur; // scale_cur
    const double ts2   = ts * ts;
    const auto nbins_d = static_cast<double>(nbins);
    const double dphi  = eta / nbins_d;

    const double d2_range      = param_limits[0].max - param_limits[0].min;
    const double f0_range      = param_limits[1].max - param_limits[1].min;
    const double alpha_2_range = 0.25 * d2_range * ts2;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");

    // Transform the parameters to coord_cur domain
    std::span<SizeType> indices_tree = leaves_origins.subspan(0, n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        indices_tree[i] = i;
    }
    poly_chebyshev_transform_accel_batch(leaves_tree, indices_tree, coord_cur,
                                         coord_prev, n_leaves);

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ dparam_new_ptr        = dparam_new.data();
    double* __restrict__ shift_bins_ptr        = shift_bins.data();

    // Step + Shift
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto alpha_2_sig_cur = leaves_tree_ptr[lo + 1];
        const auto alpha_1_sig_cur = leaves_tree_ptr[lo + 3];
        const auto f0              = leaves_tree_ptr[lo + 6];

        // Base step in Chebyshev units: dphi * C / f0 (same for all alpha_k)
        const auto dfactor   = utils::kCval / f0;
        const auto base_step = dphi * dfactor;
        // Compute steps
        const auto alpha_1_range   = dfactor * f0_range * ts;
        const auto alpha_2_sig_new = std::min(base_step, alpha_2_range);
        const auto alpha_1_sig_new = std::min(base_step, alpha_1_range);
        dparam_new_ptr[fb + 0]     = alpha_2_sig_new;
        dparam_new_ptr[fb + 1]     = alpha_1_sig_new;

        // Compute shift bins
        const double inv_base = nbins_d / dfactor;
        shift_bins_ptr[fb + 0] =
            std::abs(alpha_2_sig_cur - alpha_2_sig_new) * inv_base;
        shift_bins_ptr[fb + 1] =
            std::abs(alpha_1_sig_cur - alpha_1_sig_new) * inv_base;
    }

    // Early Exit: Check if any leaf needs branching
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - utils::kEps)) {
            any_branching = true;
            break;
        }
    }
    // Fast path: no branching at all
    if (!any_branching) {
        std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                    n_leaves * kLeavesStride * sizeof(double));
        for (SizeType i = 0; i < n_leaves; ++i) {
            leaves_origins_ptr[i] = i;
        }
        return n_leaves;
    }

    // Branching
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo         = i * kLeavesStride;
        const auto fb         = i * kParams;
        const auto d2_cur     = leaves_tree_ptr[lo + 0];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d1_cur     = leaves_tree_ptr[lo + 2];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 3];

        // Branch d2-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            1, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
    }

    // Fill leaves_origins
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo            = i * kLeavesStride;
        const SizeType fb            = i * kParams;
        const SizeType n_d2_branches = branch_ws.scratch_counts[fb + 0];
        const SizeType n_d1_branches = branch_ws.scratch_counts[fb + 1];
        const SizeType d2_offset     = (fb + 0) * branch_max;
        const SizeType d1_offset     = (fb + 1) * branch_max;

        for (SizeType a = 0; a < n_d2_branches; ++a) {
            for (SizeType b = 0; b < n_d1_branches; ++b) {
                const SizeType bo = out_leaves * kLeavesStride;
                leaves_branch_ptr[bo + 0] =
                    branch_ws.scratch_params[d2_offset + a];
                leaves_branch_ptr[bo + 1] = branch_ws.scratch_dparams[fb + 0];
                leaves_branch_ptr[bo + 2] =
                    branch_ws.scratch_params[d1_offset + b];
                leaves_branch_ptr[bo + 3] = branch_ws.scratch_dparams[fb + 1];
                // Fill d0 and f0 directly from leaves_tree
                std::memcpy(leaves_branch_ptr + bo + 4,
                            leaves_tree_ptr + lo + 4, 4 * sizeof(double));

                leaves_origins_ptr[out_leaves] = i;
                ++out_leaves;
            }
        }
    }

    return out_leaves;
}

SizeType
poly_chebyshev_branch_jerk_batch(std::span<double> leaves_tree,
                                 std::span<double> leaves_branch,
                                 std::span<SizeType> leaves_origins,
                                 std::pair<double, double> coord_cur,
                                 std::pair<double, double> coord_prev,
                                 SizeType nbins,
                                 double eta,
                                 std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 utils::BranchingWorkspaceView branch_ws) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");
    const auto [_, ts] = coord_cur; // scale_cur
    const double ts2   = ts * ts;
    const double ts3   = ts2 * ts;
    const auto nbins_d = static_cast<double>(nbins);
    const double dphi  = eta / nbins_d;

    const double d3_range      = param_limits[0].max - param_limits[0].min;
    const double d2_range      = param_limits[1].max - param_limits[1].min;
    const double f0_range      = param_limits[2].max - param_limits[2].min;
    const double alpha_3_range = d3_range * ts3 / 24.0;
    const double alpha_2_range = d2_range * ts2 / 4.0;

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");
    // Transform the parameters to coord_cur domain
    std::span<SizeType> indices_tree = leaves_origins.subspan(0, n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        indices_tree[i] = i;
    }
    poly_chebyshev_transform_accel_batch(leaves_tree, indices_tree, coord_cur,
                                         coord_prev, n_leaves);

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ dparam_new_ptr        = dparam_new.data();
    double* __restrict__ shift_bins_ptr        = shift_bins.data();

    // Step + Shift
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto alpha_3_sig_cur = leaves_tree_ptr[lo + 1];
        const auto alpha_2_sig_cur = leaves_tree_ptr[lo + 3];
        const auto alpha_1_sig_cur = leaves_tree_ptr[lo + 5];
        const auto f0              = leaves_tree_ptr[lo + 8];

        // Base step in Chebyshev units: dphi * C / f0 (same for all alpha_k)
        const auto dfactor   = utils::kCval / f0;
        const auto base_step = dphi * dfactor;
        // Compute steps
        const auto alpha_1_range =
            (dfactor * f0_range * ts) + (d3_range * ts3 / 8.0);
        const auto alpha_3_sig_new = std::min(base_step, alpha_3_range);
        const auto alpha_2_sig_new = std::min(base_step, alpha_2_range);
        const auto alpha_1_sig_new = std::min(base_step, alpha_1_range);
        dparam_new_ptr[fb + 0]     = alpha_3_sig_new;
        dparam_new_ptr[fb + 1]     = alpha_2_sig_new;
        dparam_new_ptr[fb + 2]     = alpha_1_sig_new;

        // Compute shift bins
        const double inv_base = nbins_d / dfactor;
        shift_bins_ptr[fb + 0] =
            std::abs(alpha_3_sig_cur - alpha_3_sig_new) * inv_base;
        shift_bins_ptr[fb + 1] =
            std::abs(alpha_2_sig_cur - alpha_2_sig_new) * inv_base;
        shift_bins_ptr[fb + 2] =
            std::abs(alpha_1_sig_cur - alpha_1_sig_new) * inv_base;
    }

    // Early Exit: Check if any leaf needs branching
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - utils::kEps)) {
            any_branching = true;
            break;
        }
    }
    // Fast path: no branching at all
    if (!any_branching) {
        std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                    n_leaves * kLeavesStride * sizeof(double));
        for (SizeType i = 0; i < n_leaves; ++i) {
            leaves_origins_ptr[i] = i;
        }
        return n_leaves;
    }

    // Branching
    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo         = i * kLeavesStride;
        const auto fb         = i * kParams;
        const auto d3_cur     = leaves_tree_ptr[lo + 0];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d2_cur     = leaves_tree_ptr[lo + 2];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d1_cur     = leaves_tree_ptr[lo + 4];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 5];

        // Branch d3-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d3_cur, d3_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            1, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            2, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
    }

    // Fill leaves_origins
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo            = i * kLeavesStride;
        const SizeType fb            = i * kParams;
        const SizeType n_d3_branches = branch_ws.scratch_counts[fb + 0];
        const SizeType n_d2_branches = branch_ws.scratch_counts[fb + 1];
        const SizeType n_d1_branches = branch_ws.scratch_counts[fb + 2];
        const SizeType d3_offset     = (fb + 0) * branch_max;
        const SizeType d2_offset     = (fb + 1) * branch_max;
        const SizeType d1_offset     = (fb + 2) * branch_max;

        for (SizeType a = 0; a < n_d3_branches; ++a) {
            for (SizeType b = 0; b < n_d2_branches; ++b) {
                for (SizeType c = 0; c < n_d1_branches; ++c) {
                    const SizeType bo = out_leaves * kLeavesStride;
                    leaves_branch_ptr[bo + 0] =
                        branch_ws.scratch_params[d3_offset + a];
                    leaves_branch_ptr[bo + 1] =
                        branch_ws.scratch_dparams[fb + 0];
                    leaves_branch_ptr[bo + 2] =
                        branch_ws.scratch_params[d2_offset + b];
                    leaves_branch_ptr[bo + 3] =
                        branch_ws.scratch_dparams[fb + 1];
                    leaves_branch_ptr[bo + 4] =
                        branch_ws.scratch_params[d1_offset + c];
                    leaves_branch_ptr[bo + 5] =
                        branch_ws.scratch_dparams[fb + 2];
                    // Fill d0 and f0 directly from leaves_tree
                    std::memcpy(leaves_branch_ptr + bo + 6,
                                leaves_tree_ptr + lo + 6, 4 * sizeof(double));

                    leaves_origins_ptr[out_leaves] = i;
                    ++out_leaves;
                }
            }
        }
    }

    return out_leaves;
}

SizeType
poly_chebyshev_branch_snap_batch(std::span<double> leaves_tree,
                                 std::span<double> leaves_branch,
                                 std::span<SizeType> leaves_origins,
                                 std::pair<double, double> coord_cur,
                                 std::pair<double, double> coord_prev,
                                 SizeType nbins,
                                 double eta,
                                 std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 utils::BranchingWorkspaceView branch_ws) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_equal(leaves_tree.size(), n_leaves * kLeavesStride,
                             "leaves_tree size mismatch");
    error_check::check_greater_equal(leaves_branch.size(),
                                     n_leaves * branch_max * kLeavesStride,
                                     "leaves_branch size mismatch");
    error_check::check_greater_equal(leaves_origins.size(),
                                     n_leaves * branch_max,
                                     "leaves_origins size mismatch");
    const auto [_, ts] = coord_cur; // scale_cur
    const double ts2   = ts * ts;
    const double ts3   = ts2 * ts;
    const double ts4   = ts2 * ts2;
    const auto nbins_d = static_cast<double>(nbins);
    const double dphi  = eta / nbins_d;

    const double d4_range      = param_limits[0].max - param_limits[0].min;
    const double d3_range      = param_limits[1].max - param_limits[1].min;
    const double d2_range      = param_limits[2].max - param_limits[2].min;
    const double f0_range      = param_limits[3].max - param_limits[3].min;
    const double alpha_4_range = d4_range * ts4 / 192.0;
    const double alpha_3_range = d3_range * ts3 / 24.0;
    const double alpha_2_range =
        (d2_range * ts2 / 4.0) + (d4_range * ts4 / 48.0);

    // Use leaves_branch memory as workspace.
    const SizeType workspace_size      = leaves_branch.size();
    const SizeType single_batch_params = n_leaves * kParams;

    // Get spans from workspace
    std::span<double> dparam_new =
        leaves_branch.subspan(0, single_batch_params);
    std::span<double> shift_bins =
        leaves_branch.subspan(single_batch_params, single_batch_params);
    const auto workspace_acquired_size = (single_batch_params * 2);
    error_check::check_less_equal(workspace_acquired_size, workspace_size,
                                  "workspace size mismatch");
    // Transform the parameters to coord_cur domain
    std::span<SizeType> indices_tree = leaves_origins.subspan(0, n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        indices_tree[i] = i;
    }
    poly_chebyshev_transform_accel_batch(leaves_tree, indices_tree, coord_cur,
                                         coord_prev, n_leaves);

    const double* __restrict__ leaves_tree_ptr = leaves_tree.data();
    SizeType* __restrict__ leaves_origins_ptr  = leaves_origins.data();
    double* __restrict__ leaves_branch_ptr     = leaves_branch.data();
    double* __restrict__ dparam_new_ptr        = dparam_new.data();
    double* __restrict__ shift_bins_ptr        = shift_bins.data();

    // Step + Shift
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kLeavesStride;
        const SizeType fb = i * kParams;

        const auto alpha_4_sig_cur = leaves_tree_ptr[lo + 1];
        const auto alpha_3_sig_cur = leaves_tree_ptr[lo + 3];
        const auto alpha_2_sig_cur = leaves_tree_ptr[lo + 5];
        const auto alpha_1_sig_cur = leaves_tree_ptr[lo + 7];
        const auto f0              = leaves_tree_ptr[lo + 10];

        // Base step in Chebyshev units: dphi * C / f0 (same for all alpha_k)
        const auto dfactor   = utils::kCval / f0;
        const auto base_step = dphi * dfactor;
        // Compute steps
        const auto alpha_1_range =
            (dfactor * f0_range * ts) + (d3_range * ts3 / 8.0);
        const auto alpha_4_sig_new = std::min(base_step, alpha_4_range);
        const auto alpha_3_sig_new = std::min(base_step, alpha_3_range);
        const auto alpha_2_sig_new = std::min(base_step, alpha_2_range);
        const auto alpha_1_sig_new = std::min(base_step, alpha_1_range);
        dparam_new_ptr[fb + 0]     = alpha_4_sig_new;
        dparam_new_ptr[fb + 1]     = alpha_3_sig_new;
        dparam_new_ptr[fb + 2]     = alpha_2_sig_new;
        dparam_new_ptr[fb + 3]     = alpha_1_sig_new;

        // Compute shift bins
        const double inv_base = nbins_d / dfactor;
        shift_bins_ptr[fb + 0] =
            std::abs(alpha_4_sig_cur - alpha_4_sig_new) * inv_base;
        shift_bins_ptr[fb + 1] =
            std::abs(alpha_3_sig_cur - alpha_3_sig_new) * inv_base;
        shift_bins_ptr[fb + 2] =
            std::abs(alpha_2_sig_cur - alpha_2_sig_new) * inv_base;
        shift_bins_ptr[fb + 3] =
            std::abs(alpha_1_sig_cur - alpha_1_sig_new) * inv_base;
    }

    // Early Exit: Check if any leaf needs branching
    bool any_branching = false;
    for (SizeType i = 0; i < n_leaves * kParams; ++i) {
        if (shift_bins_ptr[i] >= (eta - utils::kEps)) {
            any_branching = true;
            break;
        }
    }
    // Fast path: no branching at all
    if (!any_branching) {
        std::memcpy(leaves_branch_ptr, leaves_tree_ptr,
                    n_leaves * kLeavesStride * sizeof(double));
        for (SizeType i = 0; i < n_leaves; ++i) {
            leaves_origins_ptr[i] = i;
        }
        return n_leaves;
    }

    // Loop 2: branching
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo     = i * kLeavesStride;
        const SizeType fb     = i * kParams;
        const auto d4_cur     = leaves_tree_ptr[lo + 0];
        const auto d4_sig_cur = leaves_tree_ptr[lo + 1];
        const auto d3_cur     = leaves_tree_ptr[lo + 2];
        const auto d3_sig_cur = leaves_tree_ptr[lo + 3];
        const auto d2_cur     = leaves_tree_ptr[lo + 4];
        const auto d2_sig_cur = leaves_tree_ptr[lo + 5];
        const auto d1_cur     = leaves_tree_ptr[lo + 6];
        const auto d1_sig_cur = leaves_tree_ptr[lo + 7];

        // Branch d4-d1 parameters
        psr_utils::branch_one_param_padded(
            0, d4_cur, d4_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            1, d3_cur, d3_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            2, d2_cur, d2_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
        psr_utils::branch_one_param_padded(
            3, d1_cur, d1_sig_cur, eta, shift_bins_ptr, dparam_new_ptr,
            branch_ws.scratch_params, branch_ws.scratch_dparams,
            branch_ws.scratch_counts, fb, branch_max);
    }

    // Fill leaves_origins
    SizeType out_leaves = 0;
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo            = i * kLeavesStride;
        const SizeType fb            = i * kParams;
        const SizeType n_d4_branches = branch_ws.scratch_counts[fb + 0];
        const SizeType n_d3_branches = branch_ws.scratch_counts[fb + 1];
        const SizeType n_d2_branches = branch_ws.scratch_counts[fb + 2];
        const SizeType n_d1_branches = branch_ws.scratch_counts[fb + 3];
        const SizeType d4_offset     = (fb + 0) * branch_max;
        const SizeType d3_offset     = (fb + 1) * branch_max;
        const SizeType d2_offset     = (fb + 2) * branch_max;
        const SizeType d1_offset     = (fb + 3) * branch_max;

        for (SizeType a = 0; a < n_d4_branches; ++a) {
            for (SizeType b = 0; b < n_d3_branches; ++b) {
                for (SizeType c = 0; c < n_d2_branches; ++c) {
                    for (SizeType d = 0; d < n_d1_branches; ++d) {
                        const SizeType bo = out_leaves * kLeavesStride;
                        leaves_branch_ptr[bo + 0] =
                            branch_ws.scratch_params[d4_offset + a];
                        leaves_branch_ptr[bo + 1] =
                            branch_ws.scratch_dparams[fb + 0];
                        leaves_branch_ptr[bo + 2] =
                            branch_ws.scratch_params[d3_offset + b];
                        leaves_branch_ptr[bo + 3] =
                            branch_ws.scratch_dparams[fb + 1];
                        leaves_branch_ptr[bo + 4] =
                            branch_ws.scratch_params[d2_offset + c];
                        leaves_branch_ptr[bo + 5] =
                            branch_ws.scratch_dparams[fb + 2];
                        leaves_branch_ptr[bo + 6] =
                            branch_ws.scratch_params[d1_offset + d];
                        leaves_branch_ptr[bo + 7] =
                            branch_ws.scratch_dparams[fb + 3];
                        // Fill d0 and f0 directly from leaves_tree
                        std::memcpy(leaves_branch_ptr + bo + 8,
                                    leaves_tree_ptr + lo + 8,
                                    4 * sizeof(double));

                        leaves_origins_ptr[out_leaves] = i;
                        ++out_leaves;
                    }
                }
            }
        }
    }

    return out_leaves;
}

void poly_chebyshev_resolve_accel_batch(
    std::span<const double> leaves_tree,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    std::span<const ParamLimit> param_limits,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_accel_init,
    SizeType n_freq_init,
    SizeType nbins,
    SizeType n_leaves) {
    constexpr SizeType kParams       = 2;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[0];
    const auto& lim_freq  = param_limits[1];

    // Pre-compute constants
    const auto dt_add  = t0_add - t0_cur;
    const auto dt_init = t0_init - t0_cur;
    const auto dt      = dt_add - dt_init;
    const auto dt2     = ((dt_add * dt_add) - (dt_init * dt_init));
    const auto inv_ts  = 1.0 / scale_cur;
    const auto inv_ts2 = inv_ts * inv_ts;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo      = i * kLeavesStride;
        const auto alpha_2 = leaves_tree[lo + 0];
        const auto alpha_1 = leaves_tree[lo + 2];
        const auto f0      = leaves_tree[lo + 6];
        const auto a_new   = 2.0 * alpha_2 * inv_ts2;
        const auto delta_v = 4.0 * alpha_2 * inv_ts2 * dt;
        const auto delta_d =
            (alpha_1 * inv_ts * dt) + (2.0 * alpha_2 * inv_ts2 * dt2);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
        const auto delay_rel = delta_d * utils::kInvCval;

        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
        phase_shift[i]   = psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
    }
}

void poly_chebyshev_resolve_jerk_batch(std::span<const double> leaves_tree,
                                       std::span<SizeType> param_indices,
                                       std::span<float> phase_shift,
                                       std::span<const ParamLimit> param_limits,
                                       std::pair<double, double> coord_add,
                                       std::pair<double, double> coord_cur,
                                       std::pair<double, double> coord_init,
                                       SizeType n_accel_init,
                                       SizeType n_freq_init,
                                       SizeType nbins,
                                       SizeType n_leaves) {
    constexpr SizeType kParams       = 3;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[1];
    const auto& lim_freq  = param_limits[2];

    // Pre-compute constants to avoid repeated calculations
    const auto dt_add   = t0_add - t0_cur;
    const auto dt_init  = t0_init - t0_cur;
    const auto dt2_add  = dt_add * dt_add;
    const auto dt2_init = dt_init * dt_init;
    const auto dt       = dt_add - dt_init;
    const auto dt2      = dt2_add - dt2_init;
    const auto dt3      = (dt2_add * dt_add) - (dt2_init * dt_init);
    const auto inv_ts   = 1.0 / scale_cur;
    const auto inv_ts2  = inv_ts * inv_ts;
    const auto inv_ts3  = inv_ts2 * inv_ts;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo      = i * kLeavesStride;
        const auto alpha_3 = leaves_tree[lo + 0];
        const auto alpha_2 = leaves_tree[lo + 2];
        const auto alpha_1 = leaves_tree[lo + 4];
        const auto f0      = leaves_tree[lo + 8];
        const auto a_new =
            ((2.0 * alpha_2) + (12.0 * alpha_3 * (dt_add * inv_ts))) * inv_ts2;
        const auto delta_v = ((4.0 * alpha_2 * inv_ts2) * dt) +
                             ((12.0 * alpha_3 * inv_ts3) * dt2);
        const auto delta_d = ((alpha_1 - (3.0 * alpha_3)) * inv_ts * dt) +
                             ((2.0 * alpha_2 * inv_ts2) * dt2) +
                             ((4.0 * alpha_3 * inv_ts3) * dt3);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
        const auto delay_rel = delta_d * utils::kInvCval;

        // Calculate relative phase
        phase_shift[i] = psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
    }
}

void poly_chebyshev_resolve_snap_batch(std::span<const double> leaves_tree,
                                       std::span<SizeType> param_indices,
                                       std::span<float> phase_shift,
                                       std::span<const ParamLimit> param_limits,
                                       std::pair<double, double> coord_add,
                                       std::pair<double, double> coord_cur,
                                       std::pair<double, double> coord_init,
                                       SizeType n_accel_init,
                                       SizeType n_freq_init,
                                       SizeType nbins,
                                       SizeType n_leaves) {
    constexpr SizeType kParams       = 4;
    constexpr SizeType kParamStride  = 2;
    constexpr SizeType kLeavesStride = (kParams + 2) * kParamStride;

    error_check::check_greater_equal(leaves_tree.size(),
                                     n_leaves * kLeavesStride,
                                     "leaves_tree size mismatch");
    error_check::check_greater_equal(param_indices.size(), n_leaves,
                                     "param_indices size mismatch");
    error_check::check_greater_equal(phase_shift.size(), n_leaves,
                                     "phase_shift size mismatch");
    error_check::check_equal(param_limits.size(), kParams,
                             "param_limits size mismatch");

    const auto [t0_cur, scale_cur]   = coord_cur;
    const auto [t0_init, scale_init] = coord_init;
    const auto [t0_add, scale_add]   = coord_add;

    const auto& lim_accel = param_limits[2];
    const auto& lim_freq  = param_limits[3];

    // Pre-compute constants to avoid repeated calculations
    const auto dt_add   = t0_add - t0_cur;
    const auto dt_init  = t0_init - t0_cur;
    const auto dt2_add  = dt_add * dt_add;
    const auto dt2_init = dt_init * dt_init;
    const auto dt       = dt_add - dt_init;
    const auto dt2      = dt2_add - dt2_init;
    const auto dt3      = ((dt2_add * dt_add) - (dt2_init * dt_init));
    const auto dt4      = ((dt2_add * dt2_add) - (dt2_init * dt2_init));
    const auto inv_ts   = 1.0 / scale_cur;
    const auto inv_ts2  = inv_ts * inv_ts;
    const auto inv_ts3  = inv_ts2 * inv_ts;
    const auto inv_ts4  = inv_ts3 * inv_ts;

    for (SizeType i = 0; i < n_leaves; ++i) {
        const auto lo      = i * kLeavesStride;
        const auto alpha_4 = leaves_tree[lo + 0];
        const auto alpha_3 = leaves_tree[lo + 2];
        const auto alpha_2 = leaves_tree[lo + 4];
        const auto alpha_1 = leaves_tree[lo + 6];
        const auto f0      = leaves_tree[lo + 10];
        const auto a_new =
            ((2.0 * alpha_2) + (12.0 * alpha_3 * (dt_add * inv_ts)) +
             (48.0 * alpha_4 * (dt_add * inv_ts) * (dt_add * inv_ts)) -
             (8.0 * alpha_4)) *
            inv_ts2;
        const auto delta_v =
            (((4.0 * alpha_2) - (16.0 * alpha_4)) * inv_ts2 * dt) +
            ((12.0 * alpha_3 * inv_ts3) * dt2) +
            ((32.0 * alpha_4 * inv_ts4) * dt3);

        const auto delta_d =
            ((alpha_1 - (3.0 * alpha_3)) * inv_ts * dt) +
            (((2.0 * alpha_2) - (8.0 * alpha_4)) * inv_ts2 * dt2) +
            ((4.0 * alpha_3 * inv_ts3) * dt3) +
            ((8.0 * alpha_4 * inv_ts4) * dt4);
        // Calculates new frequency based on the first-order Doppler
        // approximation:
        const auto f_new     = f0 * (1.0 - (delta_v * utils::kInvCval));
        const auto delay_rel = delta_d * utils::kInvCval;

        // Calculate relative phase
        phase_shift[i] = psr_utils::get_phase_idx(dt, f0, nbins, delay_rel);
        // Find nearest grid indices
        const auto idx_a = psr_utils::get_nearest_idx_analytical(
            a_new, lim_accel, n_accel_init);
        const auto idx_f =
            psr_utils::get_nearest_idx_analytical(f_new, lim_freq, n_freq_init);
        param_indices[i] = (idx_a * n_freq_init) + idx_f;
    }
}

template <SizeType NPARAMS>
SizeType
poly_chebyshev_branch_batch_impl(std::span<double> leaves_tree,
                                 std::span<double> leaves_branch,
                                 std::span<SizeType> leaves_origins,
                                 std::pair<double, double> coord_cur,
                                 std::pair<double, double> coord_prev,
                                 SizeType nbins,
                                 double eta,
                                 std::span<const ParamLimit> param_limits,
                                 SizeType branch_max,
                                 SizeType n_leaves,
                                 utils::BranchingWorkspaceView branch_ws) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Chebyshev order");
    if constexpr (NPARAMS == 2) {
        return poly_chebyshev_branch_accel_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, coord_prev,
            nbins, eta, param_limits, branch_max, n_leaves, branch_ws);
    } else if constexpr (NPARAMS == 3) {
        return poly_chebyshev_branch_jerk_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, coord_prev,
            nbins, eta, param_limits, branch_max, n_leaves, branch_ws);
    } else if constexpr (NPARAMS == 4) {
        return poly_chebyshev_branch_snap_batch(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, coord_prev,
            nbins, eta, param_limits, branch_max, n_leaves, branch_ws);
    }
}

template <SizeType NPARAMS>
void poly_chebyshev_resolve_batch_impl(std::span<const double> leaves_tree,
                                       std::span<SizeType> param_indices,
                                       std::span<float> phase_shift,
                                       std::span<const ParamLimit> param_limits,
                                       std::pair<double, double> coord_add,
                                       std::pair<double, double> coord_cur,
                                       std::pair<double, double> coord_init,
                                       SizeType n_accel_init,
                                       SizeType n_freq_init,
                                       SizeType nbins,
                                       SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Chebyshev order");
    if constexpr (NPARAMS == 2) {
        poly_chebyshev_resolve_accel_batch(
            leaves_tree, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_chebyshev_resolve_jerk_batch(
            leaves_tree, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_chebyshev_resolve_snap_batch(
            leaves_tree, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
    }
}

template <SizeType NPARAMS>
void poly_chebyshev_transform_batch_impl(std::span<double> leaves_tree,
                                         std::span<SizeType> indices_tree,
                                         std::pair<double, double> coord_next,
                                         std::pair<double, double> coord_cur,
                                         SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Chebyshev order");
    if constexpr (NPARAMS == 2) {
        poly_chebyshev_transform_accel_batch(leaves_tree, indices_tree,
                                             coord_next, coord_cur, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_chebyshev_transform_jerk_batch(leaves_tree, indices_tree,
                                            coord_next, coord_cur, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_chebyshev_transform_snap_batch(leaves_tree, indices_tree,
                                            coord_next, coord_cur, n_leaves);
    }
}

template <SizeType NPARAMS>
void poly_cheby_to_taylor_batch_impl(std::span<double> leaves_tree,
                                     std::pair<double, double> coord_report,
                                     SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Chebyshev order");
    if constexpr (NPARAMS == 2) {
        poly_cheby_to_taylor_accel_batch(leaves_tree, coord_report, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_cheby_to_taylor_jerk_batch(leaves_tree, coord_report, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_cheby_to_taylor_snap_batch(leaves_tree, coord_report, n_leaves);
    }
}

template <SizeType NPARAMS>
void poly_taylor_to_cheby_batch_impl(std::span<double> leaves_tree,
                                     std::pair<double, double> coord_init,
                                     SizeType n_leaves) {
    static_assert(NPARAMS == 2 || NPARAMS == 3 || NPARAMS == 4,
                  "Unsupported Taylor order");
    if constexpr (NPARAMS == 2) {
        poly_taylor_to_cheby_accel_batch(leaves_tree, coord_init, n_leaves);
    } else if constexpr (NPARAMS == 3) {
        poly_taylor_to_cheby_jerk_batch(leaves_tree, coord_init, n_leaves);
    } else if constexpr (NPARAMS == 4) {
        poly_taylor_to_cheby_snap_batch(leaves_tree, coord_init, n_leaves);
    }
}

} // namespace

SizeType poly_chebyshev_seed(std::span<const SizeType> param_grid_count_init,
                             std::span<const double> dparams_init,
                             std::span<const ParamLimit> param_limits,
                             std::span<double> seed_leaves,
                             std::pair<double, double> coord_init,
                             SizeType n_params) {
    constexpr SizeType kParamStride = 2U;
    error_check::check_equal(param_grid_count_init.size(), n_params,
                             "param_grid_count_init size mismatch");
    error_check::check_equal(dparams_init.size(), n_params,
                             "dparams_init size mismatch");
    error_check::check_equal(param_limits.size(), n_params,
                             "param_limits size mismatch");
    const SizeType leaves_stride = (n_params + 2) * kParamStride;
    SizeType n_leaves            = 1;
    for (const auto count : param_grid_count_init) {
        n_leaves *= count;
    }
    const auto n_accel_init = param_grid_count_init[n_params - 2];
    const auto n_freq_init  = param_grid_count_init[n_params - 1];
    const auto& lim_accel   = param_limits[n_params - 2];
    const auto& lim_freq    = param_limits[n_params - 1];
    const auto d_freq_cur   = dparams_init[n_params - 1];
    error_check::check_equal(n_leaves, n_accel_init * n_freq_init,
                             "n_leaves mismatch");

    // Create parameter sets tensor: (n_param_sets, poly_order + 2, 2)
    error_check::check_greater_equal(seed_leaves.size(),
                                     n_leaves * leaves_stride,
                                     "seed_leaves size mismatch");
    SizeType leaf_idx = 0;
    for (SizeType accel_idx = 0; accel_idx < n_accel_init; ++accel_idx) {
        const auto accel_cur =
            psr_utils::get_param_val_at_idx(lim_accel, n_accel_init, accel_idx);
        for (SizeType freq_idx = 0; freq_idx < n_freq_init; ++freq_idx) {
            const auto freq_cur = psr_utils::get_param_val_at_idx(
                lim_freq, n_freq_init, freq_idx);
            const auto lo = leaf_idx * leaves_stride;
            // Copy till d2 (acceleration)
            for (SizeType j = 0; j < n_params - 1; ++j) {
                seed_leaves[lo + (j * kParamStride) + 0] = 0;
                seed_leaves[lo + (j * kParamStride) + 1] = dparams_init[j];
            }
            seed_leaves[lo + ((n_params - 2) * kParamStride) + 0] = accel_cur;
            // Update frequency to velocity
            // f = f0(1 - v / C) => dv = -(C/f0) * df
            seed_leaves[lo + ((n_params - 1) * kParamStride) + 0] = 0;
            seed_leaves[lo + ((n_params - 1) * kParamStride) + 1] =
                d_freq_cur * (utils::kCval / freq_cur);
            // intialize d0 (measure from t=t_init) and store f0
            seed_leaves[lo + ((n_params + 0) * kParamStride) + 0] = 0;
            seed_leaves[lo + ((n_params + 0) * kParamStride) + 1] = 0;
            seed_leaves[lo + ((n_params + 1) * kParamStride) + 0] = freq_cur;
            // Store basis flag (0: Polynomial, 1: Physical)
            seed_leaves[lo + ((n_params + 1) * kParamStride) + 1] = 0;
            ++leaf_idx;
        }
    }
    error_check::check_equal(leaf_idx, n_leaves, "n_leaves mismatch");

    // Convert to Chebyshev basis
    auto dispatch = [&]<SizeType N>() {
        return poly_taylor_to_cheby_batch_impl<N>(seed_leaves, coord_init,
                                                  n_leaves);
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
        throw std::invalid_argument("Unsupported Taylor order");
    }
    return n_leaves;
}

SizeType poly_chebyshev_branch_batch(std::span<double> leaves_tree,
                                     std::span<double> leaves_branch,
                                     std::span<SizeType> leaves_origins,
                                     std::pair<double, double> coord_cur,
                                     std::pair<double, double> coord_prev,
                                     SizeType nbins,
                                     double eta,
                                     std::span<const ParamLimit> param_limits,
                                     SizeType branch_max,
                                     SizeType n_leaves,
                                     SizeType n_params,
                                     utils::BranchingWorkspaceView branch_ws) {

    auto dispatch = [&]<SizeType N>() {
        return poly_chebyshev_branch_batch_impl<N>(
            leaves_tree, leaves_branch, leaves_origins, coord_cur, coord_prev,
            nbins, eta, param_limits, branch_max, n_leaves, branch_ws);
    };
    switch (n_params) {
    case 2:
        return dispatch.template operator()<2>();
        break;
    case 3:
        return dispatch.template operator()<3>();
        break;
    case 4:
        return dispatch.template operator()<4>();
        break;
    default:
        throw std::invalid_argument("Unsupported Taylor order");
    }
}

void poly_chebyshev_resolve_batch(std::span<const double> leaves_branch,
                                  std::span<SizeType> param_indices,
                                  std::span<float> phase_shift,
                                  std::span<const ParamLimit> param_limits,
                                  std::pair<double, double> coord_add,
                                  std::pair<double, double> coord_cur,
                                  std::pair<double, double> coord_init,
                                  SizeType n_accel_init,
                                  SizeType n_freq_init,
                                  SizeType nbins,
                                  SizeType n_leaves,
                                  SizeType n_params) {
    auto dispatch = [&]<SizeType N>() {
        return poly_chebyshev_resolve_batch_impl<N>(
            leaves_branch, param_indices, phase_shift, param_limits, coord_add,
            coord_cur, coord_init, n_accel_init, n_freq_init, nbins, n_leaves);
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
        throw std::invalid_argument("Unsupported Taylor order");
    }
}

void poly_chebyshev_transform_batch(std::span<double> leaves_tree,
                                    std::span<SizeType> indices_tree,
                                    std::pair<double, double> coord_next,
                                    std::pair<double, double> coord_cur,
                                    SizeType n_leaves,
                                    SizeType n_params) {
    auto dispatch = [&]<SizeType N>() {
        return poly_chebyshev_transform_batch_impl<N>(
            leaves_tree, indices_tree, coord_next, coord_cur, n_leaves);
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
        throw std::invalid_argument("Unsupported Chebyshev order");
    }
}

void poly_chebyshev_report_batch(std::span<double> leaves_tree,
                                 std::pair<double, double> coord_report,
                                 SizeType n_leaves,
                                 SizeType n_params) {
    auto dispatch = [&]<SizeType N>() {
        return poly_cheby_to_taylor_batch_impl<N>(leaves_tree, coord_report,
                                                  n_leaves);
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
        throw std::invalid_argument("Unsupported Chebyshev order");
    }
    poly_taylor_report_batch(leaves_tree, coord_report, n_leaves, n_params);
}

std::vector<double> generate_bp_poly_chebyshev_approx(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    std::span<const ParamLimit> param_limits,
    double tseg_ffa,
    SizeType nsegments,
    SizeType nbins,
    double eta,
    SizeType ref_seg,
    IndexType isuggest,
    SizeType branch_max) {
    error_check::check_equal(param_grid_count_init.size(), param_limits.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_grid_count_init.size(), param_limits.size(),
        "param_grid_count_init and param_limits must have the same size");
    std::vector<double> branching_pattern(nsegments - 1);
    const auto n_params      = param_grid_count_init.size();
    const auto leaves_stride = (n_params + 2) * 2;
    std::vector<double> branch_leaves(branch_max * leaves_stride);
    std::vector<double> leaf_data(leaves_stride);
    std::vector<SizeType> leaf_origins(branch_max);
    utils::BranchingWorkspace branch_ws(1, branch_max, n_params);
    auto branch_ws_view = branch_ws.get_view();

    psr_utils::MiddleOutScheme snail_scheme(nsegments, ref_seg, tseg_ffa);
    const auto coord_init = snail_scheme.get_coord(0);
    SizeType n_leaves     = 1;
    for (const auto count : param_grid_count_init) {
        n_leaves *= count;
    }
    std::vector<double> seed_leaves(n_leaves * leaves_stride);
    const auto n_leaves_seed =
        poly_chebyshev_seed(param_grid_count_init, dparams_init, param_limits,
                            seed_leaves, coord_init, n_params);
    error_check::check_equal(n_leaves_seed, n_leaves, "n_leaves mismatch");
    // Get isuggest-th leaf
    if (isuggest < 0) { // Negative index
        isuggest = static_cast<IndexType>(n_leaves + isuggest);
    }
    error_check::check_greater_equal(isuggest, 0,
                                     "isuggest must be non-negative");
    error_check::check_less(isuggest, n_leaves,
                            "isuggest must be less than n_leaves");
    // Copy isuggest-th leaf to leaf_data
    auto leaf = std::span(seed_leaves)
                    .subspan((leaves_stride * isuggest), leaves_stride);
    std::ranges::copy(leaf, leaf_data.begin());
    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_prev = snail_scheme.get_coord(prune_level - 1);
        const auto coord_next = snail_scheme.get_coord(prune_level);
        const auto coord_cur  = snail_scheme.get_current_coord(prune_level);
        const auto n_leaves_branch = poly_chebyshev_branch_batch(
            leaf_data, branch_leaves, leaf_origins, coord_cur, coord_prev,
            nbins, eta, param_limits, branch_max, 1, n_params, branch_ws_view);
        auto leaves_span =
            std::span(branch_leaves).first(n_leaves_branch * leaves_stride);
        std::vector<SizeType> indices_branch(n_leaves_branch);
        std::iota(indices_branch.begin(), indices_branch.end(), 0U);
        branching_pattern[prune_level - 1] =
            static_cast<double>(n_leaves_branch);
        poly_chebyshev_transform_batch(leaves_span, indices_branch, coord_next,
                                       coord_cur, n_leaves_branch, n_params);
        // Copy first leaf to leaf_data
        auto first_leaf_span = std::span(branch_leaves).first(leaves_stride);
        std::ranges::copy(first_leaf_span, leaf_data.begin());
    }
    // Check if any branches is truncated due to branch_max
    if (std::ranges::any_of(branching_pattern, [branch_max](double value) {
            return static_cast<SizeType>(value) == branch_max;
        })) {
        throw std::runtime_error("Branching pattern is truncated due to "
                                 "branch_max. Increase branch_max.");
    }
    return branching_pattern;
}

std::vector<double>
generate_bp_poly_chebyshev(std::span<const std::vector<double>> param_arr,
                           std::span<const double> dparams,
                           std::span<const ParamLimit> param_limits,
                           double tseg_ffa,
                           SizeType nsegments,
                           SizeType nbins,
                           double eta,
                           SizeType ref_seg) {
    error_check::check_equal(param_arr.size(), dparams.size(),
                             "param_arr and dparams must have the same size");
    error_check::check_equal(
        param_arr.size(), param_limits.size(),
        "param_arr and param_limits must have the same size");
    const auto n_params  = dparams.size();
    const auto& f0_batch = param_arr.back(); // Last array is frequency
    const auto n_freqs   = f0_batch.size();  // Number of frequency bins

    // Snail Scheme
    psr_utils::MiddleOutScheme snail_scheme(nsegments, ref_seg, tseg_ffa);
    std::vector<double> weights(n_freqs, 1.0);
    std::vector<double> branching_pattern(nsegments - 1);

    // Initialize dparam_cur_batch - each frequency gets the same dparams
    std::vector<double> dparam_cur_batch(n_freqs * n_params);
    for (SizeType i = 0; i < n_freqs; ++i) {
        std::ranges::copy(dparams, dparam_cur_batch.begin() +
                                       static_cast<IndexType>(i * n_params));
    }
    // f = f0(1 - v / C) => dv = -(C/f0) * df
    for (SizeType i = 0; i < n_freqs; ++i) {
        dparam_cur_batch[(i * n_params) + n_params - 1] =
            dparam_cur_batch[(i * n_params) + n_params - 1] *
            (utils::kCval / f0_batch[i]);
    }

    std::vector<double> dparam_new_batch(n_freqs * n_params, 0.0);
    std::vector<double> shift_bins_batch(n_freqs * n_params, 0.0);
    std::vector<double> dparam_cur_next(n_freqs * n_params, 0.0);
    std::vector<double> n_branches(n_freqs, 1);

    for (SizeType prune_level = 1; prune_level < nsegments; ++prune_level) {
        const auto coord_prev     = snail_scheme.get_coord(prune_level - 1);
        const auto coord_next     = snail_scheme.get_coord(prune_level);
        const auto coord_cur      = snail_scheme.get_current_coord(prune_level);
        const auto [_, scale_cur] = coord_cur;

        // Transform the parameters to coord_cur domain
        transforms::shift_cheb_errors_batch(dparam_cur_batch, coord_cur.second,
                                            coord_prev.second, n_freqs,
                                            n_params);

        // Calculate optimal parameter steps and shift bins
        psr_utils::poly_cheb_step_vec_limited(n_params, scale_cur, nbins, eta,
                                              f0_batch, param_limits,
                                              dparam_new_batch);
        psr_utils::poly_cheb_shift_vec(dparam_cur_batch, dparam_new_batch,
                                       nbins, f0_batch, shift_bins_batch,
                                       n_freqs, n_params);

        std::ranges::fill(n_branches, 1.0);
        // Determine branching needs
        for (SizeType i = 0; i < n_freqs; ++i) {
            for (SizeType j = 0; j < n_params; ++j) {
                const auto idx = (i * n_params) + j;
                if (shift_bins_batch[idx] < (eta - utils::kEps)) {
                    dparam_cur_next[idx] = dparam_cur_batch[idx];
                    continue;
                }
                const auto ratio      = (dparam_cur_batch[idx] + utils::kEps) /
                                        (dparam_new_batch[idx]);
                const auto num_points = std::max(
                    1, static_cast<int>(std::ceil(ratio - utils::kEps)));
                n_branches[i] *= static_cast<double>(num_points);
                dparam_cur_next[idx] =
                    dparam_cur_batch[idx] / static_cast<double>(num_points);
            }
        }

        // Compute average branching factor and update weights
        double children = 0.0;
        double parents  = 0.0;
        for (SizeType i = 0; i < n_freqs; ++i) {
            children += weights[i] * n_branches[i];
            parents += weights[i];
            weights[i] *= n_branches[i];
        }
        branching_pattern[prune_level - 1] = children / parents;

        // Transform dparams to the next segment
        transforms::shift_cheb_errors_batch(dparam_cur_next, coord_next.second,
                                            coord_cur.second, n_freqs,
                                            n_params);
        std::ranges::copy(dparam_cur_next, dparam_cur_batch.begin());
    }
    return branching_pattern;
}

} // namespace loki::core