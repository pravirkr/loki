#pragma once

/**
 * @file taylor_ffa_cuda.cuh
 * @brief Device implementations declared alongside core/taylor_ffa.hpp (CUDA
 * only).
 */

#include <cstdint>

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "lib/core/taylor_ffa.hpp"
#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"

namespace loki::core {

void ffa_taylor_resolve_freq_batch_cuda(
    cuda::std::span<const uint32_t> param_arr_count,
    cuda::std::span<const uint32_t> ncoords_offsets,
    cuda::std::span<const ParamLimit> param_limits,
    coord::FFACoordFreqDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    double tseg_brute,
    SizeType nbins,
    cudaStream_t stream);

void ffa_taylor_resolve_poly_batch_cuda(
    cuda::std::span<const uint32_t> param_arr_count,
    cuda::std::span<const uint32_t> ncoords_offsets,
    cuda::std::span<const ParamLimit> param_limits,
    coord::FFACoordDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    SizeType latter,
    double tseg_brute,
    SizeType nbins,
    SizeType n_params,
    cudaStream_t stream);

/// Per-(level, accel row) values of the accel resolve that do not depend
/// on the frequency: the parent accel index (times n_freq_prev), and per
/// half (tail, head) the factor 1 - v / c and the time t - d / c.
struct FFAAccelResolveRow {
    uint32_t idx_a_base;
    uint32_t pad;
    double w[2];
    double u[2];
};

/// One level's view of the accel resolve tables (see
/// resolve_accel_coord in ffa_resolve_device.cuh).
struct FFAAccelResolveLevel {
    const FFAAccelResolveRow* rows; // this level's rows (n_accel_cur)
    const double* cols;             // this level's freqs (n_freq_cur)
    uint32_t n_accel_cur;
    uint32_t n_freq_cur;
    uint32_t n_freq_prev;
    uint32_t nbins;
    double f_min;
    double f_max;
};

/**
 * @brief Accel (n_params = 2) resolve of tail and head in one pass, from
 * per-level row and column tables: bit-identical to two
 * ffa_taylor_resolve_poly_batch_cuda calls (latter 0 and 1) with a fraction
 * of the FP64 work.
 *
 * @p rows needs sum over levels of n_accel and @p cols sum of n_freq
 * entries; @p row_col_offsets holds the per-level row offsets then the
 * per-level column offsets (2 * (n_levels + 1) values, host-computed by
 * accel_resolve_table_offsets). Coordinates past @p ncoords_resolve (e.g.
 * the final level, which the fused sweep resolves on the fly) are left
 * unwritten; the tables cover every level.
 */
void ffa_taylor_resolve_accel_tables_cuda(
    cuda::std::span<const uint32_t> param_arr_count,
    cuda::std::span<const uint32_t> ncoords_offsets,
    cuda::std::span<const ParamLimit> param_limits,
    cuda::std::span<const uint32_t> row_col_offsets,
    FFAAccelResolveRow* rows,
    double* cols,
    coord::FFACoordDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    SizeType total_rows,
    SizeType total_cols,
    double tseg_brute,
    SizeType nbins,
    cudaStream_t stream,
    SizeType ncoords_resolve = SIZE_MAX);

} // namespace loki::core
