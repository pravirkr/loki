#include <algorithm>

#include <cuda/atomic>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ffa_resolve_device.cuh"
#include "lib/cuda/kernel_utils.cuh"
#include "lib/cuda/taylor_cuda.cuh"
#include "lib/detail/utils.hpp"

namespace loki::core {

namespace {

template <int LATTER>
__device__ __forceinline__ void ffa_taylor_resolve_accel_batch_device(
    const uint32_t* __restrict__ param_arr_count,
    const uint32_t* __restrict__ ncoords_offsets,
    const ParamLimit* __restrict__ param_limits,
    coord::FFACoordDPtrs coords_ptrs,
    uint32_t n_levels,
    uint32_t ncoords_total,
    double tseg_brute,
    uint32_t nbins) {
    constexpr uint32_t kParams = 2;
    const uint32_t tid         = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= ncoords_total) {
        return;
    }

    // Find which level this tid belongs to
    const uint32_t i_level =
        utils::find_ffa_level(ncoords_offsets, tid, n_levels);
    // Skip level 0
    if (i_level == 0) {
        return;
    }
    // Decompose index (f fast moving)
    // idx = a * n_freq + f
    const uint32_t local_tid   = tid - ncoords_offsets[i_level];
    const uint32_t n_accel_cur = param_arr_count[(i_level * kParams) + 0];
    const uint32_t n_freq_cur  = param_arr_count[(i_level * kParams) + 1];
    const uint32_t n_accel_prev =
        param_arr_count[((i_level - 1) * kParams) + 0];
    const uint32_t n_freq_prev = param_arr_count[((i_level - 1) * kParams) + 1];
    const uint32_t freq_idx    = local_tid % n_freq_cur;
    const uint32_t accel_idx   = (local_tid / n_freq_cur) % n_accel_cur;

    const double a_cur = utils::get_param_val_at_idx_device(
        param_limits[0].min, param_limits[0].max, n_accel_cur, accel_idx);
    const double f_cur = utils::get_param_val_at_idx_device(
        param_limits[1].min, param_limits[1].max, n_freq_cur, freq_idx);

    const double tsegment   = ldexp(tseg_brute, static_cast<int>(i_level - 1));
    const double delta_t    = (static_cast<double>(LATTER) - 0.5) * tsegment;
    const double delta_t_sq = delta_t * delta_t;
    const double v_new      = (a_cur * delta_t);
    const double d_new      = (a_cur * 0.5 * delta_t_sq);

    // Frequency-specific calculations
    const double f_new     = f_cur * (1.0 - v_new * utils::kInvCval);
    const double delay_rel = d_new * utils::kInvCval;
    const float relative_phase =
        utils::get_phase_idx_device(delta_t, f_cur, nbins, delay_rel);

    const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
        a_cur, param_limits[0].min, param_limits[0].max, n_accel_prev);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_new, param_limits[1].min, param_limits[1].max, n_freq_prev);

    const uint32_t final_idx = (idx_a * n_freq_prev) + idx_f;

    if constexpr (LATTER == 0) {
        coords_ptrs.i_tail[tid]     = final_idx;
        coords_ptrs.shift_tail[tid] = relative_phase;
    } else {
        coords_ptrs.i_head[tid]     = final_idx;
        coords_ptrs.shift_head[tid] = relative_phase;
    }
}

template <int LATTER>
__device__ __forceinline__ void ffa_taylor_resolve_jerk_batch_device(
    const uint32_t* __restrict__ param_arr_count,
    const uint32_t* __restrict__ ncoords_offsets,
    const ParamLimit* __restrict__ param_limits,
    coord::FFACoordDPtrs coords_ptrs,
    uint32_t n_levels,
    uint32_t ncoords_total,
    double tseg_brute,
    uint32_t nbins,
    uint32_t param_stride) {
    constexpr uint32_t kParams = 3;
    const uint32_t tid         = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= ncoords_total) {
        return;
    }

    // Find which level this tid belongs to
    const uint32_t i_level =
        utils::find_ffa_level(ncoords_offsets, tid, n_levels);
    // Skip level 0
    if (i_level == 0) {
        return;
    }
    // Decompose index (f fast moving)
    // idx = (j * n_accel + a) * n_freq + f
    const uint32_t po        = param_stride - kParams;
    const uint32_t local_tid = tid - ncoords_offsets[i_level];
    const uint32_t n_jerk_cur =
        param_arr_count[(i_level * param_stride) + po + 0];
    const uint32_t n_accel_cur =
        param_arr_count[(i_level * param_stride) + po + 1];
    const uint32_t n_freq_cur =
        param_arr_count[(i_level * param_stride) + po + 2];
    const uint32_t n_jerk_prev =
        param_arr_count[((i_level - 1) * param_stride) + po + 0];
    const uint32_t n_accel_prev =
        param_arr_count[((i_level - 1) * param_stride) + po + 1];
    const uint32_t n_freq_prev =
        param_arr_count[((i_level - 1) * param_stride) + po + 2];
    const uint32_t freq_idx  = local_tid % n_freq_cur;
    const uint32_t accel_idx = (local_tid / n_freq_cur) % n_accel_cur;
    const uint32_t jerk_idx  = local_tid / (n_freq_cur * n_accel_cur);

    const double j_cur = utils::get_param_val_at_idx_device(
        param_limits[po + 0].min, param_limits[po + 0].max, n_jerk_cur,
        jerk_idx);
    const double a_cur = utils::get_param_val_at_idx_device(
        param_limits[po + 1].min, param_limits[po + 1].max, n_accel_cur,
        accel_idx);
    const double f_cur = utils::get_param_val_at_idx_device(
        param_limits[po + 2].min, param_limits[po + 2].max, n_freq_cur,
        freq_idx);

    const double tsegment   = ldexp(tseg_brute, static_cast<int>(i_level - 1));
    const double delta_t    = (static_cast<double>(LATTER) - 0.5) * tsegment;
    const double delta_t_sq = delta_t * delta_t;
    const double a_new      = a_cur + (j_cur * delta_t);
    const double v_new      = (a_cur * delta_t) + (0.5 * j_cur * delta_t_sq);
    const double d_new =
        (a_cur * 0.5 * delta_t_sq) + (j_cur * delta_t_sq * delta_t / 6.0);

    // Frequency-specific calculations
    const double f_new     = f_cur * (1.0 - v_new * utils::kInvCval);
    const double delay_rel = d_new * utils::kInvCval;
    const float relative_phase =
        utils::get_phase_idx_device(delta_t, f_cur, nbins, delay_rel);

    const uint32_t idx_j = utils::get_nearest_idx_analytical_device(
        j_cur, param_limits[po + 0].min, param_limits[po + 0].max, n_jerk_prev);
    const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
        a_new, param_limits[po + 1].min, param_limits[po + 1].max,
        n_accel_prev);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_new, param_limits[po + 2].min, param_limits[po + 2].max, n_freq_prev);

    const uint32_t final_idx =
        (idx_j * n_accel_prev * n_freq_prev) + (idx_a * n_freq_prev) + idx_f;

    if constexpr (LATTER == 0) {
        coords_ptrs.i_tail[tid]     = final_idx;
        coords_ptrs.shift_tail[tid] = relative_phase;
    } else {
        coords_ptrs.i_head[tid]     = final_idx;
        coords_ptrs.shift_head[tid] = relative_phase;
    }
}

__global__ void kernel_ffa_resolve_taylor_freq_batch(
    const uint32_t* __restrict__ param_arr_count,
    const uint32_t* __restrict__ ncoords_offsets,
    const ParamLimit* __restrict__ param_limits,
    coord::FFACoordFreqDPtrs coords_ptrs,
    uint32_t n_levels,
    uint32_t ncoords_total,
    double tseg_brute,
    uint32_t nbins) {
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= ncoords_total) {
        return;
    }

    // Find which level this tid belongs to
    const uint32_t i_level =
        utils::find_ffa_level(ncoords_offsets, tid, n_levels);
    // Skip level 0
    if (i_level == 0) {
        return;
    }
    // Decompose index (f fast moving)
    // idx = f
    const uint32_t local_tid   = tid - ncoords_offsets[i_level];
    const uint32_t n_freq_cur  = param_arr_count[i_level];
    const uint32_t n_freq_prev = param_arr_count[i_level - 1];
    const uint32_t freq_idx    = local_tid;

    const double delta_t = ldexp(tseg_brute, static_cast<int>(i_level - 1));
    const double f_cur   = utils::get_param_val_at_idx_device(
        param_limits[0].min, param_limits[0].max, n_freq_cur, freq_idx);
    const float relative_phase =
        utils::get_phase_idx_device(delta_t, f_cur, nbins, 0.0);
    const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
        f_cur, param_limits[0].min, param_limits[0].max, n_freq_prev);
    coords_ptrs.idx[tid]   = idx_f;
    coords_ptrs.shift[tid] = relative_phase;
}

template <SizeType NPARAMS, int LATTER>
__global__ void kernel_ffa_resolve_taylor_poly_batch(
    const uint32_t* __restrict__ param_arr_count,
    const uint32_t* __restrict__ ncoords_offsets,
    const ParamLimit* __restrict__ param_limits,
    coord::FFACoordDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    double tseg_brute,
    SizeType nbins) {
    static_assert(NPARAMS > 1 && NPARAMS <= 5 && LATTER >= 0 && LATTER <= 1,
                  "Unsupported number of parameters or latter");

    if constexpr (NPARAMS == 2) {
        ffa_taylor_resolve_accel_batch_device<LATTER>(
            param_arr_count, ncoords_offsets, param_limits, coords_ptrs,
            n_levels, ncoords_total, tseg_brute, nbins);
    } else {
        ffa_taylor_resolve_jerk_batch_device<LATTER>(
            param_arr_count, ncoords_offsets, param_limits, coords_ptrs,
            n_levels, ncoords_total, tseg_brute, nbins, NPARAMS);
    }
}

// Row (level, accel) and column (level, freq) tables of the accel resolve.
// Each value is the same expression ffa_taylor_resolve_accel_batch_device
// evaluates, so it rounds the same way.
__global__ void
kernel_ffa_resolve_accel_tables(const uint32_t* __restrict__ param_arr_count,
                                const ParamLimit* __restrict__ param_limits,
                                const uint32_t* __restrict__ row_off,
                                const uint32_t* __restrict__ col_off,
                                uint32_t n_levels,
                                uint32_t total_rows,
                                uint32_t total_cols,
                                double tseg_brute,
                                FFAAccelResolveRow* __restrict__ rows,
                                double* __restrict__ cols) {
    constexpr uint32_t kParams = 2;
    const uint32_t tid         = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < total_rows) {
        const uint32_t i_level = utils::find_ffa_level(row_off, tid, n_levels);
        const uint32_t accel_idx   = tid - row_off[i_level];
        const uint32_t n_accel_cur = param_arr_count[(i_level * kParams) + 0];
        const uint32_t n_accel_prev =
            param_arr_count[((i_level - 1) * kParams) + 0];
        const uint32_t n_freq_prev =
            param_arr_count[((i_level - 1) * kParams) + 1];
        const double a_cur = utils::get_param_val_at_idx_device(
            param_limits[0].min, param_limits[0].max, n_accel_cur, accel_idx);
        const double tsegment =
            ldexp(tseg_brute, static_cast<int>(i_level - 1));
        FFAAccelResolveRow row{};
#pragma unroll
        for (int latter = 0; latter < 2; ++latter) {
            const double delta_t =
                (static_cast<double>(latter) - 0.5) * tsegment;
            const double delta_t_sq = delta_t * delta_t;
            const double v_new      = (a_cur * delta_t);
            const double d_new      = (a_cur * 0.5 * delta_t_sq);
            const double delay_rel  = d_new * utils::kInvCval;
            row.w[latter]           = 1.0 - v_new * utils::kInvCval;
            row.u[latter]           = delta_t - delay_rel;
        }
        const uint32_t idx_a = utils::get_nearest_idx_analytical_device(
            a_cur, param_limits[0].min, param_limits[0].max, n_accel_prev);
        row.idx_a_base = idx_a * n_freq_prev;
        rows[tid]      = row;
        return;
    }
    const uint32_t cid = tid - total_rows;
    if (cid >= total_cols) {
        return;
    }
    const uint32_t i_level    = utils::find_ffa_level(col_off, cid, n_levels);
    const uint32_t freq_idx   = cid - col_off[i_level];
    const uint32_t n_freq_cur = param_arr_count[(i_level * kParams) + 1];
    cols[cid]                 = utils::get_param_val_at_idx_device(
        param_limits[1].min, param_limits[1].max, n_freq_cur, freq_idx);
}

// Tail and head of every accel coordinate from the tables. The frequency
// shift f * (1 - v / c), the nearest-index division and the phase are per
// coordinate, as in ffa_taylor_resolve_accel_batch_device.
__global__ void
kernel_ffa_resolve_accel_fast(const uint32_t* __restrict__ param_arr_count,
                              const uint32_t* __restrict__ ncoords_offsets,
                              const ParamLimit* __restrict__ param_limits,
                              const uint32_t* __restrict__ row_off,
                              const uint32_t* __restrict__ col_off,
                              const FFAAccelResolveRow* __restrict__ rows,
                              const double* __restrict__ cols,
                              coord::FFACoordDPtrs coords_ptrs,
                              uint32_t n_levels,
                              uint32_t ncoords_total,
                              uint32_t nbins) {
    constexpr uint32_t kParams = 2;
    const uint32_t tid         = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= ncoords_total) {
        return;
    }
    const uint32_t i_level =
        utils::find_ffa_level(ncoords_offsets, tid, n_levels);
    if (i_level == 0) {
        return;
    }
    const FFAAccelResolveLevel lvl{
        .rows        = rows + row_off[i_level],
        .cols        = cols + col_off[i_level],
        .n_accel_cur = param_arr_count[(i_level * kParams) + 0],
        .n_freq_cur  = param_arr_count[(i_level * kParams) + 1],
        .n_freq_prev = param_arr_count[((i_level - 1) * kParams) + 1],
        .nbins       = nbins,
        .f_min       = param_limits[1].min,
        .f_max       = param_limits[1].max};
    const detail::AccelCoord c =
        detail::resolve_accel_coord(lvl, tid - ncoords_offsets[i_level]);
    coords_ptrs.i_tail[tid]     = c.i_tail;
    coords_ptrs.shift_tail[tid] = c.shift_tail;
    coords_ptrs.i_head[tid]     = c.i_head;
    coords_ptrs.shift_head[tid] = c.shift_head;
}

} // namespace

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
    SizeType ncoords_resolve) {
    constexpr int kBlockSize = 256;
    const uint32_t* row_off  = row_col_offsets.data();
    ncoords_total            = std::min(ncoords_total, ncoords_resolve);
    const uint32_t* col_off  = row_off + n_levels + 1;
    const SizeType n_tab     = total_rows + total_cols;
    if (n_tab > 0) {
        const dim3 grid((n_tab + kBlockSize - 1) / kBlockSize);
        cuda_utils::check_kernel_launch_params(grid, dim3(kBlockSize));
        kernel_ffa_resolve_accel_tables<<<grid, kBlockSize, 0, stream>>>(
            param_arr_count.data(), param_limits.data(), row_off, col_off,
            static_cast<uint32_t>(n_levels), static_cast<uint32_t>(total_rows),
            static_cast<uint32_t>(total_cols), tseg_brute, rows, cols);
        cuda_utils::check_last_cuda_error(
            "FFA accel resolve tables kernel launch failed");
    }
    const dim3 grid((ncoords_total + kBlockSize - 1) / kBlockSize);
    cuda_utils::check_kernel_launch_params(grid, dim3(kBlockSize));
    kernel_ffa_resolve_accel_fast<<<grid, kBlockSize, 0, stream>>>(
        param_arr_count.data(), ncoords_offsets.data(), param_limits.data(),
        row_off, col_off, rows, cols, coords_ptrs,
        static_cast<uint32_t>(n_levels), static_cast<uint32_t>(ncoords_total),
        static_cast<uint32_t>(nbins));
    cuda_utils::check_last_cuda_error("FFA accel resolve kernel launch failed");
}

void ffa_taylor_resolve_freq_batch_cuda(
    cuda::std::span<const uint32_t> param_arr_count,
    cuda::std::span<const uint32_t> ncoords_offsets,
    cuda::std::span<const ParamLimit> param_limits,
    coord::FFACoordFreqDPtrs coords_ptrs,
    SizeType n_levels,
    SizeType ncoords_total,
    double tseg_brute,
    SizeType nbins,
    cudaStream_t stream) {
    // Better occupancy than 512 for many kernels
    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((ncoords_total + kBlockSize - 1) / kBlockSize);
    cuda_utils::check_kernel_launch_params(grid, block);

    kernel_ffa_resolve_taylor_freq_batch<<<grid, block, 0, stream>>>(
        param_arr_count.data(), ncoords_offsets.data(), param_limits.data(),
        coords_ptrs, n_levels, ncoords_total, tseg_brute, nbins);
    cuda_utils::check_last_cuda_error(
        "FFA Taylor (freq) resolve kernel launch failed");
}

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
    cudaStream_t stream) {
    // Better occupancy than 512 for many kernels
    constexpr int kBlockSize = 256;
    const dim3 block(kBlockSize);
    const dim3 grid((ncoords_total + kBlockSize - 1) / kBlockSize);
    cuda_utils::check_kernel_launch_params(grid, block);

    // Two-level dispatch for complete compile-time specialization
    auto dispatch = [&]<SizeType N, int L>() {
        kernel_ffa_resolve_taylor_poly_batch<N, L><<<grid, block, 0, stream>>>(
            param_arr_count.data(), ncoords_offsets.data(), param_limits.data(),
            coords_ptrs, n_levels, ncoords_total, tseg_brute, nbins);
    };
    if (latter == 0) {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, 0>();
            break;
        case 3:
            dispatch.template operator()<3, 0>();
            break;
        case 4:
            dispatch.template operator()<4, 0>();
            break;
        case 5:
            dispatch.template operator()<5, 0>();
            break;
        default:
            throw std::invalid_argument("Unsupported n_params");
        }
    } else {
        switch (n_params) {
        case 2:
            dispatch.template operator()<2, 1>();
            break;
        case 3:
            dispatch.template operator()<3, 1>();
            break;
        case 4:
            dispatch.template operator()<4, 1>();
            break;
        case 5:
            dispatch.template operator()<5, 1>();
            break;
        default:
            throw std::invalid_argument("Unsupported n_params");
        }
    }
    cuda_utils::check_last_cuda_error(
        "FFA Taylor (poly) resolve kernel launch failed");
}

} // namespace loki::core
