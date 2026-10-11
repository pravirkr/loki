#pragma once

/**
 * @file ffa_resolve_device.cuh
 * @brief Device resolve of one accel (n_params = 2) FFA coordinate from the
 * per-level row / column tables. Shared by kernel_ffa_resolve_accel_fast and
 * the fused final-level scorers, so both produce the same bits. CUDA only.
 */

#include <cstdint>

#include <cuda_runtime.h>

#include "lib/cuda/kernel_utils.cuh"
#include "lib/cuda/taylor_ffa_cuda.cuh"

namespace loki::core::detail {

struct AccelCoord {
    uint32_t i_tail;
    float shift_tail;
    uint32_t i_head;
    float shift_head;
};

// Coordinate `local` of a level (f fast moving): the parent index and phase
// shift of each half, as ffa_taylor_resolve_accel_batch_device defines them.
__device__ __forceinline__ AccelCoord
resolve_accel_coord(const FFAAccelResolveLevel& lvl, uint32_t local) {
    const uint32_t freq_idx      = local % lvl.n_freq_cur;
    const uint32_t accel_idx     = (local / lvl.n_freq_cur) % lvl.n_accel_cur;
    const FFAAccelResolveRow row = lvl.rows[accel_idx];
    const double f_cur           = lvl.cols[freq_idx];
    AccelCoord c{};
#pragma unroll
    for (int latter = 0; latter < 2; ++latter) {
        const double f_new   = f_cur * row.w[latter];
        const uint32_t idx_f = utils::get_nearest_idx_analytical_device(
            f_new, lvl.f_min, lvl.f_max, lvl.n_freq_prev);
        // get_phase_idx_device with (proper_time - delay) precomputed.
        const double total_phase = row.u[latter] * f_cur;
        const double norm_phase  = total_phase - floor(total_phase);
        double iphase            = norm_phase * static_cast<double>(lvl.nbins);
        if (iphase >= static_cast<double>(lvl.nbins)) {
            iphase = 0.0;
        }
        const uint32_t final_idx = row.idx_a_base + idx_f;
        if (latter == 0) {
            c.i_tail     = final_idx;
            c.shift_tail = static_cast<float>(iphase);
        } else {
            c.i_head     = final_idx;
            c.shift_head = static_cast<float>(iphase);
        }
    }
    return c;
}

} // namespace loki::core::detail
