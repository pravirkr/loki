#pragma once

/**
 * @file planner_memory.hpp
 * @brief Memory reserve shared by the FFA and EP region planners. Internal.
 */

namespace loki::algorithms::detail {

/**
 * @brief Process memory (GiB) that the planners' memory models do not cover.
 *
 * The models count every large buffer exactly. What remains is hard to account
 * deterministically: heap fragmentation and allocator arenas, thread stacks,
 * small per-task objects (FFA plans, FFTW plans, logging buffers) and the
 * interpreter or application around the library. A fixed amount, rather than
 * a fraction of the limit, keeps large limits usable and small limits
 * feasible. See docs/memory.md.
 */
inline constexpr double kUnmodelledReserveGB = 0.5;

/**
 * @brief Free device memory (GiB) the GPU planners leave alone: the CUDA
 * runtime and context, kernel images, cuFFT work areas and what other
 * processes allocate meanwhile. Applied by every GPU sweep to the device's
 * free memory before it takes the smaller of that and max_process_memory_gb.
 */
inline constexpr double kDeviceReserveGB = 1.0;

/// Memory available to the modelled buffers under @p max_process_memory_gb.
[[nodiscard]] constexpr double
effective_memory_limit_gb(double max_process_memory_gb) noexcept {
    return max_process_memory_gb - kUnmodelledReserveGB;
}

} // namespace loki::algorithms::detail
