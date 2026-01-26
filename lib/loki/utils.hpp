#pragma once

#include <optional>
#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

inline constexpr double kCval    = 299792458.0;            // m/s
inline constexpr double kInvCval = 3.3356409519815204e-09; // s/m
inline constexpr double kEps     = 1e-12;

/**
 * @brief Computes max_i(x[i] - y[i]) for two non-overlapping float arrays.
 *
 * @param x,y Non‑overlapping input arrays with at least @p size elements.
 * @param size Number of elements to process.
 * @return  The maximum of the pair‑wise differences.
 */
float diff_max(const float* __restrict__ x,
               const float* __restrict__ y,
               SizeType size) noexcept;

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(const float* __restrict__ x,
                         float* __restrict__ out,
                         SizeType nbins,
                         SizeType nsum) noexcept;

/**
 * @brief  Locate the index of the element **closest** to @p val in a sorted
 *         array.
 *
 * Breaks ties toward the *lower* index, i.e. when `val` lies exactly in the
 * middle of two equal‑distant neighbours the one with the smaller index is
 * returned.
 *
 * @param   arr_sorted  Monotonically non‑decreasing array.
 * @param   val         Search value (must be finite).
 * @param   rtol        Relative tolerance for floating-point comparison
 * (default: 1e-5).
 * @param   atol        Absolute tolerance for floating-point comparison
 *                      (default: 1e-8).
 * @return  Index of the nearest element.
 *
 * @throws  std::invalid_argument if the array is empty or @p val is NaN.
 *
 * @complex **O(log n)** comparisons via `std::ranges::lower_bound`.
 */
[[nodiscard]] SizeType
find_nearest_sorted_idx(std::span<const double> arr_sorted,
                        double val,
                        double rtol = 1e-5,
                        double atol = 1e-8);

/**
 * @brief  Two‑pointer **amortised O(1)** nearest‑index finder.
 *
 * Find the index of the closest value in a sorted array using a linear scan
 * starting from a hint index.
 *
 * In case of a tie, the index of the smaller value is returned.
 * Behavior is undefined if the array is not sorted in ascending order.
 * The hint_idx is updated to the returned index for use in subsequent calls.
 *
 * @param   arr_sorted  Monotonically non‑decreasing array.
 * @param   val         Search value (finite).
 * @param   hint_idx    [in/out]  Reference to an index used as a starting point
 * for the scan, updated to the returned index.
 * @param   rtol        Relative tolerance for floating-point comparison
 * (default: 1e-5).
 * @param   atol        Absolute tolerance for floating-point comparison
 *                      (default: 1e-8).
 * @return  Index of the nearest element (same tie rule as the binary version).
 *
 * @throws  std::invalid_argument if the array is empty, @p val is NaN.
 */
[[nodiscard]] SizeType
find_nearest_sorted_idx_scan(std::span<const double> arr_sorted,
                             double val,
                             SizeType& hint_idx,
                             double rtol = 1e-5,
                             double atol = 1e-8);

/**
 * @brief  Return @p num indices centred (as much as possible) around
 *         @p target_idx.
 *
 *
 * @param   indices      Sorted list of indices.
 * @param   target_idx   Index around which to centre the window.
 * @param   num          Number of neighbours to return (must be > 0).
 * @return  `std::vector` containing the selected neighbours.
 *
 * @throws  std::invalid_argument on empty @p indices or zero @p num.
 */
std::vector<SizeType> find_neighbouring_indices(
    std::span<const SizeType> indices, SizeType target_idx, SizeType num);

/**
 * @brief  Generate `num_samples` evenly spaced points in [`start`, `stop`].
 *
 * @param   start        First value (included).
 * @param   stop         Last value (included if @p endpoint is true).
 * @param   num_samples  Number of points to generate.
 * @param   endpoint     When *true* the sequence **includes** @p stop.  When
 *                       *false* the sequence ends one step before @p stop.
 * @return  Vector of size @p num_samples (possibly empty).
 *
 * @note    By construction `(stop - start)` is evenly divisible by
 *          `(num_samples - 1)` when `endpoint == true`.
 */
std::vector<double> linspace(double start,
                             double stop,
                             SizeType num_samples = 50,
                             bool endpoint        = true) noexcept;

std::vector<SizeType>
determine_ref_segs(SizeType nsegments,
                   std::optional<SizeType> n_runs,
                   std::optional<std::vector<SizeType>> ref_segs);

} // namespace loki::utils
