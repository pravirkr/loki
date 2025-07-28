#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

inline constexpr double kCval = 299792458.0;

/**
 * @brief Computes max_i(x[i] - y[i]) for two non-overlapping float arrays.
 *
 * @param x,y Non‑overlapping input arrays with at least @p size elements.
 * @param size Number of elements to process.
 * @return  The maximum of the pair‑wise differences.
 */
[[nodiscard]] float diff_max(const float* __restrict__ x,
                             const float* __restrict__ y,
                             SizeType size) noexcept;

// out[nsum] = x[0] + x[1] + ... + x[size-1] + x[0] + x[1] + ...
void circular_prefix_sum(std::span<const float> x, std::span<float> out);

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
 * @return  Index of the nearest element.
 *
 * @throws  std::invalid_argument if the array is empty or @p val is NaN.
 *
 * @complex **O(log n)** comparisons via `std::ranges::lower_bound`.
 */
[[nodiscard]] SizeType
find_nearest_sorted_idx(std::span<const double> arr_sorted, double val);

/**
 * @brief  Two‑pointer **amortised O(1)** nearest‑index finder.
 *
 * Picks up searching from @p hint_idx, making it *extremely* efficient when
 * successive queries have monotonic or near‑monotonic values.
 *
 * The function updates @p hint_idx so that the next call can resume from the
 * most recent lower‑bound position.
 *
 * @param   arr_sorted  Monotonically non‑decreasing array.
 * @param   val         Search value (finite).
 * @param   hint_idx    [in/out]  Starting position hint; becomes the lower
 *                     bound after the call.
 * @return  Index of the nearest element (same tie rule as the binary version).
 *
 * @throws  std::invalid_argument if the array is empty, @p val is NaN.
 */
[[nodiscard]] SizeType find_nearest_sorted_idx_scan(
    std::span<const double> arr_sorted, double val, SizeType& hint_idx);

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

} // namespace loki::utils
