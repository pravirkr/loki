#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/types.hpp"

namespace loki::algorithms {

/**
 * @brief Exclusion mask over the FFA base grid used by the EP search.
 *
 * @details The base grid is the 2D (acceleration, frequency) grid of the last
 * FFA level. A grid cell is addressed by `coord_idx = ia * n_freq + if`, which
 * is exactly the index produced by the EP `resolve()` stage and the layout of
 * the EP seed leaves. The mask is a dense bitset (1 bit per cell), so lookups
 * are O(1) and the memory footprint (`n_cells / 8` bytes) is negligible
 * compared to the FFA fold.
 *
 * Windows are given in physical units and rasterised with the same
 * value-to-index mapping as `resolve()` (see
 * `psr_utils::get_nearest_idx_analytical`), so any leaf whose instantaneous
 * (a, f) lies inside a window maps to a masked cell.
 */
class GridMask {
public:
    GridMask() = default;
    /**
     * @param lim_accel Acceleration limits of the base grid (m/s^2).
     * @param n_accel Number of acceleration cells.
     * @param lim_freq Frequency limits of the base grid (Hz).
     * @param n_freq Number of frequency cells.
     */
    GridMask(ParamLimit lim_accel,
             SizeType n_accel,
             ParamLimit lim_freq,
             SizeType n_freq);

    ~GridMask()                                = default;
    GridMask(const GridMask&)                  = default;
    GridMask& operator=(const GridMask&)       = default;
    GridMask(GridMask&&) noexcept              = default;
    GridMask& operator=(GridMask&&) noexcept   = default;

    [[nodiscard]] SizeType get_n_accel() const noexcept { return m_n_accel; }
    [[nodiscard]] SizeType get_n_freq() const noexcept { return m_n_freq; }
    [[nodiscard]] SizeType get_n_cells() const noexcept {
        return m_n_accel * m_n_freq;
    }
    /// @brief True if no cell is masked (fast-path skip for callers).
    [[nodiscard]] bool empty() const noexcept { return m_n_set == 0; }
    /// @brief Number of masked cells.
    [[nodiscard]] SizeType count() const noexcept { return m_n_set; }
    /// @brief Number of windows rasterised so far (including harmonics).
    [[nodiscard]] SizeType get_n_windows() const noexcept {
        return m_n_windows;
    }
    /// @brief Memory usage of the bitset in GiB.
    [[nodiscard]] float get_memory_usage_gib() const noexcept;

    /// @brief True if the base-grid cell `coord_idx` is masked.
    [[nodiscard]] bool is_masked(SizeType coord_idx) const noexcept {
        return ((m_bits[coord_idx >> 6U] >> (coord_idx & 63U)) & 1ULL) != 0ULL;
    }

    /**
     * @brief Rasterise a window (and optionally its harmonics) onto the grid.
     *
     * @details Cells whose value range intersects the window are set. Windows
     * disjoint from the grid are ignored. For `k in [2, n_harmonics]` the
     * frequency range is also scaled by `k` and `1/k`; the acceleration range
     * is kept (kinematic units are harmonic-invariant).
     */
    void add_window(const ParamWindow& window, SizeType n_harmonics = 0);

    /// @brief Rasterise a set of windows.
    void add_windows(std::span<const ParamWindow> windows,
                     SizeType n_harmonics = 0);

    /**
     * @brief Reset this mask to a copy of `base`.
     * @details Reuses the existing allocation when the geometry matches.
     */
    void assign(const GridMask& base);

    /// @brief Clear all bits.
    void clear() noexcept;

    /**
     * @brief Drop resolved leaves that map to a masked cell.
     *
     * @details Compacts (in-place, order-preserving) the four arrays produced
     * by Branch/Resolve: the leaves (with `leaves_stride` doubles per leaf),
     * their tree origins, their base-grid indices and their phase shifts.
     *
     * @return Number of surviving leaves.
     */
    SizeType filter_resolved(std::span<double> leaves,
                             std::span<SizeType> origins,
                             std::span<SizeType> param_idx,
                             std::span<float> phase_shift,
                             SizeType leaves_stride,
                             SizeType n_leaves) const;

    /**
     * @brief Select the seed indices that are not masked.
     *
     * @details Seed leaf `i` sits on base-grid cell `i`, so this simply
     * writes the unmasked indices in `[0, n_seeds)` to `keep_indices`.
     *
     * @return Number of kept seeds.
     */
    SizeType select_seeds(std::span<SizeType> keep_indices,
                          SizeType n_seeds) const;

private:
    ParamLimit m_lim_accel{};
    ParamLimit m_lim_freq{};
    SizeType m_n_accel{};
    SizeType m_n_freq{};
    SizeType m_n_set{};
    SizeType m_n_windows{};
    std::vector<uint64_t> m_bits;

    /// Rasterise a single window without harmonic expansion.
    void add_window_single(double f_lo,
                           double f_hi,
                           double a_lo,
                           double a_hi) noexcept;
    /// Set bits [lo, hi] (inclusive) and update the set count.
    void set_range(SizeType lo, SizeType hi) noexcept;
};

} // namespace loki::algorithms
