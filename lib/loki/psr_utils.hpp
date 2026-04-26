#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/utils.hpp"

namespace loki::psr_utils {

/**
 * @brief Computes the absolute phase index for a periodic signal.
 *
 * The phase is calculated as the fractional part of the total cycles
 * ((time + delay) * frequency) and then scaled by the number of bins.
 *
 * @param proper_time The time of the event (e.g., arrival time).
 * @param freq The frequency of the periodic signal in Hz.
 * @param nbins The number of phase bins to divide the period into.
 * @param delay A time delay offset to be added to proper_time.
 * @return The phase bin index as a float, in the range [0, nbins).
 */
inline float get_phase_idx(double proper_time,
                           double freq,
                           SizeType nbins,
                           double delay = 0.0) {
    error_check::check_greater_equal(freq, 0.0, "Frequency must be positive");
    error_check::check_greater_equal(nbins, 1,
                                     "Number of bins must be positive");
    // Calculate the total phase in cycles (can be negative or > 1)
    const double total_phase = (proper_time - delay) * freq;
    // Normalize phase to [0, 1) interval
    double norm_phase = total_phase - std::floor(total_phase);
    // Scale the normalized phase to [0, nbins) and convert to float
    double iphase = norm_phase * static_cast<double>(nbins);
    if (iphase >= static_cast<double>(nbins)) {
        iphase = 0.0;
    }
    return static_cast<float>(iphase);
}

// When using, iphase ∈ [0, nbins), half-up rounding is intentional and
// deterministic.
inline uint32_t get_phase_idx_uint(double proper_time,
                                   double freq,
                                   SizeType nbins,
                                   double delay = 0.0) {
    const float iphase = get_phase_idx(proper_time, freq, nbins, delay);
    auto iphase_int    = static_cast<uint32_t>(iphase + 0.5F);
    if (iphase_int == nbins) {
        iphase_int = 0;
    }
    return iphase_int;
}

// Grid size for frequency and its derivatives {f_k, ..., f}.
std::vector<double> poly_taylor_step_f(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double t_ref = 0.0);

// Grid for parameters {d_k,... d_2, f} based on the Taylor expansion.
std::vector<double> poly_taylor_step_d_f(SizeType nparams,
                                         double tobs,
                                         SizeType fold_bins,
                                         double tol_bins,
                                         double f_max,
                                         double t_ref = 0.0);

// Grid for parameters {d_k,... d_2, d_1} based on the Taylor expansion.
std::vector<double> poly_taylor_step_d(SizeType nparams,
                                       double tobs,
                                       SizeType fold_bins,
                                       double tol_bins,
                                       double f_max,
                                       double t_ref = 0.0);

void poly_taylor_step_d_vec(SizeType nparams,
                            double tobs,
                            SizeType nbins,
                            double eta,
                            std::span<const double> f_max,
                            std::span<double> dparams_batch,
                            double t_ref = 0.0);

// Check if a parameter should be split
bool split_f(double df_old,
             double df_new,
             double tobs_new,
             SizeType k,
             double nbins,
             double eta,
             double t_ref = 0.0);

// Compute the bin shift for parameters {d_k,... d_2, f}.
std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType nbins,
                                        double f_cur,
                                        double t_ref = 0.0);

void poly_taylor_shift_d_vec(std::span<const double> dparam_old,
                             std::span<const double> dparam_new,
                             double tobs_new,
                             SizeType nbins,
                             std::span<const double> f_cur,
                             double t_ref,
                             std::span<double> shift_bins_batch,
                             SizeType nbatch,
                             SizeType nparams);

void poly_cheb_step_vec(SizeType n_params,
                        SizeType nbins,
                        double eta,
                        std::span<const double> f0_batch,
                        std::span<double> dparams_batch);

void poly_cheb_shift_vec(std::span<const double> dparam_old,
                         std::span<const double> dparam_new,
                         SizeType nbins,
                         std::span<const double> f_cur,
                         std::span<double> shift_bins_batch,
                         SizeType nbatch,
                         SizeType nparams);

/**
 * @brief Perfectly sub-divide a parent parameter cell into contiguous child cells.
 *
 * DESIGN NOTE: Exact Contiguous Splitting
 * This physically partitions the parent cell into `num_points` equal sub-cells.
 * The outermost edges of the extreme child cells sit perfectly flush with the
 * boundaries of the parent cell, ensuring zero overlap and zero gaps between
 * adjacent branches in the hierarchical tree.
 *
 * The function assumes that parameter values outside the allowed search
 * domain will be handled elsewhere (e.g. in the FFA init step). Therefore
 * it does not enforce parameter limits internally.
 *
 * @param param_cur Current parameter value (centre of the parent cell).
 * @param dparam_cur Current grid spacing of the parameter dimension. Should be trunctaed as per
 * search range.
 * @param dparam_new Desired grid spacing for the refined search stage. The actual spacing
 * used may differ slightly in order to maintain symmetry.
 * @return std::tuple<std::vector<double>, double> Array of new parameter values
 * for the child cells and the actual grid spacing used for the refined search
 * stage.
 */
std::tuple<std::vector<double>, double>
branch_param(double param_cur, double dparam_cur, double dparam_new);

// Refine a parameter range around a current value with a finer step size.
// The output is written to a pre-allocated array in a slice to write into
// (shape MAX_BRANCH_VALS,).
// Returns the refined dparam and the number of params written.
std::pair<double, SizeType> branch_param_padded(std::span<double> out_values,
                                                double param_cur,
                                                double dparam_cur,
                                                double dparam_new,
                                                SizeType branch_max);

std::pair<double, SizeType> branch_dparam_crackle(double dparam_cur,
                                                  double dparam_new,
                                                  SizeType branch_max);

void branch_crackle_padded(std::span<double> out_values,
                           double param_cur,
                           double dparam_cur,
                           SizeType num_points);

inline void branch_one_param_padded(SizeType p,
                                    double cur,
                                    double sig_cur,
                                    double eta,
                                    const double* __restrict__ shift_bins_ptr,
                                    const double* __restrict__ dparam_new_ptr,
                                    double* __restrict__ scratch_params_ptr,
                                    double* __restrict__ scratch_dparams_ptr,
                                    SizeType* __restrict__ scratch_counts_ptr,
                                    SizeType flat_base,
                                    SizeType branch_max) {
    const SizeType pad_offset = (flat_base + p) * branch_max;

    if (shift_bins_ptr[flat_base + p] >= (eta - utils::kFloatEps)) {
        auto slice =
            std::span<double>(scratch_params_ptr + pad_offset, branch_max);
        auto [dparam_act, count] = branch_param_padded(
            slice, cur, sig_cur, dparam_new_ptr[flat_base + p], branch_max);
        scratch_dparams_ptr[flat_base + p] = dparam_act;
        scratch_counts_ptr[flat_base + p]  = count;
    } else {
        scratch_params_ptr[pad_offset]     = cur;
        scratch_dparams_ptr[flat_base + p] = sig_cur;
        scratch_counts_ptr[flat_base + p]  = 1;
    }
}

inline void
branch_one_param_padded_crackle(SizeType p,
                                double cur,
                                double sig_cur,
                                double eta,
                                const double* __restrict__ shift_bins_ptr,
                                const double* __restrict__ dparam_new_ptr,
                                double* __restrict__ scratch_params_ptr,
                                double* __restrict__ scratch_dparams_ptr,
                                SizeType* __restrict__ scratch_counts_ptr,
                                SizeType flat_base,
                                SizeType branch_max) {
    const SizeType pad_offset = (flat_base + p) * branch_max;

    if (shift_bins_ptr[flat_base + p] >= (eta - utils::kFloatEps)) {
        auto [dparam_act, count] = branch_dparam_crackle(
            sig_cur, dparam_new_ptr[flat_base + p], branch_max);
        scratch_dparams_ptr[flat_base + p] = dparam_act;
        scratch_counts_ptr[flat_base + p]  = count;
    } else {
        scratch_dparams_ptr[flat_base + p] = sig_cur;
        scratch_counts_ptr[flat_base + p]  = 1;
    }
    scratch_params_ptr[pad_offset] = cur;
}

// Count the number of parameters in a range.
inline SizeType range_param_count(double vmin, double vmax, double dv) {
    error_check::check_greater(vmax, vmin, "vmax must be greater than vmin");
    error_check::check_greater(dv, 0, "dv must be positive");
    if (dv >= (vmax - vmin)) {
        return 1;
    }
    return static_cast<SizeType>(std::ceil((vmax - vmin) / dv));
}

/**
 * @brief Generate an array of cell centres that perfectly tile the parameter
 * space.
 *
 * @param vmin Minimum boundary of the parameter space.
 * @param vmax Maximum boundary of the parameter space.
 * @param dv Desired step size. Actual spacing will be <= dv to ensure perfect
 * tiling.
 * @return Array of cell centres uniformly spaced.
 */
inline std::vector<double> range_param(double vmin, double vmax, double dv) {
    error_check::check_greater(vmax, vmin, "vmax must be greater than vmin");
    error_check::check_greater(dv, 0, "dv must be positive");
    if (dv >= (vmax - vmin)) {
        return {(vmax + vmin) / 2.0};
    }
    const auto npoints = static_cast<SizeType>(std::ceil((vmax - vmin) / dv));
    const double dv_actual = (vmax - vmin) / static_cast<double>(npoints);

    std::vector<double> result(npoints);
    const double start = vmin + (0.5 * dv_actual);
    // np.linspace(vmin + (dv_actual / 2.0), vmax - (dv_actual / 2.0), npoints)
    for (SizeType i = 0; i < npoints; ++i) {
        result[i] = start + (dv_actual * static_cast<double>(i));
    }
    return result;
}

/**
 * @brief Compute the parameter value at a specific grid index on the fly.
 *
 * This lazily evaluates the exact contiguous tiling geometry without allocating
 * the full array.
 *
 * @param limit The parameter limits (min and max).
 * @param count The total number of points in the grid.
 * @param i The target index.
 * @return The physical centre value of the cell at index i.
 */
inline double
get_param_val_at_idx(const ParamLimit& limit, SizeType count, SizeType i) {
    // Safeguard against 0 or 1 point grids (center the point exactly in the
    // middle)
    if (count <= 1) {
        return (limit.max + limit.min) / 2.0;
    }
    const double dv_actual =
        (limit.max - limit.min) / static_cast<double>(count);
    return limit.min + (dv_actual * (static_cast<double>(i) + 0.5));
}

/**
 * @brief Get the nearest index in a uniformly spaced grid.
 *
 * @param val The value to find the nearest index for.
 * @param vmin The minimum value of the grid.
 * @param vmax The maximum value of the grid.
 * @param count The number of points in the grid.
 * @return The nearest index.
 */
inline SizeType get_nearest_idx_analytical(double val,
                                           const ParamLimit& limit,
                                           SizeType count) {
    if (count <= 1) {
        return 0;
    }
    const double raw_idx = static_cast<double>(count) * (val - limit.min) /
                           (limit.max - limit.min);
    const auto idx       = static_cast<int>(raw_idx + utils::kFloatEps);
    if (idx < 0) {
        return 0;
    }
    if (idx >= static_cast<int>(count)) {
        return count - 1;
    }
    return static_cast<SizeType>(idx);
}

/**
 * @class MiddleOutScheme
 * @brief A utility class for "middle-out" indexing of segments for hierarchical
 * algorithms.
 */
class MiddleOutScheme {
public:
    MiddleOutScheme() = default;
    MiddleOutScheme(SizeType nsegments, SizeType ref_idx, double tsegment = 1.0)
        : m_nsegments(nsegments),
          m_ref_idx(ref_idx),
          m_tsegment(tsegment),
          m_data(nsegments) {
        error_check::check(nsegments > 0, "nsegments must be greater than 0.");
        error_check::check(ref_idx < nsegments,
                           "ref_idx must be less than nsegments.");
        error_check::check(tsegment >= 0.0, "tsegment must be non-negative.");
        // np.argsort(np.abs(np.arange(nseg) - ref_idx), kind="stable")
        std::iota(m_data.begin(), m_data.end(), 0);
        std::ranges::sort(m_data, [ref = m_ref_idx](SizeType a, SizeType b) {
            auto da = (a < ref) ? (ref - a) : (a - ref);
            auto db = (b < ref) ? (ref - b) : (b - ref);
            if (da != db) {
                return da < db;
            }
            return a < b;
        });
    }
    SizeType get_nsegments() const { return m_nsegments; }
    SizeType get_ref_idx() const { return m_ref_idx; }
    double get_tsegment() const { return m_tsegment; }
    std::vector<SizeType> get_data() const { return m_data; }

    /**
     * @brief Returns the reference time at the middle of the reference segment.
     */
    [[nodiscard]] double ref_time() const {
        return (static_cast<double>(m_ref_idx) + 0.5) * m_tsegment;
    }

    /**
     * @brief Gets the segment index at the specified hierarchical level.
     * @param level The hierarchical level (0 is the reference segment).
     * @return The segment index at the given level.
     */
    [[nodiscard]] SizeType get_segment_idx(SizeType level) const {
        if (level >= m_nsegments) {
            throw std::out_of_range(
                std::format("level must be in [0, {}].", m_nsegments - 1));
        }
        return m_data[level];
    }

    /**
     * @brief Gets the coordinate (ref and scale) for the combined interval up
     * to a given level.
     *
     * The reference time is the center of the time interval covered by all
     * segments from level 0 to the specified level. The scale is the half-width
     * of this interval.
     *
     * @param level The current hierarchical level.
     * @return A pair containing the reference and scale in seconds.
     */
    [[nodiscard]] std::pair<double, double> get_coord(SizeType level) const {
        if (level >= m_nsegments) {
            throw std::out_of_range(
                std::format("level must be in [0, {}].", m_nsegments - 1));
        }
        auto scheme_till_now =
            std::views::take(m_data, static_cast<IndexType>(level + 1));
        const auto [min_it, max_it] =
            std::ranges::minmax_element(scheme_till_now);

        const auto min_val   = static_cast<double>(*min_it);
        const auto max_val   = static_cast<double>(*max_it);
        const auto ref_val   = (min_val + max_val + 1.0) / 2.0;
        const auto scale_val = ref_val - min_val;

        return {ref_val * m_tsegment, scale_val * m_tsegment};
    }

    /**
     * @brief Gets the coordinate (ref and scale) for the single segment at the
     * given level.
     * @param level The hierarchical level.
     * @return A pair containing the reference and scale for the segment in
     * seconds.
     */
    [[nodiscard]] std::pair<double, double>
    get_segment_coord(SizeType level) const {
        const auto current_idx = static_cast<double>(get_segment_idx(level));
        const auto ref_val     = (current_idx + 0.5) * m_tsegment;
        const auto scale_val   = 0.5 * m_tsegment;
        return {ref_val, scale_val};
    }

    /**
     * @brief Gets the segment indices and coordinates for all segments up to
     * the given level.
     * @param level The hierarchical level.
     * @return A pair containing the segment indices and coordinates.
     */
    [[nodiscard]] std::pair<std::vector<SizeType>,
                            std::vector<std::pair<double, double>>>
    get_segment_coords_so_far(SizeType level) const {
        if (level >= m_nsegments) {
            throw std::out_of_range(
                std::format("level must be in [0, {}].", m_nsegments - 1));
        }
        std::vector<SizeType> idx_segments(level + 1);
        std::vector<std::pair<double, double>> coord_segments(level + 1);
        for (SizeType i = 0; i <= level; ++i) {
            idx_segments[i]   = get_segment_idx(i);
            coord_segments[i] = get_segment_coord(i);
        }
        return {idx_segments, coord_segments};
    }

    /**
     * @brief Gets the coordinate (ref and scale) for the accumulated current
     * segment at a given level.
     * @param level The hierarchical level.
     * @return A pair containing the reference and scale for the segment in
     * seconds.
     */
    [[nodiscard]] std::pair<double, double>
    get_current_coord(SizeType level) const {
        if (level == 0) {
            return get_coord(level);
        }
        const auto [prev_ref, prev_scale] = get_coord(level - 1);
        const auto [cur_ref, cur_scale]   = get_coord(level);
        return {prev_ref, cur_scale};
    }

    /**
     * @brief Gets the min and max valid segment indices up to a given pruning
     * level.
     * @param prune_level The hierarchical level to consider.
     * @return A pair containing the minimum and maximum segment indices.
     */
    [[nodiscard]] std::pair<SizeType, SizeType>
    get_valid(SizeType prune_level) const {
        if (prune_level == 0) {
            throw std::invalid_argument(
                std::format("prune_level must be greater than 0."));
        }
        if (prune_level > m_nsegments) {
            throw std::out_of_range(std::format(
                "prune_level must be in [0, {}].", m_nsegments - 1));
        }

        auto scheme_till_now =
            std::views::take(m_data, static_cast<IndexType>(prune_level));
        const auto [min_it, max_it] =
            std::ranges::minmax_element(scheme_till_now);
        return {*min_it, *max_it};
    }

    /**
     * @brief Gets the shift of the current coordinate reference from the
     * starting reference.
     * @param level The hierarchical level.
     * @return The difference in seconds.
     */
    [[nodiscard]] double get_delta(SizeType level) const {
        return get_coord(level).first - ref_time();
    }

    /**
     * @brief Gets the total number of segments.
     */
    [[nodiscard]] SizeType size() const { return m_nsegments; }

private:
    SizeType m_nsegments{};
    SizeType m_ref_idx{};
    double m_tsegment{};
    std::vector<SizeType> m_data;
};

} // namespace loki::psr_utils