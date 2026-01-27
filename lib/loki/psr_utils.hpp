#pragma once

#include <algorithm>
#include <format>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"

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
float get_phase_idx(double proper_time,
                    double freq,
                    SizeType nbins,
                    double delay = 0.0);

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
                            SizeType fold_bins,
                            double tol_bins,
                            std::span<const double> f_max,
                            std::span<double> dparams_batch,
                            double t_ref = 0.0);

// Check if a parameter should be split
bool split_f(double df_old,
             double df_new,
             double tobs_new,
             SizeType k,
             double fold_bins,
             double tol_bins,
             double t_ref = 0.0);

// Compute the bin shift for parameters {d_k,... d_2, f}.
std::vector<double> poly_taylor_shift_d(std::span<const double> dparam_old,
                                        std::span<const double> dparam_new,
                                        double tobs_new,
                                        SizeType fold_bins,
                                        double f_cur,
                                        double t_ref = 0.0);

void poly_taylor_shift_d_vec(std::span<const double> dparam_old,
                             std::span<const double> dparam_new,
                             double tobs_new,
                             SizeType fold_bins,
                             std::span<const double> f_cur,
                             double t_ref,
                             std::span<double> shift_bins_batch,
                             SizeType nbatch,
                             SizeType nparams);

// Refine a parameter range around a current value with a finer step size.
std::tuple<std::vector<double>, double>
branch_param(double param_cur,
             double dparam_cur,
             double dparam_new,
             double param_min = std::numeric_limits<double>::lowest(),
             double param_max = std::numeric_limits<double>::max());

// Refine a parameter range around a current value with a finer step size.
// The output is written to a pre-allocated array in a slice to write into
// (shape MAX_BRANCH_VALS,).
// Returns the refined dparam and the number of params written.
std::pair<double, SizeType>
branch_param_padded(std::span<double> out_values,
                    double param_cur,
                    double dparam_cur,
                    double dparam_new,
                    double param_min = std::numeric_limits<double>::lowest(),
                    double param_max = std::numeric_limits<double>::max());

inline void branch_one_param_padded(int p,
                                    double cur,
                                    double sig_cur,
                                    double sig_new,
                                    double pmin,
                                    double pmax,
                                    double eta,
                                    const double* __restrict shift_bins_ptr,
                                    double* __restrict scratch_params_ptr,
                                    double* __restrict scratch_dparams_ptr,
                                    SizeType* __restrict scratch_counts_ptr,
                                    SizeType flat_base,
                                    SizeType branch_max) {
    constexpr double kEps     = 1e-12;
    const SizeType pad_offset = (flat_base + p) * branch_max;

    if (shift_bins_ptr[flat_base + p] >= (eta - kEps)) {
        auto slice =
            std::span<double>(scratch_params_ptr + pad_offset, branch_max);
        auto [dparam_act, count] = psr_utils::branch_param_padded(
            slice, cur, sig_cur, sig_new, pmin, pmax);
        scratch_dparams_ptr[flat_base + p] = dparam_act;
        scratch_counts_ptr[flat_base + p]  = count;
    } else {
        scratch_params_ptr[pad_offset]     = cur;
        scratch_dparams_ptr[flat_base + p] = sig_cur;
        scratch_counts_ptr[flat_base + p]  = 1;
    }
}

// Count the number of parameters in a range.
SizeType range_param_count(double vmin, double vmax, double dv);

// Generate an evenly spaced array of values between vmin and vmax.
std::vector<double> range_param(double vmin, double vmax, double dv);

/**
 * @class MiddleOutScheme
 * @brief A utility class for "middle-out" indexing of segments for hierarchical
 * algorithms.
 */
class MiddleOutScheme {
public:
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
    SizeType m_nsegments;
    SizeType m_ref_idx;
    double m_tsegment;
    std::vector<SizeType> m_data;
};

} // namespace loki::psr_utils