#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

namespace loki::core {

// Old method (not used anymore)
std::tuple<std::vector<SizeType>, float>
ffa_taylor_resolve_generic(std::span<const double> pset_cur,
                           std::span<const SizeType> param_grid_count_prev,
                           std::span<const ParamLimit> param_limits,
                           SizeType ffa_level,
                           SizeType latter,
                           double tseg_brute,
                           SizeType nbins);

void ffa_taylor_resolve_freq_batch(SizeType n_freqs_cur,
                                   SizeType n_freqs_prev,
                                   const ParamLimit& lim_freq,
                                   std::span<coord::FFACoordFreq> coords,
                                   SizeType ffa_level,
                                   double tseg_brute,
                                   SizeType nbins,
                                   int nthreads);

void ffa_taylor_resolve_poly_batch(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    SizeType latter,
    double tseg_brute,
    SizeType nbins,
    SizeType n_params,
    int nthreads);

/**
 * @brief Accel (n_params = 2) coordinates, both halves, of rows
 * [row_begin, row_end) of FFA level `ffa_level` (a row is one accel trial of
 * param_grid_count_cur[1] frequencies). `coords` holds those rows only.
 * Bit-identical to the rows of ffa_taylor_resolve_poly_batch; rows are
 * resolved in parallel.
 */
void ffa_taylor_resolve_accel_rows(
    std::span<const SizeType> param_grid_count_cur,
    std::span<const SizeType> param_grid_count_prev,
    std::span<const ParamLimit> param_limits,
    std::span<coord::FFACoord> coords,
    SizeType ffa_level,
    double tseg_brute,
    SizeType nbins,
    SizeType row_begin,
    SizeType row_end,
    int nthreads);

} // namespace loki::core
