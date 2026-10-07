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

} // namespace loki::core
