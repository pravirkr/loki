#pragma once

#include <cstddef>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::core {

std::tuple<std::vector<SizeType>, double>
ffa_taylor_resolve(std::span<const double> pset_cur,
                   std::span<const std::vector<double>> param_arr,
                   SizeType ffa_level,
                   SizeType latter,
                   double tseg_brute,
                   SizeType nbins);

} // namespace loki::core
