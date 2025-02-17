#pragma once

#include <cstddef>
#include <span>

#include <loki/loki_types.hpp>

namespace loki {

std::tuple<std::vector<SizeType>, SizeType>
ffa_resolve(std::span<const float> pset_cur,
            std::span<const std::vector<float>> parr_prev,
            SizeType ffa_level,
            SizeType latter,
            float tseg_brute,
            SizeType nbins);

} // namespace loki