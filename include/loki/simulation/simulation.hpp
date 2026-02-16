#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::simulation {

void generate_folded_profile(std::span<float> profile,
                             SizeType nbins,
                             float ducy   = 0.1F,
                             float center = 0.5F);

std::vector<float> generate_folded_profile(SizeType nbins = 100,
                                           float ducy     = 0.1F,
                                           float center   = 0.5F);

} // namespace loki::simulation