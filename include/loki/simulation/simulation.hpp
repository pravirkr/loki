#pragma once

#include <vector>

#include "loki/common/types.hpp"

namespace loki::simulation {

std::vector<float> generate_folded_profile(SizeType nbins = 100,
                                           float ducy     = 0.1F,
                                           float center   = 0.5F);

} // namespace loki::simulation