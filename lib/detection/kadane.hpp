#pragma once

#include <span>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::detection {

struct BoxcarKadaneCache {
    std::vector<float> biases;
    SizeType n_biases;
    std::vector<float> fold_norm_buffer;

    BoxcarKadaneCache(std::span<const float> biases, SizeType nbins);
};

// Compute the S/N of a batch of folded profiles using the Kadane algorithm
SizeType
score_and_filter_max_kadane_with_cache(std::span<const float> folds,
                                       std::span<float> scores,
                                       std::span<SizeType> indices_filtered,
                                       float threshold,
                                       SizeType nprofiles,
                                       SizeType nbins,
                                       BoxcarKadaneCache& cache);

} // namespace loki::detection