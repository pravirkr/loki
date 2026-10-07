// Explicit instantiations of the RegScorer kernels of the CUDA
// DynamicThresholdScheme; see cuda/thresholds_kernels_cuda.cuh.

#include "lib/cuda/thresholds_kernels_cuda.cuh"

namespace loki::detection::detail {

template struct ScorerLaunch<RegScorer<32, 16>>;
template struct ScorerLaunch<RegScorer<32, 32>>;
template struct ScorerLaunch<RegScorer<64, 16>>;
template struct ScorerLaunch<RegScorer<64, 32>>;
template struct ScorerLaunch<RegScorer<64, 64>>;
template struct ScorerLaunch<RegScorer<128, 16>>;
template struct ScorerLaunch<RegScorer<128, 32>>;
template struct ScorerLaunch<RegScorer<128, 64>>;

} // namespace loki::detection::detail
