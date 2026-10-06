// Explicit instantiations of the GenericScorer kernels of the CUDA
// DynamicThresholdScheme; see cuda/thresholds_kernels_cuda.cuh.

#include "cuda/thresholds_kernels_cuda.cuh"

namespace loki::detection::detail {

template struct ScorerLaunch<GenericScorer<32>>;
template struct ScorerLaunch<GenericScorer<64>>;
template struct ScorerLaunch<GenericScorer<128>>;
template struct ScorerLaunch<GenericScorer<256>>;
template struct ScorerLaunch<GenericScorer<512>>;
template struct ScorerLaunch<GenericScorer<1024>>;

} // namespace loki::detection::detail
