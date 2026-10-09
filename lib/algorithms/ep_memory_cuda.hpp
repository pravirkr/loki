#pragma once

/**
 * @file ep_memory_cuda.hpp
 * @brief Device queries the CUDA policy of the EP memory model needs.
 * Internal. Declarations only: the definitions live in lib/cuda/ and exist in
 * GPU builds, so callers reach them only behind the GPU dispatch guard of a
 * facade translation unit (docs/architecture.md, rule 4).
 */

#include <string>

#include "loki/common/types.hpp"

namespace loki::algorithms::detail {

/// Bytes of CUB temporary storage that an EP workspace for @p max_n_leaves
/// leaves (batch_size * branch_max) allocates on @p device. Queries CUB
/// itself, so the model follows the library version; memoised.
SizeType ep_cuda_cub_scratch_bytes(SizeType max_n_leaves, int device);

/// Free device memory (GiB) of @p device right now.
double ep_cuda_free_memory_gb(int device);

/// Compute capability of @p device, e.g. "sm_89". Informational: plan caches
/// record it and do not reject a plan because of it.
std::string ep_cuda_device_arch(int device);

} // namespace loki::algorithms::detail
