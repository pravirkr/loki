#include "lib/algorithms/ep_memory_cuda.hpp"

#include <format>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::algorithms::detail {

SizeType ep_cuda_cub_scratch_bytes(SizeType max_n_leaves, int device) {
    static std::mutex mutex;
    static std::map<std::pair<int, SizeType>, SizeType> memo;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::pair{device, max_n_leaves};
    if (const auto it = memo.find(key); it != memo.end()) {
        return it->second;
    }
    cuda_utils::CudaSetDeviceGuard device_guard(device);
    const SizeType bytes =
        memory::CUBScratchArena::temp_bytes_for(max_n_leaves);
    memo.emplace(key, bytes);
    return bytes;
}

double ep_cuda_free_memory_gb(int device) {
    cuda_utils::CudaSetDeviceGuard device_guard(device);
    return cuda_utils::get_cuda_memory_usage().first;
}

std::string ep_cuda_device_arch(int device) {
    cuda_utils::CudaSetDeviceGuard device_guard(device);
    cudaDeviceProp prop{};
    cuda_utils::check_cuda_call(
        cudaGetDeviceProperties(&prop, device),
        "EPRegionPlanner: cudaGetDeviceProperties failed");
    return std::format("sm_{}{}", prop.major, prop.minor);
}

} // namespace loki::algorithms::detail
