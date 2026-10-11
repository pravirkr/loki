#include <cuda_runtime.h>

#include "lib/common/dispatch.hpp"
#include "lib/cuda/cuda_utils.cuh"

namespace loki::detail {

void initialize_device_gpu(int device) {
    cuda_utils::check_cuda_call(cudaSetDevice(device),
                                "initialize_device: cudaSetDevice failed");
    // The first runtime call that needs a context creates it.
    cuda_utils::check_cuda_call(cudaFree(nullptr),
                                "initialize_device: context creation failed");
}

} // namespace loki::detail
