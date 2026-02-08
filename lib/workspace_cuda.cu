#include "loki/utils/workspace.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "loki/common/types.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::utils {

BranchingWorkspaceCUDA::BranchingWorkspaceCUDA(size_t batch_size,
                                               size_t branch_max,
                                               size_t nparams,
                                               cudaStream_t stream)
    : scratch_params(batch_size * nparams * branch_max),
      scratch_dparams(batch_size * nparams),
      scratch_counts(batch_size * nparams),
      leaf_branch_count(batch_size * branch_max),
      leaf_output_offset(batch_size * branch_max),
      m_stream(stream) {
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(
            nullptr, cub_temp_bytes, static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), batch_size * branch_max, stream),
        "cub::DeviceScan::ExclusiveSum failed");
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&cub_temp_storage, cub_temp_bytes, stream),
        "cudaMallocAsync failed");
}

BranchingWorkspaceCUDA::~BranchingWorkspaceCUDA() {
    if (cub_temp_storage != nullptr) {
        cudaFreeAsync(cub_temp_storage, m_stream);
    }
}

BranchingWorkspaceCUDAView BranchingWorkspaceCUDA::get_view() noexcept {
    return BranchingWorkspaceCUDAView{
        .scratch_params    = thrust::raw_pointer_cast(scratch_params.data()),
        .scratch_dparams   = thrust::raw_pointer_cast(scratch_dparams.data()),
        .scratch_counts    = thrust::raw_pointer_cast(scratch_counts.data()),
        .leaf_branch_count = thrust::raw_pointer_cast(leaf_branch_count.data()),
        .leaf_output_offset =
            thrust::raw_pointer_cast(leaf_output_offset.data()),
        .cub_temp_storage = cub_temp_storage,
        .cub_temp_bytes   = cub_temp_bytes};
}

float BranchingWorkspaceCUDA::get_memory_usage() const noexcept {
    const auto total_memory = (scratch_params.size() * sizeof(double)) +
                              (scratch_dparams.size() * sizeof(double)) +
                              (scratch_counts.size() * sizeof(uint32_t)) +
                              (leaf_branch_count.size() * sizeof(uint32_t)) +
                              (leaf_output_offset.size() * sizeof(uint32_t)) +
                              (cub_temp_bytes);
    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}

// DeviceCounter implementation
DeviceCounter::DeviceCounter() {
    cuda_utils::check_cuda_call(
        cudaMalloc(&d_ptr, sizeof(uint32_t)),
        "Failed to allocate device memory for DeviceCounter");
    cuda_utils::check_cuda_call(
        cudaMallocHost(&h_ptr, sizeof(uint32_t)),
        "Failed to allocate pinned memory for DeviceCounter");
    // Safe default state
    *h_ptr = 0;
    cuda_utils::check_cuda_call(cudaMemset(d_ptr, 0, sizeof(uint32_t)),
                                "Failed to initialize DeviceCounter");
}

DeviceCounter::~DeviceCounter() {
    if (d_ptr) {
        cudaFree(d_ptr);
    }
    if (h_ptr) {
        cudaFreeHost(h_ptr);
    }
}

void DeviceCounter::reset(cudaStream_t stream) {
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_ptr, 0, sizeof(uint32_t), stream),
        "Failed to reset DeviceCounter");
}

uint32_t DeviceCounter::value_sync(cudaStream_t stream) {
    cuda_utils::check_cuda_call(cudaMemcpyAsync(h_ptr, d_ptr, sizeof(uint32_t),
                                                cudaMemcpyDeviceToHost, stream),
                                "Failed to copy DeviceCounter value to host");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed in value_sync");
    return *h_ptr;
}

} // namespace loki::utils