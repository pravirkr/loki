#pragma once

#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::utils {

struct BranchingWorkspaceView {
    double* __restrict__ scratch_params;
    double* __restrict__ scratch_dparams;
    SizeType* __restrict__ scratch_counts;
};

struct BranchingWorkspace {
    std::vector<double> scratch_params;
    std::vector<double> scratch_dparams;
    std::vector<SizeType> scratch_counts;

    BranchingWorkspace(SizeType batch_size,
                       SizeType branch_max,
                       SizeType nparams)
        : scratch_params(batch_size * nparams * branch_max),
          scratch_dparams(batch_size * nparams),
          scratch_counts(batch_size * nparams) {}

    [[nodiscard]] BranchingWorkspaceView get_view() noexcept {
        return BranchingWorkspaceView{.scratch_params  = scratch_params.data(),
                                      .scratch_dparams = scratch_dparams.data(),
                                      .scratch_counts  = scratch_counts.data()};
    }

    [[nodiscard]] float get_memory_usage() const noexcept {
        const auto total_memory = (scratch_params.size() * sizeof(double)) +
                                  (scratch_dparams.size() * sizeof(double)) +
                                  (scratch_counts.size() * sizeof(SizeType));
        return static_cast<float>(total_memory) /
               static_cast<float>(1ULL << 30U);
    }
};

#ifdef LOKI_ENABLE_CUDA
struct BranchingWorkspaceCUDAView {
    double* __restrict__ scratch_params;
    double* __restrict__ scratch_dparams;
    uint32_t* __restrict__ scratch_counts;
    uint32_t* __restrict__ leaf_branch_count;
    uint32_t* __restrict__ leaf_output_offset;
    void* __restrict__ cub_temp_storage;
    SizeType cub_temp_bytes;
};

struct BranchingWorkspaceCUDA {
    thrust::device_vector<double> scratch_params;
    thrust::device_vector<double> scratch_dparams;
    thrust::device_vector<uint32_t> scratch_counts;
    thrust::device_vector<uint32_t> leaf_branch_count;
    thrust::device_vector<uint32_t> leaf_output_offset;
    void* cub_temp_storage  = nullptr;
    SizeType cub_temp_bytes = 0;
    cudaStream_t m_stream   = nullptr;

    BranchingWorkspaceCUDA(size_t batch_size,
                           size_t branch_max,
                           size_t nparams,
                           cudaStream_t stream = nullptr);
    ~BranchingWorkspaceCUDA();
    BranchingWorkspaceCUDA(const BranchingWorkspaceCUDA&)            = delete;
    BranchingWorkspaceCUDA& operator=(const BranchingWorkspaceCUDA&) = delete;
    BranchingWorkspaceCUDA(BranchingWorkspaceCUDA&&) noexcept        = delete;
    BranchingWorkspaceCUDA&
    operator=(BranchingWorkspaceCUDA&&) noexcept = delete;

    [[nodiscard]] BranchingWorkspaceCUDAView get_view() noexcept;

    [[nodiscard]] float get_memory_usage() const noexcept;
};

struct DeviceCounter {
    uint32_t* d_ptr = nullptr;
    uint32_t* h_ptr = nullptr; // pinned

    DeviceCounter();
    ~DeviceCounter();
    DeviceCounter(const DeviceCounter&)                      = delete;
    DeviceCounter& operator=(const DeviceCounter&)           = delete;
    DeviceCounter(DeviceCounter&& other) noexcept            = delete;
    DeviceCounter& operator=(DeviceCounter&& other) noexcept = delete;

    void reset(cudaStream_t stream);
    [[nodiscard]] uint32_t* data() noexcept { return d_ptr; }
    [[nodiscard]] const uint32_t* data() const noexcept { return d_ptr; }
    [[nodiscard]] uint32_t value_sync(cudaStream_t stream);
};

#endif // LOKI_ENABLE_CUDA

} // namespace loki::utils