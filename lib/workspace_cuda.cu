#include "loki/common/types.hpp"
#include "loki/utils/workspace.hpp"

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "loki/cuda_utils.cuh"

namespace loki::utils {

CUBScratchArena::CUBScratchArena(SizeType batch_size,
                                 SizeType branch_max,
                                 cudaStream_t stream)
    : m_stream(stream) {
    const SizeType max_n_leaves = batch_size * branch_max;
    SizeType reduce_bytes       = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Sum(
            nullptr, reduce_bytes, static_cast<uint8_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), max_n_leaves, stream),
        "cub::DeviceReduce::Sum failed");
    SizeType scan_bytes = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(
            nullptr, scan_bytes, static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), max_n_leaves, stream),
        "cub::DeviceScan::ExclusiveSum failed");
    SizeType flagged_bytes = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceSelect::Flagged(
            nullptr, flagged_bytes, thrust::make_counting_iterator<uint32_t>(0),
            static_cast<const uint8_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<::cuda::std::int64_t>(max_n_leaves), stream),
        "cub::DeviceSelect::Flagged failed");
    // size ws.cub_temp_storage to flagged_bytes
    cub_temp_bytes = std::max({reduce_bytes, scan_bytes, flagged_bytes});
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&cub_temp_storage, cub_temp_bytes, stream),
        "cudaMallocAsync failed");
    cuda_utils::check_cuda_call(
        cudaMallocAsync(&d_reduce_out, sizeof(uint32_t), stream),
        "cudaMallocAsync failed");
}

CUBScratchArena::~CUBScratchArena() {
    if (cub_temp_storage != nullptr) {
        cudaFreeAsync(cub_temp_storage, m_stream);
    }
    if (d_reduce_out != nullptr) {
        cudaFreeAsync(d_reduce_out, m_stream);
    }
}

float CUBScratchArena::get_memory_usage() const noexcept {
    return static_cast<float>(cub_temp_bytes) / static_cast<float>(1ULL << 30U);
}

void CUBScratchArena::convert_mask_to_indices(
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> indices,
    SizeType n_leaves) {
    auto counting_it = thrust::make_counting_iterator<uint32_t>(0);
    cuda_utils::check_cuda_call(
        cub::DeviceSelect::Flagged(
            cub_temp_storage, cub_temp_bytes, counting_it,
            validation_mask.data(), indices.data(), d_reduce_out,
            static_cast<::cuda::std::int64_t>(n_leaves), m_stream),
        "cub::DeviceSelect::Flagged failed");
}

BranchingWorkspaceCUDA::BranchingWorkspaceCUDA(SizeType batch_size,
                                               SizeType branch_max,
                                               SizeType nparams)
    : scratch_params(batch_size * nparams * branch_max),
      scratch_dparams(batch_size * nparams),
      scratch_counts(batch_size * nparams),
      leaf_branch_count(batch_size),
      leaf_output_offset(batch_size) {}

BranchingWorkspaceCUDAView BranchingWorkspaceCUDA::get_view() noexcept {
    return BranchingWorkspaceCUDAView{
        .scratch_params    = thrust::raw_pointer_cast(scratch_params.data()),
        .scratch_dparams   = thrust::raw_pointer_cast(scratch_dparams.data()),
        .scratch_counts    = thrust::raw_pointer_cast(scratch_counts.data()),
        .leaf_branch_count = thrust::raw_pointer_cast(leaf_branch_count.data()),
        .leaf_output_offset =
            thrust::raw_pointer_cast(leaf_output_offset.data()),
    };
}

float BranchingWorkspaceCUDA::get_memory_usage() const noexcept {
    const auto total_memory = (scratch_params.size() * sizeof(double)) +
                              (scratch_dparams.size() * sizeof(double)) +
                              (scratch_counts.size() * sizeof(uint32_t)) +
                              (leaf_branch_count.size() * sizeof(uint32_t)) +
                              (leaf_output_offset.size() * sizeof(uint32_t));

    return static_cast<float>(total_memory) / static_cast<float>(1ULL << 30U);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneWorkspaceCUDA<FoldTypeCUDA>::PruneWorkspaceCUDA(SizeType batch_size,
                                                     SizeType branch_max,
                                                     SizeType nparams,
                                                     SizeType nbins)
    : batch_size(batch_size),
      branch_max(branch_max),
      nparams(nparams),
      nbins(nbins),
      max_branched_leaves(batch_size * branch_max),
      leaves_stride((nparams + 2) * kLeavesParamStride),
      folds_stride(2 * nbins),
      branched_leaves_d(max_branched_leaves * leaves_stride),
      branched_folds_d(max_branched_leaves * folds_stride),
      branched_scores_d(max_branched_leaves),
      branched_indices_d(max_branched_leaves),
      branched_param_idx_d(max_branched_leaves),
      branched_phase_shift_d(max_branched_leaves),
      validation_mask_d(max_branched_leaves) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float PruneWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage() const noexcept {
    const auto total_memory = (branched_leaves_d.size() * sizeof(double)) +
                              (branched_folds_d.size() * sizeof(FoldTypeCUDA)) +
                              (branched_scores_d.size() * sizeof(float)) +
                              (branched_indices_d.size() * sizeof(uint32_t)) +
                              (branched_param_idx_d.size() * sizeof(uint32_t)) +
                              (branched_phase_shift_d.size() * sizeof(float)) +
                              (validation_mask_d.size() * sizeof(uint8_t));

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
    if (d_ptr != nullptr) {
        cudaFree(d_ptr);
    }
    if (h_ptr != nullptr) {
        cudaFreeHost(h_ptr);
    }
}

void DeviceCounter::reset(cudaStream_t stream) { // NOLINT
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(d_ptr, 0, sizeof(uint32_t), stream),
        "Failed to reset DeviceCounter");
}

uint32_t DeviceCounter::value_sync(cudaStream_t stream) { // NOLINT
    cuda_utils::check_cuda_call(cudaMemcpyAsync(h_ptr, d_ptr, sizeof(uint32_t),
                                                cudaMemcpyDeviceToHost, stream),
                                "Failed to copy DeviceCounter value to host");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed in value_sync");
    return *h_ptr;
}

// Explicit instantiation of PruneWorkspaceCUDA
template class PruneWorkspaceCUDA<float>;
template class PruneWorkspaceCUDA<ComplexTypeCUDA>;

} // namespace loki::utils