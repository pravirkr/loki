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

template <SupportedFoldType FoldType> struct PruneWorkspace {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size;
    SizeType branch_max;
    SizeType nparams;
    SizeType nbins;
    SizeType max_branched_leaves;
    SizeType leaves_stride;
    SizeType folds_stride;

    std::vector<double> branched_leaves;
    std::vector<FoldType> branched_folds;
    std::vector<float> branched_scores;
    // Scratch space for indices
    std::vector<SizeType> branched_indices;
    // Scratch space for resolving parameters
    std::vector<SizeType> branched_param_idx;
    std::vector<float> branched_phase_shift;

    PruneWorkspace(SizeType batch_size,
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
          branched_leaves(max_branched_leaves * leaves_stride),
          branched_folds(max_branched_leaves * folds_stride),
          branched_scores(max_branched_leaves),
          branched_indices(max_branched_leaves),
          branched_param_idx(max_branched_leaves),
          branched_phase_shift(max_branched_leaves) {}

    float get_memory_usage() const noexcept {
        const auto total_memory =
            (branched_leaves.size() * sizeof(double)) +
            (branched_folds.size() * sizeof(FoldType)) +
            (branched_scores.size() * sizeof(float)) +
            (branched_indices.size() * sizeof(SizeType)) +
            (branched_param_idx.size() * sizeof(SizeType)) +
            (branched_phase_shift.size() * sizeof(float));
        return static_cast<float>(total_memory) /
               static_cast<float>(1ULL << 30U);
    }
}; // End PruneWorkspace definition

template <SupportedFoldType FoldType> struct EPWorkspace {
    PruneWorkspace<FoldType> prune;
    BranchingWorkspace branch;

    EPWorkspace(SizeType batch_size,
                SizeType branch_max,
                SizeType nparams,
                SizeType nbins)
        : prune(batch_size, branch_max, nparams, nbins),
          branch(batch_size, branch_max, nparams) {}

    [[nodiscard]] float get_memory_usage() const noexcept {
        return prune.get_memory_usage() + branch.get_memory_usage();
    }
}; // End EPWorkspace definition

#ifdef LOKI_ENABLE_CUDA

struct CUBScratchArena {
    void* cub_temp_storage  = nullptr;
    SizeType cub_temp_bytes = 0;
    uint32_t* d_reduce_out  = nullptr;
    cudaStream_t m_stream   = nullptr;

    CUBScratchArena(SizeType batch_size,
                    SizeType branch_max,
                    cudaStream_t stream);
    ~CUBScratchArena();
    CUBScratchArena(const CUBScratchArena&)                = delete;
    CUBScratchArena& operator=(const CUBScratchArena&)     = delete;
    CUBScratchArena(CUBScratchArena&&) noexcept            = delete;
    CUBScratchArena& operator=(CUBScratchArena&&) noexcept = delete;

    [[nodiscard]] float get_memory_usage() const noexcept;

    void convert_mask_to_indices(cuda::std::span<const uint8_t> validation_mask,
                                 cuda::std::span<uint32_t> indices,
                                 SizeType n_leaves);
};

struct BranchingWorkspaceCUDAView {
    double* __restrict__ scratch_params;
    double* __restrict__ scratch_dparams;
    uint32_t* __restrict__ scratch_counts;
    uint32_t* __restrict__ leaf_branch_count;
    uint32_t* __restrict__ leaf_output_offset;
};

struct BranchingWorkspaceCUDA {
    thrust::device_vector<double> scratch_params;
    thrust::device_vector<double> scratch_dparams;
    thrust::device_vector<uint32_t> scratch_counts;
    thrust::device_vector<uint32_t> leaf_branch_count;
    thrust::device_vector<uint32_t> leaf_output_offset;

    BranchingWorkspaceCUDA(SizeType batch_size,
                           SizeType branch_max,
                           SizeType nparams);
    [[nodiscard]] BranchingWorkspaceCUDAView get_view() noexcept;
    [[nodiscard]] float get_memory_usage() const noexcept;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct PruneWorkspaceCUDA {
    constexpr static SizeType kLeavesParamStride = 2;
    SizeType batch_size;
    SizeType branch_max;
    SizeType nparams;
    SizeType nbins;
    SizeType max_branched_leaves;
    SizeType leaves_stride;
    SizeType folds_stride;

    thrust::device_vector<double> branched_leaves_d;
    thrust::device_vector<FoldTypeCUDA> branched_folds_d;
    thrust::device_vector<float> branched_scores_d;
    // Scratch space for indices
    thrust::device_vector<uint32_t> branched_indices_d;
    // Scratch space for resolving parameters
    thrust::device_vector<uint32_t> branched_param_idx_d;
    thrust::device_vector<float> branched_phase_shift_d;
    // Scratch space for validation mask
    thrust::device_vector<uint8_t> validation_mask_d;

    PruneWorkspaceCUDA(SizeType batch_size,
                       SizeType branch_max,
                       SizeType nparams,
                       SizeType nbins);

    [[nodiscard]] float get_memory_usage() const noexcept;
};

template <SupportedFoldTypeCUDA FoldTypeCUDA> struct EPWorkspaceCUDA {
    PruneWorkspaceCUDA<FoldTypeCUDA> prune;
    BranchingWorkspaceCUDA branch;
    CUBScratchArena scratch;

    EPWorkspaceCUDA(SizeType batch_size,
                    SizeType branch_max,
                    SizeType nparams,
                    SizeType nbins,
                    cudaStream_t stream = nullptr)
        : prune(batch_size, branch_max, nparams, nbins),
          branch(batch_size, branch_max, nparams),
          scratch(batch_size, branch_max, stream) {}

    [[nodiscard]] float get_memory_usage() const noexcept {
        return prune.get_memory_usage() + branch.get_memory_usage() +
               scratch.get_memory_usage();
    }
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
    [[nodiscard]] uint32_t* data() noexcept { return d_ptr; } // NOLINT
    [[nodiscard]] const uint32_t* data() const noexcept { return d_ptr; }
    [[nodiscard]] uint32_t value_sync(cudaStream_t stream);
};

#endif // LOKI_ENABLE_CUDA

} // namespace loki::utils