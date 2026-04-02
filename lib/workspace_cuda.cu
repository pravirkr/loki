#include "loki/utils/workspace.hpp"

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <utility>

#include "loki/core/taylor_ffa.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::memory {

// --- FFAWorkspaceCUDA implementation ---
template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFAWorkspaceCUDA<FoldTypeCUDA>::FFAWorkspaceCUDA(
    const plans::FFAPlan<HostFoldT>& ffa_plan) {
    const auto n_levels     = ffa_plan.get_n_levels();
    const auto n_params     = ffa_plan.get_n_params();
    const auto buffer_size  = ffa_plan.get_buffer_size();
    const auto coord_size   = ffa_plan.get_coord_size();
    const bool is_freq_only = n_params == 1;
    fold_internal_d.resize(buffer_size, DeviceFoldT{});
    m_param_counts_d.resize(n_levels * n_params);
    m_ncoords_offsets_d.resize(n_levels + 1);
    m_param_limits_d.resize(n_params);

    if (is_freq_only) {
        coords_freq_d.resize(coord_size);
    } else {
        coords_d.resize(coord_size);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFAWorkspaceCUDA<FoldTypeCUDA>::FFAWorkspaceCUDA(SizeType buffer_size,
                                                 SizeType coord_size,
                                                 SizeType n_levels,
                                                 SizeType n_params) {
    const bool is_freq_only = n_params == 1;
    fold_internal_d.resize(buffer_size, DeviceFoldT{});
    m_param_counts_d.resize(n_levels * n_params);
    m_ncoords_offsets_d.resize(n_levels + 1);
    m_param_limits_d.resize(n_params);

    if (is_freq_only) {
        coords_freq_d.resize(coord_size);
    } else {
        coords_d.resize(coord_size);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAWorkspaceCUDA<FoldTypeCUDA>::validate(
    const plans::FFAPlan<HostFoldT>& ffa_plan) const {
    const auto n_levels     = ffa_plan.get_n_levels();
    const auto n_params     = ffa_plan.get_n_params();
    const auto buffer_size  = ffa_plan.get_buffer_size();
    const auto coord_size   = ffa_plan.get_coord_size();
    const bool is_freq_only = n_params == 1;

    error_check::check_greater_equal(
        fold_internal_d.size(), buffer_size,
        "FFAWorkspaceCUDA: fold_internal buffer too small");
    error_check::check_greater_equal(
        m_param_counts_d.size(), n_levels * n_params,
        "FFAWorkspaceCUDA: param counts buffer too small");
    error_check::check_greater_equal(
        m_ncoords_offsets_d.size(), n_levels + 1,
        "FFAWorkspaceCUDA: ncoords offsets buffer too small");
    if (is_freq_only) {
        error_check::check_greater_equal(coords_freq_d.idx.size(), coord_size,
                                         "FFAWorkspaceCUDA: coordinates "
                                         "not allocated for enough levels");
    } else {
        error_check::check_greater_equal(coords_d.i_tail.size(), coord_size,
                                         "FFAWorkspaceCUDA: coordinates "
                                         "not allocated for enough levels");
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAWorkspaceCUDA<FoldTypeCUDA>::resolve_coordinates_freq(
    const plans::FFAPlan<HostFoldT>& ffa_plan, cudaStream_t stream) {
    copy_plan_to_device(ffa_plan, stream);

    const auto n_levels      = ffa_plan.get_n_levels();
    const auto ncoords_total = ffa_plan.get_coord_size();
    const auto tseg_brute    = ffa_plan.get_config().get_tseg_brute();
    const auto nbins         = ffa_plan.get_config().get_nbins();
    auto coord_ptrs          = coords_freq_d.get_raw_ptrs();
    core::ffa_taylor_resolve_freq_batch_cuda(
        cuda_utils::as_span(m_param_counts_d),
        cuda_utils::as_span(m_ncoords_offsets_d),
        cuda_utils::as_span(m_param_limits_d), coord_ptrs, n_levels,
        ncoords_total, tseg_brute, nbins, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAWorkspaceCUDA<FoldTypeCUDA>::resolve_coordinates(
    const plans::FFAPlan<HostFoldT>& ffa_plan, cudaStream_t stream) {
    copy_plan_to_device(ffa_plan, stream);

    const auto n_levels      = ffa_plan.get_n_levels();
    const auto n_params      = ffa_plan.get_n_params();
    const auto ncoords_total = ffa_plan.get_coord_size();
    const auto tseg_brute    = ffa_plan.get_config().get_tseg_brute();
    const auto nbins         = ffa_plan.get_config().get_nbins();
    auto coord_ptrs          = coords_d.get_raw_ptrs();
    // Tail coordinates
    core::ffa_taylor_resolve_poly_batch_cuda(
        cuda_utils::as_span(m_param_counts_d),
        cuda_utils::as_span(m_ncoords_offsets_d),
        cuda_utils::as_span(m_param_limits_d), coord_ptrs, n_levels,
        ncoords_total, 0, tseg_brute, nbins, n_params, stream);

    // Head coordinates
    core::ffa_taylor_resolve_poly_batch_cuda(
        cuda_utils::as_span(m_param_counts_d),
        cuda_utils::as_span(m_ncoords_offsets_d),
        cuda_utils::as_span(m_param_limits_d), coord_ptrs, n_levels,
        ncoords_total, 1, tseg_brute, nbins, n_params, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAWorkspaceCUDA<FoldTypeCUDA>::copy_plan_to_device(
    const plans::FFAPlan<HostFoldT>& ffa_plan, cudaStream_t stream) {

    const auto param_counts    = ffa_plan.get_param_counts_flat();
    const auto ncoords_offsets = ffa_plan.get_ncoords_offsets();
    const auto param_limits    = ffa_plan.get_config().get_param_limits();

    // Copy resolve ingredients to device
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_param_counts_d.data()),
                        param_counts.data(),
                        param_counts.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync param counts failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ncoords_offsets_d.data()),
                        ncoords_offsets.data(),
                        ncoords_offsets.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync ncoords offsets failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_param_limits_d.data()),
                        param_limits.data(),
                        param_limits.size() * sizeof(ParamLimit),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync param limits failed");

    // MANDATORY sync because param_counts and other host memory goes out of
    // scope on the next line!
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize failed");
}

// --- BranchingWorkspaceCUDA implementation ---
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

void BranchingWorkspaceCUDA::validate(SizeType batch_size,
                                      SizeType branch_max,
                                      SizeType nparams) const {
    error_check::check_equal(
        scratch_params.size(), batch_size * nparams * branch_max,
        "BranchingWorkspaceCUDA: scratch_params size is too small");
    error_check::check_equal(
        scratch_dparams.size(), batch_size * nparams,
        "BranchingWorkspaceCUDA: scratch_dparams size is too small");
    error_check::check_equal(
        scratch_counts.size(), batch_size * nparams,
        "BranchingWorkspaceCUDA: scratch_counts size is too small");
    error_check::check_equal(
        leaf_branch_count.size(), batch_size,
        "BranchingWorkspaceCUDA: leaf_branch_count size is too small");
    error_check::check_equal(
        leaf_output_offset.size(), batch_size,
        "BranchingWorkspaceCUDA: leaf_output_offset size is too small");
}

// --- PruneWorkspaceCUDA implementation ---
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

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneWorkspaceCUDA<FoldTypeCUDA>::validate(SizeType batch_size,
                                                SizeType branch_max) const {
    error_check::check_equal(
        branched_leaves_d.size(), batch_size * branch_max * leaves_stride,
        "PruneWorkspaceCUDA: branched_leaves_d size is too small");
    error_check::check_equal(
        branched_folds_d.size(), batch_size * branch_max * folds_stride,
        "PruneWorkspaceCUDA: branched_folds_d size is too small");
    error_check::check_equal(
        branched_scores_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: branched_scores_d size is too small");
    error_check::check_equal(
        branched_indices_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: branched_indices_d size is too small");
    error_check::check_equal(
        branched_param_idx_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: branched_param_idx_d size is too small");
    error_check::check_equal(
        branched_phase_shift_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: branched_phase_shift_d size is too small");
    error_check::check_equal(
        validation_mask_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: validation_mask_d size is too small");
}

// --- CUBScratchArena implementation ---
CUBScratchArena::CUBScratchArena(SizeType batch_size,
                                 SizeType branch_max,
                                 cudaStream_t stream)
    : stream(stream) {
    const SizeType max_n_leaves = batch_size * branch_max;

    SizeType reduce_bytes = 0;
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
        cudaFreeAsync(cub_temp_storage, stream);
    }
    if (d_reduce_out != nullptr) {
        cudaFreeAsync(d_reduce_out, stream);
    }
}

CUBScratchArena::CUBScratchArena(CUBScratchArena&& other) noexcept
    : cub_temp_storage(std::exchange(other.cub_temp_storage, nullptr)),
      cub_temp_bytes(std::exchange(other.cub_temp_bytes, 0)),
      d_reduce_out(std::exchange(other.d_reduce_out, nullptr)),
      stream(std::exchange(other.stream, nullptr)) {}

CUBScratchArena& CUBScratchArena::operator=(CUBScratchArena&& other) noexcept {
    if (this != &other) {
        // Free current resources before stealing from other
        if (cub_temp_storage != nullptr)
            cudaFreeAsync(cub_temp_storage, stream);
        if (d_reduce_out != nullptr)
            cudaFreeAsync(d_reduce_out, stream);

        cub_temp_storage = std::exchange(other.cub_temp_storage, nullptr);
        cub_temp_bytes   = std::exchange(other.cub_temp_bytes, 0);
        d_reduce_out     = std::exchange(other.d_reduce_out, nullptr);
        stream           = std::exchange(other.stream, nullptr);
    }
    return *this;
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
            static_cast<::cuda::std::int64_t>(n_leaves), stream),
        "cub::DeviceSelect::Flagged failed");
}

// --- EPWorkspaceCUDA implementation ---
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPWorkspaceCUDA<FoldTypeCUDA>::EPWorkspaceCUDA(SizeType batch_size,
                                               SizeType branch_max,
                                               SizeType max_sugg,
                                               SizeType ncoords_ffa,
                                               SizeType nparams,
                                               SizeType nbins,
                                               cudaStream_t stream)
    : world_tree(max_sugg, nparams, nbins, batch_size * branch_max),
      prune(batch_size, branch_max, nparams, nbins),
      branch(batch_size, branch_max, nparams),
      scratch(batch_size, branch_max, stream) {
    seed_leaves_d.resize(ncoords_ffa * world_tree.get_leaves_stride());
    seed_scores_d.resize(ncoords_ffa);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float EPWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage() const noexcept {
    const auto base_gb = world_tree.get_memory_usage() +
                         prune.get_memory_usage() + branch.get_memory_usage() +
                         scratch.get_memory_usage();
    const auto extra_gb =
        static_cast<float>(seed_leaves_d.size() * sizeof(double) +
                           seed_scores_d.size() * sizeof(float)) /
        static_cast<float>(1ULL << 30U);
    return base_gb + extra_gb;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void EPWorkspaceCUDA<FoldTypeCUDA>::validate(SizeType batch_size,
                                             SizeType branch_max,
                                             SizeType max_sugg,
                                             SizeType ncoords_ffa,
                                             SizeType nparams,
                                             SizeType nbins) const {
    const auto leaves_stride = (nparams + 2) * 2;
    error_check::check_greater_equal(
        seed_scores_d.size(), ncoords_ffa,
        "EPWorkspaceCUDA: seed_scores size is too small");
    error_check::check_equal(seed_leaves_d.size(), ncoords_ffa * leaves_stride,
                             "EPWorkspaceCUDA: seed_leaves size is too small");
    world_tree.validate(max_sugg, nparams, nbins, batch_size * branch_max);
    prune.validate(batch_size, branch_max);
    branch.validate(batch_size, branch_max, nparams);
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

template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPWorkspaceCUDA<FoldTypeCUDA>::~EPWorkspaceCUDA() = default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPWorkspaceCUDA<FoldTypeCUDA>::EPWorkspaceCUDA(EPWorkspaceCUDA&&) noexcept =
    default;
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPWorkspaceCUDA<FoldTypeCUDA>&
EPWorkspaceCUDA<FoldTypeCUDA>::operator=(EPWorkspaceCUDA&&) noexcept = default;

// Explicit instantiation
template struct FFAWorkspaceCUDA<float>;
template struct FFAWorkspaceCUDA<ComplexTypeCUDA>;
template struct PruneWorkspaceCUDA<float>;
template struct PruneWorkspaceCUDA<ComplexTypeCUDA>;
template struct EPWorkspaceCUDA<float>;
template struct EPWorkspaceCUDA<ComplexTypeCUDA>;

} // namespace loki::memory