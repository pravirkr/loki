#include "lib/cuda/workspace_cuda.cuh"

#include <algorithm>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "lib/cuda/cub_helpers.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/kernels_cuda.cuh"
#include "lib/cuda/taylor_ffa_cuda.cuh"
#include "lib/detail/error_check.hpp"

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
    fold_internal_d.resize(buffer_size);
    m_param_counts_d.resize(n_levels * n_params);
    m_ncoords_offsets_d.resize(n_levels + 1);
    m_param_limits_d.resize(n_params);

    if (is_freq_only) {
        coords_freq_d.resize(coord_size);
        m_fuse_check_d.resize(3 * n_levels);
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
    fold_internal_d.resize(buffer_size);
    m_param_counts_d.resize(n_levels * n_params);
    m_ncoords_offsets_d.resize(n_levels + 1);
    m_param_limits_d.resize(n_params);

    if (is_freq_only) {
        coords_freq_d.resize(coord_size);
        m_fuse_check_d.resize(3 * n_levels);
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
SizeType FFAWorkspaceCUDA<FoldTypeCUDA>::get_buffers_bytes() const noexcept {
    return (fold_internal_d.size() * sizeof(DeviceFoldT)) +
           (coords_d.i_tail.size() * sizeof(uint32_t)) +
           (coords_d.shift_tail.size() * sizeof(float)) +
           (coords_d.i_head.size() * sizeof(uint32_t)) +
           (coords_d.shift_head.size() * sizeof(float)) +
           (coords_freq_d.idx.size() * sizeof(uint32_t)) +
           (coords_freq_d.shift.size() * sizeof(float));
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType
FFAWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage_bytes() const noexcept {
    return get_buffers_bytes() + (m_param_counts_d.size() * sizeof(uint32_t)) +
           (m_ncoords_offsets_d.size() * sizeof(uint32_t)) +
           (m_fuse_check_d.size() * sizeof(uint32_t)) +
           (m_param_limits_d.size() * sizeof(ParamLimit));
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
    m_param_counts_h.assign(param_counts.begin(), param_counts.end());
    m_ncoords_offsets_h.assign(ncoords_offsets.begin(), ncoords_offsets.end());
    m_param_limits_h.assign(param_limits.begin(), param_limits.end());

    // Pageable sources: each copy returns once its source is staged, and
    // the members outlive this call, so no stream sync is needed.
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_param_counts_d.data()),
                        m_param_counts_h.data(),
                        m_param_counts_h.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync param counts failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_ncoords_offsets_d.data()),
                        m_ncoords_offsets_h.data(),
                        m_ncoords_offsets_h.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync ncoords offsets failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_param_limits_d.data()),
                        m_param_limits_h.data(),
                        m_param_limits_h.size() * sizeof(ParamLimit),
                        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync param limits failed");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::vector<uint32_t> FFAWorkspaceCUDA<FoldTypeCUDA>::check_fuse_levels_freq(
    const plans::FFAPlan<HostFoldT>& ffa_plan, cudaStream_t stream) {
    const auto n_levels = ffa_plan.get_n_levels();
    error_check::check_greater_equal(
        m_fuse_check_d.size(), 3 * n_levels,
        "FFAWorkspaceCUDA: fuse check scratch too small");
    const auto offsets = ffa_plan.get_ncoords_offsets();
    return core::ffa_freq_fuse_check_levels_cuda(
        thrust::raw_pointer_cast(coords_freq_d.idx.data()),
        std::span<const uint32_t>(offsets.data(), n_levels),
        ffa_plan.get_ncoords(), thrust::raw_pointer_cast(m_fuse_check_d.data()),
        stream);
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

SizeType BranchingWorkspaceCUDA::get_memory_usage_bytes() const noexcept {
    return (scratch_params.size() * sizeof(double)) +
           (scratch_dparams.size() * sizeof(double)) +
           (scratch_counts.size() * sizeof(uint32_t)) +
           (leaf_branch_count.size() * sizeof(uint32_t)) +
           (leaf_output_offset.size() * sizeof(uint32_t));
}

float BranchingWorkspaceCUDA::get_memory_usage_gib() const noexcept {
    return static_cast<float>(get_memory_usage_bytes()) /
           static_cast<float>(1ULL << 30U);
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
                                                     SizeType nbins,
                                                     SizeType nsegments)
    : batch_size(batch_size),
      branch_max(branch_max),
      nparams(nparams),
      nbins(nbins),
      nsegments(nsegments),
      max_branched_leaves(batch_size * branch_max),
      max_branched_param_idx(
          std::max(max_branched_leaves, nsegments * batch_size)),
      leaves_stride((nparams + 2) * kLeavesParamStride),
      folds_stride(2 * nbins),
      branched_leaves_d(max_branched_leaves * leaves_stride),
      branched_folds_d(max_branched_leaves * folds_stride),
      branched_scores_d(max_branched_leaves),
      branched_indices_d(max_branched_leaves),
      branched_param_idx_d(max_branched_param_idx),
      branched_phase_shift_d(max_branched_param_idx),
      validation_mask_d(max_branched_leaves),
      filtered_mask_d(max_branched_leaves) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType
PruneWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage_bytes() const noexcept {
    return (branched_leaves_d.size() * sizeof(double)) +
           (branched_folds_d.size() * sizeof(FoldTypeCUDA)) +
           (branched_scores_d.size() * sizeof(float)) +
           (branched_indices_d.size() * sizeof(uint32_t)) +
           (branched_param_idx_d.size() * sizeof(uint32_t)) +
           (branched_phase_shift_d.size() * sizeof(float)) +
           (validation_mask_d.size() * sizeof(uint8_t)) +
           (filtered_mask_d.size() * sizeof(uint8_t));
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float PruneWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage_gib() const noexcept {
    return static_cast<float>(get_memory_usage_bytes()) /
           static_cast<float>(1ULL << 30U);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneWorkspaceCUDA<FoldTypeCUDA>::validate(SizeType batch_size,
                                                SizeType branch_max,
                                                SizeType nsegments) const {
    const auto max_branched_param_idx =
        std::max(batch_size * branch_max, nsegments * batch_size);
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
        branched_param_idx_d.size(), max_branched_param_idx,
        "PruneWorkspaceCUDA: branched_param_idx_d size is too small");
    error_check::check_equal(
        branched_phase_shift_d.size(), max_branched_param_idx,
        "PruneWorkspaceCUDA: branched_phase_shift_d size is too small");
    error_check::check_equal(
        validation_mask_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: validation_mask_d size is too small");
    error_check::check_equal(
        filtered_mask_d.size(), batch_size * branch_max,
        "PruneWorkspaceCUDA: filtered_mask_d size is too small");
}

// --- CUBScratchArena implementation ---
SizeType CUBScratchArena::temp_bytes_for(SizeType max_n_leaves,
                                         cudaStream_t stream) {
    // 1a. DeviceReduce::Sum (uint8 mask → uint32 count via cast iterator)
    auto dummy_cast_it = thrust::make_transform_iterator(
        static_cast<const uint8_t*>(nullptr), cub_helpers::Uint8ToUint32{});
    SizeType reduce_bytes = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Sum(nullptr, reduce_bytes, dummy_cast_it,
                               static_cast<uint32_t*>(nullptr),
                               static_cast<int>(max_n_leaves), stream),
        "cub::DeviceReduce::Sum sizing failed");

    // 1b. DeviceScan::ExclusiveSum
    SizeType scan_bytes = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
                                      static_cast<uint32_t*>(nullptr),
                                      static_cast<uint32_t*>(nullptr),
                                      static_cast<int>(max_n_leaves), stream),
        "cub::DeviceScan::ExclusiveSum sizing failed");

    // 1c. DeviceSelect::Flagged (counting iterator + uint8 flags)
    SizeType flagged_bytes = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceSelect::Flagged(
            nullptr, flagged_bytes, thrust::make_counting_iterator<uint32_t>(0),
            static_cast<const uint8_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<::cuda::std::int64_t>(max_n_leaves), stream),
        "cub::DeviceSelect::Flagged sizing failed");

    // 1d. DeviceReduce::Reduce (masked min/max over float scores)
    auto dummy_minmax_it = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(0),
        cub_helpers::ScoreToMinMaxFloat{nullptr, nullptr});
    const MinMaxFloat minmax_identity{std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::lowest()};
    SizeType reduce_bytes_minmax = 0;
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Reduce(
            nullptr, reduce_bytes_minmax, dummy_minmax_it,
            static_cast<MinMaxFloat*>(nullptr), static_cast<int>(max_n_leaves),
            cub_helpers::MinMaxReduce{}, minmax_identity, stream),
        "cub::DeviceReduce::Reduce sizing failed");

    return std::max(
        {reduce_bytes, scan_bytes, flagged_bytes, reduce_bytes_minmax});
}

CUBScratchArena::CUBScratchArena(SizeType batch_size,
                                 SizeType branch_max,
                                 cudaStream_t stream)
    : max_n_leaves(batch_size * branch_max) {
    // A throwing constructor runs no destructor: free what was allocated.
    try {
        // One buffer large enough for every operation.
        cub_temp_bytes = temp_bytes_for(max_n_leaves, stream);
        cuda_utils::check_cuda_call(
            cudaMallocAsync(&cub_temp_storage, cub_temp_bytes, stream),
            "cudaMallocAsync cub_temp_storage failed");
        // ---- 3. Allocate device-side output scalars ------------------------
        cuda_utils::check_cuda_call(
            cudaMallocAsync(&d_reduce_out, sizeof(uint32_t), stream),
            "cudaMallocAsync d_reduce_out failed");
        cuda_utils::check_cuda_call(
            cudaMallocAsync(&d_minmax_out, sizeof(MinMaxFloat), stream),
            "cudaMallocAsync d_minmax_out failed");
        cuda_utils::check_cuda_call(
            cudaMallocHost(&h_scalars, kNumHostSlots * sizeof(uint32_t)),
            "cudaMallocHost h_scalars failed");
        cuda_utils::check_cuda_call(
            cudaMallocHost(&h_minmax, sizeof(MinMaxFloat)),
            "cudaMallocHost h_minmax failed");
        cuda_utils::check_cuda_call(
            cudaEventCreateWithFlags(&minmax_ready, cudaEventDisableTiming),
            "cudaEventCreate minmax_ready failed");
    } catch (...) {
        release();
        throw;
    }
}

void CUBScratchArena::release() noexcept {
    // Use synchronous cudaFree (not cudaFreeAsync) so that destruction is
    // safe regardless of whether the original construction stream is still
    // alive. Callers must ensure no in-flight GPU work uses these pointers
    // at the point of destruction.
    if (cub_temp_storage != nullptr) {
        cudaFree(cub_temp_storage);
    }
    if (d_reduce_out != nullptr) {
        cudaFree(d_reduce_out);
    }
    if (d_minmax_out != nullptr) {
        cudaFree(d_minmax_out);
    }
    if (h_scalars != nullptr) {
        cudaFreeHost(h_scalars);
    }
    if (h_minmax != nullptr) {
        cudaFreeHost(h_minmax);
    }
    if (minmax_ready != nullptr) {
        cudaEventDestroy(minmax_ready);
    }
}

CUBScratchArena::~CUBScratchArena() { release(); }

CUBScratchArena::CUBScratchArena(CUBScratchArena&& other) noexcept
    : cub_temp_storage(std::exchange(other.cub_temp_storage, nullptr)),
      cub_temp_bytes(std::exchange(other.cub_temp_bytes, 0)),
      d_reduce_out(std::exchange(other.d_reduce_out, nullptr)),
      d_minmax_out(std::exchange(other.d_minmax_out, nullptr)),
      max_n_leaves(std::exchange(other.max_n_leaves, 0)),
      h_scalars(std::exchange(other.h_scalars, nullptr)),
      h_minmax(std::exchange(other.h_minmax, nullptr)),
      minmax_ready(std::exchange(other.minmax_ready, nullptr)) {}

CUBScratchArena& CUBScratchArena::operator=(CUBScratchArena&& other) noexcept {
    if (this != &other) {
        // Free current resources before stealing from other
        release();
        cub_temp_storage = std::exchange(other.cub_temp_storage, nullptr);
        cub_temp_bytes   = std::exchange(other.cub_temp_bytes, 0);
        d_reduce_out     = std::exchange(other.d_reduce_out, nullptr);
        d_minmax_out     = std::exchange(other.d_minmax_out, nullptr);
        max_n_leaves     = std::exchange(other.max_n_leaves, 0);
        h_scalars        = std::exchange(other.h_scalars, nullptr);
        h_minmax         = std::exchange(other.h_minmax, nullptr);
        minmax_ready     = std::exchange(other.minmax_ready, nullptr);
    }
    return *this;
}

SizeType CUBScratchArena::get_memory_usage_bytes() const noexcept {
    return cub_temp_bytes + sizeof(uint32_t) + sizeof(MinMaxFloat);
}

float CUBScratchArena::get_memory_usage_gib() const noexcept {
    return static_cast<float>(get_memory_usage_bytes()) /
           static_cast<float>(1ULL << 30U);
}

void CUBScratchArena::convert_mask_to_indices(
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> indices,
    SizeType n_leaves,
    cudaStream_t stream) {
    auto counting_it = thrust::make_counting_iterator<uint32_t>(0);
    cuda_utils::check_cuda_call(
        cub::DeviceSelect::Flagged(
            cub_temp_storage, cub_temp_bytes, counting_it,
            validation_mask.data(), indices.data(), d_reduce_out,
            static_cast<::cuda::std::int64_t>(n_leaves), stream),
        "cub::DeviceSelect::Flagged failed");
}

void CUBScratchArena::compute_min_max_scores_async(
    cuda::std::span<const float> scores,
    cuda::std::span<const uint8_t> mask,
    SizeType n_leaves,
    cudaStream_t stream) {
    auto counting_it  = thrust::make_counting_iterator<int>(0);
    auto transform_it = thrust::make_transform_iterator(
        counting_it,
        cub_helpers::ScoreToMinMaxFloat{scores.data(), mask.data()});
    const MinMaxFloat identity{std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::lowest()};
    cuda_utils::check_cuda_call(
        cub::DeviceReduce::Reduce(
            cub_temp_storage, cub_temp_bytes, transform_it, d_minmax_out,
            static_cast<int>(n_leaves), cub_helpers::MinMaxReduce{}, identity,
            stream),
        "cub::DeviceReduce::Reduce failed");
    cuda_utils::check_cuda_call(cudaMemcpyAsync(h_minmax, d_minmax_out,
                                                sizeof(MinMaxFloat),
                                                cudaMemcpyDeviceToHost, stream),
                                "cudaMemcpyAsync minmax out failed");
    cuda_utils::check_cuda_call(cudaEventRecord(minmax_ready, stream),
                                "cudaEventRecord minmax_ready failed");
}

MinMaxFloat CUBScratchArena::wait_min_max() const {
    cuda_utils::check_cuda_call(cudaEventSynchronize(minmax_ready),
                                "cudaEventSynchronize minmax_ready failed");
    return *h_minmax;
}

// --- EPWorkspaceCUDA implementation ---
template <SupportedFoldTypeCUDA FoldTypeCUDA>
EPWorkspaceCUDA<FoldTypeCUDA>::EPWorkspaceCUDA(SizeType batch_size,
                                               SizeType branch_max,
                                               SizeType max_sugg,
                                               SizeType ncoords_ffa,
                                               SizeType nparams,
                                               SizeType nbins,
                                               SizeType nsegments,
                                               cudaStream_t stream)
    : world_tree(max_sugg, nparams, nbins, batch_size * branch_max),
      prune(batch_size, branch_max, nparams, nbins, nsegments),
      branch(batch_size, branch_max, nparams),
      scratch(batch_size, branch_max, stream) {
    seed_leaves_d.resize(ncoords_ffa * world_tree.get_leaves_stride());
    seed_scores_d.resize(ncoords_ffa);
    idx_segments_d.resize(nsegments);
    coord_segments_d.resize(nsegments);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float EPWorkspaceCUDA<FoldTypeCUDA>::get_seed_memory_usage_gib()
    const noexcept {
    const auto bytes = (seed_leaves_d.size() * sizeof(double)) +
                       (seed_scores_d.size() * sizeof(float));
    return static_cast<float>(bytes) / static_cast<float>(1ULL << 30U);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float EPWorkspaceCUDA<FoldTypeCUDA>::get_segment_coords_memory_usage_gib()
    const noexcept {
    const auto bytes =
        (idx_segments_d.size() * sizeof(uint32_t)) +
        (coord_segments_d.size() * sizeof(cuda::std::pair<double, double>));
    return static_cast<float>(bytes) / static_cast<float>(1ULL << 30U);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType
EPWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage_bytes() const noexcept {
    return world_tree.get_memory_usage_bytes() +
           prune.get_memory_usage_bytes() + branch.get_memory_usage_bytes() +
           scratch.get_memory_usage_bytes() +
           (seed_leaves_d.size() * sizeof(double)) +
           (seed_scores_d.size() * sizeof(float)) +
           (idx_segments_d.size() * sizeof(uint32_t)) +
           (coord_segments_d.size() * sizeof(cuda::std::pair<double, double>));
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float EPWorkspaceCUDA<FoldTypeCUDA>::get_memory_usage_gib() const noexcept {
    return static_cast<float>(get_memory_usage_bytes()) /
           static_cast<float>(1ULL << 30U);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void EPWorkspaceCUDA<FoldTypeCUDA>::validate(SizeType batch_size,
                                             SizeType branch_max,
                                             SizeType max_sugg,
                                             SizeType ncoords_ffa,
                                             SizeType nparams,
                                             SizeType nbins,
                                             SizeType nsegments) const {
    const auto leaves_stride = (nparams + 2) * 2;
    error_check::check_greater_equal(
        seed_scores_d.size(), ncoords_ffa,
        "EPWorkspaceCUDA: seed_scores size is too small");
    error_check::check_equal(seed_leaves_d.size(), ncoords_ffa * leaves_stride,
                             "EPWorkspaceCUDA: seed_leaves size is too small");
    world_tree.validate(max_sugg, nparams, nbins, batch_size * branch_max);
    prune.validate(batch_size, branch_max, nsegments);
    branch.validate(batch_size, branch_max, nparams);
}

// DeviceCounter implementation
DeviceCounter::DeviceCounter() {
    d_ptr = nullptr;
    h_ptr = nullptr;
    cuda_utils::check_cuda_call(
        cudaMalloc(&d_ptr, sizeof(uint32_t)),
        "Failed to allocate device memory for DeviceCounter");
    try {
        cuda_utils::check_cuda_call(
            cudaMallocHost(&h_ptr, sizeof(uint32_t)),
            "Failed to allocate pinned memory for DeviceCounter");
    } catch (...) {
        cudaFree(d_ptr);
        d_ptr = nullptr;
        throw;
    }
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

// --- GPU storage factories for the public workspace handles ---

namespace detail {

template <SupportedFoldType FoldType>
std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu(const plans::FFAPlan<FoldType>& ffa_plan,
                       int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    return std::make_unique<FFAWorkspaceCudaStorage<CudaFoldType<FoldType>>>(
        ffa_plan);
}

template <SupportedFoldType FoldType>
std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu(SizeType buffer_size,
                       SizeType coord_size,
                       SizeType n_levels,
                       SizeType n_params,
                       int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    return std::make_unique<FFAWorkspaceCudaStorage<CudaFoldType<FoldType>>>(
        buffer_size, coord_size, n_levels, n_params);
}

template <SupportedFoldType FoldType>
std::unique_ptr<loki::detail::DeviceStorage>
make_ep_workspace_gpu(SizeType batch_size,
                      SizeType branch_max,
                      SizeType max_sugg,
                      SizeType ncoords_ffa,
                      SizeType nparams,
                      SizeType nbins,
                      SizeType nsegments,
                      int device_id) {
    cuda_utils::CudaSetDeviceGuard device_guard(device_id);
    // Allocated on the default stream; EPMultiPass on this workspace runs
    // on its own stream, which the default stream orders before.
    return std::make_unique<EPWorkspaceCudaStorage<CudaFoldType<FoldType>>>(
        batch_size, branch_max, max_sugg, ncoords_ffa, nparams, nbins,
        nsegments, /*stream=*/nullptr);
}

template std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu<float>(const plans::FFAPlan<float>&, int);
template std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu<ComplexType>(const plans::FFAPlan<ComplexType>&, int);
template std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu<float>(SizeType, SizeType, SizeType, SizeType, int);
template std::unique_ptr<loki::detail::DeviceStorage>
make_ffa_workspace_gpu<ComplexType>(
    SizeType, SizeType, SizeType, SizeType, int);
template std::unique_ptr<loki::detail::DeviceStorage>
make_ep_workspace_gpu<float>(
    SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, int);
template std::unique_ptr<loki::detail::DeviceStorage>
make_ep_workspace_gpu<ComplexType>(
    SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType, int);

} // namespace detail

} // namespace loki::memory
