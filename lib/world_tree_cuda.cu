#include "loki/utils/world_tree.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

#include <cuda/std/limits>
#include <cuda/std/span>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/cub_helpers.cuh"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"

namespace loki::memory {
namespace {

// Warp-per-row kernel (for most workloads)
template <SupportedFoldTypeCUDA FoldTypeCUDA>
__global__ void kernel_scatter_to_circular_copy_warp_per_row(
    const double* __restrict__ src_leaves,
    const FoldTypeCUDA* __restrict__ src_folds,
    const float* __restrict__ src_scores,
    const uint32_t* __restrict__ src_indices,
    double* __restrict__ dst_leaves,
    FoldTypeCUDA* __restrict__ dst_folds,
    float* __restrict__ dst_scores,
    uint32_t dst_start_slot,
    uint32_t capacity,
    uint32_t slots_to_write,
    uint32_t leaves_stride,
    uint32_t folds_stride) {
    constexpr uint32_t kWarpSize = 32;

    const uint32_t lane            = threadIdx.x & (kWarpSize - 1);
    const uint32_t warp_in_block   = threadIdx.x / kWarpSize;
    const uint32_t warps_per_block = blockDim.x / kWarpSize;

    const uint32_t global_warp = (blockIdx.x * warps_per_block) + warp_in_block;
    const uint32_t total_warps = gridDim.x * warps_per_block;

    for (uint32_t row = global_warp; row < slots_to_write; row += total_warps) {
        const uint32_t src_idx = src_indices[row];
        uint32_t dst_idx       = dst_start_slot + row;
        if (dst_idx >= capacity) {
            dst_idx -= capacity;
        }
        for (uint32_t j = lane; j < leaves_stride; j += kWarpSize) {
            dst_leaves[(dst_idx * leaves_stride) + j] =
                src_leaves[(src_idx * leaves_stride) + j];
        }
        for (uint32_t j = lane; j < folds_stride; j += kWarpSize) {
            dst_folds[(dst_idx * folds_stride) + j] =
                src_folds[(src_idx * folds_stride) + j];
        }
        // Score: lane 0 only
        if (lane == 0) {
            dst_scores[dst_idx] = src_scores[src_idx];
        }
    }
}

// Multi-warp per-row kernel (for large workloads)
template <SizeType WarpsPerRow, SupportedFoldTypeCUDA FoldTypeCUDA>
__global__ void kernel_scatter_to_circular_copy_multiwarp_per_row(
    const double* __restrict__ src_leaves,
    const FoldTypeCUDA* __restrict__ src_folds,
    const float* __restrict__ src_scores,
    const uint32_t* __restrict__ src_indices,
    double* __restrict__ dst_leaves,
    FoldTypeCUDA* __restrict__ dst_folds,
    float* __restrict__ dst_scores,
    uint32_t dst_start_slot,
    uint32_t capacity,
    uint32_t slots_to_write,
    uint32_t leaves_stride,
    uint32_t folds_stride) {
    constexpr uint32_t kWarpSize = 32;

    const uint32_t lane            = threadIdx.x & (kWarpSize - 1);
    const uint32_t warp_in_block   = threadIdx.x / kWarpSize;
    const uint32_t warps_per_block = blockDim.x / kWarpSize;

    const uint32_t global_warp = (blockIdx.x * warps_per_block) + warp_in_block;
    const uint32_t total_warps = gridDim.x * warps_per_block;

    uint32_t row              = global_warp / WarpsPerRow;
    const uint32_t subwarp    = global_warp % WarpsPerRow;
    const uint32_t row_stride = total_warps / WarpsPerRow;

    for (; row < slots_to_write; row += row_stride) {
        const uint32_t src_idx = src_indices[row];
        uint32_t dst_idx       = dst_start_slot + row;
        if (dst_idx >= capacity) {
            dst_idx -= capacity;
        }
        if (subwarp == 0) {
            for (uint32_t j = lane; j < leaves_stride; j += kWarpSize) {
                dst_leaves[(dst_idx * leaves_stride) + j] =
                    src_leaves[(src_idx * leaves_stride) + j];
            }
            // Score: lane 0 only
            if (lane == 0) {
                dst_scores[dst_idx] = src_scores[src_idx];
            }
        }
        for (uint32_t j = (subwarp * kWarpSize) + lane; j < folds_stride;
             j += (WarpsPerRow * kWarpSize)) {
            dst_folds[(dst_idx * folds_stride) + j] =
                src_folds[(src_idx * folds_stride) + j];
        }
    }
}

template <typename T>
__global__ void compact_in_place_kernel(const uint8_t* __restrict__ keep,
                                        const uint32_t* __restrict__ prefix,
                                        T* __restrict__ data,
                                        uint32_t start,
                                        uint32_t capacity,
                                        uint32_t size,
                                        uint32_t stride) {
    const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= size || keep[i] == 0) {
        return;
    }

    const uint32_t dst_logical = prefix[i];
    const uint32_t src_lin     = start + i;
    const uint32_t dst_lin     = start + dst_logical;
    const uint32_t src_phys = src_lin < capacity ? src_lin : src_lin - capacity;
    const uint32_t dst_phys = dst_lin < capacity ? dst_lin : dst_lin - capacity;
    // Copy full row
    for (uint32_t j = 0; j < stride; ++j) {
        data[(dst_phys * stride) + j] = data[(src_phys * stride) + j];
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void scatter_to_circular_copy_cuda(const double* __restrict__ src_leaves,
                                   const FoldTypeCUDA* __restrict__ src_folds,
                                   const float* __restrict__ src_scores,
                                   const uint32_t* __restrict__ src_indices,
                                   double* __restrict__ dst_leaves,
                                   FoldTypeCUDA* __restrict__ dst_folds,
                                   float* __restrict__ dst_scores,
                                   uint32_t dst_start_slot,
                                   uint32_t capacity,
                                   uint32_t slots_to_write,
                                   uint32_t leaves_stride,
                                   uint32_t folds_stride,
                                   cudaStream_t stream) {
    constexpr SizeType kThreadsPerBlock = 256; // 8 warps
    constexpr SizeType kWarpsPerBlock   = kThreadsPerBlock / 32;

    SizeType warps_per_row = 1;
    if (folds_stride > 512) {
        warps_per_row = 2;
    }
    if (folds_stride > 1024) {
        warps_per_row = 4;
    }

    const SizeType work_warps = slots_to_write * warps_per_row;
    const SizeType blocks_per_grid =
        (work_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 block_dim(kThreadsPerBlock);
    const dim3 grid_dim(blocks_per_grid);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    if (warps_per_row == 4) {
        kernel_scatter_to_circular_copy_multiwarp_per_row<4, FoldTypeCUDA>
            <<<grid_dim, block_dim, 0, stream>>>(
                src_leaves, src_folds, src_scores, src_indices, dst_leaves,
                dst_folds, dst_scores, dst_start_slot, capacity, slots_to_write,
                leaves_stride, folds_stride);
    } else if (warps_per_row == 2) {
        kernel_scatter_to_circular_copy_multiwarp_per_row<2, FoldTypeCUDA>
            <<<grid_dim, block_dim, 0, stream>>>(
                src_leaves, src_folds, src_scores, src_indices, dst_leaves,
                dst_folds, dst_scores, dst_start_slot, capacity, slots_to_write,
                leaves_stride, folds_stride);
    } else {
        kernel_scatter_to_circular_copy_warp_per_row<FoldTypeCUDA>
            <<<grid_dim, block_dim, 0, stream>>>(
                src_leaves, src_folds, src_scores, src_indices, dst_leaves,
                dst_folds, dst_scores, dst_start_slot, capacity, slots_to_write,
                leaves_stride, folds_stride);
    }
    cuda_utils::check_last_cuda_error("scatter_to_circular_copy_cuda failed");
}

struct ScoreAboveThresholdFunctor {
    const float* __restrict__ scores_ptr;
    float threshold;

    __host__ __device__ bool operator()(uint32_t idx) const {
        return scores_ptr[idx] >= threshold;
    }
};

struct CircularMaskFunctor {
    const float* __restrict__ scores_ptr;
    uint32_t start;
    uint32_t capacity;
    float threshold;

    __host__ __device__ uint8_t operator()(uint32_t logical_idx) const {
        const uint32_t x    = start + logical_idx;
        const uint32_t phys = (x < capacity) ? x : x - capacity;
        return (scores_ptr[phys] > threshold) ? 1 : 0;
    }
};
} // namespace

/**
 * @brief GPU-resident circular buffer for world tree data
 *
 * This class maintains all data in GPU global memory and provides
 * the same circular buffer semantics as the CPU version.
 *
 * Layout (all indices modulo m_capacity):
 *
 * ┌────────┬───────────────┬───────────────┐
 * │ unused │  READ REGION  │  WRITE REGION │
 * └────────┴───────────────┴───────────────┘
 *            ^ m_head         ^ m_write_start
 *            size = m_size_old      size = m_size
 *
 * @tparam FoldTypeCUDA Either float or ComplexTypeCUDA
 */
template <SupportedFoldTypeCUDA FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>::WorldTreeCUDA(SizeType capacity,
                                           SizeType nparams,
                                           SizeType nbins,
                                           SizeType max_batch_size)
    : m_capacity(capacity),
      m_nparams(nparams),
      m_nbins(nbins),
      m_max_batch_size(max_batch_size),
      m_leaves_stride((nparams + 2) * kParamStride),
      m_folds_stride(2 * nbins),
      m_leaves(m_capacity * m_leaves_stride, 0.0),
      m_folds(m_capacity * m_folds_stride, FoldTypeCUDA{}),
      m_scores(m_capacity, 0.0F),
      m_scratch_leaves(m_capacity * (nparams * kParamStride), 0.0),
      m_scratch_scores((m_capacity + max_batch_size), 0.0F),
      m_scratch_pending_indices(max_batch_size, 0),
      m_scratch_prefix(m_capacity, 0),
      m_scratch_mask(m_capacity, 0) {
    // Validate inputs
    error_check::check_greater(m_capacity, SizeType{0},
                               "SuggestionTreeCUDA: capacity must be > 0");
    error_check::check_greater(m_nparams, SizeType{0},
                               "SuggestionTreeCUDA: nparams must be > 0");
    error_check::check_greater(m_nbins, SizeType{0},
                               "SuggestionTreeCUDA: nbins must be > 0");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_size_lb() const noexcept {
    return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
}

// Get raw span of data
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const double>
WorldTreeCUDA<FoldTypeCUDA>::get_leaves_span() const noexcept {
    return {thrust::raw_pointer_cast(m_leaves.data()), m_leaves.size()};
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const FoldTypeCUDA>
WorldTreeCUDA<FoldTypeCUDA>::get_folds_span() const noexcept {
    return {thrust::raw_pointer_cast(m_folds.data()), m_folds.size()};
}
template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const float>
WorldTreeCUDA<FoldTypeCUDA>::get_scores_span() const noexcept {
    return {thrust::raw_pointer_cast(m_scores.data()), m_scores.size()};
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_score_max() const noexcept {
    if (m_size == 0) {
        return 0.0F;
    }
    const SizeType start = get_current_start_idx();
    if (start + m_size <= m_capacity) {
        // Contiguous case - direct reduction
        return thrust::reduce(thrust::device, m_scores.begin() + start,
                              m_scores.begin() + start + m_size,
                              cuda::std::numeric_limits<float>::lowest(),
                              ThrustMaxOp<float>());
    } // Wrapped case - two reductions
    const SizeType first_count = m_capacity - start;

    float max1 = thrust::reduce(
        thrust::device, m_scores.begin() + start, m_scores.end(),
        cuda::std::numeric_limits<float>::lowest(), ThrustMaxOp<float>());
    float max2 = thrust::reduce(thrust::device, m_scores.begin(),
                                m_scores.begin() + (m_size - first_count),
                                cuda::std::numeric_limits<float>::lowest(),
                                ThrustMaxOp<float>());
    return std::max(max1, max2);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_score_min() const noexcept {
    if (m_size == 0) {
        return 0.0F;
    }
    const SizeType start = get_current_start_idx();

    if (start + m_size <= m_capacity) {
        return thrust::reduce(thrust::device, m_scores.begin() + start,
                              m_scores.begin() + start + m_size,
                              cuda::std::numeric_limits<float>::max(),
                              ThrustMinOp<float>());
    }
    const SizeType first_count = m_capacity - start;

    float min1 = thrust::reduce(
        thrust::device, m_scores.begin() + start, m_scores.end(),
        cuda::std::numeric_limits<float>::max(), ThrustMinOp<float>());
    float min2 = thrust::reduce(thrust::device, m_scores.begin(),
                                m_scores.begin() + (m_size - first_count),
                                cuda::std::numeric_limits<float>::max(),
                                ThrustMinOp<float>());
    return std::min(min1, min2);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_memory_usage() const noexcept {
    const auto base_bytes =
        (m_leaves.size() * sizeof(double)) +
        (m_folds.size() * sizeof(FoldTypeCUDA)) +
        (m_scores.size() * sizeof(float)) +
        (m_scratch_leaves.size() * sizeof(double)) +
        (m_scratch_scores.size() * sizeof(float)) +
        (m_scratch_pending_indices.size() * sizeof(uint32_t)) +
        (m_scratch_prefix.size() * sizeof(uint32_t)) +
        (m_scratch_mask.size() * sizeof(uint8_t));

    return static_cast<float>(base_bytes) / static_cast<float>(1ULL << 30U);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::pair<cuda::std::span<const double>, SizeType>
WorldTreeCUDA<FoldTypeCUDA>::get_leaves_span(SizeType n_leaves) const {
    const SizeType available = m_size_old - m_read_consumed;
    error_check::check_less_equal(
        n_leaves, available,
        "get_leaves_span: n_leaves exceeds available space");

    // Compute physical start: relative to current head of read region
    const auto physical_start =
        get_circular_index(m_read_consumed, m_head, m_capacity);
    // Compute how many contiguous elements we can take before wrap
    const auto actual_size = std::min(n_leaves, m_capacity - physical_start);
    // Return span with correct byte size (actual_size elements, not bytes)
    return {cuda::std::span<const double>(
                thrust::raw_pointer_cast(m_leaves.data()) +
                    (physical_start * m_leaves_stride),
                actual_size * m_leaves_stride),
            actual_size};
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::pair<cuda::std::span<double>, SizeType>
WorldTreeCUDA<FoldTypeCUDA>::get_leaves_span(SizeType n_leaves) {
    const SizeType available = m_size_old - m_read_consumed;
    error_check::check_less_equal(
        n_leaves, available,
        "get_leaves_span: n_leaves exceeds available space");

    // Compute physical start: relative to current head of read region
    const auto physical_start =
        get_circular_index(m_read_consumed, m_head, m_capacity);
    // Compute how many contiguous elements we can take before wrap
    const auto actual_size = std::min(n_leaves, m_capacity - physical_start);
    // Return span with correct byte size (actual_size elements, not bytes)
    return {cuda::std::span<double>(thrust::raw_pointer_cast(m_leaves.data()) +
                                        (physical_start * m_leaves_stride),
                                    actual_size * m_leaves_stride),
            actual_size};
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<double> WorldTreeCUDA<FoldTypeCUDA>::get_leaves_contiguous_span(
    cudaStream_t stream) noexcept {
    const auto start_idx     = get_current_start_idx();
    const auto report_stride = m_nparams * kParamStride;
    const SizeType src_pitch = m_leaves_stride * sizeof(double);
    const SizeType dst_pitch = report_stride * sizeof(double);
    const SizeType row_bytes = report_stride * sizeof(double);

    const double* __restrict__ src = thrust::raw_pointer_cast(m_leaves.data());
    double* __restrict__ dst =
        thrust::raw_pointer_cast(m_scratch_leaves.data());

    // Check if data wraps around
    if (start_idx + m_size <= m_capacity) {
        // Contiguous case - single copy
        cuda_utils::check_cuda_call(
            cudaMemcpy2DAsync(dst, dst_pitch, src + start_idx * m_leaves_stride,
                              src_pitch, row_bytes, m_size,
                              cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy2DAsync leaves failed");
    } else {
        // Wrapped case - two contiguous copies
        const auto first_part      = m_capacity - start_idx;
        const SizeType second_part = m_size - first_part;
        cuda_utils::check_cuda_call(
            cudaMemcpy2DAsync(dst, dst_pitch, src + start_idx * m_leaves_stride,
                              src_pitch, row_bytes, first_part,
                              cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy2DAsync leaves failed");
        cuda_utils::check_cuda_call(
            cudaMemcpy2DAsync(dst + first_part * report_stride, dst_pitch, src,
                              src_pitch, row_bytes, second_part,
                              cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy2DAsync leaves failed");
    }
    cuda_utils::check_cuda_call(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize get_leaves_contiguous_span failed");
    return {thrust::raw_pointer_cast(m_scratch_leaves.data()),
            m_size * report_stride};
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<float> WorldTreeCUDA<FoldTypeCUDA>::get_scores_contiguous_span(
    cudaStream_t stream) noexcept {
    const auto start_idx = get_current_start_idx();
    copy_from_circular(
        thrust::raw_pointer_cast(m_scores.data()), start_idx, m_size,
        SizeType{1}, thrust::raw_pointer_cast(m_scratch_scores.data()), stream);
    return {thrust::raw_pointer_cast(m_scratch_scores.data()), m_size};
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_physical_start_idx() const {
    error_check::check(m_is_updating,
                       "WorldTreeCUDA: get_physical_start_idx only valid "
                       "during updates");
    // Compute physical start: relative to current head of read region
    return get_circular_index(m_read_consumed, m_head, m_capacity);
}

// Mutation operations

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::set_size(SizeType size) noexcept {
    m_size     = size;
    m_head     = 0;
    m_size_old = 0;
    error_check::check_less_equal(m_size, m_capacity,
                                  "WorldTreeCUDA: Invalid size after set_size");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::reset() noexcept {
    m_size          = 0;
    m_head          = 0;
    m_size_old      = 0;
    m_is_updating   = false;
    m_read_consumed = 0;
    m_write_head    = 0;
    m_write_start   = 0;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::prepare_in_place_update() {
    error_check::check_equal(
        m_is_updating, false,
        "Cannot prepare for update while already updating");
    m_size_old      = m_size;
    m_write_start   = get_circular_index(m_size, m_head, m_capacity);
    m_write_head    = m_write_start;
    m_size          = 0; // The new size starts at 0
    m_is_updating   = true;
    m_read_consumed = 0;
    validate_circular_buffer_state();
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::finalize_in_place_update() {
    error_check::check_equal(
        m_read_consumed, m_size_old,
        "finalize_in_place_update: not all old data consumed");
    m_head          = m_write_start;
    m_size_old      = 0;
    m_is_updating   = false;
    m_read_consumed = 0;
    m_write_head    = 0; // Reset for safety
    m_write_start   = 0;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::consume_read(SizeType n) {
    error_check::check_less_equal(m_read_consumed + n, m_size_old,
                                  "WorldTreeCUDA: read_consumed overflow");
    m_read_consumed += n;

    // Validate circular buffer invariant
    error_check::check_less_equal(
        m_size_old - m_read_consumed + m_size, m_capacity,
        "WorldTreeCUDA: circular buffer invariant violated");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::add_initial(
    cuda::std::span<const double> leaves_batch,
    cuda::std::span<const FoldTypeCUDA> folds_batch,
    cuda::std::span<const float> scores_batch,
    SizeType slots_to_write,
    cudaStream_t stream) {
    error_check::check_less_equal(
        slots_to_write, m_capacity,
        "WorldTreeCUDA: Suggestions too large to add.");
    error_check::check_equal(slots_to_write, scores_batch.size(),
                             "slots_to_write must match batch_scores size");

    reset(); // Start fresh
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()),
                        leaves_batch.data(),
                        slots_to_write * m_leaves_stride * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync leaves failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()),
                        folds_batch.data(),
                        slots_to_write * m_folds_stride * sizeof(FoldTypeCUDA),
                        cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync folds failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_scores.data()),
                        scores_batch.data(), slots_to_write * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync scores failed");
    m_size = slots_to_write;
    error_check::check_less_equal(
        m_size, m_capacity, "WorldTreeCUDA: Invalid size after add_initial.");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::add_batch_scattered(
    cuda::std::span<const double> leaves_batch,
    cuda::std::span<const FoldTypeCUDA> folds_batch,
    cuda::std::span<const float> scores_batch,
    cuda::std::span<const uint32_t> indices_batch,
    float current_threshold,
    SizeType slots_to_write,
    cudaStream_t stream) {
    if (slots_to_write == 0) {
        return current_threshold;
    }
    double* __restrict__ m_leaves_ptr =
        thrust::raw_pointer_cast(m_leaves.data());
    FoldTypeCUDA* __restrict__ m_folds_ptr =
        thrust::raw_pointer_cast(m_folds.data());
    float* __restrict__ m_scores_ptr =
        thrust::raw_pointer_cast(m_scores.data());

    // Fast path: Check if we have enough space immediately
    SizeType space_left = calculate_space_left();
    if (slots_to_write <= space_left) {
        scatter_to_circular_copy_cuda(leaves_batch.data(), folds_batch.data(),
                                      scores_batch.data(), indices_batch.data(),
                                      m_leaves_ptr, m_folds_ptr, m_scores_ptr,
                                      m_write_head, m_capacity, slots_to_write,
                                      m_leaves_stride, m_folds_stride, stream);
        cuda_utils::check_cuda_call(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize scatter_to_circular_copy_cuda failed");
        m_write_head =
            get_circular_index(slots_to_write, m_write_head, m_capacity);
        m_size += slots_to_write;
        return current_threshold;
    }

    // Slow path: Overflow & Pruning (Union Top-K Strategy)
    error_check::check_less_equal(
        slots_to_write, m_max_batch_size,
        "WorldTreeCUDA: Suggestions too large to add.");
    // Determine the Global Pruning Threshold
    const float effective_threshold = get_prune_threshold(
        scores_batch, slots_to_write, current_threshold, stream);
    // Remove old items that are strictly worse than the new global pruning
    // threshold.
    prune_on_overload(effective_threshold, stream);

    ScoreAboveThresholdFunctor functor{.scores_ptr = scores_batch.data(),
                                       .threshold  = effective_threshold};
    const auto end_it = thrust::copy_if(
        thrust::cuda::par.on(stream), indices_batch.begin(),
        indices_batch.begin() + static_cast<IndexType>(slots_to_write),
        indices_batch.begin(), // stencil = src_idx
        m_scratch_pending_indices.begin(), functor);

    const SizeType pending_count =
        static_cast<SizeType>(end_it - m_scratch_pending_indices.begin());

    space_left          = calculate_space_left();
    const auto n_to_add = std::min(pending_count, space_left);
    if (n_to_add < pending_count) {
        spdlog::warn(
            "WorldTreeCUDA: Unexpected pruning deficit - pending_count={}, "
            "space_left={}, m_size={}, slots_to_write={}, threshold={}",
            pending_count, space_left, m_size, slots_to_write,
            effective_threshold);
    }
    if (n_to_add == 0) {
        return effective_threshold;
    }
    scatter_to_circular_copy_cuda(
        leaves_batch.data(), folds_batch.data(), scores_batch.data(),
        thrust::raw_pointer_cast(m_scratch_pending_indices.data()),
        m_leaves_ptr, m_folds_ptr, m_scores_ptr, m_write_head, m_capacity,
        n_to_add, m_leaves_stride, m_folds_stride, stream);
    cuda_utils::check_cuda_call(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize scatter_to_circular_copy_cuda failed");
    m_write_head = get_circular_index(n_to_add, m_write_head, m_capacity);
    m_size += n_to_add;
    return effective_threshold;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::validate(SizeType capacity,
                                           SizeType nparams,
                                           SizeType nbins,
                                           SizeType max_batch_size) const {
    error_check::check_equal(m_capacity, capacity,
                             "WorldTreeCUDA: capacity mismatch");
    error_check::check_equal(m_nparams, nparams,
                             "WorldTreeCUDA: nparams mismatch");
    error_check::check_equal(m_nbins, nbins, "WorldTreeCUDA: nbins mismatch");
    error_check::check_greater_equal(m_leaves.size(), capacity * (nparams + 2),
                                     "WorldTreeCUDA: leaves size mismatch");
    error_check::check_greater_equal(m_folds.size(), capacity * 2 * nbins,
                                     "WorldTreeCUDA: folds size mismatch");
    error_check::check_greater_equal(m_scores.size(), capacity,
                                     "WorldTreeCUDA: scores size mismatch");
    error_check::check_greater_equal(
        m_scratch_leaves.size(), capacity * nparams,
        "WorldTreeCUDA: scratch_leaves size mismatch");
    error_check::check_greater_equal(
        m_scratch_scores.size(), capacity + max_batch_size,
        "WorldTreeCUDA: scratch_scores size mismatch");
    error_check::check_greater_equal(
        m_scratch_pending_indices.size(), max_batch_size,
        "WorldTreeCUDA: scratch_pending_indices size mismatch");
    error_check::check_greater_equal(
        m_scratch_prefix.size(), capacity,
        "WorldTreeCUDA: scratch_prefix size mismatch");
    error_check::check_greater_equal(
        m_scratch_mask.size(), capacity,
        "WorldTreeCUDA: scratch_mask size mismatch");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
template <typename T>
void WorldTreeCUDA<FoldTypeCUDA>::copy_from_circular(
    const T* __restrict__ src,
    SizeType src_start_slot,
    SizeType slots,
    SizeType stride,
    T* __restrict__ dst,
    cudaStream_t stream) const noexcept {
    const auto first_slots = std::min(slots, m_capacity - src_start_slot);
    const auto first_elems = first_slots * stride;
    const auto src_offset  = src_start_slot * stride;

    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(dst, src + src_offset, first_elems * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync copy_from_circular failed");
    const auto second_slots = slots - first_slots;
    if (second_slots > 0) {
        const auto second_elems = second_slots * stride;
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(dst + first_elems, src, second_elems * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync copy_from_circular failed");
    }
    cuda_utils::check_cuda_call(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize copy_from_circular failed");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::get_current_start_idx() const noexcept {
    return m_is_updating ? m_write_start : m_head;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
constexpr SizeType WorldTreeCUDA<FoldTypeCUDA>::get_circular_index(
    SizeType logical_idx, SizeType start, SizeType capacity) noexcept {
    return (start + logical_idx) % capacity;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType WorldTreeCUDA<FoldTypeCUDA>::calculate_space_left() const noexcept {
    const auto remaining_old = static_cast<IndexType>(m_size_old) -
                               static_cast<IndexType>(m_read_consumed);
    error_check::check_greater_equal(
        remaining_old, 0, "calculate_space_left: Invalid buffer state");

    const auto space_left = static_cast<IndexType>(m_capacity) -
                            (remaining_old + static_cast<IndexType>(m_size));
    error_check::check_greater_equal(
        space_left, 0, "calculate_space_left: Invalid buffer state");
    return static_cast<SizeType>(space_left);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float WorldTreeCUDA<FoldTypeCUDA>::get_prune_threshold(
    cuda::std::span<const float> scores_batch,
    SizeType slots_to_write,
    float current_threshold,
    cudaStream_t stream) noexcept {
    if (slots_to_write == 0) {
        return current_threshold;
    }
    const auto space_left       = calculate_space_left();
    const auto total_candidates = m_size + slots_to_write;
    const auto total_capacity   = m_size + space_left;
    if (total_candidates <= total_capacity) {
        return current_threshold;
    }
    error_check::check_greater_equal(
        m_scratch_scores.size(), total_candidates,
        "get_prune_threshold: Invalid scratch buffer size");
    // Gather all candidates (Buffer + Batch) into scratch
    // Copy active circular buffer scores to scratch [0 ... m_size]
    const auto start_idx = get_current_start_idx();
    copy_from_circular(
        thrust::raw_pointer_cast(m_scores.data()), start_idx, m_size,
        SizeType{1}, thrust::raw_pointer_cast(m_scratch_scores.data()), stream);
    // Append batch scores [m_size ... total]
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(
            thrust::raw_pointer_cast(m_scratch_scores.data()) + m_size,
            thrust::raw_pointer_cast(scores_batch.data()),
            slots_to_write * sizeof(float), cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync get_prune_threshold failed");
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize get_prune_threshold "
                                "append batch scores failed");

    // The item at 'total_capacity-1' (in descending order) is the smallest
    // score we keep (i.e. the threshold).
    thrust::sort(thrust::cuda::par.on(stream), m_scratch_scores.begin(),
                 m_scratch_scores.begin() + total_candidates,
                 cuda::std::greater<float>());
    cuda_utils::check_last_cuda_error("thrust::sort failed");
    const float kth = m_scratch_scores[total_capacity - 1];
    // Median score for severity (restricted range)
    const float mid = m_scratch_scores[total_candidates / 2];
    // To break ties, we use the next representable value greater than the
    // topk and median scores.
    const float topk_threshold =
        std::nextafter(kth, std::numeric_limits<float>::max());
    const float median_threshold =
        std::nextafter(mid, std::numeric_limits<float>::max());
    return std::max({current_threshold, topk_threshold, median_threshold});
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::prune_on_overload(float threshold,
                                                    cudaStream_t stream) {
    error_check::check(
        m_is_updating,
        "WorldTreeCUDA: prune_on_overload only allowed during updates");
    if (m_size == 0) {
        return;
    }

    const SizeType start_idx = get_current_start_idx();
    // Mark survivors (logical indices) in keep mask
    CircularMaskFunctor functor{.scores_ptr =
                                    thrust::raw_pointer_cast(m_scores.data()),
                                .start     = static_cast<uint32_t>(start_idx),
                                .capacity  = static_cast<uint32_t>(m_capacity),
                                .threshold = threshold};
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(m_size),
                      m_scratch_mask.begin(), functor);

    // Exclusive prefix sum to get logical indices to keep
    thrust::exclusive_scan(thrust::cuda::par.on(stream), m_scratch_mask.begin(),
                           m_scratch_mask.begin() + m_size,
                           m_scratch_prefix.begin(), uint32_t{0});

    // In-place compaction (three arrays)
    const dim3 block_dim(256);
    const dim3 grid_dim((m_size + block_dim.x - 1) / block_dim.x);
    cuda_utils::check_kernel_launch_params(grid_dim, block_dim);
    compact_in_place_kernel<<<grid_dim, block_dim, 0, stream>>>(
        thrust::raw_pointer_cast(m_scratch_mask.data()),
        thrust::raw_pointer_cast(m_scratch_prefix.data()),
        thrust::raw_pointer_cast(m_leaves.data()), start_idx, m_capacity,
        m_size, m_leaves_stride);
    cuda_utils::check_last_cuda_error("compact_in_place_kernel leaves failed");
    compact_in_place_kernel<<<grid_dim, block_dim, 0, stream>>>(
        thrust::raw_pointer_cast(m_scratch_mask.data()),
        thrust::raw_pointer_cast(m_scratch_prefix.data()),
        thrust::raw_pointer_cast(m_folds.data()), start_idx, m_capacity, m_size,
        m_folds_stride);
    cuda_utils::check_last_cuda_error("compact_in_place_kernel folds failed");
    compact_in_place_kernel<<<grid_dim, block_dim, 0, stream>>>(
        thrust::raw_pointer_cast(m_scratch_mask.data()),
        thrust::raw_pointer_cast(m_scratch_prefix.data()),
        thrust::raw_pointer_cast(m_scores.data()), start_idx, m_capacity,
        m_size, 1);
    cuda_utils::check_last_cuda_error("compact_in_place_kernel scores failed");
    cuda_utils::check_cuda_call(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize compact_in_place_kernel scores failed");
    auto last_prefix = m_scratch_prefix[m_size - 1];
    auto last_keep   = m_scratch_mask[m_size - 1];
    m_size           = last_prefix + last_keep;
    m_write_head     = get_circular_index(m_size, start_idx, m_capacity);
    error_check::check_less_equal(
        m_size, m_capacity,
        "WorldTreeCUDA: Invalid size after prune_on_overload");
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void WorldTreeCUDA<FoldTypeCUDA>::validate_circular_buffer_state() {
    // Core invariant: total used space <= capacity
    const auto total_used = (m_size_old - m_read_consumed) + m_size;
    error_check::check_less_equal(total_used, m_capacity,
                                  "Circular buffer invariant violated: "
                                  "total_used > capacity");

    error_check::check_less_equal(m_read_consumed, m_size_old,
                                  "read_consumed cannot exceed size_old");

    error_check::check_less(m_head, m_capacity, "head must be within capacity");

    if (m_is_updating) {
        error_check::check_less(m_write_start, m_capacity,
                                "write_start must be within capacity");
        error_check::check_less(m_write_head, m_capacity,
                                "write_head must be within capacity");
    }
}

// Explicit instantiation
template class WorldTreeCUDA<float>;
template class WorldTreeCUDA<ComplexTypeCUDA>;
template void
WorldTreeCUDA<float>::copy_from_circular<float>(const float*,
                                                SizeType,
                                                SizeType,
                                                SizeType,
                                                float*,
                                                cudaStream_t) const noexcept;

template void WorldTreeCUDA<ComplexTypeCUDA>::copy_from_circular<float>(
    const float*, SizeType, SizeType, SizeType, float*, cudaStream_t)
    const noexcept;

} // namespace loki::memory