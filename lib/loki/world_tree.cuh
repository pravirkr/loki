#pragma once

#include <algorithm>
#include <format>
#include <stdexcept>

#include <cuda/std/complex>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cub/cub.cuh>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"

namespace loki::utils {

struct CircularIndexFunctor {
    SizeType start;
    SizeType capacity;

    __host__ __device__ CircularIndexFunctor(SizeType s, SizeType c)
        : start(s),
          capacity(c) {}

    __host__ __device__ SizeType operator()(SizeType logical_idx) const {
        const SizeType x = start + logical_idx;
        return (x < capacity) ? x : x - capacity;
    }
};

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
template <SupportedFoldTypeCUDA FoldTypeCUDA = float> class WorldTreeCUDA {
public:
    /**
     * @brief Constructor for the WorldTreeCUDA class.
     *
     * Initializes the internal arrays with the given maximum number of
     * candidates, number of parameters, and number of bins.
     *
     * @param capacity Maximum number of candidates to hold
     * @param nparams Number of parameters
     * @param nbins Number of bins
     */
    WorldTreeCUDA(SizeType capacity, SizeType nparams, SizeType nbins)
        : m_capacity(capacity),
          m_nparams(nparams),
          m_nbins(nbins),
          m_leaves_stride((nparams + 2) * kParamStride),
          m_folds_stride(2 * nbins),
          m_leaves(m_capacity * m_leaves_stride, 0.0),
          m_folds(m_capacity * m_folds_stride, FoldTypeCUDA{}),
          m_scores(m_capacity, 0.0F),
          m_scratch_leaves(m_capacity * (nparams * kParamStride), 0.0),
          m_scratch_scores(m_capacity, 0.0F),
          m_scratch_pending_indices(m_capacity, 0),
          m_scratch_mask(m_capacity, 0) {
        // Validate inputs
        error_check::check_greater(m_capacity, SizeType{0},
                                   "SuggestionTreeCUDA: capacity must be > 0");
        error_check::check_greater(m_nparams, SizeType{0},
                                   "SuggestionTreeCUDA: nparams must be > 0");
        error_check::check_greater(m_nbins, SizeType{0},
                                   "SuggestionTreeCUDA: nbins must be > 0");
    }

    ~WorldTreeCUDA()                                   = default;
    WorldTreeCUDA(WorldTreeCUDA&&) noexcept            = default;
    WorldTreeCUDA& operator=(WorldTreeCUDA&&) noexcept = default;
    WorldTreeCUDA(const WorldTreeCUDA&)                = delete;
    WorldTreeCUDA& operator=(const WorldTreeCUDA&)     = delete;

    // Size and capacity queries
    SizeType get_capacity() const noexcept { return m_capacity; }
    SizeType get_nparams() const noexcept { return m_nparams; }
    SizeType get_nbins() const noexcept { return m_nbins; }
    SizeType get_leaves_stride() const noexcept { return m_leaves_stride; }
    SizeType get_folds_stride() const noexcept { return m_folds_stride; }
    SizeType get_size() const noexcept { return m_size; }
    SizeType get_size_old() const noexcept { return m_size_old; }
    float get_size_lb() const noexcept {
        return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
    }

    // Get raw span of data
    cuda::std::span<const double> get_leaves_span() noexcept {
        return {thrust::raw_pointer_cast(m_leaves.data()), m_leaves.size()};
    }
    cuda::std::span<const FoldTypeCUDA> get_folds_span() noexcept {
        return {thrust::raw_pointer_cast(m_folds.data()), m_folds.size()};
    }
    cuda::std::span<float> get_scores_span() noexcept {
        return {thrust::raw_pointer_cast(m_scores.data()), m_scores.size()};
    }

    /**
     * @brief Get maximum score in current region
     */
    float get_score_max() const noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        const SizeType start = get_current_start_idx();
        if (start + m_size <= m_capacity) {
            // Contiguous case - direct reduction
            return thrust::reduce(thrust::device, m_scores.begin() + start,
                                  m_scores.begin() + start + m_size,
                                  -std::numeric_limits<float>::infinity(),
                                  thrust::maximum<float>());
        } // Wrapped case - two reductions
        const SizeType first_count = m_capacity - start;
        float max1                 = thrust::reduce(
            thrust::device, m_scores.begin() + start, m_scores.end(),
            -std::numeric_limits<float>::infinity(), thrust::maximum<float>());
        float max2 = thrust::reduce(thrust::device, m_scores.begin(),
                                    m_scores.begin() + (m_size - first_count),
                                    -std::numeric_limits<float>::infinity(),
                                    thrust::maximum<float>());
        return std::max(max1, max2);
    }

    /**
     * @brief Get minimum score in current region
     */
    float get_score_min() const noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        const SizeType start = get_current_start_idx();

        if (start + m_size <= m_capacity) {
            return thrust::reduce(thrust::device, m_scores.begin() + start,
                                  m_scores.begin() + start + m_size,
                                  std::numeric_limits<float>::infinity(),
                                  thrust::minimum<float>());
        }
        const SizeType first_count = m_capacity - start;
        float min1                 = thrust::reduce(
            thrust::device, m_scores.begin() + start, m_scores.end(),
            std::numeric_limits<float>::infinity(), thrust::minimum<float>());
        float min2 = thrust::reduce(thrust::device, m_scores.begin(),
                                    m_scores.begin() + (m_size - first_count),
                                    std::numeric_limits<float>::infinity(),
                                    thrust::minimum<float>());
        return std::min(min1, min2);
    }

    /**
     * @brief Estimate GPU memory usage in GiB
     *
     * Includes both base storage and estimated peak temporary allocations.
     */
    float get_memory_usage() const noexcept {
        const auto base_bytes =
            (m_leaves.size() * sizeof(double)) +
            (m_folds.size() * sizeof(FoldTypeCUDA)) +
            (m_scores.size() * sizeof(float)) +
            (m_scratch_leaves.size() * sizeof(double)) +
            (m_scratch_scores.size() * sizeof(float)) +
            (m_scratch_pending_indices.size() * sizeof(SizeType)) +
            (m_scratch_mask.size() * sizeof(uint8_t));

        return static_cast<float>(base_bytes) / static_cast<float>(1ULL << 30U);
    }

    /**
     * @brief Get span over leaves for processing
     *
     * During updates, returns span over readable (old) region.
     * The span may be truncated at buffer wrap point.
     *
     * @param n_leaves Number of leaves to access
     * @return Pair of (span, actual_size), where actual_size <= requested
     * n_leaves, limited to contiguous segment before wrap.
     */
    std::pair<cuda::std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const {
        const SizeType available = m_size_old - m_read_consumed;
        error_check::check_less_equal(
            n_leaves, available,
            "get_leaves_span: n_leaves exceeds available space");

        // Compute physical start: relative to current head of read region
        const auto physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);
        // Compute how many contiguous elements we can take before wrap
        const auto actual_size =
            std::min(n_leaves, m_capacity - physical_start);
        // Return span with correct byte size (actual_size elements, not bytes)
        return {cuda::std::span<const double>(
                    thrust::raw_pointer_cast(m_leaves.data()) +
                        (physical_start * m_leaves_stride),
                    actual_size * m_leaves_stride),
                actual_size};
    }

    cuda::std::span<double> get_leaves_contiguous_span() noexcept {
        const auto start_idx     = get_current_start_idx();
        const auto report_stride = m_nparams * kParamStride;
        const SizeType src_pitch = m_leaves_stride * sizeof(double);
        const SizeType dst_pitch = report_stride * sizeof(double);
        const SizeType row_bytes = report_stride * sizeof(double);

        const double* __restrict__ src =
            thrust::raw_pointer_cast(m_leaves.data());
        double* __restrict__ dst =
            thrust::raw_pointer_cast(m_scratch_leaves.data());

        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single copy
            cudaMemcpy2DAsync(dst, dst_pitch, src + start_idx * m_leaves_stride,
                              src_pitch, row_bytes, m_size,
                              cudaMemcpyDeviceToDevice);
        } else {
            // Wrapped case - two contiguous copies
            const auto first_part      = m_capacity - start_idx;
            const SizeType second_part = m_size - first_part;
            cudaMemcpy2DAsync(dst, dst_pitch, src + start_idx * m_leaves_stride,
                              src_pitch, row_bytes, first_part,
                              cudaMemcpyDeviceToDevice);
            cudaMemcpy2DAsync(dst + first_part * report_stride, dst_pitch, src,
                              src_pitch, row_bytes, second_part,
                              cudaMemcpyDeviceToDevice);
        }
        return cuda::std::span<double>(
                   thrust::raw_pointer_cast(m_scratch_leaves.data()))
            .first(m_size * report_stride);
    }

    /**
     * @brief Get span over contiguous scores (for saving to file)
     *
     * Returns span over contiguous scores for m_size elements in the current
     * region.
     *
     * @return Span over contiguous scores
     */
    cuda::std::span<float> get_scores_contiguous_span() noexcept {
        copy_scores_to_scratch();
        cudaDeviceSynchronize();
        return {thrust::raw_pointer_cast(m_scratch_scores.data()), m_size};
    }

    // Mutation operations

    /**
     * @brief Set size externally (for initialization)
     */
    void set_size(SizeType size) noexcept {
        m_size     = size;
        m_head     = 0;
        m_size_old = 0;
        error_check::check_less_equal(
            m_size, m_capacity, "WorldTreeCUDA: Invalid size after set_size");
    }

    /**
     * @brief Reset buffer to empty state
     */
    void reset() noexcept {
        m_size          = 0;
        m_head          = 0;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;
        m_write_head    = 0;
        m_write_start   = 0;
    }

    /**
     * @brief Prepare for in-place update
     *
     * Freezes current data as read region, opens write region.
     */
    void prepare_in_place_update() {
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

    /**
     * @brief Finalize in-place update
     *
     * Promotes the write region to be the new read region.
     * All old data must have been consumed.
     */
    void finalize_in_place_update() {
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

    /**
     * @brief Advance read consumed counter
     */
    void consume_read(SizeType n) {
        error_check::check_less_equal(m_read_consumed + n, m_size_old,
                                      "WorldTreeCUDA: read_consumed overflow");
        m_read_consumed += n;

        // Validate circular buffer invariant
        error_check::check_less_equal(
            m_size_old - m_read_consumed + m_size, m_capacity,
            "WorldTreeCUDA: circular buffer invariant violated");
    }

    /**
     * @brief Compute physical indices from logical indices
     */
    void
    compute_physical_indices(cuda::std::span<const SizeType> logical_indices,
                             cuda::std::span<SizeType> physical_indices,
                             SizeType n_leaves) const {
        if (!m_is_updating) {
            throw std::runtime_error(
                "compute_physical_indices: only valid during updates");
        }
        error_check::check_greater_equal(
            logical_indices.size(), n_leaves,
            "compute_physical_indices: logical_indices size insufficient");
        error_check::check_greater_equal(
            physical_indices.size(), n_leaves,
            "compute_physical_indices: physical_indices size insufficient");

        // Compute physical start: relative to current head of read region
        const SizeType physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);

        CircularIndexFunctor functor(physical_start, m_capacity);
        thrust::transform(thrust::device, logical_indices.begin(),
                          logical_indices.begin() +
                              static_cast<IndexType>(n_leaves),
                          physical_indices.begin(), functor);
    }

    /**
     * @brief Add initial batch (resets buffer first)
     */
    void add_initial(cuda::std::span<const double> batch_leaves,
                     cuda::std::span<const FoldTypeCUDA> batch_folds,
                     cuda::std::span<const float> batch_scores,
                     SizeType slots_to_write) {
        error_check::check_less_equal(
            slots_to_write, m_capacity,
            "WorldTreeCUDA: Suggestions too large to add.");
        error_check::check_equal(slots_to_write, batch_scores.size(),
                                 "slots_to_write must match batch_scores size");

        reset(); // Start fresh
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()),
                        batch_leaves.data(),
                        slots_to_write * m_leaves_stride * sizeof(double),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()),
                        batch_folds.data(),
                        slots_to_write * m_folds_stride * sizeof(FoldTypeCUDA),
                        cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_scores.data()),
                        batch_scores.data(), slots_to_write * sizeof(float),
                        cudaMemcpyDeviceToDevice);
        m_size = slots_to_write;
        error_check::check_less_equal(
            m_size, m_capacity,
            "WorldTreeCUDA: Invalid size after add_initial.");
    }

    /**
     * @brief Add batch during update with threshold filtering
     *
     * @return Updated threshold after any trimming
     * Adds filtered batch to write region. If full, trims write region via
     * median threshold. Loops until all candidates fit, reclaiming space
     * from consumed old candidates.
     */
    float add_batch(cuda::std::span<const double> batch_leaves,
                    cuda::std::span<const FoldTypeCUDA> batch_folds,
                    cuda::std::span<const float> batch_scores,
                    float current_threshold,
                    SizeType slots_to_write) {
        if (slots_to_write == 0) {
            return current_threshold;
        }
        // Fast path: Check if we have enough space immediately
        IndexType space_left = calculate_space_left();
        if (static_cast<IndexType>(slots_to_write) <= space_left) {
            // Check if we can do contiguous copy (no wrapping)
            if (m_write_head + slots_to_write <= m_capacity) {
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()) +
                                    (m_write_head * m_leaves_stride),
                                thrust::raw_pointer_cast(batch_leaves.data()),
                                slots_to_write * m_leaves_stride *
                                    sizeof(double),
                                cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()) +
                                    (m_write_head * m_folds_stride),
                                thrust::raw_pointer_cast(batch_folds.data()),
                                slots_to_write * m_folds_stride *
                                    sizeof(FoldTypeCUDA),
                                cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(m_scores.data()) + m_write_head,
                    thrust::raw_pointer_cast(batch_scores.data()),
                    slots_to_write * sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                // Wrapped case
                const auto first_part  = m_capacity - m_write_head;
                const auto second_part = slots_to_write - first_part;
                // Copy first part [m_write_head, m_capacity)
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()) +
                                    (m_write_head * m_leaves_stride),
                                thrust::raw_pointer_cast(batch_leaves.data()),
                                first_part * m_leaves_stride * sizeof(double),
                                cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()) +
                                    (m_write_head * m_folds_stride),
                                thrust::raw_pointer_cast(batch_folds.data()),
                                first_part * m_folds_stride *
                                    sizeof(FoldTypeCUDA),
                                cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(m_scores.data()) + m_write_head,
                    thrust::raw_pointer_cast(batch_scores.data()),
                    first_part * sizeof(float), cudaMemcpyDeviceToDevice);
                // Copy second part [0, second_part)
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()),
                                thrust::raw_pointer_cast(batch_leaves.data()) +
                                    (first_part * m_leaves_stride),
                                second_part * m_leaves_stride * sizeof(double),
                                cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()),
                                thrust::raw_pointer_cast(batch_folds.data()) +
                                    (first_part * m_folds_stride),
                                second_part * m_folds_stride *
                                    sizeof(FoldTypeCUDA),
                                cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(m_scores.data()),
                    thrust::raw_pointer_cast(batch_scores.data()) + first_part,
                    second_part * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            m_write_head =
                get_circular_index(slots_to_write, m_write_head, m_capacity);
            m_size += slots_to_write;
            return current_threshold;
        }

        // Slow path: Overflow & Pruning
        float effective_threshold = current_threshold;
        auto pending_indices_span = cuda::std::span<SizeType>(
            thrust::raw_pointer_cast(m_scratch_pending_indices.data()),
            slots_to_write);
        thrust::sequence(pending_indices_span.begin(),
                         pending_indices_span.end());
        SizeType pending_count  = slots_to_write;
        SizeType pending_offset = 0;
        while (pending_offset < pending_count) {
            space_left = calculate_space_left();

            if (space_left < 0 ||
                space_left > static_cast<IndexType>(m_capacity)) {
                throw std::runtime_error(
                    std::format("WorldTreeCUDA: Invalid space left ({}) "
                                "after add_batch. Buffer overflow.",
                                space_left));
            }

            if (space_left == 0) {
                // Buffer is full, try to prune the newly added candidates.
                const float new_threshold = prune_on_overload();
                effective_threshold =
                    std::max(effective_threshold, new_threshold);
                // Re-filter with new threshold
                SizeType new_pending = 0;
                for (SizeType i = pending_offset; i < pending_count; ++i) {
                    const auto idx = pending_indices_span[i];
                    if (batch_scores[idx] >= effective_threshold) {
                        pending_indices_span[new_pending++] = idx;
                    }
                }
                pending_count  = new_pending;
                pending_offset = 0;
                continue;
            }

            const SizeType n_to_add =
                std::min(pending_count - pending_offset,
                         static_cast<SizeType>(space_left));

            // Batch copy
            for (SizeType i = 0; i < n_to_add; ++i) {
                const auto src_idx = pending_indices_span[pending_offset + i];
                const auto dst_idx = m_write_head;
                const auto leaves_src = src_idx * m_leaves_stride;
                const auto leaves_dst = dst_idx * m_leaves_stride;
                const auto folds_src  = src_idx * m_folds_stride;
                const auto folds_dst  = dst_idx * m_folds_stride;
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(m_leaves.data()) + leaves_dst,
                    thrust::raw_pointer_cast(batch_leaves.data()) + leaves_src,
                    m_leaves_stride * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(m_folds.data()) + folds_dst,
                    thrust::raw_pointer_cast(batch_folds.data()) + folds_src,
                    m_folds_stride * sizeof(FoldTypeCUDA),
                    cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(
                    thrust::raw_pointer_cast(m_scores.data()) + dst_idx,
                    thrust::raw_pointer_cast(batch_scores.data()) + src_idx,
                    sizeof(float), cudaMemcpyDeviceToDevice);
                m_write_head++;
                if (m_write_head == m_capacity) {
                    m_write_head = 0;
                }
            }
            m_size += n_to_add;
            pending_offset += n_to_add;
        }

        return effective_threshold;
    }

private:
    static constexpr SizeType kParamStride = 2;

    // Configuration
    SizeType m_capacity;
    SizeType m_nparams;
    SizeType m_nbins;
    SizeType m_leaves_stride;
    SizeType m_folds_stride;

    // Device storage
    thrust::device_vector<double> m_leaves;
    thrust::device_vector<FoldTypeCUDA> m_folds;
    thrust::device_vector<float> m_scores;

    // Circular buffer state (host-side tracking)
    SizeType m_head{0};
    SizeType m_size{0};
    SizeType m_size_old{0};
    SizeType m_write_head{0};
    SizeType m_write_start{0};
    bool m_is_updating{false};
    SizeType m_read_consumed{0};

    // Scratch buffer for in-place operations
    thrust::device_vector<double> m_scratch_leaves;
    thrust::device_vector<float> m_scratch_scores;
    thrust::device_vector<SizeType> m_scratch_pending_indices;
    thrust::device_vector<uint8_t> m_scratch_mask;

    /**
     * @brief Get the starting index for current region
     */
    SizeType get_current_start_idx() const noexcept {
        return m_is_updating ? m_write_start : m_head;
    }

    /**
     * @brief Compute physical index from logical index in circular buffer
     * @param logical_idx Logical index (0-based from start of valid region)
     * @param start Starting physical index of the region
     * @param capacity Total buffer capacity
     * @return Physical index in the buffer
     */
    static constexpr SizeType get_circular_index(SizeType logical_idx,
                                                 SizeType start,
                                                 SizeType capacity) noexcept {
        return (start + logical_idx) % capacity;
    }

    /**
     * @brief Calculate available space in buffer
     */
    IndexType calculate_space_left() const noexcept {
        const auto remaining_old = static_cast<IndexType>(m_size_old) -
                                   static_cast<IndexType>(m_read_consumed);
        error_check::check_greater_equal(
            remaining_old, 0, "calculate_space_left: Invalid buffer state");
        return static_cast<IndexType>(m_capacity) -
               (remaining_old + static_cast<IndexType>(m_size));
    }

    /**
     * @brief Copy current scores to scratch buffer
     *
     * Copies scores for m_size elements in the current region to scratch
     * buffer.
     */
    void copy_scores_to_scratch() noexcept {
        if (m_size == 0) {
            return;
        }

        float* __restrict__ dst =
            thrust::raw_pointer_cast(m_scratch_scores.data());
        const float* __restrict__ src =
            thrust::raw_pointer_cast(m_scores.data());

        const SizeType start = get_current_start_idx();
        if (start + m_size <= m_capacity) {
            // Single contiguous copy
            cudaMemcpyAsync(dst, src + start, m_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
        } else {
            // Wrapped case: two memcpy
            const SizeType first_part = m_capacity - start;
            cudaMemcpyAsync(dst, src + start, first_part * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(dst + first_part, src,
                            (m_size - first_part) * sizeof(float),
                            cudaMemcpyDeviceToDevice);
        }
    }

    /**
     * @brief Compute prune threshold of current region using Thrust sort
     *
     * Uses nth_element for O(n) complexity.
     *
     * @return Prune threshold value
     */
    float get_prune_threshold() noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        copy_scores_to_scratch();
        const auto scores_span = cuda::std::span<float>(
            thrust::raw_pointer_cast(m_scratch_scores.data()), m_size);
        // Optimized Radix Sort
        thrust::sort(scores_span.begin(), scores_span.end());
        const float median_val = scores_span[m_size / 2];
        return median_val;
    }

    /**
     * @brief Prune write region by threshold with in-place update.
     * @param mode The prune mode to use
     * @return The threshold used
     *
     * Single-pass approach: safely compacts in-place because write never
     * overtakes read (write_logical ≤ read_logical always holds).
     */
    float prune_on_overload() {
        if (!m_is_updating) {
            throw std::runtime_error("WorldTreeCUDA: prune_on_overload: only "
                                     "allowed during updates");
        }
        if (m_size == 0) {
            return 0.0F;
        }

        // Compute median score of the *newly added* candidates.
        const float threshold    = get_prune_threshold();
        const SizeType start_idx = get_current_start_idx();
        const SizeType old_size  = m_size;
        // Use logical indices to abstract circular wrapping
        SizeType phys_read  = start_idx;
        SizeType phys_write = start_idx;
        SizeType kept_count = 0;
        // Single-pass: scan and compact in-place
        for (SizeType read_logical = 0; read_logical < old_size;
             ++read_logical) {
            // Check filter condition
            if (m_scores[phys_read] >= threshold) {
                // Move data only if gaps were created
                if (phys_read != phys_write) {
                    cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()) +
                                        (phys_write * m_leaves_stride),
                                    thrust::raw_pointer_cast(m_leaves.data()) +
                                        (phys_read * m_leaves_stride),
                                    m_leaves_stride * sizeof(double),
                                    cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()) +
                                        (phys_write * m_folds_stride),
                                    thrust::raw_pointer_cast(m_folds.data()) +
                                        (phys_read * m_folds_stride),
                                    m_folds_stride * sizeof(FoldTypeCUDA),
                                    cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(m_scores.data()) + phys_write,
                        thrust::raw_pointer_cast(m_scores.data()) + phys_read,
                        sizeof(float), cudaMemcpyDeviceToDevice);
                }
                // Advance write pointer (with wrap)
                kept_count++;
                phys_write++;
                if (phys_write == m_capacity) {
                    phys_write = 0;
                }
            }
            // Always advance read pointer (with wrap)
            phys_read++;
            if (phys_read == m_capacity) {
                phys_read = 0;
            }
        }
        // Update valid size
        m_size = kept_count;
        // After trimming, the write head must be updated to the new end
        if (m_is_updating) {
            m_write_head = phys_write;
        }
        error_check::check_less_equal(
            m_size, m_capacity,
            "WorldTreeCUDA: Invalid size after prune_on_overload");
        return threshold;
    }

    void validate_circular_buffer_state() {
        // Core invariant: total used space <= capacity
        const auto total_used = (m_size_old - m_read_consumed) + m_size;
        error_check::check_less_equal(total_used, m_capacity,
                                      "Circular buffer invariant violated: "
                                      "total_used > capacity");

        error_check::check_less_equal(m_read_consumed, m_size_old,
                                      "read_consumed cannot exceed size_old");

        error_check::check_less(m_head, m_capacity,
                                "head must be within capacity");

        if (m_is_updating) {
            error_check::check_less(m_write_start, m_capacity,
                                    "write_start must be within capacity");
            error_check::check_less(m_write_head, m_capacity,
                                    "write_head must be within capacity");
        }
    }
};

} // namespace loki::utils

#if 0
public:
    /**
     * @brief Add initial batch (resets buffer first)
     * optimized to use direct cudaMemcpy where possible
     */
     void add_initial(cuda::std::span<const double> batch_leaves,
        cuda::std::span<const FoldTypeCUDA> batch_folds,
        cuda::std::span<const float> batch_scores,
        SizeType slots_to_write) {
error_check::check_less_equal(slots_to_write, m_capacity,
                         "WorldTreeCUDA: Suggestions too large to add.");
// Consistency check
if(batch_scores.size() != slots_to_write) {
throw std::runtime_error("batch_scores size mismatch");
}

reset(); // Start fresh

// Direct async copies - fastest possible method
// Leaves
cudaMemcpyAsync(thrust::raw_pointer_cast(m_leaves.data()), 
           batch_leaves.data(),
           slots_to_write * m_leaves_stride * sizeof(double),
           cudaMemcpyDeviceToDevice);

// Folds
cudaMemcpyAsync(thrust::raw_pointer_cast(m_folds.data()), 
           batch_folds.data(),
           slots_to_write * m_folds_stride * sizeof(FoldTypeCUDA),
           cudaMemcpyDeviceToDevice);
           
// Scores
cudaMemcpyAsync(thrust::raw_pointer_cast(m_scores.data()), 
           batch_scores.data(),
           slots_to_write * sizeof(float),
           cudaMemcpyDeviceToDevice);

m_size = slots_to_write;
// In initial add, write_head moves to the end of what we just wrote
// effectively same as simple linear buffer
m_write_start = 0;
m_write_head = (m_size == m_capacity) ? 0 : m_size;

// Ensure CPU sees the updated size immediately if needed, 
// but typically we trust the stream. m_size is host-side.
}

/**
* @brief Add batch with filtering. 
* Optimizes memory access by avoiding modulo in the scatter kernel.
*/
float add_batch(cuda::std::span<const double> batch_leaves,
       cuda::std::span<const FoldTypeCUDA> batch_folds,
       cuda::std::span<const float> batch_scores,
       float current_threshold,
       SizeType slots_to_write) {
if (slots_to_write == 0) return current_threshold;

float effective_threshold = current_threshold;

// 1. Filter indices of elements passing threshold
// We reuse a persistent device vector to avoid allocation overhead if possible
// but for now local is fine.
thrust::device_vector<SizeType> filtered_indices(slots_to_write);

// We need an iterator that counts 0..slots_to_write
auto counting_iter = thrust::make_counting_iterator(SizeType{0});

// Custom predicate
const float* d_batch_scores = batch_scores.data();
auto pred = [d_batch_scores, effective_threshold] __device__ (SizeType i) {
return d_batch_scores[i] >= effective_threshold;
};

auto end_iter = thrust::copy_if(thrust::device, 
                           counting_iter, 
                           counting_iter + slots_to_write, 
                           filtered_indices.begin(), 
                           pred);

SizeType num_pending = end_iter - filtered_indices.begin();
SizeType processed = 0;

while (processed < num_pending) {
IndexType space_left = calculate_space_left();

if (space_left == 0) {
   // Prune
   float new_threshold = prune_on_overload();
   effective_threshold = std::max(effective_threshold, new_threshold);
   
   // Re-filter the REMAINING pending items
   // It is cheaper to re-run copy_if on the *remaining* filtered_indices 
   // than to manage a complex state, but typically we just re-scan the *original*
   // list logic or filter the already filtered list.
   // For exact match with CPU: CPU re-checks *entire* list against new threshold.
   // Optimization: Filter the *current* filtered_indices in-place.
   
   auto new_end = thrust::remove_if(thrust::device,
                                   filtered_indices.begin() + processed,
                                   filtered_indices.begin() + num_pending,
                                   [d_batch_scores, effective_threshold] __device__ (SizeType idx) {
                                       return d_batch_scores[idx] < effective_threshold; 
                                   });
   num_pending = new_end - filtered_indices.begin();
   continue; 
}

SizeType n_to_add = std::min((SizeType)space_left, num_pending - processed);

// Perform the copy
// We split the write into 1 or 2 contiguous chunks to avoid modulo in kernel
SizeType write_idx = m_write_head;
SizeType contig_space = m_capacity - write_idx;

SizeType chunk1 = std::min(n_to_add, contig_space);
SizeType chunk2 = n_to_add - chunk1;

auto scatter_chunk = [&](SizeType count, SizeType dst_start_idx, SizeType src_offset_in_filtered) {
   if (count == 0) return;
   
   // Pointers
   double* d_dst_leaves = thrust::raw_pointer_cast(m_leaves.data());
   FoldTypeCUDA* d_dst_folds = thrust::raw_pointer_cast(m_folds.data());
   float* d_dst_scores = thrust::raw_pointer_cast(m_scores.data());
   
   const SizeType* d_filtered_idx = thrust::raw_pointer_cast(filtered_indices.data()) + src_offset_in_filtered;
   
   // Launch Kernel (using for_each for simplicity, but logically it's a kernel)
   thrust::for_each_n(thrust::device, 
                      thrust::make_counting_iterator(SizeType{0}), 
                      count,
                      [=, leaves_stride=m_leaves_stride, folds_stride=m_folds_stride] 
                      __device__ (SizeType i) {
                          SizeType src_batch_idx = d_filtered_idx[i];
                          SizeType dst_idx = dst_start_idx + i;
                          
                          // Copy Score
                          d_dst_scores[dst_idx] = d_batch_scores[src_batch_idx];
                          
                          // Copy Leaves (Strided)
                          for(int k=0; k<leaves_stride; ++k) {
                               d_dst_leaves[dst_idx * leaves_stride + k] = 
                                   batch_leaves[src_batch_idx * leaves_stride + k];
                          }
                          
                          // Copy Folds (Strided)
                          for(int k=0; k<folds_stride; ++k) {
                               d_dst_folds[dst_idx * folds_stride + k] = 
                                   batch_folds[src_batch_idx * folds_stride + k];
                          }
                      });
};

// Chunk 1: [write_head, m_capacity or end of batch)
scatter_chunk(chunk1, m_write_head, processed);

// Chunk 2: [0, remaining)
if (chunk2 > 0) {
   scatter_chunk(chunk2, 0, processed + chunk1);
}

m_size += n_to_add;
processed += n_to_add;

// Update write head
m_write_head += n_to_add;
if (m_write_head >= m_capacity) m_write_head -= m_capacity;
}

return effective_threshold;
}

/**
* @brief Prune on overload using split-loop strategy
*/
float prune_on_overload() {
if (!m_is_updating) throw std::runtime_error("prune_on_overload: update only");
if (m_size == 0) return 0.0F;

// 1. Get Median
// get_score_median() creates a temp copy and sorts it. Correct.
float threshold = get_score_median();

// 2. Build Mask
thrust::device_vector<bool> keep_mask(m_size);
bool* d_mask = thrust::raw_pointer_cast(keep_mask.data());
const float* d_scores = thrust::raw_pointer_cast(m_scores.data());

auto regions = get_active_regions(m_scores);

// Fill mask for first region
if (!regions.first.empty()) {
const float* ptr1 = regions.first.data();
thrust::transform(thrust::device, 
                 ptr1, ptr1 + regions.first.size(), 
                 d_mask,
                 [threshold] __device__ (float val) { return val >= threshold; });
}

// Fill mask for second region
if (!regions.second.empty()) {
const float* ptr2 = regions.second.data();
thrust::transform(thrust::device, 
                 ptr2, ptr2 + regions.second.size(), 
                 d_mask + regions.first.size(),
                 [threshold] __device__ (float val) { return val >= threshold; });
}

// 3. Keep
keep(keep_mask);
return threshold;
}

/**
* @brief Compact buffer in-place (conceptually) using temporary storage
* Optimized to avoid modulo and repeated kernel launches
*/
void keep(const thrust::device_vector<bool>& keep_mask) {
SizeType count = thrust::count(keep_mask.begin(), keep_mask.end(), true);
if (count == m_size) return;
if (count == 0) {
m_size = 0;
if(m_is_updating) m_write_head = m_write_start;
return;
}

// 1. Scan to find new positions
thrust::device_vector<SizeType> dst_offsets(m_size);
thrust::exclusive_scan(thrust::device, keep_mask.begin(), keep_mask.end(), dst_offsets.begin());

// 2. Compact into temp buffers
// We MUST use temp buffers because parallel compaction in-place on circular buffer is hard 
// to do safely without race conditions or complex atomic logic.
thrust::device_vector<double> temp_leaves(count * m_leaves_stride);
thrust::device_vector<FoldTypeCUDA> temp_folds(count * m_folds_stride);
thrust::device_vector<float> temp_scores(count);

// Raw Pointers
double* d_temp_leaves = thrust::raw_pointer_cast(temp_leaves.data());
FoldTypeCUDA* d_temp_folds = thrust::raw_pointer_cast(temp_folds.data());
float* d_temp_scores = thrust::raw_pointer_cast(temp_scores.data());

double* d_src_leaves = thrust::raw_pointer_cast(m_leaves.data());
FoldTypeCUDA* d_src_folds = thrust::raw_pointer_cast(m_folds.data());
float* d_src_scores = thrust::raw_pointer_cast(m_scores.data());

const bool* d_mask = thrust::raw_pointer_cast(keep_mask.data());
const SizeType* d_offsets = thrust::raw_pointer_cast(dst_offsets.data());

// We launch ONE kernel to move everything.
// But we must handle the Circular Read. 
// We split the kernel launch into 2 parts (Region 1 and Region 2) to ensure
// coalesced reads from source.

auto regions_scores = get_active_regions(m_scores); // Just to get sizes/pointers
SizeType len1 = regions_scores.first.size();
SizeType len2 = regions_scores.second.size();

auto move_kernel = [=, leaves_stride=m_leaves_stride, folds_stride=m_folds_stride] 
              __device__ (SizeType i, SizeType src_phys_idx) {
if (d_mask[i]) {
   SizeType dst_idx = d_offsets[i];
   
   // Move Score
   d_temp_scores[dst_idx] = d_src_scores[src_phys_idx];
   
   // Move Leaves
   for(int k=0; k<leaves_stride; ++k) {
       d_temp_leaves[dst_idx * leaves_stride + k] = 
           d_src_leaves[src_phys_idx * leaves_stride + k];
   }
   
   // Move Folds
   for(int k=0; k<folds_stride; ++k) {
       d_temp_folds[dst_idx * folds_stride + k] = 
           d_src_folds[src_phys_idx * folds_stride + k];
   }
}
};

// Launch for Region 1
SizeType start1 = get_current_start_idx();
thrust::for_each_n(thrust::device, 
              thrust::make_counting_iterator(SizeType{0}), 
              len1,
              [=] __device__ (SizeType i) {
                  move_kernel(i, start1 + i);
              });

// Launch for Region 2 (if exists)
if (len2 > 0) {
thrust::for_each_n(thrust::device, 
                  thrust::make_counting_iterator(SizeType{0}), 
                  len2,
                  [=] __device__ (SizeType i) {
                      move_kernel(len1 + i, i); // src index starts at 0 for region 2, logical index is len1+i
                  });
}

// 3. Copy Back (Contiguous)
// Now we have contiguous temp data. We copy it back to [m_write_start ... ]
// We might wrap again during write back.
// This is essentially "add_initial" logic but preserving the circular start/end logic if strict,
// OR we can just reset the buffer to be contiguous starting at m_write_start.
// Actually, to preserve the "circular window" semantics for the reader, we usually just overwrite 
// starting at m_write_start and wrap.

SizeType start_write = m_write_start; // Usually where we keep the "start" of valid data

// Split copy back
SizeType back_chunk1 = std::min(count, m_capacity - start_write);
SizeType back_chunk2 = count - back_chunk1;

auto copy_back = [&](SizeType count_items, SizeType src_offset, SizeType dst_idx_start) {
// Scores
cudaMemcpyAsync(d_src_scores + dst_idx_start, 
                d_temp_scores + src_offset, 
                count_items * sizeof(float), cudaMemcpyDeviceToDevice);

// Leaves
cudaMemcpyAsync(d_src_leaves + dst_idx_start * m_leaves_stride, 
                d_temp_leaves + src_offset * m_leaves_stride, 
                count_items * m_leaves_stride * sizeof(double), cudaMemcpyDeviceToDevice);
                
// Folds
cudaMemcpyAsync(d_src_folds + dst_idx_start * m_folds_stride, 
                d_temp_folds + src_offset * m_folds_stride, 
                count_items * m_folds_stride * sizeof(FoldTypeCUDA), cudaMemcpyDeviceToDevice);
};

copy_back(back_chunk1, 0, start_write);
if (back_chunk2 > 0) {
copy_back(back_chunk2, back_chunk1, 0);
}

m_size = count;
if(m_is_updating) {
m_write_head = (m_write_start + m_size) % m_capacity;
}
}
#endif