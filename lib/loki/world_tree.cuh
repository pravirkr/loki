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
     * @param mode Mode of the world tree (e.g. Taylor, Chebyshev)
     */
    WorldTreeCUDA(SizeType capacity,
                  SizeType nparams,
                  SizeType nbins,
                  std::string_view mode)
        : m_capacity(capacity),
          m_nparams(nparams),
          m_nbins(nbins),
          m_mode(mode),
          m_leaves_stride((nparams + 2) * kParamStride),
          m_folds_stride(2 * nbins),
          m_leaves(m_capacity * m_leaves_stride, 0.0),
          m_folds(m_capacity * m_folds_stride, FoldTypeCUDA{}),
          m_scores(m_capacity, 0.0F) {
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
    std::string_view get_mode() const noexcept { return m_mode; }
    SizeType get_leaves_stride() const noexcept { return m_leaves_stride; }
    SizeType get_folds_stride() const noexcept { return m_folds_stride; }
    SizeType get_size() const noexcept { return m_size; }
    SizeType get_size_old() const noexcept { return m_size_old; }
    float get_size_lb() const noexcept {
        return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
    }

    // Get raw pointer to data
    double* get_leaves_ptr() noexcept {
        return thrust::raw_pointer_cast(m_leaves.data());
    }
    const double* get_leaves_ptr() const noexcept {
        return thrust::raw_pointer_cast(m_leaves.data());
    }
    FoldTypeCUDA* get_folds_ptr() noexcept {
        return thrust::raw_pointer_cast(m_folds.data());
    }
    const FoldTypeCUDA* get_folds_ptr() const noexcept {
        return thrust::raw_pointer_cast(m_folds.data());
    }
    float* get_scores_ptr() noexcept {
        return thrust::raw_pointer_cast(m_scores.data());
    }
    const float* get_scores_ptr() const noexcept {
        return thrust::raw_pointer_cast(m_scores.data());
    }

    /**
     * @brief Get copy of current scores
     *
     * Returns scores for m_size elements in the current region.
     *
     * @return Device vector of scores
     */
    thrust::device_vector<float> get_scores() const {
        thrust::device_vector<float> scores(m_size);
        if (m_size == 0) {
            return scores;
        }

        float* dst       = thrust::raw_pointer_cast(scores.data());
        const float* src = thrust::raw_pointer_cast(m_scores.data());

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

        return scores;
    }

    /**
     * @brief Get maximum score in current region
     */
    float get_score_max() const {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto regions = get_active_regions(m_scores);
        float max_val      = thrust::reduce(
            regions.first.begin(), regions.first.end(),
            -std::numeric_limits<float>::infinity(), thrust::maximum<float>());
        if (!regions.second.empty()) {
            max_val =
                thrust::reduce(regions.second.begin(), regions.second.end(),
                               max_val, thrust::maximum<float>());
        }
        return max_val;
    }

    /**
     * @brief Get minimum score in current region
     */
    float get_score_min() const {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto regions = get_active_regions(m_scores);
        float min_val      = thrust::reduce(
            regions.first.begin(), regions.first.end(),
            std::numeric_limits<float>::infinity(), thrust::minimum<float>());
        if (!regions.second.empty()) {
            min_val =
                thrust::reduce(regions.second.begin(), regions.second.end(),
                               min_val, thrust::minimum<float>());
        }
        return min_val;
    }

    /**
     * @brief Compute median score of current region using Thrust sort
     *
     * Note: This involves a device-side sort which modifies a temporary copy.
     *
     * @return Median score value
     */
    float get_score_median() const {
        if (m_size == 0) {
            return 0.0F;
        }
        thrust::device_vector<float> linear_scores = get_scores();
        // Optimized Radix Sort
        thrust::sort(linear_scores.begin(), linear_scores.end());
        const float median_val = linear_scores[m_size / 2];
        return median_val;
    }

    /**
     * @brief Estimate GPU memory usage in GiB
     * @return Memory usage in GiB
     */
    float get_memory_usage() const noexcept {
        const auto total_bytes = (m_leaves.size() * sizeof(double)) +
                                 (m_folds.size() * sizeof(FoldTypeCUDA)) +
                                 (m_scores.size() * sizeof(float));

        return static_cast<float>(total_bytes) /
               static_cast<float>(1ULL << 30U);
    }

    // Mutation operations

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

        const SizeType span_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);

        CircularIndexFunctor functor(span_start, m_capacity);
        thrust::transform(thrust::device, logical_indices.begin(),
                          logical_indices.begin() +
                              static_cast<IndexType>(n_leaves),
                          physical_indices.begin(), functor);
    }

    /**
     * @brief Overload for thrust device_vectors
     */
    void compute_physical_indices(
        const thrust::device_vector<SizeType>& logical_indices,
        thrust::device_vector<SizeType>& physical_indices,
        SizeType n_leaves) const {

        error_check::check_greater_equal(
            logical_indices.size(), n_leaves,
            "compute_physical_indices: logical_indices size insufficient");
        error_check::check_greater_equal(
            physical_indices.size(), n_leaves,
            "compute_physical_indices: physical_indices size insufficient");

        compute_physical_indices(
            thrust::raw_pointer_cast(logical_indices.data()),
            thrust::raw_pointer_cast(physical_indices.data()), n_leaves);
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

        float effective_threshold = current_threshold;

        // Create device vector for pending indices
        thrust::device_vector<SizeType> pending_indices(slots_to_write);
        thrust::device_vector<SizeType> filtered_indices(slots_to_write);

        auto update_candidates = [&]() -> SizeType {
            // Generate sequence 0, 1, 2, ...
            thrust::sequence(pending_indices.begin(), pending_indices.end());

            // Filter by threshold using copy_if
            auto end_iter = thrust::copy_if(
                thrust::device, pending_indices.begin(), pending_indices.end(),
                filtered_indices.begin(),
                [batch_scores, effective_threshold] __device__(SizeType idx) {
                    return batch_scores[idx] >= effective_threshold;
                });

            return static_cast<SizeType>(end_iter - filtered_indices.begin());
        };

        SizeType num_pending = update_candidates();

        SizeType processed = 0;
        while (processed < num_pending) {
            const auto space_left = calculate_space_left();

            if (space_left < 0 ||
                space_left > static_cast<IndexType>(m_capacity)) {
                throw std::runtime_error(
                    std::format("SuggestionTreeCUDA: Invalid space left ({}) "
                                "after add_batch. Buffer overflow.",
                                space_left));
            }

            if (space_left == 0) {
                // Trim and get new threshold
                const float new_threshold = prune_on_overload();
                effective_threshold =
                    std::max(effective_threshold, new_threshold);

                // Re-filter with new threshold
                num_pending = update_candidates();
                processed   = 0;
                continue;
            }

            const SizeType n_to_add = std::min(
                num_pending - processed, static_cast<SizeType>(space_left));

            // Batch copy using custom kernel or element-wise for now
            // For simplicity, we do element-wise but this could be optimized
            // with a custom CUDA kernel for strided scatter
            add_elements_to_write_region(
                batch_leaves.data(), batch_folds.data(), batch_scores.data(),
                thrust::raw_pointer_cast(filtered_indices.data()) + processed,
                n_to_add);

            processed += n_to_add;
        }

        return effective_threshold;
    }

    /**
     * @brief Prune write region by median threshold
     * @return The median threshold used
     */
    float prune_on_overload() {
        if (!m_is_updating) {
            throw std::runtime_error(
                "prune_on_overload: only allowed during updates");
        }
        if (m_size == 0) {
            return 0.0F;
        }

        // Compute median score of the *newly added* candidates.
        const float threshold = get_score_median();

        // Create boolean mask for scores >= threshold
        thrust::device_vector<bool> keep_mask(m_size);
        const SizeType start_idx = m_write_start;
        const float* scores_ptr  = thrust::raw_pointer_cast(m_scores.data());

        thrust::transform(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(m_size), keep_mask.begin(),
            [scores_ptr, threshold, start_idx,
             cap = m_capacity] __device__(SizeType i) {
                return scores_ptr[(start_idx + i) % cap] >= threshold;
            });

        keep(keep_mask);
        return threshold;
    }

private:
    static constexpr SizeType kParamStride = 2;

    // Configuration
    SizeType m_capacity;
    SizeType m_nparams;
    SizeType m_nbins;
    std::string_view m_mode;
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

    // Helper structure to represent the two contiguous segments
    // of a circular buffer.
    template <typename T> struct CircularViewCUDA {
        cuda::std::span<T> first;
        cuda::std::span<T> second; // Empty if buffer doesn't wrap
    };

    // Generic helper to get active regions for any vector
    template <typename T>
    CircularViewCUDA<const T>
    get_active_regions(const thrust::device_vector<T>& arr,
                       SizeType stride = 1) const {
        if (m_size == 0) {
            return {cuda::std::span<const T>{}, cuda::std::span<const T>{}};
        }

        const auto start = get_current_start_idx();
        // Handle stride for leaves/folds
        const auto start_offset = start * stride;
        if (start + m_size <= m_capacity) {
            return {cuda::std::span<const T>{
                        thrust::raw_pointer_cast(arr.data()) + start_offset,
                        m_size * stride},
                    cuda::std::span<const T>{}};
        }
        const auto first_count  = m_capacity - start;
        const auto second_count = m_size - first_count;
        return {cuda::std::span<const T>{thrust::raw_pointer_cast(arr.data()) +
                                             start_offset,
                                         first_count * stride},
                cuda::std::span<const T>{thrust::raw_pointer_cast(arr.data()),
                                         second_count * stride}};
    }

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
        return static_cast<IndexType>(m_capacity) -
               ((static_cast<IndexType>(m_size_old) -
                 static_cast<IndexType>(m_read_consumed)) +
                static_cast<IndexType>(m_size));
    }

    /**
     * @brief Add elements to write region (strided scatter)
     */
    void add_elements_to_write_region(const double* d_batch_leaves,
                                      const FoldTypeCUDA* d_batch_folds,
                                      const float* d_batch_scores,
                                      const SizeType* d_src_indices,
                                      SizeType count) {
        if (count == 0) {
            return;
        }

        // Get raw pointers for device operations
        double* leaves_ptr      = thrust::raw_pointer_cast(m_leaves.data());
        FoldTypeCUDA* folds_ptr = thrust::raw_pointer_cast(m_folds.data());
        float* scores_ptr       = thrust::raw_pointer_cast(m_scores.data());

        const SizeType leaves_stride = m_leaves_stride;
        const SizeType folds_stride  = m_folds_stride;
        const SizeType capacity      = m_capacity;
        SizeType write_head          = m_write_head;

        // For each element, we need to copy strided data
        // This is done with a custom operation using for_each
        // A more efficient approach would be a custom CUDA kernel

        // First, compute destination indices
        thrust::device_vector<SizeType> dst_indices(count);
        thrust::sequence(dst_indices.begin(), dst_indices.end(), write_head);
        thrust::transform(
            thrust::device, dst_indices.begin(), dst_indices.end(),
            dst_indices.begin(),
            [capacity] __device__(SizeType idx) { return idx % capacity; });

        // Copy scores (simple scatter)
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(count),
            [d_batch_scores, d_src_indices, scores_ptr,
             dst_ptr = thrust::raw_pointer_cast(
                 dst_indices.data())] __device__(SizeType i) {
                scores_ptr[dst_ptr[i]] = d_batch_scores[d_src_indices[i]];
            });

        // Copy leaves (strided scatter)
        // Each element has leaves_stride doubles to copy
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(count * leaves_stride),
            [d_batch_leaves, d_src_indices, leaves_ptr, leaves_stride,
             dst_ptr = thrust::raw_pointer_cast(
                 dst_indices.data())] __device__(SizeType flat_idx) {
                const SizeType elem_idx = flat_idx / leaves_stride;
                const SizeType offset   = flat_idx % leaves_stride;
                const SizeType src_idx  = d_src_indices[elem_idx];
                const SizeType dst_idx  = dst_ptr[elem_idx];
                leaves_ptr[(dst_idx * leaves_stride) + offset] =
                    d_batch_leaves[(src_idx * leaves_stride) + offset];
            });

        // Copy folds (strided scatter)
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(count * folds_stride),
            [d_batch_folds, d_src_indices, folds_ptr, folds_stride,
             dst_ptr = thrust::raw_pointer_cast(
                 dst_indices.data())] __device__(SizeType flat_idx) {
                const SizeType elem_idx = flat_idx / folds_stride;
                const SizeType offset   = flat_idx % folds_stride;
                const SizeType src_idx  = d_src_indices[elem_idx];
                const SizeType dst_idx  = dst_ptr[elem_idx];
                folds_ptr[(dst_idx * folds_stride) + offset] =
                    d_batch_folds[(src_idx * folds_stride) + offset];
            });

        m_write_head = (write_head + count) % capacity;
        m_size += count;
    }

    /**
     * @brief Compact current region keeping only marked elements
     *
     * Uses stream compaction pattern with Thrust.
     */
    void keep(const thrust::device_vector<bool>& keep_mask) {
        const SizeType count =
            thrust::count(keep_mask.begin(), keep_mask.end(), true);

        if (count == m_size) {
            return; // Nothing to remove
        }

        if (count == 0) {
            m_size = 0;
            if (m_is_updating) {
                m_write_head = m_write_start;
            }
            return;
        }

        const SizeType start_idx = get_current_start_idx();

        // Compute exclusive scan of keep_mask to get destination indices
        thrust::device_vector<SizeType> dst_offsets(m_size);
        thrust::exclusive_scan(thrust::device, keep_mask.begin(),
                               keep_mask.end(), dst_offsets.begin(),
                               SizeType{0}, thrust::plus<SizeType>());

        // Get raw pointers
        double* leaves_ptr      = thrust::raw_pointer_cast(m_leaves.data());
        FoldTypeCUDA* folds_ptr = thrust::raw_pointer_cast(m_folds.data());
        float* scores_ptr       = thrust::raw_pointer_cast(m_scores.data());

        const bool* mask_ptr = thrust::raw_pointer_cast(keep_mask.data());
        const SizeType* offset_ptr =
            thrust::raw_pointer_cast(dst_offsets.data());

        const SizeType leaves_stride = m_leaves_stride;
        const SizeType folds_stride  = m_folds_stride;
        const SizeType capacity      = m_capacity;

        // We need temporary storage for compacted data to avoid overwrites
        thrust::device_vector<double> temp_leaves(count * leaves_stride);
        thrust::device_vector<FoldTypeCUDA> temp_folds(count * folds_stride);
        thrust::device_vector<float> temp_scores(count);

        double* temp_leaves_ptr = thrust::raw_pointer_cast(temp_leaves.data());
        FoldTypeCUDA* temp_folds_ptr =
            thrust::raw_pointer_cast(temp_folds.data());
        float* temp_scores_ptr = thrust::raw_pointer_cast(temp_scores.data());

        // Copy kept elements to temporary storage
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(m_size),
            [mask_ptr, offset_ptr, scores_ptr, temp_scores_ptr, start_idx,
             capacity] __device__(SizeType i) {
                if (mask_ptr[i]) {
                    const SizeType src_phys        = (start_idx + i) % capacity;
                    temp_scores_ptr[offset_ptr[i]] = scores_ptr[src_phys];
                }
            });

        // Copy leaves
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(m_size),
            [mask_ptr, offset_ptr, leaves_ptr, temp_leaves_ptr, start_idx,
             capacity, leaves_stride] __device__(SizeType i) {
                if (mask_ptr[i]) {
                    const SizeType src_phys = (start_idx + i) % capacity;
                    const SizeType dst_off  = offset_ptr[i];
                    for (SizeType j = 0; j < leaves_stride; ++j) {
                        temp_leaves_ptr[(dst_off * leaves_stride) + j] =
                            leaves_ptr[(src_phys * leaves_stride) + j];
                    }
                }
            });

        // Copy folds
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(m_size),
            [mask_ptr, offset_ptr, folds_ptr, temp_folds_ptr, start_idx,
             capacity, folds_stride] __device__(SizeType i) {
                if (mask_ptr[i]) {
                    const SizeType src_phys = (start_idx + i) % capacity;
                    const SizeType dst_off  = offset_ptr[i];
                    for (SizeType j = 0; j < folds_stride; ++j) {
                        temp_folds_ptr[(dst_off * folds_stride) + j] =
                            folds_ptr[(src_phys * folds_stride) + j];
                    }
                }
            });

        // Copy back from temporary storage to main buffers starting at
        // start_idx
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(count),
            [scores_ptr, temp_scores_ptr, start_idx,
             capacity] __device__(SizeType i) {
                const SizeType dst_phys = (start_idx + i) % capacity;
                scores_ptr[dst_phys]    = temp_scores_ptr[i];
            });

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(count),
            [leaves_ptr, temp_leaves_ptr, start_idx, capacity,
             leaves_stride] __device__(SizeType i) {
                const SizeType dst_phys = (start_idx + i) % capacity;
                for (SizeType j = 0; j < leaves_stride; ++j) {
                    leaves_ptr[(dst_phys * leaves_stride) + j] =
                        temp_leaves_ptr[(i * leaves_stride) + j];
                }
            });

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(SizeType{0}),
            thrust::make_counting_iterator(count),
            [folds_ptr, temp_folds_ptr, start_idx, capacity,
             folds_stride] __device__(SizeType i) {
                const SizeType dst_phys = (start_idx + i) % capacity;
                for (SizeType j = 0; j < folds_stride; ++j) {
                    folds_ptr[(dst_phys * folds_stride) + j] =
                        temp_folds_ptr[(i * folds_stride) + j];
                }
            });

        m_size = count;
        if (m_is_updating) {
            m_write_head = (m_write_start + m_size) % m_capacity;
        }
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