#pragma once

#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

namespace loki::memory {

// Helper structure to represent the two contiguous segments
// of a circular buffer.
template <typename T> struct CircularView {
    std::span<T> first;
    std::span<T> second; // Empty if buffer doesn't wrap
};

/**
 * @brief A memory-efficient circular buffer for managing hierarchical search
 * candidates.
 *
 * This class implements a circular buffer for iterative search algorithms to
 * store, prune, and update "candidates". It minimizes memory allocations by
 * avoiding defragmentation and using in-place updates.
 *
 * **Circular Buffer Design:**
 * - Buffer is never defragmented - data remains in circular layout
 * - New candidates are generated from old ones within the same buffer, split
 *   into a "read region" (current iteration's input) and a "write region"
 *   (next iteration's candidates).
 * - Automatic pruning when buffer fills up
 *
 * **State Transitions:**
 * 1. `prepare_for_in_place_update()` - Split buffer: READ region (old) + WRITE
 * region (new, empty)
 * 2. `get_leaves_span()` + `advance_read_consumed()` - Read old data
 * incrementally
 * 3. `add_batch()` - Write new data, auto-trim if needed
 * 4. `finalize_in_place_update()` - Promote WRITE region to be the new READ
 * region
 *
 * WorldTree contains following arrays:
 * - leaves: Parameter sets, shape (capacity, nparams + 2, 2)
 * - folds: Folded profiles, shape (capacity, 2, nbins)
 * - scores: Scores for each leaf, shape (capacity)
 *
 * @tparam FoldType Element type for folded profiles.
 */
template <SupportedFoldType FoldType = float> class WorldTree {
public:
    WorldTree() = default;

    /**
     * @brief Constructor for the WorldTree class.
     *
     * Initializes the internal arrays with the given maximum number of
     * candidates, number of parameters, and number of bins.
     *
     * @param capacity Maximum number of candidates to hold
     * @param nparams Number of parameters
     * @param nbins Number of bins
     * @param max_batch_size Maximum number of candidates that can be added in a
     * single batch. This is used to allocate the scratch buffer.
     */
    WorldTree(SizeType capacity,
              SizeType nparams,
              SizeType nbins,
              SizeType max_batch_size);

    ~WorldTree()                               = default;
    WorldTree(const WorldTree&)                = delete;
    WorldTree& operator=(const WorldTree&)     = delete;
    WorldTree(WorldTree&&) noexcept            = default;
    WorldTree& operator=(WorldTree&&) noexcept = default;

    // Getters
    [[nodiscard]] const std::vector<double>& get_leaves() const noexcept {
        return m_leaves;
    }
    [[nodiscard]] const std::vector<FoldType>& get_folds() const noexcept {
        return m_folds;
    }
    [[nodiscard]] const std::vector<float>& get_scores() const noexcept {
        return m_scores;
    }
    [[nodiscard]] SizeType get_capacity() const noexcept { return m_capacity; }
    [[nodiscard]] SizeType get_nparams() const noexcept { return m_nparams; }
    [[nodiscard]] SizeType get_nbins() const noexcept { return m_nbins; }
    [[nodiscard]] SizeType get_max_batch_size() const noexcept {
        return m_max_batch_size;
    }
    [[nodiscard]] SizeType get_leaves_stride() const noexcept {
        return m_leaves_stride;
    }
    [[nodiscard]] SizeType get_folds_stride() const noexcept {
        return m_folds_stride;
    }
    [[nodiscard]] SizeType get_size() const noexcept { return m_size; }
    [[nodiscard]] SizeType get_size_old() const noexcept { return m_size_old; }
    [[nodiscard]] float get_size_lb() const noexcept;
    /// @brief Get maximum score in current region
    [[nodiscard]] float get_score_max() const noexcept;
    /// @brief Get minimum score in current region
    [[nodiscard]] float get_score_min() const noexcept;
    /// @brief Estimate memory usage in GiB, includes both base storage and
    // estimated peak temporary allocations (scratch buffer).
    [[nodiscard]] float get_memory_usage_gib() const noexcept;

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
    [[nodiscard]] std::pair<std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const;

    /**
     * @brief Get a mutable span over leaves for processing
     *
     * @param n_leaves Number of leaves to access
     * @return Pair of (span, actual_size), where actual_size <= requested
     * n_leaves, limited to contiguous segment before wrap.
     */
    [[nodiscard]] std::pair<std::span<double>, SizeType>
    get_leaves_span(SizeType n_leaves);
    /// @brief Returns a zero-copy two-part view of the leaves circular buffer
    /// (for in-place reporting). Both spans point directly into m_leaves.
    [[nodiscard]] CircularView<double> get_leaves_circular_view() noexcept;
    [[nodiscard]] CircularView<const double>
    get_leaves_circular_view() const noexcept;

    /// @brief Returns a zero-copy two-part view of the leaves circular buffer
    /// (for in-place reporting). Both spans point directly into m_leaves.
    [[nodiscard]] CircularView<FoldType> get_folds_circular_view() noexcept;
    [[nodiscard]] CircularView<const FoldType>
    get_folds_circular_view() const noexcept;

    /// @brief Returns a zero-copy two-part view of the scores circular buffer
    /// (for saving to file). Both spans point directly into m_scores.
    [[nodiscard]] CircularView<float> get_scores_circular_view() noexcept;
    [[nodiscard]] CircularView<const float>
    get_scores_circular_view() const noexcept;

    /// @brief Returns a zero-copy two-part view of the scores circular buffer
    /// (for saving to file). Both spans point directly into m_scores.
    [[nodiscard]] CircularView<float> get_scores_ep_circular_view() noexcept;
    [[nodiscard]] CircularView<const float>
    get_scores_ep_circular_view() const noexcept;

    /// @brief Get physical start index
    [[nodiscard]] SizeType get_physical_start_idx() const;

    /// @brief Set size externally (for initialization)
    void set_size(SizeType size) noexcept;
    /// @brief Reset buffer to empty state
    void reset() noexcept;
    /// @brief Prepare for in-place update, freezes current data as read region,
    /// opens write region.
    void prepare_in_place_update();
    /// @brief Finalize in-place update, promotes the write region to be the new
    /// read region. All old data must have been consumed.
    void finalize_in_place_update();
    /// @brief Advance read consumed counter
    void consume_read(SizeType n);

    /// @brief Convert logical indices to physical indices
    /**
     * @brief Convert logical indices to physical indices
     *
     * @param logical_indices Logical indices to convert
     * @param n_leaves Number of leaves to convert
     */
    void convert_to_physical_indices(std::span<SizeType> logical_indices,
                                     SizeType n_leaves) const;

    // Get the best candidate (highest score)
    [[nodiscard]] std::
        tuple<std::span<const double>, std::span<const FoldType>, float>
        get_best() const;

    /**
     * @brief Add initial batch (resets buffer first)
     */
    void add_initial(std::span<const double> leaves_batch,
                     std::span<const FoldType> folds_batch,
                     std::span<const float> scores_batch,
                     SizeType slots_to_write);
    // Add a candidate leaf to the Tree if there is space
    [[nodiscard]] bool add(std::span<const double> leaf,
                           std::span<const FoldType> fold,
                           float score);

    /**
     * @brief Add batch during update with threshold filtering, scattered.
     *
     * @return Updated threshold after any trimming
     * Adds filtered batch to write region. If full, trims write region via
     * top-k threshold. Makes sure all candidates fit, reclaiming space
     * from consumed old candidates.
     * leaves_batch, folds_batch and scores_batch are scattered and
     * indices_batch is the physical indices of the leaves.
     */
    [[nodiscard]] float
    add_batch_scattered(std::span<const double> leaves_batch,
                        std::span<const FoldType> folds_batch,
                        std::span<const float> scores_batch,
                        std::span<const SizeType> indices_batch,
                        float current_threshold,
                        SizeType slots_to_write);
    // Prune to keep only unique candidates
    void deduplicate();

    void validate(SizeType capacity,
                  SizeType nparams,
                  SizeType nbins,
                  SizeType max_batch_size) const;

private:
    constexpr static SizeType kParamStride = 2;

    // Configuration
    SizeType m_capacity{};
    SizeType m_nparams{};
    SizeType m_nbins{};
    SizeType m_max_batch_size{};
    SizeType m_leaves_stride{};
    SizeType m_folds_stride{};

    // Host-side storage
    std::vector<double> m_leaves;  // Shape: (capacity, nparams + 2, 2)
    std::vector<FoldType> m_folds; // Shape: (capacity, 2, nbins)
    std::vector<float> m_scores;   // Shape: (capacity)
    std::vector<float> m_scores_ep; // Shape: (capacity)

    // Circular buffer state
    bool m_is_updating{false};
    SizeType m_head{0};     // Index of the first valid element
    SizeType m_size{0};     // Number of valid elements in the buffer
    SizeType m_size_old{0}; // Number of elements from previous iteration

    // In-place update state
    SizeType m_write_head{0}; // Index where next element will be written
    SizeType m_write_start{0};
    SizeType m_read_consumed{0};

    // Scratch buffer for in-place operations
    std::vector<float> m_scratch_scores;
    std::vector<SizeType> m_scratch_pending_indices;
    std::vector<uint8_t> m_scratch_mask;

    // Generic helper to get active regions for any vector
    template <typename T>
    CircularView<T> get_active_regions(std::span<T> arr,
                                       SizeType stride = 1) const noexcept;

    /**
     * @brief Copy slots from contiguous source to circular buffer
     *
     * @param src Source pointer (contiguous)
     * @param dst Destination pointer (circular buffer)
     * @param dst_start_slot Starting slot in destination (circular buffer)
     * @param slots Number of slots to copy
     * @param stride Stride of the elements
     */
    template <typename T>
    void copy_to_circular(const T* __restrict__ src,
                          T* __restrict__ dst,
                          SizeType dst_start_slot,
                          SizeType slots,
                          SizeType stride) const noexcept;

    void scatter_to_circular_copy(const double* __restrict__ src_leaves,
                                  const FoldType* __restrict__ src_folds,
                                  const float* __restrict__ src_scores,
                                  const SizeType* __restrict__ src_indices,
                                  SizeType slots_to_write) noexcept;

    /**
     * @brief Copy slots from circular buffer to contiguous destination
     *
     * @param src Source pointer (circular buffer)
     * @param src_start_slot Starting slot in source (circular buffer)
     * @param slots Number of slots to copy from circular buffer
     * @param stride Stride of the elements
     * @param dst Destination pointer (contiguous)
     */
    template <typename T>
    void copy_from_circular(const T* __restrict__ src,
                            SizeType src_start_slot,
                            SizeType slots,
                            SizeType stride,
                            T* __restrict__ dst) const noexcept;

    /**
     * @brief Get starting index for current region
     */
    SizeType get_current_start_idx() const noexcept;

    /**
     * @brief Compute physical index from logical index in circular buffer
     * @param logical_idx Logical index (0-based from start of valid region)
     * @param start Starting physical index of the region
     * @param capacity Total buffer capacity
     * @return Physical index in the buffer
     */
    static constexpr SizeType get_circular_index(SizeType logical_idx,
                                                 SizeType start,
                                                 SizeType capacity) noexcept;

    /**
     * @brief Calculate available space in buffer
     * @return The number of slots available in the write region
     */
    SizeType calculate_space_left() const;

    /**
     * @brief Get prune threshold in current region
     *
     * find a threshold in (Buffer + Batch) so that keeping only scores
     * strictly above it yields at most total_capacity items.
     */
    float get_prune_threshold(std::span<const float> scores_batch,
                              std::span<const SizeType> indices_batch,
                              SizeType slots_to_write,
                              float current_threshold) noexcept;

    /**
     * @brief Prune write region by threshold with in-place update.
     * @param threshold The threshold to use
     *
     * Single-pass approach: safely compacts in-place because write never
     * overtakes read (write_logical ≤ read_logical always holds).
     */
    void prune_on_overload(float threshold);

    /**
     * @brief Compact current region keeping only marked elements
     * using boolean mask. Only affects [start_idx, start_idx + m_size);
     * updates m_size.
     */
    void keep(std::span<const uint8_t> keep_mask);

    // Memory-efficient uniqueness detection
    // Tie-break: keep first occurrence when scores are equal.
    void compute_uniqueness_mask_in_scratch() noexcept;

    void validate_circular_buffer_state();
}; // End WorldTree definition

using WorldTreeFloat   = WorldTree<float>;
using WorldTreeComplex = WorldTree<ComplexType>;

#ifdef LOKI_ENABLE_CUDA

// Helper structure to represent the two contiguous segments
// of a circular buffer.
template <typename T> struct CircularViewCUDA {
    cuda::std::span<T> first;
    cuda::std::span<T> second; // Empty if buffer doesn't wrap
};

template <SupportedFoldTypeCUDA FoldTypeCUDA = float> class WorldTreeCUDA {
public:
    WorldTreeCUDA() = default;

    /**
     * @brief Constructor for the WorldTreeCUDA class.
     *
     * Initializes the internal arrays with the given maximum number of
     * candidates, number of parameters, and number of bins.
     *
     * @param capacity Maximum number of candidates to hold
     * @param nparams Number of parameters
     * @param nbins Number of bins
     * @param max_batch_size Maximum number of candidates that can be added in a
     * single batch. This is used to allocate the scratch buffer.
     */
    WorldTreeCUDA(SizeType capacity,
                  SizeType nparams,
                  SizeType nbins,
                  SizeType max_batch_size);

    ~WorldTreeCUDA()                                   = default;
    WorldTreeCUDA(const WorldTreeCUDA&)                = delete;
    WorldTreeCUDA& operator=(const WorldTreeCUDA&)     = delete;
    WorldTreeCUDA(WorldTreeCUDA&&) noexcept            = default;
    WorldTreeCUDA& operator=(WorldTreeCUDA&&) noexcept = default;

    // Getters
    [[nodiscard]] cuda::std::span<const double>
    get_leaves_span() const noexcept;
    [[nodiscard]] cuda::std::span<const FoldTypeCUDA>
    get_folds_span() const noexcept;
    [[nodiscard]] cuda::std::span<const float> get_scores_span() const noexcept;
    [[nodiscard]] SizeType get_capacity() const noexcept { return m_capacity; }
    [[nodiscard]] SizeType get_nparams() const noexcept { return m_nparams; }
    [[nodiscard]] SizeType get_nbins() const noexcept { return m_nbins; }
    [[nodiscard]] SizeType get_max_batch_size() const noexcept {
        return m_max_batch_size;
    }
    [[nodiscard]] SizeType get_leaves_stride() const noexcept {
        return m_leaves_stride;
    }
    [[nodiscard]] SizeType get_folds_stride() const noexcept {
        return m_folds_stride;
    }
    [[nodiscard]] SizeType get_size() const noexcept { return m_size; }
    [[nodiscard]] SizeType get_size_old() const noexcept { return m_size_old; }

    /// @brief Get size in log2(size), useful for reporting.
    [[nodiscard]] float get_size_lb() const noexcept;
    /// @brief Get maximum score in current region
    [[nodiscard]] float get_score_max(cudaStream_t stream) const noexcept;
    /// @brief Get minimum score in current region
    [[nodiscard]] float get_score_min(cudaStream_t stream) const noexcept;
    /// @brief Estimate GPU memory usage in GiB, includes both base storage and
    /// estimated peak temporary allocations.
    [[nodiscard]] float get_memory_usage_gib() const noexcept;

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
    [[nodiscard]] std::pair<cuda::std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const;

    /**
     * @brief Get mutable span over leaves for processing
     *
     * During updates, returns span over readable (old) region.
     * The span may be truncated at buffer wrap point.
     *
     * @param n_leaves Number of leaves to access
     * @return Pair of (span, actual_size), where actual_size <= requested
     * n_leaves, limited to contiguous segment before wrap.
     */
    [[nodiscard]] std::pair<cuda::std::span<double>, SizeType>
    get_leaves_span(SizeType n_leaves);

    /// @brief Returns a zero-copy two-part view of the leaves circular buffer
    /// (for in-place reporting). Both spans point directly into m_leaves.
    [[nodiscard]] CircularViewCUDA<double> get_leaves_circular_view() noexcept;
    [[nodiscard]] CircularViewCUDA<const double>
    get_leaves_circular_view() const noexcept;

    [[nodiscard]] CircularViewCUDA<FoldTypeCUDA> get_folds_circular_view() noexcept;
    [[nodiscard]] CircularViewCUDA<const FoldTypeCUDA>
    get_folds_circular_view() const noexcept;

    /// @brief Returns a zero-copy two-part view of the scores circular buffer
    /// (for saving to file). Both spans point directly into m_scores.
    [[nodiscard]] CircularViewCUDA<float> get_scores_circular_view() noexcept;
    [[nodiscard]] CircularViewCUDA<const float>
    get_scores_circular_view() const noexcept;

    [[nodiscard]] CircularViewCUDA<float> get_scores_ep_circular_view() noexcept;
    [[nodiscard]] CircularViewCUDA<const float>
    get_scores_ep_circular_view() const noexcept;

    /// @brief Get physical start index
    [[nodiscard]] SizeType get_physical_start_idx() const;

    // Circular buffer control
    /// @brief Set size externally (for initialization)
    void set_size(SizeType size) noexcept;
    /// @brief Reset buffer to empty state
    void reset() noexcept;
    /// @brief Prepare for in-place update, freezes current data as read region,
    /// opens write region.
    void prepare_in_place_update();
    /// @brief Finalize in-place update, promotes the write region to be the new
    /// read region. All old data must have been consumed.
    void finalize_in_place_update();
    /// @brief Advance read consumed counter
    void consume_read(SizeType n);

    // Add an initial set of candidate leaves to the Tree (reset buffer first)
    void add_initial(cuda::std::span<const double> leaves_batch,
                     cuda::std::span<const FoldTypeCUDA> folds_batch,
                     cuda::std::span<const float> scores_batch,
                     SizeType slots_to_write,
                     cudaStream_t stream);
    /**
     * @brief Add batch during update with threshold filtering, scattered.
     *
     * @return Updated threshold after any trimming
     * Adds filtered batch to write region. If full, trims write region via
     * top-k threshold. Makes sure all candidates fit, reclaiming space
     * from consumed old candidates.
     * leaves_batch, folds_batch and scores_batch are scattered and
     * indices_batch is the physical indices of the leaves.
     */
    [[nodiscard]] float
    add_batch_scattered(cuda::std::span<const double> leaves_batch,
                        cuda::std::span<const FoldTypeCUDA> folds_batch,
                        cuda::std::span<const float> scores_batch,
                        cuda::std::span<const uint32_t> indices_batch,
                        float current_threshold,
                        SizeType slots_to_write,
                        cudaStream_t stream);

    // Validation
    void validate(SizeType capacity,
                  SizeType nparams,
                  SizeType nbins,
                  SizeType max_batch_size) const;

private:
    static constexpr SizeType kParamStride = 2;

    // Configuration
    SizeType m_capacity{};
    SizeType m_nparams{};
    SizeType m_nbins{};
    SizeType m_max_batch_size{};
    SizeType m_leaves_stride{};
    SizeType m_folds_stride{};

    // Device storage
    thrust::device_vector<double> m_leaves;
    thrust::device_vector<FoldTypeCUDA> m_folds;
    thrust::device_vector<float> m_scores;
    thrust::device_vector<float> m_scores_ep;

    // Circular buffer state (host-side tracking)
    bool m_is_updating{false};
    SizeType m_head{0};
    SizeType m_size{0};
    SizeType m_size_old{0};
    SizeType m_write_head{0};
    SizeType m_write_start{0};
    SizeType m_read_consumed{0};

    // Scratch buffer for in-place operations
    thrust::device_vector<float> m_scratch_scores;
    thrust::device_vector<uint32_t> m_scratch_indices_1;
    thrust::device_vector<uint32_t> m_scratch_indices_2;
    thrust::device_vector<uint8_t> m_scratch_mask;

    // Generic helper to get active regions for any vector
    template <typename T>
    CircularViewCUDA<T> get_active_regions(cuda::std::span<T> arr,
                                           SizeType stride = 1) const noexcept;

    /**
     * @brief Copy slots from circular buffer to contiguous destination
     *
     * @param src Source pointer (circular buffer)
     * @param src_start_slot Starting slot in source (circular buffer)
     * @param slots Number of slots to copy from circular buffer
     * @param stride Stride of the elements
     * @param dst Destination pointer (contiguous)
     */
    template <typename T>
    void copy_from_circular(const T* __restrict__ src,
                            SizeType src_start_slot,
                            SizeType slots,
                            SizeType stride,
                            T* __restrict__ dst,
                            cudaStream_t stream) const noexcept;

    /**
     * @brief Get the starting index for current region
     */
    SizeType get_current_start_idx() const noexcept;

    /**
     * @brief Compute physical index from logical index in circular buffer
     * @param logical_idx Logical index (0-based from start of valid region)
     * @param start Starting physical index of the region
     * @param capacity Total buffer capacity
     * @return Physical index in the buffer
     */
    static constexpr SizeType get_circular_index(SizeType logical_idx,
                                                 SizeType start,
                                                 SizeType capacity) noexcept;

    /**
     * @brief Calculate available space in buffer
     */
    SizeType calculate_space_left() const noexcept;

    /**
     * @brief Get prune threshold in current region
     *
     * find a threshold in (Buffer + Batch) so that keeping only scores strictly
     * above it yields at most total_capacity items.
     */
    float get_prune_threshold(cuda::std::span<const float> scores_batch,
                              cuda::std::span<const uint32_t> indices_batch,
                              SizeType slots_to_write,
                              float current_threshold,
                              cudaStream_t stream) noexcept;

    /**
     * @brief Prune write region by threshold with in-place update.
     * @param mode The prune mode to use
     * @return The threshold used
     *
     * Single-pass approach: safely compacts in-place because write never
     * overtakes read (write_logical ≤ read_logical always holds).
     */
    void prune_on_overload(float threshold, cudaStream_t stream);

    void validate_circular_buffer_state();
};

using WorldTreeCUDAFloat   = WorldTreeCUDA<float>;
using WorldTreeCUDAComplex = WorldTreeCUDA<ComplexTypeCUDA>;
#endif // LOKI_ENABLE_CUDA

} // namespace loki::memory