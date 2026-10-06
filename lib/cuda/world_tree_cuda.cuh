#pragma once

/**
 * @file world_tree_cuda.cuh
 * @brief Device-memory WorldTree for CUDA EP pruning. Internal.
 */

#include <cstdint>
#include <tuple>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"
#include "lib/cuda/types_cuda.cuh"

namespace loki::memory {

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

    [[nodiscard]] CircularViewCUDA<FoldTypeCUDA>
    get_folds_circular_view() noexcept;
    [[nodiscard]] CircularViewCUDA<const FoldTypeCUDA>
    get_folds_circular_view() const noexcept;

    /// @brief Returns a zero-copy two-part view of the scores circular buffer
    /// (for saving to file). Both spans point directly into m_scores.
    [[nodiscard]] CircularViewCUDA<float> get_scores_circular_view() noexcept;
    [[nodiscard]] CircularViewCUDA<const float>
    get_scores_circular_view() const noexcept;

    [[nodiscard]] CircularViewCUDA<float>
    get_scores_ep_circular_view() noexcept;
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

} // namespace loki::memory
