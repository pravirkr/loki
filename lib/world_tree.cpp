#include "loki/utils/world_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <unordered_map>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"

namespace loki::utils {

namespace {
struct BestIndices {
    float score;
    SizeType win_idx;
};

} // namespace

/**
Circular buffer layout (all indices modulo m_capacity):

┌────────┬───────────────┬───────────────┐
│ unused │  READ REGION  │  WRITE REGION │
└────────┴───────────────┴───────────────┘
           ^ m_head         ^ m_write_start
           size = m_size_old      size = m_size
During an iteration we:
1. prepare_in_place_update() – freezes READ, opens empty WRITE.
2. Consumers call consume_read() as they read.
3. Producers call add()/add_batch() into WRITE.
4. If WRITE fills, prune_on_overload() may delete from WRITE only.
5. finalize_in_place_update() – promotes WRITE to READ.
6. Buffer is circular and not defragmented. It starts at m_head and wraps
   around to m_write_start.

All operations keep the invariant
   m_size_old - m_read_consumed + m_size ≤ m_capacity
- Read region: Old candidates (m_head to m_head + m_size_old -
m_read_consumed).
- Write region: New candidates (m_write_start to m_write_head).
*/
template <SupportedFoldType FoldType> class WorldTree<FoldType>::Impl {
public:
    Impl(SizeType capacity,
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
          m_folds(m_capacity * m_folds_stride, FoldType{}),
          m_scores(m_capacity, 0.0F),
          m_scratch_leaves(m_capacity * (nparams * kParamStride), 0.0),
          m_scratch_scores((m_capacity + max_batch_size), 0.0F),
          m_scratch_pending_indices(max_batch_size, 0),
          m_scratch_mask(m_capacity, 0) {
        // Validate inputs
        error_check::check_greater(
            m_capacity, 0, "WorldTree: capacity must be greater than 0");
        error_check::check_greater(m_nparams, 0,
                                   "WorldTree: nparams must be greater than 0");
        error_check::check_greater(m_nbins, 0,
                                   "WorldTree: nbins must be greater than 0");
        error_check::check_greater(
            m_max_batch_size, 0,
            "WorldTree: max_batch_size must be greater than 0");
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    // Size and getters
    const std::vector<double>& get_leaves() const noexcept { return m_leaves; }
    const std::vector<FoldType>& get_folds() const noexcept { return m_folds; }
    const std::vector<float>& get_scores() const noexcept { return m_scores; }
    SizeType get_capacity() const noexcept { return m_capacity; }
    SizeType get_nparams() const noexcept { return m_nparams; }
    SizeType get_nbins() const noexcept { return m_nbins; }
    SizeType get_max_batch_size() const noexcept { return m_max_batch_size; }
    SizeType get_leaves_stride() const noexcept { return m_leaves_stride; }
    SizeType get_folds_stride() const noexcept { return m_folds_stride; }
    SizeType get_size() const noexcept { return m_size; }
    SizeType get_size_old() const noexcept { return m_size_old; }
    float get_size_lb() const noexcept {
        return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
    }

    /**
     * @brief Get maximum score in current region
     */
    float get_score_max() const noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto regions = get_active_regions(m_scores);
        float max_val      = *std::ranges::max_element(regions.first);
        if (!regions.second.empty()) {
            max_val =
                std::max(max_val, *std::ranges::max_element(regions.second));
        }
        return max_val;
    }

    /**
     * @brief Get minimum score in current region
     */
    float get_score_min() const noexcept {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto regions = get_active_regions(m_scores);
        float min_val      = *std::ranges::min_element(regions.first);
        if (!regions.second.empty()) {
            min_val =
                std::min(min_val, *std::ranges::min_element(regions.second));
        }
        return min_val;
    }

    /**
     * @brief Estimate memory usage in GiB
     *
     * Includes both base storage and estimated peak temporary allocations
     * (scratch buffer).
     */
    float get_memory_usage() const noexcept {
        const auto base_bytes =
            (m_leaves.size() * sizeof(double)) +
            (m_folds.size() * sizeof(FoldType)) +
            (m_scores.size() * sizeof(float)) +
            (m_scratch_leaves.size() * sizeof(double)) +
            (m_scratch_scores.size() * sizeof(float)) +
            (m_scratch_pending_indices.size() * sizeof(SizeType)) +
            (m_scratch_mask.size() * sizeof(uint8_t));
        // Peak temporary allocations (worst case scenario)
        // best_by_key map in compute_uniqueness_mask_inplace (worst case:
        // all unique)
        // const auto extra_temp_bytes =
        //    (m_capacity * (sizeof(int64_t) + sizeof(BestIndices)));
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
    std::pair<std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const {
        const SizeType available = m_size_old - m_read_consumed;
        error_check::check_less_equal(
            n_leaves, available,
            "get_leaves_span: requested range exceeds available data");

        // Compute physical start: relative to current head of read region
        const auto physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);
        // Compute how many contiguous elements we can take before wrap
        const auto actual_size =
            std::min(n_leaves, m_capacity - physical_start);
        // Return span with correct byte size (actual_size elements, not bytes)
        return {{m_leaves.data() + (physical_start * m_leaves_stride),
                 actual_size * m_leaves_stride},
                actual_size};
    }

    std::span<double> get_leaves_contiguous_span() noexcept {
        const auto start_idx     = get_current_start_idx();
        const auto report_stride = m_nparams * kParamStride;
        const auto first_part    = std::min(m_size, m_capacity - start_idx);
        for (SizeType i = 0; i < first_part; ++i) {
            const auto src_offset = (start_idx + i) * m_leaves_stride;
            const auto dst_offset = i * report_stride;
            std::copy_n(m_leaves.begin() + src_offset, report_stride,
                        m_scratch_leaves.begin() + dst_offset);
        }
        const auto second_part = m_size - first_part;
        if (second_part > 0) {
            for (SizeType i = 0; i < second_part; ++i) {
                const auto src_offset = i * m_leaves_stride;
                const auto dst_offset = (first_part + i) * report_stride;
                std::copy_n(m_leaves.begin() + src_offset, report_stride,
                            m_scratch_leaves.begin() + dst_offset);
            }
        }
        return {m_scratch_leaves.data(), m_size * report_stride};
    }

    /**
     * @brief Get span over contiguous scores (for saving to file)
     *
     * Returns span over contiguous scores for m_size elements in the current
     * region.
     *
     * @return Span over contiguous scores
     */
    std::span<float> get_scores_contiguous_span() noexcept {
        const auto start_idx = get_current_start_idx();
        copy_from_circular(m_scores.data() + start_idx, start_idx, m_size,
                           SizeType{1}, m_scratch_scores.data());
        return {m_scratch_scores.data(), m_size};
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
            m_size, m_capacity, "WorldTree: Invalid size after set_size()");
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
        error_check::check(
            !m_is_updating,
            "WorldTree: Cannot prepare for update while already updating");
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
                                      "WorldTree: read_consumed overflow");
        m_read_consumed += n;

        // Validate circular buffer invariant
        error_check::check_less_equal(
            m_size_old - m_read_consumed + m_size, m_capacity,
            "WorldTree: circular buffer invariant violated");
    }

    /**
     * @brief Compute physical indices from logical indices
     */
    void compute_physical_indices(std::span<const SizeType> logical_indices,
                                  std::span<SizeType> physical_indices,
                                  SizeType n_leaves) const {
        error_check::check(
            m_is_updating,
            "WorldTree: compute_physical_indices only valid during updates");
        error_check::check_greater_equal(
            logical_indices.size(), n_leaves,
            "compute_physical_indices: logical_indices size insufficient");
        error_check::check_greater_equal(
            physical_indices.size(), n_leaves,
            "compute_physical_indices: physical_indices size insufficient");

        // Compute physical start: relative to current head of read region
        const auto physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);
        for (SizeType i = 0; i < n_leaves; ++i) {
            physical_indices[i] = get_circular_index(
                logical_indices[i], physical_start, m_capacity);
        }
    }

    std::tuple<std::span<const double>, std::span<const FoldType>, float>
    get_best() const {
        if (m_size == 0) {
            return {{}, {}, 0.0F};
        }

        const auto regions = get_active_regions(m_scores);
        // Find max iterator in first region
        auto it1              = std::ranges::max_element(regions.first);
        float max_score       = *it1;
        SizeType best_offset  = std::distance(regions.first.begin(), it1);
        SizeType absolute_idx = get_current_start_idx() + best_offset;

        // Check second region if it exists
        if (!regions.second.empty()) {
            auto it2 = std::ranges::max_element(regions.second);
            if (*it2 > max_score) {
                max_score    = *it2;
                best_offset  = std::distance(regions.second.begin(), it2);
                absolute_idx = best_offset; // Second region starts at 0
            }
        }
        return {std::span{m_leaves.data() + (absolute_idx * m_leaves_stride),
                          m_leaves_stride},
                std::span{m_folds.data() + (absolute_idx * m_folds_stride),
                          m_folds_stride},
                max_score};
    }

    /**
     * @brief Add initial batch (resets buffer first)
     */
    void add_initial(std::span<const double> batch_leaves,
                     std::span<const FoldType> batch_folds,
                     std::span<const float> batch_scores,
                     SizeType slots_to_write) {
        error_check::check_less_equal(
            slots_to_write, m_capacity,
            "WorldTree: Suggestions too large to add.");
        error_check::check_equal(slots_to_write, batch_scores.size(),
                                 "slots_to_write must match batch_scores size");

        reset(); // Start fresh
        std::copy_n(batch_leaves.begin(), slots_to_write * m_leaves_stride,
                    m_leaves.begin());
        std::copy_n(batch_folds.begin(), slots_to_write * m_folds_stride,
                    m_folds.begin());
        std::copy_n(batch_scores.begin(), slots_to_write, m_scores.begin());
        m_size = slots_to_write;
        error_check::check_less_equal(
            m_size, m_capacity, "WorldTree: Invalid size after add_initial.");
    }

    bool add(std::span<const double> leaf,
             std::span<const FoldType> fold,
             float score) {
        if (m_size_old + m_size >= m_capacity) {
            return false;
        }
        const auto write_idx = m_write_head;
        std::ranges::copy(leaf,
                          m_leaves.begin() + (write_idx * m_leaves_stride));
        std::ranges::copy(fold, m_folds.begin() + (write_idx * m_folds_stride));
        m_scores[write_idx] = score;
        m_write_head        = get_circular_index(1, m_write_head, m_capacity);
        ++m_size;
        return true;
    }

    /**
     * @brief Add batch during update with threshold filtering
     *
     * @return Updated threshold after any trimming
     * Adds filtered batch to write region. If full, trims write region via
     * top-k threshold. Makes sure all candidates fit, reclaiming space
     * from consumed old candidates.
     */
    float add_batch(std::span<const double> batch_leaves,
                    std::span<const FoldType> batch_folds,
                    std::span<const float> batch_scores,
                    float current_threshold,
                    SizeType slots_to_write) {
        // Always use scores_batch to get the correct batch size
        if (slots_to_write == 0) {
            return current_threshold;
        }
        // Fast path: Check if we have enough space immediately
        IndexType space_left = calculate_space_left();
        if (static_cast<IndexType>(slots_to_write) <= space_left) {
            copy_to_circular(batch_leaves.data(), m_leaves.data(), m_write_head,
                             slots_to_write, m_leaves_stride);
            copy_to_circular(batch_folds.data(), m_folds.data(), m_write_head,
                             slots_to_write, m_folds_stride);
            copy_to_circular(batch_scores.data(), m_scores.data(), m_write_head,
                             slots_to_write, SizeType{1});
            m_write_head =
                get_circular_index(slots_to_write, m_write_head, m_capacity);
            m_size += slots_to_write;
            return current_threshold;
        }

        // Slow path: Overflow & Pruning (Median + Top-K Strategy)
        error_check::check_less_equal(
            slots_to_write, m_max_batch_size,
            "WorldTree: Suggestions too large to add.");
        // Determine the Global Pruning Threshold
        const float effective_threshold = get_prune_threshold(
            batch_scores, slots_to_write, current_threshold);
        // Remove old items that are strictly worse than the new global pruning
        // threshold.
        prune_on_overload(effective_threshold);

        // Add Qualifying Batch Items
        auto pending_indices =
            std::span(m_scratch_pending_indices.data(), slots_to_write);
        SizeType pending_count = 0;
        for (SizeType i = 0; i < slots_to_write; ++i) {
            if (batch_scores[i] > effective_threshold) {
                pending_indices[pending_count++] = i;
            }
        }
        space_left = calculate_space_left();
        const auto n_to_add =
            std::min(pending_count, static_cast<SizeType>(space_left));
        if (n_to_add == 0) {
            return effective_threshold;
        }

        // Batched copy
        for (SizeType i = 0; i < n_to_add; ++i) {
            const auto src_idx = pending_indices[i];
            const auto dst_idx = m_write_head;
            std::copy_n(batch_leaves.begin() + (src_idx * m_leaves_stride),
                        m_leaves_stride,
                        m_leaves.begin() + (dst_idx * m_leaves_stride));
            std::copy_n(batch_folds.begin() + (src_idx * m_folds_stride),
                        m_folds_stride,
                        m_folds.begin() + (dst_idx * m_folds_stride));
            m_scores[dst_idx] = batch_scores[src_idx];
            m_write_head++;
            if (m_write_head == m_capacity) {
                m_write_head = 0;
            }
        }

        m_size += n_to_add;
        return effective_threshold;
    }

    void deduplicate() {
        error_check::check(
            m_is_updating,
            "WorldTree: deduplicate only allowed during updates");
        if (m_size == 0) {
            return;
        }
        compute_uniqueness_mask_in_scratch();
        const auto keep_mask_span =
            std::span<const uint8_t>(m_scratch_mask.data(), m_size);
        keep(keep_mask_span);
    }

private:
    constexpr static SizeType kParamStride = 2;
    SizeType m_capacity;
    SizeType m_nparams;
    SizeType m_nbins;
    SizeType m_max_batch_size;
    SizeType m_leaves_stride{};
    SizeType m_folds_stride;

    std::vector<double> m_leaves;  // Shape: (capacity, nparams + 2, 2)
    std::vector<FoldType> m_folds; // Shape: (capacity, 2, nbins)
    std::vector<float> m_scores;   // Shape: (capacity)

    // Circular buffer state
    SizeType m_head{0};     // Index of the first valid element
    SizeType m_size{0};     // Number of valid elements in the buffer
    SizeType m_size_old{0}; // Number of elements from previous iteration

    // In-place update state
    SizeType m_write_head{0}; // Index where next element will be written
    SizeType m_write_start{0};
    bool m_is_updating{false};
    SizeType m_read_consumed{0};

    // Scratch buffer for in-place operations
    std::vector<double> m_scratch_leaves;
    std::vector<float> m_scratch_scores;
    std::vector<SizeType> m_scratch_pending_indices;
    std::vector<uint8_t> m_scratch_mask;

    // Helper structure to represent the two contiguous segments
    // of a circular buffer.
    template <typename T> struct CircularView {
        std::span<T> first;
        std::span<T> second; // Empty if buffer doesn't wrap
    };

    // Generic helper to get active regions for any vector
    template <typename T>
    CircularView<const T> get_active_regions(const std::vector<T>& arr,
                                             SizeType stride = 1) const {
        if (m_size == 0) {
            return {{}, {}};
        }

        const auto start = get_current_start_idx();
        // Handle stride for leaves/folds
        const auto start_offset = start * stride;
        if (start + m_size <= m_capacity) {
            return {{arr.data() + start_offset, m_size * stride}, {}};
        }
        const auto first_count  = m_capacity - start;
        const auto second_count = m_size - first_count;
        return {{arr.data() + start_offset, first_count * stride},
                {arr.data(), second_count * stride}};
    }

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
                          SizeType stride) const noexcept {
        const auto first_slots = std::min(slots, m_capacity - dst_start_slot);
        const auto first_elems = first_slots * stride;
        const auto dst_offset  = dst_start_slot * stride;

        std::copy_n(src, first_elems, dst + dst_offset);
        const auto second_slots = slots - first_slots;
        if (second_slots > 0) {
            const auto second_elems = second_slots * stride;
            std::copy_n(src + first_elems, second_elems, dst);
        }
    }

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
                            T* __restrict__ dst) const noexcept {
        const auto first_slots = std::min(slots, m_capacity - src_start_slot);
        const auto first_elems = first_slots * stride;
        const auto src_offset  = src_start_slot * stride;

        std::copy_n(src + src_offset, first_elems, dst);
        const auto second_slots = slots - first_slots;
        if (second_slots > 0) {
            const auto second_elems = second_slots * stride;
            std::copy_n(src, second_elems, dst + first_elems);
        }
    }

    /**
     * @brief Get starting index for current region
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
     * @return The number of slots available in the write region
     */
    IndexType calculate_space_left() const {
        const auto remaining_old = static_cast<IndexType>(m_size_old) -
                                   static_cast<IndexType>(m_read_consumed);
        error_check::check_greater_equal(
            remaining_old, 0, "calculate_space_left: Invalid buffer state");
        return static_cast<IndexType>(m_capacity) -
               (remaining_old + static_cast<IndexType>(m_size));
    }

    /**
     * @brief Get prune threshold in current region
     *
     * find a threshold in (Buffer + Batch) so that keeping only scores strictly
     * above it yields at most total_capacity items.
     */
    float get_prune_threshold(std::span<const float> batch_scores,
                              SizeType slots_to_write,
                              float current_threshold) noexcept {
        const auto space_left       = calculate_space_left();
        const auto total_candidates = m_size + slots_to_write;
        const auto total_capacity = m_size + static_cast<SizeType>(space_left);
        if (total_candidates <= total_capacity) {
            return current_threshold;
        }
        error_check::check_greater_equal(
            m_scratch_scores.size(), total_candidates,
            "get_prune_threshold: Invalid scratch buffer size");
        // Gather all candidates (Buffer + Batch) into scratch
        // Copy active circular buffer scores to scratch [0 ... m_size]
        const auto start_idx = get_current_start_idx();
        copy_from_circular(m_scores.data() + start_idx, start_idx, m_size,
                           SizeType{1}, m_scratch_scores.data());
        // Append batch scores [m_size ... total]
        std::copy_n(batch_scores.begin(), slots_to_write,
                    m_scratch_scores.begin() + m_size);

        // The item at 'total_capacity-1' (in descending order) is the smallest
        // score we keep (i.e. the threshold).
        const auto begin = m_scratch_scores.begin();
        const auto end   = begin + total_candidates;
        auto kth         = begin + total_capacity - 1;
        std::nth_element(begin, kth, end, std::greater<float>());
        // To break ties, we use the next representable value greater than the
        // topk score.
        const float topk_threshold =
            std::nextafter(*kth, std::numeric_limits<float>::infinity());

        // Median score for severity (restricted range)
        auto mid = begin + total_candidates / 2;
        if (mid <= kth) {
            std::nth_element(begin, mid, kth + 1);
        } else {
            std::nth_element(kth + 1, mid, end);
        }
        const float median_threshold =
            std::nextafter(*mid, std::numeric_limits<float>::infinity());
        return std::max({current_threshold, topk_threshold, median_threshold});
    }

    /**
     * @brief Prune write region by threshold with in-place update.
     * @param threshold The threshold to use
     *
     * Single-pass approach: safely compacts in-place because write never
     * overtakes read (write_logical ≤ read_logical always holds).
     */
    void prune_on_overload(float threshold) {
        error_check::check(
            m_is_updating,
            "WorldTree: prune_on_overload only allowed during updates");
        if (m_size == 0) {
            return;
        }
        const auto start_idx    = get_current_start_idx();
        const SizeType old_size = m_size;
        // Use logical indices to abstract circular wrapping
        SizeType phys_read  = start_idx;
        SizeType phys_write = start_idx;
        SizeType kept_count = 0;
        // Single-pass: scan and compact in-place
        for (SizeType read_logical = 0; read_logical < old_size;
             ++read_logical) {
            // Check filter condition
            if (m_scores[phys_read] > threshold) {
                // Move data only if gaps were created
                if (phys_read != phys_write) {
                    std::copy_n(
                        m_leaves.begin() + (phys_read * m_leaves_stride),
                        m_leaves_stride,
                        m_leaves.begin() + (phys_write * m_leaves_stride));
                    std::copy_n(m_folds.begin() + (phys_read * m_folds_stride),
                                m_folds_stride,
                                m_folds.begin() +
                                    (phys_write * m_folds_stride));
                    m_scores[phys_write] = m_scores[phys_read];
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
        m_write_head = phys_write;
        error_check::check_less_equal(m_size, m_capacity,
                                      "WorldTree: Invalid size after keep");
    }

    /**
     * @brief Compact current region keeping only marked elements
     * using boolean mask. Only affects [start_idx, start_idx + m_size);
     * updates m_size.
     */
    void keep(std::span<const uint8_t> keep_mask) {
        // Count how many elements to keep
        const SizeType count =
            std::accumulate(keep_mask.begin(), keep_mask.end(), SizeType{0});
        if (count == m_size) {
            return;
        }
        if (count == 0) {
            m_size = 0;
            if (m_is_updating) {
                m_write_head = m_write_start;
            }
            return;
        }
        const auto start_idx    = get_current_start_idx();
        const SizeType old_size = m_size;
        // Use logical indices to abstract circular wrapping
        SizeType phys_read  = start_idx;
        SizeType phys_write = start_idx;
        SizeType kept_count = 0;
        // Single-pass: scan and compact in-place
        for (SizeType read_logical = 0; read_logical < old_size;
             ++read_logical) {
            // Check filter condition
            if (keep_mask[read_logical] != 0U) {
                // Move data only if gaps were created
                if (phys_read != phys_write) {
                    std::copy_n(
                        m_leaves.begin() + (phys_read * m_leaves_stride),
                        m_leaves_stride,
                        m_leaves.begin() + (phys_write * m_leaves_stride));
                    std::copy_n(m_folds.begin() + (phys_read * m_folds_stride),
                                m_folds_stride,
                                m_folds.begin() +
                                    (phys_write * m_folds_stride));
                    m_scores[phys_write] = m_scores[phys_read];
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
        error_check::check_less_equal(m_size, m_capacity,
                                      "WorldTree: Invalid size after keep");
    }

    // Memory-efficient uniqueness detection
    // Tie-break: keep first occurrence when scores are equal.
    void compute_uniqueness_mask_in_scratch() {
        if (m_size == 0) {
            return;
        }
        std::fill_n(m_scratch_mask.begin(), m_size, uint8_t{0});
        std::unordered_map<int64_t, BestIndices> best_by_key;
        best_by_key.reserve(m_size);

        const auto start_idx = get_current_start_idx();
        for (SizeType i = 0; i < m_size; ++i) {
            const auto buffer_idx =
                get_circular_index(i, start_idx, m_capacity);
            const auto leaves_offset = buffer_idx * m_leaves_stride;
            const auto val1 = m_leaves[leaves_offset + ((m_nparams - 2) * 2)];
            const auto val2 = m_leaves[leaves_offset + ((m_nparams - 1) * 2)];
            const auto key =
                static_cast<int64_t>(std::nearbyint((val1 + val2) * 1e9));
            const auto score = m_scores[buffer_idx];

            auto it = best_by_key.find(key);
            if (it == best_by_key.end()) {
                best_by_key.emplace(key, BestIndices{score, i});
            } else if (score > it->second.score) {
                it->second.score   = score;
                it->second.win_idx = i;
            }
        }

        // Mark the winners
        for (const auto& kv : best_by_key) {
            m_scratch_mask[kv.second.win_idx] = 1;
        }
    }

    void validate_circular_buffer_state() {
        // Core invariant: total used space <= capacity
        const auto total_used = (m_size_old - m_read_consumed) + m_size;
        error_check::check_less_equal(total_used, m_capacity,
                                      "Circular buffer invariant violated: "
                                      "total_used > capacity");

        // State consistency checks
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

}; // End WorldTree::Impl definition

// Public interface implementation
template <SupportedFoldType FoldType>
WorldTree<FoldType>::WorldTree(SizeType capacity,
                               SizeType nparams,
                               SizeType nbins,
                               SizeType max_batch_size)
    : m_impl(std::make_unique<Impl>(capacity, nparams, nbins, max_batch_size)) {
}
template <SupportedFoldType FoldType>
WorldTree<FoldType>::~WorldTree() = default;
template <SupportedFoldType FoldType>
WorldTree<FoldType>::WorldTree(WorldTree&& other) noexcept = default;
template <SupportedFoldType FoldType>
WorldTree<FoldType>&
WorldTree<FoldType>::operator=(WorldTree<FoldType>&& other) noexcept = default;
// Getters
template <SupportedFoldType FoldType>
const std::vector<double>& WorldTree<FoldType>::get_leaves() const noexcept {
    return m_impl->get_leaves();
}
template <SupportedFoldType FoldType>
const std::vector<FoldType>& WorldTree<FoldType>::get_folds() const noexcept {
    return m_impl->get_folds();
}
template <SupportedFoldType FoldType>
const std::vector<float>& WorldTree<FoldType>::get_scores() const noexcept {
    return m_impl->get_scores();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_capacity() const noexcept {
    return m_impl->get_capacity();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_max_batch_size() const noexcept {
    return m_impl->get_max_batch_size();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_leaves_stride() const noexcept {
    return m_impl->get_leaves_stride();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_folds_stride() const noexcept {
    return m_impl->get_folds_stride();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_size() const noexcept {
    return m_impl->get_size();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_size_old() const noexcept {
    return m_impl->get_size_old();
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::get_size_lb() const noexcept {
    return m_impl->get_size_lb();
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::get_score_max() const noexcept {
    return m_impl->get_score_max();
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::get_score_min() const noexcept {
    return m_impl->get_score_min();
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
template <SupportedFoldType FoldType>
std::pair<std::span<const double>, SizeType>
WorldTree<FoldType>::get_leaves_span(SizeType n_leaves) const {
    return m_impl->get_leaves_span(n_leaves);
}
template <SupportedFoldType FoldType>
std::span<double> WorldTree<FoldType>::get_leaves_contiguous_span() noexcept {
    return m_impl->get_leaves_contiguous_span();
}
template <SupportedFoldType FoldType>
std::span<float> WorldTree<FoldType>::get_scores_contiguous_span() noexcept {
    return m_impl->get_scores_contiguous_span();
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::set_size(SizeType size) noexcept {
    m_impl->set_size(size);
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::reset() noexcept {
    m_impl->reset();
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::prepare_in_place_update() {
    m_impl->prepare_in_place_update();
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::finalize_in_place_update() {
    m_impl->finalize_in_place_update();
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::consume_read(SizeType n) {
    m_impl->consume_read(n);
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::compute_physical_indices(
    std::span<const SizeType> logical_indices,
    std::span<SizeType> physical_indices,
    SizeType n_leaves) const {
    m_impl->compute_physical_indices(logical_indices, physical_indices,
                                     n_leaves);
}
// Other methods
template <SupportedFoldType FoldType>
std::tuple<std::span<const double>, std::span<const FoldType>, float>
WorldTree<FoldType>::get_best() const {
    return m_impl->get_best();
}
template <SupportedFoldType FoldType>
bool WorldTree<FoldType>::add(std::span<const double> leaf,
                              std::span<const FoldType> fold,
                              float score) {
    return m_impl->add(leaf, fold, score);
}
template <SupportedFoldType FoldType>
void WorldTree<FoldType>::add_initial(std::span<const double> batch_leaves,
                                      std::span<const FoldType> batch_folds,
                                      std::span<const float> batch_scores,
                                      SizeType slots_to_write) {
    m_impl->add_initial(batch_leaves, batch_folds, batch_scores,
                        slots_to_write);
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::add_batch(std::span<const double> batch_leaves,
                                     std::span<const FoldType> batch_folds,
                                     std::span<const float> batch_scores,
                                     float current_threshold,
                                     SizeType slots_to_write) {
    return m_impl->add_batch(batch_leaves, batch_folds, batch_scores,
                             current_threshold, slots_to_write);
}
template <SupportedFoldType FoldType> void WorldTree<FoldType>::deduplicate() {
    m_impl->deduplicate();
}
// Explicit instantiation
template class WorldTree<float>;
template class WorldTree<ComplexType>;
} // namespace loki::utils