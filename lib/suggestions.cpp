#include "loki/utils/suggestions.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/transforms.hpp"

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
         std::string_view mode)
        : m_capacity(capacity),
          m_nparams(nparams),
          m_nbins(nbins),
          m_mode(mode),
          m_leaves_stride((nparams + 2) * kParamStride),
          m_folds_stride(2 * nbins),
          m_leaves(m_capacity * m_leaves_stride, 0.0),
          m_folds(m_capacity * m_folds_stride, FoldType{}),
          m_scores(m_capacity, 0.0F),
          m_scratch_scores(m_capacity, 0.0F),
          m_scratch_pending_indices(m_capacity, 0),
          m_scratch_mask(m_capacity, 0) {
        // Validate inputs
        error_check::check_greater(
            m_capacity, 0, "WorldTree: capacity must be greater than 0");
        error_check::check_greater(m_nparams, 0,
                                   "WorldTree: nparams must be greater than 0");
        error_check::check_greater(m_nbins, 0,
                                   "WorldTree: nbins must be greater than 0");
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

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

    const std::vector<double>& get_leaves() const noexcept { return m_leaves; }

    /**
     * @brief Get span over leaves for processing
     *
     * During updates, returns span over readable (old) region.
     * The span may be truncated at buffer wrap point.
     *
     * @param n_leaves Number of leaves to access
     * @return Pair of (span, actual_contiguous_count)
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

    const std::vector<FoldType>& get_folds() const noexcept { return m_folds; }

    /**
     * @brief Get span over folds for kernel consumption
     *
     * @param n_folds Number of fold entries to access
     * @return Pair of (span, actual_contiguous_count)
     */
    std::pair<std::span<const FoldType>, SizeType>
    get_folds_span(SizeType n_folds) const {
        const SizeType available = m_size_old - m_read_consumed;
        error_check::check_less_equal(
            n_folds, available,
            "get_folds_span: requested range exceeds available data");

        const auto physical_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);
        const auto actual_size = std::min(n_folds, m_capacity - physical_start);
        return {{m_folds.data() + (physical_start * m_folds_stride),
                 actual_size * m_folds_stride},
                actual_size};
    }

    /**
     * @brief Get copy of current scores
     *
     * Returns scores for m_size elements in the current region.
     *
     * @return Vector of scores
     */
    std::vector<float> get_scores() const {
        std::vector<float> scores(m_size);
        if (m_size == 0) {
            return scores;
        }
        const auto start_idx = get_current_start_idx();
        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single fast copy
            std::copy_n(m_scores.begin() + start_idx, m_size, scores.begin());
        } else {
            // Wrapped case - two contiguous copies
            const auto first_part = m_capacity - start_idx;
            std::copy_n(m_scores.begin() + start_idx, first_part,
                        scores.begin());
            std::copy_n(m_scores.begin(), m_size - first_part,
                        scores.begin() + first_part);
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
    float get_score_min() const {
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
     * @brief Get median score in current region
     *
     * Uses nth_element for O(n) complexity.
     * For even-sized arrays, returns lower median.
     */
    float get_score_median() {
        if (m_size == 0) {
            return 0.0F;
        }
        copy_scores_to_scratch();
        const auto mid = m_scratch_scores.begin() + m_size / 2;
        std::ranges::nth_element(m_scratch_scores, mid);
        return *mid;
    }

    /**
     * @brief Estimate memory usage in GiB
     *
     * Includes both base storage and estimated peak temporary allocations.
     */
    float get_memory_usage() const noexcept {
        const auto base_bytes = (m_leaves.size() * sizeof(double)) +
                                (m_folds.size() * sizeof(FoldType)) +
                                (2 * m_scores.size() * sizeof(float)) +
                                (m_capacity * sizeof(uint32_t)) +
                                (m_capacity * sizeof(uint8_t));
        // Peak temporary allocations (worst case scenario)
        SizeType peak_temp_bytes = 0;

        // 2. get_transformed() temp allocation
        const auto transform_bytes =
            m_capacity * m_leaves_stride * sizeof(double);
        peak_temp_bytes = std::max(peak_temp_bytes, transform_bytes);
        // 3. trim operations temp allocations
        const auto trim_temp_bytes =
            // idx array in trim_half
            (m_capacity * sizeof(SizeType)) +
            // moves vector in keep (worst case: all elements move)
            (m_capacity * sizeof(std::pair<SizeType, SizeType>)) +
            // best_by_key map in compute_uniqueness_mask_inplace (worst case:
            // all unique)
            (m_capacity * (sizeof(int64_t) + sizeof(BestIndices))) +
            // keep_mask
            (m_capacity * sizeof(bool));

        peak_temp_bytes = std::max(peak_temp_bytes, trim_temp_bytes);

        return static_cast<float>(base_bytes + peak_temp_bytes) /
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
            m_size, m_capacity, "WorldTree: Invalid size after set_size()");
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

        // logical_indices are relative to current batch span. The span starts
        // at this physical location:
        const auto span_start =
            get_circular_index(m_read_consumed, m_head, m_capacity);
        for (SizeType i = 0; i < n_leaves; ++i) {
            physical_indices[i] =
                get_circular_index(logical_indices[i], span_start, m_capacity);
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

    std::vector<double>
    get_transformed(std::pair<double, double> /*coord_mid*/) const {
        // Copy all leaves rows
        std::vector<double> contig_leaves(m_size * m_leaves_stride, 0.0);
        const auto start_idx = get_current_start_idx();

        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single copy with stride
            for (SizeType i = 0; i < m_size; ++i) {
                const auto src_offset = (start_idx + i) * m_leaves_stride;
                const auto dst_offset = i * m_leaves_stride;
                std::copy_n(m_leaves.begin() + src_offset, m_leaves_stride,
                            contig_leaves.begin() + dst_offset);
            }
        } else {
            // Wrapped case
            const auto first_part = m_capacity - start_idx;
            for (SizeType i = 0; i < first_part; ++i) {
                const auto src_offset = (start_idx + i) * m_leaves_stride;
                const auto dst_offset = i * m_leaves_stride;
                std::copy_n(m_leaves.begin() + src_offset, m_leaves_stride,
                            contig_leaves.begin() + dst_offset);
            }
            for (SizeType i = 0; i < m_size - first_part; ++i) {
                const auto src_offset = i * m_leaves_stride;
                const auto dst_offset = (first_part + i) * m_leaves_stride;
                std::copy_n(m_leaves.begin() + src_offset, m_leaves_stride,
                            contig_leaves.begin() + dst_offset);
            }
        }
        // Transform in-place
        if (m_mode == "taylor") {
            transforms::report_leaves_taylor_batch(contig_leaves, m_size,
                                                   m_nparams);
        } else if (m_mode == "chebyshev") {
            throw std::runtime_error("Chebyshev mode not implemented.");
        } else {
            throw std::runtime_error(std::format(
                "Suggestion struct mode must be taylor or chebyshev."));
        }
        // Report only param_sets rows
        const auto param_sets_stride = m_nparams * kParamStride;
        std::vector<double> contig_param_sets(m_size * param_sets_stride, 0.0);
        for (SizeType i = 0; i < m_size; ++i) {
            const auto src_offset = i * m_leaves_stride;
            const auto dst_offset = i * param_sets_stride;
            std::copy_n(contig_leaves.begin() + src_offset, param_sets_stride,
                        contig_param_sets.begin() + dst_offset);
        }
        return contig_param_sets;
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

    /**
     * @brief Add batch during update with threshold filtering
     *
     * @return Updated threshold after any trimming
     * Adds filtered batch to write region. If full, trims write region via
     * median threshold. Loops until all candidates fit, reclaiming space
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
        const IndexType space_left = calculate_space_left();
        if (static_cast<IndexType>(slots_to_write) <= space_left) {
            for (SizeType i = 0; i < slots_to_write; ++i) {
                const auto write_idx  = m_write_head;
                const auto leaves_src = i * m_leaves_stride;
                const auto leaves_dst = write_idx * m_leaves_stride;
                const auto folds_src  = i * m_folds_stride;
                const auto folds_dst  = write_idx * m_folds_stride;
                std::copy_n(batch_leaves.begin() + leaves_src, m_leaves_stride,
                            m_leaves.begin() + leaves_dst);
                std::copy_n(batch_folds.begin() + folds_src, m_folds_stride,
                            m_folds.begin() + folds_dst);
                m_scores[write_idx] = batch_scores[i];
                m_write_head++;
                if (m_write_head == m_capacity) {
                    m_write_head = 0;
                }
            }
            m_size += slots_to_write;
            return current_threshold;
        }

        // Slow path: Overflow & Pruning
        auto effective_threshold = current_threshold;

        // Create candidate indices
        auto pending_indices_span =
            std::span(m_scratch_pending_indices).first(slots_to_write);
        for (SizeType i = 0; i < slots_to_write; ++i) {
            pending_indices_span[i] = i;
        }
        SizeType pending_count  = slots_to_write;
        SizeType pending_offset = 0;

        auto update_candidates = [&]() {
            pending_count = 0;
            for (SizeType i = 0; i < slots_to_write; ++i) {
                if (batch_scores[i] >= effective_threshold) {
                    pending_indices_span[pending_count++] = i;
                }
            }
        };

        while (pending_offset < pending_count) {
            // Using IndexType to safely handle potential negative results
            const IndexType space_left = calculate_space_left();

            if (space_left < 0 ||
                space_left > static_cast<IndexType>(m_capacity)) {
                throw std::runtime_error(
                    std::format("WorldTree: Invalid space left ({}) after "
                                "add_batch. Buffer overflow.",
                                space_left));
            }
            if (space_left == 0) {
                // Buffer is full, try to prune the newly added candidates.
                const auto new_threshold = prune_on_overload();
                effective_threshold =
                    std::max(effective_threshold, new_threshold);
                // Re-filter after new threshold
                update_candidates();
                pending_offset = 0;
                continue; // Try again with new threshold
            }

            const auto remaining = pending_count - pending_offset;
            const auto n_to_add =
                std::min(remaining, static_cast<SizeType>(space_left));

            // Batched copy
            for (SizeType i = 0; i < n_to_add; ++i) {
                const auto src_idx = pending_indices_span[pending_offset + i];
                const auto dst_idx = m_write_head;
                const auto leaves_src = src_idx * m_leaves_stride;
                const auto leaves_dst = dst_idx * m_leaves_stride;
                const auto folds_src  = src_idx * m_folds_stride;
                const auto folds_dst  = dst_idx * m_folds_stride;
                std::copy_n(batch_leaves.begin() + leaves_src, m_leaves_stride,
                            m_leaves.begin() + leaves_dst);
                std::copy_n(batch_folds.begin() + folds_src, m_folds_stride,
                            m_folds.begin() + folds_dst);
                m_scores[dst_idx] = batch_scores[src_idx];
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

        // Create a boolean mask for scores >= threshold
        std::vector<bool> keep_mask(m_size);
        const auto regions = get_active_regions(m_scores);
        // Fill first part of mask
        for (size_t i = 0; i < regions.first.size(); ++i) {
            keep_mask[i] = regions.first[i] >= threshold;
        }
        // Fill second part (if exists)
        size_t offset = regions.first.size();
        for (size_t i = 0; i < regions.second.size(); ++i) {
            keep_mask[offset + i] = regions.second[i] >= threshold;
        }
        keep(keep_mask);
        return threshold;
    }

    float prune_half() {
        if (m_size == 0) {
            return 0.0F;
        }

        // To guarantee progress, we will keep at most half of the
        // candidates.
        const SizeType n_to_keep = m_size / 2;
        // Find the score of the (n_to_keep)-th best candidate. This will
        // be our new threshold.
        copy_scores_to_scratch();
        // We want the score that keeps top n_to_keep.
        // nth_element puts the n_to_keep-th largest element at the 'nth'
        // position. We use greater<float> to sort descending.
        auto nth_iter = m_scratch_scores.begin() + n_to_keep;
        std::nth_element(m_scratch_scores.begin(), nth_iter,
                         m_scratch_scores.end(), std::greater<float>());
        const float threshold = *nth_iter;

        // Build mask (Split loop to avoid modulo)
        std::vector<bool> keep_mask(m_size, false);
        SizeType kept_count = 0;

        // Helper to process a region
        auto process_region = [&](std::span<const float> score_span,
                                  SizeType mask_offset) {
            for (size_t i = 0; i < score_span.size(); ++i) {
                if (score_span[i] > threshold) {
                    keep_mask[mask_offset + i] = true;
                    kept_count++;
                }
            }
        };

        const auto regions = get_active_regions(m_scores);
        process_region(regions.first, 0);
        if (!regions.second.empty()) {
            process_region(regions.second, regions.first.size());
        }

        // Handle tie-breaking (fill up to n_to_keep with items == threshold)
        if (kept_count < n_to_keep) {
            auto fill_region = [&](std::span<const float> score_span,
                                   SizeType mask_offset) {
                for (size_t i = 0;
                     i < score_span.size() && kept_count < n_to_keep; ++i) {
                    if (!keep_mask[mask_offset + i] &&
                        score_span[i] == threshold) {
                        keep_mask[mask_offset + i] = true;
                        kept_count++;
                    }
                }
            };

            fill_region(regions.first, 0);
            if (!regions.second.empty()) {
                fill_region(regions.second, regions.first.size());
            }
        }

        keep(keep_mask);
        return threshold;
    }

    void deduplicate() {
        if (!m_is_updating) {
            throw std::runtime_error(
                "deduplicate: only allowed during updates");
        }
        if (m_size == 0) {
            return;
        }
        const auto keep_mask = compute_uniqueness_mask_inplace();
        keep(keep_mask);
    }

    float deduplicate_and_prune_on_overload() {
        if (!m_is_updating) {
            throw std::runtime_error(
                "trim_repeats_threshold: only allowed during updates");
        }
        if (m_size == 0) {
            return 0.0F;
        }
        auto keep_mask       = compute_uniqueness_mask_inplace();
        const auto threshold = get_score_median();
        const auto regions   = get_active_regions(m_scores);

        auto filter_region = [&](std::span<const float> scores,
                                 SizeType mask_offset) {
            for (size_t i = 0; i < scores.size(); ++i) {
                if (scores[i] < threshold) {
                    keep_mask[mask_offset + i] = false;
                }
            }
        };

        // Update keep_mask to keep only scores >= threshold
        filter_region(regions.first, 0);
        if (!regions.second.empty()) {
            filter_region(regions.second, regions.first.size());
        }
        keep(keep_mask);
        return threshold;
    }

private:
    constexpr static SizeType kParamStride = 2;
    SizeType m_capacity;
    SizeType m_nparams;
    SizeType m_nbins;
    std::string m_mode;
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
    std::vector<float> m_scratch_scores;
    std::vector<uint32_t> m_scratch_pending_indices;
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
     */
    IndexType calculate_space_left() const noexcept {
        return static_cast<IndexType>(m_capacity) -
               ((static_cast<IndexType>(m_size_old) -
                 static_cast<IndexType>(m_read_consumed)) +
                static_cast<IndexType>(m_size));
    }

    /**
     * @brief Copy current scores to scratch buffer
     *
     * Copies scores for m_size elements in the current region to scratch
     * buffer.
     */
    void copy_scores_to_scratch() {
        if (m_size == 0) {
            return;
        }
        const auto start_idx = get_current_start_idx();
        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single fast copy
            std::copy_n(m_scores.begin() + start_idx, m_size,
                        m_scratch_scores.begin());
        } else {
            // Wrapped case - two contiguous copies
            const auto first_part = m_capacity - start_idx;
            std::copy_n(m_scores.begin() + start_idx, first_part,
                        m_scratch_scores.begin());
            std::copy_n(m_scores.begin(), m_size - first_part,
                        m_scratch_scores.begin() + first_part);
        }
    }

    /**
     * @brief Compact current region keeping only marked elements
     * using boolean mask. Only affects [start_idx, start_idx + m_size);
     * updates m_size.
     */
    void keep(const std::vector<bool>& keep_mask) {
        // Count how many elements to keep
        const SizeType count =
            std::count(keep_mask.begin(), keep_mask.end(), true);
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
        const auto start_idx = get_current_start_idx();
        // Collect all moves first, then do bulk operations
        std::vector<std::pair<SizeType, SizeType>> moves;
        moves.reserve(count);

        SizeType write_idx = start_idx;
        // Lambda to process logic over a contiguous range of physical indices
        // associated with a contiguous range of logical indices (mask)
        auto plan_moves = [&](SizeType phys_start, SizeType n_items,
                              SizeType mask_offset) {
            for (SizeType i = 0; i < n_items; ++i) {
                if (keep_mask[mask_offset + i]) {
                    const SizeType src_idx = phys_start + i;
                    if (write_idx != src_idx) {
                        moves.emplace_back(src_idx, write_idx);
                    }
                    write_idx++;
                    if (write_idx == m_capacity) {
                        write_idx = 0;
                    }
                }
            }
        };

        // Split Loop
        if (start_idx + m_size <= m_capacity) {
            plan_moves(start_idx, m_size, 0);
        } else {
            const auto first_len = m_capacity - start_idx;
            plan_moves(start_idx, first_len, 0);
            plan_moves(0, m_size - first_len, first_len);
        }

        // Execute moves
        for (const auto& [src, dst] : moves) {
            std::copy_n(m_leaves.begin() + (src * m_leaves_stride),
                        m_leaves_stride,
                        m_leaves.begin() + (dst * m_leaves_stride));
            std::copy_n(m_folds.begin() + (src * m_folds_stride),
                        m_folds_stride,
                        m_folds.begin() + (dst * m_folds_stride));
            m_scores[dst] = m_scores[src];
        }
        // Update valid size
        m_size = count;
        // After trimming, the write head must be updated to the new end
        if (m_is_updating) {
            m_write_head =
                get_circular_index(m_size, m_write_start, m_capacity);
        }
        error_check::check_less_equal(m_size, m_capacity,
                                      "WorldTree: Invalid size after keep");
    }

    // Memory-efficient uniqueness detection
    // Tie-break: keep first occurrence when scores are equal.
    std::vector<bool> compute_uniqueness_mask_inplace() {
        std::vector<bool> keep_mask(m_size, false);
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
            keep_mask[kv.second.win_idx] = true;
        }
        return keep_mask;
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
                               std::string_view mode)
    : m_impl(std::make_unique<Impl>(capacity, nparams, nbins, mode)) {}
template <SupportedFoldType FoldType>
WorldTree<FoldType>::~WorldTree() = default;
template <SupportedFoldType FoldType>
WorldTree<FoldType>::WorldTree(WorldTree&& other) noexcept = default;
template <SupportedFoldType FoldType>
WorldTree<FoldType>&
WorldTree<FoldType>::operator=(WorldTree<FoldType>&& other) noexcept = default;
// Getters
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_capacity() const noexcept {
    return m_impl->get_capacity();
}
template <SupportedFoldType FoldType>
size_t WorldTree<FoldType>::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
template <SupportedFoldType FoldType>
SizeType WorldTree<FoldType>::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
template <SupportedFoldType FoldType>
std::string_view WorldTree<FoldType>::get_mode() const noexcept {
    return m_impl->get_mode();
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
const std::vector<double>& WorldTree<FoldType>::get_leaves() const noexcept {
    return m_impl->get_leaves();
}
template <SupportedFoldType FoldType>
std::pair<std::span<const double>, SizeType>
WorldTree<FoldType>::get_leaves_span(SizeType n_leaves) const {
    return m_impl->get_leaves_span(n_leaves);
}
template <SupportedFoldType FoldType>
const std::vector<FoldType>& WorldTree<FoldType>::get_folds() const noexcept {
    return m_impl->get_folds();
}
template <SupportedFoldType FoldType>
std::vector<float> WorldTree<FoldType>::get_scores() const noexcept {
    return m_impl->get_scores();
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
float WorldTree<FoldType>::get_score_median() const noexcept {
    return m_impl->get_score_median();
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
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
std::vector<double> WorldTree<FoldType>::get_transformed(
    std::pair<double, double> coord_mid) const {
    return m_impl->get_transformed(coord_mid);
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
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::prune_on_overload() {
    return m_impl->prune_on_overload();
}
template <SupportedFoldType FoldType> void WorldTree<FoldType>::deduplicate() {
    m_impl->deduplicate();
}
template <SupportedFoldType FoldType>
float WorldTree<FoldType>::deduplicate_and_prune_on_overload() {
    return m_impl->deduplicate_and_prune_on_overload();
}

// Explicit instantiation
template class WorldTree<float>;
template class WorldTree<ComplexType>;
} // namespace loki::utils