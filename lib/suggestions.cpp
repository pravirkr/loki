#include "loki/utils/suggestions.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
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

1. prepare_for_in_place_update()
      – freezes READ, opens empty WRITE.
2. Consumers call advance_read_consumed() as they read.
3. Producers call add()/add_batch() into WRITE.
4. If WRITE fills, trim_*() may delete from WRITE only.
5. finalize_in_place_update()
      – promotes WRITE to READ.
6. Buffer is circular and not defragmented. It starts at m_head and wraps
   around to m_write_start.

All operations keep the invariant
   m_size_old - m_read_consumed + m_size ≤ m_capacity
- Read region: Old suggestions (m_head to m_head + m_size_old -
m_read_consumed).
- Write region: New suggestions (m_write_start to m_write_head).
*/
template <typename FoldType> class SuggestionTree<FoldType>::Impl {
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
          m_scores(m_capacity, 0.0F) {
        m_folds = std::vector<FoldType>(m_capacity * m_folds_stride, FoldType{});
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    const std::vector<double>& get_leaves() const noexcept { return m_leaves; }
    std::pair<std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const {
        // During updates, we read from the old region starting at m_head
        const SizeType available = m_size_old - m_read_consumed;
        error_check::check_less_equal(
            n_leaves, available,
            "get_leaves_span: requested range exceeds available data");

        // Compute physical start: relative to current head of read region
        const SizeType physical_start = (m_head + m_read_consumed) % m_capacity;

        // Compute how many contiguous elements we can take before wrap
        const auto dist_to_end = m_capacity - physical_start;
        const auto actual_size = std::min(n_leaves, dist_to_end);

        // Return span with correct byte size (actual_size elements, not bytes)
        return {{m_leaves.data() + (physical_start * m_leaves_stride),
                 actual_size * m_leaves_stride},
                actual_size};
    }

    const std::vector<FoldType>& get_folds() const noexcept { return m_folds; }
    std::vector<float> get_scores() const {
        std::vector<float> scores(m_size);
        if (m_size == 0) {
            return scores;
        }
        const auto start_idx = get_current_start_idx();
        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single fast copy
            std::copy(m_scores.begin() + start_idx,
                      m_scores.begin() + start_idx + m_size, scores.begin());
        } else {
            // Wrapped case - two contiguous copies
            const auto first_part = m_capacity - start_idx;
            std::copy(m_scores.begin() + start_idx, m_scores.end(),
                      scores.begin());
            std::copy(m_scores.begin(),
                      m_scores.begin() + (m_size - first_part),
                      scores.begin() + first_part);
        }
        return scores;
    }
    SizeType get_capacity() const noexcept { return m_capacity; }
    SizeType get_nparams() const noexcept { return m_nparams; }
    SizeType get_nbins() const noexcept { return m_nbins; }
    std::string_view get_mode() const noexcept { return m_mode; }
    SizeType get_leaves_stride() const noexcept { return m_leaves_stride; }
    SizeType get_folds_stride() const noexcept { return m_folds_stride; }
    SizeType get_nsugg() const noexcept { return m_size; }
    SizeType get_nsugg_old() const noexcept { return m_size_old; }
    float get_nsugg_lb() const noexcept {
        return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
    }
    float get_score_max() const {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto start_idx = get_current_start_idx();
        float max_score      = std::numeric_limits<float>::lowest();
        for (SizeType i = 0; i < m_size; ++i) {
            max_score =
                std::max(max_score, m_scores[(start_idx + i) % m_capacity]);
        }
        return max_score;
    }

    float get_score_min() const {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto start_idx = get_current_start_idx();
        float min_score      = std::numeric_limits<float>::max();
        for (SizeType i = 0; i < m_size; ++i) {
            min_score =
                std::min(min_score, m_scores[(start_idx + i) % m_capacity]);
        }
        return min_score;
    }

    float get_score_median() const {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto start_idx = get_current_start_idx();
        std::vector<SizeType> idx(m_size);
        std::iota(idx.begin(), idx.end(), 0);
        auto score_at = [&](SizeType i) {
            return m_scores[(start_idx + i) % m_capacity];
        };
        const auto mid = idx.begin() + m_size / 2;
        std::nth_element(
            idx.begin(), mid, idx.end(),
            [&](SizeType a, SizeType b) { return score_at(a) < score_at(b); });

        return score_at(*mid);
    }

    float get_memory_usage() const noexcept {
        const auto total_base_bytes = (m_leaves.size() * sizeof(double)) +
                                      (m_folds.size() * sizeof(FoldType)) +
                                      (m_scores.size() * sizeof(float));
        // Peak temporary allocations (worst case scenario)
        SizeType peak_temp_bytes = 0;

        // 1. get_scores() temp allocation
        peak_temp_bytes = std::max(peak_temp_bytes, m_capacity * sizeof(float));
        // 2. get_transformed() temp allocation
        const auto transform_bytes =
            m_capacity * m_leaves_stride * sizeof(double);
        peak_temp_bytes = std::max(peak_temp_bytes, transform_bytes);
        // 3. trim operations temp allocations
        const auto trim_temp_bytes =
            // scores_copy in get_score_median
            (m_capacity * sizeof(float)) +
            // idx array in trim_half
            (m_capacity * sizeof(SizeType)) +
            // moves vector in keep (worst case: all elements move)
            (m_capacity * sizeof(std::pair<SizeType, SizeType>)) +
            // best_by_key map in compute_uniqueness_mask_inplace (worst case:
            // all unique)
            (m_capacity * (sizeof(int64_t) + sizeof(BestIndices))) +
            // keep_mask
            (m_capacity * sizeof(bool)) +
            // pending_indices in add_batch
            (m_capacity * sizeof(SizeType));

        peak_temp_bytes = std::max(peak_temp_bytes, trim_temp_bytes);

        return static_cast<float>(total_base_bytes + peak_temp_bytes) /
               static_cast<float>(1ULL << 30U);
    }

    void set_nsugg(SizeType nsugg) noexcept {
        m_size     = nsugg;
        m_head     = 0;
        m_size_old = 0;
        error_check::check_less_equal(m_size, m_capacity,
                                      "SuggestionTree: Invalid size after "
                                      "set_nsugg. Buffer overflow.");
    }
    void reset() noexcept {
        m_size          = 0;
        m_head          = 0;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;
    }

    void prepare_for_in_place_update() {
        error_check::check_equal(m_is_updating, false,
                                 "Cannot prepare for update while already "
                                 "updating");
        m_size_old      = m_size;
        m_write_start   = (m_head + m_size) % m_capacity;
        m_write_head    = m_write_start;
        m_size          = 0; // The new size starts at 0
        m_is_updating   = true;
        m_read_consumed = 0;
        validate_circular_buffer_state();
    }

    void finalize_in_place_update() {
        error_check::check_equal(
            m_read_consumed, m_size_old,
            "finalize_in_place_update: not all old data consumed");
        m_head          = m_write_start;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;
    }

    void advance_read_consumed(SizeType n) {
        error_check::check_less_equal(m_read_consumed + n, m_size_old,
                                      "SuggestionTree: read_consumed overflow");
        m_read_consumed += n;

        // Validate circular buffer invariant
        error_check::check_less_equal(
            m_size_old - m_read_consumed + m_size, m_capacity,
            "SuggestionTree: circular buffer invariant violated");
    }

    void compute_physical_indices(std::span<const SizeType> logical_indices,
                                  std::span<SizeType> physical_indices,
                                  SizeType n_leaves) const {
        if (!m_is_updating) {
            throw std::runtime_error(
                "compute_physical_indices: only valid during updates");
        }
        error_check::check_greater_equal(
            logical_indices.size(), n_leaves,
            "compute_physical_indices: logical_indices "
            "size must be greater than or equal to n_leaves");
        error_check::check_greater_equal(physical_indices.size(), n_leaves,
                                         "compute_physical_indices: "
                                         "physical_indices size must be "
                                         "greater than or equal to n_leaves");

        // logical_indices are relative to current batch span. The span starts
        // at this physical location:
        const auto span_start_physical =
            (m_head + m_read_consumed) % m_capacity;
        for (SizeType i = 0; i < n_leaves; ++i) {
            const auto span_relative_idx = logical_indices[i];
            physical_indices[i] =
                (span_start_physical + span_relative_idx) % m_capacity;
        }
    }

    std::tuple<std::span<const double>, std::span<const FoldType>, float>
    get_best() const {
        if (m_size == 0) {
            return {std::span<const double>{}, std::span<const FoldType>{},
                    0.0F};
        }
        const auto start_idx = get_current_start_idx();
        float max_score      = std::numeric_limits<float>::lowest();
        SizeType idx_max_rel = 0;
        for (SizeType i = 0; i < m_size; ++i) {
            const auto current_idx = (start_idx + i) % m_capacity;
            if (m_scores[current_idx] > max_score) {
                max_score   = m_scores[current_idx];
                idx_max_rel = i;
            }
        }
        const auto idx_max_abs = (start_idx + idx_max_rel) % m_capacity;

        // Extract the best parameter set and fold using views
        return {std::span{m_leaves.data() + (idx_max_abs * m_leaves_stride),
                          m_leaves_stride},
                std::span{m_folds.data() + (idx_max_abs * m_folds_stride),
                          m_folds_stride},
                m_scores[idx_max_abs]};
    }

    std::vector<double>
    get_transformed(std::pair<double, double> coord_mid) const {
        // Copy all leaves rows
        std::vector<double> contig_leaves(m_size * m_leaves_stride, 0.0);
        const auto start_idx = get_current_start_idx();

        // Check if data wraps around
        if (start_idx + m_size <= m_capacity) {
            // Contiguous case - single copy with stride
            for (SizeType i = 0; i < m_size; ++i) {
                const auto src_offset = (start_idx + i) * m_leaves_stride;
                const auto dst_offset = i * m_leaves_stride;
                std::copy(m_leaves.begin() + src_offset,
                          m_leaves.begin() + src_offset + m_leaves_stride,
                          contig_leaves.begin() + dst_offset);
            }
        } else {
            // Wrapped case
            const auto first_part = m_capacity - start_idx;
            for (SizeType i = 0; i < first_part; ++i) {
                const auto src_offset = (start_idx + i) * m_leaves_stride;
                const auto dst_offset = i * m_leaves_stride;
                std::copy(m_leaves.begin() + src_offset,
                          m_leaves.begin() + src_offset + m_leaves_stride,
                          contig_leaves.begin() + dst_offset);
            }
            for (SizeType i = 0; i < m_size - first_part; ++i) {
                const auto src_offset = i * m_leaves_stride;
                const auto dst_offset = (first_part + i) * m_leaves_stride;
                std::copy(m_leaves.begin() + src_offset,
                          m_leaves.begin() + src_offset + m_leaves_stride,
                          contig_leaves.begin() + dst_offset);
            }
        }
        // Transform in-place
        if (m_mode == "taylor") {
            transforms::report_leaves_taylor_batch(contig_leaves, m_size,
                                                   m_nparams);
        } else if (m_mode == "chebyshev") {
            transforms::report_leaves_chebyshev_batch(contig_leaves, coord_mid,
                                                      m_size, m_nparams);
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
            std::copy(contig_leaves.begin() + src_offset,
                      contig_leaves.begin() + src_offset + param_sets_stride,
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
        m_write_head        = (m_write_head + 1) % m_capacity;
        ++m_size;
        return true;
    }

    void add_initial(std::span<const double> batch_leaves,
                     std::span<const FoldType> batch_folds,
                     std::span<const float> batch_scores,
                     SizeType slots_to_write) {
        error_check::check_less_equal(slots_to_write, m_capacity,
                                      "SuggestionTree: Suggestions too large "
                                      "to add.");
        error_check::check_equal(slots_to_write, batch_scores.size(),
                                 "slots_to_write must match batch_scores size");

        reset(); // Start fresh
        std::ranges::copy(batch_leaves, m_leaves.begin());
        std::ranges::copy(batch_folds, m_folds.begin());
        std::ranges::copy(batch_scores, m_scores.begin());
        m_size = slots_to_write;
        error_check::check_less_equal(m_size, m_capacity,
                                      "SuggestionTree: Invalid size after "
                                      "add_initial. Buffer overflow.");
    }

    // Adds filtered batch to write region. If full, trims write region via
    // median threshold. Loops until all candidates fit, reclaiming space
    // from consumed old suggestions.
    float add_batch(std::span<const double> batch_leaves,
                    std::span<const FoldType> batch_folds,
                    std::span<const float> batch_scores,
                    float current_threshold,
                    SizeType slots_to_write) {
        // Always use scores_batch to get the correct batch size
        if (slots_to_write == 0) {
            return current_threshold;
        }
        auto effective_threshold = current_threshold;

        // Create initial mask for scores >= threshold
        std::vector<SizeType> pending_indices;
        pending_indices.reserve(slots_to_write);

        auto update_candidates = [&]() {
            pending_indices.clear();
            for (SizeType i = 0; i < slots_to_write; ++i) {
                if (batch_scores[i] >= effective_threshold) {
                    pending_indices.push_back(i);
                }
            }
        };

        update_candidates();

        while (!pending_indices.empty()) {
            // Using IndexType to safely handle potential negative results
            const auto space_left = static_cast<IndexType>(m_capacity) -
                                    ((static_cast<IndexType>(m_size_old) -
                                      static_cast<IndexType>(m_read_consumed)) +
                                     static_cast<IndexType>(m_size));

            if (space_left < 0 ||
                space_left > static_cast<IndexType>(m_capacity)) {
                throw std::runtime_error(
                    std::format("SuggestionTree: Invalid space left ({}) after "
                                "add_batch. Buffer overflow.",
                                space_left));
            }
            if (space_left == 0) {
                // Buffer is full, try to trim the newly added suggestions.
                const auto new_threshold_from_trim = trim_threshold();
                effective_threshold =
                    std::max(effective_threshold, new_threshold_from_trim);
                // Re-filter after new threshold
                update_candidates();
                continue; // Try again with new threshold
            }

            const auto n_to_add_now = std::min(
                pending_indices.size(), static_cast<SizeType>(space_left));

            // Batched assignment
            for (SizeType i = 0; i < n_to_add_now; ++i) {
                const auto src_idx               = pending_indices[i];
                const auto dst_idx               = m_write_head;
                const SizeType leaves_src_offset = src_idx * m_leaves_stride;
                const SizeType leaves_dst_offset = dst_idx * m_leaves_stride;
                const SizeType folds_src_offset  = src_idx * m_folds_stride;
                const SizeType folds_dst_offset  = dst_idx * m_folds_stride;
                std::ranges::copy(
                    batch_leaves.subspan(leaves_src_offset, m_leaves_stride),
                    m_leaves.begin() + leaves_dst_offset);
                std::ranges::copy(
                    batch_folds.subspan(folds_src_offset, m_folds_stride),
                    m_folds.begin() + folds_dst_offset);
                m_scores[dst_idx] = batch_scores[src_idx];

                m_write_head = (m_write_head + 1) % m_capacity;
            }
            m_size += n_to_add_now;
            // Remove added candidates from the list
            pending_indices.erase(pending_indices.begin(),
                                  pending_indices.begin() +
                                      static_cast<IndexType>(n_to_add_now));
        }
        return effective_threshold;
    }

    float trim_threshold() {
        if (!m_is_updating) {
            throw std::runtime_error(
                "trim_threshold: only allowed during updates");
        }
        if (m_size == 0) {
            return 0.0F;
        }
        // Compute median score of the *newly added* suggestions.
        const float threshold = get_score_median();

        // Create a boolean mask for scores >= threshold
        std::vector<bool> indices(m_size, false);
        for (SizeType i = 0; i < m_size; ++i) {
            const auto current_idx = (m_write_start + i) % m_capacity;
            indices[i]             = m_scores[current_idx] >= threshold;
        }

        keep(indices);
        return threshold;
    }

    float trim_half() {
        if (m_size == 0) {
            return 0.0F;
        }

        // To guarantee progress, we will keep at most half of the
        // suggestions.
        const SizeType n_to_keep = m_size / 2;
        // Find the score of the (n_to_keep)-th best suggestion. This will
        // be our new threshold.
        const auto start_idx = get_current_start_idx();
        // Indices proxy instead of scores copy
        std::vector<SizeType> idx(m_size);
        std::iota(idx.begin(), idx.end(), 0);

        auto nth      = idx.begin() + n_to_keep;
        auto score_at = [&](SizeType i) {
            return m_scores[(start_idx + i) % m_capacity];
        };

        // Find the threshold score that would keep the top `n_to_keep`
        // elements. We sort in descending order to find the n_to_keep-th
        // largest element.
        std::nth_element(
            idx.begin(), nth, idx.end(),
            [&](SizeType a, SizeType b) { return score_at(a) > score_at(b); });

        const float threshold = score_at(*nth);

        // Build mask: one pass, keep > threshold; fill == threshold until
        // quota.
        std::vector<bool> keep_mask(m_size, false);
        SizeType kept_count = 0;
        for (SizeType i = 0; i < m_size; ++i) {
            if (score_at(i) > threshold) {
                keep_mask[i] = true;
                ++kept_count;
            }
        }
        for (SizeType i = 0; kept_count < n_to_keep && i < m_size; ++i) {
            if (!keep_mask[i] && score_at(i) == threshold) {
                keep_mask[i] = true;
                ++kept_count;
            }
        }

        keep(keep_mask);
        return threshold;
    }

    void trim_repeats() {
        if (!m_is_updating) {
            throw std::runtime_error(
                "trim_repeats: only allowed during updates");
        }
        if (m_size == 0) {
            return;
        }
        const auto keep_mask = compute_uniqueness_mask_inplace();
        keep(keep_mask);
    }

    float trim_repeats_threshold() {
        if (!m_is_updating) {
            throw std::runtime_error(
                "trim_repeats_threshold: only allowed during updates");
        }
        if (m_size == 0) {
            return 0.0F;
        }
        auto keep_mask       = compute_uniqueness_mask_inplace();
        const auto threshold = get_score_median();
        const auto start_idx = get_current_start_idx();
        // Update keep_mask to keep only scores >= threshold
        for (SizeType i = 0; i < m_size; ++i) {
            const auto buffer_idx = (start_idx + i) % m_capacity;
            if (m_scores[buffer_idx] < threshold) {
                keep_mask[i] = false;
            }
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

    SizeType get_current_start_idx() const noexcept {
        return m_is_updating ? m_write_start : m_head;
    }

    // Compacts the current region (write or full, based on m_is_updating)
    // using boolean mask. Only affects [start_idx, start_idx + m_size);
    // updates m_size.
    void keep(const std::vector<bool>& indices) {
        // Count how many elements to keep
        SizeType count = std::count(indices.begin(), indices.end(), true);
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
        for (SizeType read_idx_rel = 0; read_idx_rel < m_size; ++read_idx_rel) {
            if (indices[read_idx_rel]) {
                const auto src_idx = (start_idx + read_idx_rel) % m_capacity;
                if (write_idx != src_idx) {
                    moves.push_back({src_idx, write_idx});
                }
                write_idx = (write_idx + 1) % m_capacity;
            }
        }
        for (const auto& move : moves) {
            std::copy(m_leaves.begin() + (move.first * m_leaves_stride),
                      m_leaves.begin() + (move.first * m_leaves_stride) +
                          m_leaves_stride,
                      m_leaves.begin() + (move.second * m_leaves_stride));
            std::copy(m_folds.begin() + (move.first * m_folds_stride),
                      m_folds.begin() + (move.first * m_folds_stride) +
                          m_folds_stride,
                      m_folds.begin() + (move.second * m_folds_stride));
            m_scores[move.second] = m_scores[move.first];
        }
        // Update valid size
        m_size = count;
        // After trimming, the write head must be updated to the new end
        if (m_is_updating) {
            m_write_head = (m_write_start + m_size) % m_capacity;
        }
        error_check::check_less_equal(m_size, m_capacity,
                                      "SuggestionTree: Invalid size after "
                                      "keep. Buffer overflow.");
    }

    // Memory-efficient uniqueness detection
    // Tie-break: keep first occurrence when scores are equal.
    std::vector<bool> compute_uniqueness_mask_inplace() {
        std::vector<bool> keep_mask(m_size, false);
        std::unordered_map<int64_t, BestIndices> best_by_key;
        best_by_key.reserve(m_size);

        const auto start_idx = get_current_start_idx();
        for (SizeType i = 0; i < m_size; ++i) {
            const auto buffer_idx    = (start_idx + i) % m_capacity;
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

}; // End SuggestionTree::Impl definition

// Public interface implementation
template <typename FoldType>
SuggestionTree<FoldType>::SuggestionTree(SizeType capacity,
                                         SizeType nparams,
                                         SizeType nbins,
                                         std::string_view mode)
    : m_impl(std::make_unique<Impl>(capacity, nparams, nbins, mode)) {}
template <typename FoldType>
SuggestionTree<FoldType>::~SuggestionTree() = default;
template <typename FoldType>
SuggestionTree<FoldType>::SuggestionTree(SuggestionTree&& other) noexcept =
    default;
template <typename FoldType>
SuggestionTree<FoldType>& SuggestionTree<FoldType>::operator=(
    SuggestionTree<FoldType>&& other) noexcept = default;
// Getters
template <typename FoldType>
SizeType SuggestionTree<FoldType>::get_capacity() const noexcept {
    return m_impl->get_capacity();
}
template <typename FoldType>
size_t SuggestionTree<FoldType>::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
template <typename FoldType>
SizeType SuggestionTree<FoldType>::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
template <typename FoldType>
std::string_view SuggestionTree<FoldType>::get_mode() const noexcept {
    return m_impl->get_mode();
}
template <typename FoldType>
SizeType SuggestionTree<FoldType>::get_leaves_stride() const noexcept {
    return m_impl->get_leaves_stride();
}
template <typename FoldType>
SizeType SuggestionTree<FoldType>::get_folds_stride() const noexcept {
    return m_impl->get_folds_stride();
}
template <typename FoldType>
const std::vector<double>&
SuggestionTree<FoldType>::get_leaves() const noexcept {
    return m_impl->get_leaves();
}
template <typename FoldType>
std::pair<std::span<const double>, SizeType>
SuggestionTree<FoldType>::get_leaves_span(SizeType n_leaves) const {
    return m_impl->get_leaves_span(n_leaves);
}
template <typename FoldType>
const std::vector<FoldType>&
SuggestionTree<FoldType>::get_folds() const noexcept {
    return m_impl->get_folds();
}
template <typename FoldType>
std::vector<float> SuggestionTree<FoldType>::get_scores() const noexcept {
    return m_impl->get_scores();
}
template <typename FoldType>
SizeType SuggestionTree<FoldType>::get_nsugg() const noexcept {
    return m_impl->get_nsugg();
}
template <typename FoldType>
SizeType SuggestionTree<FoldType>::get_nsugg_old() const noexcept {
    return m_impl->get_nsugg_old();
}
template <typename FoldType>
float SuggestionTree<FoldType>::get_nsugg_lb() const noexcept {
    return m_impl->get_nsugg_lb();
}
template <typename FoldType>
float SuggestionTree<FoldType>::get_score_max() const noexcept {
    return m_impl->get_score_max();
}
template <typename FoldType>
float SuggestionTree<FoldType>::get_score_min() const noexcept {
    return m_impl->get_score_min();
}
template <typename FoldType>
float SuggestionTree<FoldType>::get_score_median() const noexcept {
    return m_impl->get_score_median();
}
template <typename FoldType>
float SuggestionTree<FoldType>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
template <typename FoldType>
void SuggestionTree<FoldType>::set_nsugg(SizeType nsugg) noexcept {
    m_impl->set_nsugg(nsugg);
}
template <typename FoldType> void SuggestionTree<FoldType>::reset() noexcept {
    m_impl->reset();
}
template <typename FoldType>
void SuggestionTree<FoldType>::prepare_for_in_place_update() {
    m_impl->prepare_for_in_place_update();
}
template <typename FoldType>
void SuggestionTree<FoldType>::finalize_in_place_update() {
    m_impl->finalize_in_place_update();
}
template <typename FoldType>
void SuggestionTree<FoldType>::advance_read_consumed(SizeType n) {
    m_impl->advance_read_consumed(n);
}
template <typename FoldType>
void SuggestionTree<FoldType>::compute_physical_indices(
    std::span<const SizeType> logical_indices,
    std::span<SizeType> physical_indices,
    SizeType n_leaves) const {
    m_impl->compute_physical_indices(logical_indices, physical_indices,
                                     n_leaves);
}
// Other methods
template <typename FoldType>
std::tuple<std::span<const double>, std::span<const FoldType>, float>
SuggestionTree<FoldType>::get_best() const {
    return m_impl->get_best();
}
template <typename FoldType>
std::vector<double> SuggestionTree<FoldType>::get_transformed(
    std::pair<double, double> coord_mid) const {
    return m_impl->get_transformed(coord_mid);
}
template <typename FoldType>
bool SuggestionTree<FoldType>::add(std::span<const double> leaf,
                                   std::span<const FoldType> fold,
                                   float score) {
    return m_impl->add(leaf, fold, score);
}
template <typename FoldType>
void SuggestionTree<FoldType>::add_initial(
    std::span<const double> batch_leaves,
    std::span<const FoldType> batch_folds,
    std::span<const float> batch_scores,
    SizeType slots_to_write) {
    m_impl->add_initial(batch_leaves, batch_folds, batch_scores,
                        slots_to_write);
}
template <typename FoldType>
float SuggestionTree<FoldType>::add_batch(std::span<const double> batch_leaves,
                                          std::span<const FoldType> batch_folds,
                                          std::span<const float> batch_scores,
                                          float current_threshold,
                                          SizeType slots_to_write) {
    return m_impl->add_batch(batch_leaves, batch_folds, batch_scores,
                             current_threshold, slots_to_write);
}
template <typename FoldType> float SuggestionTree<FoldType>::trim_threshold() {
    return m_impl->trim_threshold();
}
template <typename FoldType> void SuggestionTree<FoldType>::trim_repeats() {
    m_impl->trim_repeats();
}
template <typename FoldType>
float SuggestionTree<FoldType>::trim_repeats_threshold() {
    return m_impl->trim_repeats_threshold();
}

// Explicit instantiation
template class SuggestionTree<float>;
template class SuggestionTree<ComplexType>;
} // namespace loki::utils