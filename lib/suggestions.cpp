#include "loki/utils/suggestions.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <unordered_map>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/views/xview.hpp>

#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/psr_utils.hpp"

namespace loki::utils {

namespace {

std::vector<SizeType>
get_unique_indices_scores(const xt::xtensor<double, 3>& params,
                          std::span<const float> scores) {
    const SizeType nparams = params.shape()[0];
    std::vector<SizeType> unique_indices;
    unique_indices.reserve(nparams);

    std::unordered_map<int64_t, bool> unique_dict;
    std::unordered_map<int64_t, float> m_scoresdict;
    std::unordered_map<int64_t, SizeType> count_dict;

    unique_dict.reserve(nparams);
    m_scoresdict.reserve(nparams);
    count_dict.reserve(nparams);

    SizeType count = 0;
    for (SizeType i = 0; i < nparams; ++i) {
        if (params.shape()[1] < 2) {
            continue;
        }

        // Use the sum of the last two elements' first values as the key
        const double val1 = params(i, params.shape()[1] - 2, 0);
        const double val2 = params(i, params.shape()[1] - 1, 0);
        const auto key = static_cast<int64_t>(std::round((val1 + val2) * 1e9));

        auto it = unique_dict.find(key);
        if (it != unique_dict.end() && it->second) {
            // Found existing entry, update if current score is better
            if (scores[i] > m_scoresdict[key]) {
                m_scoresdict[key]   = scores[i];
                SizeType idx        = count_dict[key];
                unique_indices[idx] = i;
            }
        } else {
            // New unique entry
            unique_dict[key]  = true;
            m_scoresdict[key] = scores[i];
            count_dict[key]   = count;
            unique_indices.push_back(i);
            ++count;
        }
    }

    // Resize to actual count of unique entries
    unique_indices.resize(count);
    return unique_indices;
}
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
      – promotes WRITE to READ and defragments to
        make [0, m_size) contiguous for next round.

All operations keep the invariant
   m_size_old - m_read_consumed + m_size ≤ m_capacity
- Read region: Old suggestions (m_head to m_head + m_size_old -
m_read_consumed).
- Write region: New suggestions (m_write_start to m_write_head).
*/
template <typename FoldType> class SuggestionStruct<FoldType>::Impl {
public:
    Impl(SizeType capacity, SizeType nparams, SizeType nbins)
        : m_capacity(capacity),
          m_nparams(nparams),
          m_nbins(nbins),
          m_leaves_stride((nparams + 2) * 2),
          m_combined_res_stride(2 * nbins),
          m_backtrack_stride(nparams + 2) {
        m_param_sets =
            xt::xtensor<double, 3>({m_capacity, m_nparams + 2, 2}, 0.0);
        m_folds = xt::xtensor<FoldType, 3>({m_capacity, 2, m_nbins});

        // Initialize folds based on type
        if constexpr (std::is_same_v<FoldType, std::complex<float>>) {
            m_folds.fill(std::complex<float>(0.0F, 0.0F));
        } else {
            m_folds.fill(FoldType{});
        }

        m_scores     = std::vector<float>(m_capacity, 0.0F);
        m_backtracks = xt::xtensor<SizeType, 2>({m_capacity, m_nparams + 2}, 0);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = default;
    Impl& operator=(Impl&&)      = default;

    const xt::xtensor<double, 3>& get_param_sets() const noexcept {
        return m_param_sets;
    }
    const xt::xtensor<FoldType, 3>& get_folds() const noexcept {
        return m_folds;
    }
    std::vector<float> get_scores() const {
        std::vector<float> valid_scores;
        if (m_size == 0) {
            return valid_scores;
        }
        valid_scores.reserve(m_size);
        const auto start_idx = m_is_updating ? m_write_start : m_head;
        for (SizeType i = 0; i < m_size; ++i) {
            valid_scores.push_back(m_scores[(start_idx + i) % m_capacity]);
        }
        return valid_scores;
    }
    const xt::xtensor<SizeType, 2>& get_backtracks() const noexcept {
        return m_backtracks;
    }
    SizeType get_max_sugg() const noexcept { return m_capacity; }
    SizeType get_nparams() const noexcept { return m_nparams; }
    SizeType get_nbins() const noexcept { return m_nbins; }
    SizeType get_nsugg() const noexcept { return m_size; }
    SizeType get_nsugg_old() const noexcept { return m_size_old; }
    float get_nsugg_lb() const noexcept {
        return m_size > 0 ? std::log2(static_cast<float>(m_size)) : 0.0F;
    }
    float get_score_max() const {
        if (m_size == 0) {
            return 0.0F;
        }
        const auto start_idx = m_is_updating ? m_write_start : m_head;
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
        const auto start_idx = m_is_updating ? m_write_start : m_head;
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
        const auto start_idx = m_is_updating ? m_write_start : m_head;
        std::vector<float> scores_copy;
        scores_copy.reserve(m_size);
        for (SizeType i = 0; i < m_size; ++i) {
            scores_copy.push_back(m_scores[(start_idx + i) % m_capacity]);
        }
        const auto mid = scores_copy.begin() + m_size / 2;
        std::nth_element(scores_copy.begin(), mid, scores_copy.end());
        return *mid;
    }

    SizeType get_memory_usage() const noexcept {
        return (m_param_sets.size() * sizeof(double)) +
               (m_folds.size() * sizeof(FoldType)) +
               (m_scores.size() * sizeof(float)) +
               (m_backtracks.size() * sizeof(SizeType));
    }

    void set_nsugg(SizeType nsugg) noexcept {
        m_size     = nsugg;
        m_head     = 0;
        m_size_old = 0;
        error_check::check(
            m_size <= m_capacity,
            "SuggestionStruct: Invalid size after set_nsugg. Buffer overflow.");
    }
    void reset() noexcept {
        m_size          = 0;
        m_head          = 0;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;
    }

    void prepare_for_in_place_update() {
        m_size_old      = m_size;
        m_write_start   = (m_head + m_size) % m_capacity;
        m_write_head    = m_write_start;
        m_size          = 0; // The new size starts at 0
        m_is_updating   = true;
        m_read_consumed = 0;
    }

    void finalize_in_place_update() {
        m_head          = m_write_start;
        m_size_old      = 0;
        m_is_updating   = false;
        m_read_consumed = 0;

        // Always defragment to ensure data is contiguous and starts at 0
        // for the next iteration. This simplifies the read logic immensely.
        defragment();
    }

    void advance_read_consumed(SizeType n) {
        error_check::check(m_read_consumed + n <= m_size_old,
                           "SuggestionStruct: read_consumed overflow");
        m_read_consumed += n;
    }

    std::tuple<xt::xtensor<double, 2>, xt::xtensor<FoldType, 2>, float>
    get_best() const {
        if (m_size == 0) {
            // Return empty tensors and 0 score
            xt::xtensor<double, 2> empty_params({m_nparams + 2, 2}, 0.0);
            xt::xtensor<FoldType, 2> empty_folds({2, m_nbins});

            // Initialize empty_folds based on FoldType
            if constexpr (std::is_same_v<FoldType, std::complex<float>>) {
                empty_folds.fill(std::complex<float>(0.0F, 0.0F));
            } else {
                empty_folds.fill(FoldType{});
            }

            return std::make_tuple(empty_params, empty_folds, 0.0F);
        }

        // Find the index of the maximum score
        float max_score      = std::numeric_limits<float>::lowest();
        SizeType idx_max_rel = 0;
        for (SizeType i = 0; i < m_size; ++i) {
            if (m_scores[(m_head + i) % m_capacity] > max_score) {
                max_score   = m_scores[(m_head + i) % m_capacity];
                idx_max_rel = i;
            }
        }
        const auto idx_max_abs = (m_head + idx_max_rel) % m_capacity;
        // Extract the best parameter set and fold using views
        auto best_params =
            xt::view(m_param_sets, idx_max_abs, xt::all(), xt::all());
        auto best_folds = xt::view(m_folds, idx_max_abs, xt::all(), xt::all());
        // Return copies
        return std::make_tuple(best_params, best_folds, m_scores[idx_max_abs]);
    }

    xt::xtensor<double, 3> get_transformed(double delta_t) const {
        // Extract all parameter sets except the last two rows for each
        // suggestion using views
        // This method now needs to handle potentially non-contiguous data
        // For simplicity, we create a contiguous copy first.
        if (m_param_sets.shape()[1] <= 2) {
            return xt::xtensor<double, 3>({0, 0, 0});
        }
        xt::xtensor<double, 3> contig_params({m_size, m_nparams + 2, 2});
        const auto start_idx = m_is_updating ? m_write_start : m_head;
        for (SizeType i = 0; i < m_size; ++i) {
            const auto src_idx = (start_idx + i) % m_capacity;
            xt::view(contig_params, i, xt::all(), xt::all()) =
                xt::view(m_param_sets, src_idx, xt::all(), xt::all());
        }

        // Extract all but the last two parameters for each suggestion
        xt::xtensor<double, 3> transformed_params({m_size, m_nparams, 2});
        xt::view(transformed_params, xt::all(), xt::all(), xt::all()) =
            xt::view(contig_params, xt::all(), xt::range(0, m_nparams),
                     xt::all());

        // Call the batch shift function
        xt::xtensor<double, 3> trans_params;
        if (m_nparams < 4) {
            auto [shifted, _] =
                psr_utils::shift_params_batch(transformed_params, delta_t);
            trans_params = std::move(shifted);
        } else if (m_nparams == 4) {
            auto [shifted, _] = psr_utils::shift_params_circular_batch(
                transformed_params, delta_t);
            trans_params = std::move(shifted);
        } else {
            throw std::runtime_error(std::format(
                "Suggestion struct must have less than 4 parameters."));
        }
        return trans_params;
    }

    bool add(const xt::xtensor<double, 2>& param_set,
             const xt::xtensor<FoldType, 2>& fold,
             float score,
             const std::vector<SizeType>& backtrack) {
        if (m_size_old + m_size >= m_capacity) {
            return false;
        }
        const auto write_idx = m_write_head;
        // Copy data to the corresponding position using views
        xt::view(m_param_sets, write_idx, xt::all(), xt::all()) = param_set;
        xt::view(m_folds, write_idx, xt::all(), xt::all())      = fold;
        m_scores[write_idx]                                     = score;
        // Convert backtrack vector to xtensor view and assign
        xt::xtensor<SizeType, 1> bt_tensor = xt::adapt(backtrack);
        xt::view(m_backtracks, write_idx, xt::range(0, bt_tensor.size())) =
            bt_tensor;

        m_write_head = (m_write_head + 1) % m_capacity;
        ++m_size;
        return true;
    }

    void add_initial(const xt::xtensor<double, 3>& param_sets_batch,
                     const xt::xtensor<FoldType, 3>& folds_batch,
                     const std::vector<float>& scores_batch,
                     const xt::xtensor<SizeType, 2>& backtracks_batch) {
        const auto slots_to_write = scores_batch.size();
        if (slots_to_write > m_capacity) {
            throw std::runtime_error(std::format(
                "SuggestionStruct: Suggestions too large to add: {} > {}",
                slots_to_write, m_capacity));
        }
        reset(); // Start fresh
        // Copy all data efficiently using views
        for (SizeType i = 0; i < slots_to_write; ++i) {
            xt::view(m_param_sets, i, xt::all(), xt::all()) =
                xt::view(param_sets_batch, i, xt::all(), xt::all());
            xt::view(m_folds, i, xt::all(), xt::all()) =
                xt::view(folds_batch, i, xt::all(), xt::all());
            xt::view(m_backtracks, i, xt::all()) =
                xt::view(backtracks_batch, i, xt::all());
            m_scores[i] = scores_batch[i];
        }
        m_size = slots_to_write;
        error_check::check(m_size <= m_capacity,
                           "SuggestionStruct: Invalid size after add_initial. "
                           "Buffer overflow.");
    }

    // Adds filtered batch to write region. If full, trims write region via
    // median threshold. Loops until all candidates fit, reclaiming space from
    // consumed old suggestions.
    float add_batch(const xt::xtensor<double, 3>& param_sets_batch,
                    const xt::xtensor<FoldType, 3>& folds_batch,
                    std::span<const float> scores_batch,
                    const xt::xtensor<SizeType, 2>& backtracks_batch,
                    float current_threshold) {
        // Always use scores_batch to get the correct batch size
        const auto slots_to_write = scores_batch.size();
        if (slots_to_write == 0) {
            return current_threshold;
        }
        auto effective_threshold = current_threshold;

        std::span<double> m_param_sets_span(m_param_sets.data(),
                                            m_param_sets.size());
        std::span<FoldType> m_folds_span(m_folds.data(), m_folds.size());
        std::span<SizeType> m_backtracks_span(m_backtracks.data(),
                                              m_backtracks.size());
        std::span<float> m_scores_span(m_scores.data(), m_scores.size());
        std::span<const double> param_sets_batch_span(param_sets_batch.data(),
                                                      param_sets_batch.size());
        std::span<const FoldType> folds_batch_span(folds_batch.data(),
                                                   folds_batch.size());
        std::span<const SizeType> backtracks_batch_span(
            backtracks_batch.data(), backtracks_batch.size());

        // Create initial mask for scores >= threshold
        std::vector<SizeType> pending_indices;
        pending_indices.reserve(slots_to_write);

        auto update_candidates = [&]() {
            pending_indices.clear();
            for (SizeType i = 0; i < slots_to_write; ++i) {
                if (scores_batch[i] >= effective_threshold) {
                    pending_indices.push_back(i);
                }
            }
        };

        update_candidates();

        while (!pending_indices.empty()) {
            const auto space_left =
                m_capacity - ((m_size_old - m_read_consumed) + m_size);
            if (space_left < 0 || space_left > m_capacity) {
                throw std::runtime_error(std::format(
                    "SuggestionStruct: Invalid space left ({}) after "
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

            const auto n_to_add_now =
                std::min(pending_indices.size(), space_left);

            // Batched assignment
            for (SizeType i = 0; i < n_to_add_now; ++i) {
                const auto src_idx               = pending_indices[i];
                const auto dst_idx               = m_write_head;
                const SizeType leaves_src_offset = src_idx * m_leaves_stride;
                const SizeType leaves_dst_offset = dst_idx * m_leaves_stride;
                const SizeType combined_res_src_offset =
                    src_idx * m_combined_res_stride;
                const SizeType combined_res_dst_offset =
                    dst_idx * m_combined_res_stride;
                const SizeType backtrack_src_offset =
                    src_idx * m_backtrack_stride;
                const SizeType backtrack_dst_offset =
                    dst_idx * m_backtrack_stride;
                std::copy(param_sets_batch_span.begin() + leaves_src_offset,
                          param_sets_batch_span.begin() + leaves_src_offset +
                              m_leaves_stride,
                          m_param_sets_span.begin() + leaves_dst_offset);
                std::copy(folds_batch_span.begin() + combined_res_src_offset,
                          folds_batch_span.begin() + combined_res_src_offset +
                              m_combined_res_stride,
                          m_folds_span.begin() + combined_res_dst_offset);
                std::copy(backtracks_batch_span.begin() + backtrack_src_offset,
                          backtracks_batch_span.begin() + backtrack_src_offset +
                              m_backtrack_stride,
                          m_backtracks_span.begin() + backtrack_dst_offset);
                m_scores_span[dst_idx] = scores_batch[src_idx];

                m_write_head = (m_write_head + 1) % m_capacity;
            }
            m_size += n_to_add_now;
            error_check::check(m_size <= m_capacity,
                               "SuggestionStruct: Invalid size after "
                               "add_batch. Buffer overflow.");

            // Remove added candidates from the list
            pending_indices.erase(pending_indices.begin(),
                                  pending_indices.begin() +
                                      static_cast<IndexType>(n_to_add_now));
        }
        return effective_threshold;
    }

    float trim_threshold() {
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
        // After trimming, the write head must be updated to the new end of the
        // block.
        if (m_is_updating) {
            m_write_head = (m_write_start + m_size) % m_capacity;
        }
        return threshold;
    }

    float trim_half() {
        if (m_size == 0) {
            return 0.0F;
        }

        // To guarantee progress, we will keep at most half of the suggestions.
        const SizeType n_to_keep = m_size / 2;

        // Find the score of the (n_to_keep)-th best suggestion. This will be
        // our new threshold.
        std::vector<float> scores_copy;
        scores_copy.reserve(m_size);
        const auto start_idx = m_is_updating ? m_write_start : m_head;
        for (SizeType i = 0; i < m_size; ++i) {
            scores_copy.push_back(m_scores[(start_idx + i) % m_capacity]);
        }

        // Find the threshold score that would keep the top `n_to_keep`
        // elements. We sort in descending order to find the n_to_keep-th
        // largest element.
        auto nth = scores_copy.begin() + n_to_keep;
        std::nth_element(scores_copy.begin(), nth, scores_copy.end(),
                         std::greater<float>());
        const float threshold =
            (n_to_keep < m_size) ? *nth : scores_copy.back();

        // Create a mask to keep the top n_to_keep suggestions, breaking ties
        // arbitrarily but consistently.
        std::vector<bool> indices(m_size, false);
        SizeType kept_count = 0;
        // Keep all suggestions with a score strictly greater than the
        // threshold.
        for (SizeType i = 0; i < m_size; ++i) {
            const auto current_idx = (start_idx + i) % m_capacity;
            if (m_scores[current_idx] > threshold) {
                indices[i] = true;
                kept_count++;
            }
        }
        // Keep suggestions with a score equal to the threshold until we reach
        // n_to_keep.
        if (kept_count < n_to_keep) {
            for (SizeType i = 0; i < m_size; ++i) {
                if (kept_count >= n_to_keep) {
                    break;
                }
                const auto current_idx = (start_idx + i) % m_capacity;
                if (m_scores[current_idx] == threshold) {
                    indices[i] = true;
                    kept_count++;
                }
            }
        }

        keep(indices);
        // After trimming, the write head must be updated to the new end of the
        // block.
        if (m_is_updating) {
            m_write_head = (m_write_start + m_size) % m_capacity;
        }
        return threshold;
    }

    void trim_repeats() {
        if (m_size == 0) {
            return;
        }
        const auto start_idx = m_is_updating ? m_write_start : m_head;
        xt::xtensor<double, 3> contig_params({m_size, m_nparams + 2, 2});
        std::vector<float> contig_scores(m_size);
        for (SizeType i = 0; i < m_size; ++i) {
            const auto src_idx = (start_idx + i) % m_capacity;
            xt::view(contig_params, i, xt::all(), xt::all()) =
                xt::view(m_param_sets, src_idx, xt::all(), xt::all());
            contig_scores[i] = m_scores[src_idx];
        }

        // Get unique indices on contiguous copy
        const auto unique_idx = get_unique_indices_scores(
            contig_params, std::span<const float>(contig_scores));

        // Convert indices to boolean mask
        std::vector<bool> idx_bool(m_size, false);
        for (SizeType idx : unique_idx) {
            idx_bool[idx] = true;
        }

        keep(idx_bool);
        // After trimming, the write head must be updated to the new end of the
        // block.
        if (m_is_updating) {
            m_write_head = (m_write_start + m_size) % m_capacity;
        }
    }

    float trim_repeats_threshold() {
        if (m_size == 0) {
            return 0.0F;
        }

        const float threshold = get_score_median();
        const auto start_idx  = m_is_updating ? m_write_start : m_head;

        xt::xtensor<double, 3> contig_params({m_size, m_nparams + 2, 2});
        std::vector<float> contig_scores(m_size);
        for (SizeType i = 0; i < m_size; ++i) {
            const auto src_idx = (start_idx + i) % m_capacity;
            xt::view(contig_params, i, xt::all(), xt::all()) =
                xt::view(m_param_sets, src_idx, xt::all(), xt::all());
            contig_scores[i] = m_scores[src_idx];
        }

        auto unique_idx = get_unique_indices_scores(
            contig_params, std::span<const float>(contig_scores));

        // Create boolean mask for unique indices and scores >= threshold
        std::vector<bool> idx_bool(m_size, false);
        for (SizeType idx : unique_idx) {
            if (contig_scores[idx] >= threshold) {
                idx_bool[idx] = true;
            }
        }

        keep(idx_bool);
        // After trimming, the write head must be updated to the new end of the
        // block.
        if (m_is_updating) {
            m_write_head = (m_write_start + m_size) % m_capacity;
        }
        return threshold;
    }

private:
    xt::xtensor<double, 3> m_param_sets;   // Shape: (capacity, nparams + 2, 2)
    xt::xtensor<FoldType, 3> m_folds;      // Shape: (capacity, 2, nbins)
    std::vector<float> m_scores;           // Shape: (capacity)
    xt::xtensor<SizeType, 2> m_backtracks; // Shape: (capacity, nparams + 2)

    SizeType m_capacity;
    SizeType m_nparams;
    SizeType m_nbins;
    SizeType m_leaves_stride;
    SizeType m_combined_res_stride;
    SizeType m_backtrack_stride;

    // Circular buffer state
    SizeType m_head{0};     // Index of the first valid element
    SizeType m_size{0};     // Number of valid elements in the buffer
    SizeType m_size_old{0}; // Number of elements from previous iteration

    // In-place update state
    SizeType m_write_head{0}; // Index where next element will be written
    SizeType m_write_start{0};
    bool m_is_updating{false};
    SizeType m_read_consumed{0};

    // Compacts the current region (write or full, based on m_is_updating) using
    // boolean mask.
    // Only affects [start_idx, start_idx + m_size); updates m_size.
    void keep(const std::vector<bool>& indices) {
        // Count how many elements to keep
        SizeType count = std::count(indices.begin(), indices.end(), true);
        if (count == m_size) {
            return;
        }
        if (count == 0) {
            m_size = 0;
            return;
        }
        const auto start_idx = m_is_updating ? m_write_start : m_head;
        SizeType write_idx   = start_idx;
        for (SizeType read_idx_rel = 0; read_idx_rel < m_size; ++read_idx_rel) {
            if (indices[read_idx_rel]) {
                const auto src_idx = (start_idx + read_idx_rel) % m_capacity;
                if (write_idx != src_idx) {
                    xt::view(m_param_sets, write_idx, xt::all(), xt::all()) =
                        xt::view(m_param_sets, src_idx, xt::all(), xt::all());
                    xt::view(m_folds, write_idx, xt::all(), xt::all()) =
                        xt::view(m_folds, src_idx, xt::all(), xt::all());
                    xt::view(m_backtracks, write_idx, xt::all()) =
                        xt::view(m_backtracks, src_idx, xt::all());
                    m_scores[write_idx] = m_scores[src_idx];
                }
                write_idx = (write_idx + 1) % m_capacity;
            }
        }
        // Update valid size
        m_size = count;
        error_check::check(
            m_size <= m_capacity,
            "SuggestionStruct: Invalid size after keep. Buffer overflow.");
    }

    // Moves valid data (m_head to m_head + m_size) to buffer start (index 0).
    // Handles wrapped data with temp buffers to prevent overwrite.
    // Post-condition: m_head == 0, data is contiguous.
    void defragment() {
        if (m_head == 0) {
            return; // Already contiguous at the start
        }
        if (m_size == 0) {
            m_head = 0;
            return;
        }
        // Handle wrapped vs. non-wrapped data
        if (m_head + m_size <= m_capacity) {
            // Data is contiguous but not at the start – copy via temp buffer
            // (safety first, to avoid overlap with the write region).
            xt::xtensor<double, 3> params_slice =
                xt::view(m_param_sets, xt::range(m_head, m_head + m_size),
                         xt::all(), xt::all());
            xt::xtensor<FoldType, 3> folds_slice =
                xt::view(m_folds, xt::range(m_head, m_head + m_size), xt::all(),
                         xt::all());
            xt::xtensor<SizeType, 2> backtracks_slice = xt::view(
                m_backtracks, xt::range(m_head, m_head + m_size), xt::all());
            xt::view(m_param_sets, xt::range(0, m_size), xt::all(), xt::all()) =
                params_slice;
            xt::view(m_folds, xt::range(0, m_size), xt::all(), xt::all()) =
                folds_slice;
            xt::view(m_backtracks, xt::range(0, m_size), xt::all()) =
                backtracks_slice;
            std::vector<float> tmp_scores(m_scores.begin() + m_head,
                                          m_scores.begin() + m_head + m_size);
            std::ranges::copy(tmp_scores, m_scores.begin());

        } else {
            // Data is wrapped around the end of the buffer.
            const auto part1_size = m_capacity - m_head;
            const auto part2_size = m_size - part1_size;

            // Use a temporary buffer for the first part to avoid overwriting
            // it.
            xt::xtensor<double, 3> temp_params =
                xt::view(m_param_sets, xt::range(m_head, m_capacity), xt::all(),
                         xt::all());
            xt::xtensor<FoldType, 3> temp_folds = xt::view(
                m_folds, xt::range(m_head, m_capacity), xt::all(), xt::all());
            std::vector<float> temp_scores(m_scores.begin() + m_head,
                                           m_scores.end());
            xt::xtensor<SizeType, 2> temp_backtracks = xt::view(
                m_backtracks, xt::range(m_head, m_capacity), xt::all());

            // Move the second part (from the start of the buffer) to its new
            // position
            xt::view(m_param_sets, xt::range(part1_size, m_size), xt::all(),
                     xt::all()) =
                xt::view(m_param_sets, xt::range(0, part2_size), xt::all(),
                         xt::all());
            xt::view(m_folds, xt::range(part1_size, m_size), xt::all(),
                     xt::all()) = xt::view(m_folds, xt::range(0, part2_size),
                                           xt::all(), xt::all());
            std::copy(m_scores.begin(), m_scores.begin() + part2_size,
                      m_scores.begin() + part1_size);
            xt::view(m_backtracks, xt::range(part1_size, m_size), xt::all()) =
                xt::view(m_backtracks, xt::range(0, part2_size), xt::all());

            // Copy the first part from the temporary buffer to the start.
            xt::view(m_param_sets, xt::range(0, part1_size), xt::all(),
                     xt::all()) = temp_params;
            xt::view(m_folds, xt::range(0, part1_size), xt::all(), xt::all()) =
                temp_folds;
            std::ranges::copy(temp_scores, m_scores.begin());
            xt::view(m_backtracks, xt::range(0, part1_size), xt::all()) =
                temp_backtracks;
        }

        m_head = 0;
    }
}; // End SuggestionStruct::Impl definition

// Public interface implementation
template <typename FoldType>
SuggestionStruct<FoldType>::SuggestionStruct(SizeType max_sugg,
                                             SizeType nparams,
                                             SizeType nbins)
    : m_impl(std::make_unique<Impl>(max_sugg, nparams, nbins)) {}
template <typename FoldType>
SuggestionStruct<FoldType>::~SuggestionStruct() = default;
template <typename FoldType>
SuggestionStruct<FoldType>::SuggestionStruct(
    SuggestionStruct&& other) noexcept = default;
template <typename FoldType>
SuggestionStruct<FoldType>& SuggestionStruct<FoldType>::operator=(
    SuggestionStruct<FoldType>&& other) noexcept = default;
// Getters
template <typename FoldType>
const xt::xtensor<double, 3>&
SuggestionStruct<FoldType>::get_param_sets() const noexcept {
    return m_impl->get_param_sets();
}
template <typename FoldType>
const xt::xtensor<FoldType, 3>&
SuggestionStruct<FoldType>::get_folds() const noexcept {
    return m_impl->get_folds();
}
template <typename FoldType>
std::vector<float> SuggestionStruct<FoldType>::get_scores() const noexcept {
    return m_impl->get_scores();
}
template <typename FoldType>
const xt::xtensor<SizeType, 2>&
SuggestionStruct<FoldType>::get_backtracks() const noexcept {
    return m_impl->get_backtracks();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_max_sugg() const noexcept {
    return m_impl->get_max_sugg();
}
template <typename FoldType>
size_t SuggestionStruct<FoldType>::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_nbins() const noexcept {
    return m_impl->get_nbins();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_nsugg() const noexcept {
    return m_impl->get_nsugg();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_nsugg_old() const noexcept {
    return m_impl->get_nsugg_old();
}
template <typename FoldType>
float SuggestionStruct<FoldType>::get_nsugg_lb() const noexcept {
    return m_impl->get_nsugg_lb();
}
template <typename FoldType>
float SuggestionStruct<FoldType>::get_score_max() const noexcept {
    return m_impl->get_score_max();
}
template <typename FoldType>
float SuggestionStruct<FoldType>::get_score_min() const noexcept {
    return m_impl->get_score_min();
}
template <typename FoldType>
float SuggestionStruct<FoldType>::get_score_median() const noexcept {
    return m_impl->get_score_median();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
template <typename FoldType>
void SuggestionStruct<FoldType>::set_nsugg(SizeType nsugg) noexcept {
    m_impl->set_nsugg(nsugg);
}
template <typename FoldType> void SuggestionStruct<FoldType>::reset() noexcept {
    m_impl->reset();
}
template <typename FoldType>
void SuggestionStruct<FoldType>::prepare_for_in_place_update() {
    m_impl->prepare_for_in_place_update();
}
template <typename FoldType>
void SuggestionStruct<FoldType>::finalize_in_place_update() {
    m_impl->finalize_in_place_update();
}
template <typename FoldType>
void SuggestionStruct<FoldType>::advance_read_consumed(SizeType n) {
    m_impl->advance_read_consumed(n);
}
// Other methods
template <typename FoldType>
std::tuple<xt::xtensor<double, 2>, xt::xtensor<FoldType, 2>, float>
SuggestionStruct<FoldType>::get_best() const {
    return m_impl->get_best();
}
template <typename FoldType>
xt::xtensor<double, 3>
SuggestionStruct<FoldType>::get_transformed(double delta_t) const {
    return m_impl->get_transformed(delta_t);
}
template <typename FoldType>
bool SuggestionStruct<FoldType>::add(const xt::xtensor<double, 2>& param_set,
                                     const xt::xtensor<FoldType, 2>& fold,
                                     float score,
                                     const std::vector<SizeType>& backtrack) {
    return m_impl->add(param_set, fold, score, backtrack);
}
template <typename FoldType>
void SuggestionStruct<FoldType>::add_initial(
    const xt::xtensor<double, 3>& param_sets_batch,
    const xt::xtensor<FoldType, 3>& folds_batch,
    const std::vector<float>& scores_batch,
    const xt::xtensor<SizeType, 2>& backtracks_batch) {
    m_impl->add_initial(param_sets_batch, folds_batch, scores_batch,
                        backtracks_batch);
}
template <typename FoldType>
float SuggestionStruct<FoldType>::add_batch(
    const xt::xtensor<double, 3>& param_sets_batch,
    const xt::xtensor<FoldType, 3>& folds_batch,
    std::span<const float> scores_batch,
    const xt::xtensor<SizeType, 2>& backtracks_batch,
    float current_threshold) {
    return m_impl->add_batch(param_sets_batch, folds_batch, scores_batch,
                             backtracks_batch, current_threshold);
}
template <typename FoldType>
float SuggestionStruct<FoldType>::trim_threshold() {
    return m_impl->trim_threshold();
}
template <typename FoldType> void SuggestionStruct<FoldType>::trim_repeats() {
    m_impl->trim_repeats();
}
template <typename FoldType>
float SuggestionStruct<FoldType>::trim_repeats_threshold() {
    return m_impl->trim_repeats_threshold();
}

// Explicit instantiation
template class SuggestionStruct<float>;
template class SuggestionStruct<ComplexType>;
} // namespace loki::utils