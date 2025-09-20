#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

/**
 * @brief A memory-efficient circular buffer for managing search suggestions.
 *
 * This class implements a circular buffer for iterative search algorithms to
 * store, prune, and update "suggestions". It minimizes memory allocations by
 * avoiding defragmentation and using in-place updates.
 *
 * **Circular Buffer Design:**
 * - Buffer is never defragmented - data remains in circular layout
 * - New suggestions are generated from old ones within the same buffer, split
 *   into a "read region" (current iteration's input) and a "write region"
 *   (next iteration's candidates).
 * - Automatic trimming when buffer fills up
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
 * Suggestions struct contains following arrays:
 * - leaves: Parameter sets, shape (capacity, nparams + 2, 2)
 * - folds: Folded profiles, shape (capacity, 2, nbins)
 * - scores: Scores for each suggestion, shape (capacity)
 *
 * @tparam FoldType The type of the folded profiles.
 */
template <typename FoldType = float> class SuggestionTree {
public:
    /**
     * @brief Constructor for the SuggestionTree class.
     *
     * Initializes the internal arrays with the given maximum number of
     * suggestions, number of parameters, and number of bins.
     *
     * @param capacity Maximum number of suggestions to hold
     * @param nparams Number of parameters
     * @param nbins Number of bins
     * @param mode Mode of the suggestion tree (e.g. Taylor, Chebyshev)
     */
    SuggestionTree(SizeType capacity,
                   SizeType nparams,
                   SizeType nbins,
                   std::string_view mode);

    ~SuggestionTree();
    SuggestionTree(SuggestionTree&&) noexcept;
    SuggestionTree& operator=(SuggestionTree&&) noexcept;
    SuggestionTree(const SuggestionTree&)            = delete;
    SuggestionTree& operator=(const SuggestionTree&) = delete;

    // Getters
    [[nodiscard]] SizeType get_capacity() const noexcept;
    [[nodiscard]] SizeType get_nparams() const noexcept;
    [[nodiscard]] SizeType get_nbins() const noexcept;
    [[nodiscard]] std::string_view get_mode() const noexcept;
    [[nodiscard]] SizeType get_leaves_stride() const noexcept;
    [[nodiscard]] SizeType get_folds_stride() const noexcept;
    [[nodiscard]] const std::vector<double>& get_leaves() const noexcept;
    // Returns {span, actual_size}, where actual_size <= requested n_leaves,
    // limited to contiguous segment before wrap.
    std::pair<std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const;
    [[nodiscard]] const std::vector<FoldType>& get_folds() const noexcept;
    [[nodiscard]] std::vector<float> get_scores() const noexcept;

    [[nodiscard]] SizeType get_nsugg() const noexcept;
    [[nodiscard]] SizeType get_nsugg_old() const noexcept;
    [[nodiscard]] float get_nsugg_lb() const noexcept;
    [[nodiscard]] float get_score_max() const noexcept;
    [[nodiscard]] float get_score_min() const noexcept;
    [[nodiscard]] float get_score_median() const noexcept;
    [[nodiscard]] float get_memory_usage() const noexcept;

    void set_nsugg(SizeType nsugg) noexcept;
    void reset() noexcept;
    void prepare_for_in_place_update();
    void finalize_in_place_update();
    void advance_read_consumed(SizeType n);

    void compute_physical_indices(std::span<const SizeType> logical_indices,
                                  std::span<SizeType> physical_indices,
                                  SizeType n_leaves) const;

    // Get the best suggestion (highest score)
    [[nodiscard]] std::
        tuple<std::span<const double>, std::span<const FoldType>, float>
        get_best() const;
    // Transform the search parameters to given coordinate
    [[nodiscard]] std::vector<double>
    get_transformed(std::pair<double, double> coord_mid) const;
    // Add a suggestion to the struct if there is space
    [[nodiscard]] bool add(std::span<const double> leaf,
                           std::span<const FoldType> fold,
                           float score);
    // Add an initial set of suggestions to the struct
    void add_initial(std::span<const double> batch_leaves,
                     std::span<const FoldType> batch_folds,
                     std::span<const float> batch_scores,
                     SizeType slots_to_write);
    // Add a batch of suggestions to the struct if there is space
    [[nodiscard]] float add_batch(std::span<const double> batch_leaves,
                                  std::span<const FoldType> batch_folds,
                                  std::span<const float> batch_scores,
                                  float current_threshold,
                                  SizeType slots_to_write);
    // Trim to keep only suggestions with scores >= median
    [[nodiscard]] float trim_threshold();
    // Trim repeated suggestions
    void trim_repeats();
    // Trim repeated suggestions and keep only those with scores >= median
    [[nodiscard]] float trim_repeats_threshold();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using SuggestionTreeFloat   = SuggestionTree<float>;
using SuggestionTreeComplex = SuggestionTree<ComplexType>;

} // namespace loki::utils