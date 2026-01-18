#pragma once

#include <memory>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"

namespace loki::utils {

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

    ~WorldTree();
    WorldTree(WorldTree&&) noexcept;
    WorldTree& operator=(WorldTree&&) noexcept;
    WorldTree(const WorldTree&)            = delete;
    WorldTree& operator=(const WorldTree&) = delete;

    // Getters
    [[nodiscard]] const std::vector<double>& get_leaves() const noexcept;
    [[nodiscard]] const std::vector<FoldType>& get_folds() const noexcept;
    [[nodiscard]] const std::vector<float>& get_scores() const noexcept;
    [[nodiscard]] SizeType get_capacity() const noexcept;
    [[nodiscard]] SizeType get_nparams() const noexcept;
    [[nodiscard]] SizeType get_nbins() const noexcept;
    [[nodiscard]] SizeType get_max_batch_size() const noexcept;
    [[nodiscard]] SizeType get_leaves_stride() const noexcept;
    [[nodiscard]] SizeType get_folds_stride() const noexcept;
    [[nodiscard]] SizeType get_size() const noexcept;
    [[nodiscard]] SizeType get_size_old() const noexcept;
    [[nodiscard]] float get_size_lb() const noexcept;
    [[nodiscard]] float get_score_max() const noexcept;
    [[nodiscard]] float get_score_min() const noexcept;
    [[nodiscard]] float get_memory_usage() const noexcept;

    // Returns {span, actual_size}, where actual_size <= requested n_leaves,
    // limited to contiguous segment before wrap.
    [[nodiscard]] std::pair<std::span<const double>, SizeType>
    get_leaves_span(SizeType n_leaves) const;
    // Returns span over contiguous leaves (for reporting)
    [[nodiscard]] std::span<double> get_leaves_contiguous_span() noexcept;
    // Returns span over contiguous scores (for saving to file)
    [[nodiscard]] std::span<float> get_scores_contiguous_span() noexcept;

    void set_size(SizeType size) noexcept;
    void reset() noexcept;
    void prepare_in_place_update();
    void finalize_in_place_update();
    void consume_read(SizeType n);

    void convert_to_physical_indices(std::span<SizeType> logical_indices,
                                     SizeType n_leaves) const;

    // Get the best candidate (highest score)
    [[nodiscard]] std::
        tuple<std::span<const double>, std::span<const FoldType>, float>
        get_best() const;

    // Add an initial set of candidate leaves to the Tree
    void add_initial(std::span<const double> leaves_batch,
                     std::span<const FoldType> folds_batch,
                     std::span<const float> scores_batch,
                     SizeType slots_to_write);
    // Add a candidate leaf to the Tree if there is space
    [[nodiscard]] bool add(std::span<const double> leaf,
                           std::span<const FoldType> fold,
                           float score);
    // Add a batch of candidate leaves to the Tree if there is space
    [[nodiscard]] float add_batch(std::span<const double> leaves_batch,
                                  std::span<const FoldType> folds_batch,
                                  std::span<const float> scores_batch,
                                  float current_threshold,
                                  SizeType slots_to_write);
    // Prune to keep only unique candidates
    void deduplicate();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

using WorldTreeFloat   = WorldTree<float>;
using WorldTreeComplex = WorldTree<ComplexType>;

} // namespace loki::utils