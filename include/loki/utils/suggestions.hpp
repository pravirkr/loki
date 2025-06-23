#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "loki/common/types.hpp"

namespace loki::utils {

template <typename FoldType = float> class SuggestionStruct {
public:
    /**
     * A struct to hold suggestions for pruning. Creates following arrays:
     * - param_sets: Parameter sets, shape (max_sugg, nparams + 2, 2)
     * - folds: Folded profiles, shape (max_sugg, 2, nbins)
     * - scores: Scores for each suggestion, shape (max_sugg)
     * - backtracks: Backtracks, shape (max_sugg, nparams + 2)
     *
     * @param max_sugg Maximum number of suggestions to hold
     * @param nparams Number of parameters
     * @param nbins Number of bins
     */
    SuggestionStruct(SizeType max_sugg, SizeType nparams, SizeType nbins);

    ~SuggestionStruct();
    SuggestionStruct(SuggestionStruct&&) noexcept;
    SuggestionStruct& operator=(SuggestionStruct&&) noexcept;
    SuggestionStruct(const SuggestionStruct&)            = delete;
    SuggestionStruct& operator=(const SuggestionStruct&) = delete;

    // Getters
    [[nodiscard]] const xt::xtensor<double, 3>& get_param_sets() const noexcept;
    [[nodiscard]] const xt::xtensor<FoldType, 3>& get_folds() const noexcept;
    [[nodiscard]] const std::vector<float>& get_scores() const noexcept;
    [[nodiscard]] const xt::xtensor<SizeType, 2>&
    get_backtracks() const noexcept;
    [[nodiscard]] SizeType get_max_sugg() const noexcept;
    [[nodiscard]] SizeType get_nparams() const noexcept;
    [[nodiscard]] SizeType get_nbins() const noexcept;
    [[nodiscard]] SizeType get_nsugg() const noexcept;
    [[nodiscard]] float get_nsugg_lb() const noexcept;
    [[nodiscard]] float get_score_max() const noexcept;
    [[nodiscard]] float get_score_min() const noexcept;
    [[nodiscard]] float get_score_median() const noexcept;

    void set_nsugg(SizeType nsugg) noexcept;
    void reset() noexcept;

    // Get the best suggestion (highest score)
    [[nodiscard]] std::
        tuple<xt::xtensor<double, 2>, xt::xtensor<FoldType, 2>, float>
        get_best() const;
    // Transform the search parameters to some given t_ref
    [[nodiscard]] xt::xtensor<double, 3> get_transformed(double delta_t) const;
    // Add a suggestion to the struct if there is space
    [[nodiscard]] bool add(const xt::xtensor<double, 2>& param_set,
                           const xt::xtensor<FoldType, 2>& fold,
                           float score,
                           const std::vector<SizeType>& backtrack);
    // Add an initial set of suggestions to the struct
    void add_initial(const xt::xtensor<double, 3>& param_sets_batch,
                     const xt::xtensor<FoldType, 3>& folds_batch,
                     const std::vector<float>& scores_batch,
                     const xt::xtensor<SizeType, 2>& backtracks_batch);
    // Add a batch of suggestions to the struct if there is space
    [[nodiscard]] float
    add_batch(const xt::xtensor<double, 3>& param_sets_batch,
              const xt::xtensor<FoldType, 3>& folds_batch,
              const std::vector<float>& scores_batch,
              const xt::xtensor<SizeType, 2>& backtracks_batch,
              float current_threshold);
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

using SuggestionStructFloat   = SuggestionStruct<float>;
using SuggestionStructComplex = SuggestionStruct<ComplexType>;

} // namespace loki::utils