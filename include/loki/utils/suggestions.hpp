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
     * A struct to hold suggestions for pruning.
     *
     * @param param_sets Parameter sets: shape (nsuggestions, nparams, 2)
     * @param folds Folded profiles: shape (nsuggestions, nbins, 2)
     * @param scores Scores for each suggestion (nsuggestions)
     * @param backtracks Backtracks: shape (nsuggestions, 2 + nparams)
     */
    SuggestionStruct(xt::xtensor<double, 3> param_sets,
                     xt::xtensor<FoldType, 3> folds,
                     std::vector<float> scores,
                     xt::xtensor<SizeType, 2> backtracks);

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
    [[nodiscard]] SizeType get_valid_size() const noexcept;
    [[nodiscard]] SizeType get_size() const noexcept;
    [[nodiscard]] SizeType get_nparams() const noexcept;
    [[nodiscard]] float get_size_lb() const noexcept;
    [[nodiscard]] float get_score_max() const noexcept;
    [[nodiscard]] float get_score_min() const noexcept;
    [[nodiscard]] float get_score_median() const noexcept;

    // Setter
    void set_valid_size(SizeType valid_size) noexcept;

    // Create a new empty suggestion struct with specified capacity
    [[nodiscard]] SuggestionStruct get_new(SizeType max_sugg) const;
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
    // Add a batch of suggestions to the struct if there is space
    [[nodiscard]] float
    add_batch(const xt::xtensor<double, 3>& param_sets_batch,
              const xt::xtensor<FoldType, 3>& folds_batch,
              const std::vector<float>& scores_batch,
              const xt::xtensor<SizeType, 2>& backtracks_batch,
              float current_threshold);
    // Trim to keep only suggestions with scores >= median
    [[nodiscard]] float trim_threshold();
    // Return only the valid portion of the struct, excluding garbage data
    [[nodiscard]] SuggestionStruct trim_empty() const;
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