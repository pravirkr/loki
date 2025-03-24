#pragma once

#include <cmath>
#include <span>
#include <tuple>
#include <vector>

#include <xtensor/xtensor.hpp>

#include <loki/loki_types.hpp>

class SuggestionStruct {
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
                     xt::xtensor<float, 3> folds,
                     std::vector<float> scores,
                     xt::xtensor<SizeType, 2> backtracks);

    // Create a new empty suggestion struct with specified capacity
    [[nodiscard]] SuggestionStruct get_new(SizeType max_sugg) const;

    // Get the best suggestion (highest score)
    [[nodiscard]] std::
        tuple<xt::xtensor<double, 2>, xt::xtensor<float, 2>, float>
        get_best() const;

    // Transform the search parameters to some given t_ref
    [[nodiscard]] xt::xtensor<double, 3> get_transformed(double delta_t) const;

    // Add a suggestion to the struct if there is space
    bool add(const xt::xtensor<double, 2>& param_set,
             const xt::xtensor<float, 2>& fold,
             float score,
             const std::vector<SizeType>& backtrack);

    // Trim to keep only suggestions with scores >= median
    [[nodiscard]] float trim_threshold();

    // Return only the valid portion of the struct, excluding garbage data
    [[nodiscard]] SuggestionStruct trim_empty() const;

    // Trim repeated suggestions
    void trim_repeats();

    // Trim repeated suggestions and keep only those with scores >= median
    [[nodiscard]] float trim_repeats_threshold();

    // Getters
    [[nodiscard]] const xt::xtensor<double, 3>& param_sets() const {
        return m_param_sets;
    }
    [[nodiscard]] const xt::xtensor<float, 3>& folds() const { return m_folds; }
    [[nodiscard]] const std::vector<float>& scores() const { return m_scores; }
    [[nodiscard]] const xt::xtensor<SizeType, 2>& backtracks() const {
        return m_backtracks;
    }
    [[nodiscard]] SizeType valid_size() const { return m_valid_size; }
    [[nodiscard]] SizeType size() const { return m_size; }
    [[nodiscard]] size_t nparams() const { return m_param_sets.shape()[1]; }

    // Additional properties
    [[nodiscard]] float size_lb() const {
        return m_valid_size > 0 ? std::log2(static_cast<float>(m_valid_size))
                                : 0.0F;
    }
    [[nodiscard]] float score_max() const;
    [[nodiscard]] float score_min() const;
    [[nodiscard]] float score_median() const;

private:
    // Optimized in-place update
    void keep(const std::vector<bool>& indices);

    // Member variables
    xt::xtensor<double, 3> m_param_sets;   // Shape: (nsuggestions, nparams, 2)
    xt::xtensor<float, 3> m_folds;         // Shape: (nsuggestions, nbins, 2)
    std::vector<float> m_scores;           // Shape: (nsuggestions)
    xt::xtensor<SizeType, 2> m_backtracks; // Shape: (nsuggestions, 2 + nparams)
    SizeType m_valid_size;
    SizeType m_size;
};

// Helper functions
std::vector<SizeType> get_unique_indices(const xt::xtensor<double, 3>& params);
std::vector<SizeType>
get_unique_indices_scores(const xt::xtensor<double, 3>& params,
                          std::span<const float> scores);