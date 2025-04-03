#include "loki/suggestions.hpp"

#include <algorithm>
#include <ranges>
#include <span>
#include <unordered_map>

#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

#include <omp.h>

#include "loki/loki_types.hpp"

SuggestionStruct::SuggestionStruct(xt::xtensor<double, 3> param_sets,
                                   xt::xtensor<float, 3> folds,
                                   std::vector<float> scores,
                                   xt::xtensor<SizeType, 2> backtracks)
    : m_param_sets(std::move(param_sets)),
      m_folds(std::move(folds)),
      m_scores(std::move(scores)),
      m_backtracks(std::move(backtracks)),
      m_valid_size(m_param_sets.shape()[0]),
      m_size(m_param_sets.shape()[0]) {}

SuggestionStruct SuggestionStruct::get_new(SizeType max_sugg) const {
    // Create empty tensors with the same shapes but different first dimension
    xt::xtensor<double, 3> param_sets(
        {max_sugg, m_param_sets.shape()[1], m_param_sets.shape()[2]}, 0.0);
    xt::xtensor<float, 3> folds(
        {max_sugg, m_folds.shape()[1], m_folds.shape()[2]}, 0.0F);
    std::vector<float> scores(max_sugg, 0.0F);
    xt::xtensor<SizeType, 2> backtracks({max_sugg, m_backtracks.shape()[1]}, 0);

    auto result = SuggestionStruct(std::move(param_sets), std::move(folds),
                                   std::move(scores), std::move(backtracks));
    // Set valid_size to 0 since this is a new empty struct
    result.m_valid_size = 0;
    return result;
}

std::tuple<xt::xtensor<double, 2>, xt::xtensor<float, 2>, float>
SuggestionStruct::get_best() const {
    if (m_valid_size == 0) {
        // Return empty tensors and 0 score
        xt::xtensor<double, 2> empty_params(
            {m_param_sets.shape()[1], m_param_sets.shape()[2]}, 0.0);
        xt::xtensor<float, 2> empty_folds(
            {m_folds.shape()[1], m_folds.shape()[2]}, 0.0F);
        return std::make_tuple(empty_params, empty_folds, 0.0F);
    }

    // Find the index of the maximum score
    SizeType idx_max = std::ranges::distance(
        m_scores.begin(),
        std::ranges::max_element(m_scores | std::views::take(m_valid_size)));

    // Extract the best parameter set and fold using views
    auto best_params = xt::view(m_param_sets, idx_max, xt::all(), xt::all());
    auto best_folds  = xt::view(m_folds, idx_max, xt::all(), xt::all());
    // Return copies
    return std::make_tuple(best_params, best_folds, m_scores[idx_max]);
}

xt::xtensor<double, 3> SuggestionStruct::get_transformed(double delta_t) const {
    // Extract all parameter sets except the last two rows for each suggestion
    // using views
    if (m_param_sets.shape()[1] <= 2) {
        return xt::xtensor<double, 3>({0, 0, 0});
    }
    xt::xtensor<double, 3> transformed_params(
        {m_valid_size, m_param_sets.shape()[1] - 2, m_param_sets.shape()[2]});

    // Extract all but the last two parameters for each suggestion
    for (SizeType i = 0; i < m_valid_size; ++i) {
        auto view =
            xt::view(m_param_sets, i, xt::range(0, m_param_sets.shape()[1] - 2),
                     xt::all());
        xt::view(transformed_params, i, xt::all(), xt::all()) = view;
    }

    // Call the batch shift function
    // return psr_utils::shift_params_batch(transformed_params, delta_t);
    return transformed_params;
}

bool SuggestionStruct::add(const xt::xtensor<double, 2>& param_set,
                           const xt::xtensor<float, 2>& fold,
                           float score,
                           const std::vector<SizeType>& backtrack) {
    if (m_valid_size >= m_size) {
        return false;
    }

    // Copy data to the corresponding position using views
    xt::view(m_param_sets, m_valid_size, xt::all(), xt::all()) = param_set;
    xt::view(m_folds, m_valid_size, xt::all(), xt::all())      = fold;
    m_scores[m_valid_size]                                     = score;
    // Convert backtrack vector to xtensor view and assign
    xt::xtensor<SizeType, 1> bt_tensor = xt::adapt(backtrack);
    xt::view(m_backtracks, m_valid_size, xt::range(0, bt_tensor.size())) =
        bt_tensor;

    ++m_valid_size;
    return true;
}

float SuggestionStruct::trim_threshold() {
    if (m_valid_size == 0) {
        return 0.0F;
    }
    // Compute median score
    const float threshold = score_median();

    // Create a boolean mask for scores >= threshold
    std::vector<bool> indices(m_valid_size, false);
    for (SizeType i = 0; i < m_valid_size; ++i) {
        indices[i] = m_scores[i] >= threshold;
    }

    keep(indices);
    return threshold;
}

SuggestionStruct SuggestionStruct::trim_empty() const {
    // Return a new SuggestionStruct with only the valid entries
    if (m_valid_size == 0) {
        return get_new(0);
    }

    // Create views of the valid portions
    auto param_view = xt::view(m_param_sets, xt::range(0, m_valid_size),
                               xt::all(), xt::all());
    auto m_foldsview =
        xt::view(m_folds, xt::range(0, m_valid_size), xt::all(), xt::all());
    auto m_backtracksview =
        xt::view(m_backtracks, xt::range(0, m_valid_size), xt::all());

    // Evaluate views to create actual tensors
    xt::xtensor<double, 3> param_sets   = xt::eval(param_view);
    xt::xtensor<float, 3> folds         = xt::eval(m_foldsview);
    xt::xtensor<SizeType, 2> backtracks = xt::eval(m_backtracksview);

    std::vector<float> scores(m_scores.begin(),
                              m_scores.begin() +
                                  static_cast<IndexType>(m_valid_size));

    return {std::move(param_sets), std::move(folds), std::move(scores),
            std::move(backtracks)};
}

void SuggestionStruct::trim_repeats() {
    if (m_valid_size == 0) {
        return;
    }

    // Get unique indices
    const auto unique_idx = get_unique_indices_scores(
        xt::view(m_param_sets, xt::range(0, m_valid_size), xt::all(),
                 xt::all()),
        std::span<const float>(m_scores.data(), m_valid_size));

    // Convert indices to boolean mask
    std::vector<bool> idx_bool(m_valid_size, false);
    for (SizeType idx : unique_idx) {
        idx_bool[idx] = true;
    }

    keep(idx_bool);
}

float SuggestionStruct::trim_repeats_threshold() {
    if (m_valid_size == 0) {
        return 0.0F;
    }

    // Calculate median score
    const float threshold = score_median();
    // Get unique indices
    auto unique_idx = get_unique_indices_scores(
        xt::view(m_param_sets, xt::range(0, m_valid_size), xt::all(),
                 xt::all()),
        std::span<const float>(m_scores.data(), m_valid_size));

    // Create boolean mask for unique indices and scores >= threshold
    std::vector<bool> idx_bool(m_valid_size, false);
    for (SizeType idx : unique_idx) {
        idx_bool[idx] = m_scores[idx] >= threshold;
    }

    keep(idx_bool);
    return threshold;
}

void SuggestionStruct::keep(const std::vector<bool>& indices) {
    // Count how many elements to keep
    SizeType count = std::count(indices.begin(), indices.end(), true);
    if (count == 0) {
        m_valid_size = 0;
        return;
    }
    auto idx_valid = xt::argwhere(xt::adapt(indices));
    for (SizeType i = 0; i < count; ++i) {
        SizeType valid_idx = idx_valid[i][0]; // Extract scalar index
        xt::view(m_param_sets, i, xt::all(), xt::all()) =
            xt::view(m_param_sets, valid_idx, xt::all(), xt::all());
        xt::view(m_folds, i, xt::all(), xt::all()) =
            xt::view(m_folds, valid_idx, xt::all(), xt::all());
        xt::view(m_backtracks, i, xt::all()) =
            xt::view(m_backtracks, valid_idx, xt::all());
        m_scores[i] = m_scores[valid_idx];
    }
    // Update valid size
    m_valid_size = count;
}

float SuggestionStruct::score_max() const {
    if (m_valid_size == 0) {
        return 0.0F;
    }
    return *std::ranges::max_element(m_scores | std::views::take(m_valid_size));
}

float SuggestionStruct::score_min() const {
    if (m_valid_size == 0) {
        return 0.0F;
    }
    return *std::ranges::min_element(m_scores | std::views::take(m_valid_size));
}

float SuggestionStruct::score_median() const {
    if (m_valid_size == 0) {
        return 0.0F;
    }
    auto scores_view = m_scores | std::views::take(m_valid_size);
    std::vector<float> scores_copy(scores_view.begin(), scores_view.end());
    const auto mid = static_cast<IndexType>(m_valid_size) / 2;
    std::nth_element(scores_copy.begin(), scores_copy.begin() + mid,
                     scores_copy.end());
    return scores_copy[mid];
}

std::vector<SizeType> get_unique_indices(const xt::xtensor<double, 3>& params) {
    const SizeType nparams = params.shape()[0];
    std::vector<SizeType> unique_indices;
    unique_indices.reserve(nparams);
    std::unordered_map<int64_t, bool> unique_dict;
    unique_dict.reserve(nparams);

    for (SizeType i = 0; i < nparams; ++i) {
        // Use the last element's first value as the key
        const double val = params(i, params.shape()[1] - 1, 0);
        const auto key   = static_cast<int64_t>(std::round(val * 1e9));

        if (unique_dict.find(key) == unique_dict.end()) {
            unique_dict[key] = true;
            unique_indices.push_back(i);
        }
    }

    return unique_indices;
}

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