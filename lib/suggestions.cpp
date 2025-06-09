#include "loki/utils/suggestions.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <ranges>
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

namespace loki::utils {

namespace {
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
} // namespace

template <typename FoldType> class SuggestionStruct<FoldType>::Impl {
public:
    Impl(xt::xtensor<double, 3> param_sets,
         xt::xtensor<FoldType, 3> folds,
         std::vector<float> scores,
         xt::xtensor<SizeType, 2> backtracks)
        : m_param_sets(std::move(param_sets)),
          m_folds(std::move(folds)),
          m_scores(std::move(scores)),
          m_backtracks(std::move(backtracks)),
          m_valid_size(m_param_sets.shape()[0]),
          m_size(m_param_sets.shape()[0]) {}

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
    const std::vector<float>& get_scores() const noexcept { return m_scores; }
    const xt::xtensor<SizeType, 2>& get_backtracks() const noexcept {
        return m_backtracks;
    }
    SizeType get_valid_size() const noexcept { return m_valid_size; }
    SizeType get_size() const noexcept { return m_size; }
    size_t get_nparams() const noexcept { return m_param_sets.shape()[1]; }
    float get_size_lb() const noexcept {
        return m_valid_size > 0 ? std::log2(static_cast<float>(m_valid_size))
                                : 0.0F;
    }
    float get_score_max() const {
        if (m_valid_size == 0) {
            return 0.0F;
        }
        return *std::ranges::max_element(m_scores |
                                         std::views::take(m_valid_size));
    }

    float get_score_min() const {
        if (m_valid_size == 0) {
            return 0.0F;
        }
        return *std::ranges::min_element(m_scores |
                                         std::views::take(m_valid_size));
    }

    float get_score_median() const {
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

    void set_valid_size(SizeType valid_size) noexcept {
        m_valid_size = valid_size;
    }

    std::tuple<xt::xtensor<double, 2>, xt::xtensor<FoldType, 2>, float>
    get_best() const {
        if (m_valid_size == 0) {
            // Return empty tensors and 0 score
            xt::xtensor<double, 2> empty_params(
                {m_param_sets.shape()[1], m_param_sets.shape()[2]}, 0.0);
            xt::xtensor<FoldType, 2> empty_folds(
                {m_folds.shape()[1], m_folds.shape()[2]});

            // Initialize empty_folds based on FoldType
            if constexpr (std::is_same_v<FoldType, std::complex<float>>) {
                empty_folds.fill(std::complex<float>(0.0F, 0.0F));
            } else {
                empty_folds.fill(FoldType{});
            }

            return std::make_tuple(empty_params, empty_folds, 0.0F);
        }

        // Find the index of the maximum score
        SizeType idx_max = std::ranges::distance(
            m_scores.begin(), std::ranges::max_element(
                                  m_scores | std::views::take(m_valid_size)));

        // Extract the best parameter set and fold using views
        auto best_params =
            xt::view(m_param_sets, idx_max, xt::all(), xt::all());
        auto best_folds = xt::view(m_folds, idx_max, xt::all(), xt::all());
        // Return copies
        return std::make_tuple(best_params, best_folds, m_scores[idx_max]);
    }

    xt::xtensor<double, 3> get_transformed(double delta_t) const {
        // Extract all parameter sets except the last two rows for each
        // suggestion using views
        if (m_param_sets.shape()[1] <= 2) {
            return xt::xtensor<double, 3>({0, 0, 0});
        }
        xt::xtensor<double, 3> transformed_params({m_valid_size,
                                                   m_param_sets.shape()[1] - 2,
                                                   m_param_sets.shape()[2]});

        // Extract all but the last two parameters for each suggestion
        for (SizeType i = 0; i < m_valid_size; ++i) {
            auto view =
                xt::view(m_param_sets, i,
                         xt::range(0, m_param_sets.shape()[1] - 2), xt::all());
            xt::view(transformed_params, i, xt::all(), xt::all()) = view;
        }

        // Call the batch shift function
        // return psr_utils::shift_params_batch(transformed_params, delta_t);
        return transformed_params;
    }

    bool add(const xt::xtensor<double, 2>& param_set,
             const xt::xtensor<FoldType, 2>& fold,
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

    float add_batch(const xt::xtensor<double, 3>& param_sets_batch,
                    const xt::xtensor<FoldType, 3>& folds_batch,
                    const std::vector<float>& scores_batch,
                    const xt::xtensor<SizeType, 2>& backtracks_batch,
                    float current_threshold) {
        const auto num_to_add = scores_batch.size();
        if (num_to_add == 0) {
            return current_threshold;
        }
        auto effective_threshold = current_threshold;

        // Create initial mask for scores >= threshold
        std::vector<SizeType> candidate_indices;
        candidate_indices.reserve(num_to_add);

        auto update_candidates = [&]() {
            candidate_indices.clear();
            for (SizeType i = 0; i < num_to_add; ++i) {
                if (scores_batch[i] >= effective_threshold) {
                    candidate_indices.push_back(i);
                }
            }
        };

        update_candidates();

        while (!candidate_indices.empty()) {
            const auto space_left = m_size - m_valid_size;
            if (space_left == 0) {
                // Buffer is full, try to trim
                const auto new_threshold_from_trim = trim_threshold();
                effective_threshold =
                    std::max(effective_threshold, new_threshold_from_trim);
                // Re-filter after new threshold
                update_candidates();
                continue; // Try again with new threshold
            }

            const auto n_to_add =
                std::min(candidate_indices.size(), space_left);
            const auto pos = m_valid_size;

            // Batched assignment - much more efficient than individual adds
            for (SizeType i = 0; i < n_to_add; ++i) {
                const auto src_idx = candidate_indices[i];
                const auto dst_idx = pos + i;

                xt::view(m_param_sets, dst_idx, xt::all(), xt::all()) =
                    xt::view(param_sets_batch, src_idx, xt::all(), xt::all());
                xt::view(m_folds, dst_idx, xt::all(), xt::all()) =
                    xt::view(folds_batch, src_idx, xt::all(), xt::all());
                m_scores[dst_idx] = scores_batch[src_idx];
                xt::view(m_backtracks, dst_idx, xt::all()) =
                    xt::view(backtracks_batch, src_idx, xt::all());
            }
            m_valid_size += n_to_add;

            // Remove added candidates from the list
            candidate_indices.erase(candidate_indices.begin(),
                                    candidate_indices.begin() +
                                        static_cast<IndexType>(n_to_add));
        }
        return effective_threshold;
    }

    float trim_threshold() {
        if (m_valid_size == 0) {
            return 0.0F;
        }
        // Compute median score
        const float threshold = get_score_median();

        // Create a boolean mask for scores >= threshold
        std::vector<bool> indices(m_valid_size, false);
        for (SizeType i = 0; i < m_valid_size; ++i) {
            indices[i] = m_scores[i] >= threshold;
        }

        keep(indices);
        return threshold;
    }

    void trim_repeats() {
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

    float trim_repeats_threshold() {
        if (m_valid_size == 0) {
            return 0.0F;
        }

        // Calculate median score
        const float threshold = get_score_median();
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

    // Helper method to create data for new SuggestionStruct
    std::tuple<xt::xtensor<double, 3>,
               xt::xtensor<FoldType, 3>,
               std::vector<float>,
               xt::xtensor<SizeType, 2>>
    create_new_data(SizeType max_sugg) const {
        // Create empty tensors with the same shapes but different first
        // dimension
        xt::xtensor<double, 3> param_sets(
            {max_sugg, m_param_sets.shape()[1], m_param_sets.shape()[2]}, 0.0);
        xt::xtensor<FoldType, 3> folds(
            {max_sugg, m_folds.shape()[1], m_folds.shape()[2]});

        // Initialize folds based on type
        if constexpr (std::is_same_v<FoldType, std::complex<float>>) {
            folds.fill(std::complex<float>(0.0F, 0.0F));
        } else {
            folds.fill(FoldType{});
        }

        std::vector<float> scores(max_sugg, 0.0F);
        xt::xtensor<SizeType, 2> backtracks({max_sugg, m_backtracks.shape()[1]},
                                            0);
        return std::make_tuple(std::move(param_sets), std::move(folds),
                               std::move(scores), std::move(backtracks));
    }

    // Helper method to create trimmed data
    std::tuple<xt::xtensor<double, 3>,
               xt::xtensor<FoldType, 3>,
               std::vector<float>,
               xt::xtensor<SizeType, 2>>
    create_trimmed_data() const {
        // Return a new SuggestionStruct with only the valid entries
        if (m_valid_size == 0) {
            return create_new_data(0);
        }

        // Create views of the valid portions
        auto param_view = xt::view(m_param_sets, xt::range(0, m_valid_size),
                                   xt::all(), xt::all());
        auto folds_view =
            xt::view(m_folds, xt::range(0, m_valid_size), xt::all(), xt::all());
        auto backtracks_view =
            xt::view(m_backtracks, xt::range(0, m_valid_size), xt::all());

        // Evaluate views to create actual tensors
        xt::xtensor<double, 3> param_sets   = xt::eval(param_view);
        xt::xtensor<FoldType, 3> folds      = xt::eval(folds_view);
        xt::xtensor<SizeType, 2> backtracks = xt::eval(backtracks_view);

        std::vector<float> scores(m_scores.begin(),
                                  m_scores.begin() +
                                      static_cast<IndexType>(m_valid_size));

        return std::make_tuple(std::move(param_sets), std::move(folds),
                               std::move(scores), std::move(backtracks));
    }

private:
    xt::xtensor<double, 3> m_param_sets;   // Shape: (nsuggestions, nparams, 2)
    xt::xtensor<FoldType, 3> m_folds;      // Shape: (nsuggestions, nbins, 2)
    std::vector<float> m_scores;           // Shape: (nsuggestions)
    xt::xtensor<SizeType, 2> m_backtracks; // Shape: (nsuggestions, 2 + nparams)
    SizeType m_valid_size;
    SizeType m_size;

    // Optimized in-place update
    void keep(const std::vector<bool>& indices) {
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

}; // End SuggestionStruct::Impl definition

// Public interface implementation
template <typename FoldType>
SuggestionStruct<FoldType>::SuggestionStruct(
    xt::xtensor<double, 3> param_sets,
    xt::xtensor<FoldType, 3> folds,
    std::vector<float> scores,
    xt::xtensor<SizeType, 2> backtracks)
    : m_impl(std::make_unique<Impl>(std::move(param_sets),
                                    std::move(folds),
                                    std::move(scores),
                                    std::move(backtracks))) {}
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
const std::vector<float>&
SuggestionStruct<FoldType>::get_scores() const noexcept {
    return m_impl->get_scores();
}
template <typename FoldType>
const xt::xtensor<SizeType, 2>&
SuggestionStruct<FoldType>::get_backtracks() const noexcept {
    return m_impl->get_backtracks();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_valid_size() const noexcept {
    return m_impl->get_valid_size();
}
template <typename FoldType>
SizeType SuggestionStruct<FoldType>::get_size() const noexcept {
    return m_impl->get_size();
}
template <typename FoldType>
size_t SuggestionStruct<FoldType>::get_nparams() const noexcept {
    return m_impl->get_nparams();
}
template <typename FoldType>
float SuggestionStruct<FoldType>::get_size_lb() const noexcept {
    return m_impl->get_size_lb();
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
void SuggestionStruct<FoldType>::set_valid_size(SizeType valid_size) noexcept {
    m_impl->set_valid_size(valid_size);
}

// Methods that return SuggestionStruct - implemented at the outer level
template <typename FoldType>
SuggestionStruct<FoldType>
SuggestionStruct<FoldType>::get_new(SizeType max_sugg) const {
    auto [param_sets, folds, scores, backtracks] =
        m_impl->create_new_data(max_sugg);
    auto result = SuggestionStruct(std::move(param_sets), std::move(folds),
                                   std::move(scores), std::move(backtracks));
    // Set valid_size to 0 since this is a new empty struct
    result.set_valid_size(0);
    return result;
}

template <typename FoldType>
SuggestionStruct<FoldType> SuggestionStruct<FoldType>::trim_empty() const {
    auto [param_sets, folds, scores, backtracks] =
        m_impl->create_trimmed_data();
    return {std::move(param_sets), std::move(folds), std::move(scores),
            std::move(backtracks)};
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
float SuggestionStruct<FoldType>::add_batch(
    const xt::xtensor<double, 3>& param_sets_batch,
    const xt::xtensor<FoldType, 3>& folds_batch,
    const std::vector<float>& scores_batch,
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