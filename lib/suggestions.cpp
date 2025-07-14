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
#include "loki/psr_utils.hpp"

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
    Impl(SizeType max_sugg, SizeType nparams, SizeType nbins)
        : m_max_sugg(max_sugg),
          m_nparams(nparams),
          m_nbins(nbins) {
        // Create empty tensors with the same shapes but different first
        // dimension
        m_param_sets =
            xt::xtensor<double, 3>({m_max_sugg, m_nparams + 2, 2}, 0.0);
        m_folds = xt::xtensor<FoldType, 3>({m_max_sugg, 2, m_nbins});

        // Initialize folds based on type
        if constexpr (std::is_same_v<FoldType, std::complex<float>>) {
            m_folds.fill(std::complex<float>(0.0F, 0.0F));
        } else {
            m_folds.fill(FoldType{});
        }

        m_scores     = std::vector<float>(m_max_sugg, 0.0F);
        m_backtracks = xt::xtensor<SizeType, 2>({m_max_sugg, m_nparams + 2}, 0);
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
    const std::vector<float>& get_scores() const noexcept { return m_scores; }
    const xt::xtensor<SizeType, 2>& get_backtracks() const noexcept {
        return m_backtracks;
    }
    SizeType get_max_sugg() const noexcept { return m_max_sugg; }
    SizeType get_nparams() const noexcept { return m_nparams; }
    SizeType get_nbins() const noexcept { return m_nbins; }
    SizeType get_nsugg() const noexcept { return m_nsugg; }
    float get_nsugg_lb() const noexcept {
        return m_nsugg > 0 ? std::log2(static_cast<float>(m_nsugg)) : 0.0F;
    }
    float get_score_max() const {
        if (m_nsugg == 0) {
            return 0.0F;
        }
        return *std::ranges::max_element(m_scores | std::views::take(m_nsugg));
    }

    float get_score_min() const {
        if (m_nsugg == 0) {
            return 0.0F;
        }
        return *std::ranges::min_element(m_scores | std::views::take(m_nsugg));
    }

    float get_score_median() const {
        if (m_nsugg == 0) {
            return 0.0F;
        }
        auto scores_view = m_scores | std::views::take(m_nsugg);
        std::vector<float> scores_copy(scores_view.begin(), scores_view.end());
        const auto mid = static_cast<IndexType>(m_nsugg) / 2;
        std::nth_element(scores_copy.begin(), scores_copy.begin() + mid,
                         scores_copy.end());
        return scores_copy[mid];
    }

    void set_nsugg(SizeType nsugg) noexcept { m_nsugg = nsugg; }
    void reset() noexcept { m_nsugg = 0; }

    std::tuple<xt::xtensor<double, 2>, xt::xtensor<FoldType, 2>, float>
    get_best() const {
        if (m_nsugg == 0) {
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
        SizeType idx_max = std::ranges::distance(
            m_scores.begin(),
            std::ranges::max_element(m_scores | std::views::take(m_nsugg)));

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
        xt::xtensor<double, 3> transformed_params({m_nsugg, m_nparams, 2});

        // Extract all but the last two parameters for each suggestion
        for (SizeType i = 0; i < m_nsugg; ++i) {
            auto view =
                xt::view(m_param_sets, i, xt::range(0, m_nparams), xt::all());
            xt::view(transformed_params, i, xt::all(), xt::all()) = view;
        }

        // Call the batch shift function
        xt::xtensor<double, 3> trans_params;
        if (m_nparams < 4) {
            auto [trans_params, _] =
                psr_utils::shift_params_batch(transformed_params, delta_t);
        } else if (m_nparams == 4) {
            auto [trans_params, _] = psr_utils::shift_params_circular_batch(
                transformed_params, delta_t);
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
        if (m_nsugg >= m_max_sugg) {
            return false;
        }
        // Copy data to the corresponding position using views
        xt::view(m_param_sets, m_nsugg, xt::all(), xt::all()) = param_set;
        xt::view(m_folds, m_nsugg, xt::all(), xt::all())      = fold;
        m_scores[m_nsugg]                                     = score;
        // Convert backtrack vector to xtensor view and assign
        xt::xtensor<SizeType, 1> bt_tensor = xt::adapt(backtrack);
        xt::view(m_backtracks, m_nsugg, xt::range(0, bt_tensor.size())) =
            bt_tensor;

        ++m_nsugg;
        return true;
    }

    void add_initial(const xt::xtensor<double, 3>& param_sets_batch,
                     const xt::xtensor<FoldType, 3>& folds_batch,
                     const std::vector<float>& scores_batch,
                     const xt::xtensor<SizeType, 2>& backtracks_batch) {
        const auto num_to_add = scores_batch.size();
        if (num_to_add > m_max_sugg) {
            throw std::runtime_error(std::format(
                "SuggestionStruct: Suggestions too large to add: {} > {}",
                num_to_add, m_max_sugg));
        }
        // Copy all data efficiently using views
        for (SizeType i = 0; i < num_to_add; ++i) {
            xt::view(m_param_sets, i, xt::all(), xt::all()) =
                xt::view(param_sets_batch, i, xt::all(), xt::all());
            xt::view(m_folds, i, xt::all(), xt::all()) =
                xt::view(folds_batch, i, xt::all(), xt::all());
            xt::view(m_backtracks, i, xt::all()) =
                xt::view(backtracks_batch, i, xt::all());
            m_scores[i] = scores_batch[i];
        }
        m_nsugg = num_to_add;
    }

    float add_batch(const xt::xtensor<double, 3>& param_sets_batch,
                    const xt::xtensor<FoldType, 3>& folds_batch,
                    std::span<const float> scores_batch,
                    const xt::xtensor<SizeType, 2>& backtracks_batch,
                    float current_threshold) {
        // Always use scores_batch to get the correct batch size
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
            const auto space_left = m_max_sugg - m_nsugg;
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
            const auto pos = m_nsugg;

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
            m_nsugg += n_to_add;

            // Remove added candidates from the list
            candidate_indices.erase(candidate_indices.begin(),
                                    candidate_indices.begin() +
                                        static_cast<IndexType>(n_to_add));
        }
        return effective_threshold;
    }

    float trim_threshold() {
        if (m_nsugg == 0) {
            return 0.0F;
        }
        // Compute median score
        const float threshold = get_score_median();

        // Create a boolean mask for scores >= threshold
        std::vector<bool> indices(m_nsugg, false);
        for (SizeType i = 0; i < m_nsugg; ++i) {
            indices[i] = m_scores[i] >= threshold;
        }

        keep(indices);
        return threshold;
    }

    void trim_repeats() {
        if (m_nsugg == 0) {
            return;
        }

        // Get unique indices
        const auto unique_idx = get_unique_indices_scores(
            xt::view(m_param_sets, xt::range(0, m_nsugg), xt::all(), xt::all()),
            std::span<const float>(m_scores.data(), m_nsugg));

        // Convert indices to boolean mask
        std::vector<bool> idx_bool(m_nsugg, false);
        for (SizeType idx : unique_idx) {
            idx_bool[idx] = true;
        }

        keep(idx_bool);
    }

    float trim_repeats_threshold() {
        if (m_nsugg == 0) {
            return 0.0F;
        }

        // Calculate median score
        const float threshold = get_score_median();
        // Get unique indices
        auto unique_idx = get_unique_indices_scores(
            xt::view(m_param_sets, xt::range(0, m_nsugg), xt::all(), xt::all()),
            std::span<const float>(m_scores.data(), m_nsugg));

        // Create boolean mask for unique indices and scores >= threshold
        std::vector<bool> idx_bool(m_nsugg, false);
        for (SizeType idx : unique_idx) {
            idx_bool[idx] = m_scores[idx] >= threshold;
        }

        keep(idx_bool);
        return threshold;
    }

private:
    xt::xtensor<double, 3> m_param_sets;   // Shape: (nsugg, nparams + 2, 2)
    xt::xtensor<FoldType, 3> m_folds;      // Shape: (nsugg, 2, nbins)
    std::vector<float> m_scores;           // Shape: (nsugg)
    xt::xtensor<SizeType, 2> m_backtracks; // Shape: (nsugg, nparams + 2)

    SizeType m_max_sugg;
    SizeType m_nparams;
    SizeType m_nbins;
    SizeType m_nsugg{};

    // Optimized in-place update
    void keep(const std::vector<bool>& indices) {
        // Count how many elements to keep
        SizeType count = std::count(indices.begin(), indices.end(), true);
        if (count == 0) {
            m_nsugg = 0;
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
        m_nsugg = count;
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
void SuggestionStruct<FoldType>::set_nsugg(SizeType nsugg) noexcept {
    m_impl->set_nsugg(nsugg);
}
template <typename FoldType> void SuggestionStruct<FoldType>::reset() noexcept {
    m_impl->reset();
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