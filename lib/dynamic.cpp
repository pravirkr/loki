#include "loki/core/dynamic.hpp"

#include <algorithm>
#include <span>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/detection/score.hpp"
#include "loki/kernels.hpp"
#include "loki/search/configs.hpp"

namespace loki::core {

template <typename FoldType>
PruneTaylorDPFuncts<FoldType>::PruneTaylorDPFuncts(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size)
    : m_param_arr(param_arr.begin(), param_arr.end()),
      m_dparams(dparams.begin(), dparams.end()),
      m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_boxcar_widths_cache(m_cfg.get_score_widths(), m_cfg.get_nbins()) {
    m_branching_pattern = poly_taylor_branching_pattern(
        m_param_arr, m_dparams, m_cfg.get_param_limits(), m_nseg_ffa,
        m_tseg_ffa, m_cfg.get_nbins(), m_cfg.get_tol_bins());
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        m_irfft_executor =
            std::make_unique<utils::IrfftExecutor>(m_cfg.get_nbins());
        m_shift_buffer.resize(1); // Not needed for complex
        const auto max_batch_size = m_batch_size * get_branch_max();
        m_batch_folds_buffer.resize(max_batch_size * 2 * m_cfg.get_nbins());
    } else {
        m_shift_buffer.resize(2 * m_cfg.get_nbins());
        m_batch_folds_buffer.resize(1); // Not needed for float
    }
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::load(std::span<const FoldType> ffa_fold,
                                         SizeType seg_idx) const
    -> std::span<const FoldType> {
    SizeType n_param_sets = 1;
    for (const auto& arr : m_param_arr) {
        n_param_sets *= arr.size();
    }
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        return ffa_fold.subspan(seg_idx * n_param_sets * 2 *
                                    m_cfg.get_nbins_f(),
                                n_param_sets * 2 * m_cfg.get_nbins_f());
    } else {
        return ffa_fold.subspan(seg_idx * n_param_sets * 2 * m_cfg.get_nbins(),
                                n_param_sets * 2 * m_cfg.get_nbins());
    }
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> batch_leaves,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_init,
    std::span<SizeType> param_idx_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType n_leaves,
    SizeType n_params) const {
    kPolyResolveFuncs[m_cfg.get_prune_poly_order() - 2](
        batch_leaves, coord_add, coord_init, m_param_arr, param_idx_flat_batch,
        relative_phase_batch, m_cfg.get_nbins(), n_leaves, n_params);
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::branch(std::span<const double> batch_psets,
                                           std::pair<double, double> coord_cur,
                                           std::span<double> batch_leaves,
                                           SizeType n_batch,
                                           SizeType n_params) const
    -> std::vector<SizeType> {

    return poly_taylor_branch_batch(
        batch_psets, coord_cur, batch_leaves, n_batch, n_params,
        m_cfg.get_nbins(), m_cfg.get_tol_bins(), m_cfg.get_prune_poly_order(),
        m_cfg.get_param_limits(), get_branch_max());
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::suggest(
    std::span<const FoldType> fold_segment,
    std::pair<double, double> coord_init,
    utils::SuggestionTree<FoldType>& sugg_tree) {

    // Create scoring function based on FoldType
    detection::ScoringFunction<FoldType> scoring_func;
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        scoring_func = [this](std::span<const FoldType> folds,
                              std::span<float> out, SizeType n_batch,
                              detection::BoxcarWidthsCache& cache) {
            const auto nfft = 2 * n_batch;
            auto folds_t    = std::span<float>(m_batch_folds_buffer)
                               .first(nfft * m_cfg.get_nbins());
            m_irfft_executor->execute(folds, folds_t, static_cast<int>(nfft));
            detection::snr_boxcar_batch(folds_t, out, n_batch, cache);
        };
        poly_taylor_suggest<FoldType>(fold_segment, coord_init, m_param_arr,
                                      m_dparams, m_cfg.get_prune_poly_order(),
                                      m_cfg.get_nbins_f(), scoring_func,
                                      m_boxcar_widths_cache, sugg_tree);
    } else {
        scoring_func = [](std::span<const FoldType> folds, std::span<float> out,
                          SizeType n_batch,
                          detection::BoxcarWidthsCache& cache) {
            detection::snr_boxcar_batch(folds, out, n_batch, cache);
        };
        poly_taylor_suggest<FoldType>(fold_segment, coord_init, m_param_arr,
                                      m_dparams, m_cfg.get_prune_poly_order(),
                                      m_cfg.get_nbins(), scoring_func,
                                      m_boxcar_widths_cache, sugg_tree);
    }
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::score(std::span<const FoldType> batch_folds,
                                          std::span<float> batch_scores,
                                          SizeType n_batch) noexcept {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        const auto nfft = 2 * n_batch;
        auto folds_t    = std::span<float>(m_batch_folds_buffer)
                           .first(nfft * m_cfg.get_nbins());
        m_irfft_executor->execute(batch_folds, folds_t, static_cast<int>(nfft));
        detection::snr_boxcar_batch(folds_t, batch_scores, n_batch,
                                    m_boxcar_widths_cache);
    } else {
        detection::snr_boxcar_batch(batch_folds, batch_scores, n_batch,
                                    m_boxcar_widths_cache);
    }
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::load_shift_add(
    std::span<const FoldType> batch_folds_suggest,
    std::span<const SizeType> batch_isuggest,
    std::span<const FoldType> ffa_fold_segment,
    std::span<const SizeType> batch_param_idx,
    std::span<const float> batch_phase_shift,
    std::span<FoldType> batch_folds_out,
    SizeType n_batch) noexcept {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        kernels::shift_add_complex_recurrence_batch(
            batch_folds_suggest.data(), batch_isuggest.data(),
            ffa_fold_segment.data(), batch_param_idx.data(),
            batch_phase_shift.data(), batch_folds_out.data(),
            m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_batch);
    } else {
        kernels::shift_add_buffer_batch(
            batch_folds_suggest.data(), batch_isuggest.data(),
            ffa_fold_segment.data(), batch_param_idx.data(),
            batch_phase_shift.data(), batch_folds_out.data(),
            m_shift_buffer.data(), m_cfg.get_nbins(), n_batch);
    }
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::pack(
    std::span<const FoldType> data, std::span<FoldType> out) const noexcept {
    // Placeholder for future implementation
    std::copy(data.begin(), data.end(), out.begin());
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::transform(
    std::span<const double> /*batch_leaves*/,
    std::pair<double, double> /*coord_cur*/,
    std::span<const double> /*trans_matrix*/,
    SizeType /*n_leaves*/,
    SizeType /*n_params*/) const {
    // Taylor variant doesn't transform - just return original leaf
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::get_transform_matrix(
    std::pair<double, double> /*coord_cur*/,
    std::pair<double, double> /*coord_prev*/) const -> std::vector<double> {
    // Return identity matrix for Taylor variant
    // Size should match parameter dimensions (typically 4x4 for poly_order=4)
    const SizeType size = m_cfg.get_prune_poly_order() + 1;
    std::vector<double> identity(size * size, 0.0);
    for (SizeType i = 0; i < size; ++i) {
        identity[(i * size) + i] = 1.0;
    }
    return identity;
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::validate(
    std::span<const double> /*batch_leaves*/,
    std::pair<double, double> /*coord_valid*/,
    const std::tuple<std::vector<double>, std::vector<double>, double>&
    /*validation_params*/,
    SizeType n_leaves,
    SizeType /*n_params*/) const -> SizeType {
    // Taylor variant doesn't filter - return all leaves
    return n_leaves;
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::get_validation_params(
    std::pair<double, double> /*coord_add*/) const
    -> std::tuple<std::vector<double>, std::vector<double>, double> {
    // Return empty validation parameters for Taylor variant
    std::vector<double> empty_arr(0);
    return std::make_tuple(empty_arr, empty_arr, 0.0);
}

// Explicit template instantiations
template class PruneTaylorDPFuncts<float>;
template class PruneTaylorDPFuncts<ComplexType>;

} // namespace loki::core