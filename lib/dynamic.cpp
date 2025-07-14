#include "loki/core/dynamic.hpp"

#include <algorithm>
#include <span>
#include <utility>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "loki/core/taylor.hpp"
#include "loki/detection/score.hpp"
#include "loki/kernels.hpp"
#include "loki/search/configs.hpp"

namespace loki::core {

template <typename FoldType>
PruneTaylorDPFuncts<FoldType>::PruneTaylorDPFuncts(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    double tseg_ffa,
    search::PulsarSearchConfig cfg)
    : m_param_arr(param_arr.begin(), param_arr.end()),
      m_dparams(dparams.begin(), dparams.end()),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)) {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        m_shift_buffer.resize(1); // Not needed for complex
    } else {
        m_shift_buffer.resize(2 * m_cfg.get_nbins());
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
auto PruneTaylorDPFuncts<FoldType>::resolve(
    const xt::xtensor<double, 3>& batch_leaves,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_init,
    SizeType n_leaves) const
    -> std::tuple<std::vector<SizeType>, std::vector<double>> {
    if (m_cfg.get_prune_poly_order() == 4) {
        return poly_taylor_resolve_snap_batch(batch_leaves, coord_add,
                                              coord_init, m_param_arr,
                                              m_cfg.get_nbins(), n_leaves);
    }
    return poly_taylor_resolve_batch(batch_leaves, coord_add, coord_init,
                                     m_param_arr, m_cfg.get_nbins(), n_leaves);
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::branch(
    const xt::xtensor<double, 3>& param_set_batch,
    std::pair<double, double> coord_cur,
    xt::xtensor<double, 3>& batch_leaves) const -> std::vector<SizeType> {

    return poly_taylor_branch_batch(
        param_set_batch, coord_cur, batch_leaves, m_cfg.get_nbins(),
        m_cfg.get_tol_bins(), m_cfg.get_prune_poly_order(),
        m_cfg.get_param_limits(), m_cfg.get_branch_max());
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::suggest(
    std::span<const FoldType> fold_segment,
    std::pair<double, double> coord_init,
    utils::SuggestionStruct<FoldType>& sugg_struct) const {

    // Create scoring function based on FoldType
    detection::ScoringFunction<FoldType> scoring_func;
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        scoring_func = [](xt::xtensor<FoldType, 3>& folds,
                          std::span<const SizeType> widths,
                          std::span<float> out) {
            detection::snr_boxcar_batch_complex(folds, widths, out);
        };
    } else {
        scoring_func = [](xt::xtensor<FoldType, 3>& folds,
                          std::span<const SizeType> widths,
                          std::span<float> out) {
            detection::snr_boxcar_batch(folds, widths, out);
        };
    }

    poly_taylor_suggest<FoldType>(fold_segment, coord_init, m_param_arr,
                                  m_dparams, m_cfg.get_prune_poly_order(),
                                  m_cfg.get_score_widths(), scoring_func,
                                  sugg_struct);
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::score(
    xt::xtensor<FoldType, 3>& combined_res_batch, std::span<float> out) const {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        detection::snr_boxcar_batch_complex(combined_res_batch,
                                            m_cfg.get_score_widths(), out);
    } else {
        detection::snr_boxcar_batch(combined_res_batch,
                                    m_cfg.get_score_widths(), out);
    }
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::load_shift_add(
    const xt::xtensor<FoldType, 3>& folds,
    std::span<const SizeType> batch_isuggest,
    std::span<const FoldType> ffa_fold_segment,
    std::span<const SizeType> batch_param_idx,
    std::span<const double> batch_shift,
    xt::xtensor<FoldType, 3>& out) {
    // Use one of the spans to get the correct batch size
    const auto n_batch = batch_isuggest.size();
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        kernels::shift_add_complex_recurrence_batch(
            folds.data(), batch_isuggest.data(), ffa_fold_segment.data(),
            batch_param_idx.data(), batch_shift.data(), out.data(),
            m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_batch);
    } else {
        // Float version: round shifts to integers properly wrapped
        std::vector<SizeType> batch_shift_rounded(batch_shift.size());
        std::transform(
            batch_shift.begin(), batch_shift.end(), batch_shift_rounded.begin(),
            [nbins = m_cfg.get_nbins()](double shift) {
                auto rounded = static_cast<SizeType>(std::round(shift));
                // Handle wrapping: if rounded equals nbins, wrap to 0
                if (rounded == nbins) {
                    return static_cast<SizeType>(0);
                }
                return rounded;
            });

        kernels::shift_add_buffer_batch(
            folds.data(), batch_isuggest.data(), ffa_fold_segment.data(),
            batch_param_idx.data(), batch_shift_rounded.data(), out.data(),
            m_shift_buffer.data(), m_cfg.get_nbins(), n_batch);
    }
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::pack(
    const xt::xtensor<FoldType, 2>& data) const -> xt::xtensor<FoldType, 1> {
    // Simple flattening - equivalent to Python's pack function
    return xt::flatten(data);
}

template <typename FoldType>
void PruneTaylorDPFuncts<FoldType>::transform(
    xt::xtensor<double, 3>& /*batch_leaves*/,
    std::pair<double, double> /*coord_cur*/,
    const xt::xtensor<double, 2>& /*trans_matrix*/) const {
    // Taylor variant doesn't transform - just return original leaf
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::get_transform_matrix(
    std::pair<double, double> /*coord_cur*/,
    std::pair<double, double> /*coord_prev*/) const -> xt::xtensor<double, 2> {
    // Return identity matrix for Taylor variant
    // Size should match parameter dimensions (typically 4x4 for poly_order=4)
    const SizeType size             = m_cfg.get_prune_poly_order() + 1;
    xt::xtensor<double, 2> identity = xt::zeros<double>({size, size});
    for (SizeType i = 0; i < size; ++i) {
        identity(i, i) = 1.0;
    }
    return identity;
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::validate(
    const xt::xtensor<double, 3>& /*batch_leaves*/,
    std::pair<double, double> /*coord_valid*/,
    const std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, double>&
    /*validation_params*/,
    SizeType n_leaves) const -> SizeType {
    // Taylor variant doesn't filter - return all leaves
    return n_leaves;
}

template <typename FoldType>
auto PruneTaylorDPFuncts<FoldType>::get_validation_params(
    std::pair<double, double> /*coord_add*/) const
    -> std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, double> {
    // Return empty validation parameters for Taylor variant
    xt::xtensor<double, 1> empty_arr = xt::zeros<double>({0});
    return std::make_tuple(empty_arr, empty_arr, 0.0);
}

// Explicit template instantiations
template class PruneTaylorDPFuncts<float>;
template class PruneTaylorDPFuncts<ComplexType>;

} // namespace loki::core