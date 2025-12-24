#include "loki/core/dynamic.hpp"

#include <algorithm>
#include <span>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/algorithms/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/detection/score.hpp"
#include "loki/kernels.hpp"
#include "loki/search/configs.hpp"

namespace loki::core {

// CRTP Base class implementation
template <SupportedFoldType FoldType, typename Derived>
BasePruneDPFuncts<FoldType, Derived>::BasePruneDPFuncts(
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
      m_boxcar_widths_cache(m_cfg.get_scoring_widths(), m_cfg.get_nbins()) {
    const plans::FFAPlan<FoldType> ffa_plan(m_cfg);
    m_branching_pattern = ffa_plan.get_branching_pattern("taylor");
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

template <SupportedFoldType FoldType, typename Derived>
std::span<const FoldType>
BasePruneDPFuncts<FoldType, Derived>::load(std::span<const FoldType> ffa_fold,
                                           SizeType seg_idx) const {
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

template <SupportedFoldType FoldType, typename Derived>
void BasePruneDPFuncts<FoldType, Derived>::score(
    std::span<const FoldType> batch_folds,
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

template <SupportedFoldType FoldType, typename Derived>
void BasePruneDPFuncts<FoldType, Derived>::load_shift_add(
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

template <SupportedFoldType FoldType, typename Derived>
void BasePruneDPFuncts<FoldType, Derived>::pack(
    std::span<const FoldType> data, std::span<FoldType> out) const noexcept {
    // Placeholder for future implementation
    std::copy(data.begin(), data.end(), out.begin());
}

template <SupportedFoldType FoldType, typename Derived>
std::vector<double> BasePruneDPFuncts<FoldType, Derived>::get_transform_matrix(
    std::pair<double, double> /*coord_cur*/,
    std::pair<double, double> /*coord_prev*/) const {
    // Return identity matrix for Taylor variant
    // Size should match parameter dimensions (typically 4x4 for poly_order=4)
    const SizeType size = m_cfg.get_prune_poly_order() + 1;
    std::vector<double> identity(size * size, 0.0);
    for (SizeType i = 0; i < size; ++i) {
        identity[(i * size) + i] = 1.0;
    }
    return identity;
}

template <SupportedFoldType FoldType, typename Derived>
SizeType BasePruneDPFuncts<FoldType, Derived>::validate(
    std::span<double> /*leaves_batch*/,
    std::span<SizeType> /*leaves_origins*/,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves,
    SizeType /*n_params*/) const {
    return n_leaves;
}

template <SupportedFoldType FoldType, typename Derived>
std::tuple<std::vector<double>, std::vector<double>, double>
BasePruneDPFuncts<FoldType, Derived>::get_validation_params(
    std::pair<double, double> /*coord_add*/) const {
    // Return empty validation parameters for Taylor variant
    std::vector<double> empty_arr(0);
    return std::make_tuple(empty_arr, empty_arr, 0.0);
}

template <SupportedFoldType FoldType, typename Derived>
SizeType BasePruneDPFuncts<FoldType, Derived>::get_branch_max() const noexcept {
    const auto max_val = *std::ranges::max_element(m_branching_pattern);
    return static_cast<SizeType>(std::ceil(max_val * 2));
}

template <SupportedFoldType FoldType, typename Derived>
std::vector<double>
BasePruneDPFuncts<FoldType, Derived>::get_branching_pattern() const noexcept {
    return m_branching_pattern;
}

// Intermediate implementation for Taylor basis
template <SupportedFoldType FoldType, typename Derived>
void BaseTaylorPruneDPFuncts<FoldType, Derived>::suggest(
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
            auto folds_t    = std::span<float>(this->m_batch_folds_buffer)
                               .first(nfft * this->m_cfg.get_nbins());
            this->m_irfft_executor->execute(folds, folds_t,
                                            static_cast<int>(nfft));
            detection::snr_boxcar_batch(folds_t, out, n_batch, cache);
        };
        poly_taylor_suggest<FoldType>(
            fold_segment, coord_init, this->m_param_arr, this->m_dparams,
            this->m_cfg.get_prune_poly_order(), this->m_cfg.get_nbins_f(),
            scoring_func, this->m_boxcar_widths_cache, sugg_tree);
    } else {
        scoring_func = [](std::span<const FoldType> folds, std::span<float> out,
                          SizeType n_batch,
                          detection::BoxcarWidthsCache& cache) {
            detection::snr_boxcar_batch(folds, out, n_batch, cache);
        };
        poly_taylor_suggest<FoldType>(
            fold_segment, coord_init, this->m_param_arr, this->m_dparams,
            this->m_cfg.get_prune_poly_order(), this->m_cfg.get_nbins(),
            scoring_func, this->m_boxcar_widths_cache, sugg_tree);
    }
}

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldType FoldType>
PrunePolyTaylorDPFuncts<FoldType>::PrunePolyTaylorDPFuncts(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size)
    : Base(param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size) {
}

template <SupportedFoldType FoldType>
std::vector<SizeType> PrunePolyTaylorDPFuncts<FoldType>::branch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    std::span<double> leaves_branch_batch,
    SizeType n_batch,
    SizeType n_params) const {
    return poly_taylor_branch_batch(
        leaves_batch, coord_cur, leaves_branch_batch, n_batch, n_params,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        this->m_cfg.get_param_limits(), this->get_branch_max());
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<SizeType> param_idx_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType n_leaves,
    SizeType n_params) const {
    kPolyResolveFuncs[this->m_cfg.get_prune_poly_order() - 2](
        leaves_batch, coord_add, coord_cur, coord_init, this->m_param_arr,
        param_idx_flat_batch, relative_phase_batch, this->m_cfg.get_nbins(),
        n_leaves, n_params, this->m_cfg.get_snap_activation_threshold());
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::transform(
    std::span<double> leaves_batch,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    SizeType n_params) const {
    kPolyTransformFuncs[this->m_cfg.get_prune_poly_order() - 2](
        leaves_batch, coord_next, coord_cur, n_leaves, n_params,
        this->m_cfg.get_use_conservative_grid(),
        this->m_cfg.get_snap_activation_threshold());
}

// Specialized implementation for Circular orbit search in Taylor basis
template <SupportedFoldType FoldType>
PruneCircTaylorDPFuncts<FoldType>::PruneCircTaylorDPFuncts(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size)
    : Base(param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size) {
}

template <SupportedFoldType FoldType>
std::vector<SizeType> PruneCircTaylorDPFuncts<FoldType>::branch(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    std::span<double> leaves_branch_batch,
    SizeType n_batch,
    SizeType n_params) const {
    return poly_taylor_branch_circular_batch(
        leaves_batch, coord_cur, leaves_branch_batch, n_batch, n_params,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        this->m_cfg.get_param_limits(), this->get_branch_max(),
        this->m_cfg.get_snap_activation_threshold());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_batch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<SizeType> param_idx_flat_batch,
    std::span<float> relative_phase_batch,
    SizeType n_leaves,
    SizeType n_params) const {
    poly_taylor_resolve_circular_batch(
        leaves_batch, coord_add, coord_cur, coord_init, this->m_param_arr,
        param_idx_flat_batch, relative_phase_batch, this->m_cfg.get_nbins(),
        n_leaves, n_params, this->m_cfg.get_snap_activation_threshold());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::transform(
    std::span<double> leaves_batch,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    SizeType n_params) const {
    poly_taylor_transform_circular_batch(
        leaves_batch, coord_next, coord_cur, n_leaves, n_params,
        this->m_cfg.get_use_conservative_grid(),
        this->m_cfg.get_snap_activation_threshold());
}

template <SupportedFoldType FoldType>
SizeType PruneCircTaylorDPFuncts<FoldType>::validate(
    std::span<double> leaves_batch,
    std::span<SizeType> leaves_origins,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves,
    SizeType n_params) const {
    return poly_taylor_validate_circular_batch(
        leaves_batch, leaves_origins, n_leaves, n_params,
        this->m_cfg.get_p_orb_min(),
        this->m_cfg.get_snap_activation_threshold());
}

// Factory function to create the correct implementation based on the kind
template <SupportedFoldType FoldType>
std::unique_ptr<PruneDPFuncts<FoldType>>
create_prune_dp_functs(std::string_view kind,
                       std::span<const std::vector<double>> param_arr,
                       std::span<const double> dparams,
                       SizeType nseg_ffa,
                       double tseg_ffa,
                       search::PulsarSearchConfig cfg,
                       SizeType batch_size) {
    const auto n_params = cfg.get_nparams();
    if (kind == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFuncts<FoldType>>(
            param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size);
    }
    if (kind == "taylor" && n_params == 5) {
        return std::make_unique<PruneCircTaylorDPFuncts<FoldType>>(
            param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size);
    }
    throw std::runtime_error(std::format(
        "Unknown pruning kind: '{}'. Valid options: 'taylor', 'circular'",
        kind));
}

// Explicit template instantiations
// Base classes need explicit instantiation for linker
template class BasePruneDPFuncts<float, PrunePolyTaylorDPFuncts<float>>;
template class BasePruneDPFuncts<ComplexType,
                                 PrunePolyTaylorDPFuncts<ComplexType>>;
template class BasePruneDPFuncts<float, PruneCircTaylorDPFuncts<float>>;
template class BasePruneDPFuncts<ComplexType,
                                 PruneCircTaylorDPFuncts<ComplexType>>;

template class BaseTaylorPruneDPFuncts<float, PrunePolyTaylorDPFuncts<float>>;
template class BaseTaylorPruneDPFuncts<ComplexType,
                                       PrunePolyTaylorDPFuncts<ComplexType>>;
template class BaseTaylorPruneDPFuncts<float, PruneCircTaylorDPFuncts<float>>;
template class BaseTaylorPruneDPFuncts<ComplexType,
                                       PruneCircTaylorDPFuncts<ComplexType>>;

// Derived classes
template class PrunePolyTaylorDPFuncts<float>;
template class PrunePolyTaylorDPFuncts<ComplexType>;
template class PruneCircTaylorDPFuncts<float>;
template class PruneCircTaylorDPFuncts<ComplexType>;

// Factory function instantiations
template std::unique_ptr<PruneDPFuncts<float>>
create_prune_dp_functs<float>(std::string_view,
                              std::span<const std::vector<double>>,
                              std::span<const double>,
                              SizeType,
                              double,
                              search::PulsarSearchConfig,
                              SizeType);
template std::unique_ptr<PruneDPFuncts<ComplexType>>
create_prune_dp_functs<ComplexType>(std::string_view,
                                    std::span<const std::vector<double>>,
                                    std::span<const double>,
                                    SizeType,
                                    double,
                                    search::PulsarSearchConfig,
                                    SizeType);

} // namespace loki::core