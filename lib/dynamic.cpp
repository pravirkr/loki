#include "loki/core/dynamic.hpp"

#include <algorithm>
#include <span>
#include <utility>

#include <spdlog/spdlog.h>

#include "loki/common/types.hpp"
#include "loki/core/circular.hpp"
#include "loki/core/taylor.hpp"
#include "loki/detection/score.hpp"
#include "loki/exceptions.hpp"
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
    SizeType batch_size,
    SizeType branch_max)
    : m_param_arr(param_arr.begin(), param_arr.end()),
      m_dparams(dparams.begin(), dparams.end()),
      m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_branch_max(branch_max),
      m_boxcar_widths_cache(m_cfg.get_scoring_widths(), m_cfg.get_nbins()) {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        m_irfft_executor =
            std::make_unique<utils::IrfftExecutor>(m_cfg.get_nbins());
        m_scratch_shifts.resize(1); // Not needed for complex
        const auto max_batch_size = m_batch_size * m_branch_max;
        m_scratch_folds.resize(max_batch_size * 2 * m_cfg.get_nbins());
    } else {
        m_scratch_shifts.resize(2 * m_cfg.get_nbins());
        m_scratch_folds.resize(1); // Not needed for float
    }
}

template <SupportedFoldType FoldType, typename Derived>
std::span<const FoldType> BasePruneDPFuncts<FoldType, Derived>::load_segment(
    std::span<const FoldType> ffa_fold, SizeType seg_idx) const {
    SizeType n_coords = 1;
    for (const auto& arr : m_param_arr) {
        n_coords *= arr.size();
    }
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        return ffa_fold.subspan(seg_idx * n_coords * 2 * nbins_f,
                                n_coords * 2 * nbins_f);
    } else {
        return ffa_fold.subspan(seg_idx * n_coords * 2 * nbins,
                                n_coords * 2 * nbins);
    }
}

template <SupportedFoldType FoldType, typename Derived>
SizeType BasePruneDPFuncts<FoldType, Derived>::validate(
    std::span<double> /*leaves_branch*/,
    std::span<SizeType> /*leaves_origins*/,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves) const {
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
void BasePruneDPFuncts<FoldType, Derived>::shift_add(
    std::span<const FoldType> folds_tree,
    std::span<const SizeType> indices_tree,
    std::span<const FoldType> folds_ffa,
    std::span<const SizeType> indices_ffa,
    std::span<const float> phase_shift,
    std::span<FoldType> folds_out,
    SizeType n_leaves) noexcept {
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        kernels::shift_add_complex_recurrence_batch(
            folds_tree.data(), indices_tree.data(), folds_ffa.data(),
            indices_ffa.data(), phase_shift.data(), folds_out.data(),
            m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_leaves);
    } else {
        kernels::shift_add_buffer_batch(
            folds_tree.data(), indices_tree.data(), folds_ffa.data(),
            indices_ffa.data(), phase_shift.data(), folds_out.data(),
            m_scratch_shifts.data(), m_cfg.get_nbins(), n_leaves);
    }
}

template <SupportedFoldType FoldType, typename Derived>
void BasePruneDPFuncts<FoldType, Derived>::score(
    std::span<const FoldType> folds_tree,
    std::span<float> scores_tree,
    SizeType n_leaves) noexcept {
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        // Ensure exact span for irfft transform
        const auto nfft = 2 * n_leaves;
        auto folds_span =
            std::span<const FoldType>(folds_tree).first(n_leaves * 2 * nbins_f);
        auto folds_t_span =
            std::span<float>(m_scratch_folds).first(nfft * nbins);
        m_irfft_executor->execute(folds_span, folds_t_span,
                                  static_cast<int>(nfft));
        detection::snr_boxcar_batch(folds_t_span, scores_tree, n_leaves, nbins,
                                    m_boxcar_widths_cache);
    } else {
        detection::snr_boxcar_batch(folds_tree, scores_tree, n_leaves, nbins,
                                    m_boxcar_widths_cache);
    }
}

template <SupportedFoldType FoldType, typename Derived>
std::vector<double> BasePruneDPFuncts<FoldType, Derived>::get_transform_matrix(
    std::pair<double, double> /*coord_cur*/,
    std::pair<double, double> /*coord_prev*/) const {
    // Return identity matrix for Taylor variant
    // Size should match parameter dimensions (typically 4x4 for poly_order=4)
    const SizeType size = m_cfg.get_nparams() + 1;
    std::vector<double> identity(size * size, 0.0);
    for (SizeType i = 0; i < size; ++i) {
        identity[(i * size) + i] = 1.0;
    }
    return identity;
}

template <SupportedFoldType FoldType, typename Derived>
void BasePruneDPFuncts<FoldType, Derived>::pack(
    std::span<const FoldType> data, std::span<FoldType> out) const noexcept {
    // Placeholder for future implementation
    std::copy(data.begin(), data.end(), out.begin());
}

// Intermediate implementation for Taylor basis
template <SupportedFoldType FoldType, typename Derived>
void BaseTaylorPruneDPFuncts<FoldType, Derived>::seed(
    std::span<const FoldType> fold_segment,
    std::pair<double, double> coord_init,
    std::span<double> seed_leaves,
    std::span<float> seed_scores) {

    const auto n_leaves =
        poly_taylor_seed(this->m_param_arr, this->m_dparams,
                         this->m_cfg.get_nparams(), coord_init, seed_leaves);
    // Fold segment is (n_leaves, 2, nbins)
    const auto nbins = this->m_cfg.get_nbins();
    error_check::check_greater_equal(seed_scores.size(), n_leaves,
                                     "seed_scores size mismatch");

    // Calculate scores
    if constexpr (std::is_same_v<FoldType, ComplexType>) {
        const auto nfft    = 2 * n_leaves;
        const auto nbins_f = this->m_cfg.get_nbins_f();
        error_check::check_equal(fold_segment.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        auto fold_segment_t =
            std::span<float>(this->m_scratch_folds).first(nfft * nbins);
        this->m_irfft_executor->execute(fold_segment, fold_segment_t,
                                        static_cast<int>(nfft));
        detection::snr_boxcar_batch(fold_segment_t, seed_scores, n_leaves,
                                    nbins, this->m_boxcar_widths_cache);

    } else {
        error_check::check_equal(fold_segment.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        detection::snr_boxcar_batch(fold_segment, seed_scores, n_leaves, nbins,
                                    this->m_boxcar_widths_cache);
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
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_arr,
           dparams,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldType FoldType>
SizeType PrunePolyTaylorDPFuncts<FoldType>::branch(
    std::span<const double> leaves_tree,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    SizeType n_leaves,
    std::span<double> scratch_params,
    std::span<double> scratch_dparams,
    std::span<SizeType> scratch_counts) const {
    return kPolyBranchFuncs[this->m_cfg.get_nparams() - 2](
        leaves_tree, coord_cur, leaves_branch, leaves_origins, n_leaves,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        this->m_cfg.get_param_limits(), this->m_branch_max, scratch_params,
        scratch_dparams, scratch_counts);
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_branch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    SizeType n_leaves) const {
    kPolyResolveFuncs[this->m_cfg.get_nparams() - 2](
        leaves_branch, coord_add, coord_cur, coord_init, this->m_param_arr,
        param_indices, phase_shift, this->m_cfg.get_nbins(), n_leaves);
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::transform(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves) const {
    kPolyTransformFuncs[this->m_cfg.get_nparams() - 2](
        leaves_tree, coord_next, coord_cur, n_leaves,
        this->m_cfg.get_use_conservative_tile());
}

template <SupportedFoldType FoldType>
void PrunePolyTaylorDPFuncts<FoldType>::report(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves) const {
    report_leaves_taylor_batch(leaves_tree, coord_report, n_leaves,
                               this->m_cfg.get_nparams());
}

// Specialized implementation for Circular orbit search in Taylor basis
template <SupportedFoldType FoldType>
PruneCircTaylorDPFuncts<FoldType>::PruneCircTaylorDPFuncts(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_arr,
           dparams,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldType FoldType>
SizeType PruneCircTaylorDPFuncts<FoldType>::branch(
    std::span<const double> leaves_tree,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    SizeType n_leaves,
    std::span<double> scratch_params,
    std::span<double> scratch_dparams,
    std::span<SizeType> scratch_counts) const {
    return circ_taylor_branch_batch(
        leaves_tree, coord_cur, leaves_branch, leaves_origins, n_leaves,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        this->m_cfg.get_param_limits(), this->m_branch_max,
        this->m_cfg.get_minimum_snap_cells(), scratch_params, scratch_dparams,
        scratch_counts);
}

template <SupportedFoldType FoldType>
SizeType PruneCircTaylorDPFuncts<FoldType>::validate(
    std::span<double> leaves_branch,
    std::span<SizeType> leaves_origins,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves) const {
    return circ_taylor_validate_batch(
        leaves_branch, leaves_origins, n_leaves, this->m_cfg.get_p_orb_min(),
        this->m_cfg.get_x_mass_const(), this->m_cfg.get_minimum_snap_cells());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::resolve(
    std::span<const double> leaves_branch,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::span<SizeType> param_indices,
    std::span<float> phase_shift,
    SizeType n_leaves) const {
    circ_taylor_resolve_batch(leaves_branch, coord_add, coord_cur, coord_init,
                              this->m_param_arr, param_indices, phase_shift,
                              this->m_cfg.get_nbins(), n_leaves,
                              this->m_cfg.get_minimum_snap_cells());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::transform(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves) const {
    circ_taylor_transform_batch(leaves_tree, coord_next, coord_cur, n_leaves,
                                this->m_cfg.get_use_conservative_tile(),
                                this->m_cfg.get_minimum_snap_cells());
}

template <SupportedFoldType FoldType>
void PruneCircTaylorDPFuncts<FoldType>::report(
    std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves) const {
    report_leaves_taylor_batch(leaves_tree, coord_report, n_leaves,
                               this->m_cfg.get_nparams());
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
                       SizeType batch_size,
                       SizeType branch_max) {
    const auto n_params = cfg.get_nparams();
    if (kind == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFuncts<FoldType>>(
            param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size,
            branch_max);
    }
    if (kind == "taylor" && n_params == 5) {
        return std::make_unique<PruneCircTaylorDPFuncts<FoldType>>(
            param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size,
            branch_max);
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
                              SizeType,
                              SizeType);
template std::unique_ptr<PruneDPFuncts<ComplexType>>
create_prune_dp_functs<ComplexType>(std::string_view,
                                    std::span<const std::vector<double>>,
                                    std::span<const double>,
                                    SizeType,
                                    double,
                                    search::PulsarSearchConfig,
                                    SizeType,
                                    SizeType);

} // namespace loki::core