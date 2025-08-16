#pragma once

#include <algorithm>
#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/suggestions.hpp"

namespace loki::core {

template <typename FoldType> class PruneTaylorDPFuncts {
public:
    PruneTaylorDPFuncts(std::span<const std::vector<double>> param_arr,
                        std::span<const double> dparams,
                        SizeType nseg_ffa,
                        double tseg_ffa,
                        search::PulsarSearchConfig cfg,
                        SizeType batch_size);

    // Core interface methods
    auto load(std::span<const FoldType> ffa_fold, SizeType seg_idx) const
        -> std::span<const FoldType>;

    void resolve(std::span<const double> batch_leaves,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_init,
                 std::span<SizeType> param_idx_flat_batch,
                 std::span<float> relative_phase_batch,
                 SizeType n_leaves,
                 SizeType n_params) const;

    auto branch(std::span<const double> batch_psets,
                std::pair<double, double> coord_cur,
                std::span<double> batch_leaves,
                SizeType n_batch,
                SizeType n_params) const -> std::vector<SizeType>;

    void suggest(std::span<const FoldType> fold_segment,
                 std::pair<double, double> coord_init,
                 utils::SuggestionTree<FoldType>& sugg_tree);

    void score(std::span<const FoldType> batch_folds,
               std::span<float> batch_scores,
               SizeType n_batch) noexcept;

    void pack(std::span<const FoldType> data,
              std::span<FoldType> out) const noexcept;

    void load_shift_add(std::span<const FoldType> batch_folds_suggest,
                        std::span<const SizeType> batch_isuggest,
                        std::span<const FoldType> ffa_fold_segment,
                        std::span<const SizeType> batch_param_idx,
                        std::span<const float> batch_phase_shift,
                        std::span<FoldType> batch_folds_out,
                        SizeType n_batch) noexcept;

    void transform(std::span<const double> batch_leaves,
                   std::pair<double, double> coord_cur,
                   std::span<const double> trans_matrix,
                   SizeType n_leaves,
                   SizeType n_params) const;

    auto get_transform_matrix(std::pair<double, double> coord_cur,
                              std::pair<double, double> coord_prev) const
        -> std::vector<double>;

    auto
    validate(std::span<const double> batch_leaves,
             std::pair<double, double> coord_valid,
             const std::tuple<std::vector<double>, std::vector<double>, double>&
                 validation_params,
             SizeType n_leaves,
             SizeType n_params) const -> SizeType;

    auto get_validation_params(std::pair<double, double> coord_add) const
        -> std::tuple<std::vector<double>, std::vector<double>, double>;

    SizeType get_branch_max() const noexcept {
        return *std::ranges::max_element(m_branching_pattern);
    }

    std::vector<SizeType> get_branching_pattern() const noexcept {
        return m_branching_pattern;
    }

private:
    std::vector<std::vector<double>> m_param_arr;
    std::vector<double> m_dparams;
    SizeType m_nseg_ffa;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
    SizeType m_batch_size;

    std::vector<SizeType> m_branching_pattern;

    // Buffer for shift-add operations
    std::vector<FoldType> m_shift_buffer;
    // Buffer for ComplexType irfft transform
    std::vector<float> m_batch_folds_buffer;
    std::unique_ptr<utils::IrfftExecutor> m_irfft_executor;
    // Cache for snr_boxcar_batch
    detection::BoxcarWidthsCache m_boxcar_widths_cache;

    // Function pointers for resolving different polynomial orders
    using PolyResolveFunc = void (*)(std::span<const double>,
                                     std::pair<double, double>,
                                     std::pair<double, double>,
                                     std::span<const std::vector<double>>,
                                     std::span<SizeType>,
                                     std::span<float>,
                                     SizeType,
                                     SizeType,
                                     SizeType);
    static constexpr std::array<PolyResolveFunc, 3> kPolyResolveFuncs = {
        poly_taylor_resolve_accel_batch, // nparams == 2
        poly_taylor_resolve_jerk_batch,  // nparams == 3
        poly_taylor_resolve_snap_batch   // nparams == 4
    };
};

// Type aliases for convenience
using PruneTaylorDPFunctsFloat   = PruneTaylorDPFuncts<float>;
using PruneTaylorDPFunctsComplex = PruneTaylorDPFuncts<ComplexType>;

} // namespace loki::core