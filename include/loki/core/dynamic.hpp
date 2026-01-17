#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/world_tree.hpp"

namespace loki::core {

// Virtual Interface - for runtime polymorphism
template <SupportedFoldType FoldType> class PruneDPFuncts {
public:
    virtual ~PruneDPFuncts() = default;

    // Delete copy/move for interface
    PruneDPFuncts()                                    = default;
    PruneDPFuncts(const PruneDPFuncts&)                = delete;
    PruneDPFuncts& operator=(const PruneDPFuncts&)     = delete;
    PruneDPFuncts(PruneDPFuncts&&) noexcept            = delete;
    PruneDPFuncts& operator=(PruneDPFuncts&&) noexcept = delete;

    // Core interface methods - all derived classes must implement these
    virtual std::span<const FoldType> load(std::span<const FoldType> ffa_fold,
                                           SizeType seg_idx) const = 0;

    virtual void seed(std::span<const FoldType> fold_segment,
                      std::pair<double, double> coord_init,
                      utils::WorldTree<FoldType>& world_tree) = 0;

    virtual SizeType branch(std::span<const double> leaves_tree,
                            std::pair<double, double> coord_cur,
                            std::pair<double, double> coord_prev,
                            std::span<double> leaves_branch,
                            std::span<SizeType> leaves_origins,
                            SizeType n_leaves,
                            SizeType n_params,
                            std::span<double> scratch_params,
                            std::span<double> scratch_dparams,
                            std::span<SizeType> scratch_counts) const = 0;

    virtual SizeType validate(std::span<double> leaves_batch,
                              std::span<SizeType> leaves_origins,
                              std::pair<double, double> coord_cur,
                              SizeType n_leaves,
                              SizeType) const = 0;

    virtual std::tuple<std::vector<double>, std::vector<double>, double>
    get_validation_params(std::pair<double, double> coord_add) const = 0;

    virtual void resolve(std::span<const double> leaves_batch,
                         std::pair<double, double> coord_add,
                         std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_init,
                         std::span<SizeType> param_idx_flat_batch,
                         std::span<float> relative_phase_batch,
                         SizeType n_leaves,
                         SizeType n_params) const = 0;

    virtual void shift_add(std::span<const FoldType> batch_folds_suggest,
                           std::span<const SizeType> batch_isuggest,
                           std::span<const FoldType> ffa_fold_segment,
                           std::span<const SizeType> batch_param_idx,
                           std::span<const float> batch_phase_shift,
                           std::span<FoldType> batch_folds_out,
                           SizeType n_batch) noexcept = 0;

    virtual void score(std::span<const FoldType> folds,
                       std::span<float> scores,
                       SizeType n_leaves) noexcept = 0;

    virtual void transform(std::span<double> leaves_batch,
                           std::pair<double, double> coord_next,
                           std::pair<double, double> coord_cur,
                           SizeType n_leaves,
                           SizeType n_params) const = 0;

    virtual std::vector<double>
    get_transform_matrix(std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_prev) const = 0;

    virtual void pack(std::span<const FoldType> data,
                      std::span<FoldType> out) const noexcept = 0;

    virtual void report(std::span<double> leaves_tree,
                        std::pair<double, double> coord_report,
                        SizeType n_leaves,
                        SizeType n_params) const = 0;

    virtual SizeType get_branch_max() const noexcept = 0;

    virtual std::vector<double> get_branching_pattern() const noexcept = 0;
};

// CRTP Base class - shared functionality for all derived classes
template <SupportedFoldType FoldType, typename Derived>
class BasePruneDPFuncts : public PruneDPFuncts<FoldType> {
protected:
    // Common members for all derived classes
    std::vector<std::vector<double>> m_param_arr;
    std::vector<double> m_dparams;
    SizeType m_nseg_ffa;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
    SizeType m_batch_size;
    std::vector<double> m_branching_pattern;

    // Buffer for shift-add operations
    std::vector<FoldType> m_shift_buffer;
    // Buffer for ComplexType irfft transform
    std::vector<float> m_batch_folds_buffer;
    std::unique_ptr<utils::IrfftExecutor> m_irfft_executor;
    // Cache for snr_boxcar_batch
    detection::BoxcarWidthsCache m_boxcar_widths_cache;

    // Constructor for all derived classes
    BasePruneDPFuncts(std::span<const std::vector<double>> param_arr,
                      std::span<const double> dparams,
                      SizeType nseg_ffa,
                      double tseg_ffa,
                      search::PulsarSearchConfig cfg,
                      SizeType batch_size);

public:
    // Common implementations shared by all variants
    std::span<const FoldType> load(std::span<const FoldType>,
                                   SizeType) const override;

    SizeType validate(std::span<double> leaves_batch,
                      std::span<SizeType> leaves_origins,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves,
                      SizeType n_params) const override;

    std::tuple<std::vector<double>, std::vector<double>, double>
    get_validation_params(std::pair<double, double> coord_add) const override;

    void shift_add(std::span<const FoldType> batch_folds_suggest,
                   std::span<const SizeType> batch_isuggest,
                   std::span<const FoldType> ffa_fold_segment,
                   std::span<const SizeType> batch_param_idx,
                   std::span<const float> batch_phase_shift,
                   std::span<FoldType> batch_folds_out,
                   SizeType n_batch) noexcept override;

    void score(std::span<const FoldType> folds,
               std::span<float> scores,
               SizeType n_leaves) noexcept override;

    std::vector<double>
    get_transform_matrix(std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_prev) const override;

    void pack(std::span<const FoldType> data,
              std::span<FoldType> out) const noexcept override;

    SizeType get_branch_max() const noexcept override;
    std::vector<double> get_branching_pattern() const noexcept override;
};

// Intermediate base for Taylor-based methods (common seed implementation)
template <SupportedFoldType FoldType, typename Derived>
class BaseTaylorPruneDPFuncts : public BasePruneDPFuncts<FoldType, Derived> {
protected:
    using Base = BasePruneDPFuncts<FoldType, Derived>;

    // Inherit constructor
    using Base::BasePruneDPFuncts;

public:
    // Common seed implementation for all Taylor variants
    void seed(std::span<const FoldType> fold_segment,
              std::pair<double, double> coord_init,
              utils::WorldTree<FoldType>& world_tree) override;
};

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldType FoldType>
class PrunePolyTaylorDPFuncts final
    : public BaseTaylorPruneDPFuncts<FoldType,
                                     PrunePolyTaylorDPFuncts<FoldType>> {
private:
    using Base =
        BaseTaylorPruneDPFuncts<FoldType, PrunePolyTaylorDPFuncts<FoldType>>;

    // Function pointers for branching different polynomial orders
    using PolyBranchFunc = SizeType (*)(std::span<const double>,
                                        std::pair<double, double>,
                                        std::span<double>,
                                        std::span<SizeType>,
                                        SizeType,
                                        SizeType,
                                        SizeType,
                                        double,
                                        const std::vector<ParamLimitType>&,
                                        SizeType,
                                        std::span<double>,
                                        std::span<double>,
                                        std::span<SizeType>);
    static constexpr std::array<PolyBranchFunc, 3> kPolyBranchFuncs = {
        poly_taylor_branch_accel_batch, // nparams == 2
        poly_taylor_branch_jerk_batch,  // nparams == 3
        poly_taylor_branch_snap_batch,  // nparams == 4
    };
    // Function pointers for resolving different polynomial orders
    using PolyResolveFunc = void (*)(std::span<const double>,
                                     std::pair<double, double>,
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
        poly_taylor_resolve_snap_batch,  // nparams == 4
    };

    // Function pointers for Transforming different polynomial orders
    using PolyTransformFunc = void (*)(std::span<double>,
                                       std::pair<double, double>,
                                       std::pair<double, double>,
                                       SizeType,
                                       SizeType,
                                       bool);
    static constexpr std::array<PolyTransformFunc, 3> kPolyTransformFuncs = {
        poly_taylor_transform_accel_batch, // nparams == 2
        poly_taylor_transform_jerk_batch,  // nparams == 3
        poly_taylor_transform_snap_batch,  // nparams == 4
    };

public:
    PrunePolyTaylorDPFuncts(std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size);

    SizeType branch(std::span<const double> leaves_tree,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    SizeType n_leaves,
                    SizeType n_params,
                    std::span<double> scratch_params,
                    std::span<double> scratch_dparams,
                    std::span<SizeType> scratch_counts) const override;

    void resolve(std::span<const double> leaves_batch,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 std::span<SizeType> param_idx_flat_batch,
                 std::span<float> relative_phase_batch,
                 SizeType n_leaves,
                 SizeType n_params) const override;

    void transform(std::span<double> leaves_batch,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   SizeType n_params) const override;

    void report(std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves,
                SizeType n_params) const override;
};

// Specialized implementation for Circular orbit search in Taylor basis
// Use only when nparams == 5
template <SupportedFoldType FoldType>
class PruneCircTaylorDPFuncts final
    : public BaseTaylorPruneDPFuncts<FoldType,
                                     PruneCircTaylorDPFuncts<FoldType>> {
private:
    using Base =
        BaseTaylorPruneDPFuncts<FoldType, PruneCircTaylorDPFuncts<FoldType>>;

public:
    PruneCircTaylorDPFuncts(std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size);

    SizeType branch(std::span<const double> leaves_tree,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    SizeType n_leaves,
                    SizeType n_params,
                    std::span<double> scratch_params,
                    std::span<double> scratch_dparams,
                    std::span<SizeType> scratch_counts) const override;

    SizeType validate(std::span<double> leaves_batch,
                      std::span<SizeType> leaves_origins,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves,
                      SizeType n_params) const override;

    void resolve(std::span<const double> leaves_batch,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 std::span<SizeType> param_idx_flat_batch,
                 std::span<float> relative_phase_batch,
                 SizeType n_leaves,
                 SizeType n_params) const override;

    void transform(std::span<double> leaves_batch,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   SizeType n_params) const override;

    void report(std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves,
                SizeType n_params) const override;
};

// Factory function to create the correct implementation based on the kind
template <SupportedFoldType FoldType>
std::unique_ptr<PruneDPFuncts<FoldType>>
create_prune_dp_functs(std::string_view kind,
                       std::span<const std::vector<double>> param_arr,
                       std::span<const double> dparams,
                       SizeType nseg_ffa,
                       double tseg_ffa,
                       search::PulsarSearchConfig cfg,
                       SizeType batch_size);

// Type aliases for convenience
using PrunePolyTaylorDPFunctsFloat   = PrunePolyTaylorDPFuncts<float>;
using PrunePolyTaylorDPFunctsComplex = PrunePolyTaylorDPFuncts<ComplexType>;
using PruneCircTaylorDPFunctsFloat   = PruneCircTaylorDPFuncts<float>;
using PruneCircTaylorDPFunctsComplex = PruneCircTaylorDPFuncts<ComplexType>;

/*
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

    void resolve(std::span<const double> leaves_batch,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 std::span<SizeType> param_idx_flat_batch,
                 std::span<float> relative_phase_batch,
                 SizeType n_leaves,
                 SizeType n_params) const;

    auto branch(std::span<const double> leaves_batch,
                std::pair<double, double> coord_cur,
                std::pair<double, double> coord_prev,
                std::span<double> leaves_branch_batch,
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

    void transform(std::span<double> leaves_batch,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   SizeType n_params) const;

    auto get_transform_matrix(std::pair<double, double> coord_cur,
                              std::pair<double, double> coord_prev) const
        -> std::vector<double>;

    auto validate(std::span<double> leaves_batch,
                  std::span<SizeType> leaves_origins,
                  std::pair<double, double> coord_cur,
                  SizeType n_leaves,
                  SizeType n_params) const -> SizeType;

    auto get_validation_params(std::pair<double, double> coord_add) const
        -> std::tuple<std::vector<double>, std::vector<double>, double>;

    SizeType get_branch_max() const noexcept {
        // Take 2x to be safe
        const auto max_val = *std::ranges::max_element(m_branching_pattern);
        return static_cast<SizeType>(std::ceil(max_val * 2));
    }

    std::vector<double> get_branching_pattern() const noexcept {
        return m_branching_pattern;
    }

private:
    std::vector<std::vector<double>> m_param_arr;
    std::vector<double> m_dparams;
    SizeType m_nseg_ffa;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
    SizeType m_batch_size;

    std::vector<double> m_branching_pattern;

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
                                     std::pair<double, double>,
                                     std::span<const std::vector<double>>,
                                     std::span<SizeType>,
                                     std::span<float>,
                                     SizeType,
                                     SizeType,
                                     SizeType,
                                     double);
    static constexpr std::array<PolyResolveFunc, 4> kPolyResolveFuncs = {
        poly_taylor_resolve_accel_batch,   // nparams == 2
        poly_taylor_resolve_jerk_batch,    // nparams == 3
        poly_taylor_resolve_snap_batch,    // nparams == 4
        poly_taylor_resolve_circular_batch // nparams == 5
    };

    // Function pointers for Transforming different polynomial orders
    using PolyTransformFunc = void (*)(std::span<double>,
                                       std::pair<double, double>,
                                       std::pair<double, double>,
                                       SizeType,
                                       SizeType,
                                       bool,
                                       double);
    static constexpr std::array<PolyTransformFunc, 4> kPolyTransformFuncs = {
        poly_taylor_transform_accel_batch,   // nparams == 2
        poly_taylor_transform_jerk_batch,    // nparams == 3
        poly_taylor_transform_snap_batch,    // nparams == 4
        poly_taylor_transform_circular_batch // nparams == 5
    };
};
*/

} // namespace loki::core