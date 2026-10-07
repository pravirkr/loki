#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/detection/kadane.hpp"
#include "lib/utils/fft_impl.hpp"
#include "lib/utils/workspace_impl.hpp"

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
    virtual std::span<const FoldType>
    load_segment(std::span<const FoldType> ffa_fold,
                 SizeType seg_idx) const = 0;

    virtual void seed(std::span<const FoldType> fold_segment,
                      std::span<double> seed_leaves,
                      std::span<float> seed_scores,
                      std::pair<double, double> coord_init) = 0;

    virtual SizeType branch(std::span<double> leaves_tree,
                            std::span<double> leaves_branch,
                            std::span<SizeType> leaves_origins,
                            std::pair<double, double> coord_cur,
                            std::pair<double, double> coord_prev,
                            SizeType n_leaves,
                            memory::BranchingWorkspace& branch_ws) const = 0;

    virtual SizeType validate(std::span<double> leaves_branch,
                              std::span<SizeType> leaves_origins,
                              std::pair<double, double> coord_cur,
                              SizeType n_leaves) const = 0;

    virtual std::tuple<std::vector<double>, std::vector<double>, double>
    get_validation_params(std::pair<double, double> coord_add) const = 0;

    virtual void resolve(std::span<const double> leaves_branch,
                         std::span<SizeType> param_indices,
                         std::span<float> phase_shift,
                         std::pair<double, double> coord_add,
                         std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_init,
                         SizeType n_leaves) const = 0;

    virtual void shift_add(std::span<const FoldType> folds_tree,
                           std::span<const SizeType> indices_tree,
                           std::span<const FoldType> folds_ffa,
                           std::span<const SizeType> indices_ffa,
                           std::span<const float> phase_shift,
                           std::span<FoldType> folds_out,
                           SizeType n_leaves,
                           SizeType physical_start_idx,
                           SizeType capacity) noexcept = 0;

    virtual SizeType score_and_filter(std::span<const FoldType> folds_tree,
                                      std::span<float> scores_tree,
                                      std::span<SizeType> indices_tree,
                                      float threshold,
                                      SizeType n_leaves) = 0;

    virtual void transform(std::span<double> leaves_tree,
                           std::span<SizeType> indices_tree,
                           std::pair<double, double> coord_next,
                           std::pair<double, double> coord_cur,
                           SizeType n_leaves) const = 0;

    virtual std::vector<double>
    get_transform_matrix(std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_prev) const = 0;

    virtual void pack(std::span<const FoldType> data,
                      std::span<FoldType> out) const noexcept = 0;

    virtual void
    ascend(std::span<const FoldType> folds_ffa,
           std::span<const double> leaves_tree,
           std::span<FoldType> folds_tree,
           std::span<float> scores_tree,
           std::span<float> scores_ep_tree,
           std::span<const SizeType> idx_segments,
           std::span<const std::pair<double, double>> coord_segments,
           std::pair<double, double> coord_cur,
           std::span<SizeType> scratch_param_indices,
           std::span<float> scratch_phase_shift,
           SizeType n_leaves) = 0;

    virtual void report(std::span<double> leaves_tree,
                        std::pair<double, double> coord_report,
                        SizeType n_leaves) const = 0;

    /** IRFFT scoring scratch (complex + real), zero for float EP. */
    [[nodiscard]] virtual float get_irfft_scratch_memory_gib() const noexcept {
        return 0.0F;
    }
};

// CRTP Base class - shared functionality for all derived classes
template <SupportedFoldType FoldType, typename Derived>
class BasePruneDPFuncts : public PruneDPFuncts<FoldType> {
protected:
    // Common members for all derived classes
    std::vector<SizeType> m_param_grid_count_init;
    std::vector<double> m_dparams_init;
    SizeType m_nseg_ffa;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
    SizeType m_batch_size;
    SizeType m_branch_max;

    SizeType m_n_coords_init{};
    // Buffer for shift-add operations
    std::vector<FoldType> m_scratch_shifts;
    // IRFFT scratch: complex copy (C2R overwrites input) + real output for
    // scoring
    std::vector<ComplexType> m_scratch_folds_c;
    std::vector<float> m_scratch_folds_r;
    math::FFTWManager m_fft_manager;
    // Cache for snr_boxcar_batch
    detection::BoxcarWidthsCache m_boxcar_widths_cache;
    detection::BoxcarKadaneCache m_boxcar_kadane_cache;

    // Constructor for all derived classes
    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility): multi-level CRTP
    BasePruneDPFuncts(std::span<const SizeType> param_grid_count_init,
                      std::span<const double> dparams_init,
                      SizeType nseg_ffa,
                      double tseg_ffa,
                      search::PulsarSearchConfig cfg,
                      SizeType batch_size,
                      SizeType branch_max);

    /** Copy complex folds to scratch, IRFFT to @p dst (ComplexType EP only). */
    void irfft_for_scoring(std::span<const ComplexType> src,
                           SizeType nfft,
                           std::span<float> dst)
        requires(std::is_same_v<FoldType, ComplexType>);

public:
    // Common implementations shared by all variants
    std::span<const FoldType> load_segment(std::span<const FoldType> ffa_fold,
                                           SizeType seg_idx) const override;

    SizeType validate(std::span<double> leaves_branch,
                      std::span<SizeType> leaves_origins,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves) const override;

    std::tuple<std::vector<double>, std::vector<double>, double>
    get_validation_params(std::pair<double, double> coord_add) const override;

    void shift_add(std::span<const FoldType> folds_tree,
                   std::span<const SizeType> indices_tree,
                   std::span<const FoldType> folds_ffa,
                   std::span<const SizeType> indices_ffa,
                   std::span<const float> phase_shift,
                   std::span<FoldType> folds_out,
                   SizeType n_leaves,
                   SizeType physical_start_idx,
                   SizeType capacity) noexcept override;

    SizeType score_and_filter(std::span<const FoldType> folds_tree,
                              std::span<float> scores_tree,
                              std::span<SizeType> indices_tree,
                              float threshold,
                              SizeType n_leaves) override;

    std::vector<double>
    get_transform_matrix(std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_prev) const override;

    void pack(std::span<const FoldType> data,
              std::span<FoldType> out) const noexcept override;

    [[nodiscard]] float get_irfft_scratch_memory_gib() const noexcept override;
};

// Intermediate base for Taylor-based methods (common seed implementation)
template <SupportedFoldType FoldType, typename Derived>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility): multi-level CRTP
class BaseTaylorPruneDPFuncts : public BasePruneDPFuncts<FoldType, Derived> {
protected:
    using Base = BasePruneDPFuncts<FoldType, Derived>;

    // Inherit constructor
    using Base::BasePruneDPFuncts;
    using Base::irfft_for_scoring;

public:
    // Common seed implementation for all Taylor variants
    void seed(std::span<const FoldType> fold_segment,
              std::span<double> seed_leaves,
              std::span<float> seed_scores,
              std::pair<double, double> coord_init) override;
};

// Intermediate base for Chebyshev-based methods (common seed implementation)
template <SupportedFoldType FoldType, typename Derived>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility): multi-level CRTP
class BaseChebyshevPruneDPFuncts : public BasePruneDPFuncts<FoldType, Derived> {
protected:
    using Base = BasePruneDPFuncts<FoldType, Derived>;

    // Inherit constructor
    using Base::BasePruneDPFuncts;
    using Base::irfft_for_scoring;

public:
    void seed(std::span<const FoldType> fold_segment,
              std::span<double> seed_leaves,
              std::span<float> seed_scores,
              std::pair<double, double> coord_init) override;
};

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldType FoldType>
class PrunePolyTaylorDPFuncts final
    : public BaseTaylorPruneDPFuncts<FoldType,
                                     PrunePolyTaylorDPFuncts<FoldType>> {
private:
    using Base =
        BaseTaylorPruneDPFuncts<FoldType, PrunePolyTaylorDPFuncts<FoldType>>;

public:
    PrunePolyTaylorDPFuncts(std::span<const SizeType> param_grid_count_init,
                            std::span<const double> dparams_init,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size,
                            SizeType branch_max);

    SizeType branch(std::span<double> leaves_tree,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    memory::BranchingWorkspace& branch_ws) const override;

    void resolve(std::span<const double> leaves_branch,
                 std::span<SizeType> param_indices,
                 std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves) const override;

    void transform(std::span<double> leaves_tree,
                   std::span<SizeType> indices_tree,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves) const override;

    void ascend(std::span<const FoldType> folds_ffa,
                std::span<const double> leaves_tree,
                std::span<FoldType> folds_tree,
                std::span<float> scores_tree,
                std::span<float> scores_ep_tree,
                std::span<const SizeType> idx_segments,
                std::span<const std::pair<double, double>> coord_segments,
                std::pair<double, double> coord_cur,
                std::span<SizeType> scratch_param_indices,
                std::span<float> scratch_phase_shift,
                SizeType n_leaves) override;

    void report(std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves) const override;
};

// Specialized implementation for Polynomial searches in Chebyshev Basis
template <SupportedFoldType FoldType>
class PrunePolyChebyshevDPFuncts final
    : public BaseChebyshevPruneDPFuncts<FoldType,
                                        PrunePolyChebyshevDPFuncts<FoldType>> {
private:
    using Base =
        BaseChebyshevPruneDPFuncts<FoldType,
                                   PrunePolyChebyshevDPFuncts<FoldType>>;

public:
    PrunePolyChebyshevDPFuncts(std::span<const SizeType> param_grid_count_init,
                               std::span<const double> dparams_init,
                               SizeType nseg_ffa,
                               double tseg_ffa,
                               search::PulsarSearchConfig cfg,
                               SizeType batch_size,
                               SizeType branch_max);

    SizeType branch(std::span<double> leaves_tree,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    memory::BranchingWorkspace& branch_ws) const override;

    void resolve(std::span<const double> leaves_branch,
                 std::span<SizeType> param_indices,
                 std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves) const override;

    void transform(std::span<double> leaves_tree,
                   std::span<SizeType> indices_tree,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves) const override;

    void ascend(std::span<const FoldType> folds_ffa,
                std::span<const double> leaves_tree,
                std::span<FoldType> folds_tree,
                std::span<float> scores_tree,
                std::span<float> scores_ep_tree,
                std::span<const SizeType> idx_segments,
                std::span<const std::pair<double, double>> coord_segments,
                std::pair<double, double> coord_cur,
                std::span<SizeType> scratch_param_indices,
                std::span<float> scratch_phase_shift,
                SizeType n_leaves) override;

    void report(std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves) const override;
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
    PruneCircTaylorDPFuncts(std::span<const SizeType> param_grid_count_init,
                            std::span<const double> dparams_init,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size,
                            SizeType branch_max);

    SizeType branch(std::span<double> leaves_tree,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    memory::BranchingWorkspace& branch_ws) const override;

    SizeType validate(std::span<double> leaves_branch,
                      std::span<SizeType> leaves_origins,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves) const override;

    void resolve(std::span<const double> leaves_branch,
                 std::span<SizeType> param_indices,
                 std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves) const override;

    void transform(std::span<double> leaves_tree,
                   std::span<SizeType> indices_tree,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves) const override;

    void ascend(std::span<const FoldType> folds_ffa,
                std::span<const double> leaves_tree,
                std::span<FoldType> folds_tree,
                std::span<float> scores_tree,
                std::span<float> scores_ep_tree,
                std::span<const SizeType> idx_segments,
                std::span<const std::pair<double, double>> coord_segments,
                std::pair<double, double> coord_cur,
                std::span<SizeType> scratch_param_indices,
                std::span<float> scratch_phase_shift,
                SizeType n_leaves) override;

    void report(std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves) const override;
};

// Factory function to create the correct implementation based on the kind
template <SupportedFoldType FoldType>
std::unique_ptr<PruneDPFuncts<FoldType>>
create_prune_dp_functs(std::string_view poly_basis,
                       std::span<const SizeType> param_grid_count_init,
                       std::span<const double> dparams_init,
                       SizeType nseg_ffa,
                       double tseg_ffa,
                       search::PulsarSearchConfig cfg,
                       SizeType batch_size,
                       SizeType branch_max);

// Type aliases for convenience
using PrunePolyTaylorDPFunctsFloat   = PrunePolyTaylorDPFuncts<float>;
using PrunePolyTaylorDPFunctsComplex = PrunePolyTaylorDPFuncts<ComplexType>;
using PruneCircTaylorDPFunctsFloat   = PruneCircTaylorDPFuncts<float>;
using PruneCircTaylorDPFunctsComplex = PruneCircTaylorDPFuncts<ComplexType>;

} // namespace loki::core
