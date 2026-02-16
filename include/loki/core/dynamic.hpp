#pragma once

#include <span>
#include <tuple>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#endif // LOKI_ENABLE_CUDA

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

    virtual SizeType branch(std::span<const double> leaves_tree,
                            std::span<double> leaves_branch,
                            std::span<SizeType> leaves_origins,
                            std::pair<double, double> coord_cur,
                            std::pair<double, double> coord_prev,
                            SizeType n_leaves,
                            utils::BranchingWorkspaceView ws) const = 0;

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
                                      SizeType n_leaves) noexcept = 0;

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

    virtual void report(std::span<double> leaves_tree,
                        std::pair<double, double> coord_report,
                        SizeType n_leaves) const = 0;
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
    // Buffer for ComplexType irfft transform
    std::vector<float> m_scratch_folds;
    std::unique_ptr<utils::IrfftExecutor> m_irfft_executor;
    // Cache for snr_boxcar_batch
    detection::BoxcarWidthsCache m_boxcar_widths_cache;

    // Constructor for all derived classes
    BasePruneDPFuncts(std::span<const SizeType> param_grid_count_init,
                      std::span<const double> dparams_init,
                      SizeType nseg_ffa,
                      double tseg_ffa,
                      search::PulsarSearchConfig cfg,
                      SizeType batch_size,
                      SizeType branch_max);

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
                              SizeType n_leaves) noexcept override;

    std::vector<double>
    get_transform_matrix(std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_prev) const override;

    void pack(std::span<const FoldType> data,
              std::span<FoldType> out) const noexcept override;
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

    SizeType branch(std::span<const double> leaves_tree,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    utils::BranchingWorkspaceView ws) const override;

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

    SizeType branch(std::span<const double> leaves_tree,
                    std::span<double> leaves_branch,
                    std::span<SizeType> leaves_origins,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    utils::BranchingWorkspaceView ws) const override;

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

#ifdef LOKI_ENABLE_CUDA

// Virtual Interface - for runtime polymorphism
template <SupportedFoldTypeCUDA FoldTypeCUDA> class PruneDPFunctsCUDA {
public:
    virtual ~PruneDPFunctsCUDA() = default;

    // Delete copy/move for interface
    PruneDPFunctsCUDA()                                        = default;
    PruneDPFunctsCUDA(const PruneDPFunctsCUDA&)                = delete;
    PruneDPFunctsCUDA& operator=(const PruneDPFunctsCUDA&)     = delete;
    PruneDPFunctsCUDA(PruneDPFunctsCUDA&&) noexcept            = delete;
    PruneDPFunctsCUDA& operator=(PruneDPFunctsCUDA&&) noexcept = delete;

    // Core interface methods - all derived classes must implement these
    virtual cuda::std::span<const FoldTypeCUDA>
    load_segment(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                 SizeType seg_idx) const = 0;

    virtual void seed(cuda::std::span<const FoldTypeCUDA> fold_segment,
                      cuda::std::span<double> seed_leaves,
                      cuda::std::span<float> seed_scores,
                      std::pair<double, double> coord_init,
                      cudaStream_t stream) = 0;

    virtual SizeType branch(cuda::std::span<const double> leaves_tree,
                            cuda::std::span<double> leaves_branch,
                            cuda::std::span<uint32_t> leaves_origins,
                            std::pair<double, double> coord_cur,
                            std::pair<double, double> coord_prev,
                            SizeType n_leaves,
                            utils::BranchingWorkspaceCUDAView ws,
                            cudaStream_t stream) = 0;

    virtual SizeType validate(cuda::std::span<double> leaves_branch,
                              cuda::std::span<uint32_t> leaves_origins,
                              std::pair<double, double> coord_cur,
                              SizeType n_leaves,
                              cudaStream_t stream) const noexcept = 0;

    virtual void resolve(cuda::std::span<const double> leaves_branch,
                         cuda::std::span<uint32_t> param_indices,
                         cuda::std::span<float> phase_shift,
                         std::pair<double, double> coord_add,
                         std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_init,
                         SizeType n_leaves,
                         cudaStream_t stream) const = 0;

    virtual void shift_add(cuda::std::span<const FoldTypeCUDA> folds_tree,
                           cuda::std::span<const uint32_t> indices_tree,
                           cuda::std::span<const FoldTypeCUDA> folds_ffa,
                           cuda::std::span<const uint32_t> indices_ffa,
                           cuda::std::span<const float> phase_shift,
                           cuda::std::span<FoldTypeCUDA> folds_out,
                           SizeType n_leaves,
                           SizeType physical_start_idx,
                           SizeType capacity,
                           cudaStream_t stream) const noexcept = 0;

    virtual SizeType
    score_and_filter(cuda::std::span<const FoldTypeCUDA> folds_tree,
                     cuda::std::span<float> scores_tree,
                     cuda::std::span<uint32_t> indices_tree,
                     float threshold,
                     SizeType n_leaves,
                     utils::DeviceCounter& counter,
                     cudaStream_t stream) noexcept = 0;

    virtual void transform(cuda::std::span<double> leaves_tree,
                           cuda::std::span<uint32_t> indices_tree,
                           std::pair<double, double> coord_next,
                           std::pair<double, double> coord_cur,
                           SizeType n_leaves,
                           cudaStream_t stream) const = 0;

    virtual void report(cuda::std::span<double> leaves_tree,
                        std::pair<double, double> coord_report,
                        SizeType n_leaves,
                        cudaStream_t stream) const = 0;
};

// CRTP Base class - shared functionality for all derived classes
template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
class BasePruneDPFunctsCUDA : public PruneDPFunctsCUDA<FoldTypeCUDA> {
protected:
    // Common members for all derived classes
    SizeType m_nseg_ffa;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
    SizeType m_batch_size;
    SizeType m_branch_max;

    SizeType m_n_coords_init{};
    thrust::device_vector<SizeType> m_param_grid_count_init_d;
    thrust::device_vector<double> m_dparams_init_d;
    thrust::device_vector<ParamLimit> m_param_limits_d;
    thrust::device_vector<uint32_t> m_boxcar_widths_d;

    // Buffer for ComplexType irfft transform
    thrust::device_vector<float> m_scratch_folds_d;
    std::unique_ptr<utils::IrfftExecutorCUDA> m_irfft_executor;

    // Constructor for all derived classes
    BasePruneDPFunctsCUDA(std::span<const SizeType> param_grid_count_init,
                          std::span<const double> dparams,
                          SizeType nseg_ffa,
                          double tseg_ffa,
                          search::PulsarSearchConfig cfg,
                          SizeType batch_size,
                          SizeType branch_max);

public:
    // Common implementations shared by all variants
    cuda::std::span<const FoldTypeCUDA>
    load_segment(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                 SizeType seg_idx) const override;

    SizeType validate(cuda::std::span<double> leaves_branch,
                      cuda::std::span<uint32_t> leaves_origins,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves,
                      cudaStream_t stream) const noexcept override;

    void shift_add(cuda::std::span<const FoldTypeCUDA> folds_tree,
                   cuda::std::span<const uint32_t> indices_tree,
                   cuda::std::span<const FoldTypeCUDA> folds_ffa,
                   cuda::std::span<const uint32_t> indices_ffa,
                   cuda::std::span<const float> phase_shift,
                   cuda::std::span<FoldTypeCUDA> folds_out,
                   SizeType n_leaves,
                   SizeType physical_start_idx,
                   SizeType capacity,
                   cudaStream_t stream) const noexcept override;

    SizeType score_and_filter(cuda::std::span<const FoldTypeCUDA> folds_tree,
                              cuda::std::span<float> scores_tree,
                              cuda::std::span<uint32_t> indices_tree,
                              float threshold,
                              SizeType n_leaves,
                              utils::DeviceCounter& counter,
                              cudaStream_t stream) noexcept override;
};

// Intermediate base for Taylor-based methods (common seed implementation)
template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
class BaseTaylorPruneDPFunctsCUDA
    : public BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived> {
protected:
    using Base = BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>;

    // Inherit constructor
    using Base::BasePruneDPFunctsCUDA;

public:
    // Common seed implementation for all Taylor variants
    void seed(cuda::std::span<const FoldTypeCUDA> fold_segment,
              cuda::std::span<double> seed_leaves,
              cuda::std::span<float> seed_scores,
              std::pair<double, double> coord_init,
              cudaStream_t stream) override;
};

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PrunePolyTaylorDPFunctsCUDA final
    : public BaseTaylorPruneDPFunctsCUDA<
          FoldTypeCUDA,
          PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>> {
private:
    using Base =
        BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA,
                                    PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>>;

public:
    PrunePolyTaylorDPFunctsCUDA(std::span<const SizeType> param_grid_count_init,
                                std::span<const double> dparams_init,
                                SizeType nseg_ffa,
                                double tseg_ffa,
                                search::PulsarSearchConfig cfg,
                                SizeType batch_size,
                                SizeType branch_max);

    SizeType branch(cuda::std::span<const double> leaves_tree,
                    cuda::std::span<double> leaves_branch,
                    cuda::std::span<uint32_t> leaves_origins,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    utils::BranchingWorkspaceCUDAView ws,
                    cudaStream_t stream) override;

    void resolve(cuda::std::span<const double> leaves_branch,
                 cuda::std::span<uint32_t> param_indices,
                 cuda::std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves,
                 cudaStream_t stream) const override;

    void transform(cuda::std::span<double> leaves_tree,
                   cuda::std::span<uint32_t> indices_tree,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   cudaStream_t stream) const override;

    void report(cuda::std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves,
                cudaStream_t stream) const override;
};

// Factory function to create the correct implementation based on the kind
template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::unique_ptr<PruneDPFunctsCUDA<FoldTypeCUDA>>
create_prune_dp_functs_cuda(std::string_view poly_basis,
                            std::span<const SizeType> param_grid_count_init,
                            std::span<const double> dparams_init,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size,
                            SizeType branch_max);

// Type aliases for convenience
using PrunePolyTaylorDPFunctsCUDAFloat = PrunePolyTaylorDPFunctsCUDA<float>;
using PrunePolyTaylorDPFunctsCUDAComplex =
    PrunePolyTaylorDPFunctsCUDA<ComplexTypeCUDA>;

#endif // LOKI_ENABLE_CUDA
} // namespace loki::core