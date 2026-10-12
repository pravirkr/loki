#pragma once

/**
 * @file dynamic_cuda.cuh
 * @brief Device implementations declared alongside core/dynamic.hpp (CUDA
 * only).
 */

#include <memory>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "lib/core/dynamic.hpp"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

namespace loki::core {

/// Where shift_add_score writes the passing leaves of a batch straight into
/// the world tree's write region (see WorldTreeCUDA::get_direct_write):
/// rows of @p leaves_batch, merged folds and scores, in leaf order.
template <SupportedFoldTypeCUDA FoldTypeCUDA> struct EPDirectWriteCUDA {
    cuda::std::span<const double> leaves_batch;
    SizeType leaves_stride{0};
    double* leaves{nullptr};
    FoldTypeCUDA* folds{nullptr};
    float* scores{nullptr};
    SizeType write_head{0};
};

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

    virtual SizeType branch(cuda::std::span<double> leaves_tree,
                            cuda::std::span<double> leaves_branch,
                            cuda::std::span<uint32_t> leaves_origins,
                            cuda::std::span<uint8_t> validation_mask,
                            std::pair<double, double> coord_cur,
                            std::pair<double, double> coord_prev,
                            SizeType n_leaves,
                            memory::BranchingWorkspaceCUDAView branch_ws,
                            memory::CUBScratchArena& scratch_ws,
                            cudaStream_t stream) = 0;

    virtual SizeType validate(cuda::std::span<double> leaves_branch,
                              cuda::std::span<uint32_t> leaves_origins,
                              cuda::std::span<uint8_t> validation_mask,
                              std::pair<double, double> coord_cur,
                              SizeType n_leaves,
                              memory::CUBScratchArena& scratch_ws,
                              cudaStream_t stream) const noexcept = 0;

    virtual void resolve(cuda::std::span<const double> leaves_branch,
                         cuda::std::span<const uint8_t> validation_mask,
                         cuda::std::span<uint32_t> param_indices,
                         cuda::std::span<float> phase_shift,
                         std::pair<double, double> coord_add,
                         std::pair<double, double> coord_cur,
                         std::pair<double, double> coord_init,
                         SizeType n_leaves,
                         cudaStream_t stream) const = 0;

    virtual void shift_add(cuda::std::span<const FoldTypeCUDA> folds_tree,
                           cuda::std::span<const uint32_t> indices_tree,
                           cuda::std::span<const uint8_t> validation_mask,
                           cuda::std::span<const FoldTypeCUDA> folds_ffa,
                           cuda::std::span<const uint32_t> indices_ffa,
                           cuda::std::span<const float> phase_shift,
                           cuda::std::span<FoldTypeCUDA> folds_out,
                           SizeType n_leaves,
                           SizeType physical_start_idx,
                           SizeType capacity,
                           cudaStream_t stream) const noexcept = 0;

    /// resolve, then transform every valid leaf to @p coord_next (the
    /// direct-write path transforms before scoring).
    virtual void
    resolve_transform(cuda::std::span<double> leaves_branch,
                      cuda::std::span<const uint8_t> validation_mask,
                      cuda::std::span<uint32_t> param_indices,
                      cuda::std::span<float> phase_shift,
                      std::pair<double, double> coord_add,
                      std::pair<double, double> coord_cur,
                      std::pair<double, double> coord_init,
                      std::pair<double, double> coord_next,
                      SizeType n_leaves,
                      cudaStream_t stream) const {
        resolve(leaves_branch, validation_mask, param_indices, phase_shift,
                coord_add, coord_cur, coord_init, n_leaves, stream);
        transform(leaves_branch, validation_mask, coord_next, coord_cur,
                  n_leaves, stream);
    }

    /// Fused shift-add and scoring of one batch; returns the number of passing
    /// leaves. The fused kernel also gives the min/max score of the valid
    /// leaves (@p minmax; the identity {max, lowest} when there are none). With
    /// @p direct the passing leaves go into the tree instead of folds_out, and
    /// the call only enqueues the work and returns 0: collect_shift_add_score()
    /// gives its results later, so the host can enqueue the next batch
    /// meanwhile.
    virtual SizeType
    shift_add_score(cuda::std::span<const FoldTypeCUDA> folds_tree,
                    cuda::std::span<const uint32_t> indices_tree,
                    cuda::std::span<const uint8_t> validation_mask,
                    cuda::std::span<const FoldTypeCUDA> folds_ffa,
                    cuda::std::span<const uint32_t> indices_ffa,
                    cuda::std::span<const float> phase_shift,
                    cuda::std::span<FoldTypeCUDA> folds_out,
                    cuda::std::span<float> scores,
                    cuda::std::span<uint8_t> filtered_mask,
                    float threshold,
                    SizeType n_leaves,
                    SizeType physical_start_idx,
                    SizeType capacity,
                    memory::CUBScratchArena& scratch_ws,
                    cudaStream_t stream,
                    const EPDirectWriteCUDA<FoldTypeCUDA>* direct = nullptr,
                    MinMaxFloat* minmax = nullptr) = 0;

    /// Results of the last direct-mode shift_add_score (waits for it):
    /// the number of passing leaves, and their min/max in @p minmax.
    virtual SizeType collect_shift_add_score(MinMaxFloat& minmax) = 0;

    virtual void transform(cuda::std::span<double> leaves_tree,
                           cuda::std::span<const uint8_t> validation_mask,
                           std::pair<double, double> coord_next,
                           std::pair<double, double> coord_cur,
                           SizeType n_leaves,
                           cudaStream_t stream) const = 0;

    virtual void ascend(
        cuda::std::span<const FoldTypeCUDA> folds_ffa,
        cuda::std::span<const double> leaves_tree,
        cuda::std::span<FoldTypeCUDA> folds_tree,
        cuda::std::span<float> scores_tree,
        cuda::std::span<float> scores_ep_tree,
        cuda::std::span<const uint32_t> idx_segments,
        cuda::std::span<const cuda::std::pair<double, double>> coord_segments,
        std::pair<double, double> coord_cur,
        cuda::std::span<uint32_t> scratch_param_indices,
        cuda::std::span<float> scratch_phase_shift,
        SizeType n_leaves,
        cudaStream_t stream) = 0;

    virtual void report(cuda::std::span<double> leaves_tree,
                        std::pair<double, double> coord_report,
                        SizeType n_leaves,
                        cudaStream_t stream) const = 0;

    /** Fused scorer statistics and tile states. */
    [[nodiscard]] virtual float get_scorer_scratch_memory_gib() const noexcept {
        return 0.0F;
    }
};

// Base class - shared functionality for all derived classes
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class BasePruneDPFunctsCUDA : public PruneDPFunctsCUDA<FoldTypeCUDA> {
protected:
    // Common members for all derived classes
    SizeType m_nseg_ffa;
    double m_tseg_ffa;
    search::PulsarSearchConfig m_cfg;
    SizeType m_batch_size;
    SizeType m_branch_max;
    SizeType m_n_coords_init{};
    // Host copies of the last two grid counts: reading the device vector
    // element by element blocks on the stream.
    SizeType m_n_accel_init{};
    SizeType m_n_freq_init{};
    thrust::device_vector<SizeType> m_param_grid_count_init_d;
    thrust::device_vector<double> m_dparams_init_d;
    thrust::device_vector<ParamLimit> m_param_limits_d;
    thrust::device_vector<uint32_t> m_boxcar_widths_d;
    thrust::device_vector<float> m_boxcar_kadane_biases_d;

    // On-chip scoring, including Bluestein for non-power-of-two widths.
    thrust::device_vector<ComplexTypeCUDA> m_bluestein_d;
    // The fused kernel's batch statistics (kStatsWords words), then the
    // tile states of its ordered compaction (direct writes; at most one per
    // leaf), zeroed by one memset; and the pinned host copy of the
    // statistics. Counted as the scorer scratch.
    static constexpr SizeType kStatsWords = 2;
    thrust::device_vector<unsigned long long> m_tile_state_d;
    std::unique_ptr<uint32_t, void (*)(uint32_t*)> m_stats_h{nullptr, nullptr};
    // Recorded after the statistics copy of a direct-mode batch.
    std::unique_ptr<CUevent_st, void (*)(cudaEvent_t)> m_stats_ready{nullptr,
                                                                     nullptr};

    /// Passing count and min/max from the pinned statistics.
    SizeType decode_stats(MinMaxFloat* minmax) const;

    // Constructor for all derived classes
    BasePruneDPFunctsCUDA(std::span<const SizeType> param_grid_count_init,
                          std::span<const double> dparams,
                          SizeType nseg_ffa,
                          double tseg_ffa,
                          search::PulsarSearchConfig cfg,
                          SizeType batch_size,
                          SizeType branch_max,
                          int device_id = 0);

public:
    // Common implementations shared by all variants
    /** Max boxcar S/N of @p n_rows complex folds (the seeds and the
     * ascend): with the on-chip transform and scorer. */
    void score_complex_rows(cuda::std::span<const ComplexTypeCUDA> folds,
                            SizeType n_rows,
                            cuda::std::span<float> scores,
                            cudaStream_t stream)
        requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>);

    cuda::std::span<const FoldTypeCUDA>
    load_segment(cuda::std::span<const FoldTypeCUDA> ffa_fold,
                 SizeType seg_idx) const override;

    SizeType validate(cuda::std::span<double> leaves_branch,
                      cuda::std::span<uint32_t> leaves_origins,
                      cuda::std::span<uint8_t> validation_mask,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves,
                      memory::CUBScratchArena& scratch_ws,
                      cudaStream_t stream) const noexcept override;

    void shift_add(cuda::std::span<const FoldTypeCUDA> folds_tree,
                   cuda::std::span<const uint32_t> indices_tree,
                   cuda::std::span<const uint8_t> validation_mask,
                   cuda::std::span<const FoldTypeCUDA> folds_ffa,
                   cuda::std::span<const uint32_t> indices_ffa,
                   cuda::std::span<const float> phase_shift,
                   cuda::std::span<FoldTypeCUDA> folds_out,
                   SizeType n_leaves,
                   SizeType physical_start_idx,
                   SizeType capacity,
                   cudaStream_t stream) const noexcept override;

    SizeType
    shift_add_score(cuda::std::span<const FoldTypeCUDA> folds_tree,
                    cuda::std::span<const uint32_t> indices_tree,
                    cuda::std::span<const uint8_t> validation_mask,
                    cuda::std::span<const FoldTypeCUDA> folds_ffa,
                    cuda::std::span<const uint32_t> indices_ffa,
                    cuda::std::span<const float> phase_shift,
                    cuda::std::span<FoldTypeCUDA> folds_out,
                    cuda::std::span<float> scores,
                    cuda::std::span<uint8_t> filtered_mask,
                    float threshold,
                    SizeType n_leaves,
                    SizeType physical_start_idx,
                    SizeType capacity,
                    memory::CUBScratchArena& scratch_ws,
                    cudaStream_t stream,
                    const EPDirectWriteCUDA<FoldTypeCUDA>* direct = nullptr,
                    MinMaxFloat* minmax = nullptr) override;

    SizeType collect_shift_add_score(MinMaxFloat& minmax) override;

    [[nodiscard]] float get_scorer_scratch_memory_gib() const noexcept override;
};

// Intermediate base for Taylor-based methods (common seed implementation)
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class BaseTaylorPruneDPFunctsCUDA : public BasePruneDPFunctsCUDA<FoldTypeCUDA> {
protected:
    using Base = BasePruneDPFunctsCUDA<FoldTypeCUDA>;

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

// Intermediate base for Chebyshev-based methods (common seed implementation)
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class BaseChebyshevPruneDPFunctsCUDA
    : public BasePruneDPFunctsCUDA<FoldTypeCUDA> {
protected:
    using Base = BasePruneDPFunctsCUDA<FoldTypeCUDA>;

    // Inherit constructor
    using Base::BasePruneDPFunctsCUDA;

public:
    void seed(cuda::std::span<const FoldTypeCUDA> fold_segment,
              cuda::std::span<double> seed_leaves,
              cuda::std::span<float> seed_scores,
              std::pair<double, double> coord_init,
              cudaStream_t stream) override;
};

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PrunePolyTaylorDPFunctsCUDA final
    : public BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA> {
private:
    using Base = BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA>;

public:
    PrunePolyTaylorDPFunctsCUDA(std::span<const SizeType> param_grid_count_init,
                                std::span<const double> dparams_init,
                                SizeType nseg_ffa,
                                double tseg_ffa,
                                search::PulsarSearchConfig cfg,
                                SizeType batch_size,
                                SizeType branch_max,
                                int device_id = 0);

    SizeType branch(cuda::std::span<double> leaves_tree,
                    cuda::std::span<double> leaves_branch,
                    cuda::std::span<uint32_t> leaves_origins,
                    cuda::std::span<uint8_t> validation_mask,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    memory::BranchingWorkspaceCUDAView ws,
                    memory::CUBScratchArena& scratch_ws,
                    cudaStream_t stream) override;

    void resolve(cuda::std::span<const double> leaves_branch,
                 cuda::std::span<const uint8_t> validation_mask,
                 cuda::std::span<uint32_t> param_indices,
                 cuda::std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves,
                 cudaStream_t stream) const override;

    void transform(cuda::std::span<double> leaves_tree,
                   cuda::std::span<const uint8_t> validation_mask,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   cudaStream_t stream) const override;

    void resolve_transform(cuda::std::span<double> leaves_branch,
                           cuda::std::span<const uint8_t> validation_mask,
                           cuda::std::span<uint32_t> param_indices,
                           cuda::std::span<float> phase_shift,
                           std::pair<double, double> coord_add,
                           std::pair<double, double> coord_cur,
                           std::pair<double, double> coord_init,
                           std::pair<double, double> coord_next,
                           SizeType n_leaves,
                           cudaStream_t stream) const override;

    void ascend(
        cuda::std::span<const FoldTypeCUDA> folds_ffa,
        cuda::std::span<const double> leaves_tree,
        cuda::std::span<FoldTypeCUDA> folds_tree,
        cuda::std::span<float> scores_tree,
        cuda::std::span<float> scores_ep_tree,
        cuda::std::span<const uint32_t> idx_segments,
        cuda::std::span<const cuda::std::pair<double, double>> coord_segments,
        std::pair<double, double> coord_cur,
        cuda::std::span<uint32_t> scratch_param_indices,
        cuda::std::span<float> scratch_phase_shift,
        SizeType n_leaves,
        cudaStream_t stream) override;

    void report(cuda::std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves,
                cudaStream_t stream) const override;
};

// Specialized implementation for Polynomial searches in Chebyshev Basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PrunePolyChebyshevDPFunctsCUDA final
    : public BaseChebyshevPruneDPFunctsCUDA<FoldTypeCUDA> {
private:
    using Base = BaseChebyshevPruneDPFunctsCUDA<FoldTypeCUDA>;

public:
    PrunePolyChebyshevDPFunctsCUDA(
        std::span<const SizeType> param_grid_count_init,
        std::span<const double> dparams_init,
        SizeType nseg_ffa,
        double tseg_ffa,
        search::PulsarSearchConfig cfg,
        SizeType batch_size,
        SizeType branch_max,
        int device_id = 0);

    SizeType branch(cuda::std::span<double> leaves_tree,
                    cuda::std::span<double> leaves_branch,
                    cuda::std::span<uint32_t> leaves_origins,
                    cuda::std::span<uint8_t> validation_mask,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    memory::BranchingWorkspaceCUDAView ws,
                    memory::CUBScratchArena& scratch_ws,
                    cudaStream_t stream) override;

    void resolve(cuda::std::span<const double> leaves_branch,
                 cuda::std::span<const uint8_t> validation_mask,
                 cuda::std::span<uint32_t> param_indices,
                 cuda::std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves,
                 cudaStream_t stream) const override;

    void transform(cuda::std::span<double> leaves_tree,
                   cuda::std::span<const uint8_t> validation_mask,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   cudaStream_t stream) const override;

    void ascend(
        cuda::std::span<const FoldTypeCUDA> folds_ffa,
        cuda::std::span<const double> leaves_tree,
        cuda::std::span<FoldTypeCUDA> folds_tree,
        cuda::std::span<float> scores_tree,
        cuda::std::span<float> scores_ep_tree,
        cuda::std::span<const uint32_t> idx_segments,
        cuda::std::span<const cuda::std::pair<double, double>> coord_segments,
        std::pair<double, double> coord_cur,
        cuda::std::span<uint32_t> scratch_param_indices,
        cuda::std::span<float> scratch_phase_shift,
        SizeType n_leaves,
        cudaStream_t stream) override;

    void report(cuda::std::span<double> leaves_tree,
                std::pair<double, double> coord_report,
                SizeType n_leaves,
                cudaStream_t stream) const override;
};

// Specialized implementation for Circular orbit search in Taylor basis
// Use only when nparams == 5
template <SupportedFoldTypeCUDA FoldTypeCUDA>
class PruneCircTaylorDPFunctsCUDA final
    : public BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA> {
private:
    using Base = BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA>;

public:
    PruneCircTaylorDPFunctsCUDA(std::span<const SizeType> param_grid_count_init,
                                std::span<const double> dparams_init,
                                SizeType nseg_ffa,
                                double tseg_ffa,
                                search::PulsarSearchConfig cfg,
                                SizeType batch_size,
                                SizeType branch_max,
                                int device_id = 0);

    SizeType branch(cuda::std::span<double> leaves_tree,
                    cuda::std::span<double> leaves_branch,
                    cuda::std::span<uint32_t> leaves_origins,
                    cuda::std::span<uint8_t> validation_mask,
                    std::pair<double, double> coord_cur,
                    std::pair<double, double> coord_prev,
                    SizeType n_leaves,
                    memory::BranchingWorkspaceCUDAView ws,
                    memory::CUBScratchArena& scratch_ws,
                    cudaStream_t stream) override;

    SizeType validate(cuda::std::span<double> leaves_branch,
                      cuda::std::span<uint32_t> leaves_origins,
                      cuda::std::span<uint8_t> validation_mask,
                      std::pair<double, double> coord_cur,
                      SizeType n_leaves,
                      memory::CUBScratchArena& scratch_ws,
                      cudaStream_t stream) const noexcept override;

    void resolve(cuda::std::span<const double> leaves_branch,
                 cuda::std::span<const uint8_t> validation_mask,
                 cuda::std::span<uint32_t> param_indices,
                 cuda::std::span<float> phase_shift,
                 std::pair<double, double> coord_add,
                 std::pair<double, double> coord_cur,
                 std::pair<double, double> coord_init,
                 SizeType n_leaves,
                 cudaStream_t stream) const override;

    void transform(cuda::std::span<double> leaves_tree,
                   cuda::std::span<const uint8_t> validation_mask,
                   std::pair<double, double> coord_next,
                   std::pair<double, double> coord_cur,
                   SizeType n_leaves,
                   cudaStream_t stream) const override;

    void ascend(
        cuda::std::span<const FoldTypeCUDA> folds_ffa,
        cuda::std::span<const double> leaves_tree,
        cuda::std::span<FoldTypeCUDA> folds_tree,
        cuda::std::span<float> scores_tree,
        cuda::std::span<float> scores_ep_tree,
        cuda::std::span<const uint32_t> idx_segments,
        cuda::std::span<const cuda::std::pair<double, double>> coord_segments,
        std::pair<double, double> coord_cur,
        cuda::std::span<uint32_t> scratch_param_indices,
        cuda::std::span<float> scratch_phase_shift,
        SizeType n_leaves,
        cudaStream_t stream) override;

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
                            SizeType branch_max,
                            int device_id = 0);

// Type aliases for convenience
using PrunePolyTaylorDPFunctsCUDAFloat = PrunePolyTaylorDPFunctsCUDA<float>;
using PrunePolyTaylorDPFunctsCUDAComplex =
    PrunePolyTaylorDPFunctsCUDA<ComplexTypeCUDA>;

} // namespace loki::core
