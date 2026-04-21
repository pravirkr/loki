#include "loki/core/dynamic.hpp"

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <thrust/copy.h>

#include "loki/common/types.hpp"
#include "loki/core/chebyshev.hpp"
#include "loki/core/circular.hpp"
#include "loki/core/taylor.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/kernels.hpp"
#include "loki/utils/fft.hpp"

namespace loki::core {

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::BasePruneDPFunctsCUDA(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_branch_max(branch_max) {
    m_boxcar_widths_d       = m_cfg.get_scoring_widths();
    const auto param_limits = m_cfg.get_param_limits();

    SizeType n_coords_init = 1;
    for (const auto count : param_grid_count_init) {
        n_coords_init *= count;
    }
    m_n_coords_init = n_coords_init;
    m_param_grid_count_init_d.resize(param_grid_count_init.size());
    m_dparams_init_d.resize(dparams_init.size());
    m_param_limits_d.resize(param_limits.size());

    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(
            thrust::raw_pointer_cast(m_param_grid_count_init_d.data()),
            param_grid_count_init.data(),
            param_grid_count_init.size() * sizeof(SizeType),
            cudaMemcpyHostToDevice),
        "cudaMemcpyAsync param grid count init failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_dparams_init_d.data()),
                        dparams_init.data(),
                        dparams_init.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
        "cudaMemcpyAsync dparams init failed");
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(m_param_limits_d.data()),
                        param_limits.data(),
                        param_limits.size() * sizeof(ParamLimit),
                        cudaMemcpyHostToDevice),
        "cudaMemcpyAsync param limits failed");

    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        m_irfft_executor =
            std::make_unique<math::IrfftExecutorCUDA>(m_cfg.get_nbins());
        const auto max_batch_size = m_batch_size * m_branch_max;
        m_scratch_folds_d.resize(max_batch_size * 2 * m_cfg.get_nbins());
    } else {
        m_scratch_folds_d.resize(1); // Not needed for float
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
cuda::std::span<const FoldTypeCUDA>
BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::load_segment(
    cuda::std::span<const FoldTypeCUDA> ffa_fold, SizeType seg_idx) const {
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        return ffa_fold.subspan(seg_idx * m_n_coords_init * 2 * nbins_f,
                                m_n_coords_init * 2 * nbins_f);
    } else {
        return ffa_fold.subspan(seg_idx * m_n_coords_init * 2 * nbins,
                                m_n_coords_init * 2 * nbins);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::validate(
    cuda::std::span<double> /*leaves_branch*/,
    cuda::std::span<uint32_t> /*leaves_origins*/,
    cuda::std::span<uint8_t> /*validation_mask*/,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves,
    memory::CUBScratchArena& /*scratch_ws*/,
    cudaStream_t /*stream*/) const noexcept {
    return n_leaves;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
void BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::shift_add(
    cuda::std::span<const FoldTypeCUDA> folds_tree,
    cuda::std::span<const uint32_t> indices_tree,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<const FoldTypeCUDA> folds_ffa,
    cuda::std::span<const uint32_t> indices_ffa,
    cuda::std::span<const float> phase_shift,
    cuda::std::span<FoldTypeCUDA> folds_out,
    SizeType n_leaves,
    SizeType physical_start_idx,
    SizeType capacity,
    cudaStream_t stream) const noexcept {
    if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
        kernels::shift_add_linear_batch_cuda(
            folds_tree.data(), indices_tree.data(), validation_mask.data(),
            folds_ffa.data(), indices_ffa.data(), phase_shift.data(),
            folds_out.data(), m_cfg.get_nbins(), n_leaves, physical_start_idx,
            capacity, stream);
    } else {
        kernels::shift_add_linear_complex_batch_cuda(
            folds_tree.data(), indices_tree.data(), validation_mask.data(),
            folds_ffa.data(), indices_ffa.data(), phase_shift.data(),
            folds_out.data(), m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_leaves,
            physical_start_idx, capacity, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::score_and_filter(
    cuda::std::span<const FoldTypeCUDA> folds_tree,
    cuda::std::span<float> scores_tree,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint8_t> filtered_mask,
    float threshold,
    SizeType n_leaves,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream) noexcept {
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        // Ensure exact span for irfft transform
        const auto nfft = 2 * n_leaves;
        auto folds_span = folds_tree.first(nfft * nbins_f);
        auto folds_t_span =
            cuda_utils::as_span(m_scratch_folds_d).first(nfft * nbins);
        m_irfft_executor->execute(folds_span, folds_t_span,
                                  static_cast<int>(nfft), stream);
        return detection::score_and_filter_max_cuda_thread_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, validation_mask, filtered_mask, threshold, n_leaves,
            nbins, scratch_ws, stream);
    } else {
        return detection::score_and_filter_max_cuda_thread_d(
            folds_tree, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, validation_mask, filtered_mask, threshold, n_leaves,
            nbins, scratch_ws, stream);
    }
}

// Intermediate implementation for Taylor basis
template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
void BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA, Derived>::seed(
    cuda::std::span<const FoldTypeCUDA> fold_segment,
    cuda::std::span<double> seed_leaves,
    cuda::std::span<float> seed_scores,
    std::pair<double, double> coord_init,
    cudaStream_t stream) {
    poly_taylor_seed_cuda(cuda_utils::as_span(this->m_param_grid_count_init_d),
                          cuda_utils::as_span(this->m_dparams_init_d),
                          cuda_utils::as_span(this->m_param_limits_d),
                          seed_leaves, coord_init, this->m_n_coords_init,
                          this->m_cfg.get_nparams(), stream);
    // Fold segment is (n_leaves, 2, nbins)
    const auto nbins   = this->m_cfg.get_nbins();
    const auto nbins_f = this->m_cfg.get_nbins_f();

    // Calculate scores
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        const auto nfft = 2 * this->m_n_coords_init;
        error_check::check_equal(fold_segment.size(), nfft * nbins_f,
                                 "fold_segment size mismatch");
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_d).first(nfft * nbins);
        this->m_irfft_executor->execute(fold_segment, folds_t_span,
                                        static_cast<int>(nfft), stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            seed_scores, this->m_n_coords_init, nbins, stream);
    } else {
        error_check::check_equal(fold_segment.size(),
                                 this->m_n_coords_init * 2 * nbins,
                                 "fold_segment size mismatch");
        detection::snr_boxcar_3d_max_cuda_d(
            fold_segment, cuda_utils::as_span(this->m_boxcar_widths_d),
            seed_scores, this->m_n_coords_init, nbins, stream);
    }
}

// Intermediate implementation for Taylor basis
template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
void BaseChebyshevPruneDPFunctsCUDA<FoldTypeCUDA, Derived>::seed(
    cuda::std::span<const FoldTypeCUDA> fold_segment,
    cuda::std::span<double> seed_leaves,
    cuda::std::span<float> seed_scores,
    std::pair<double, double> coord_init,
    cudaStream_t stream) {
    // First seed in Taylor basis
    poly_taylor_seed_cuda(cuda_utils::as_span(this->m_param_grid_count_init_d),
                          cuda_utils::as_span(this->m_dparams_init_d),
                          cuda_utils::as_span(this->m_param_limits_d),
                          seed_leaves, coord_init, this->m_n_coords_init,
                          this->m_cfg.get_nparams(), stream);
    // Then convert to Chebyshev basis
    poly_taylor_to_cheby_batch_cuda(seed_leaves, coord_init,
                                    this->m_n_coords_init,
                                    this->m_cfg.get_nparams(), stream);
    // Fold segment is (n_leaves, 2, nbins)
    const auto nbins   = this->m_cfg.get_nbins();
    const auto nbins_f = this->m_cfg.get_nbins_f();

    // Calculate scores
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        const auto nfft = 2 * this->m_n_coords_init;
        error_check::check_equal(fold_segment.size(), nfft * nbins_f,
                                 "fold_segment size mismatch");
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_d).first(nfft * nbins);
        this->m_irfft_executor->execute(fold_segment, folds_t_span,
                                        static_cast<int>(nfft), stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            seed_scores, this->m_n_coords_init, nbins, stream);
    } else {
        error_check::check_equal(fold_segment.size(),
                                 this->m_n_coords_init * 2 * nbins,
                                 "fold_segment size mismatch");
        detection::snr_boxcar_3d_max_cuda_d(
            fold_segment, cuda_utils::as_span(this->m_boxcar_widths_d),
            seed_scores, this->m_n_coords_init, nbins, stream);
    }
}

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::PrunePolyTaylorDPFunctsCUDA(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::branch(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    cuda::std::span<uint8_t> validation_mask,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    SizeType n_leaves,
    memory::BranchingWorkspaceCUDAView branch_ws,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream) {
    return poly_taylor_branch_batch_cuda(
        leaves_tree, leaves_branch, leaves_origins, validation_mask, coord_cur,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        cuda_utils::as_span(this->m_param_limits_d), this->m_branch_max,
        n_leaves, this->m_cfg.get_nparams(), branch_ws, scratch_ws, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::resolve(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves,
    cudaStream_t stream) const {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init_d[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init_d[n_params - 1];
    poly_taylor_resolve_batch_cuda(
        leaves_branch, validation_mask, param_indices, phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_add, coord_cur,
        coord_init, n_accel_init, n_freq_init, this->m_cfg.get_nbins(),
        n_leaves, this->m_cfg.get_nparams(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::transform(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    cudaStream_t stream) const {
    poly_taylor_transform_batch_cuda(
        leaves_tree, validation_mask, coord_next, coord_cur, n_leaves,
        this->m_cfg.get_nparams(), this->m_cfg.get_use_conservative_tile(),
        stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::ascend(
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
    cudaStream_t stream) {
    const auto n_params      = this->m_cfg.get_nparams();
    const auto n_accel_init  = this->m_param_grid_count_init_d[n_params - 2];
    const auto n_freq_init   = this->m_param_grid_count_init_d[n_params - 1];
    const auto nbins         = this->m_cfg.get_nbins();
    const auto nbins_f       = this->m_cfg.get_nbins_f();
    const auto n_segments    = idx_segments.size();
    const auto n_coords_init = this->m_n_coords_init;

    poly_taylor_ascend_resolve_batch_cuda(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_segments, coord_cur,
        n_accel_init, n_freq_init, nbins, n_leaves, n_params, n_segments,
        stream);

    // Copy scores to scores_ep for future backup
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(scores_ep_tree.data(), scores_tree.data(),
                        n_leaves * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream),
        "cudaMemcpyAsync scores_ep_tree failed");

    // Calculate scores
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        kernels::shift_add_ascend_linear_complex_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments, stream);
        const auto nfft = 2 * n_leaves;
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_d).first(nfft * nbins);
        this->m_irfft_executor->execute(folds_tree, folds_t_span,
                                        static_cast<int>(nfft), stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);

    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        kernels::shift_add_ascend_linear_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins, n_coords_init,
            n_leaves, n_segments, stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_tree, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::report(
    cuda::std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves,
    cudaStream_t stream) const {
    if (n_leaves == 0) {
        return;
    }
    poly_taylor_report_batch_cuda(leaves_tree, coord_report, n_leaves,
                                  this->m_cfg.get_nparams(), stream);
}

// Specialized implementation for Polynomial searches in Chebyshev Basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>::PrunePolyChebyshevDPFunctsCUDA(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>::branch(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    cuda::std::span<uint8_t> validation_mask,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_prev,
    SizeType n_leaves,
    memory::BranchingWorkspaceCUDAView branch_ws,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream) {
    return poly_chebyshev_branch_batch_cuda(
        leaves_tree, leaves_branch, leaves_origins, validation_mask, coord_cur,
        coord_prev, this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        cuda_utils::as_span(this->m_param_limits_d), this->m_branch_max,
        n_leaves, this->m_cfg.get_nparams(), branch_ws, scratch_ws, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>::resolve(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves,
    cudaStream_t stream) const {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init_d[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init_d[n_params - 1];
    poly_chebyshev_resolve_batch_cuda(
        leaves_branch, validation_mask, param_indices, phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_add, coord_cur,
        coord_init, n_accel_init, n_freq_init, this->m_cfg.get_nbins(),
        n_leaves, this->m_cfg.get_nparams(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>::transform(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    cudaStream_t stream) const {
    if (this->m_cfg.get_use_conservative_tile()) {
        throw std::logic_error(
            "Conservative tile not implemented for Chebyshev basis");
    }
    poly_chebyshev_transform_batch_cuda(leaves_tree, validation_mask,
                                        coord_next, coord_cur, n_leaves,
                                        this->m_cfg.get_nparams(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>::ascend(
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
    cudaStream_t stream) {
    const auto n_params      = this->m_cfg.get_nparams();
    const auto n_accel_init  = this->m_param_grid_count_init_d[n_params - 2];
    const auto n_freq_init   = this->m_param_grid_count_init_d[n_params - 1];
    const auto nbins         = this->m_cfg.get_nbins();
    const auto nbins_f       = this->m_cfg.get_nbins_f();
    const auto n_segments    = idx_segments.size();
    const auto n_coords_init = this->m_n_coords_init;

    poly_chebyshev_ascend_resolve_batch_cuda(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_segments, coord_cur,
        n_accel_init, n_freq_init, nbins, n_leaves, n_params, n_segments,
        stream);

    // Copy scores to scores_ep for future backup
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(scores_ep_tree.data(), scores_tree.data(),
                        n_leaves * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream),
        "cudaMemcpyAsync scores_ep_tree failed");

    // Calculate scores
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        kernels::shift_add_ascend_linear_complex_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments, stream);
        const auto nfft = 2 * n_leaves;
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_d).first(nfft * nbins);
        this->m_irfft_executor->execute(folds_tree, folds_t_span,
                                        static_cast<int>(nfft), stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);

    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        kernels::shift_add_ascend_linear_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins, n_coords_init,
            n_leaves, n_segments, stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_tree, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>::report(
    cuda::std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves,
    cudaStream_t stream) const {
    if (n_leaves == 0) {
        return;
    }
    // First convert the Chebyshev leaves to Taylor leaves
    poly_cheby_to_taylor_batch_cuda(leaves_tree, coord_report, n_leaves,
                                    this->m_cfg.get_nparams(), stream);
    // Now call the Taylor report batch
    poly_taylor_report_batch_cuda(leaves_tree, coord_report, n_leaves,
                                  this->m_cfg.get_nparams(), stream);
}

// Specialized implementation for Circular orbit search in Taylor basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::PruneCircTaylorDPFunctsCUDA(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max) {}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::branch(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    cuda::std::span<uint8_t> validation_mask,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    SizeType n_leaves,
    memory::BranchingWorkspaceCUDAView branch_ws,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream) {
    return circ_taylor_branch_batch_cuda(
        leaves_tree, leaves_branch, leaves_origins, validation_mask, coord_cur,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        cuda_utils::as_span(this->m_param_limits_d), this->m_branch_max,
        n_leaves, this->m_cfg.get_minimum_snap_cells(), branch_ws, scratch_ws,
        stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::validate(
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> /*leaves_origins*/,
    cuda::std::span<uint8_t> validation_mask,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves,
    memory::CUBScratchArena& scratch_ws,
    cudaStream_t stream) const noexcept {
    return circ_taylor_validate_batch_cuda(
        leaves_branch, validation_mask, n_leaves, this->m_cfg.get_p_orb_min(),
        this->m_cfg.get_x_mass_const(), this->m_cfg.get_minimum_snap_cells(),
        scratch_ws, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::resolve(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves,
    cudaStream_t stream) const {
    const auto n_params     = this->m_cfg.get_nparams();
    const auto n_accel_init = this->m_param_grid_count_init_d[n_params - 2];
    const auto n_freq_init  = this->m_param_grid_count_init_d[n_params - 1];
    circ_taylor_resolve_batch_cuda(
        leaves_branch, validation_mask, param_indices, phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_add, coord_cur,
        coord_init, n_accel_init, n_freq_init, this->m_cfg.get_nbins(),
        n_leaves, this->m_cfg.get_minimum_snap_cells(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::transform(
    cuda::std::span<double> leaves_tree,
    cuda::std::span<const uint8_t> validation_mask,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    cudaStream_t stream) const {
    circ_taylor_transform_batch_cuda(
        leaves_tree, validation_mask, coord_next, coord_cur, n_leaves,
        this->m_cfg.get_use_conservative_tile(),
        this->m_cfg.get_minimum_snap_cells(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::ascend(
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
    cudaStream_t stream) {
    const auto n_params      = this->m_cfg.get_nparams();
    const auto n_accel_init  = this->m_param_grid_count_init_d[n_params - 2];
    const auto n_freq_init   = this->m_param_grid_count_init_d[n_params - 1];
    const auto nbins         = this->m_cfg.get_nbins();
    const auto nbins_f       = this->m_cfg.get_nbins_f();
    const auto n_segments    = idx_segments.size();
    const auto n_coords_init = this->m_n_coords_init;

    circ_taylor_ascend_resolve_batch_cuda(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_segments, coord_cur,
        n_accel_init, n_freq_init, nbins, n_leaves, n_segments,
        this->m_cfg.get_minimum_snap_cells(), stream);

    // Copy scores to scores_ep for future backup
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(scores_ep_tree.data(), scores_tree.data(),
                        n_leaves * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream),
        "cudaMemcpyAsync scores_ep_tree failed");

    // Calculate scores
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        kernels::shift_add_ascend_linear_complex_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments, stream);
        const auto nfft = 2 * n_leaves;
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_d).first(nfft * nbins);
        this->m_irfft_executor->execute(folds_tree, folds_t_span,
                                        static_cast<int>(nfft), stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);

    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        kernels::shift_add_ascend_linear_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins, n_coords_init,
            n_leaves, n_segments, stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_tree, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>::report(
    cuda::std::span<double> leaves_tree,
    std::pair<double, double> coord_report,
    SizeType n_leaves,
    cudaStream_t stream) const {
    if (n_leaves == 0) {
        return;
    }
    poly_taylor_report_batch_cuda(leaves_tree, coord_report, n_leaves,
                                  this->m_cfg.get_nparams(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::unique_ptr<PruneDPFunctsCUDA<FoldTypeCUDA>>
create_prune_dp_functs_cuda(std::string_view poly_basis,
                            std::span<const SizeType> param_grid_count_init,
                            std::span<const double> dparams_init,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size,
                            SizeType branch_max) {
    const auto n_params = cfg.get_nparams();
    if (poly_basis == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max);
    }
    if (poly_basis == "taylor" && n_params == 5) {
        return std::make_unique<PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max);
    }
    if (poly_basis == "chebyshev" && n_params <= 4) {
        return std::make_unique<PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max);
    }
    throw std::runtime_error(std::format(
        "Unknown poly_basis: '{}'. Valid options: 'taylor', 'chebyshev'",
        poly_basis));
}

// Explicit template instantiations
// Base classes need explicit instantiation for linker
template class BasePruneDPFunctsCUDA<float, PrunePolyTaylorDPFunctsCUDA<float>>;
template class BasePruneDPFunctsCUDA<
    ComplexTypeCUDA,
    PrunePolyTaylorDPFunctsCUDA<ComplexTypeCUDA>>;
template class BasePruneDPFunctsCUDA<float, PruneCircTaylorDPFunctsCUDA<float>>;
template class BasePruneDPFunctsCUDA<
    ComplexTypeCUDA,
    PruneCircTaylorDPFunctsCUDA<ComplexTypeCUDA>>;
template class BaseTaylorPruneDPFunctsCUDA<float,
                                           PrunePolyTaylorDPFunctsCUDA<float>>;
template class BaseTaylorPruneDPFunctsCUDA<
    ComplexTypeCUDA,
    PrunePolyTaylorDPFunctsCUDA<ComplexTypeCUDA>>;
template class BaseTaylorPruneDPFunctsCUDA<float,
                                           PruneCircTaylorDPFunctsCUDA<float>>;
template class BaseTaylorPruneDPFunctsCUDA<
    ComplexTypeCUDA,
    PruneCircTaylorDPFunctsCUDA<ComplexTypeCUDA>>;

template class BaseChebyshevPruneDPFunctsCUDA<
    float,
    PrunePolyChebyshevDPFunctsCUDA<float>>;
template class BaseChebyshevPruneDPFunctsCUDA<
    ComplexTypeCUDA,
    PrunePolyChebyshevDPFunctsCUDA<ComplexTypeCUDA>>;

// Derived classes
template class PrunePolyTaylorDPFunctsCUDA<float>;
template class PrunePolyTaylorDPFunctsCUDA<ComplexTypeCUDA>;
template class PruneCircTaylorDPFunctsCUDA<float>;
template class PruneCircTaylorDPFunctsCUDA<ComplexTypeCUDA>;
template class PrunePolyChebyshevDPFunctsCUDA<float>;
template class PrunePolyChebyshevDPFunctsCUDA<ComplexTypeCUDA>;

// Factory function instantiations
template std::unique_ptr<PruneDPFunctsCUDA<float>>
create_prune_dp_functs_cuda<float>(std::string_view,
                                   std::span<const SizeType>,
                                   std::span<const double>,
                                   SizeType,
                                   double,
                                   search::PulsarSearchConfig,
                                   SizeType,
                                   SizeType);
template std::unique_ptr<PruneDPFunctsCUDA<ComplexTypeCUDA>>
create_prune_dp_functs_cuda<ComplexTypeCUDA>(std::string_view,
                                             std::span<const SizeType>,
                                             std::span<const double>,
                                             SizeType,
                                             double,
                                             search::PulsarSearchConfig,
                                             SizeType,
                                             SizeType);

} // namespace loki::core