#include "loki/core/dynamic.hpp"

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "loki/common/types.hpp"
#include "loki/core/taylor.hpp"
#include "loki/cuda_utils.cuh"
#include "loki/exceptions.hpp"
#include "loki/kernels_cuda.cuh"
#include "loki/utils/fft.hpp"

namespace loki::core {

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::BasePruneDPFunctsCUDA(
    std::span<const std::vector<double>> param_arr,
    std::span<const double> dparams,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max)
    : m_dparams(dparams.begin(), dparams.end()),
      m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_branch_max(branch_max) {
    // Copy param_arr to device
    const auto& accel_arr_grid = param_arr[0];
    const auto& freq_arr_grid  = param_arr[1];
    m_accel_grid_d.resize(accel_arr_grid.size());
    m_freq_grid_d.resize(freq_arr_grid.size());
    // copy with implicit double -> float conversion
    thrust::copy(accel_arr_grid.begin(), accel_arr_grid.end(),
                 m_accel_grid_d.begin());
    thrust::copy(freq_arr_grid.begin(), freq_arr_grid.end(),
                 m_freq_grid_d.begin());
    m_boxcar_widths_d = m_cfg.get_scoring_widths();

    std::vector<ParamLimitTypeCUDA> h_limits;
    h_limits.resize(m_cfg.get_nparams());

    for (SizeType i = 0; i < m_cfg.get_nparams(); ++i) {
        h_limits[i].min = m_cfg.get_param_limits()[i][0];
        h_limits[i].max = m_cfg.get_param_limits()[i][1];
    }
    m_param_limits_d = h_limits;
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        m_irfft_executor =
            std::make_unique<utils::IrfftExecutorCUDA>(m_cfg.get_nbins());
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
    const SizeType n_coords = m_accel_grid_d.size() * m_freq_grid_d.size();
    const auto nbins        = m_cfg.get_nbins();
    const auto nbins_f      = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        return ffa_fold.subspan(seg_idx * n_coords * 2 * nbins_f,
                                n_coords * 2 * nbins_f);
    } else {
        return ffa_fold.subspan(seg_idx * n_coords * 2 * nbins,
                                n_coords * 2 * nbins);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::validate(
    cuda::std::span<double> /*leaves_branch*/,
    cuda::std::span<uint32_t> /*leaves_origins*/,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves,
    cudaStream_t stream) const {
    return n_leaves;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
void BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::shift_add(
    cuda::std::span<const FoldTypeCUDA> folds_tree,
    cuda::std::span<const uint32_t> indices_tree,
    cuda::std::span<const FoldTypeCUDA> folds_ffa,
    cuda::std::span<const uint32_t> indices_ffa,
    cuda::std::span<const float> phase_shift,
    cuda::std::span<FoldTypeCUDA> folds_out,
    SizeType n_leaves,
    cudaStream_t stream) noexcept {
    if constexpr (std::is_same_v<FoldTypeCUDA, float>) {
        kernels::shift_add_linear_batch_cuda(
            folds_tree.data(), indices_tree.data(), folds_ffa.data(),
            indices_ffa.data(), phase_shift.data(), folds_out.data(),
            m_cfg.get_nbins(), n_leaves, stream);
    } else {
        return shift_add_linear_complex_batch_cuda(
            folds_tree.data(), indices_tree.data(), folds_ffa.data(),
            indices_ffa.data(), phase_shift.data(), folds_out.data(),
            m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_leaves, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA, typename Derived>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA, Derived>::score_and_filter(
    cuda::std::span<const FoldTypeCUDA> folds_tree,
    cuda::std::span<float> scores_tree,
    cuda::std::span<uint32_t> indices_tree,
    float threshold,
    SizeType n_leaves,
    cudaStream_t stream) noexcept {
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        // Ensure exact span for irfft transform
        const auto nfft = 2 * n_leaves;
        auto folds_span = cuda::std::span<const FoldTypeCUDA>(folds_tree)
                              .first(n_leaves * 2 * nbins_f);
        auto folds_t_span =
            cuda::std::span<float>(m_scratch_folds_d).first(nfft * nbins);
        m_irfft_executor->execute(folds_span, folds_t_span,
                                  static_cast<int>(nfft));
        return detection::score_and_filter_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, indices_tree, threshold, n_leaves, nbins, stream);
    } else {
        return detection::score_and_filter_max_cuda_d(
            folds_tree, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, indices_tree, threshold, n_leaves, nbins, stream);
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

    const auto n_leaves = poly_taylor_seed_cuda(
        this->m_accel_grid_d, this->m_freq_grid_d, this->m_dparams, seed_leaves,
        coord_init, this->m_cfg.get_nparams(), stream);
    // Fold segment is (n_leaves, 2, nbins)
    const auto nbins = this->m_cfg.get_nbins();

    // Calculate scores
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        const auto nfft    = 2 * n_leaves;
        const auto nbins_f = this->m_cfg.get_nbins_f();
        error_check::check_equal(fold_segment.size(), n_leaves * 2 * nbins_f,
                                 "fold_segment size mismatch");
        auto fold_segment_t =
            cuda_utils::as_span(this->m_scratch_folds_d).first(nfft * nbins);
        this->m_irfft_executor->execute(fold_segment, fold_segment_t,
                                        static_cast<int>(nfft));
        detection::snr_boxcar_3d_max_cuda_d(
            fold_segment_t, this->m_boxcar_widths_d, seed_scores, n_leaves,
            nbins, stream);

    } else {
        error_check::check_equal(fold_segment.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        detection::snr_boxcar_3d_max_cuda_d(
            fold_segment, this->m_boxcar_widths_d, seed_scores, n_leaves, nbins,
            stream);
    }
}

// Specialized implementation for Polynomial searches in Taylor Basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::PrunePolyTaylorDPFunctsCUDA(
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

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::branch(
    cuda::std::span<const double> leaves_tree,
    cuda::std::span<double> leaves_branch,
    cuda::std::span<uint32_t> leaves_origins,
    std::pair<double, double> coord_cur,
    std::pair<double, double> /*coord_prev*/,
    SizeType n_leaves,
    utils::BranchingWorkspaceCUDAView ws,
    cudaStream_t stream) const {
    return poly_taylor_branch_batch_cuda(
        leaves_tree, leaves_branch, leaves_origins, coord_cur,
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(),
        cuda_utils::as_span(this->m_param_limits_d), this->m_branch_max,
        n_leaves, this->m_cfg.get_nparams(), ws, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::resolve(
    cuda::std::span<const double> leaves_branch,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    SizeType n_leaves,
    cudaStream_t stream) const {
    poly_taylor_resolve_batch_cuda(
        leaves_branch, this->m_accel_grid_d, this->m_freq_grid_d, param_indices,
        phase_shift, coord_add, coord_cur, coord_init, this->m_cfg.get_nbins(),
        n_leaves, this->m_cfg.get_nparams(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::transform(
    cuda::std::span<double> leaves_tree,
    std::pair<double, double> coord_next,
    std::pair<double, double> coord_cur,
    SizeType n_leaves,
    cudaStream_t stream) const {
    poly_taylor_transform_batch_cuda(
        leaves_tree, coord_next, coord_cur, n_leaves, this->m_cfg.get_nparams(),
        this->m_cfg.get_use_conservative_tile(), stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
std::unique_ptr<PruneDPFunctsCUDA<FoldTypeCUDA>>
create_prune_dp_functs_cuda(std::string_view poly_basis,
                            std::span<const std::vector<double>> param_arr,
                            std::span<const double> dparams,
                            SizeType nseg_ffa,
                            double tseg_ffa,
                            search::PulsarSearchConfig cfg,
                            SizeType batch_size,
                            SizeType branch_max) {
    const auto n_params = cfg.get_nparams();
    if (poly_basis == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_arr, dparams, nseg_ffa, tseg_ffa, std::move(cfg), batch_size,
            branch_max);
    }
    throw std::runtime_error(std::format(
        "Unknown poly_basis: '{}'. Valid options: 'taylor'", poly_basis));
}

// Explicit template instantiations
// Base classes need explicit instantiation for linker
template class BasePruneDPFuncts<float, PrunePolyTaylorDPFuncts<float>>;
template class BasePruneDPFuncts<ComplexType,
                                 PrunePolyTaylorDPFuncts<ComplexType>>;

template class BaseTaylorPruneDPFuncts<float, PrunePolyTaylorDPFuncts<float>>;
template class BaseTaylorPruneDPFuncts<ComplexType,
                                       PrunePolyTaylorDPFuncts<ComplexType>>;

} // namespace loki::core