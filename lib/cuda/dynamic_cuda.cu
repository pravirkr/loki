#include "lib/cuda/dynamic_cuda.cuh"

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <thrust/copy.h>

#include "loki/common/types.hpp"

#include "lib/cuda/chebyshev_cuda.cuh"
#include "lib/cuda/circular_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/kadane_cuda.cuh"
#include "lib/cuda/kernels_cuda.cuh"
#include "lib/cuda/score_cuda.cuh"
#include "lib/cuda/taylor_cuda.cuh"
#include "lib/detail/error_check.hpp"

namespace loki::core {

namespace {

// Each distinct batch size costs cuFFT a new plan (about a millisecond of
// host time), so the transform is rounded up to a multiple of this many
// rows when the scratch has room. The extra rows are never scored.
constexpr SizeType kIrfftBatchGranule = 1024;

/**
 * Copy complex folds to scratch and IRFFT (cuFFT C2R overwrites input).
 * The first nfft * nbins floats of @p scratch_r receive the profiles,
 * without the 1/nbins factor when @p normalize is false.
 */
void irfft_folds_for_scoring(math::CUFFTManager& fft_manager,
                             thrust::device_vector<ComplexTypeCUDA>& scratch_c,
                             thrust::device_vector<float>& scratch_r,
                             cuda::std::span<const ComplexTypeCUDA> src,
                             SizeType nfft,
                             SizeType nbins,
                             SizeType nbins_f,
                             cudaStream_t stream,
                             bool normalize = true) {
    const auto n_complex = nfft * nbins_f;
    cuda_utils::check_cuda_call(
        cudaMemcpyAsync(thrust::raw_pointer_cast(scratch_c.data()), src.data(),
                        n_complex * sizeof(ComplexTypeCUDA),
                        cudaMemcpyDeviceToDevice, stream),
        "irfft_for_scoring: d2d copy failed");
    const auto nfft_padded =
        ((nfft + kIrfftBatchGranule - 1) / kIrfftBatchGranule) *
        kIrfftBatchGranule;
    const bool fits     = nfft_padded * nbins_f <= scratch_c.size() &&
                          nfft_padded * nbins <= scratch_r.size();
    const auto nfft_run = fits ? nfft_padded : nfft;
    fft_manager.irfft_batch(
        cuda_utils::as_span(scratch_c).first(nfft_run * nbins_f),
        cuda_utils::as_span(scratch_r).first(nfft_run * nbins), nfft_run, nbins,
        stream, normalize);
}

} // namespace

template <SupportedFoldTypeCUDA FoldTypeCUDA>
BasePruneDPFunctsCUDA<FoldTypeCUDA>::BasePruneDPFunctsCUDA(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max,
    int device_id,
    math::CUFFTManager* external_fft)
    : m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_branch_max(branch_max),
      m_fft_manager(device_id),
      m_external_fft(external_fft) {
    m_boxcar_widths_d        = m_cfg.get_scoring_widths();
    m_boxcar_kadane_biases_d = m_cfg.get_boxcar_kadane_biases();
    const auto param_limits  = m_cfg.get_param_limits();

    SizeType n_coords_init = 1;
    for (const auto count : param_grid_count_init) {
        n_coords_init *= count;
    }
    m_n_coords_init     = n_coords_init;
    const auto n_counts = param_grid_count_init.size();
    // The pruning kernels index the last two (accel, freq) grid counts.
    error_check::check_greater_equal(
        n_counts, 2,
        "BasePruneDPFunctsCUDA: need at least the accel and freq grids");
    m_n_accel_init = param_grid_count_init[n_counts - 2];
    m_n_freq_init  = param_grid_count_init[n_counts - 1];
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
        const auto nbins = m_cfg.get_nbins();
        if (m_external_fft != nullptr) {
            // The owner prepared the plans for every nbins of its sweep.
            error_check::check(m_external_fft->has_prepared(nbins),
                               "BasePruneDPFunctsCUDA: the external FFT "
                               "manager does not prepare this nbins");
        } else {
            m_fft_manager.prepare_exact_plans(
                std::span<const SizeType>(&nbins, 1));
        }
        const auto max_nfft =
            std::max(2 * m_batch_size * m_branch_max, 2 * m_n_coords_init);
        const auto nbins_f = m_cfg.get_nbins_f();
        m_scratch_folds_c_d.resize(max_nfft * nbins_f);
        m_scratch_folds_r_d.resize(max_nfft * nbins);
    } else {
        m_scratch_folds_c_d.resize(1); // Not needed for float
        m_scratch_folds_r_d.resize(1); // Not needed for float
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BasePruneDPFunctsCUDA<FoldTypeCUDA>::irfft_for_scoring(
    cuda::std::span<const ComplexTypeCUDA> src,
    SizeType nfft,
    cudaStream_t stream,
    bool normalize)
    requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>)
{
    irfft_folds_for_scoring(get_fft(), m_scratch_folds_c_d, m_scratch_folds_r_d,
                            src, nfft, m_cfg.get_nbins(), m_cfg.get_nbins_f(),
                            stream, normalize);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float BasePruneDPFunctsCUDA<FoldTypeCUDA>::get_irfft_scratch_memory_gib()
    const noexcept {
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        const auto bytes =
            (m_scratch_folds_c_d.size() * sizeof(ComplexTypeCUDA)) +
            (m_scratch_folds_r_d.size() * sizeof(float));
        return static_cast<float>(bytes) / static_cast<float>(1ULL << 30U);
    }
    return 0.0F;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const FoldTypeCUDA>
BasePruneDPFunctsCUDA<FoldTypeCUDA>::load_segment(
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

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA>::validate(
    cuda::std::span<double> /*leaves_branch*/,
    cuda::std::span<uint32_t> /*leaves_origins*/,
    cuda::std::span<uint8_t> /*validation_mask*/,
    std::pair<double, double> /*coord_cur*/,
    SizeType n_leaves,
    memory::CUBScratchArena& /*scratch_ws*/,
    cudaStream_t /*stream*/) const noexcept {
    return n_leaves;
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BasePruneDPFunctsCUDA<FoldTypeCUDA>::shift_add(
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
        core::shift_add_linear_batch_cuda(
            folds_tree.data(), indices_tree.data(), validation_mask.data(),
            folds_ffa.data(), indices_ffa.data(), phase_shift.data(),
            folds_out.data(), m_cfg.get_nbins(), n_leaves, physical_start_idx,
            capacity, stream);
    } else {
        core::shift_add_linear_complex_batch_cuda(
            folds_tree.data(), indices_tree.data(), validation_mask.data(),
            folds_ffa.data(), indices_ffa.data(), phase_shift.data(),
            folds_out.data(), m_cfg.get_nbins_f(), m_cfg.get_nbins(), n_leaves,
            physical_start_idx, capacity, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA>::score_and_filter(
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
            cuda_utils::as_span(m_scratch_folds_r_d).first(nfft * nbins);
        if (m_cfg.get_use_boxcar_kadane()) {
            irfft_for_scoring(folds_span, nfft, stream);
            return detection::score_and_filter_max_cuda_kadane_d(
                folds_t_span,
                cuda_utils::as_span(this->m_boxcar_kadane_biases_d),
                scores_tree, validation_mask, filtered_mask, threshold,
                n_leaves, nbins, scratch_ws, stream);
        }
        // The scorer applies the IRFFT's 1/nbins as it loads each bin (the
        // same multiply), which saves a pass over the profiles.
        irfft_for_scoring(folds_span, nfft, stream, /*normalize=*/false);
        const float irfft_norm = 1.0F / static_cast<float>(nbins);
        return detection::score_and_filter_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, validation_mask, filtered_mask, threshold, n_leaves,
            nbins, scratch_ws, stream, irfft_norm);
    } else {
        if (m_cfg.get_use_boxcar_kadane()) {
            return detection::score_and_filter_max_cuda_kadane_d(
                folds_tree, cuda_utils::as_span(this->m_boxcar_kadane_biases_d),
                scores_tree, validation_mask, filtered_mask, threshold,
                n_leaves, nbins, scratch_ws, stream);
        }
        return detection::score_and_filter_max_cuda_d(
            folds_tree, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, validation_mask, filtered_mask, threshold, n_leaves,
            nbins, scratch_ws, stream);
    }
}

// Intermediate implementation for Taylor basis
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BaseTaylorPruneDPFunctsCUDA<FoldTypeCUDA>::seed(
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
            cuda_utils::as_span(this->m_scratch_folds_r_d).first(nfft * nbins);
        irfft_folds_for_scoring(this->get_fft(), this->m_scratch_folds_c_d,
                                this->m_scratch_folds_r_d, fold_segment, nfft,
                                nbins, nbins_f, stream);
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
template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BaseChebyshevPruneDPFunctsCUDA<FoldTypeCUDA>::seed(
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
            cuda_utils::as_span(this->m_scratch_folds_r_d).first(nfft * nbins);
        irfft_folds_for_scoring(this->get_fft(), this->m_scratch_folds_c_d,
                                this->m_scratch_folds_r_d, fold_segment, nfft,
                                nbins, nbins_f, stream);
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
    SizeType branch_max,
    int device_id,
    math::CUFFTManager* external_fft)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max,
           device_id,
           external_fft) {}

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
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(), this->m_branch_max,
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
    const auto n_accel_init = this->m_n_accel_init;
    const auto n_freq_init  = this->m_n_freq_init;
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
    const auto n_accel_init  = this->m_n_accel_init;
    const auto n_freq_init   = this->m_n_freq_init;
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
        core::shift_add_ascend_linear_complex_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments, stream);
        const auto nfft = 2 * n_leaves;
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_r_d).first(nfft * nbins);
        irfft_folds_for_scoring(this->get_fft(), this->m_scratch_folds_c_d,
                                this->m_scratch_folds_r_d, folds_tree, nfft,
                                nbins, nbins_f, stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);

    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_batch_cuda(
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
    SizeType branch_max,
    int device_id,
    math::CUFFTManager* external_fft)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max,
           device_id,
           external_fft) {}

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
        this->m_branch_max, n_leaves, this->m_cfg.get_nparams(), branch_ws,
        scratch_ws, stream);
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
    const auto n_accel_init = this->m_n_accel_init;
    const auto n_freq_init  = this->m_n_freq_init;
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
    const auto n_accel_init  = this->m_n_accel_init;
    const auto n_freq_init   = this->m_n_freq_init;
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
        core::shift_add_ascend_linear_complex_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments, stream);
        const auto nfft = 2 * n_leaves;
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_r_d).first(nfft * nbins);
        irfft_folds_for_scoring(this->get_fft(), this->m_scratch_folds_c_d,
                                this->m_scratch_folds_r_d, folds_tree, nfft,
                                nbins, nbins_f, stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);

    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_batch_cuda(
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
    SizeType branch_max,
    int device_id,
    math::CUFFTManager* external_fft)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max,
           device_id,
           external_fft) {}

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
        this->m_cfg.get_nbins(), this->m_cfg.get_eta(), this->m_branch_max,
        n_leaves, this->m_cfg.get_propagator_significance(), branch_ws,
        scratch_ws, stream);
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
        this->m_cfg.get_x_mass_const(),
        this->m_cfg.get_validation_significance(), scratch_ws, stream);
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
    const auto n_accel_init = this->m_n_accel_init;
    const auto n_freq_init  = this->m_n_freq_init;
    circ_taylor_resolve_batch_cuda(
        leaves_branch, validation_mask, param_indices, phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_add, coord_cur,
        coord_init, n_accel_init, n_freq_init, this->m_cfg.get_nbins(),
        n_leaves, this->m_cfg.get_propagator_significance(), stream);
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
        this->m_cfg.get_propagator_significance(), stream);
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
    const auto n_accel_init  = this->m_n_accel_init;
    const auto n_freq_init   = this->m_n_freq_init;
    const auto nbins         = this->m_cfg.get_nbins();
    const auto nbins_f       = this->m_cfg.get_nbins_f();
    const auto n_segments    = idx_segments.size();
    const auto n_coords_init = this->m_n_coords_init;

    circ_taylor_ascend_resolve_batch_cuda(
        leaves_tree, scratch_param_indices, scratch_phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_segments, coord_cur,
        n_accel_init, n_freq_init, nbins, n_leaves, n_segments,
        this->m_cfg.get_propagator_significance(), stream);

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
        core::shift_add_ascend_linear_complex_batch_cuda(
            folds_ffa.data(), idx_segments.data(), scratch_param_indices.data(),
            scratch_phase_shift.data(), folds_tree.data(), nbins_f, nbins,
            n_coords_init, n_leaves, n_segments, stream);
        const auto nfft = 2 * n_leaves;
        auto folds_t_span =
            cuda_utils::as_span(this->m_scratch_folds_r_d).first(nfft * nbins);
        irfft_folds_for_scoring(this->get_fft(), this->m_scratch_folds_c_d,
                                this->m_scratch_folds_r_d, folds_tree, nfft,
                                nbins, nbins_f, stream);
        detection::snr_boxcar_3d_max_cuda_d(
            folds_t_span, cuda_utils::as_span(this->m_boxcar_widths_d),
            scores_tree, n_leaves, nbins, stream);

    } else {
        error_check::check_equal(folds_tree.size(), n_leaves * 2 * nbins,
                                 "fold_segment size mismatch");
        core::shift_add_ascend_linear_batch_cuda(
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
                            SizeType branch_max,
                            int device_id,
                            math::CUFFTManager* external_fft) {
    const auto n_params = cfg.get_nparams();
    if (poly_basis == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max, device_id, external_fft);
    }
    if (poly_basis == "taylor" && n_params == 5) {
        return std::make_unique<PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max, device_id, external_fft);
    }
    if (poly_basis == "chebyshev" && n_params <= 4) {
        return std::make_unique<PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max, device_id, external_fft);
    }
    throw std::runtime_error(std::format(
        "Unknown poly_basis: '{}'. Valid options: 'taylor', 'chebyshev'",
        poly_basis));
}

// Explicit template instantiations
// Base classes need explicit instantiation for linker
template class BasePruneDPFunctsCUDA<float>;
template class BasePruneDPFunctsCUDA<ComplexTypeCUDA>;
template class BaseTaylorPruneDPFunctsCUDA<float>;
template class BaseTaylorPruneDPFunctsCUDA<ComplexTypeCUDA>;

template class BaseChebyshevPruneDPFunctsCUDA<float>;
template class BaseChebyshevPruneDPFunctsCUDA<ComplexTypeCUDA>;

// Leaf classes
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
                                   SizeType,
                                   int,
                                   math::CUFFTManager*);
template std::unique_ptr<PruneDPFunctsCUDA<ComplexTypeCUDA>>
create_prune_dp_functs_cuda<ComplexTypeCUDA>(std::string_view,
                                             std::span<const SizeType>,
                                             std::span<const double>,
                                             SizeType,
                                             double,
                                             search::PulsarSearchConfig,
                                             SizeType,
                                             SizeType,
                                             int,
                                             math::CUFFTManager*);

} // namespace loki::core
