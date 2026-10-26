#include "lib/cuda/dynamic_cuda.cuh"

#include <bit>
#include <limits>

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <thrust/copy.h>

#include "loki/common/types.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/cuda/chebyshev_cuda.cuh"
#include "lib/cuda/circular_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/ep_score_cuda.cuh"
#include "lib/cuda/ffa_score_cuda.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/kadane_cuda.cuh"
#include "lib/cuda/kernels_cuda.cuh"
#include "lib/cuda/score_cuda.cuh"
#include "lib/cuda/taylor_cuda.cuh"
#include "lib/detail/error_check.hpp"
#include "lib/detail/utils.hpp"
#include "lib/search/ep_support.hpp"

namespace loki::core {

template <SupportedFoldTypeCUDA FoldTypeCUDA>
BasePruneDPFunctsCUDA<FoldTypeCUDA>::BasePruneDPFunctsCUDA(
    std::span<const SizeType> param_grid_count_init,
    std::span<const double> dparams_init,
    SizeType nseg_ffa,
    double tseg_ffa,
    search::PulsarSearchConfig cfg,
    SizeType batch_size,
    SizeType branch_max,
    int device_id)
    : m_nseg_ffa(nseg_ffa),
      m_tseg_ffa(tseg_ffa),
      m_cfg(std::move(cfg)),
      m_batch_size(batch_size),
      m_branch_max(branch_max) {
    (void)device_id;
    search::detail::check_ep_supported(m_cfg, "CUDA EP");
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

    const auto nbins = m_cfg.get_nbins();
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        if (ffa_fourier_needs_bluestein(nbins)) {
            const auto table = ffa_fourier_bluestein_table(nbins);
            m_bluestein_d.assign(table.begin(), table.end());
        }
    }
    // At most one tile per leaf (the memory model's bound).
    m_tile_state_d.resize(kStatsWords + (m_batch_size * m_branch_max) + 1);
    {
        uint32_t* stats_h = nullptr;
        cuda_utils::check_cuda_call(
            cudaMallocHost(&stats_h, kStatsWords * sizeof(uint64_t)),
            "BasePruneDPFunctsCUDA: cudaMallocHost failed");
        m_stats_h         = {stats_h, [](uint32_t* p) { cudaFreeHost(p); }};
        cudaEvent_t ready = nullptr;
        cuda_utils::check_cuda_call(
            cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
            "BasePruneDPFunctsCUDA: cudaEventCreate failed");
        m_stats_ready = {ready, [](cudaEvent_t e) { cudaEventDestroy(e); }};
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void BasePruneDPFunctsCUDA<FoldTypeCUDA>::score_complex_rows(
    cuda::std::span<const ComplexTypeCUDA> folds,
    SizeType n_rows,
    cuda::std::span<float> scores,
    cudaStream_t stream)
    requires(std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>)
{
    const auto nbins = m_cfg.get_nbins();
    ep_score_rows_fourier_cuda(
        folds.data(), n_rows, nbins,
        thrust::raw_pointer_cast(m_boxcar_widths_d.data()),
        m_boxcar_widths_d.size(), scores.data(),
        m_bluestein_d.empty() ? nullptr
                              : thrust::raw_pointer_cast(m_bluestein_d.data()),
        stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
float BasePruneDPFunctsCUDA<FoldTypeCUDA>::get_scorer_scratch_memory_gib()
    const noexcept {
    const auto bytes = m_tile_state_d.size() * sizeof(unsigned long long);
    return static_cast<float>(bytes) / static_cast<float>(utils::kBytesPerGB);
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
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA>::shift_add_score(
    cuda::std::span<const FoldTypeCUDA> folds_tree,
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
    const EPDirectWriteCUDA<FoldTypeCUDA>* direct,
    MinMaxFloat* minmax) {
    (void)scratch_ws;
    auto* const stats_d = reinterpret_cast<uint32_t*>(
        thrust::raw_pointer_cast(m_tile_state_d.data()));
    // The statistics, and in direct mode the tile states after them.
    SizeType words = kStatsWords;
    if (direct != nullptr) {
        words += ep_shift_add_score_tile_words<FoldTypeCUDA>(
            m_cfg.get_nbins(), n_leaves, m_cfg.get_use_boxcar_kadane());
        error_check::check_less_equal(
            words, m_tile_state_d.size(),
            "shift_add_score: batch larger than the tile states");
    }
    cuda_utils::check_cuda_call(
        cudaMemsetAsync(stats_d, 0, words * sizeof(unsigned long long), stream),
        "shift_add_score: statistics reset failed");
    EPScoreArgs<FoldTypeCUDA> args{
        .folds_tree      = folds_tree.data(),
        .indices_tree    = indices_tree.data(),
        .validation_mask = validation_mask.data(),
        .folds_ffa       = folds_ffa.data(),
        .indices_ffa     = indices_ffa.data(),
        .phase_shift     = phase_shift.data(),
        .folds_out       = folds_out.data(),
        .scores          = scores.data(),
        .filtered_mask   = filtered_mask.data(),
        .stats           = stats_d,
        .widths          = thrust::raw_pointer_cast(m_boxcar_widths_d.data()),
        .nwidths         = static_cast<uint32_t>(m_boxcar_widths_d.size()),
        .threshold       = threshold,
        .n_leaves        = static_cast<uint32_t>(n_leaves),
        .physical_start_idx = static_cast<uint32_t>(physical_start_idx),
        .capacity           = static_cast<uint32_t>(capacity),
        .nbins              = static_cast<uint32_t>(m_cfg.get_nbins()),
        .bluestein = m_bluestein_d.empty()
                         ? nullptr
                         : thrust::raw_pointer_cast(m_bluestein_d.data()),
    };
    if (m_cfg.get_use_boxcar_kadane()) {
        args.kadane_biases =
            thrust::raw_pointer_cast(m_boxcar_kadane_biases_d.data());
        args.nbiases = static_cast<uint32_t>(m_boxcar_kadane_biases_d.size());
    }
    if (direct != nullptr) {
        args.leaves_batch    = direct->leaves_batch.data();
        args.leaves_stride   = static_cast<uint32_t>(direct->leaves_stride);
        args.tree_leaves     = direct->leaves;
        args.tree_folds      = direct->folds;
        args.tree_scores     = direct->scores;
        args.tree_write_head = static_cast<uint32_t>(direct->write_head);
        args.tile_state =
            thrust::raw_pointer_cast(m_tile_state_d.data()) + kStatsWords;
    }
    ep_shift_add_score_cuda<FoldTypeCUDA>(args, stream);
    cuda_utils::check_cuda_call(cudaMemcpyAsync(m_stats_h.get(), stats_d,
                                                3 * sizeof(uint32_t),
                                                cudaMemcpyDeviceToHost, stream),
                                "shift_add_score: statistics copy failed");
    if (direct != nullptr) {
        cuda_utils::check_cuda_call(
            cudaEventRecord(m_stats_ready.get(), stream),
            "shift_add_score: cudaEventRecord failed");
        return 0;
    }
    cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                "shift_add_score: sync failed");
    return decode_stats(minmax);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType
BasePruneDPFunctsCUDA<FoldTypeCUDA>::decode_stats(MinMaxFloat* minmax) const {
    const uint32_t* const stats_h = m_stats_h.get();
    if (minmax != nullptr) {
        *minmax = {std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::lowest()};
        if (stats_h[1] != 0) {
            minmax->max = std::bit_cast<float>(ep_score_key_bits(stats_h[1]));
            minmax->min = std::bit_cast<float>(ep_score_key_bits(~stats_h[2]));
        }
    }
    return stats_h[0];
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
SizeType BasePruneDPFunctsCUDA<FoldTypeCUDA>::collect_shift_add_score(
    MinMaxFloat& minmax) {
    cuda_utils::check_cuda_call(cudaEventSynchronize(m_stats_ready.get()),
                                "collect_shift_add_score: sync failed");
    return decode_stats(&minmax);
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
        this->score_complex_rows(fold_segment, this->m_n_coords_init,
                                 seed_scores, stream);
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
        this->score_complex_rows(fold_segment, this->m_n_coords_init,
                                 seed_scores, stream);
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
    int device_id)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max,
           device_id) {}

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
void PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>::resolve_transform(
    cuda::std::span<double> leaves_branch,
    cuda::std::span<const uint8_t> validation_mask,
    cuda::std::span<uint32_t> param_indices,
    cuda::std::span<float> phase_shift,
    std::pair<double, double> coord_add,
    std::pair<double, double> coord_cur,
    std::pair<double, double> coord_init,
    std::pair<double, double> coord_next,
    SizeType n_leaves,
    cudaStream_t stream) const {
    poly_taylor_resolve_transform_batch_cuda(
        leaves_branch, validation_mask, param_indices, phase_shift,
        cuda_utils::as_span(this->m_param_limits_d), coord_add, coord_cur,
        coord_init, coord_next, this->m_n_accel_init, this->m_n_freq_init,
        this->m_cfg.get_nbins(), n_leaves, this->m_cfg.get_nparams(),
        this->m_cfg.get_use_conservative_tile(), stream);
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
        this->score_complex_rows(folds_tree, n_leaves, scores_tree, stream);

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
    int device_id)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max,
           device_id) {}

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
        this->score_complex_rows(folds_tree, n_leaves, scores_tree, stream);

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
    int device_id)
    : Base(param_grid_count_init,
           dparams_init,
           nseg_ffa,
           tseg_ffa,
           std::move(cfg),
           batch_size,
           branch_max,
           device_id) {}

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
        this->score_complex_rows(folds_tree, n_leaves, scores_tree, stream);

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
                            int device_id) {
    const auto n_params = cfg.get_nparams();
    if (poly_basis == "taylor" && n_params <= 4) {
        return std::make_unique<PrunePolyTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max, device_id);
    }
    if (poly_basis == "taylor" && n_params == 5) {
        return std::make_unique<PruneCircTaylorDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max, device_id);
    }
    if (poly_basis == "chebyshev" && n_params <= 4) {
        return std::make_unique<PrunePolyChebyshevDPFunctsCUDA<FoldTypeCUDA>>(
            param_grid_count_init, dparams_init, nseg_ffa, tseg_ffa,
            std::move(cfg), batch_size, branch_max, device_id);
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
                                   int);
template std::unique_ptr<PruneDPFunctsCUDA<ComplexTypeCUDA>>
create_prune_dp_functs_cuda<ComplexTypeCUDA>(std::string_view,
                                             std::span<const SizeType>,
                                             std::span<const double>,
                                             SizeType,
                                             double,
                                             search::PulsarSearchConfig,
                                             SizeType,
                                             SizeType,
                                             int);

} // namespace loki::core
