#include "loki/core/dynamic.hpp"

namespace loki::core {

namespace {
template <SupportedFoldTypeCUDA FoldTypeCUDA>
__global__ void seed_kernel(const FoldTypeCUDA* __restrict__ ffa_fold,
                            cuda::std::pair<double, double> coord_init,
                            const double* __restrict__ param_arr,
                            const double* __restrict__ dparams,
                            int poly_order,
                            int nbins,
                            double* __restrict__ tree_leaves,
                            FoldTypeCUDA* __restrict__ tree_folds,
                            float* __restrict__ tree_scores,
                            int n_leaves) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_leaves) {
        return;
    }
    int n_param_sets = 1;
    for (int i = 0; i < poly_order; ++i) {
        n_param_sets *= param_arr[(i * 2)];
    }
    const auto param_set = idx * n_param_sets;
    for (int i = 0; i < n_param_sets; ++i) {
        tree_leaves[idx * n_param_sets + i] = param_arr[i][param_set + i];
    }
    for (int i = 0; i < nbins; ++i) {
        tree_folds[(idx * nbins) + i] = ffa_fold[(idx * nbins) + i];
    }
    tree_scores[idx] = 0.0F;
}
} // namespace

template <SupportedFoldTypeCUDA FoldTypeCUDA>
PruneDPFunctsCUDA<FoldTypeCUDA>::PruneDPFunctsCUDA(
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
      m_branch_max(branch_max) {
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        const auto max_batch_size = m_batch_size * m_branch_max;
        m_folds_buffer_d.resize(max_batch_size * 2 * m_cfg.get_nbins());
    } else {
        m_folds_buffer_d.resize(1); // Not needed for float
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
cuda::std::span<const FoldTypeCUDA>
PruneDPFunctsCUDA<FoldTypeCUDA>::load_segment(
    cuda::std::span<const FoldTypeCUDA> ffa_fold, SizeType seg_idx) const {
    SizeType n_coords = 1;
    for (const auto& arr : m_param_arr) {
        n_coords *= arr.size();
    }
    const auto nbins   = m_cfg.get_nbins();
    const auto nbins_f = m_cfg.get_nbins_f();
    if constexpr (std::is_same_v<FoldTypeCUDA, ComplexTypeCUDA>) {
        return ffa_fold.subspan(seg_idx * n_coords * 2 * nbins_f,
                                n_coords * 2 * nbins_f);
    } else {
        return ffa_fold.subspan(seg_idx * n_coords * 2 * nbins,
                                n_coords * 2 * nbins);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void PruneDPFunctsCUDA<FoldTypeCUDA>::seed(
    cuda::std::span<const FoldTypeCUDA> ffa_fold,
    std::pair<double, double> coord_init,
    cuda::std::span<double> seed_leaves,
    cuda::std::span<float> seed_scores) {
    const auto poly_order = m_cfg.get_prune_poly_order();
    const auto nbins      = m_cfg.get_nbins();
    const auto seg_idx    = m_snail_scheme->get_ref_idx();

    const auto leaves_stride = m_world_tree->get_leaves_stride();
    thrust::device_vector<double> leaves_d(ncoords * leaves_stride);
    thrust::device_vector<float> scores_d(ncoords);

    const auto tree_folds = load_segment(ffa_fold, seg_idx);
    auto tree_leaves      = cuda::std::span<double>(
        thrust::raw_pointer_cast(leaves_d.data()), ncoords * leaves_stride);
    auto tree_scores = cuda::std::span<float>(
        thrust::raw_pointer_cast(scores_d.data()), ncoords);

    seed_kernel<FoldTypeCUDA><<<1, 1>>>(ffa_fold, coord_init, param_arr,
                                        dparams, poly_order, nbins, tree_leaves,
                                        tree_folds, tree_scores, ncoords);
}

} // namespace loki::core