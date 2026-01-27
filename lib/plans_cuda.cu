#include "loki/algorithms/plans.hpp"

#include "loki/common/coord.hpp"
#include "loki/core/taylor.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::plans {

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFAPlanCUDA<FoldTypeCUDA>::FFAPlanCUDA(
    const plans::FFAPlan<HostFoldType>& ffa_plan)
    : m_ffa_plan(std::make_unique<plans::FFAPlan<HostFoldType>>(ffa_plan)) {
    m_params_d       = m_ffa_plan->get_params_flat();
    m_param_counts_d = m_ffa_plan->get_param_counts_flat();
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAPlanCUDA<FoldTypeCUDA>::resolve_coordinates(coord::FFACoordD& coords_d,
                                                    cudaStream_t stream) {
    const auto& n_levels        = m_ffa_plan->get_n_levels();
    const auto& n_params        = m_ffa_plan->get_n_params();
    const auto& ncoords         = m_ffa_plan->get_ncoords();
    const auto& ncoords_offsets = m_ffa_plan->get_ncoords_offsets();
    const auto& tseg_brute      = m_ffa_plan->get_config().get_tseg_brute();
    const auto& nbins           = m_ffa_plan->get_config().get_nbins();
    const auto& [params_flat_offsets, params_flat_sizes] =
        m_ffa_plan->get_params_flat_sizes();

    for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
        const auto ncoords_cur = ncoords[i_level];
        const auto offset      = ncoords_offsets[i_level];

        auto coord_ptrs = coords_d.get_raw_ptrs();
        // Tail coordinates
        core::ffa_taylor_resolve_poly_batch_cuda(
            cuda_utils::as_span(m_params_d)
                .subspan(params_flat_offsets[i_level],
                         params_flat_sizes[i_level]),
            cuda_utils::as_span(m_param_counts_d)
                .subspan(i_level * n_params, n_params),
            cuda_utils::as_span(m_params_d)
                .subspan(params_flat_offsets[i_level - 1],
                         params_flat_sizes[i_level - 1]),
            cuda_utils::as_span(m_param_counts_d)
                .subspan((i_level - 1) * n_params, n_params),
            coord_ptrs.i_tail + offset, coord_ptrs.shift_tail + offset, i_level,
            0, tseg_brute, nbins, n_params, stream);

        // Head coordinates
        core::ffa_taylor_resolve_poly_batch_cuda(
            cuda_utils::as_span(m_params_d)
                .subspan(params_flat_offsets[i_level],
                         params_flat_sizes[i_level]),
            cuda_utils::as_span(m_param_counts_d)
                .subspan(i_level * n_params, n_params),
            cuda_utils::as_span(m_params_d)
                .subspan(params_flat_offsets[i_level - 1],
                         params_flat_sizes[i_level - 1]),
            cuda_utils::as_span(m_param_counts_d)
                .subspan((i_level - 1) * n_params, n_params),
            coord_ptrs.i_head + offset, coord_ptrs.shift_head + offset, i_level,
            1, tseg_brute, nbins, n_params, stream);
    }
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAPlanCUDA<FoldTypeCUDA>::resolve_coordinates_freq(
    coord::FFACoordFreqD& coords_d, cudaStream_t stream) {
    const auto& n_levels        = m_ffa_plan->get_n_levels();
    const auto& n_params        = m_ffa_plan->get_n_params();
    const auto& ncoords         = m_ffa_plan->get_ncoords();
    const auto& ncoords_offsets = m_ffa_plan->get_ncoords_offsets();
    const auto& tseg_brute      = m_ffa_plan->get_config().get_tseg_brute();
    const auto& nbins           = m_ffa_plan->get_config().get_nbins();
    const auto& [params_flat_offsets, params_flat_sizes] =
        m_ffa_plan->get_params_flat_sizes();

    for (SizeType i_level = 1; i_level < n_levels; ++i_level) {
        const auto ncoords_cur = ncoords[i_level];
        const auto offset      = ncoords_offsets[i_level];

        auto coord_ptrs = coords_d.get_raw_ptrs();
        core::ffa_taylor_resolve_poly_batch_cuda(
            cuda_utils::as_span(m_params_d)
                .subspan(params_flat_offsets[i_level],
                         params_flat_sizes[i_level]),
            cuda_utils::as_span(m_param_counts_d)
                .subspan(i_level * n_params, n_params),
            cuda_utils::as_span(m_params_d)
                .subspan(params_flat_offsets[i_level - 1],
                         params_flat_sizes[i_level - 1]),
            cuda_utils::as_span(m_param_counts_d)
                .subspan((i_level - 1) * n_params, n_params),
            cuda::std::span<uint32_t>(coord_ptrs.idx + offset, ncoords_cur),
            cuda::std::span<float>(coord_ptrs.shift + offset, ncoords_cur),
            i_level, 0, tseg_brute, nbins, n_params, stream);
    }
}

// Explicit instantiation
template class FFAPlanCUDA<ComplexTypeCUDA>;
template class FFAPlanCUDA<float>;
} // namespace loki::plans