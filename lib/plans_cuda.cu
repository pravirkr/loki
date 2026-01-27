#include "loki/algorithms/plans.hpp"

#include "loki/common/coord.hpp"
#include "loki/core/taylor.hpp"
#include "loki/cuda_utils.cuh"

namespace loki::plans {

template <SupportedFoldTypeCUDA FoldTypeCUDA>
FFAPlanCUDA<FoldTypeCUDA>::FFAPlanCUDA(
    const plans::FFAPlan<HostFoldType>& ffa_plan)
    : m_ffa_plan(std::make_unique<plans::FFAPlan<HostFoldType>>(ffa_plan)) {
    m_params_d                           = m_ffa_plan->get_params_flat();
    m_param_counts_d                     = m_ffa_plan->get_param_counts_flat();
    const auto& [params_flat_offsets, _] = m_ffa_plan->get_params_flat_sizes();
    m_params_flat_offsets_d              = params_flat_offsets;
    m_ncoords_offsets_d                  = m_ffa_plan->get_ncoords_offsets();
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAPlanCUDA<FoldTypeCUDA>::resolve_coordinates(coord::FFACoordD& coords_d,
                                                    cudaStream_t stream) {
    const auto n_levels      = m_ffa_plan->get_n_levels();
    const auto n_params      = m_ffa_plan->get_n_params();
    const auto ncoords_total = m_ffa_plan->get_coord_size();
    const auto tseg_brute    = m_ffa_plan->get_config().get_tseg_brute();
    const auto nbins         = m_ffa_plan->get_config().get_nbins();

    auto coord_ptrs = coords_d.get_raw_ptrs();
    // Tail coordinates
    core::ffa_taylor_resolve_poly_batch_cuda(
        cuda_utils::as_span(m_params_d), cuda_utils::as_span(m_param_counts_d),
        cuda_utils::as_span(m_params_flat_offsets_d),
        cuda_utils::as_span(m_ncoords_offsets_d), coord_ptrs, n_levels,
        ncoords_total, 0, tseg_brute, nbins, n_params, stream);

    // Head coordinates
    core::ffa_taylor_resolve_poly_batch_cuda(
        cuda_utils::as_span(m_params_d), cuda_utils::as_span(m_param_counts_d),
        cuda_utils::as_span(m_params_flat_offsets_d),
        cuda_utils::as_span(m_ncoords_offsets_d), coord_ptrs, n_levels,
        ncoords_total, 1, tseg_brute, nbins, n_params, stream);
}

template <SupportedFoldTypeCUDA FoldTypeCUDA>
void FFAPlanCUDA<FoldTypeCUDA>::resolve_coordinates_freq(
    coord::FFACoordFreqD& coords_d, cudaStream_t stream) {
    const auto n_levels      = m_ffa_plan->get_n_levels();
    const auto ncoords_total = m_ffa_plan->get_coord_size();
    const auto tseg_brute    = m_ffa_plan->get_config().get_tseg_brute();
    const auto nbins         = m_ffa_plan->get_config().get_nbins();

    auto coord_ptrs = coords_d.get_raw_ptrs();
    core::ffa_taylor_resolve_freq_batch_cuda(
        cuda_utils::as_span(m_params_d), cuda_utils::as_span(m_param_counts_d),
        cuda_utils::as_span(m_params_flat_offsets_d),
        cuda_utils::as_span(m_ncoords_offsets_d), coord_ptrs, n_levels,
        ncoords_total, tseg_brute, nbins, stream);
}

// Explicit instantiation
template class FFAPlanCUDA<ComplexTypeCUDA>;
template class FFAPlanCUDA<float>;
} // namespace loki::plans