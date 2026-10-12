#ifdef LOKI_ENABLE_CUDA
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/cuda/coord_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/taylor_ffa_cuda.cuh"

using loki::Backend;
using loki::ParamLimit;
using loki::SizeType;

namespace {

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d) {
    std::vector<T> h(d.size());
    loki::cuda_utils::check_cuda_call(
        cudaMemcpy(h.data(), thrust::raw_pointer_cast(d.data()),
                   d.size() * sizeof(T), cudaMemcpyDeviceToHost),
        "test D2H failed");
    return h;
}

template <typename T>
bool bit_equal(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// The table-based accel resolve against the two per-half kernels, on the
// plan of a real accel search chunk.
void check_accel_resolve(
    double f_min, double f_max, double a_max, SizeType nsamps, SizeType nbins) {
    const std::vector<ParamLimit> limits{{-a_max, a_max}, {f_min, f_max}};
    const loki::search::FFASearchConfig cfg(nsamps, 64e-6, nbins, 1.0, limits,
                                            0.2, 1.5, false);
    const loki::plans::FFAPlan<float> plan(cfg);
    const auto n_levels = plan.get_n_levels();
    const auto ncoords  = plan.get_coord_size();
    const auto counts   = plan.get_param_counts_flat();
    const auto offsets  = plan.get_ncoords_offsets();
    const auto& pc      = plan.get_param_counts();
    std::vector<uint32_t> row_col(2 * (n_levels + 1), 0U);
    SizeType rows = 0;
    SizeType cols = 0;
    for (SizeType l = 0; l < n_levels; ++l) {
        row_col[l]                = static_cast<uint32_t>(rows);
        row_col[n_levels + 1 + l] = static_cast<uint32_t>(cols);
        if (l > 0) {
            rows += pc[l][0];
            cols += pc[l][1];
        }
    }
    row_col[n_levels]           = static_cast<uint32_t>(rows);
    row_col[(2 * n_levels) + 1] = static_cast<uint32_t>(cols);

    const thrust::device_vector<uint32_t> counts_d(counts.begin(),
                                                   counts.end());
    const thrust::device_vector<uint32_t> offsets_d(offsets.begin(),
                                                    offsets.end());
    const thrust::device_vector<ParamLimit> limits_d(limits.begin(),
                                                     limits.end());
    const thrust::device_vector<uint32_t> row_col_d(row_col.begin(),
                                                    row_col.end());
    thrust::device_vector<loki::core::FFAAccelResolveRow> rows_d(rows);
    thrust::device_vector<double> cols_d(cols);
    loki::coord::FFACoordD ref;
    loki::coord::FFACoordD fast;
    ref.resize(ncoords);
    fast.resize(ncoords);
    const auto tseg = cfg.get_tseg_brute();
    for (SizeType latter = 0; latter < 2; ++latter) {
        loki::core::ffa_taylor_resolve_poly_batch_cuda(
            loki::cuda_utils::as_span(counts_d),
            loki::cuda_utils::as_span(offsets_d),
            loki::cuda_utils::as_span(limits_d), ref.get_raw_ptrs(), n_levels,
            ncoords, latter, tseg, nbins, 2, nullptr);
    }
    loki::core::ffa_taylor_resolve_accel_tables_cuda(
        loki::cuda_utils::as_span(counts_d),
        loki::cuda_utils::as_span(offsets_d),
        loki::cuda_utils::as_span(limits_d),
        loki::cuda_utils::as_span(row_col_d),
        thrust::raw_pointer_cast(rows_d.data()),
        thrust::raw_pointer_cast(cols_d.data()), fast.get_raw_ptrs(), n_levels,
        ncoords, rows, cols, tseg, nbins, nullptr);
    // Level 0 entries are never written by either path.
    const auto skip = static_cast<std::ptrdiff_t>(offsets[1]);
    auto tail = [&](auto v) { return decltype(v)(v.begin() + skip, v.end()); };
    INFO("f [" << f_min << ", " << f_max << "] a_max " << a_max << " ncoords "
               << ncoords);
    CHECK(bit_equal(tail(to_host(ref.i_tail)), tail(to_host(fast.i_tail))));
    CHECK(bit_equal(tail(to_host(ref.i_head)), tail(to_host(fast.i_head))));
    CHECK(bit_equal(tail(to_host(ref.shift_tail)),
                    tail(to_host(fast.shift_tail))));
    CHECK(bit_equal(tail(to_host(ref.shift_head)),
                    tail(to_host(fast.shift_head))));
}

} // namespace

TEST_CASE("Table-based accel resolve matches the per-half kernels bit for bit",
          "[ffa][cuda][resolve]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    check_accel_resolve(99.93, 100.036, 100.0, SizeType{1} << 25U, 64);
    check_accel_resolve(81.0, 81.6, 50.0, SizeType{1} << 23U, 64);
    check_accel_resolve(12.5, 14.0, 200.0, SizeType{1} << 23U, 256);
    check_accel_resolve(30.0, 31.0, 5.0, SizeType{1} << 22U, 128);
}
#endif // LOKI_ENABLE_CUDA
