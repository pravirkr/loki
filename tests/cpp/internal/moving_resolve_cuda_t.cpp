#ifdef LOKI_ENABLE_CUDA
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"

#include "lib/cuda/chebyshev_cuda.cuh"
#include "lib/cuda/circular_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/taylor_cuda.cuh"

#include "moving_resolve.hpp"

// The GPU counterpart of moving_resolve_t.cpp.

using namespace moving_resolve;

namespace {

enum class Basis : std::uint8_t { kTaylor, kChebyshev, kCircular };

Resolver cuda_resolver(Basis basis, SizeType n_params) {
    return [basis, n_params](const std::vector<double>& leaves, SizeType n,
                             double t_add, double t_cur,
                             std::vector<SizeType>& idx,
                             std::vector<float>& phase) {
        const auto lim = limits(n_params);
        const thrust::device_vector<double> leaves_d(leaves);
        const thrust::device_vector<uint8_t> mask_d(n, 1);
        const thrust::device_vector<ParamLimit> lim_d(lim.begin(), lim.end());
        thrust::device_vector<uint32_t> idx_d(n);
        thrust::device_vector<float> phase_d(n);
        const auto l  = loki::cuda_utils::as_span(leaves_d);
        const auto m  = loki::cuda_utils::as_span(mask_d);
        const auto i  = loki::cuda_utils::as_span(idx_d);
        const auto p  = loki::cuda_utils::as_span(phase_d);
        const auto pl = loki::cuda_utils::as_span(lim_d);
        const std::pair<double, double> add{t_add, kScale};
        const std::pair<double, double> cur{t_cur, kScale};
        const std::pair<double, double> init{kTInit, kScale};
        switch (basis) {
        case Basis::kTaylor:
            loki::core::poly_taylor_resolve_batch_cuda(
                l, m, i, p, pl, add, cur, init, kNAccel, kNFreq, kNbins, n,
                n_params, nullptr);
            break;
        case Basis::kChebyshev:
            loki::core::poly_chebyshev_resolve_batch_cuda(
                l, m, i, p, pl, add, cur, init, kNAccel, kNFreq, kNbins, n,
                n_params, nullptr);
            break;
        case Basis::kCircular:
            loki::core::circ_taylor_resolve_batch_cuda(
                l, m, i, p, pl, add, cur, init, kNAccel, kNFreq, kNbins, n,
                kSignif, nullptr);
            break;
        }
        loki::cuda_utils::check_cuda_call(cudaDeviceSynchronize(),
                                          "resolve failed");
        std::vector<uint32_t> idx_h(n);
        thrust::copy(idx_d.begin(), idx_d.end(), idx_h.begin());
        idx.assign(idx_h.begin(), idx_h.end());
        phase.resize(n);
        thrust::copy(phase_d.begin(), phase_d.end(), phase.begin());
    };
}

void check(const std::vector<Leaf>& cases, const Resolver& resolve) {
    for (const double t_cur : {kTInit, 18.3719}) {
        INFO("t_cur " << t_cur);
        const auto bad = failures(cases, resolve, t_cur);
        for (const auto& b : bad) {
            UNSCOPED_INFO(b);
        }
        CHECK(bad.empty());
    }
}

} // namespace

TEST_CASE("The CUDA moving-grid resolves pick the cell of the leaf's absolute "
          "frequency",
          "[resolve][cuda]") {
    if (!loki::is_available(loki::Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    for (const SizeType n_params : {2, 3, 4}) {
        INFO("n_params " << n_params);
        check(taylor_leaves(n_params), cuda_resolver(Basis::kTaylor, n_params));
        check(chebyshev_leaves(n_params),
              cuda_resolver(Basis::kChebyshev, n_params));
    }
    check(circular_leaves(), cuda_resolver(Basis::kCircular, 5));
}
#endif // LOKI_ENABLE_CUDA
