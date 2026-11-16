#ifdef LOKI_ENABLE_CUDA
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"

#include "lib/core/circular.hpp"
#include "lib/cuda/circular_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/utils/workspace_impl.hpp"

#include "circ_branch_replay.hpp"

// The GPU counterpart of circ_branch_bound_t.cpp: the CUDA circular branch,
// given the same parents level by level, gives every parent as many children
// as the CPU kernel, and the worst parent has BranchingForecast::max_children.

namespace {

using loki::SizeType;
using loki::test::kCircBatch;
using loki::test::kCircParams;
using loki::test::kCircSlots;
using loki::test::kCircStride;

void check_cuda_bound(bool conservative) {
    const auto cfg = loki::test::make_circ_cfg(conservative);
    const loki::plans::FFAPlan<loki::ComplexType> plan(cfg);
    const auto forecast = plan.forecast_branching("taylor", 0);

    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    cudaStream_t stream     = nullptr;
    const SizeType capacity = kCircBatch * kCircSlots;
    thrust::device_vector<double> tree_d(kCircBatch * kCircStride);
    thrust::device_vector<double> branch_d(capacity * kCircStride);
    thrust::device_vector<uint32_t> origins_d(capacity);
    thrust::device_vector<uint8_t> mask_d(capacity);
    loki::memory::BranchingWorkspaceCUDA ws_d(kCircBatch, kCircSlots,
                                              kCircParams);
    loki::memory::CUBScratchArena scratch(kCircBatch, kCircSlots, stream);
    loki::memory::BranchingWorkspace ws(kCircBatch, kCircSlots, kCircParams);
    std::vector<double> cpu_children(capacity * kCircStride);
    std::vector<SizeType> cpu_origins(capacity);
    std::vector<uint32_t> origins_h(capacity);
    bool counts_match = true;

    const SizeType max_children = loki::test::replay_circ_widths(
        cfg, plan,
        [&](std::span<const double> tree, SizeType nb, SizeType /*level*/,
            std::pair<double, double> coord_cur, std::vector<double>& children,
            std::vector<SizeType>& origins) {
            thrust::copy(tree.data(), tree.data() + tree.size(),
                         tree_d.begin());
            const SizeType n_out = loki::core::circ_taylor_branch_batch_cuda(
                loki::cuda_utils::as_span(tree_d, nb * kCircStride),
                loki::cuda_utils::as_span(branch_d),
                loki::cuda_utils::as_span(origins_d),
                loki::cuda_utils::as_span(mask_d), coord_cur, cfg.get_nbins(),
                cfg.get_eta(), kCircSlots, nb,
                cfg.get_propagator_significance(), ws_d.get_view(), scratch,
                stream);
            loki::cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                              "branch sync failed");
            thrust::copy(branch_d.begin(),
                         branch_d.begin() +
                             static_cast<long>(n_out * kCircStride),
                         children.begin());
            thrust::copy(origins_d.begin(),
                         origins_d.begin() + static_cast<long>(n_out),
                         origins_h.begin());
            std::vector<SizeType> gpu_count(nb, 0);
            for (SizeType c = 0; c < n_out; ++c) {
                origins[c] = origins_h[c];
                ++gpu_count[origins[c]];
            }
            const SizeType n_cpu = loki::core::circ_taylor_branch_batch(
                tree, cpu_children, cpu_origins, coord_cur, cfg.get_nbins(),
                cfg.get_eta(), kCircSlots, nb,
                cfg.get_propagator_significance(), ws);
            std::vector<SizeType> cpu_count(nb, 0);
            for (SizeType c = 0; c < n_cpu; ++c) {
                ++cpu_count[cpu_origins[c]];
            }
            counts_match = counts_match && gpu_count == cpu_count;
            return n_out;
        });
    REQUIRE(cudaGetLastError() == cudaSuccess);
    INFO("conservative " << conservative << ", forecast "
                         << forecast.max_children << ", CUDA kernels "
                         << max_children);
    CHECK(counts_match);
    REQUIRE(max_children < kCircSlots);
    CHECK(max_children == forecast.max_children);
}

} // namespace

TEST_CASE("CUDA circular branching stays within the forecast branch_max",
          "[ep_branching][circular][cuda]") {
    if (!loki::is_available(loki::Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    check_cuda_bound(false);
    check_cuda_bound(true);
}
#endif
