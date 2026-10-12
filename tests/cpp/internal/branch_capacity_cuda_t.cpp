#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/core/taylor.hpp"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/taylor_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/utils/workspace_impl.hpp"

using Catch::Matchers::WithinRel;
using loki::SizeType;

// The GPU counterpart of branch_capacity_t.cpp: the CUDA Taylor branch must
// never write a child past the branch workspace, and must reject a batch that
// does not fit before writing anything, as the CPU kernels do.

namespace {

constexpr SizeType kParams    = 3; // jerk, accel, freq
constexpr SizeType kStride    = (kParams + 2) * 2;
constexpr SizeType kNbins     = 64;
constexpr double kEta         = 1.0;
constexpr double kDt          = 100.0;
constexpr double kF0          = 100.0;
constexpr double kC           = 299792458.0;
constexpr double kCanaryVal   = -12345.0;
constexpr uint32_t kCanaryIx  = 987654321U;
constexpr uint8_t kCanaryMask = 7;
// Spare slots behind the span passed to the kernel. Large enough that an
// unguarded write of every child of the overflow case still lands in them.
constexpr SizeType kSpare = 1024;

// Leaves whose current steps are `ratio` times the steps the next stage needs,
// so each of the three parameters branches about `ratio` ways.
std::vector<double> make_leaves(SizeType n_leaves, double ratio) {
    const double dphi    = kEta / static_cast<double>(kNbins);
    const double dfactor = kC / kF0;
    const double d3_new  = dphi * dfactor * 24.0 / (kDt * kDt * kDt);
    const double d2_new  = dphi * dfactor * 4.0 / (kDt * kDt);
    const double d1_new  = dphi * dfactor * 1.0 / kDt;
    std::vector<double> leaves(n_leaves * kStride, 0.0);
    for (SizeType i = 0; i < n_leaves; ++i) {
        const SizeType lo = i * kStride;
        leaves[lo + 1]    = ratio * d3_new;
        leaves[lo + 3]    = ratio * d2_new;
        leaves[lo + 5]    = ratio * d1_new;
        leaves[lo + 8]    = kF0;
    }
    return leaves;
}

struct DeviceRun {
    bool threw{false};
    SizeType returned{0};
    bool canaries_intact{false};
    SizeType leaves_past_span{0};  // canary leaf slots that were overwritten
    SizeType mask_past_span{0};    // canary mask bytes that were overwritten
    std::vector<double> leaves;    // the capacity slots
    std::vector<uint32_t> origins; // the capacity slots
};

// Branch on the device into spans of exactly `capacity` leaves, followed by
// canaries that a write past the span would overwrite.
DeviceRun branch_on_device(const std::vector<double>& leaves,
                           SizeType n_leaves,
                           SizeType capacity,
                           SizeType branch_max) {
    // The kernels' launch checks read the device the guard selects, as the
    // prune functors do before calling them.
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    cudaStream_t stream = nullptr;
    const thrust::device_vector<double> leaves_d(leaves);
    thrust::device_vector<double> branch_d((capacity + kSpare) * kStride,
                                           kCanaryVal);
    thrust::device_vector<uint32_t> origins_d(capacity + kSpare, kCanaryIx);
    thrust::device_vector<uint8_t> mask_d(capacity + kSpare, kCanaryMask);
    loki::memory::BranchingWorkspaceCUDA ws(n_leaves, branch_max, kParams);
    loki::memory::CUBScratchArena scratch(n_leaves, branch_max, stream);

    DeviceRun run;
    try {
        run.returned = loki::core::poly_taylor_branch_batch_cuda(
            loki::cuda_utils::as_span(leaves_d),
            loki::cuda_utils::as_span(branch_d, capacity * kStride),
            loki::cuda_utils::as_span(origins_d, capacity),
            loki::cuda_utils::as_span(mask_d, capacity), {0.0, kDt}, kNbins,
            kEta, branch_max, n_leaves, kParams, ws.get_view(), scratch,
            stream);
    } catch (const std::runtime_error&) {
        run.threw = true;
    }
    loki::cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                      "branch sync failed");
    REQUIRE(cudaGetLastError() == cudaSuccess);

    std::vector<double> branch_h(branch_d.size());
    std::vector<uint32_t> origins_h(origins_d.size());
    std::vector<uint8_t> mask_h(mask_d.size());
    thrust::copy(branch_d.begin(), branch_d.end(), branch_h.begin());
    thrust::copy(origins_d.begin(), origins_d.end(), origins_h.begin());
    thrust::copy(mask_d.begin(), mask_d.end(), mask_h.begin());

    for (SizeType i = capacity; i < capacity + kSpare; ++i) {
        if (branch_h[i * kStride] != kCanaryVal) {
            ++run.leaves_past_span;
        }
        if (mask_h[i] != kCanaryMask) {
            ++run.mask_past_span;
        }
    }
    const auto origins_ok =
        std::all_of(origins_h.begin() + static_cast<long>(capacity),
                    origins_h.end(), [](uint32_t v) { return v == kCanaryIx; });
    run.canaries_intact =
        run.leaves_past_span == 0 && run.mask_past_span == 0 && origins_ok;
    run.leaves.assign(branch_h.begin(),
                      branch_h.begin() + static_cast<long>(capacity * kStride));
    run.origins.assign(origins_h.begin(),
                       origins_h.begin() + static_cast<long>(capacity));
    return run;
}

} // namespace

TEST_CASE("CUDA Taylor branching that fits matches the CPU branch",
          "[branch_capacity][cuda]") {
    constexpr SizeType kLeaves    = 4;
    constexpr SizeType kBranchMax = 512;
    const auto leaves             = make_leaves(kLeaves, 6.0);
    const SizeType capacity       = kLeaves * kBranchMax;

    const auto gpu = branch_on_device(leaves, kLeaves, capacity, kBranchMax);
    REQUIRE_FALSE(gpu.threw);
    REQUIRE(gpu.returned > kLeaves * 8);
    REQUIRE(gpu.returned <= capacity);
    REQUIRE(gpu.canaries_intact);

    std::vector<double> cpu_leaves(capacity * kStride, kCanaryVal);
    std::vector<SizeType> cpu_origins(capacity, kCanaryIx);
    loki::memory::BranchingWorkspace ws(kLeaves, kBranchMax, kParams);
    const auto cpu_returned = loki::core::poly_taylor_branch_batch(
        leaves, cpu_leaves, cpu_origins, {0.0, kDt}, kNbins, kEta, kBranchMax,
        kLeaves, kParams, ws);
    REQUIRE(gpu.returned == cpu_returned);
    for (SizeType i = 0; i < cpu_returned; ++i) {
        REQUIRE(gpu.origins[i] == cpu_origins[i]);
        for (SizeType j = 0; j < kStride; ++j) {
            REQUIRE_THAT(gpu.leaves[(i * kStride) + j],
                         WithinRel(cpu_leaves[(i * kStride) + j], 1e-12));
        }
    }
}

TEST_CASE("CUDA Taylor branching past the workspace throws before writing "
          "past it",
          "[branch_capacity][cuda]") {
    // About 6 ways per parameter, ~216 children per leaf, against 8 slots per
    // leaf. The batch is rejected once its total is known, before the
    // validation mask and the children are written.
    constexpr SizeType kLeaves       = 4;
    constexpr SizeType kSlotsPerLeaf = 8;
    const auto leaves                = make_leaves(kLeaves, 6.0);

    const auto gpu = branch_on_device(leaves, kLeaves, kLeaves * kSlotsPerLeaf,
                                      kSlotsPerLeaf);
    INFO("leaf slots written past the span: "
         << gpu.leaves_past_span << ", mask bytes: " << gpu.mask_past_span);
    CHECK(gpu.threw);
    CHECK(gpu.canaries_intact);
}

TEST_CASE("CUDA Taylor branching past branch_max is rejected like the CPU",
          "[branch_capacity][cuda]") {
    // Each parameter wants ~6 children but its scratch slot holds 4. The
    // outputs would fit; the per-parameter slot would not.
    constexpr SizeType kLeaves    = 4;
    constexpr SizeType kBranchMax = 4;
    const auto leaves             = make_leaves(kLeaves, 6.0);
    const SizeType capacity       = kLeaves * 512;

    const auto gpu = branch_on_device(leaves, kLeaves, capacity, kBranchMax);
    INFO("leaf slots written past the span: "
         << gpu.leaves_past_span << ", mask bytes: " << gpu.mask_past_span);
    CHECK(gpu.threw);
    CHECK(gpu.canaries_intact);

    std::vector<double> cpu_leaves(capacity * kStride, kCanaryVal);
    std::vector<SizeType> cpu_origins(capacity, kCanaryIx);
    loki::memory::BranchingWorkspace ws(kLeaves, kBranchMax, kParams);
    REQUIRE_THROWS_AS(loki::core::poly_taylor_branch_batch(
                          leaves, cpu_leaves, cpu_origins, {0.0, kDt}, kNbins,
                          kEta, kBranchMax, kLeaves, kParams, ws),
                      std::runtime_error);
}
#endif // LOKI_ENABLE_CUDA
