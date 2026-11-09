#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/core/circular.hpp"
#include "lib/cuda/circular_cuda.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/detail/psr_utils.hpp"

using Catch::Matchers::WithinRel;
using loki::Backend;
using loki::ParamLimit;
using loki::SizeType;

// The GPU counterpart of circular_mask_t.cpp: the CUDA resolve, ascend
// resolve and transform must class each leaf as the CPU does, so a crackle
// within round-off of zero takes the Taylor path on both.

namespace {

constexpr SizeType kParams       = 5;
constexpr SizeType kLeavesStride = (kParams + 2) * 2;
constexpr double kSignificance   = 2.0;
constexpr SizeType kNbins        = 64;
constexpr SizeType kNAccel       = 101;
constexpr SizeType kNFreq        = 4001;
// CPU and GPU differ by round-off only (libm against CUDA sin/cos, FMA), far
// below these; a leaf resolved on the wrong path misses by many bins.
constexpr double kPhaseTolBins = 1e-3;
constexpr double kRelTol       = 1e-9;

const std::array<ParamLimit, kParams> kLimits = {
    {{-1e-6, 1e-6}, {-1e-3, 1e-3}, {-1.0, 1.0}, {-50.0, 50.0}, {99.0, 101.0}}};
constexpr std::pair<double, double> kCoordCur  = {0.0, 400.0};
constexpr std::pair<double, double> kCoordInit = {-206.1, 100.0};
constexpr std::pair<double, double> kCoordAdd  = {412.3, 100.0};

struct Leaf {
    double crackle, dcrackle, snap, dsnap, jerk, djerk, accel, daccel;
};

// A leaf in the crackle hole: snap insignificant, jerk at 37 steps.
constexpr Leaf kHole = {.crackle  = 0.0,
                        .dcrackle = 6.99e-8,
                        .snap     = 0.0,
                        .dsnap    = 8.34e-6,
                        .jerk     = 4.5032e-2,
                        .djerk    = 1.23e-3,
                        .accel    = 11.255,
                        .daccel   = 0.147};

// Leaf 0 has a round-off crackle and leaf 1 an exact zero, so both are
// Taylor; then a real crackle child, a circular snap leaf, and the three
// children of a crackle split around zero, each with both signs of jerk.
std::vector<double> make_leaves() {
    std::vector<Leaf> leaves;
    auto residue    = kHole;
    residue.crackle = -1.4036e-24;
    residue.snap    = -3.5081e-22;
    leaves.push_back(residue);
    leaves.push_back(kHole);
    auto real    = kHole;
    real.crackle = -kHole.dcrackle / 2.0;
    leaves.push_back(real);
    auto snap = kHole;
    snap.snap = -10.0 * kHole.dsnap;
    leaves.push_back(snap);
    constexpr SizeType kChildren = 3;
    std::array<double, kChildren> children{};
    loki::psr_utils::branch_crackle_padded(
        children, 0.0, kChildren * kHole.dcrackle, kChildren);
    for (const auto c : children) {
        auto leaf    = kHole;
        leaf.crackle = c;
        leaves.push_back(leaf);
        leaf.jerk = -leaf.jerk;
        leaves.push_back(leaf);
    }
    std::vector<double> out;
    for (const auto& l : leaves) {
        const std::array<double, kLeavesStride> row = {
            l.crackle, l.dcrackle, l.snap, l.dsnap, l.jerk, l.djerk, l.accel,
            l.daccel,  0.0,        1.0e3,  0.0,     0.0,    100.0,   0.0};
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d) {
    std::vector<T> h(d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
}

void sync_device() {
    loki::cuda_utils::check_cuda_call(cudaDeviceSynchronize(),
                                      "circular mask test sync failed");
    REQUIRE(cudaGetLastError() == cudaSuccess);
}

double phase_distance(double a, double b) {
    const double d = std::abs(a - b);
    return std::min(d, static_cast<double>(kNbins) - d);
}

void check_resolved(const std::vector<SizeType>& idx_cpu,
                    const std::vector<float>& phase_cpu,
                    const std::vector<uint32_t>& idx_gpu,
                    const std::vector<float>& phase_gpu) {
    REQUIRE(idx_cpu.size() == idx_gpu.size());
    for (SizeType i = 0; i < idx_cpu.size(); ++i) {
        INFO("output " << i);
        CHECK(idx_gpu[i] == idx_cpu[i]);
        CHECK(phase_distance(phase_gpu[i], phase_cpu[i]) <= kPhaseTolBins);
    }
}

} // namespace

TEST_CASE("CUDA circular resolve classes leaves as the CPU does",
          "[circular][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const auto leaves       = make_leaves();
    const SizeType n_leaves = leaves.size() / kLeavesStride;

    std::vector<SizeType> idx_cpu(n_leaves);
    std::vector<float> phase_cpu(n_leaves);
    loki::core::circ_taylor_resolve_batch(
        leaves, idx_cpu, phase_cpu, kLimits, kCoordAdd, kCoordCur, kCoordInit,
        kNAccel, kNFreq, kNbins, n_leaves, kSignificance);

    const thrust::device_vector<double> leaves_d(leaves);
    const thrust::device_vector<uint8_t> mask_d(n_leaves, 1);
    const thrust::device_vector<ParamLimit> limits_d(kLimits.begin(),
                                                     kLimits.end());
    thrust::device_vector<uint32_t> idx_d(n_leaves);
    thrust::device_vector<float> phase_d(n_leaves);
    loki::core::circ_taylor_resolve_batch_cuda(
        loki::cuda_utils::as_span(leaves_d), loki::cuda_utils::as_span(mask_d),
        loki::cuda_utils::as_span(idx_d), loki::cuda_utils::as_span(phase_d),
        loki::cuda_utils::as_span(limits_d), kCoordAdd, kCoordCur, kCoordInit,
        kNAccel, kNFreq, kNbins, n_leaves, kSignificance, nullptr);
    sync_device();
    const auto idx_gpu   = to_host(idx_d);
    const auto phase_gpu = to_host(phase_d);
    check_resolved(idx_cpu, phase_cpu, idx_gpu, phase_gpu);
    // The round-off crackle resolves as the exact zero does
    CHECK(idx_gpu[0] == idx_gpu[1]);
    CHECK(phase_distance(phase_gpu[0], phase_gpu[1]) <= kPhaseTolBins);
}

TEST_CASE("CUDA circular ascend resolve classes leaves as the CPU does",
          "[circular][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const auto leaves       = make_leaves();
    const SizeType n_leaves = leaves.size() / kLeavesStride;
    const std::array<std::pair<double, double>, 2> segments = {kCoordInit,
                                                               kCoordAdd};
    const SizeType n_out = n_leaves * segments.size();

    std::vector<SizeType> idx_cpu(n_out);
    std::vector<float> phase_cpu(n_out);
    loki::core::circ_taylor_ascend_resolve_batch(
        leaves, idx_cpu, phase_cpu, kLimits, segments, kCoordCur, kNAccel,
        kNFreq, kNbins, n_leaves, segments.size(), kSignificance);

    const thrust::device_vector<double> leaves_d(leaves);
    const thrust::device_vector<ParamLimit> limits_d(kLimits.begin(),
                                                     kLimits.end());
    std::vector<cuda::std::pair<double, double>> segments_h;
    for (const auto& [t0, scale] : segments) {
        segments_h.emplace_back(t0, scale);
    }
    const thrust::device_vector<cuda::std::pair<double, double>> segments_d(
        segments_h);
    thrust::device_vector<uint32_t> idx_d(n_out);
    thrust::device_vector<float> phase_d(n_out);
    loki::core::circ_taylor_ascend_resolve_batch_cuda(
        loki::cuda_utils::as_span(leaves_d), loki::cuda_utils::as_span(idx_d),
        loki::cuda_utils::as_span(phase_d), loki::cuda_utils::as_span(limits_d),
        loki::cuda_utils::as_span(segments_d), kCoordCur, kNAccel, kNFreq,
        kNbins, n_leaves, segments.size(), kSignificance, nullptr);
    sync_device();
    check_resolved(idx_cpu, phase_cpu, to_host(idx_d), to_host(phase_d));
}

TEST_CASE("CUDA circular transform classes leaves as the CPU does",
          "[circular][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const auto leaves       = make_leaves();
    const SizeType n_leaves = leaves.size() / kLeavesStride;

    auto leaves_cpu = leaves;
    std::vector<SizeType> indices(n_leaves);
    std::iota(indices.begin(), indices.end(), SizeType{0});
    loki::core::circ_taylor_transform_batch(leaves_cpu, indices, kCoordAdd,
                                            kCoordCur, n_leaves, false,
                                            kSignificance);

    thrust::device_vector<double> leaves_d(leaves);
    const thrust::device_vector<uint8_t> mask_d(n_leaves, 1);
    loki::core::circ_taylor_transform_batch_cuda(
        loki::cuda_utils::as_span(leaves_d), loki::cuda_utils::as_span(mask_d),
        kCoordAdd, kCoordCur, n_leaves, false, kSignificance, nullptr);
    sync_device();
    const auto leaves_gpu = to_host(leaves_d);
    for (SizeType i = 0; i < leaves_cpu.size(); ++i) {
        INFO("leaf " << i / kLeavesStride << ", slot " << i % kLeavesStride);
        CHECK_THAT(leaves_gpu[i], WithinRel(leaves_cpu[i], kRelTol));
    }
}
#endif // LOKI_ENABLE_CUDA
