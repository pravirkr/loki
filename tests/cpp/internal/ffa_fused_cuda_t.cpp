#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/kernels_cuda.cuh"
#include "lib/cuda/workspace_cuda.cuh"

using loki::Backend;
using loki::ParamLimit;
using loki::SizeType;

namespace {

// Levels for the fused kernel: per-level parent indices and phase shifts.
struct SyntheticLevels {
    std::vector<SizeType> ncoords;
    std::vector<SizeType> nsegments;
    std::vector<std::vector<uint32_t>> idx; // level 0 empty
};

std::vector<float> random_floats(SizeType n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> out(n);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

std::vector<float> to_host(const float* d, SizeType n) {
    std::vector<float> h(n);
    loki::cuda_utils::check_cuda_call(
        cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost),
        "test D2H failed");
    return h;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Runs levels [first, first + k) unfused (ping-pong) and fused from the same
// input, and returns both outputs at level first + k - 1.
std::pair<std::vector<float>, std::vector<float>>
run_both(const std::vector<float>& input,
         std::span<const uint32_t* const> idx_d,
         std::span<const float* const> shift_d,
         std::span<const SizeType> ncoords,
         std::span<const SizeType> nsegments,
         SizeType first,
         int k,
         SizeType nbins,
         SizeType max_elems,
         SizeType misalign) {
    const SizeType stride = 2 * nbins;
    thrust::device_vector<float> buf_a(max_elems + misalign);
    thrust::device_vector<float> buf_b(max_elems + misalign);
    float* a = thrust::raw_pointer_cast(buf_a.data()) + misalign;
    float* b = thrust::raw_pointer_cast(buf_b.data()) + misalign;
    const SizeType out_level = first + static_cast<SizeType>(k) - 1;
    const SizeType out_elems =
        nsegments[out_level] * ncoords[out_level] * stride;

    // Unfused reference.
    cudaMemcpy(a, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    float* in  = a;
    float* out = b;
    for (int step = 0; step < k; ++step) {
        const SizeType lvl = first + static_cast<SizeType>(step);
        const loki::coord::FFACoordFreqDPtrs coords{
            .idx   = const_cast<uint32_t*>(idx_d[lvl]),
            .shift = const_cast<float*>(shift_d[lvl]),
            .size  = ncoords[lvl]};
        loki::core::ffa_iter_freq_cuda(in, out, coords, ncoords[lvl],
                                       ncoords[lvl - 1], nsegments[lvl], nbins,
                                       nullptr);
        std::swap(in, out);
    }
    auto unfused = to_host(in, out_elems);

    // Fused.
    cudaMemcpy(a, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(b, 0, out_elems * sizeof(float));
    std::array<const uint32_t*, 5> idxs{};
    std::array<const float*, 5> shifts{};
    std::array<uint32_t, 5> counts{};
    for (int step = 0; step < k; ++step) {
        const SizeType lvl                = first + static_cast<SizeType>(step);
        idxs[static_cast<SizeType>(step)] = idx_d[lvl];
        shifts[static_cast<SizeType>(step)] = shift_d[lvl];
        counts[static_cast<SizeType>(step)] =
            static_cast<uint32_t>(ncoords[lvl]);
    }
    loki::core::ffa_iter_freq_fused_cuda(
        a, b, idxs.data(), shifts.data(), counts.data(),
        static_cast<uint32_t>(ncoords[first - 1]),
        static_cast<uint32_t>(ncoords[out_level]),
        static_cast<uint32_t>(nsegments[first - 1]),
        static_cast<uint32_t>(nbins), k, nullptr);
    loki::cuda_utils::check_cuda_call(cudaDeviceSynchronize(),
                                      "fused kernel failed");
    auto fused = to_host(b, out_elems);
    return {std::move(unfused), std::move(fused)};
}

} // namespace

TEST_CASE("plan_ffa_freq_fuse_groups picks valid groups", "[ffa][cuda]") {
    using loki::core::plan_ffa_freq_fuse_groups;
    const auto sum = [](const std::vector<int>& g) {
        return std::accumulate(g.begin(), g.end(), 0);
    };
    const std::array<bool, 6> all_fit{false, false, false, true, true, true};
    const std::vector<SizeType> ncoords(9, 16);
    std::vector<SizeType> nsegments(9);
    for (SizeType i = 0; i < 9; ++i) {
        nsegments[i] = SizeType{256} >> i;
    }

    SECTION("clean levels: a k = 4 prefix, then the largest that fits") {
        const std::vector<uint32_t> bad(9, 0);
        const auto g =
            plan_ffa_freq_fuse_groups(ncoords, nsegments, bad, all_fit);
        CHECK(g == std::vector<int>{4, 4});
    }
    SECTION("an odd number of k = 4 groups is allowed") {
        const std::span<const SizeType> nc(ncoords.data(), 5);
        const std::span<const SizeType> ns(nsegments.data(), 5);
        const std::vector<uint32_t> bad(5, 0);
        CHECK(plan_ffa_freq_fuse_groups(nc, ns, bad, all_fit) ==
              std::vector<int>{4});
    }
    SECTION("no fused group covers a bad level") {
        std::vector<uint32_t> bad(9, 0);
        bad[3] = 1;
        const auto g =
            plan_ffa_freq_fuse_groups(ncoords, nsegments, bad, all_fit);
        CHECK(sum(g) == 8);
        SizeType level = 1;
        for (const int k : g) {
            CHECK((k == 1 || k == 3 || k == 4 || k == 5));
            if (k > 1) {
                CHECK((level > 3 || level + static_cast<SizeType>(k) <= 3));
            }
            level += static_cast<SizeType>(k);
        }
    }
    SECTION("shared memory or segment tiling rules a group out") {
        const std::vector<uint32_t> bad(9, 0);
        const std::array<bool, 6> none{};
        CHECK(plan_ffa_freq_fuse_groups(ncoords, nsegments, bad, none) ==
              std::vector<int>(8, 1));
        std::vector<SizeType> odd_segments(9, 3);
        CHECK(plan_ffa_freq_fuse_groups(ncoords, odd_segments, bad, all_fit) ==
              std::vector<int>(8, 1));
    }
}

TEST_CASE("Fused frequency merges match the unfused levels bit for bit",
          "[ffa][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const auto nbins = GENERATE(SizeType{32}, SizeType{50}, SizeType{64},
                                SizeType{96}, SizeType{192}, SizeType{384});
    // A span one float into its allocation must take the scalar path
    // (nbins 64 is the float4 path).
    const auto misalign = GENERATE(SizeType{0}, SizeType{1});
    CAPTURE(nbins, misalign);

    const std::vector<ParamLimit> limits = {{.min = 20.0, .max = 21.0}};
    const loki::search::FFASearchConfig cfg(
        /*nsamps=*/SizeType{1} << 15U, /*tsamp=*/1.0e-3, nbins, /*eta=*/1.0,
        limits, /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/false,
        /*nthreads=*/1, /*max_process_memory_gb=*/1.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/SizeType{32});
    const loki::plans::FFAPlan<float> plan(cfg);
    loki::memory::FFAWorkspaceCUDA<float> ws(plan);
    ws.resolve_coordinates_freq(plan, nullptr);
    const auto bad       = ws.check_fuse_levels_freq(plan, nullptr);
    const auto ncoords   = plan.get_ncoords();
    const auto nsegments = plan.get_nsegments();
    const auto offsets   = plan.get_ncoords_offsets();
    const auto n_levels  = ncoords.size();
    REQUIRE(n_levels >= 6);
    for (const auto b : bad) {
        CHECK(b == 0);
    }

    auto ptrs = ws.coords_freq_d.get_raw_ptrs();
    std::vector<const uint32_t*> idx_d(n_levels);
    std::vector<const float*> shift_d(n_levels);
    for (SizeType l = 0; l < n_levels; ++l) {
        idx_d[l]   = ptrs.idx + offsets[l];
        shift_d[l] = ptrs.shift + offsets[l];
    }
    const SizeType stride = 2 * nbins;
    SizeType max_elems    = 0;
    for (SizeType l = 0; l < n_levels; ++l) {
        max_elems = std::max(max_elems, nsegments[l] * ncoords[l] * stride);
    }

    for (const int k : {3, 4, 5}) {
        if (!loki::core::ffa_freq_fuse_fits_smem(k, nbins)) {
            continue;
        }
        // Start at level 1 and at level 2 (both parities of the input).
        for (const SizeType first : {SizeType{1}, SizeType{2}}) {
            CAPTURE(k, first);
            const auto input = random_floats(
                nsegments[first - 1] * ncoords[first - 1] * stride,
                static_cast<std::uint32_t>(nbins * 10 + first));
            const auto [unfused, fused] =
                run_both(input, idx_d, shift_d, ncoords, nsegments, first, k,
                         nbins, max_elems, misalign);
            CHECK(bit_equal(unfused, fused));
        }
    }
}

// Hidden ([.]): a level over 2^31 bytes (3.5 GB, 64-bit offsets). Needs
// ~7.5 GB of device memory and ~11 GB of host memory. Run with
//   loki_internal_tests "[ffa][cuda][large]"
TEST_CASE("Fused frequency merges are exact on a level over 2^31 bytes",
          "[.][ffa][cuda][large]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    constexpr SizeType kNbins            = 64;
    const std::vector<ParamLimit> limits = {{.min = 1.0, .max = 200.0}};
    const loki::search::FFASearchConfig cfg(
        /*nsamps=*/SizeType{1} << 23U, /*tsamp=*/64.0e-6, kNbins,
        /*eta=*/1.0, limits, /*ducy_max=*/0.2, /*wtsp=*/1.5,
        /*use_fourier=*/false, /*nthreads=*/1, /*max_process_memory_gb=*/32.0,
        /*octave_scale=*/2.0, /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/SizeType{128});
    const loki::plans::FFAPlan<float> plan(cfg);
    const auto ncoords    = plan.get_ncoords();
    const auto nsegments  = plan.get_nsegments();
    const auto offsets    = plan.get_ncoords_offsets();
    const auto n_levels   = ncoords.size();
    const SizeType stride = 2 * kNbins;
    SizeType max_elems    = 0;
    for (SizeType l = 0; l < n_levels; ++l) {
        max_elems = std::max(max_elems, nsegments[l] * ncoords[l] * stride);
    }
    REQUIRE(max_elems * sizeof(float) > (SizeType{1} << 31U));
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    if (free_bytes < (2 * max_elems * sizeof(float)) + (SizeType{1} << 30U)) {
        SKIP("not enough free device memory");
    }

    loki::memory::FFAWorkspaceCUDA<float> ws(SizeType{1}, plan.get_coord_size(),
                                             n_levels, 1);
    ws.resolve_coordinates_freq(plan, nullptr);
    auto ptrs = ws.coords_freq_d.get_raw_ptrs();
    std::vector<const uint32_t*> idx_d(n_levels);
    std::vector<const float*> shift_d(n_levels);
    for (SizeType l = 0; l < n_levels; ++l) {
        idx_d[l]   = ptrs.idx + offsets[l];
        shift_d[l] = ptrs.shift + offsets[l];
    }
    // Cheap deterministic input (a hash, not an RNG: 880M values).
    std::vector<float> input(nsegments[0] * ncoords[0] * stride);
    for (SizeType i = 0; i < input.size(); ++i) {
        const auto h = static_cast<std::uint32_t>(i * 2654435761ULL);
        input[i]     = static_cast<float>(h >> 8U) * 0x1p-24F - 0.5F;
    }
    const auto [unfused, fused] = run_both(
        input, idx_d, shift_d, ncoords, nsegments, 1, 4, kNbins, max_elems, 0);
    CHECK(bit_equal(unfused, fused));
}

TEST_CASE("Fused frequency merges handle coordinates without children",
          "[ffa][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    constexpr SizeType kNbins = 32;
    // Parent 1 of level 0 and parent 2 of level 1 have no children; every
    // parent has at most 2.
    const SyntheticLevels lv{
        .ncoords   = {3, 4, 5, 6},
        .nsegments = {8, 4, 2, 1},
        .idx       = {{}, {0, 0, 2, 2}, {0, 1, 1, 3, 3}, {0, 0, 1, 2, 2, 4}},
    };
    std::vector<uint32_t> idx_flat;
    std::vector<uint32_t> offsets;
    for (const auto& level : lv.idx) {
        offsets.push_back(static_cast<uint32_t>(idx_flat.size()));
        idx_flat.insert(idx_flat.end(), level.begin(), level.end());
    }
    // Level 0 has 3 coordinates but no parent indices: pad so offsets match.
    idx_flat.insert(idx_flat.begin(), 3, 0U);
    for (SizeType l = 1; l < offsets.size(); ++l) {
        offsets[l] += 3;
    }
    std::vector<float> shift_flat(idx_flat.size());
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.0F, kNbins);
    for (auto& s : shift_flat) {
        s = dist(rng);
    }
    const thrust::device_vector<uint32_t> idx_dv(idx_flat.begin(),
                                                 idx_flat.end());
    const thrust::device_vector<float> shift_dv(shift_flat.begin(),
                                                shift_flat.end());
    std::vector<const uint32_t*> idx_d;
    std::vector<const float*> shift_d;
    for (const auto off : offsets) {
        idx_d.push_back(thrust::raw_pointer_cast(idx_dv.data()) + off);
        shift_d.push_back(thrust::raw_pointer_cast(shift_dv.data()) + off);
    }

    // The contract check reports these levels clean.
    thrust::device_vector<uint32_t> scratch(3 * lv.ncoords.size());
    const auto bad = loki::core::ffa_freq_fuse_check_levels_cuda(
        thrust::raw_pointer_cast(idx_dv.data()), offsets, lv.ncoords,
        thrust::raw_pointer_cast(scratch.data()), nullptr);
    CHECK(bad == std::vector<uint32_t>(lv.ncoords.size(), 0));

    const auto input = random_floats(8 * 3 * 2 * kNbins, 11);
    const auto [unfused, fused] =
        run_both(input, idx_d, shift_d, lv.ncoords, lv.nsegments, 1, 3, kNbins,
                 8 * 6 * 2 * kNbins, 0);
    CHECK(bit_equal(unfused, fused));
}

TEST_CASE("The fusion contract check flags unsorted and wide levels",
          "[ffa][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    // Level 1 clean, level 2 unsorted, level 3 has a parent with 3
    // children, level 4 points past the previous level.
    const std::vector<SizeType> ncoords = {2, 4, 4, 5, 3};
    const std::vector<uint32_t> idx     = {0, 0,          // level 0 (unused)
                                           0, 0, 1, 1,    // level 1
                                           0, 2, 1, 3,    // level 2
                                           0, 1, 1, 1, 3, // level 3
                                           0, 4, 5};      // level 4
    const std::vector<uint32_t> offsets = {0, 2, 6, 10, 15};
    const thrust::device_vector<uint32_t> idx_d(idx.begin(), idx.end());
    thrust::device_vector<uint32_t> scratch(3 * ncoords.size());
    const auto bad = loki::core::ffa_freq_fuse_check_levels_cuda(
        thrust::raw_pointer_cast(idx_d.data()), offsets, ncoords,
        thrust::raw_pointer_cast(scratch.data()), nullptr);
    REQUIRE(bad.size() == 5);
    CHECK(bad[0] == 0);
    CHECK(bad[1] == 0);
    CHECK(bad[2] > 0);
    CHECK(bad[3] > 0);
    CHECK(bad[4] > 0);
}
#endif // LOKI_ENABLE_CUDA
