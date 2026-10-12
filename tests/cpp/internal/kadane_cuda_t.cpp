#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/kadane_cuda.cuh"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/workspace_cuda.cuh"
#include "lib/detection/kadane.hpp"

using loki::SizeType;

namespace {

// e (N(0, 1) plus a pulse) then v (positive) per profile.
std::vector<float>
make_folds(SizeType nprofiles, SizeType nbins, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> h(nprofiles * 2 * nbins);
    for (SizeType p = 0; p < nprofiles; ++p) {
        float* e = h.data() + (p * 2 * nbins);
        float* v = e + nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            e[j] = normal(rng);
            v[j] = 0.5F + (0.1F * std::abs(normal(rng)));
        }
        // A pulse of a width that depends on the profile.
        for (SizeType j = 0; j < 1 + (p % 7); ++j) {
            e[((p * 7) + j) % nbins] += 3.0F;
        }
    }
    return h;
}

} // namespace

TEST_CASE("Kadane CUDA scorer matches the CPU reference",
          "[kadane][cuda][score]") {
    if (!loki::is_available(loki::Backend::kCUDA)) {
        SKIP("CUDA backend not available");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    using loki::cuda_utils::as_span;
    constexpr SizeType kProfiles = 1003;
    const std::vector<float> all_biases{1.42F, 0.76F, 0.41F};
    std::vector<SizeType> sizes;
    for (SizeType n = 2; n <= 300; n += (n < 70 ? 1 : 11)) {
        sizes.push_back(n);
    }
    for (const SizeType n :
         {SizeType{320}, SizeType{512}, SizeType{777}, SizeType{1024}}) {
        sizes.push_back(n);
    }
    for (const SizeType nbins : sizes) {
        for (const SizeType nbiases : {SizeType{2}, SizeType{3}}) {
            INFO("nbins " << nbins << " nbiases " << nbiases);
            const auto folds =
                make_folds(kProfiles, nbins, static_cast<unsigned>(nbins));
            const std::vector<float> biases(all_biases.begin(),
                                            all_biases.begin() +
                                                static_cast<long>(nbiases));
            std::vector<uint8_t> valid(kProfiles);
            for (SizeType p = 0; p < kProfiles; ++p) {
                valid[p] = (p % 13) == 4 ? 0 : 1;
            }
            constexpr float kThreshold = 2.0F;

            std::vector<float> ref(kProfiles);
            std::vector<SizeType> ref_idx(kProfiles);
            loki::detection::BoxcarKadaneCache cache(biases, nbins);
            (void)loki::detection::score_and_filter_max_kadane_with_cache(
                folds, ref, ref_idx, kThreshold, kProfiles, nbins, cache);

            const thrust::device_vector<float> d_folds  = folds;
            const thrust::device_vector<float> d_biases = biases;
            thrust::device_vector<float> d_scores(kProfiles, -7.0F);
            const thrust::device_vector<uint8_t> d_valid = valid;
            thrust::device_vector<uint8_t> d_mask(kProfiles, 1);
            loki::memory::CUBScratchArena scratch(kProfiles, 1, nullptr);
            const auto count =
                loki::detection::score_and_filter_max_cuda_kadane_d(
                    as_span(d_folds), as_span(d_biases), as_span(d_scores),
                    as_span(d_valid), as_span(d_mask), kThreshold, kProfiles,
                    nbins, scratch, nullptr);
            const thrust::host_vector<float> scores = d_scores;
            const thrust::host_vector<uint8_t> mask = d_mask;

            // float-level differences (rsqrt, summation order) can flip the
            // best width of a near-tie: allow a few.
            SizeType flips = 0;
            SizeType npass = 0;
            for (SizeType p = 0; p < kProfiles; ++p) {
                if (valid[p] == 0) {
                    CHECK(mask[p] == 0);
                    continue;
                }
                npass += mask[p];
                CHECK((mask[p] != 0) == (scores[p] >= kThreshold));
                flips += std::abs(scores[p] - ref[p]) > 5.0e-4F ? 1U : 0U;
            }
            CHECK(count == npass);
            CHECK(flips <= (kProfiles / 100) + 1);
        }
    }
    // Large profiles (the segment kernel's rows above 48 KiB of shared
    // memory): a strong boxcar stands out from the noise.
    for (const SizeType nbins :
         {SizeType{1500}, SizeType{2048}, SizeType{4000}}) {
        INFO("nbins " << nbins);
        constexpr SizeType kN = 20;
        auto folds            = make_folds(kN, nbins, 11);
        for (SizeType p = 0; p < kN; p += 2) {
            for (SizeType j = 0; j < 100; ++j) {
                folds[(p * 2 * nbins) + ((p * 31 + j) % nbins)] += 6.0F;
            }
        }
        const thrust::device_vector<float> d_folds  = folds;
        const thrust::device_vector<float> d_biases = all_biases;
        thrust::device_vector<float> d_scores(kN, -7.0F);
        const thrust::device_vector<uint8_t> d_valid(kN, 1);
        thrust::device_vector<uint8_t> d_mask(kN, 1);
        loki::memory::CUBScratchArena scratch(kN, 1, nullptr);
        (void)loki::detection::score_and_filter_max_cuda_kadane_d(
            as_span(d_folds), as_span(d_biases), as_span(d_scores),
            as_span(d_valid), as_span(d_mask), 0.0F, kN, nbins, scratch,
            nullptr);
        const thrust::host_vector<float> scores = d_scores;
        for (SizeType p = 0; p < kN; ++p) {
            if (p % 2 == 0) {
                CHECK(scores[p] > 20.0F);
            } else {
                CHECK(scores[p] < 12.0F);
            }
        }
    }
    // A bias count the kernels do not handle.
    {
        const thrust::device_vector<float> d_biases(4, 1.0F);
        thrust::device_vector<float> d_folds(2 * 32);
        thrust::device_vector<float> d_scores(1);
        thrust::device_vector<uint8_t> d_valid(1, 1);
        thrust::device_vector<uint8_t> d_mask(1);
        loki::memory::CUBScratchArena scratch(1, 1, nullptr);
        CHECK_THROWS(loki::detection::score_and_filter_max_cuda_kadane_d(
            as_span(d_folds), as_span(d_biases), as_span(d_scores),
            as_span(d_valid), as_span(d_mask), 0.0F, 1, 32, scratch, nullptr));
    }
}
#endif // LOKI_ENABLE_CUDA
