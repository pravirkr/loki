#ifdef LOKI_ENABLE_CUDA
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "lib/cuda/thresholds_kernels_cuda.cuh"

// GenericScorer<MAX_BINS>::run_warp (one warp per trial, used for padded
// widths >= 128) must give the same bits as run (one thread per trial, the
// reference implementation): the same stored fold and the same max boxcar
// S/N, for every row. nvcc compiles this file.

namespace {

using loki::detection::detail::BoxWidth;
using loki::detection::detail::GenericScorer;
using loki::detection::detail::TrialSimParams;

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Row r: noise stream base 2 r + 1; odd rows add the profile (H1 branch).
__device__ float row_scale(uint32_t row) {
    return (row & 1U) != 0U ? 2.5F : 0.0F;
}

template <uint32_t MAX_BINS>
__global__ void score_thread_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    float* __restrict__ scores,
                                    TrialSimParams p,
                                    uint32_t nrows) {
    const uint32_t row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const auto off = static_cast<uint64_t>(row) * p.nbins_padded;
    scores[row]    = GenericScorer<MAX_BINS>::template run<true>(
        in + off, out + off, row_scale(row), (2ULL * row) + 1ULL, p);
}

// One warp per row (blockDim.x == 32).
template <uint32_t MAX_BINS>
__global__ void score_warp_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  float* __restrict__ scores,
                                  TrialSimParams p) {
    __shared__ float s_prefix[MAX_BINS];
    const uint32_t row = blockIdx.x;
    const auto off     = static_cast<uint64_t>(row) * p.nbins_padded;
    const float score  = GenericScorer<MAX_BINS>::template run_warp<true>(
        in + off, out + off, s_prefix, row_scale(row), (2ULL * row) + 1ULL, p);
    if (threadIdx.x == 0U) {
        scores[row] = score;
    }
}

std::vector<BoxWidth> box_widths(uint32_t nbins) {
    std::vector<uint32_t> ws{1,  2,  3,  5,   8,   13,  21,
                             34, 55, 89, 144, 233, 377, 610};
    ws.push_back(nbins / 2);
    ws.push_back(nbins - 1); // the widest box: almost every window wraps
    std::vector<BoxWidth> out;
    const auto n = static_cast<float>(nbins);
    for (const auto width : ws) {
        if (width == 0 || width >= nbins) {
            continue;
        }
        const auto w  = static_cast<float>(width);
        const float h = std::sqrt((n - w) / (n * w));
        const float b = w * h / (n - w);
        out.push_back(BoxWidth{width, h + b, b});
    }
    return out;
}

struct Scored {
    std::vector<float> folds;
    std::vector<float> scores;
};

template <uint32_t MAX_BINS> void compare_paths(uint32_t nbins) {
    constexpr uint32_t kRows = 96;
    const uint32_t padded    = (nbins + 3U) & ~3U;
    REQUIRE(padded <= MAX_BINS);

    std::mt19937 gen(0xA11CEU + nbins);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> in(static_cast<std::size_t>(kRows) * padded, 0.0F);
    std::vector<float> profile(padded, 0.0F);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t j = 0; j < nbins; ++j) {
            in[(static_cast<std::size_t>(r) * padded) + j] = normal(gen);
        }
    }
    for (uint32_t j = 0; j < nbins; ++j) {
        const float x =
            (static_cast<float>(j) - (0.3F * static_cast<float>(nbins))) /
            (0.05F * static_cast<float>(nbins));
        profile[j] = std::exp(-0.5F * x * x);
    }
    const auto widths = box_widths(nbins);

    const thrust::device_vector<float> in_d(in.begin(), in.end());
    const thrust::device_vector<float> prof_d(profile.begin(), profile.end());
    const thrust::device_vector<BoxWidth> widths_d(widths.begin(),
                                                   widths.end());
    const TrialSimParams p{
        .profile      = thrust::raw_pointer_cast(prof_d.data()),
        .widths       = thrust::raw_pointer_cast(widths_d.data()),
        .box_by_len   = nullptr, // read only by RegScorer
        .nwidths      = static_cast<uint32_t>(widths.size()),
        .nbins        = nbins,
        .nbins_padded = padded,
        .bias_snr     = 0.0F,
        .var_in       = 1.0F,
        .var_add      = 0.25F,
        .seed         = 0x5EED0000ULL + nbins,
    };

    auto run_path = [&](bool warp) {
        thrust::device_vector<float> out_d(in.size(), -1.0F);
        thrust::device_vector<float> scores_d(kRows, -1.0F);
        auto* out        = thrust::raw_pointer_cast(out_d.data());
        auto* scores     = thrust::raw_pointer_cast(scores_d.data());
        const auto* in_p = thrust::raw_pointer_cast(in_d.data());
        if (warp) {
            score_warp_kernel<MAX_BINS><<<kRows, 32>>>(in_p, out, scores, p);
        } else {
            score_thread_kernel<MAX_BINS>
                <<<(kRows + 31U) / 32U, 32>>>(in_p, out, scores, p, kRows);
        }
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        Scored s;
        s.folds.resize(in.size());
        s.scores.resize(kRows);
        thrust::copy(out_d.begin(), out_d.end(), s.folds.begin());
        thrust::copy(scores_d.begin(), scores_d.end(), s.scores.begin());
        return s;
    };

    const auto ref  = run_path(false);
    const auto warp = run_path(true);
    for (uint32_t r = 0; r < kRows; ++r) {
        INFO("nbins " << nbins << " row " << r);
        REQUIRE(std::isfinite(ref.scores[r]));
        REQUIRE(std::bit_cast<uint32_t>(warp.scores[r]) ==
                std::bit_cast<uint32_t>(ref.scores[r]));
    }
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (std::bit_cast<uint32_t>(warp.folds[i]) !=
            std::bit_cast<uint32_t>(ref.folds[i])) {
            FAIL("nbins " << nbins << ": stored fold differs at " << i);
        }
    }
}

} // namespace

TEST_CASE("CUDA scorer: the warp path gives the same bits as the thread path",
          "[thresholds][cuda]") {
    if (!cuda_device_available()) {
        SKIP("no CUDA device");
    }
    compare_paths<128>(100);
    compare_paths<128>(128);
    compare_paths<256>(130); // padded to 132
    compare_paths<256>(256);
    compare_paths<512>(509); // padded to 512
    compare_paths<1024>(1000);
    compare_paths<1024>(1021); // padded to 1024
    compare_paths<1024>(1024);
}
#endif
