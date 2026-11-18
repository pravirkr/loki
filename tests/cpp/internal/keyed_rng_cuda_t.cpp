#ifdef LOKI_ENABLE_CUDA
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/detail/keyed_rng.hpp"
#include "lib/detail/math.hpp"

// The keyed PCG32 streams and the inverse-CDF normal draw give the same bits
// on the device as on the host (the spectral zap's noise fill). nvcc
// compiles this file.

using loki::SizeType;

namespace {

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

constexpr int kDraws = 4;

/// Per key k: kDraws raw words, then kDraws normals with mean 0.5 and
/// stddev sd[k], all from make_keyed_pcg32(seed, 1, 0, k, 0, 0).
__global__ void kernel_draw(std::uint64_t seed,
                            const float* __restrict__ lut,
                            const float* __restrict__ sd,
                            std::uint32_t nkeys,
                            std::uint32_t* __restrict__ words,
                            float* __restrict__ normals) {
    const std::uint32_t k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k >= nkeys) {
        return;
    }
    auto raw = loki::math::make_keyed_pcg32(seed, 1U, 0U, k, 0U, 0U);
    for (int i = 0; i < kDraws; ++i) {
        words[(k * kDraws) + i] = raw();
    }
    auto rng = loki::math::make_keyed_pcg32(seed, 1U, 0U, k, 0U, 0U);
    for (int i = 0; i < kDraws; ++i) {
        normals[(k * kDraws) + i] =
            loki::math::normal_from_lut(rng(), lut, 0.5F, sd[k]);
    }
}

} // namespace

TEST_CASE("Keyed PCG32 and normal draws match the host bit for bit",
          "[rng][keyed_rng]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    const auto lut = loki::math::NormalSampler::lut();
    REQUIRE(lut.size() == loki::math::kNormalLutSize);
    constexpr std::uint32_t kKeys = 4096;
    std::vector<float> sd(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) {
        sd[k] = 0.1F + (1.7e-3F * static_cast<float>(k));
    }
    thrust::device_vector<float> d_lut(lut.begin(), lut.end());
    thrust::device_vector<float> d_sd(sd.begin(), sd.end());
    constexpr std::array<std::uint64_t, 3> kSeeds{0, 7, 0xdeadbeefcafef00dULL};
    for (const std::uint64_t seed : kSeeds) {
        thrust::device_vector<std::uint32_t> d_words(kKeys * kDraws);
        thrust::device_vector<float> d_normals(kKeys * kDraws);
        kernel_draw<<<(kKeys + 127) / 128, 128>>>(
            seed, thrust::raw_pointer_cast(d_lut.data()),
            thrust::raw_pointer_cast(d_sd.data()), kKeys,
            thrust::raw_pointer_cast(d_words.data()),
            thrust::raw_pointer_cast(d_normals.data()));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<std::uint32_t> words(d_words.size());
        std::vector<float> normals(d_normals.size());
        thrust::copy(d_words.begin(), d_words.end(), words.begin());
        thrust::copy(d_normals.begin(), d_normals.end(), normals.begin());

        SizeType word_mismatch   = 0;
        SizeType normal_mismatch = 0;
        for (std::uint32_t k = 0; k < kKeys; ++k) {
            auto raw = loki::math::make_keyed_pcg32(seed, 1U, 0U, k, 0U, 0U);
            for (int i = 0; i < kDraws; ++i) {
                word_mismatch += words[(k * kDraws) + i] != raw() ? 1U : 0U;
            }
            auto rng = loki::math::make_keyed_pcg32(seed, 1U, 0U, k, 0U, 0U);
            std::array<float, kDraws> host{};
            loki::math::NormalSampler::generate(rng, host, 0.5F, sd[k]);
            for (int i = 0; i < kDraws; ++i) {
                normal_mismatch +=
                    std::bit_cast<std::uint32_t>(normals[(k * kDraws) + i]) !=
                            std::bit_cast<std::uint32_t>(host[i])
                        ? 1U
                        : 0U;
            }
        }
        CHECK(word_mismatch == 0);
        CHECK(normal_mismatch == 0);
    }
}

#endif // LOKI_ENABLE_CUDA
