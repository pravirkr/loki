#ifdef LOKI_ENABLE_CUDA
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

#include "lib/cuda/ep_fourier_device.cuh"
#include "lib/cuda/ffa_score_device.cuh"

using loki::Backend;
using loki::SizeType;

namespace {

constexpr SizeType kOut = 8;

// The FFA phasor before 2026-10-15: the cycle count reduced as
// t_hi * inv_nbins, exact only for a power-of-two nbins.
__device__ loki::core::detail::Phasor
old_fourier_phasor(uint32_t k, float shift, float inv_nbins) {
    const auto kf    = static_cast<float>(k);
    const float t_hi = kf * shift;
    const float t_lo = fmaf(kf, shift, -t_hi);
    const float c_hi = t_hi * inv_nbins;
    const float c_lo = t_lo * inv_nbins;
    const float frac = (c_hi - rintf(c_hi)) + c_lo;
    loki::core::detail::Phasor ph;
    sincospif(-2.0F * frac, &ph.s, &ph.c);
    return ph;
}

// Per (k, shift): cos/sin of the EP phasor, of the FFA phasor, of the old EP
// path (float-rounded phase, __sincosf) and of the old FFA phasor.
__global__ void kernel_phasors(const uint32_t* k,
                               const float* shift,
                               uint32_t n,
                               uint32_t nbins,
                               float* out) {
    const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }
    float* o       = out + (kOut * static_cast<SizeType>(i));
    const auto nf  = static_cast<float>(nbins);
    const auto inv = __frcp_rn(nf);
    loki::core::detail::ep_phasor(k[i], shift[i], nbins, o[0], o[1]);
    const auto ffa =
        loki::core::detail::fourier_phasor(k[i], shift[i], nf, inv);
    o[2] = ffa.c;
    o[3] = ffa.s;
    __sincosf(loki::core::detail::fourier_phase(k[i], shift[i], nbins), &o[5],
              &o[4]);
    const auto old = old_fourier_phasor(k[i], shift[i], inv);
    o[6]           = old.c;
    o[7]           = old.s;
}

struct MaxErr {
    double ep{};
    double ffa{};
    double legacy{};
    double old_ffa{};
    // The new FFA phasor has the old one's bits (cos and sin) everywhere.
    bool ffa_same_bits{true};
};

// Largest |phasor - exp(-2 pi i k shift / nbins)| over every harmonic k of
// nbins and random shifts in [0, nbins), against a double reference.
MaxErr max_errors(uint32_t nbins, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0F, static_cast<float>(nbins));
    std::vector<uint32_t> ks;
    std::vector<float> shifts;
    for (uint32_t k = 0; k <= nbins / 2; ++k) {
        for (int j = 0; j < 64; ++j) {
            ks.push_back(k);
            shifts.push_back(dist(rng));
        }
    }
    const auto n = static_cast<uint32_t>(ks.size());
    const thrust::device_vector<uint32_t> k_d(ks.begin(), ks.end());
    const thrust::device_vector<float> s_d(shifts.begin(), shifts.end());
    thrust::device_vector<float> out_d(kOut * static_cast<SizeType>(n));
    kernel_phasors<<<(n + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(k_d.data()),
        thrust::raw_pointer_cast(s_d.data()), n, nbins,
        thrust::raw_pointer_cast(out_d.data()));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const thrust::host_vector<float> out = out_d;

    MaxErr err;
    for (uint32_t i = 0; i < n; ++i) {
        const double t = std::fmod(static_cast<double>(ks[i]) *
                                       static_cast<double>(shifts[i]),
                                   static_cast<double>(nbins));
        const double phase =
            -2.0 * std::numbers::pi * t / static_cast<double>(nbins);
        const double c   = std::cos(phase);
        const double s   = std::sin(phase);
        const SizeType o = kOut * i;
        const auto dev   = [&](SizeType j) {
            return std::hypot(out[o + j] - c, out[o + j + 1] - s);
        };
        err.ep            = std::max(err.ep, dev(0));
        err.ffa           = std::max(err.ffa, dev(2));
        err.legacy        = std::max(err.legacy, dev(4));
        err.old_ffa       = std::max(err.old_ffa, dev(6));
        err.ffa_same_bits = err.ffa_same_bits &&
                            std::bit_cast<uint32_t>(out[o + 2]) ==
                                std::bit_cast<uint32_t>(out[o + 6]) &&
                            std::bit_cast<uint32_t>(out[o + 3]) ==
                                std::bit_cast<uint32_t>(out[o + 7]);
    }
    return err;
}

} // namespace

TEST_CASE("Fourier phasors (EP, FFA) are accurate for any nbins",
          "[ep][ffa][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    std::mt19937 rng(20261014U);
    for (const uint32_t nbins : {50U, 64U, 100U, 128U, 257U, 1000U, 1024U}) {
        const auto err = max_errors(nbins, rng);
        UNSCOPED_INFO("nbins " << nbins << ": ep_phasor " << err.ep
                               << ", fourier_phasor " << err.ffa
                               << ", old fourier_phasor " << err.old_ffa
                               << ", fourier_phase + __sincosf " << err.legacy);
        CHECK(err.ep < 1e-6);
        CHECK(err.ffa < 1e-6);
        // A power-of-two nbins keeps the FFA outputs bit for bit.
        if (std::has_single_bit(nbins)) {
            CHECK(err.ffa_same_bits);
        }
    }
}
#endif // LOKI_ENABLE_CUDA
