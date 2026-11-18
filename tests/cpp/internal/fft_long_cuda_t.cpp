#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/fft_long_cuda.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/utils/fft_impl.hpp"

// cuFFT whole-series transforms (preprocessing). nvcc compiles this file.

using loki::ComplexType;
using loki::ComplexTypeCUDA;
using loki::SizeType;
using loki::math::LongRealFFT;

namespace {

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<float> random_series(SizeType n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(n);
    for (auto& v : x) {
        v = nd(rng);
    }
    return x;
}

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d) {
    std::vector<T> h(d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
}

/// Largest |a - b| over the largest |b| (relative to the spectrum's scale).
double max_rel_diff(std::span<const ComplexTypeCUDA> a,
                    std::span<const ComplexType> b) {
    double diff  = 0.0;
    double scale = 0.0;
    for (SizeType k = 0; k < b.size(); ++k) {
        const std::complex<double> ad(a[k].real(), a[k].imag());
        const std::complex<double> bd(b[k].real(), b[k].imag());
        diff  = std::max(diff, std::abs(ad - bd));
        scale = std::max(scale, std::abs(bd));
    }
    return diff / scale;
}

} // namespace

TEST_CASE("LongRealFFT matches pocketfft and round-trips",
          "[fft][LongRealFFT]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    // Powers of two, odd and composite lengths, one prime.
    const SizeType n =
        GENERATE(2, 3, 16, 1000, 4096, 6561, 65536, 100003, 1U << 20U);
    const SizeType nf = (n / 2) + 1;
    const auto x      = random_series(n, static_cast<unsigned>(n));

    std::vector<ComplexType> cpu_spec(nf);
    auto x_copy = x;
    loki::math::rfft_batch(x_copy, cpu_spec, 1, n, 1);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    {
        LongRealFFT fft(n, 0, stream);
        REQUIRE(fft.size() == n);
        thrust::device_vector<float> d_x(x.begin(), x.end());
        thrust::device_vector<ComplexTypeCUDA> d_spec(nf);
        fft.rfft(thrust::raw_pointer_cast(d_x.data()),
                 thrust::raw_pointer_cast(d_spec.data()));
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        const auto spec = to_host(d_spec);
        // Float transforms: error grows ~ log n; 1e-5 of the peak is loose.
        CHECK(max_rel_diff(spec, cpu_spec) < 1e-5);
        // The input is left unchanged.
        CHECK(to_host(d_x) == x);

        // A second call gives the same bits.
        thrust::device_vector<ComplexTypeCUDA> d_spec2(nf);
        fft.rfft(thrust::raw_pointer_cast(d_x.data()),
                 thrust::raw_pointer_cast(d_spec2.data()));
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        const auto spec2 = to_host(d_spec2);
        bool same        = true;
        for (SizeType k = 0; k < nf; ++k) {
            same = same && spec[k] == spec2[k];
        }
        CHECK(same);

        // Round trip, scaled by 1 / n as on the host.
        thrust::device_vector<float> d_back(n);
        fft.irfft(thrust::raw_pointer_cast(d_spec.data()),
                  thrust::raw_pointer_cast(d_back.data()));
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        const auto back = to_host(d_back);
        double err      = 0.0;
        for (SizeType i = 0; i < n; ++i) {
            err = std::max(err, static_cast<double>(std::abs(back[i] - x[i])));
        }
        CHECK(err < 1e-4);
    }
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("LongRealFFT work area and limits", "[fft][LongRealFFT]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    const loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    CHECK(LongRealFFT::workspace_bytes(1) == 0);
    // A long transform needs a work area; the estimate does not allocate.
    CHECK(LongRealFFT::workspace_bytes(SizeType{1} << 27U) > 0);
    CHECK_THROWS_AS(LongRealFFT(1, 0, nullptr), std::invalid_argument);
}

#endif // LOKI_ENABLE_CUDA
