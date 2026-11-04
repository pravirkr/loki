#ifdef LOKI_ENABLE_CUDA
#include "lib/cuda/fft_cuda.cuh"

#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/common/types.hpp"

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/types_cuda.cuh"
#include "lib/utils/fft_impl.hpp"

// GPU FFT tests. nvcc compiles this file (Thrust device vectors); the CPU FFT
// tests stay in fft_t.cpp, a host translation unit.

using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinAbs;
using loki::ComplexType;
using loki::ComplexTypeCUDA;
using loki::SizeType;
using loki::math::CpuFFT;
using loki::math::irfft_batch;
using loki::math::rfft_batch;

namespace {

constexpr float kFftTol = 1e-4F;

auto float_near(float margin) {
    return [margin](float actual, float expected) {
        return WithinAbs(expected, margin).match(actual);
    };
}

auto cuda_complex_eq(const ComplexTypeCUDA& lhs, const ComplexTypeCUDA& rhs) {
    return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}

std::vector<float> make_sine_batch(SizeType batch_size, SizeType n_real) {
    std::vector<float> data(batch_size * n_real);
    for (SizeType b = 0; b < batch_size; ++b) {
        for (SizeType i = 0; i < n_real; ++i) {
            data[(b * n_real) + i] = std::sin(2.0F * std::numbers::pi_v<float> *
                                              static_cast<float>(i + 1 + b) /
                                              static_cast<float>(n_real));
        }
    }
    return data;
}

void require_round_trip(std::span<const float> real_in,
                        std::span<const float> recovered) {
    REQUIRE_THAT(recovered, RangeEquals(real_in, float_near(kFftTol)));
}

void require_reals_near(std::span<const float> lhs,
                        std::span<const float> rhs,
                        float margin) {
    REQUIRE_THAT(lhs, RangeEquals(rhs, float_near(margin)));
}

} // namespace

using loki::math::GpuFFTManager;
using loki::math::irfft_batch_cuda;
using loki::math::rfft_batch_cuda;

namespace {

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<float> random_rows(SizeType batch_size, SizeType n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(batch_size * n);
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

} // namespace

TEST_CASE("GpuFFTManager matches the host FFT for direct and Bluestein lengths",
          "[fft][GpuFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const SizeType n = GENERATE(2, 3, 5, 7, 16, 31, 32, 50, 64, 100, 127, 128,
                                333, 512, 1000, 1024, 1500, 2047, 2048);
    constexpr SizeType kBatch = 37;
    const SizeType nf         = (n / 2) + 1;
    const auto real_in = random_rows(kBatch, n, static_cast<unsigned>(n));

    std::vector<ComplexType> cpu_spec(kBatch * nf);
    std::vector<float> cpu_real(kBatch * n);
    CpuFFT cpu;
    auto real_copy = real_in;
    cpu.rfft_batch(real_copy, cpu_spec, kBatch, n, 1);
    auto spec_copy = cpu_spec;
    cpu.irfft_batch(spec_copy, cpu_real, kBatch, n, 1);

    GpuFFTManager gpu(0);
    gpu.prepare_plans(std::span<const SizeType>(&n, 1));
    REQUIRE(gpu.has_prepared(n));
    REQUIRE(gpu.n_cached_plans() == 1);
    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(kBatch * nf);
    thrust::device_vector<float> d_back(kBatch * n);
    gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                   loki::cuda_utils::as_span(d_spec), kBatch, n);
    // C2R of the host spectrum, so both directions are checked on their own.
    std::vector<ComplexTypeCUDA> cpu_spec_c(cpu_spec.size());
    for (SizeType i = 0; i < cpu_spec.size(); ++i) {
        cpu_spec_c[i] = {cpu_spec[i].real(), cpu_spec[i].imag()};
    }
    thrust::device_vector<ComplexTypeCUDA> d_cpu_spec(cpu_spec_c.begin(),
                                                      cpu_spec_c.end());
    gpu.irfft_batch(loki::cuda_utils::as_span(d_cpu_spec),
                    loki::cuda_utils::as_span(d_back), kBatch, n);
    cudaDeviceSynchronize();

    // Float-level agreement: errors scale like sqrt(n) of the unit-variance
    // input in the spectrum and like 1 / sqrt(n) in the normalized C2R.
    const auto spec = to_host(d_spec);
    const float tol_spec =
        2e-6F * static_cast<float>(n) * std::sqrt(static_cast<float>(n));
    for (SizeType i = 0; i < spec.size(); ++i) {
        REQUIRE_THAT(spec[i].real(), WithinAbs(cpu_spec[i].real(), tol_spec));
        REQUIRE_THAT(spec[i].imag(), WithinAbs(cpu_spec[i].imag(), tol_spec));
    }
    for (SizeType b = 0; b < kBatch; ++b) {
        REQUIRE(spec[b * nf].imag() == 0.0F);
        if (n % 2 == 0) {
            REQUIRE(spec[(b * nf) + (n / 2)].imag() == 0.0F);
        }
    }
    const auto back = to_host(d_back);
    require_reals_near(back, cpu_real,
                       2e-5F * std::sqrt(static_cast<float>(n)));
    require_reals_near(cpu_real, real_in, 1e-4F);
    // The input of the C2R is left unchanged.
    REQUIRE(to_host(d_cpu_spec) == cpu_spec_c);
}

TEST_CASE("GpuFFTManager rows do not depend on the batch",
          "[fft][GpuFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    const SizeType n          = GENERATE(64, 100);
    constexpr SizeType kBatch = 33;
    const SizeType nf         = (n / 2) + 1;
    const auto real_in        = random_rows(kBatch, n, 7);
    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_all(kBatch * nf);
    thrust::device_vector<ComplexTypeCUDA> d_one(nf);
    GpuFFTManager gpu(0);
    gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                   loki::cuda_utils::as_span(d_all), kBatch, n);
    // Row 17 alone.
    gpu.rfft_batch(loki::cuda_utils::as_span(d_real).subspan(17 * n, n),
                   loki::cuda_utils::as_span(d_one), 1, n);
    thrust::device_vector<float> d_r_all(kBatch * n);
    thrust::device_vector<float> d_r_one(n);
    gpu.irfft_batch(loki::cuda_utils::as_span(d_all),
                    loki::cuda_utils::as_span(d_r_all), kBatch, n);
    gpu.irfft_batch(loki::cuda_utils::as_span(d_one),
                    loki::cuda_utils::as_span(d_r_one), 1, n);
    const auto all   = to_host(d_all);
    const auto one   = to_host(d_one);
    const auto r_all = to_host(d_r_all);
    const auto r_one = to_host(d_r_one);
    for (SizeType k = 0; k < nf; ++k) {
        REQUIRE(cuda_complex_eq(all[(17 * nf) + k], one[k]));
    }
    for (SizeType j = 0; j < n; ++j) {
        REQUIRE(r_all[(17 * n) + j] == r_one[j]);
    }
}

TEST_CASE("GpuFFTManager limits", "[fft][GpuFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);
    GpuFFTManager gpu(0);
    const std::vector<SizeType> bad{1, loki::math::kGpuFFTMaxN + 1};
    for (const SizeType n : bad) {
        REQUIRE_THROWS(gpu.prepare_plans(std::span<const SizeType>(&n, 1)));
    }
    // An empty batch does nothing.
    thrust::device_vector<float> d_real(64);
    thrust::device_vector<ComplexTypeCUDA> d_spec(33);
    gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                   loki::cuda_utils::as_span(d_spec), 0, 64);
    // Undersized buffers are rejected.
    REQUIRE_THROWS(gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                                  loki::cuda_utils::as_span(d_spec), 2, 64));
    REQUIRE(gpu.n_cached_plans() == 0);
}

TEST_CASE("rfft_batch_cuda / irfft_batch_cuda convenience wrappers",
          "[fft][GpuFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 50;
    constexpr SizeType batch_size = 16;
    const SizeType n_complex      = (n_real / 2) + 1;
    auto real_in                  = make_sine_batch(batch_size, n_real);

    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(batch_size * n_complex);
    thrust::device_vector<float> d_recovered(batch_size * n_real);

    rfft_batch_cuda(loki::cuda_utils::as_span(d_real),
                    loki::cuda_utils::as_span(d_spec), batch_size, n_real);
    irfft_batch_cuda(loki::cuda_utils::as_span(d_spec),
                     loki::cuda_utils::as_span(d_recovered), batch_size,
                     n_real);

    std::vector<float> gpu_recovered(batch_size * n_real);
    thrust::copy(d_recovered.begin(), d_recovered.end(), gpu_recovered.begin());
    require_round_trip(real_in, gpu_recovered);
}

#endif // LOKI_ENABLE_CUDA
