#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include "loki/common/types.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/utils/fft_impl.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/fft_cuda.cuh"
#include "lib/cuda/types_cuda.cuh"
#endif // LOKI_ENABLE_CUDA

using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinAbs;
using loki::ComplexType;
using loki::SizeType;
using loki::math::FFTWManager;
using loki::math::irfft_batch;
using loki::math::kFFTBatchSizeMax;
using loki::math::rfft_batch;

#ifdef LOKI_ENABLE_CUDA
using loki::ComplexTypeCUDA;
#endif

namespace {

// Moderate batch for fast unit tests; stress cases use kFFTBatchSizeMax.
constexpr SizeType kFFTTestBatch = 256;
constexpr float kFftTol          = 1e-4F;
constexpr float kFftRepeatTol    = 1e-5F;

auto float_near(float margin) {
    return [margin](float actual, float expected) {
        return WithinAbs(expected, margin).match(actual);
    };
}

auto complex_near(float margin) {
    return [margin](const ComplexType& actual, const ComplexType& expected) {
        return WithinAbs(expected.real(), margin).match(actual.real()) &&
               WithinAbs(expected.imag(), margin).match(actual.imag());
    };
}

#ifdef LOKI_ENABLE_CUDA
auto cuda_complex_near(float margin) {
    return
        [margin](const ComplexTypeCUDA& actual, const ComplexType& expected) {
            return WithinAbs(expected.real(), margin).match(actual.real()) &&
                   WithinAbs(expected.imag(), margin).match(actual.imag());
        };
}

auto cuda_complex_eq(const ComplexTypeCUDA& lhs, const ComplexTypeCUDA& rhs) {
    return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}
#endif

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

TEST_CASE("FFTWManager max_howmany caps cached decomposition",
          "[fft][FFTWManager]") {
    const SizeType n_real      = 64;
    const SizeType batch_size  = 512;
    const SizeType max_howmany = 256;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> cached_out(batch_size * ((n_real / 2) + 1));
    std::vector<ComplexType> ephemeral_out(cached_out.size());

    FFTWManager prepared;
    prepared.prepare_plans(std::span<const SizeType>(&n_real, 1), max_howmany);
    REQUIRE(prepared.has_prepared(n_real));

    prepared.rfft_batch(real_in, cached_out, batch_size, n_real, 8);

    FFTWManager ephemeral;
    ephemeral.rfft_batch(real_in, ephemeral_out, batch_size, n_real, 8);

    REQUIRE_THAT(cached_out, RangeEquals(ephemeral_out, complex_near(kFftTol)));
}

TEST_CASE("FFTWManager round-trip with prepared cache", "[fft][FFTWManager]") {
    const SizeType n_real     = 64;
    const SizeType batch_size = 32;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager manager;
    manager.prepare_plans(std::span<const SizeType>(&n_real, 1));
    manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
    manager.irfft_batch(spectrum, recovered, batch_size, n_real, 4);

    require_round_trip(real_in, recovered);
}

TEST_CASE("FFTWManager prepare_plans extend and reject shrink",
          "[fft][FFTWManager]") {
    const SizeType n_real = 64;

    FFTWManager manager;
    manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 256);
    const SizeType plans_after_first = manager.n_cached_plans();

    manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 256);
    REQUIRE(manager.n_cached_plans() == plans_after_first);

    manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 1024);
    REQUIRE(manager.n_cached_plans() > plans_after_first);

    REQUIRE_THROWS_AS(
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 128),
        loki::error_check::DetailedException);
}

TEST_CASE("FFTWManager rfft preserves real input", "[fft][FFTWManager]") {
    const SizeType n_real     = 64;
    const SizeType batch_size = 40;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));

    SECTION("ephemeral") {
        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
        REQUIRE_THAT(real_in, RangeEquals(real_before));
    }
    SECTION("ladder") {
        FFTWManager manager;
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 16);
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
        REQUIRE_THAT(real_in, RangeEquals(real_before));
    }
}

TEST_CASE("FFTWManager irfft round-trip without PRESERVE_INPUT",
          "[fft][FFTWManager]") {
    const SizeType n_real     = 64;
    const SizeType batch_size = 40;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);

    SECTION("ephemeral") {
        std::vector<ComplexType> spectrum_work = spectrum;
        FFTWManager manager;
        manager.irfft_batch(spectrum_work, recovered, batch_size, n_real, 4);
        require_round_trip(real_in, recovered);
    }
    SECTION("ladder") {
        std::vector<ComplexType> spectrum_work = spectrum;
        FFTWManager manager;
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 16);
        manager.irfft_batch(spectrum_work, recovered, batch_size, n_real, 4);
        require_round_trip(real_in, recovered);
    }
    SECTION("exact-howmany") {
        std::vector<ComplexType> spectrum_work = spectrum;
        FFTWManager manager;
        manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
        manager.irfft_batch(spectrum_work, recovered, batch_size, n_real, 1);
        require_round_trip(real_in, recovered);
    }
}

TEST_CASE("FFTWManager preserves input when batch exceeds kFFTBatchSizeMax",
          "[fft][FFTWManager][stress]") {
    const SizeType n_real     = 32;
    const SizeType batch_size = kFFTBatchSizeMax + 1;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager manager;
    manager.rfft_batch(real_in, spectrum, batch_size, n_real, 8);
    REQUIRE_THAT(real_in, RangeEquals(real_before));

    const auto spectrum_before                = spectrum;
    std::vector<ComplexType> spectrum_scratch = spectrum;
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 8);
    REQUIRE_THAT(spectrum, RangeEquals(spectrum_before));
    require_round_trip(real_in, recovered);
}

TEST_CASE("FFTWManager exact-howmany lazy cache reuses plans",
          "[fft][FFTWManager]") {
    const SizeType n_real     = 64;
    const SizeType batch_size = 200;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered_a(batch_size * n_real);
    std::vector<float> recovered_b(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);

    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    REQUIRE(manager.n_cached_plans() == 0);

    manager.irfft_batch(spectrum, recovered_a, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 1);

    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);
    manager.irfft_batch(spectrum, recovered_b, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 1);

    require_reals_near(recovered_b, recovered_a, kFftRepeatTol);
}

TEST_CASE("FFTWManager exact-howmany caches two chunk sizes",
          "[fft][FFTWManager]") {
    const SizeType n_real      = 32;
    const SizeType max_howmany = 128;
    const SizeType batch_size  = 200;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);

    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1),
                                max_howmany);
    const auto spectrum_before                = spectrum;
    std::vector<ComplexType> spectrum_scratch = spectrum;
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 2);
    REQUIRE_THAT(spectrum, RangeEquals(spectrum_before));

    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);
    const auto spectrum_before_b = spectrum;
    spectrum_scratch             = spectrum;
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 2);
    REQUIRE_THAT(spectrum, RangeEquals(spectrum_before_b));

    REQUIRE_THROWS_AS(
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 1),
        loki::error_check::DetailedException);
}

TEST_CASE("FFTWManager exact vs ladder mode conflict", "[fft][FFTWManager]") {
    const SizeType n_real = 64;
    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    REQUIRE_THROWS_AS(
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1)),
        loki::error_check::DetailedException);
}

TEST_CASE("FFTWManager chunking and empty batch", "[fft][FFTWManager]") {
    const SizeType n_real = 64;

    SECTION("nthreads greater than batch_size") {
        const SizeType batch_size = 3;
        auto real_in              = make_sine_batch(batch_size, n_real);
        const auto real_before    = real_in;
        std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
        std::vector<float> recovered(batch_size * n_real);

        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 8);
        REQUIRE_THAT(real_in, RangeEquals(real_before));
        manager.irfft_batch(spectrum, recovered, batch_size, n_real, 8);
        require_round_trip(real_in, recovered);
    }

    SECTION("batch larger than single worker chunk round-trips") {
        const SizeType batch_size = kFFTTestBatch;
        auto real_in              = make_sine_batch(batch_size, n_real);
        std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
        std::vector<float> recovered(batch_size * n_real);
        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
        manager.irfft_batch(spectrum, recovered, batch_size, n_real, 4);
        require_round_trip(real_in, recovered);
    }

    SECTION("batch_size zero is a no-op") {
        FFTWManager manager;
        std::vector<float> empty_real;
        std::vector<ComplexType> empty_spec;
        manager.rfft_batch(empty_real, empty_spec, 0, n_real, 4);
        manager.irfft_batch(empty_spec, empty_real, 0, n_real, 4);
    }

    SECTION("odd n_real") {
        const SizeType odd_n      = 33;
        const SizeType batch_size = 8;
        auto real_in              = make_sine_batch(batch_size, odd_n);
        const auto real_before    = real_in;
        std::vector<ComplexType> spectrum(batch_size * ((odd_n / 2) + 1));
        std::vector<float> recovered(batch_size * odd_n);

        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, odd_n, 2);
        REQUIRE_THAT(real_in, RangeEquals(real_before));
        manager.irfft_batch(spectrum, recovered, batch_size, odd_n, 2);
        require_round_trip(real_in, recovered);
    }
}

TEST_CASE("FFTWManager rfft preserves real input; irfft round-trip",
          "[fft][FFTWManager]") {
    const SizeType n_real     = 64;
    const SizeType batch_size = 16;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    rfft_batch(real_in, spectrum, batch_size, n_real, 2);
    REQUIRE_THAT(real_in, RangeEquals(real_before));

    irfft_batch(spectrum, recovered, batch_size, n_real, 2);
    require_round_trip(real_in, recovered);
}

TEST_CASE("EP-style copy before irfft preserves source spectrum",
          "[fft][FFTWManager]") {
    const SizeType n_real     = 64;
    const SizeType batch_size = 16;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<ComplexType> spectrum_scratch(spectrum.size());
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);
    const auto spectrum_before = spectrum;

    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    std::ranges::copy(spectrum, spectrum_scratch.begin());
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 1);

    REQUIRE_THAT(spectrum, RangeEquals(spectrum_before));
    require_round_trip(real_in, recovered);
}

#ifdef LOKI_ENABLE_CUDA

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

TEST_CASE("GpuFFTManager matches FFTW for direct and Bluestein lengths",
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
    FFTWManager cpu;
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
    // C2R of FFTW's spectrum, so both directions are checked on their own.
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
