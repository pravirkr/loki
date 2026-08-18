#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/utils/fft.hpp"

#ifdef LOKI_ENABLE_CUDA
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "loki/cuda_utils.cuh"
#endif // LOKI_ENABLE_CUDA

using Catch::Matchers::WithinAbs;
using loki::ComplexType;
using loki::SizeType;
using loki::math::FFTWManager;
using loki::math::irfft_batch;
using loki::math::kFFTBatchSizeMax;
using loki::math::rfft_batch;

namespace {

std::vector<float> make_sine_batch(SizeType batch_size, SizeType n_real) {
    std::vector<float> data(batch_size * n_real);
    for (SizeType b = 0; b < batch_size; ++b) {
        for (SizeType i = 0; i < n_real; ++i) {
            data[(b * n_real) + i] = std::sin(2.0F * static_cast<float>(M_PI) *
                                              static_cast<float>(i + 1 + b) /
                                              static_cast<float>(n_real));
        }
    }
    return data;
}

void require_reals_equal(std::span<const float> lhs,
                         std::span<const float> rhs) {
    REQUIRE(lhs.size() == rhs.size());
    for (SizeType i = 0; i < lhs.size(); ++i) {
        REQUIRE(lhs[i] == rhs[i]);
    }
}

void require_spectra_equal(std::span<const ComplexType> lhs,
                           std::span<const ComplexType> rhs) {
    REQUIRE(lhs.size() == rhs.size());
    for (SizeType i = 0; i < lhs.size(); ++i) {
        REQUIRE(lhs[i] == rhs[i]);
    }
}

void require_round_trip(std::span<const float> real_in,
                        std::span<const float> recovered) {
    REQUIRE(real_in.size() == recovered.size());
    for (SizeType i = 0; i < real_in.size(); ++i) {
        REQUIRE_THAT(recovered[i], WithinAbs(real_in[i], 1e-4F));
    }
}

} // namespace

TEST_CASE("FFTWManager max_howmany caps cached decomposition",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real      = 64;
    constexpr SizeType batch_size  = 5000;
    constexpr SizeType max_howmany = 256;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> cached_out(batch_size * ((n_real / 2) + 1));
    std::vector<ComplexType> ephemeral_out(cached_out.size());

    FFTWManager prepared;
    prepared.prepare_plans(std::span<const SizeType>(&n_real, 1), max_howmany);
    REQUIRE(prepared.has_prepared(n_real));

    prepared.rfft_batch(real_in, cached_out, batch_size, n_real, 8);

    FFTWManager ephemeral;
    ephemeral.rfft_batch(real_in, ephemeral_out, batch_size, n_real, 8);

    for (SizeType i = 0; i < cached_out.size(); ++i) {
        REQUIRE_THAT(cached_out[i].real(),
                     WithinAbs(ephemeral_out[i].real(), 1e-4F));
        REQUIRE_THAT(cached_out[i].imag(),
                     WithinAbs(ephemeral_out[i].imag(), 1e-4F));
    }
}

TEST_CASE("FFTWManager round-trip with prepared cache", "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 32;

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
    constexpr SizeType n_real = 64;

    FFTWManager manager;
    manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 256);
    const SizeType plans_after_first = manager.n_cached_plans();

    manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 256);
    REQUIRE(manager.n_cached_plans() == plans_after_first);

    manager.prepare_plans(std::span<const SizeType>(&n_real, 1),
                          kFFTBatchSizeMax);
    REQUIRE(manager.n_cached_plans() > plans_after_first);

    REQUIRE_THROWS_AS(
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 128),
        loki::error_check::DetailedException);
}

TEST_CASE("FFTWManager rfft preserves real input", "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 40;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));

    SECTION("ephemeral") {
        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
        require_reals_equal(real_in, real_before);
    }
    SECTION("ladder") {
        FFTWManager manager;
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 16);
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
        require_reals_equal(real_in, real_before);
    }
}

TEST_CASE("FFTWManager irfft round-trip without PRESERVE_INPUT",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 40;

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
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 32;
    constexpr SizeType batch_size = kFFTBatchSizeMax + 1;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager manager;
    manager.rfft_batch(real_in, spectrum, batch_size, n_real, 8);
    require_reals_equal(real_in, real_before);

    const auto spectrum_before = spectrum;
    std::vector<ComplexType> spectrum_scratch = spectrum;
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 8);
    require_spectra_equal(spectrum, spectrum_before);
    require_round_trip(real_in, recovered);
}

TEST_CASE("FFTWManager exact-howmany lazy cache reuses plans",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 200;

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

    for (SizeType i = 0; i < recovered_a.size(); ++i) {
        REQUIRE_THAT(recovered_b[i], WithinAbs(recovered_a[i], 1e-5F));
    }
}

TEST_CASE("FFTWManager exact-howmany caches two chunk sizes",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 32;
    constexpr SizeType batch_size = kFFTBatchSizeMax + 1;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);

    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    const auto spectrum_before = spectrum;
    std::vector<ComplexType> spectrum_scratch = spectrum;
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 2);
    require_spectra_equal(spectrum, spectrum_before);

    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);
    const auto spectrum_before_b = spectrum;
    spectrum_scratch = spectrum;
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 2);
    require_spectra_equal(spectrum, spectrum_before_b);

    REQUIRE_THROWS_AS(
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 1),
        loki::error_check::DetailedException);
}

TEST_CASE("FFTWManager exact vs ladder mode conflict", "[fft][FFTWManager]") {
    constexpr SizeType n_real = 64;
    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    REQUIRE_THROWS_AS(
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1)),
        loki::error_check::DetailedException);
}

TEST_CASE("FFTWManager chunking and empty batch", "[fft][FFTWManager]") {
    constexpr SizeType n_real = 64;

    SECTION("nthreads greater than batch_size") {
        constexpr SizeType batch_size = 3;
        auto real_in                  = make_sine_batch(batch_size, n_real);
        const auto real_before        = real_in;
        std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
        std::vector<float> recovered(batch_size * n_real);

        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 8);
        require_reals_equal(real_in, real_before);
        manager.irfft_batch(spectrum, recovered, batch_size, n_real, 8);
        require_round_trip(real_in, recovered);
    }

    SECTION("batch_size just below kFFTBatchSizeMax") {
        constexpr SizeType batch_size = kFFTBatchSizeMax - 1;
        auto real_in                  = make_sine_batch(batch_size, n_real);
        std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, n_real, 4);
        REQUIRE(spectrum.size() == batch_size * ((n_real / 2) + 1));
    }

    SECTION("batch_size zero is a no-op") {
        FFTWManager manager;
        std::vector<float> empty_real;
        std::vector<ComplexType> empty_spec;
        manager.rfft_batch(empty_real, empty_spec, 0, n_real, 4);
        manager.irfft_batch(empty_spec, empty_real, 0, n_real, 4);
    }

    SECTION("odd n_real") {
        constexpr SizeType odd_n      = 33;
        constexpr SizeType batch_size = 8;
        auto real_in                  = make_sine_batch(batch_size, odd_n);
        const auto real_before        = real_in;
        std::vector<ComplexType> spectrum(batch_size * ((odd_n / 2) + 1));
        std::vector<float> recovered(batch_size * odd_n);

        FFTWManager manager;
        manager.rfft_batch(real_in, spectrum, batch_size, odd_n, 2);
        require_reals_equal(real_in, real_before);
        manager.irfft_batch(spectrum, recovered, batch_size, odd_n, 2);
        require_round_trip(real_in, recovered);
    }
}

TEST_CASE("FFTWManager rfft preserves real input; irfft round-trip",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 16;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    rfft_batch(real_in, spectrum, batch_size, n_real, 2);
    require_reals_equal(real_in, real_before);

    irfft_batch(spectrum, recovered, batch_size, n_real, 2);
    require_round_trip(real_in, recovered);
}

TEST_CASE("EP-style copy before irfft preserves source spectrum",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 16;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<ComplexType> spectrum_scratch(spectrum.size());
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);
    const auto spectrum_before = spectrum;

    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    std::copy(spectrum.begin(), spectrum.end(), spectrum_scratch.begin());
    manager.irfft_batch(spectrum_scratch, recovered, batch_size, n_real, 1);

    require_spectra_equal(spectrum, spectrum_before);
    require_round_trip(real_in, recovered);
}

#ifdef LOKI_ENABLE_CUDA

using loki::ComplexTypeCUDA;
using loki::math::CUFFTManager;
using loki::math::irfft_batch_cuda;
using loki::math::rfft_batch_cuda;

namespace {

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace

TEST_CASE("CUFFTManager round-trip matches FFTWManager",
          "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 32;
    auto real_in                  = make_sine_batch(batch_size, n_real);
    const SizeType n_complex      = (n_real / 2) + 1;

    std::vector<ComplexType> cpu_spec(batch_size * n_complex);
    std::vector<float> cpu_recovered(batch_size * n_real);
    FFTWManager cpu;
    cpu.prepare_plans(std::span<const SizeType>(&n_real, 1));
    cpu.rfft_batch(real_in, cpu_spec, batch_size, n_real, 1);
    cpu.irfft_batch(cpu_spec, cpu_recovered, batch_size, n_real, 1);

    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(batch_size * n_complex);
    thrust::device_vector<float> d_recovered(batch_size * n_real);

    CUFFTManager gpu(0);
    gpu.prepare_plans(std::span<const SizeType>(&n_real, 1));
    gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                   loki::cuda_utils::as_span(d_spec), batch_size, n_real);
    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec),
                    loki::cuda_utils::as_span(d_recovered), batch_size, n_real);

    std::vector<ComplexTypeCUDA> gpu_spec(batch_size * n_complex);
    std::vector<float> gpu_recovered(batch_size * n_real);
    thrust::copy(d_spec.begin(), d_spec.end(), gpu_spec.begin());
    thrust::copy(d_recovered.begin(), d_recovered.end(), gpu_recovered.begin());

    for (SizeType i = 0; i < cpu_spec.size(); ++i) {
        REQUIRE_THAT(gpu_spec[i].real(), WithinAbs(cpu_spec[i].real(), 1e-4F));
        REQUIRE_THAT(gpu_spec[i].imag(), WithinAbs(cpu_spec[i].imag(), 1e-4F));
    }
    require_round_trip(real_in, gpu_recovered);
    require_round_trip(cpu_recovered, gpu_recovered);
}

TEST_CASE("CUFFTManager leftover chunk and plan reuse", "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 64;
    constexpr SizeType max_batch  = 8;
    constexpr SizeType batch_size = max_batch + 2;
    auto real_in                  = make_sine_batch(batch_size, n_real);
    const SizeType n_complex      = (n_real / 2) + 1;

    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(batch_size * n_complex);
    thrust::device_vector<float> d_recovered(batch_size * n_real);

    CUFFTManager gpu(0);
    gpu.prepare_plans(std::span<const SizeType>(&n_real, 1), max_batch);
    REQUIRE(gpu.has_prepared(n_real));
    REQUIRE(gpu.n_cached_plans() == 2);

    gpu.rfft_batch(
        loki::cuda_utils::as_span(d_real).first(max_batch * n_real),
        loki::cuda_utils::as_span(d_spec).first(max_batch * n_complex),
        max_batch, n_real);
    REQUIRE(gpu.n_cached_plans() == 2);

    gpu.rfft_batch(
        loki::cuda_utils::as_span(d_real).first(max_batch * n_real),
        loki::cuda_utils::as_span(d_spec).first(max_batch * n_complex),
        max_batch, n_real);
    REQUIRE(gpu.n_cached_plans() == 2);

    gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                   loki::cuda_utils::as_span(d_spec), batch_size, n_real);
    REQUIRE(gpu.n_cached_plans() == 3);

    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec),
                    loki::cuda_utils::as_span(d_recovered), batch_size, n_real);
    REQUIRE(gpu.n_cached_plans() == 4);

    std::vector<float> gpu_recovered(batch_size * n_real);
    thrust::copy(d_recovered.begin(), d_recovered.end(), gpu_recovered.begin());
    require_round_trip(real_in, gpu_recovered);

    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec),
                    loki::cuda_utils::as_span(d_recovered), batch_size, n_real);
    REQUIRE(gpu.n_cached_plans() == 4);
}

TEST_CASE("CUFFTManager C2R overwrites complex input", "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 16;
    auto real_in                  = make_sine_batch(batch_size, n_real);
    const SizeType n_complex      = (n_real / 2) + 1;

    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(batch_size * n_complex);
    thrust::device_vector<float> d_recovered(batch_size * n_real);

    CUFFTManager gpu(0);
    gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                   loki::cuda_utils::as_span(d_spec), batch_size, n_real);

    std::vector<ComplexTypeCUDA> spec_before(batch_size * n_complex);
    thrust::copy(d_spec.begin(), d_spec.end(), spec_before.begin());

    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec),
                    loki::cuda_utils::as_span(d_recovered), batch_size, n_real);

    std::vector<ComplexTypeCUDA> spec_after(batch_size * n_complex);
    thrust::copy(d_spec.begin(), d_spec.end(), spec_after.begin());

    bool overwritten = false;
    for (SizeType i = 0; i < spec_before.size(); ++i) {
        if (spec_before[i].real() != spec_after[i].real() ||
            spec_before[i].imag() != spec_after[i].imag()) {
            overwritten = true;
            break;
        }
    }
    REQUIRE(overwritten);
}

TEST_CASE("CUFFTManager empty batch is a no-op", "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real = 64;
    thrust::device_vector<float> empty_real;
    thrust::device_vector<ComplexTypeCUDA> empty_spec;
    CUFFTManager gpu(0);
    gpu.rfft_batch(loki::cuda_utils::as_span(empty_real),
                   loki::cuda_utils::as_span(empty_spec), 0, n_real);
    gpu.irfft_batch(loki::cuda_utils::as_span(empty_spec),
                    loki::cuda_utils::as_span(empty_real), 0, n_real);
    REQUIRE(gpu.n_cached_plans() == 0);
}

TEST_CASE("CUFFTManager prepare_plans reject shrink", "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real = 64;
    CUFFTManager gpu(0);
    gpu.prepare_plans(std::span<const SizeType>(&n_real, 1), 256);
    const SizeType plans_after_first = gpu.n_cached_plans();
    gpu.prepare_plans(std::span<const SizeType>(&n_real, 1), 256);
    REQUIRE(gpu.n_cached_plans() == plans_after_first);
    REQUIRE_THROWS_AS(
        gpu.prepare_plans(std::span<const SizeType>(&n_real, 1), 128),
        loki::error_check::DetailedException);
}

TEST_CASE("CUFFTManager exact-batch lazy cache reuses plans",
          "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 200;
    const SizeType n_complex      = (n_real / 2) + 1;
    auto real_in                  = make_sine_batch(batch_size, n_real);

    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(batch_size * n_complex);
    thrust::device_vector<float> d_recovered_a(batch_size * n_real);
    thrust::device_vector<float> d_recovered_b(batch_size * n_real);

    CUFFTManager spectral(0);
    spectral.rfft_batch(loki::cuda_utils::as_span(d_real),
                        loki::cuda_utils::as_span(d_spec), batch_size, n_real);

    CUFFTManager gpu(0);
    gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    REQUIRE(gpu.has_prepared(n_real));
    REQUIRE(gpu.n_cached_plans() == 0);

    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec),
                    loki::cuda_utils::as_span(d_recovered_a), batch_size,
                    n_real);
    REQUIRE(gpu.n_cached_plans() == 1);

    spectral.rfft_batch(loki::cuda_utils::as_span(d_real),
                        loki::cuda_utils::as_span(d_spec), batch_size, n_real);
    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec),
                    loki::cuda_utils::as_span(d_recovered_b), batch_size,
                    n_real);
    REQUIRE(gpu.n_cached_plans() == 1);

    std::vector<float> recovered_a(batch_size * n_real);
    std::vector<float> recovered_b(batch_size * n_real);
    thrust::copy(d_recovered_a.begin(), d_recovered_a.end(),
                 recovered_a.begin());
    thrust::copy(d_recovered_b.begin(), d_recovered_b.end(),
                 recovered_b.begin());
    for (SizeType i = 0; i < recovered_a.size(); ++i) {
        REQUIRE_THAT(recovered_b[i], WithinAbs(recovered_a[i], 1e-5F));
    }
}

TEST_CASE("CUFFTManager exact-batch caches two chunk sizes",
          "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 32;
    constexpr SizeType max_batch  = 8;
    constexpr SizeType batch_size = max_batch + 1;
    const SizeType n_complex      = (n_real / 2) + 1;
    auto real_in                  = make_sine_batch(batch_size, n_real);

    thrust::device_vector<float> d_real(real_in.begin(), real_in.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec(batch_size * n_complex);
    thrust::device_vector<ComplexTypeCUDA> d_spec_scratch(batch_size *
                                                         n_complex);
    thrust::device_vector<float> d_recovered(batch_size * n_real);

    CUFFTManager spectral(0);
    spectral.rfft_batch(loki::cuda_utils::as_span(d_real),
                        loki::cuda_utils::as_span(d_spec), batch_size, n_real);
    std::vector<ComplexTypeCUDA> spec_before(batch_size * n_complex);
    thrust::copy(d_spec.begin(), d_spec.end(), spec_before.begin());

    CUFFTManager gpu(0);
    gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1), max_batch);
    thrust::copy(d_spec.begin(), d_spec.end(), d_spec_scratch.begin());
    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec_scratch),
                    loki::cuda_utils::as_span(d_recovered), batch_size, n_real);
    REQUIRE(gpu.n_cached_plans() == 2);

    std::vector<ComplexTypeCUDA> spec_after(batch_size * n_complex);
    thrust::copy(d_spec.begin(), d_spec.end(), spec_after.begin());
    for (SizeType i = 0; i < spec_before.size(); ++i) {
        REQUIRE(spec_before[i].real() == spec_after[i].real());
        REQUIRE(spec_before[i].imag() == spec_after[i].imag());
    }

    std::vector<float> gpu_recovered(batch_size * n_real);
    thrust::copy(d_recovered.begin(), d_recovered.end(), gpu_recovered.begin());
    require_round_trip(real_in, gpu_recovered);

    spectral.rfft_batch(loki::cuda_utils::as_span(d_real),
                        loki::cuda_utils::as_span(d_spec), batch_size, n_real);
    thrust::copy(d_spec.begin(), d_spec.end(), d_spec_scratch.begin());
    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec_scratch),
                    loki::cuda_utils::as_span(d_recovered), batch_size, n_real);
    REQUIRE(gpu.n_cached_plans() == 2);

    REQUIRE_THROWS_AS(
        gpu.rfft_batch(loki::cuda_utils::as_span(d_real),
                       loki::cuda_utils::as_span(d_spec), batch_size, n_real),
        loki::error_check::DetailedException);
}

TEST_CASE("CUFFTManager exact vs max-batch mode conflict",
          "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real = 64;

    SECTION("exact then prepare_plans") {
        CUFFTManager gpu(0);
        gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
        REQUIRE_THROWS_AS(
            gpu.prepare_plans(std::span<const SizeType>(&n_real, 1)),
            loki::error_check::DetailedException);
    }
    SECTION("prepare_plans then exact") {
        CUFFTManager gpu(0);
        gpu.prepare_plans(std::span<const SizeType>(&n_real, 1));
        REQUIRE_THROWS_AS(
            gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1)),
            loki::error_check::DetailedException);
    }
    SECTION("exact reject shrink") {
        CUFFTManager gpu(0);
        gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1), 256);
        gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1), 256);
        REQUIRE_THROWS_AS(
            gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1), 128),
            loki::error_check::DetailedException);
    }
}

TEST_CASE("CUFFTManager exact-batch two sizes round-trip",
          "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real  = 64;
    constexpr SizeType batch_a = 16;
    constexpr SizeType batch_b = 24;
    const SizeType n_complex   = (n_real / 2) + 1;
    auto real_in_a             = make_sine_batch(batch_a, n_real);
    auto real_in_b             = make_sine_batch(batch_b, n_real);

    thrust::device_vector<float> d_real_a(real_in_a.begin(), real_in_a.end());
    thrust::device_vector<float> d_real_b(real_in_b.begin(), real_in_b.end());
    thrust::device_vector<ComplexTypeCUDA> d_spec_a(batch_a * n_complex);
    thrust::device_vector<ComplexTypeCUDA> d_spec_b(batch_b * n_complex);
    thrust::device_vector<float> d_out_a(batch_a * n_real);
    thrust::device_vector<float> d_out_b(batch_b * n_real);

    CUFFTManager spectral(0);
    spectral.rfft_batch(loki::cuda_utils::as_span(d_real_a),
                        loki::cuda_utils::as_span(d_spec_a), batch_a, n_real);
    spectral.rfft_batch(loki::cuda_utils::as_span(d_real_b),
                        loki::cuda_utils::as_span(d_spec_b), batch_b, n_real);

    CUFFTManager gpu(0);
    gpu.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec_a),
                    loki::cuda_utils::as_span(d_out_a), batch_a, n_real);
    gpu.irfft_batch(loki::cuda_utils::as_span(d_spec_b),
                    loki::cuda_utils::as_span(d_out_b), batch_b, n_real);
    REQUIRE(gpu.n_cached_plans() == 2);

    std::vector<float> out_a(batch_a * n_real);
    std::vector<float> out_b(batch_b * n_real);
    thrust::copy(d_out_a.begin(), d_out_a.end(), out_a.begin());
    thrust::copy(d_out_b.begin(), d_out_b.end(), out_b.begin());
    require_round_trip(real_in_a, out_a);
    require_round_trip(real_in_b, out_b);
}

TEST_CASE("rfft_batch_cuda / irfft_batch_cuda convenience wrappers",
          "[fft][CUFFTManager]") {
    if (!cuda_device_available()) {
        SKIP("No CUDA device");
    }
    loki::cuda_utils::CudaSetDeviceGuard device_guard(0);

    constexpr SizeType n_real     = 64;
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
