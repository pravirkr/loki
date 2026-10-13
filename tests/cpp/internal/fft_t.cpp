#include <algorithm>
#include <cmath>
#include <complex>
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
#include <pocketfft_hdronly.h>

#include "loki/common/types.hpp"

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
using loki::math::CpuFFT;
using loki::math::irfft_batch;
using loki::math::rfft_batch;

#ifdef LOKI_ENABLE_CUDA
using loki::ComplexTypeCUDA;
#endif

namespace {

constexpr float kFftTol = 1e-4F;

auto float_near(float margin) {
    return [margin](float actual, float expected) {
        return WithinAbs(expected, margin).match(actual);
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

#ifdef LOKI_ENABLE_CUDA
void require_reals_near(std::span<const float> lhs,
                        std::span<const float> rhs,
                        float margin) {
    REQUIRE_THAT(lhs, RangeEquals(rhs, float_near(margin)));
}
#endif // LOKI_ENABLE_CUDA

} // namespace

TEST_CASE("CpuFFT matches pocketfft in double precision", "[fft][CpuFFT]") {
    // Lane FFT lengths (direct, odd, Bluestein) and pocketfft lengths
    // (n = 1 and n > kLaneFFTMaxSize).
    const SizeType n_real =
        GENERATE(SizeType{1}, SizeType{2}, SizeType{3}, SizeType{17},
                 SizeType{50}, SizeType{64}, SizeType{509}, SizeType{1024},
                 SizeType{1025}, SizeType{3000}, SizeType{1} << 14U);
    const SizeType batch_size =
        GENERATE(SizeType{1}, SizeType{5}, SizeType{40});
    const int nthreads       = GENERATE(1, 3);
    const SizeType n_complex = (n_real / 2) + 1;
    INFO("n_real=" << n_real << " batch=" << batch_size
                   << " nthreads=" << nthreads);
    std::mt19937 rng(static_cast<unsigned>(n_real + batch_size));
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> real_d(batch_size * n_real);
    std::vector<std::complex<double>> spec_d(batch_size * n_complex);
    for (auto& x : real_d) {
        x = dist(rng);
    }
    for (auto& c : spec_d) {
        c = {dist(rng), dist(rng)};
    }
    std::vector<float> real_f(real_d.begin(), real_d.end());
    std::vector<ComplexType> spec_f(spec_d.begin(), spec_d.end());
    // Round the double inputs to float so both see the same values.
    std::ranges::copy(real_f, real_d.begin());
    std::ranges::copy(spec_f, spec_d.begin());

    const pocketfft::shape_t shape{batch_size, n_real};
    const pocketfft::stride_t stride_r{
        static_cast<std::ptrdiff_t>(n_real * sizeof(double)), sizeof(double)};
    const pocketfft::stride_t stride_c{
        static_cast<std::ptrdiff_t>(n_complex * sizeof(std::complex<double>)),
        sizeof(std::complex<double>)};
    std::vector<std::complex<double>> fwd_ref(batch_size * n_complex);
    pocketfft::r2c(shape, stride_r, stride_c, /*axis=*/1, pocketfft::FORWARD,
                   real_d.data(), fwd_ref.data(), 1.0);
    std::vector<double> inv_ref(batch_size * n_real);
    pocketfft::c2r(shape, stride_c, stride_r, /*axis=*/1, pocketfft::BACKWARD,
                   spec_d.data(), inv_ref.data(),
                   1.0 / static_cast<double>(n_real));

    CpuFFT manager;
    std::vector<ComplexType> fwd(batch_size * n_complex);
    manager.rfft_batch(real_f, fwd, batch_size, n_real, nthreads);
    std::vector<float> inv(batch_size * n_real);
    manager.irfft_batch(spec_f, inv, batch_size, n_real, nthreads);

    const double tol = 2e-6 * std::log2(static_cast<double>(2 * n_real));
    for (SizeType row = 0; row < batch_size; ++row) {
        double scale = 0.0;
        double err   = 0.0;
        for (SizeType k = 0; k < n_complex; ++k) {
            const auto ref = fwd_ref[(row * n_complex) + k];
            scale          = std::max(scale, std::abs(ref));
            err =
                std::max(err, std::abs(ref - std::complex<double>(
                                                 fwd[(row * n_complex) + k])));
        }
        INFO("forward row " << row);
        CHECK(err <= tol * scale);
        scale = 0.0;
        err   = 0.0;
        for (SizeType j = 0; j < n_real; ++j) {
            const double ref = inv_ref[(row * n_real) + j];
            scale            = std::max(scale, std::abs(ref));
            err = std::max(err, std::abs(ref - inv[(row * n_real) + j]));
        }
        INFO("inverse row " << row);
        CHECK(err <= tol * scale);
    }
}

TEST_CASE("CpuFFT leaves its input unchanged", "[fft][CpuFFT]") {
    const SizeType n_real     = GENERATE(SizeType{64}, SizeType{3000});
    const SizeType batch_size = 9;
    const SizeType n_complex  = (n_real / 2) + 1;
    const auto real_in        = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * n_complex);
    CpuFFT manager;
    manager.rfft_batch(real_in, spectrum, batch_size, n_real, 2);
    REQUIRE_THAT(real_in, RangeEquals(make_sine_batch(batch_size, n_real)));
    const auto spectrum_before = spectrum;
    std::vector<float> recovered(batch_size * n_real);
    manager.irfft_batch(spectrum, recovered, batch_size, n_real, 2);
    REQUIRE(std::memcmp(spectrum.data(), spectrum_before.data(),
                        spectrum.size() * sizeof(ComplexType)) == 0);
    require_round_trip(real_in, recovered);
}

TEST_CASE("CpuFFT executes on misaligned batch slices", "[fft][CpuFFT]") {
    // Slices of an odd n_complex batch, and buffers off a SIMD boundary.
    const SizeType n_real     = GENERATE(SizeType{64}, SizeType{128});
    const SizeType n_complex  = (n_real / 2) + 1;
    const SizeType batch_size = 77;
    const int nthreads        = GENERATE(1, 3);

    auto real_ref = make_sine_batch(batch_size, n_real);
    // Shift both buffers by one element off their allocation.
    std::vector<float> real_storage(real_ref.size() + 1);
    std::ranges::copy(real_ref, real_storage.begin() + 1);
    const auto real_in = std::span<float>(real_storage).subspan(1);
    std::vector<ComplexType> spec_storage((batch_size * n_complex) + 1);
    const auto spectrum =
        std::span<ComplexType>(spec_storage).subspan(1, batch_size * n_complex);
    std::vector<float> back_storage(real_ref.size() + 1);
    const auto recovered = std::span<float>(back_storage).subspan(1);

    CpuFFT manager;
    manager.rfft_batch(real_in, spectrum, batch_size, n_real, nthreads);
    std::vector<ComplexType> spectrum_ref(batch_size * n_complex);
    CpuFFT reference;
    reference.rfft_batch(real_ref, spectrum_ref, batch_size, n_real, 1);
    // The same row pairs: identical whatever the alignment or thread count.
    REQUIRE(std::memcmp(spectrum.data(), spectrum_ref.data(),
                        spectrum_ref.size() * sizeof(ComplexType)) == 0);
    manager.irfft_batch(spectrum, recovered, batch_size, n_real, nthreads);
    require_round_trip(real_ref, recovered);
}

TEST_CASE("CpuFFT records prepared and built lengths", "[fft][CpuFFT]") {
    const std::vector<SizeType> lazy{50};
    const std::vector<SizeType> eager{64, 3000};
    CpuFFT manager;
    CHECK_FALSE(manager.has_prepared(50));
    manager.prepare_exact_plans(lazy);
    CHECK(manager.has_prepared(50));
    CHECK(manager.n_cached_plans() == 0);
    manager.prepare_plans(eager);
    CHECK(manager.has_prepared(64));
    CHECK(manager.has_prepared(3000));
    CHECK(manager.n_cached_plans() == 2);
    // A lazily registered length is built on first use.
    std::vector<ComplexType> spec(2 * 26);
    std::vector<float> real(2 * 50);
    manager.irfft_batch(spec, real, 2, 50, 1);
    CHECK(manager.n_cached_plans() == 3);
    // Moving keeps the bookkeeping.
    const CpuFFT moved(std::move(manager));
    CHECK(moved.has_prepared(50));
    CHECK(moved.n_cached_plans() == 3);
}

TEST_CASE("CpuFFT empty batch and size checks", "[fft][CpuFFT]") {
    const SizeType n_real = 64;
    CpuFFT manager;
    std::vector<float> empty_real;
    std::vector<ComplexType> empty_spec;
    manager.rfft_batch(empty_real, empty_spec, 0, n_real, 4);
    manager.irfft_batch(empty_spec, empty_real, 0, n_real, 4);
    std::vector<float> real(3 * n_real);
    std::vector<ComplexType> spec(2 * ((n_real / 2) + 1));
    CHECK_THROWS(manager.rfft_batch(real, spec, 3, n_real, 1));
    CHECK_THROWS(manager.irfft_batch(spec, real, 3, n_real, 1));
    CHECK_THROWS(manager.rfft_batch(real, spec, 3, 0, 1));
}

TEST_CASE("CircularConvolver matches a direct circular convolution", "[fft]") {
    // More rows in n1 than in n2, and lane and pocketfft row lengths.
    const SizeType ny  = GENERATE(SizeType{8}, SizeType{50}, SizeType{2048});
    const SizeType n1x = 3;
    const SizeType n2x = 2;
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> n1(n1x * ny);
    std::vector<float> n2(n2x * ny);
    for (auto& x : n1) {
        x = dist(rng);
    }
    for (auto& x : n2) {
        x = dist(rng);
    }
    loki::math::CircularConvolver convolver(n1x, n2x, ny);
    std::vector<float> out(n1x * n2x * ny);
    convolver.convolve(n1, n2, out);
    INFO("ny=" << ny);
    for (SizeType i = 0; i < n1x; ++i) {
        for (SizeType j = 0; j < n2x; ++j) {
            for (SizeType k = 0; k < ny; k += (ny > 64 ? 97 : 1)) {
                double ref = 0.0;
                for (SizeType m = 0; m < ny; ++m) {
                    ref += static_cast<double>(n1[(i * ny) + m]) *
                           n2[(j * ny) + ((k + ny - m) % ny)];
                }
                const float got = out[(((i * n2x) + j) * ny) + k];
                REQUIRE_THAT(got, WithinAbs(ref, 1e-4 * std::sqrt(double(ny))));
            }
        }
    }
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
