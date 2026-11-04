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

using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinAbs;
using loki::ComplexType;
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
