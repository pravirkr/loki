#include <cmath>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"
#include "loki/exceptions.hpp"
#include "loki/utils/fft.hpp"

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

TEST_CASE("FFTWManager irfft preserves complex input", "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 40;

    auto real_in = make_sine_batch(batch_size, n_real);
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    FFTWManager spectral;
    spectral.rfft_batch(real_in, spectrum, batch_size, n_real, 1);
    const auto spectrum_before = spectrum;

    SECTION("ephemeral") {
        FFTWManager manager;
        manager.irfft_batch(spectrum, recovered, batch_size, n_real, 4);
        require_spectra_equal(spectrum, spectrum_before);
        require_round_trip(real_in, recovered);
    }
    SECTION("ladder") {
        FFTWManager manager;
        manager.prepare_plans(std::span<const SizeType>(&n_real, 1), 16);
        manager.irfft_batch(spectrum, recovered, batch_size, n_real, 4);
        require_spectra_equal(spectrum, spectrum_before);
        require_round_trip(real_in, recovered);
    }
    SECTION("exact-howmany") {
        FFTWManager manager;
        manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
        manager.irfft_batch(spectrum, recovered, batch_size, n_real, 1);
        require_spectra_equal(spectrum, spectrum_before);
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
    manager.irfft_batch(spectrum, recovered, batch_size, n_real, 8);
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
    const auto spectrum_before = spectrum;

    FFTWManager manager;
    manager.prepare_exact_plans(std::span<const SizeType>(&n_real, 1));
    manager.irfft_batch(spectrum, recovered, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 2);
    require_spectra_equal(spectrum, spectrum_before);

    manager.irfft_batch(spectrum, recovered, batch_size, n_real, 1);
    REQUIRE(manager.n_cached_plans() == 2);

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

TEST_CASE("free rfft_batch and irfft_batch preserve input",
          "[fft][FFTWManager]") {
    constexpr SizeType n_real     = 64;
    constexpr SizeType batch_size = 16;

    auto real_in           = make_sine_batch(batch_size, n_real);
    const auto real_before = real_in;
    std::vector<ComplexType> spectrum(batch_size * ((n_real / 2) + 1));
    std::vector<float> recovered(batch_size * n_real);

    rfft_batch(real_in, spectrum, batch_size, n_real, 2);
    require_reals_equal(real_in, real_before);

    const auto spectrum_before = spectrum;
    irfft_batch(spectrum, recovered, batch_size, n_real, 2);
    require_spectra_equal(spectrum, spectrum_before);
    require_round_trip(real_in, recovered);
}
