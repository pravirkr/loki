#include "lib/cpu/lane_fft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/types.hpp"

#include "lib/cpu/boxcar_kernels.hpp"

using loki::ComplexType;
using loki::SizeType;
namespace lane_fft = loki::lane_fft;

namespace {

using DComplex = std::complex<double>;

/// Unnormalised e^{sign 2 pi i jk / n} in double, the angle reduced exactly.
DComplex root(SizeType jk, SizeType n, double sign) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(jk % n) /
                         static_cast<double>(n);
    return std::polar(1.0, sign * angle);
}

/// c2r reference: Hermitian extension of x[0..n/2], imaginary parts of the
/// DC and Nyquist bins ignored, normalised by 1/n.
std::vector<double> ref_inverse(const ComplexType* x, SizeType n) {
    std::vector<double> out(n, 0.0);
    for (SizeType j = 0; j < n; ++j) {
        DComplex acc = 0.0;
        for (SizeType k = 0; k < n; ++k) {
            DComplex xk =
                k <= n / 2 ? DComplex(x[k]) : std::conj(DComplex(x[n - k]));
            if (k == 0 || 2 * k == n) {
                xk = xk.real();
            }
            acc += xk * root(j * k, n, 1.0);
        }
        out[j] = acc.real() / static_cast<double>(n);
    }
    return out;
}

std::vector<DComplex> ref_forward(const float* x, SizeType n) {
    std::vector<DComplex> out((n / 2) + 1, 0.0);
    for (SizeType k = 0; k <= n / 2; ++k) {
        for (SizeType j = 0; j < n; ++j) {
            out[k] += static_cast<double>(x[j]) * root(j * k, n, -1.0);
        }
    }
    return out;
}

} // namespace

TEST_CASE("Lane FFT inverse and forward match a double DFT", "[fft][lane]") {
    // Powers of two, mixed radices, odd sizes and Bluestein sizes (prime
    // factors > 13: 17, 34, 509).
    const SizeType n =
        GENERATE(SizeType{2}, SizeType{3}, SizeType{8}, SizeType{17},
                 SizeType{34}, SizeType{50}, SizeType{64}, SizeType{125},
                 SizeType{256}, SizeType{509}, SizeType{800}, SizeType{1024});
    // Rows pair up (e, v): a lone row, partial and full groups of 2 kLanes
    // rows, odd counts, and several groups with a tail.
    const SizeType nrows = GENERATE(SizeType{1}, SizeType{2}, SizeType{31},
                                    SizeType{32}, SizeType{75});
    const int nthreads   = GENERATE(1, 3);
    const SizeType nf    = (n / 2) + 1;
    std::mt19937 rng(static_cast<unsigned>(n * 131 + nrows));
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<ComplexType> spectra(nrows * nf);
    for (auto& c : spectra) {
        c = {dist(rng), dist(rng)};
    }
    std::vector<float> samples(nrows * n);
    for (auto& x : samples) {
        x = dist(rng);
    }
    INFO("n=" << n << " nrows=" << nrows << " nthreads=" << nthreads);
    // Rounding grows like log n; the bound is loose but far below any
    // algorithmic error (a wrong twiddle or index is O(1)).
    const double tol = 1e-6 * std::log2(static_cast<double>(2 * n));

    std::vector<float> inv(nrows * n, -1.0F);
    lane_fft::inverse_batch(spectra.data(), inv.data(), nrows, n, nthreads);
    for (SizeType row = 0; row < nrows; ++row) {
        const auto ref = ref_inverse(spectra.data() + (row * nf), n);
        double scale   = 0.0;
        double err     = 0.0;
        for (SizeType j = 0; j < n; ++j) {
            scale = std::max(scale, std::abs(ref[j]));
            err   = std::max(err, std::abs(ref[j] - inv[(row * n) + j]));
        }
        INFO("inverse row " << row);
        CHECK(err <= tol * scale);
    }

    std::vector<ComplexType> fwd(nrows * nf, ComplexType(-1.0F, -1.0F));
    lane_fft::forward_batch(samples.data(), fwd.data(), nrows, n, nthreads);
    for (SizeType row = 0; row < nrows; ++row) {
        const auto ref = ref_forward(samples.data() + (row * n), n);
        double scale   = 0.0;
        double err     = 0.0;
        for (SizeType k = 0; k < nf; ++k) {
            scale = std::max(scale, std::abs(ref[k]));
            err =
                std::max(err, std::abs(ref[k] - DComplex(fwd[(row * nf) + k])));
        }
        INFO("forward row " << row);
        CHECK(err <= tol * scale);
        // A real signal's DC and (even n) Nyquist bins are exactly real.
        CHECK(fwd[row * nf].imag() == 0.0F);
        if (n % 2 == 0) {
            CHECK(fwd[(row * nf) + nf - 1].imag() == 0.0F);
        }
    }
}

TEST_CASE("Lane FFT lane output equals its row output", "[fft][lane]") {
    const SizeType n =
        GENERATE(SizeType{8}, SizeType{50}, SizeType{67}, SizeType{512});
    const SizeType nprofiles = GENERATE(SizeType{5}, SizeType{16});
    const SizeType nf        = (n / 2) + 1;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<ComplexType> spectra(nprofiles * 2 * nf);
    for (auto& c : spectra) {
        c = {dist(rng), dist(rng)};
    }
    std::vector<float> rows(nprofiles * 2 * n);
    lane_fft::EVTransform transform(n);
    transform.inverse_rows(spectra.data(), 2 * nprofiles, rows.data());
    transform.inverse(spectra.data(), nprofiles);
    INFO("n=" << n << " nprofiles=" << nprofiles);
    for (SizeType lane = 0; lane < lane_fft::kLanes; ++lane) {
        for (SizeType j = 0; j < n; ++j) {
            const float e = transform.e_lanes()[(j * lane_fft::kLanes) + lane];
            const float v = transform.v_lanes()[(j * lane_fft::kLanes) + lane];
            if (lane < nprofiles) {
                REQUIRE(std::memcmp(&e, &rows[(lane * 2 * n) + j], sizeof e) ==
                        0);
                REQUIRE(std::memcmp(&v, &rows[(lane * 2 * n) + n + j],
                                    sizeof v) == 0);
            } else {
                REQUIRE(e == 0.0F);
                REQUIRE(v == 0.0F);
            }
        }
    }
}

TEST_CASE("Lane FFT matches a double DFT for every n up to 1024",
          "[fft][lane]") {
    // One profile (two rows) per size; a root table keeps the O(n^2)
    // reference cheap. Covers every radix mix and Bluestein size.
    std::mt19937 rng(17);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    SizeType nfail = 0;
    for (SizeType n = 2; n <= 1024; ++n) {
        const SizeType nf = (n / 2) + 1;
        std::vector<DComplex> roots(n);
        for (SizeType t = 0; t < n; ++t) {
            roots[t] = root(t, n, 1.0);
        }
        std::vector<ComplexType> spectra(2 * nf);
        for (auto& c : spectra) {
            c = {dist(rng), dist(rng)};
        }
        std::vector<float> samples(2 * n);
        for (auto& x : samples) {
            x = dist(rng);
        }
        std::vector<float> inv(2 * n);
        std::vector<ComplexType> fwd(2 * nf);
        lane_fft::inverse_batch(spectra.data(), inv.data(), 2, n, 1);
        lane_fft::forward_batch(samples.data(), fwd.data(), 2, n, 1);
        const double tol = 1e-6 * std::log2(static_cast<double>(2 * n));
        double err_inv   = 0.0;
        double err_fwd   = 0.0;
        for (SizeType row = 0; row < 2; ++row) {
            const ComplexType* x = spectra.data() + (row * nf);
            double scale         = 0.0;
            double err           = 0.0;
            for (SizeType j = 0; j < n; ++j) {
                DComplex acc = 0.0;
                for (SizeType k = 0; k < n; ++k) {
                    DComplex xk = k <= n / 2 ? DComplex(x[k])
                                             : std::conj(DComplex(x[n - k]));
                    if (k == 0 || 2 * k == n) {
                        xk = xk.real();
                    }
                    acc += xk * roots[(j * k) % n];
                }
                const double ref = acc.real() / static_cast<double>(n);
                scale            = std::max(scale, std::abs(ref));
                err = std::max(err, std::abs(ref - inv[(row * n) + j]));
            }
            err_inv = std::max(err_inv, err / scale);
            scale   = 0.0;
            err     = 0.0;
            for (SizeType k = 0; k < nf; ++k) {
                DComplex acc = 0.0;
                for (SizeType j = 0; j < n; ++j) {
                    acc += static_cast<double>(samples[(row * n) + j]) *
                           std::conj(roots[(j * k) % n]);
                }
                scale = std::max(scale, std::abs(acc));
                err   = std::max(err,
                                 std::abs(acc - DComplex(fwd[(row * nf) + k])));
            }
            err_fwd = std::max(err_fwd, err / scale);
        }
        if (err_inv > tol || err_fwd > tol) {
            ++nfail;
            UNSCOPED_INFO("n=" << n << " inverse " << err_inv << " forward "
                               << err_fwd);
        }
    }
    CHECK(nfail == 0);
}

TEST_CASE("Lane FFT rejects n < 2", "[fft][lane]") {
    CHECK_FALSE(lane_fft::EVTransform::supports(0));
    CHECK_FALSE(lane_fft::EVTransform::supports(1));
    CHECK(lane_fft::EVTransform::supports(2));
    CHECK(lane_fft::EVTransform::supports(1021));
}

TEST_CASE("Boxcar scorer on lane input is bit-exact with row input",
          "[fft][lane][score]") {
    const SizeType n =
        GENERATE(SizeType{16}, SizeType{50}, SizeType{64}, SizeType{256});
    const SizeType nprofiles = GENERATE(SizeType{3}, SizeType{16});
    const SizeType nf        = (n / 2) + 1;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<ComplexType> spectra(nprofiles * 2 * nf);
    for (SizeType p = 0; p < nprofiles; ++p) {
        for (SizeType k = 0; k < nf; ++k) {
            spectra[(p * 2 * nf) + k] = {dist(rng), dist(rng)};
            // A positive variance profile: a large DC term.
            spectra[(p * 2 * nf) + nf + k] =
                k == 0 ? ComplexType(static_cast<float>(4 * n), 0.0F)
                       : ComplexType(dist(rng), dist(rng));
        }
    }
    const std::vector<SizeType> widths{1, 2, 3, 5, 8, n / 4};
    loki::detection::detail::BoxcarScorer3D lanes(n, widths.data(),
                                                  widths.size());
    loki::detection::detail::BoxcarScorer3D rows(n, widths.data(),
                                                 widths.size());
    lane_fft::EVTransform transform(n);
    std::vector<float> folds(nprofiles * 2 * n);
    transform.inverse_rows(spectra.data(), 2 * nprofiles, folds.data());
    std::vector<float> from_rows(nprofiles * widths.size());
    rows.score(folds.data(), nprofiles, [&](SizeType p, SizeType iw, float s) {
        from_rows[(p * widths.size()) + iw] = s;
    });
    transform.inverse(spectra.data(), nprofiles);
    std::vector<float> from_lanes(nprofiles * widths.size(), -1.0F);
    lanes.score_lanes(transform.e_lanes(), transform.v_lanes(), nprofiles,
                      [&](SizeType p, SizeType iw, float s) {
                          from_lanes[(p * widths.size()) + iw] = s;
                      });
    INFO("n=" << n << " nprofiles=" << nprofiles);
    REQUIRE(std::memcmp(from_rows.data(), from_lanes.data(),
                        from_rows.size() * sizeof(float)) == 0);
}
