#include "loki/algorithms/fold.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"

using Catch::Approx;
using loki::Backend;
using loki::ComplexType;
using loki::Exec;
using loki::SizeType;
using loki::algorithms::BruteFold;
using loki::algorithms::compute_brute_fold;

TEST_CASE("BruteFold CPU executes time and complex domain folding",
          "[fold][cpu]") {
    constexpr SizeType kNsamps      = 1024;
    constexpr SizeType kSegmentLen  = 256;
    constexpr SizeType kNbins       = 32;
    constexpr double kTsamp         = 1.0e-3;
    const std::vector<double> freqs = {2.0, 5.0, 10.0};

    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = static_cast<float>(std::sin(2.0 * std::numbers::pi * 5.0 *
                                              static_cast<double>(i) * kTsamp));
    }

    SECTION("Time domain float") {
        BruteFold<float> bf(freqs, kSegmentLen, kNbins, kNsamps, kTsamp, 0.0,
                            Exec::cpu(1));
        const SizeType fold_size = bf.get_fold_size();
        REQUIRE(fold_size ==
                (kNsamps / kSegmentLen) * freqs.size() * 2 * kNbins);

        std::vector<float> fold(fold_size, 0.0F);
        bf.execute(ts_e, ts_v, fold);

        bool has_nonzero = false;
        for (const float v : fold) {
            if (std::abs(v) > 1e-4F) {
                has_nonzero = true;
                break;
            }
        }
        REQUIRE(has_nonzero);

        auto conv_fold = compute_brute_fold<float>(
            ts_e, ts_v, freqs, kSegmentLen, kNbins, kTsamp, 0.0, Exec::cpu(1));
        REQUIRE(conv_fold.size() == fold_size);
        for (SizeType i = 0; i < fold_size; ++i) {
            REQUIRE(conv_fold[i] == Approx(fold[i]).margin(1e-5F));
        }
    }

    SECTION("Fourier domain ComplexType") {
        BruteFold<ComplexType> bf(freqs, kSegmentLen, kNbins, kNsamps, kTsamp,
                                  0.0, Exec::cpu(1));
        const SizeType fold_size = bf.get_fold_size();
        REQUIRE(fold_size == (kNsamps / kSegmentLen) * freqs.size() * 2 *
                                 ((kNbins / 2) + 1));

        std::vector<ComplexType> fold(fold_size, ComplexType{0.0F, 0.0F});
        bf.execute(ts_e, ts_v, fold);

        bool has_nonzero = false;
        for (const auto& v : fold) {
            if (std::abs(v.real()) > 1e-4F || std::abs(v.imag()) > 1e-4F) {
                has_nonzero = true;
                break;
            }
        }
        REQUIRE(has_nonzero);
    }
}

TEST_CASE("BruteFold CPU Fourier: any segment length matches a double DFT",
          "[fold][cpu]") {
    // The SIMD fold reads one phasor row per frequency at stride segment_len,
    // so lengths that are not a multiple of the vector width start rows off
    // the vector alignment.
    constexpr double kTsamp  = 64e-6;
    constexpr SizeType kNseg = 3;
    const SizeType seg_len =
        GENERATE(SizeType{33}, SizeType{100}, SizeType{256}, SizeType{300});
    const SizeType nbins   = GENERATE(SizeType{16}, SizeType{33});
    const SizeType nsamps  = kNseg * seg_len;
    const SizeType nbins_f = (nbins / 2) + 1;
    const double t_ref     = 0.5 * static_cast<double>(seg_len) * kTsamp;
    const std::vector<double> freqs = {0.7, 3.1, 47.0, 99.3, 211.9};
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> ts_e(nsamps);
    std::vector<float> ts_v(nsamps);
    for (SizeType i = 0; i < nsamps; ++i) {
        ts_e[i] = nd(rng);
        ts_v[i] = 1.0F + (0.1F * std::abs(nd(rng)));
    }
    BruteFold<ComplexType> bf(freqs, seg_len, nbins, nsamps, kTsamp, t_ref,
                              Exec::cpu(1));
    std::vector<ComplexType> fold(bf.get_fold_size());
    bf.execute(ts_e, ts_v, fold);
    REQUIRE(fold.size() == kNseg * freqs.size() * 2 * nbins_f);
    for (SizeType iseg = 0; iseg < kNseg; ++iseg) {
        for (SizeType ifreq = 0; ifreq < freqs.size(); ++ifreq) {
            for (SizeType ch = 0; ch < 2; ++ch) {
                const auto& ts = ch == 0 ? ts_e : ts_v;
                const auto row = ((iseg * freqs.size() + ifreq) * 2) + ch;
                double scale   = 0.0;
                for (SizeType i = 0; i < seg_len; ++i) {
                    scale += std::abs(ts[(iseg * seg_len) + i]);
                }
                for (SizeType k = 0; k < nbins_f; ++k) {
                    double re = 0.0;
                    double im = 0.0;
                    for (SizeType i = 0; i < seg_len; ++i) {
                        const double t =
                            (static_cast<double>(i) * kTsamp) - t_ref;
                        const double ph = -2.0 * std::numbers::pi *
                                          freqs[ifreq] *
                                          static_cast<double>(k) * t;
                        const double x  = ts[(iseg * seg_len) + i];
                        re += x * std::cos(ph);
                        im += x * std::sin(ph);
                    }
                    const auto got = fold[(row * nbins_f) + k];
                    REQUIRE(got.real() == Approx(re).margin(1e-5 * scale));
                    REQUIRE(got.imag() == Approx(im).margin(1e-5 * scale));
                }
            }
        }
    }
}

TEST_CASE("BruteFold CUDA parity with CPU", "[fold][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }

    constexpr SizeType kNsamps      = 1024;
    constexpr SizeType kSegmentLen  = 256;
    constexpr SizeType kNbins       = 32;
    constexpr double kTsamp         = 1.0e-3;
    const std::vector<double> freqs = {2.0, 5.0, 10.0};

    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = static_cast<float>(std::sin(2.0 * std::numbers::pi * 5.0 *
                                              static_cast<double>(i) * kTsamp));
    }

    SECTION("Time domain parity") {
        BruteFold<float> bf_cpu(freqs, kSegmentLen, kNbins, kNsamps, kTsamp,
                                0.0, Exec::cpu(1));
        BruteFold<float> bf_cuda(freqs, kSegmentLen, kNbins, kNsamps, kTsamp,
                                 0.0, Exec::cuda(0));

        std::vector<float> fold_cpu(bf_cpu.get_fold_size(), 0.0F);
        std::vector<float> fold_cuda(bf_cuda.get_fold_size(), 0.0F);

        bf_cpu.execute(ts_e, ts_v, fold_cpu);
        bf_cuda.execute(ts_e, ts_v, fold_cuda);

        REQUIRE(fold_cpu.size() == fold_cuda.size());
        for (SizeType i = 0; i < fold_cpu.size(); ++i) {
            REQUIRE(fold_cuda[i] == Approx(fold_cpu[i]).margin(1e-4F));
        }
    }

    SECTION("Fourier domain parity") {
        BruteFold<ComplexType> bf_cpu(freqs, kSegmentLen, kNbins, kNsamps,
                                      kTsamp, 0.0, Exec::cpu(1));
        BruteFold<ComplexType> bf_cuda(freqs, kSegmentLen, kNbins, kNsamps,
                                       kTsamp, 0.0, Exec::cuda(0));

        std::vector<ComplexType> fold_cpu(bf_cpu.get_fold_size(),
                                          ComplexType{0.0F, 0.0F});
        std::vector<ComplexType> fold_cuda(bf_cuda.get_fold_size(),
                                           ComplexType{0.0F, 0.0F});

        bf_cpu.execute(ts_e, ts_v, fold_cpu);
        bf_cuda.execute(ts_e, ts_v, fold_cuda);

        REQUIRE(fold_cpu.size() == fold_cuda.size());
        // CUDA sums the DFT terms directly, so float32 rounding scales with
        // the largest harmonic rather than with each element.
        float peak = 0.0F;
        for (const auto& x : fold_cpu) {
            peak = std::max(peak, std::abs(x));
        }
        const float tol = 1e-4F * peak;
        for (SizeType i = 0; i < fold_cpu.size(); ++i) {
            REQUIRE(fold_cuda[i].real() ==
                    Approx(fold_cpu[i].real()).margin(tol));
            REQUIRE(fold_cuda[i].imag() ==
                    Approx(fold_cpu[i].imag()).margin(tol));
        }
    }
}

TEST_CASE("BruteFold CUDA Fourier parity: reference time and long segments",
          "[fold][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    // A sweep folds about the segment centre (t_ref != 0), and long periods
    // give segments longer than one shared-memory tile (2048 samples).
    constexpr double kTsamp = 64e-6;
    const SizeType seg_len  = GENERATE(SizeType{512}, SizeType{8192});
    const SizeType nbins    = GENERATE(SizeType{16}, SizeType{64});
    const SizeType nsamps   = 4 * seg_len;
    const double t_ref      = 0.5 * static_cast<double>(seg_len) * kTsamp;
    const std::vector<double> freqs = {0.7, 3.1, 47.0, 99.3};
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> ts_e(nsamps);
    std::vector<float> ts_v(nsamps);
    for (SizeType i = 0; i < nsamps; ++i) {
        ts_e[i] = nd(rng);
        ts_v[i] = 1.0F + (0.1F * std::abs(nd(rng)));
    }
    BruteFold<ComplexType> bf_cpu(freqs, seg_len, nbins, nsamps, kTsamp, t_ref,
                                  Exec::cpu(1));
    BruteFold<ComplexType> bf_cuda(freqs, seg_len, nbins, nsamps, kTsamp, t_ref,
                                   Exec::cuda(0));
    std::vector<ComplexType> fold_cpu(bf_cpu.get_fold_size());
    std::vector<ComplexType> fold_cuda(bf_cuda.get_fold_size());
    bf_cpu.execute(ts_e, ts_v, fold_cpu);
    bf_cuda.execute(ts_e, ts_v, fold_cuda);
    REQUIRE(fold_cpu.size() == fold_cuda.size());
    // Both reach float32 rounding of sums of seg_len terms: within a few
    // 1e-6 of the row's largest term (the DC of the variance row).
    const SizeType nbins_f = (nbins / 2) + 1;
    for (SizeType row = 0; row < fold_cpu.size() / nbins_f; ++row) {
        float peak = 0.0F;
        for (SizeType k = 0; k < nbins_f; ++k) {
            peak = std::max(peak, std::abs(fold_cpu[(row * nbins_f) + k]));
        }
        const float tol = 2e-5F * peak;
        for (SizeType k = 0; k < nbins_f; ++k) {
            const auto a = fold_cpu[(row * nbins_f) + k];
            const auto b = fold_cuda[(row * nbins_f) + k];
            REQUIRE(b.real() == Approx(a.real()).margin(tol));
            REQUIRE(b.imag() == Approx(a.imag()).margin(tol));
        }
    }
}
