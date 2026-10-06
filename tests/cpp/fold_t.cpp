#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "loki/algorithms/fold.hpp"
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
        ts_e[i] = std::sin(2.0 * M_PI * 5.0 * static_cast<double>(i) * kTsamp);
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
        for (float v : fold) {
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

#ifdef LOKI_ENABLE_CUDA
TEST_CASE("BruteFold CUDA parity with CPU and backward compatibility",
          "[fold][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        return;
    }

    constexpr SizeType kNsamps      = 1024;
    constexpr SizeType kSegmentLen  = 256;
    constexpr SizeType kNbins       = 32;
    constexpr double kTsamp         = 1.0e-3;
    const std::vector<double> freqs = {2.0, 5.0, 10.0};

    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = std::sin(2.0 * M_PI * 5.0 * static_cast<double>(i) * kTsamp);
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

        // Backward compatibility class BruteFoldFloatCUDA
        loki::algorithms::BruteFoldFloatCUDA bf_compat(
            freqs, kSegmentLen, kNbins, kNsamps, kTsamp, 0.0, 0);
        std::vector<float> fold_compat(bf_compat.get_fold_size(), 0.0F);
        bf_compat.execute(ts_e, ts_v, fold_compat);
        for (SizeType i = 0; i < fold_cpu.size(); ++i) {
            REQUIRE(fold_compat[i] == Approx(fold_cpu[i]).margin(1e-4F));
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
        for (SizeType i = 0; i < fold_cpu.size(); ++i) {
            REQUIRE(fold_cuda[i].real() ==
                    Approx(fold_cpu[i].real()).margin(1e-3F));
            REQUIRE(fold_cuda[i].imag() ==
                    Approx(fold_cpu[i].imag()).margin(1e-3F));
        }
    }
}
#endif
