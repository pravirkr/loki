#include <ios>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

using Catch::Approx;
using loki::Backend;
using loki::ComplexType;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::compute_ffa;
using loki::algorithms::compute_ffa_scores;
using loki::algorithms::FFA;
using loki::algorithms::FFAFourier;
using loki::algorithms::FFATime;
using loki::search::PulsarSearchConfig;

TEST_CASE("FFA backend dispatch CPU", "[ffa][backend]") {
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
    }

    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 10.0}};
    const PulsarSearchConfig cfg(
        kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.5, limits,
        /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/1,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, /*bseg_brute=*/128);

    SECTION("FFATime with Exec::cpu") {
        FFATime ffa(cfg, /*show_progress=*/false, Exec::cpu(1));
        std::vector<float> fold(ffa.get_plan().get_buffer_size(), 0.0F);
        ffa.execute(ts_e, ts_v, fold);
        fold.resize(ffa.get_plan().get_fold_size());
        REQUIRE(!fold.empty());

        auto [conv_fold, plan] =
            compute_ffa<float>(ts_e, ts_v, cfg, /*quiet=*/true,
                               /*show_progress=*/false, Exec::cpu(1));
        REQUIRE(conv_fold.size() == fold.size());
        for (SizeType i = 0; i < fold.size(); ++i) {
            REQUIRE(conv_fold[i] == Approx(fold[i]).margin(1e-5F));
        }
    }

    SECTION("FFAFourier with Exec::cpu") {
        const PulsarSearchConfig fourier_cfg(
            kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.5, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/true,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
            /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, /*bseg_brute=*/128);
        FFAFourier ffa(fourier_cfg, /*show_progress=*/false, Exec::cpu(1));
        std::vector<ComplexType> fold(ffa.get_plan().get_buffer_size());
        ffa.execute(ts_e, ts_v, fold);
        fold.resize(ffa.get_plan().get_fold_size());
        REQUIRE(!fold.empty());

        // execute returning to time
        std::vector<float> fold_time(2 * ffa.get_plan().get_buffer_size());
        ffa.execute(ts_e, ts_v, fold_time);
        fold_time.resize(ffa.get_plan().get_fold_size_time());
        REQUIRE(!fold_time.empty());

        auto [ref_fold_time, ref_plan_time] =
            loki::algorithms::compute_ffa_fourier_return_to_time(
                ts_e, ts_v, fourier_cfg, /*quiet=*/true,
                /*show_progress=*/false, Exec::cpu(1));
        REQUIRE(fold_time.size() == ref_fold_time.size());
        for (SizeType i = 0; i < fold_time.size(); ++i) {
            REQUIRE(fold_time[i] == ref_fold_time[i]);
        }

        // Lossy Fourier (nbins > nbins_min_lossy_bf) starts from the exact same
        // time-domain brute fold as FFATime; verify return-to-time matches both
        // compute_ffa_fourier_return_to_time and FFATime closely.
        const PulsarSearchConfig lossy_cfg(
            kNsamps, kTsamp, /*nbins=*/64, /*eta=*/0.5, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/true,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
            /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, /*bseg_brute=*/128);
        FFAFourier ffa_lossy(lossy_cfg, /*show_progress=*/false, Exec::cpu(1));
        std::vector<float> lossy_time(2 *
                                      ffa_lossy.get_plan().get_buffer_size());
        ffa_lossy.execute(ts_e, ts_v, lossy_time);
        lossy_time.resize(ffa_lossy.get_plan().get_fold_size_time());
        auto [lossy_ref, lossy_plan] =
            loki::algorithms::compute_ffa_fourier_return_to_time(
                ts_e, ts_v, lossy_cfg, /*quiet=*/true,
                /*show_progress=*/false, Exec::cpu(1));
        REQUIRE(lossy_time == lossy_ref);
    }

    SECTION("Exec::cuda availability check") {
        if (loki::is_available(Backend::kCUDA)) {
            REQUIRE_NOTHROW(FFATime(cfg, false, Exec::cuda(0)));
        } else {
            REQUIRE_THROWS_AS(FFATime(cfg, false, Exec::cuda(0)),
                              std::invalid_argument);
        }
    }
}

TEST_CASE("compute_ffa_scores CPU vs CUDA parity", "[ffa][backend][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
    }

    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 10.0}};

    for (const bool use_fourier : {false, true}) {
        DYNAMIC_SECTION("use_fourier = " << std::boolalpha << use_fourier) {
            const PulsarSearchConfig cfg(
                kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.5, limits,
                /*ducy_max=*/0.2, /*wtsp=*/1.5, use_fourier, /*nthreads=*/1,
                /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
                /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32,
                /*bseg_brute=*/128);

            auto [cpu_scores, cpu_plan] =
                compute_ffa_scores(ts_e, ts_v, cfg, /*quiet=*/true,
                                   /*show_progress=*/false, Exec::cpu(1));
            auto [cuda_scores, cuda_plan] =
                compute_ffa_scores(ts_e, ts_v, cfg, /*quiet=*/true,
                                   /*show_progress=*/false, Exec::cuda(0));

            REQUIRE(cpu_scores.size() == cuda_scores.size());
            REQUIRE(!cpu_scores.empty());
            for (SizeType i = 0; i < cpu_scores.size(); ++i) {
                REQUIRE(cuda_scores[i] ==
                        Approx(cpu_scores[i]).margin(1.0e-2F));
            }
        }
    }
}

// nbins where the fused GPU merges change k or fall back to unfused because
// of the shared-memory tile (dynamic plus static), and the float4 path (64).
TEST_CASE("compute_ffa_scores CPU vs CUDA parity at fused-tile edges",
          "[ffa][backend][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto nbins = GENERATE(SizeType{50}, SizeType{64}, SizeType{96},
                                SizeType{192}, SizeType{384});
    CAPTURE(nbins);
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    const std::vector<float> ts_v(kNsamps, 1.0F);
    for (auto& v : ts_e) {
        v = dist(rng);
    }
    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 6.0}};
    const PulsarSearchConfig cfg(
        kNsamps, kTsamp, nbins, /*eta=*/0.5, limits, /*ducy_max=*/0.2,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/1,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, /*bseg_brute=*/32);

    auto [cpu_scores, cpu_plan] =
        compute_ffa_scores(ts_e, ts_v, cfg, /*quiet=*/true,
                           /*show_progress=*/false, Exec::cpu(1));
    auto [cuda_scores, cuda_plan] =
        compute_ffa_scores(ts_e, ts_v, cfg, /*quiet=*/true,
                           /*show_progress=*/false, Exec::cuda(0));
    REQUIRE(cpu_scores.size() == cuda_scores.size());
    REQUIRE(!cpu_scores.empty());
    for (SizeType i = 0; i < cpu_scores.size(); ++i) {
        REQUIRE(cuda_scores[i] == Approx(cpu_scores[i]).margin(1.0e-2F));
    }
}

TEST_CASE("snr_boxcar_2d and snr_boxcar_2d_max CPU vs CUDA parity",
          "[score][backend][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    constexpr SizeType kNprofiles = 64;
    constexpr SizeType kNbins     = 128;
    std::vector<float> folds(kNprofiles * kNbins);
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& val : folds) {
        val = dist(rng);
    }
    const std::vector<SizeType> widths = {1, 2, 4, 8, 16};

    // 2D per-width
    std::vector<float> scores_cpu(kNprofiles * widths.size());
    std::vector<float> scores_cuda(kNprofiles * widths.size());
    loki::detection::snr_boxcar_2d(folds, widths, scores_cpu, kNprofiles,
                                   kNbins, 1.0F, Exec::cpu(1));
    loki::detection::snr_boxcar_2d(folds, widths, scores_cuda, kNprofiles,
                                   kNbins, 1.0F, Exec::cuda(0));

    REQUIRE(scores_cpu.size() == scores_cuda.size());
    for (SizeType i = 0; i < scores_cpu.size(); ++i) {
        REQUIRE(scores_cuda[i] == Approx(scores_cpu[i]).margin(1.0e-4F));
    }

    // 2D max
    std::vector<float> max_cpu(kNprofiles);
    std::vector<float> max_cuda(kNprofiles);
    loki::detection::snr_boxcar_2d_max(folds, widths, max_cpu, kNprofiles,
                                       kNbins, 1.0F, Exec::cpu(1));
    loki::detection::snr_boxcar_2d_max(folds, widths, max_cuda, kNprofiles,
                                       kNbins, 1.0F, Exec::cuda(0));

    REQUIRE(max_cpu.size() == max_cuda.size());
    for (SizeType i = 0; i < max_cpu.size(); ++i) {
        REQUIRE(max_cuda[i] == Approx(max_cpu[i]).margin(1.0e-4F));
    }
}

TEST_CASE("FFA non-power-of-2 nsamps matches zero-padded reference and CUDA",
          "[ffa][backend][nsamps]") {
    constexpr SizeType kNsamps       = 10000;
    constexpr SizeType kNsampsPadded = 16384;
    constexpr SizeType kBsegBrute    = 128;
    constexpr double kTsamp          = 1.0e-3;

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
    }
    std::vector<float> ts_e_pad(kNsampsPadded, 0.0F);
    std::vector<float> ts_v_pad(kNsampsPadded, 0.0F);
    std::copy(ts_e.begin(), ts_e.end(), ts_e_pad.begin());
    std::copy(ts_v.begin(), ts_v.end(), ts_v_pad.begin());

    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 8.0}};

    for (const bool use_fourier : {false, true}) {
        for (const SizeType nbins : {SizeType{32}, SizeType{64}}) {
            const loki::search::FFASearchConfig cfg_np2(
                kNsamps, kTsamp, nbins, /*eta=*/0.5, limits,
                /*ducy_max=*/0.2, /*wtsp=*/1.5, use_fourier, /*nthreads=*/2,
                /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
                /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, kBsegBrute);
            const loki::search::FFASearchConfig cfg_pad(
                kNsampsPadded, kTsamp, nbins, /*eta=*/0.5, limits,
                /*ducy_max=*/0.2, /*wtsp=*/1.5, use_fourier, /*nthreads=*/2,
                /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
                /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, kBsegBrute);

            auto [scores_np2, plan_np2] =
                compute_ffa_scores(ts_e, ts_v, cfg_np2, /*quiet=*/true,
                                   /*show_progress=*/false, Exec::cpu(2));
            auto [scores_pad, plan_pad] =
                compute_ffa_scores(ts_e_pad, ts_v_pad, cfg_pad, /*quiet=*/true,
                                   /*show_progress=*/false, Exec::cpu(2));

            REQUIRE(scores_np2.size() == scores_pad.size());
            for (SizeType i = 0; i < scores_np2.size(); ++i) {
                REQUIRE(scores_np2[i] == Approx(scores_pad[i]).margin(1.0e-5F));
            }

            if (loki::is_available(Backend::kCUDA)) {
                auto [scores_cuda, plan_cuda] =
                    compute_ffa_scores(ts_e, ts_v, cfg_np2, /*quiet=*/true,
                                       /*show_progress=*/false, Exec::cuda(0));
                REQUIRE(scores_cuda.size() == scores_np2.size());
                for (SizeType i = 0; i < scores_np2.size(); ++i) {
                    REQUIRE(scores_cuda[i] ==
                            Approx(scores_np2[i]).margin(1.0e-2F));
                }
            }
        }
    }
}
