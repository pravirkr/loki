#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <stdexcept>
#include <vector>

#include "loki/algorithms/ffa.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using Catch::Approx;
using loki::Backend;
using loki::ComplexType;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::compute_ffa;
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
        REQUIRE(!fold_time.empty());
    }

    SECTION("Exec::cuda availability check") {
#ifndef LOKI_ENABLE_CUDA
        REQUIRE_THROWS_AS(FFATime(cfg, false, Exec::cuda(0)),
                          std::invalid_argument);
#endif
    }
}
