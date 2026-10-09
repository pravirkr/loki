#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/core/dynamic.hpp"
#include "lib/utils/fft_impl.hpp"

namespace {

using loki::ComplexType;
using loki::ParamLimit;
using loki::SizeType;
using loki::math::FFTWManager;
using loki::search::PulsarSearchConfig;

PulsarSearchConfig make_cfg(bool use_fourier) {
    constexpr SizeType kNsamps = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 145.0},
    };
    return {kNsamps,
            64e-6,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            use_fourier,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*bseg_ffa=*/kNsamps / 8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

template <typename FoldType>
std::unique_ptr<loki::core::PruneDPFuncts<FoldType>>
make_functs(const PulsarSearchConfig& cfg, FFTWManager* fft) {
    const loki::plans::FFAPlan<FoldType> plan(cfg);
    return loki::core::create_prune_dp_functs<FoldType>(
        "taylor", plan.get_param_counts().back(),
        plan.get_dparams_actual().back(), plan.get_nsegments().back(),
        plan.get_tsegments().back(), cfg, /*batch_size=*/64,
        /*branch_max=*/16, fft);
}

} // namespace

TEST_CASE("Prune functors reuse the FFT plans of an external manager",
          "[ep_freq_sweep][fft]") {
    const auto cfg = make_cfg(/*use_fourier=*/true);
    FFTWManager shared;
    const std::vector<SizeType> n_reals{cfg.get_nbins()};
    shared.prepare_exact_plans(n_reals);

    const auto functs = make_functs<ComplexType>(cfg, &shared);
    REQUIRE(functs != nullptr);
    CHECK(shared.has_prepared(cfg.get_nbins()));
    // Building the functor creates no plan: they are made on first use, in
    // the manager the sweep owns.
    CHECK(shared.n_cached_plans() == 0);
}

TEST_CASE("Prune functors refuse an external manager without their plans",
          "[ep_freq_sweep][fft]") {
    const auto cfg = make_cfg(/*use_fourier=*/true);
    FFTWManager unprepared;
    CHECK_THROWS(make_functs<ComplexType>(cfg, &unprepared));
}

TEST_CASE("Prune functors without an external manager keep their own",
          "[ep_freq_sweep][fft]") {
    REQUIRE_NOTHROW(make_functs<ComplexType>(make_cfg(true), nullptr));
    // Time-domain folds need no FFT manager, so the pointer is ignored.
    REQUIRE_NOTHROW(make_functs<float>(make_cfg(false), nullptr));
}
