#include <catch2/catch_test_macros.hpp>

#include <random>
#include <stdexcept>
#include <vector>

#include "loki/algorithms/ffa.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/fft.hpp"
#include "loki/utils/workspace.hpp"

using loki::Backend;
using loki::ComplexType;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::FFA;
using loki::math::FFTManager;
using loki::memory::EPWorkspace;
using loki::memory::FFAWorkspace;
using loki::search::PulsarSearchConfig;

namespace {

constexpr SizeType kNsamps = 1U << 14U;
constexpr double kTsamp    = 1.0e-3;

PulsarSearchConfig make_cfg(bool use_fourier) {
    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 10.0}};
    return {kNsamps,
            kTsamp,
            /*nbins=*/32,
            /*eta=*/0.5,
            limits,
            /*ducy_max=*/0.2,
            /*wtsp=*/1.5,
            use_fourier,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/32,
            /*bseg_brute=*/128};
}

template <typename FoldType> void check_shared_matches_owned(bool use_fourier) {
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    const std::vector<float> ts_v(kNsamps, 1.0F);
    for (auto& x : ts_e) {
        x = dist(rng);
    }
    const auto cfg = make_cfg(use_fourier);

    FFA<FoldType> owned(cfg, /*show_progress=*/false);
    std::vector<FoldType> fold_owned(owned.get_plan().get_buffer_size());
    owned.execute(ts_e, ts_v, fold_owned);

    FFAWorkspace<FoldType> workspace(owned.get_plan(), Exec::cpu());
    FFTManager fft_manager(Exec::cpu());
    // Two instances on the same buffers: the second must not see stale state.
    for (int run = 0; run < 2; ++run) {
        FFA<FoldType> shared(workspace, fft_manager, cfg,
                             /*show_progress=*/false);
        std::vector<FoldType> fold_shared(shared.get_plan().get_buffer_size());
        shared.execute(ts_e, ts_v, fold_shared);
        const auto fold_size = shared.get_plan().get_fold_size();
        for (SizeType i = 0; i < fold_size; ++i) {
            REQUIRE(fold_shared[i] == fold_owned[i]);
        }
    }
}

} // namespace

TEST_CASE("FFA on a shared workspace matches an owning FFA",
          "[workspace][ffa]") {
    SECTION("time domain") { check_shared_matches_owned<float>(false); }
    SECTION("Fourier domain") { check_shared_matches_owned<ComplexType>(true); }
}

TEST_CASE("Workspace and FFT handles report their backend",
          "[workspace][fft]") {
    const auto cfg = make_cfg(false);
    const FFA<float> ffa(cfg, /*show_progress=*/false);

    const FFAWorkspace<float> ws(ffa.get_plan(), Exec::cpu());
    REQUIRE_FALSE(ws.empty());
    REQUIRE(ws.exec().backend == Backend::kCPU);

    const EPWorkspace<float> ep_ws(/*batch_size=*/16, /*branch_max=*/4,
                                   /*max_sugg=*/64, /*ncoords_ffa=*/8,
                                   /*nparams=*/2, /*nbins=*/32,
                                   /*nsegments=*/4, Exec::cpu());
    REQUIRE(ep_ws.exec().backend == Backend::kCPU);
    REQUIRE(ep_ws.get_memory_usage_gib() > 0.0F);

    FFTManager fft(Exec::cpu());
    const std::vector<SizeType> n_reals = {32};
    REQUIRE_FALSE(fft.has_prepared(32));
    fft.prepare_plans(n_reals);
    REQUIRE(fft.has_prepared(32));
    REQUIRE(fft.n_cached_plans() > 0);
}

TEST_CASE("Empty handles are rejected", "[workspace][ffa]") {
    const auto cfg = make_cfg(false);
    FFAWorkspace<float> ws;
    FFTManager fft;
    REQUIRE(ws.empty());
    REQUIRE(fft.empty());
    REQUIRE_FALSE(fft.has_prepared(32));
    REQUIRE_THROWS_AS(FFA<float>(ws, fft, cfg, /*show_progress=*/false),
                      std::invalid_argument);
}

TEST_CASE("GPU handles are rejected by a CPU-only build", "[workspace][fft]") {
    if (loki::is_available(Backend::kCUDA)) {
        SKIP("CUDA backend is built");
    }
    REQUIRE_THROWS_AS(FFTManager(Exec::cuda(0)), std::invalid_argument);
    REQUIRE_THROWS_AS(FFAWorkspace<float>(1024, 64, 4, 1, Exec::cuda(0)),
                      std::invalid_argument);
}

TEST_CASE("A workspace on another backend than the FFA is rejected",
          "[workspace][ffa]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    const auto cfg = make_cfg(false);
    const FFA<float> owned(cfg, /*show_progress=*/false);
    FFAWorkspace<float> ws(owned.get_plan(), Exec::cpu());
    FFTManager fft(Exec::cpu());
    REQUIRE_THROWS_AS(FFA<float>(ws, fft, cfg, Exec::cuda(0)),
                      std::invalid_argument);
}
