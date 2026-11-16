// An interrupt request (SIGINT/SIGTERM under the CLI's guard) stops an EP or
// FFA sweep and its candidate postprocessing at a checkpoint and leaves no
// staging, scratch or partial files behind.

#include <algorithm>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/backend.hpp"
#include "loki/common/interrupt.hpp"
#include "loki/common/types.hpp"
#include "loki/io/observation.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"

#include "temp_dir.hpp"

using loki::Backend;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::EPFreqSweep;
using loki::pipelines::FFAFreqSweep;
using loki::search::CandidateOptions;
using loki::search::PulsarSearchConfig;
using loki::test::TempDir;

namespace {

PulsarSearchConfig make_cfg() {
    constexpr SizeType kNsamps = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 142.0},
    };
    return {kNsamps,
            64e-6,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            /*nthreads=*/2,
            /*max_process_memory_gb=*/4.0,
            /*region_growth_factor=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*nsegments=*/8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

std::pair<std::vector<float>, std::vector<float>> noise(SizeType n) {
    std::vector<float> ts_e(n);
    std::vector<float> ts_v(n, 1.0F);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::ranges::generate(ts_e, [&] { return dist(rng); });
    return {std::move(ts_e), std::move(ts_v)};
}

CandidateOptions small_options() {
    CandidateOptions options;
    options.memory_gb = 0.05;
    return options;
}

/// Entries of @p dir other than @p keep.
std::vector<std::string> leftovers(const std::filesystem::path& dir,
                                   const std::vector<std::string>& keep) {
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (std::ranges::find(keep, name) == keep.end()) {
            names.push_back(name);
        }
    }
    return names;
}

/// Clears the request even when a check fails.
struct ClearInterrupt {
    ClearInterrupt() { loki::clear_interrupt(); }
    ~ClearInterrupt() { loki::clear_interrupt(); }
    ClearInterrupt(const ClearInterrupt&)            = delete;
    ClearInterrupt& operator=(const ClearInterrupt&) = delete;
    ClearInterrupt(ClearInterrupt&&)                 = delete;
    ClearInterrupt& operator=(ClearInterrupt&&)      = delete;
};

void check_sweep_interrupt(Exec exec) {
    const ClearInterrupt clear;
    const TempDir outdir("loki-ep-interrupt");
    const auto cfg          = make_cfg();
    const auto [ts_e, ts_v] = noise(cfg.get_nsamps());
    const auto plan         = outdir / "plan.h5";
    const std::vector<SizeType> ref_segs{2U, 4U};

    SECTION("diagnostic-only sweep") {
        EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, plan,
                          std::nullopt, ref_segs, exec);
        loki::request_interrupt();
        CHECK_THROWS_AS(sweep.execute(ts_e, ts_v, outdir.path(), "run"),
                        loki::Interrupted);
        loki::clear_interrupt();
        CHECK(leftovers(outdir.path(), {"plan.h5"}).empty());
    }
    SECTION("candidate-enabled sweep") {
        EPFreqSweep sweep(cfg, small_options(), false, 0.1F, "taylor", 0.1F, {},
                          plan, std::nullopt, ref_segs, exec);
        loki::request_interrupt();
        CHECK_THROWS_AS(sweep.execute(ts_e, ts_v, outdir.path(), "run",
                                      loki::io::ObservationMetadata{}),
                        loki::Interrupted);
        loki::clear_interrupt();
        CHECK(leftovers(outdir.path(), {"plan.h5"}).empty());
    }
}

} // namespace

TEST_CASE("an interrupt stops a CPU EP sweep without leftovers",
          "[interrupt][ep_freq_sweep]") {
    check_sweep_interrupt(Exec::cpu());
}

TEST_CASE("an interrupt stops a CUDA EP sweep without leftovers",
          "[interrupt][ep_freq_sweep][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    check_sweep_interrupt(Exec::cuda(0));
}

TEST_CASE("an interrupt stops EP candidate publication without leftovers",
          "[interrupt][ep_freq_sweep]") {
    const ClearInterrupt clear;
    const TempDir outdir("loki-ep-publish-interrupt");
    const auto cfg          = make_cfg();
    const auto [ts_e, ts_v] = noise(cfg.get_nsamps());
    const std::vector<SizeType> ref_segs{2U, 4U};
    // A diagnostic-only sweep writes the spool publication starts from.
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, std::nullopt,
                      std::nullopt, ref_segs);
    sweep.execute(ts_e, ts_v, outdir.path(), "spool");
    const auto spool  = outdir / "spool_ep_results.h5";
    const auto output = outdir / "out_ep_results.h5";
    REQUIRE(std::filesystem::exists(spool));

    const auto scratch          = outdir / "scratch";
    auto options                = small_options();
    options.temporary_directory = scratch;
    loki::request_interrupt();
    CHECK_THROWS_AS(EPFreqSweep::publish_candidates(
                        spool, output, options, loki::io::ObservationMetadata{},
                        /*snr_min=*/0.0),
                    loki::Interrupted);
    loki::clear_interrupt();
    CHECK_FALSE(std::filesystem::exists(output));
    CHECK(leftovers(outdir.path(), {"spool_ep_results.h5", "scratch"}).empty());
    if (std::filesystem::exists(scratch)) {
        CHECK(leftovers(scratch, {}).empty());
    }
    // The spool is untouched: publication succeeds once the request is gone.
    REQUIRE_NOTHROW(EPFreqSweep::publish_candidates(
        spool, output, options, loki::io::ObservationMetadata{}, 0.0));
    CHECK(std::filesystem::exists(output));
}

namespace {

void check_ffa_interrupt(Exec exec) {
    const ClearInterrupt clear;
    const TempDir outdir("loki-ffa-interrupt");
    constexpr SizeType kNsamps = 1U << 14U;
    const std::vector<ParamLimit> limits{{50.0, 200.0}};
    const PulsarSearchConfig cfg(
        kNsamps, 6.4e-5, /*nbins=*/64, /*eta=*/1.0, limits, /*ducy_max=*/0.2,
        /*wtsp=*/1.5, /*use_fourier=*/true, /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0, /*region_growth_factor=*/2.0,
        /*nbins_max=*/128, /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/std::nullopt, /*nsegments=*/std::nullopt,
        /*snr_min=*/3.0, /*max_passing_candidates=*/1U << 20U);
    const auto [ts_e, ts_v] = noise(kNsamps);

    SECTION("flat detection file") {
        CandidateOptions off;
        off.enabled = false;
        FFAFreqSweep sweep(cfg, off, false, exec);
        loki::request_interrupt();
        CHECK_THROWS_AS(sweep.execute(ts_e, ts_v, outdir.path(), "run"),
                        loki::Interrupted);
        loki::clear_interrupt();
        CHECK(leftovers(outdir.path(), {}).empty());
    }
    SECTION("candidate-enabled sweep") {
        FFAFreqSweep sweep(cfg, small_options(), false, exec);
        loki::request_interrupt();
        CHECK_THROWS_AS(sweep.execute(ts_e, ts_v, outdir.path(), "run"),
                        loki::Interrupted);
        loki::clear_interrupt();
        CHECK(leftovers(outdir.path(), {}).empty());
    }
}

} // namespace

TEST_CASE("an interrupt stops a CPU FFA sweep without leftovers",
          "[interrupt][ffa_sweep]") {
    check_ffa_interrupt(Exec::cpu());
}

TEST_CASE("an interrupt stops a CUDA FFA sweep without leftovers",
          "[interrupt][ffa_sweep][cuda]") {
    if (!loki::is_available(Backend::kCUDA)) {
        SKIP("needs a CUDA build");
    }
    check_ffa_interrupt(Exec::cuda(0));
}
