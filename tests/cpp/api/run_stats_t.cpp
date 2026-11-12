// /run_stats: measured stage timings and counts derived from a results file.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/io/observation.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"
#include "loki/search/provenance.hpp"

#include "temp_dir.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::pipelines::EPFreqSweep;
using loki::search::PulsarSearchConfig;
using loki::test::TempDir;

TEST_CASE("write_run_stats records timings and EP run counts",
          "[provenance][ep_freq_sweep]") {
    const TempDir outdir("loki-run-stats");
    constexpr SizeType kNsamps = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 142.0},
    };
    const PulsarSearchConfig cfg(
        kNsamps, 64e-6, /*nbins=*/32, /*eta=*/1.0, limits, /*ducy_max=*/0.3,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64, /*bseg_brute=*/1024,
        /*nsegments=*/8, /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U, /*prune_poly_order=*/2);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    // Noise plus a 141 Hz pulse (duty 0.1), so the final leaves pass snr_min.
    for (SizeType i = 0; i < kNsamps; ++i) {
        const double phase = 141.0 * static_cast<double>(i) * 64e-6;
        ts_e[i] = noise(rng) + (phase - std::floor(phase) < 0.1 ? 0.3F : 0.0F);
    }
    const std::vector<SizeType> ref_segs{2U, 4U};
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, std::nullopt,
                      std::nullopt, ref_segs);
    sweep.execute(ts_e, ts_v, outdir.path(), "diag");
    const auto diag = outdir / "diag_ep_results.h5";

    const loki::search::RunStats stats{
        .timing_s    = {{"load", 1.5}, {"search", 2.0}, {"total", 4.0}},
        .peak_rss_gb = 0.25,
    };
    loki::search::write_run_stats(diag, stats);
    // Replacing an existing group is allowed (a re-run of the CLI step).
    loki::search::write_run_stats(diag, stats);
    {
        const HighFive::File h5(diag.string(), HighFive::File::ReadOnly);
        REQUIRE(h5.exist("run_stats"));
        const auto g = h5.getGroup("run_stats");
        CHECK(g.getAttribute("timing_load_s").read<double>() == 1.5);
        CHECK(g.getAttribute("timing_total_s").read<double>() == 4.0);
        CHECK(g.getAttribute("peak_rss_gb").read<double>() == 0.25);
        CHECK(g.getAttribute("n_runs").read<std::uint64_t>() == 2);
        CHECK(g.getAttribute("n_runs_overloaded").read<std::uint64_t>() == 0);
        CHECK(g.getAttribute("n_overloads").read<std::uint64_t>() == 0);
        CHECK(g.getAttribute("plan_peak_memory_gb").read<double>() > 0.0);
        CHECK_FALSE(g.exist("detections_per_nbins"));
    }

    // A common product: the per-nbins counts add up to the detections.
    loki::search::CandidateOptions options;
    options.memory_gb  = 0.05;
    const auto product = outdir / "product_ep_results.h5";
    EPFreqSweep::publish_candidates(diag, product, options,
                                    loki::io::ObservationMetadata{}, 0.0);
    loki::search::write_run_stats(product, {});
    const HighFive::File h5(product.string(), HighFive::File::ReadOnly);
    const auto g = h5.getGroup("run_stats");
    CHECK_FALSE(g.hasAttribute("peak_rss_gb"));
    CHECK(g.getAttribute("n_runs").read<std::uint64_t>() == 2);
    const auto counts = g.getDataSet("detections_per_nbins/count")
                            .read<std::vector<std::uint64_t>>();
    const auto n_detections =
        h5.getAttribute("n_detections").read<std::uint64_t>();
    REQUIRE(n_detections > 0);
    CHECK(std::accumulate(counts.begin(), counts.end(), std::uint64_t{0}) ==
          n_detections);
    CHECK(g.getDataSet("detections_per_nbins/nbins")
              .read<std::vector<std::uint32_t>>() ==
          std::vector<std::uint32_t>{32});
}
