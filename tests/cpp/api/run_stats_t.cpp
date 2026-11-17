// /run_stats: measured stage timings and counts derived from a results file.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <hdf5.h>
#include <highfive/highfive.hpp>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

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
        /*max_process_memory_gb=*/4.0, /*region_growth_factor=*/2.0,
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

    // Overloaded runs are summarised in one warning, per chunk, with the
    // usual remedies (the file itself is unchanged apart from /run_stats).
    {
        const HighFive::File h5(diag.string(), HighFive::File::ReadWrite);
        const auto chunk    = h5.getGroup("chunks").getGroup("chunk_0000");
        const auto runs     = chunk.getGroup("runs");
        const auto stats_ds = runs.getGroup(runs.listObjectNames().front())
                                  .getDataSet("level_stats");
        const auto n_levels = stats_ds.getSpace().getElementCount();
        // Write only the n_overloads member: 1 at every level.
        const hid_t member = H5Tcreate(H5T_COMPOUND, sizeof(std::uint64_t));
        REQUIRE(member >= 0);
        REQUIRE(H5Tinsert(member, "n_overloads", 0, H5T_NATIVE_UINT64) >= 0);
        const std::vector<std::uint64_t> ones(n_levels, 1U);
        const auto status = H5Dwrite(stats_ds.getId(), member, H5S_ALL, H5S_ALL,
                                     H5P_DEFAULT, ones.data());
        H5Tclose(member);
        REQUIRE(status >= 0);
    }
    std::ostringstream log;
    const auto logger = spdlog::default_logger();
    const auto sink   = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
    logger->sinks().push_back(sink);
    loki::search::write_run_stats(diag, stats);
    std::erase(logger->sinks(), sink);
    CHECK_THAT(log.str(),
               Catch::Matchers::ContainsSubstring("EP tree overloads") &&
                   Catch::Matchers::ContainsSubstring("in 1 of 2 runs") &&
                   Catch::Matchers::ContainsSubstring("pulsar_mask"));
    {
        const HighFive::File h5(diag.string(), HighFive::File::ReadOnly);
        const auto g = h5.getGroup("run_stats");
        CHECK(g.getAttribute("n_runs_overloaded").read<std::uint64_t>() == 1);
        CHECK(g.getAttribute("n_overloads").read<std::uint64_t>() > 0);
    }

    // A common product of the EP defaults: the per-nbins counts add up to
    // the detections, and leaves_dropped to the reduction's counts.
    auto options       = loki::search::CandidateOptions::ep_defaults();
    options.memory_gb  = 0.05;
    const auto product = outdir / "product_ep_results.h5";
    EPFreqSweep::publish_candidates(diag, product, options,
                                    loki::io::ObservationMetadata{}, 0.0);
    loki::search::write_run_provenance(
        product, {.config_effective     = "abc",
                  .command_line         = {},
                  .loki_version         = {},
                  .config_toml          = "x = 1\n",
                  .publish_command_line = "loki publish"});
    loki::search::write_run_stats(product, {});
    const HighFive::File h5(product.string(), HighFive::File::ReadOnly);
    H5F_info2_t file_info{};
    REQUIRE(H5Fget_info2(h5.getId(), &file_info) >= 0);
    CHECK(file_info.super.version >= 2U);
    CHECK(h5.getAttribute("config_toml").read<std::string>() == "x = 1\n");
    CHECK(h5.getAttribute("publish_command_line").read<std::string>() ==
          "loki publish");
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
    // Provenance: this build, and FNV-1a 64 of the configuration text.
    CHECK_FALSE(g.getAttribute("loki_version").read<std::string>().empty());
    CHECK_FALSE(g.getAttribute("loki_commit").read<std::string>().empty());
    CHECK(g.getAttribute("config_hash").read<std::string>() ==
          "fnv1a64:e71fa2190541574b");
    CHECK(g.hasAttribute("input_path"));
    CHECK(g.getAttribute("max_cell_dropped_snr").read<float>() >= 0.0F);
    CHECK(g.getAttribute("leaves_dropped").read<std::uint64_t>() ==
          h5.getAttribute("n_cell_dropped").read<std::uint64_t>() +
              h5.getAttribute("n_dominated").read<std::uint64_t>());
}
