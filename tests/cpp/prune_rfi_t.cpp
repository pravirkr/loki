#include <filesystem>
#include <format>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/algorithms/prune.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/cands.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"
#include "loki/utils/world_tree.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::EPMultiPassTime;
using loki::algorithms::ParamWindow;
using loki::algorithms::PruneRFIConfig;
using loki::cands::HarvestBuffer;
using loki::cands::PruneResultWriter;
using loki::cands::PruneStatsCollection;
using loki::memory::CircularView;
using loki::search::PulsarSearchConfig;

namespace {

PulsarSearchConfig make_small_cfg() {
    constexpr SizeType kNsamps = 1U << 16U;
    constexpr double kTsamp    = 64e-6;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 145.0},
    };
    return PulsarSearchConfig(
        kNsamps, kTsamp, /*nbins=*/32, /*eta=*/1.0, limits, /*ducy_max=*/0.3,
        /*wtsp=*/1.5, /*use_fourier=*/false, /*nthreads=*/1,
        /*max_process_memory_gb=*/8.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/1024, /*bseg_ffa=*/kNsamps / 16, /*snr_min=*/5.0,
        /*max_passing_candidates=*/1U << 22U, /*prune_poly_order=*/2);
}

std::pair<std::vector<float>, std::vector<float>>
make_noise_series(SizeType nsamps, unsigned seed) {
    std::vector<float> ts_e(nsamps);
    std::vector<float> ts_v(nsamps, 1.0F);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    for (auto& x : ts_e) {
        x = noise(rng);
    }
    return {std::move(ts_e), std::move(ts_v)};
}

std::filesystem::path result_path(const std::filesystem::path& outdir,
                                  std::string_view prefix,
                                  SizeType nsegments) {
    return outdir /
           std::format("{}_pruning_nstages_{}_results.h5", prefix, nsegments);
}

} // namespace

TEST_CASE("HarvestBuffer stores optional folds", "[prune_rfi]") {
    HarvestBuffer<float> with_folds(8, 16, true);
    HarvestBuffer<float> no_folds(8, 16, false);
    const std::vector<double> leaf(8, 1.5);
    const std::vector<float> fold(16, 0.25F);
    with_folds.push(leaf, fold, 11.0F, 4, 2, 1.25);
    no_folds.push(leaf, fold, 11.0F, 4, 2, 1.25);
    REQUIRE(with_folds.size() == 1);
    REQUIRE(no_folds.size() == 1);
    REQUIRE(with_folds.stores_folds());
    REQUIRE_FALSE(no_folds.stores_folds());
    REQUIRE(with_folds.get_folds().size() == 16);
    REQUIRE(no_folds.get_folds().empty());
    REQUIRE(with_folds.get_scores()[0] == 11.0F);
    REQUIRE(with_folds.get_levels()[0] == 4);
    REQUIRE(with_folds.get_seg_idx()[0] == 2);
    REQUIRE(with_folds.get_t_ref()[0] == 1.25);
    with_folds.clear();
    REQUIRE(with_folds.empty());
}

TEST_CASE("PruneStats RFI summaries", "[prune_rfi]") {
    loki::cands::PruneStats stats{};
    stats.n_leaves_phy  = 0;
    stats.n_leaves_surv = 0;
    REQUIRE(stats.surv_frac() == 0.0);
    stats.n_leaves_phy    = 10;
    stats.n_leaves_surv   = 5;
    stats.n_leaves_masked = 2;
    stats.n_harvested     = 1;
    REQUIRE(stats.surv_frac() == 0.5);
    REQUIRE(stats.get_summary().find("masked: 2") != std::string::npos);
    REQUIRE(stats.get_summary().find("harvested: 1") != std::string::npos);

    loki::cands::PruneStatsCollection coll;
    coll.update_stats(stats);
    const auto summary = coll.get_stats_summary();
    REQUIRE(summary.find("masked: 2") != std::string::npos);
    REQUIRE(summary.find("harvested: 1") != std::string::npos);
}

TEST_CASE("PruneResultWriter writes harvest groups", "[prune_rfi]") {
    const auto path =
        std::filesystem::temp_directory_path() / "loki_harvest_io_test.h5";
    std::filesystem::remove(path);

    const std::vector<float> thresholds(15, 1.5F);
    {
        PruneResultWriter writer(path, PruneResultWriter::Mode::kWrite);
        writer.write_metadata({"accel", "freq"}, 16, 1024, thresholds, {});
    }

    CircularView<double> empty_leaves{{}, {}};
    CircularView<float> empty_scores{{}, {}};
    const std::vector<SizeType> snail{0, 1, 2};
    PruneStatsCollection stats;

    PruneResultWriter appender(path, PruneResultWriter::Mode::kAppend);
    appender.write_run_results("000_00", snail, empty_leaves, empty_scores,
                               empty_scores, 0.0, 0, 2, stats);

    HarvestBuffer<float> empty_harvest(8, 16, true);
    appender.write_run_harvest("000_00", empty_harvest, 2, 0);

    appender.write_run_results("001_00", snail, empty_leaves, empty_scores,
                               empty_scores, 0.0, 0, 2, stats);
    HarvestBuffer<float> harvest(8, 16, true);
    const std::vector<double> leaf(8, 2.0);
    const std::vector<float> fold(16, 0.5F);
    harvest.push(leaf, fold, 12.5F, 6, 4, 3.0);
    appender.write_run_harvest("001_00", harvest, 2, 7);

    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    auto empty_g = file.getGroup("runs").getGroup("000_00").getGroup("harvest");
    SizeType n_total_empty = 0;
    empty_g.getAttribute("n_harvested_total").read(n_total_empty);
    REQUIRE(n_total_empty == 0);
    REQUIRE(empty_g.getDataSet("scores").getSpace().getDimensions()[0] == 0);

    auto hg = file.getGroup("runs").getGroup("001_00").getGroup("harvest");
    SizeType n_total = 0;
    hg.getAttribute("n_harvested_total").read(n_total);
    REQUIRE(n_total == 7);
    std::vector<float> scores;
    hg.getDataSet("scores").read(scores);
    REQUIRE(scores.size() == 1);
    REQUIRE(scores[0] == 12.5F);
    REQUIRE(hg.exist("folds"));
    const auto fold_dims = hg.getDataSet("folds").getSpace().getDimensions();
    REQUIRE(fold_dims[0] == 1);
    REQUIRE(fold_dims[1] == 2);
    REQUIRE(fold_dims[2] == 8);
}

TEST_CASE("EPMultiPassTime executes with default RFI config", "[prune_rfi]") {
    auto cfg = make_small_cfg();
    const auto nsegments =
        loki::plans::FFAPlan<float>(cfg).get_nsegments().back();
    const std::vector<float> thresholds(nsegments - 1, 1.5F);
    const std::vector<SizeType> ref_segs{nsegments / 2};
    auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps(), 0);

    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_rfi_cpp_default";
    std::filesystem::create_directories(outdir);

    EPMultiPassTime ep(cfg, thresholds, /*n_runs=*/std::nullopt, ref_segs,
                       /*ascend_levels=*/{}, /*max_sugg=*/1U << 14U,
                       /*batch_size=*/256, "taylor", /*show_progress=*/false);
    REQUIRE_NOTHROW(ep.execute(ts_e, ts_v, outdir, "cpp_rfi"));
}

TEST_CASE("EPMultiPassTime full-grid mask empties the tree", "[prune_rfi]") {
    auto cfg = make_small_cfg();
    const auto nsegments =
        loki::plans::FFAPlan<float>(cfg).get_nsegments().back();
    const std::vector<float> thresholds(nsegments - 1, 1.5F);
    const std::vector<SizeType> ref_segs{nsegments / 2};
    auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps(), 1);

    const auto limits = cfg.get_param_limits();
    PruneRFIConfig rfi;
    rfi.pulsar_mask.push_back(
        ParamWindow{.f_lo = limits[1].min, .f_hi = limits[1].max});

    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_rfi_cpp_mask";
    std::filesystem::create_directories(outdir);

    EPMultiPassTime ep(cfg, thresholds, /*n_runs=*/std::nullopt, ref_segs,
                       /*ascend_levels=*/{}, /*max_sugg=*/1U << 14U,
                       /*batch_size=*/256, "taylor", /*show_progress=*/false,
                       rfi);
    REQUIRE_NOTHROW(ep.execute(ts_e, ts_v, outdir, "cpp_mask"));

    const auto path = result_path(outdir, "cpp_mask", nsegments);
    REQUIRE(std::filesystem::exists(path));
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    auto runs = file.getGroup("runs");
    REQUIRE_FALSE(runs.listObjectNames().empty());
    auto run        = runs.getGroup(runs.listObjectNames().front());
    const auto dims = run.getDataSet("param_sets").getSpace().getDimensions();
    REQUIRE(dims[0] == 0);
}

TEST_CASE("EPMultiPassTime early harvest writes a harvest group",
          "[prune_rfi]") {
    auto cfg = make_small_cfg();
    const auto nsegments =
        loki::plans::FFAPlan<float>(cfg).get_nsegments().back();
    const std::vector<float> thresholds(nsegments - 1, 1.5F);
    const std::vector<SizeType> ref_segs{nsegments / 2};
    auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps(), 2);

    PruneRFIConfig rfi;
    rfi.harvest_scheme.assign(nsegments - 1, 0.0F);

    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_rfi_cpp_harvest";
    std::filesystem::create_directories(outdir);

    EPMultiPassTime ep(cfg, thresholds, /*n_runs=*/std::nullopt, ref_segs,
                       /*ascend_levels=*/{}, /*max_sugg=*/1U << 14U,
                       /*batch_size=*/256, "taylor", /*show_progress=*/false,
                       rfi);
    REQUIRE_NOTHROW(ep.execute(ts_e, ts_v, outdir, "cpp_harvest"));

    const auto path = result_path(outdir, "cpp_harvest", nsegments);
    REQUIRE(std::filesystem::exists(path));
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    auto runs = file.getGroup("runs");
    REQUIRE_FALSE(runs.listObjectNames().empty());
    auto run = runs.getGroup(runs.listObjectNames().front());
    REQUIRE(run.exist("harvest"));
    auto harvest     = run.getGroup("harvest");
    SizeType n_total = 0;
    harvest.getAttribute("n_harvested_total").read(n_total);
    REQUIRE(n_total > 0);
    REQUIRE(harvest.exist("scores"));
    REQUIRE(harvest.exist("param_sets"));
}

TEST_CASE(
    "make_default_harvest_scheme disables early stages and offsets threshold",
    "[prune_rfi]") {
    const std::vector<float> thresh{1.0F, 2.0F, 3.0F, 4.0F,  5.0F,  6.0F,
                                    7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    const auto harvest =
        loki::algorithms::make_default_harvest_scheme(thresh, /*min_level=*/5,
                                                      /*offset=*/10.0F,
                                                      /*min_snr=*/15.0F);
    REQUIRE(harvest.size() == thresh.size());
    // Levels 1..4 (indices 0..3) are disabled
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE_FALSE(loki::algorithms::is_harvest_enabled(harvest[i]));
        REQUIRE(harvest[i] == std::numeric_limits<float>::max());
    }
    // Level 5 (index 4): thresh=5.0, 5+10=15.0 == min_snr
    REQUIRE(loki::algorithms::is_harvest_enabled(harvest[4]));
    REQUIRE(harvest[4] == 15.0F);
    // Level 12 (index 11): thresh=12.0, 12+10=22.0 > min_snr
    REQUIRE(loki::algorithms::is_harvest_enabled(harvest[11]));
    REQUIRE(harvest[11] == 22.0F);
}

TEST_CASE("EPMultiPassTime with make_default_harvest_scheme runs cleanly",
          "[prune_rfi]") {
    auto cfg = make_small_cfg();
    const auto nsegments =
        loki::plans::FFAPlan<float>(cfg).get_nsegments().back();
    const std::vector<float> thresholds(nsegments - 1, 1.5F);
    const std::vector<SizeType> ref_segs{nsegments / 2};
    auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps(), 3);

    PruneRFIConfig rfi;
    rfi.harvest_scheme = loki::algorithms::make_default_harvest_scheme(
        thresholds, /*min_level=*/10, /*offset=*/10.0F, /*min_snr=*/15.0F);

    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_rfi_cpp_default_harvest";
    std::filesystem::create_directories(outdir);

    EPMultiPassTime ep(cfg, thresholds, /*n_runs=*/std::nullopt, ref_segs,
                       /*ascend_levels=*/{}, /*max_sugg=*/1U << 14U,
                       /*batch_size=*/256, "taylor", /*show_progress=*/false,
                       rfi);
    REQUIRE_NOTHROW(ep.execute(ts_e, ts_v, outdir, "cpp_default_harvest"));

    const auto path = result_path(outdir, "cpp_default_harvest", nsegments);
    REQUIRE(std::filesystem::exists(path));
}

