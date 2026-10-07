#include "loki/pipelines/ep_freq_sweep.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <highfive/highfive.hpp>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::EPRegionPlanner;
using loki::pipelines::EPFreqSweep;
using loki::search::PulsarSearchConfig;

namespace {

PulsarSearchConfig make_test_cfg(double max_memory_gb = 4.0,
                                 double f_min         = 140.0,
                                 double f_max         = 145.0) {
    constexpr SizeType kNsamps = 1U << 16U;
    constexpr double kTsamp    = 64e-6;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = f_min, .max = f_max},
    };
    const int nthreads =
        std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 8);
    return {kNsamps,
            kTsamp,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            nthreads,
            max_memory_gb,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*bseg_ffa=*/kNsamps / 8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

std::pair<std::vector<float>, std::vector<float>>
make_noise_series(SizeType nsamps, unsigned seed = 42) {
    std::vector<float> ts_e(nsamps);
    std::vector<float> ts_v(nsamps, 1.0F);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    for (auto& x : ts_e) {
        x = noise(rng);
    }
    return {std::move(ts_e), std::move(ts_v)};
}

} // namespace

TEST_CASE("EPRegionPlanner plans valid memory-bounded chunks",
          "[ep_freq_sweep]") {
    const auto cfg = make_test_cfg(4.0, 140.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, /*min_pd=*/0.1F, "taylor",
                                         /*ref_ducy=*/0.1F);

    REQUIRE(planner.get_nchunks() > 0);
    const auto& chunks = planner.get_chunk_cfgs();
    const auto& stats  = planner.get_stats();

    REQUIRE(stats.get_nchunks() == chunks.size());
    CHECK(stats.get_max_memory_gb() <= cfg.get_max_process_memory_gb());
    CHECK(stats.get_max_sugg() >= 1024U);

    for (const auto& chunk : chunks) {
        CHECK(chunk.nominal_f_start < chunk.nominal_f_end);
        CHECK(chunk.actual_f_start <= chunk.nominal_f_start);
        CHECK(chunk.actual_f_end >= chunk.nominal_f_end);
        CHECK(chunk.chunk_memory_gb <= cfg.get_max_process_memory_gb());
        CHECK(chunk.max_sugg >= 1024U);
        CHECK(chunk.branch_max >= 32U);
        CHECK(!chunk.threshold_scheme.empty());
        CHECK(!chunk.branching_pattern.empty());
        CHECK(chunk.peak_complexity >= 1.0);
    }
}

TEST_CASE("EPRegionPlanner plans a band spanning two FFA regions",
          "[ep_freq_sweep]") {
    // 70-145 Hz spans two period octaves: 32 bins above 72.5 Hz, 64 below
    const auto cfg = make_test_cfg(4.0, 70.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, /*min_pd=*/0.1F, "taylor",
                                         /*ref_ducy=*/0.1F);

    std::set<SizeType> region_nbins;
    for (const auto& chunk : planner.get_chunk_cfgs()) {
        region_nbins.insert(chunk.cfg.get_nbins());
        CHECK(!chunk.threshold_scheme.empty());
    }
    CHECK(region_nbins == std::set<SizeType>{32U, 64U});
}

TEST_CASE("EPRegionPlanner stats report the maxima over its chunks",
          "[ep_freq_sweep]") {
    const auto cfg = make_test_cfg(4.0, 70.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, /*min_pd=*/0.1F, "taylor",
                                         /*ref_ducy=*/0.1F);
    const auto& stats = planner.get_stats();
    REQUIRE(stats.get_chunk_stats().size() == planner.get_nchunks());

    SizeType branch_max = 0;
    double memory_gb    = 0.0;
    for (const auto& chunk : stats.get_chunk_stats()) {
        branch_max = std::max(branch_max, chunk.branch_max);
        memory_gb  = std::max(memory_gb, chunk.memory_gb);
    }
    REQUIRE(branch_max > 0);
    REQUIRE(memory_gb > 0.0);
    CHECK(stats.get_max_branch_max() == branch_max);
    CHECK(stats.get_max_memory_gb() == static_cast<float>(memory_gb));
}

TEST_CASE("EPRegionPlanner HDF5 cache round-trip and validation",
          "[ep_freq_sweep]") {
    const auto outdir     = std::filesystem::temp_directory_path();
    const auto cache_file = outdir / "loki_test_ep_plan_cache_fast.h5";

    std::error_code ec;
    std::filesystem::remove(cache_file, ec);

    const auto cfg = make_test_cfg(4.0, 140.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, /*min_pd=*/0.1F, "taylor",
                                         /*ref_ducy=*/0.1F, cache_file);

    REQUIRE(std::filesystem::exists(cache_file));
    const auto& orig_chunks = planner.get_chunk_cfgs();

    // Reload from cache into a new planner
    EPRegionPlanner<float> reloaded(cfg, /*min_pd=*/0.1F, "taylor",
                                    /*ref_ducy=*/0.1F, cache_file);

    REQUIRE(reloaded.get_nchunks() == planner.get_nchunks());
    const auto& loaded_chunks = reloaded.get_chunk_cfgs();

    for (SizeType i = 0; i < orig_chunks.size(); ++i) {
        CHECK(std::abs(orig_chunks[i].nominal_f_start -
                       loaded_chunks[i].nominal_f_start) < 1e-9);
        CHECK(std::abs(orig_chunks[i].nominal_f_end -
                       loaded_chunks[i].nominal_f_end) < 1e-9);
        CHECK(std::abs(orig_chunks[i].actual_f_start -
                       loaded_chunks[i].actual_f_start) < 1e-9);
        CHECK(std::abs(orig_chunks[i].actual_f_end -
                       loaded_chunks[i].actual_f_end) < 1e-9);
        CHECK(orig_chunks[i].max_sugg == loaded_chunks[i].max_sugg);
        CHECK(orig_chunks[i].ncoords == loaded_chunks[i].ncoords);
        CHECK(orig_chunks[i].branch_max == loaded_chunks[i].branch_max);
        CHECK(orig_chunks[i].threshold_scheme ==
              loaded_chunks[i].threshold_scheme);
        CHECK(orig_chunks[i].branching_pattern ==
              loaded_chunks[i].branching_pattern);
    }

    // Direct cache mismatch tests by modifying copy of cache file
    const auto make_modified_cache =
        [&](const std::string& filename, const std::string& attr,
            const auto& val) -> std::filesystem::path {
        const auto mod_path = outdir / filename;
        std::filesystem::copy_file(
            cache_file, mod_path,
            std::filesystem::copy_options::overwrite_existing);
        const HighFive::File f(mod_path.string(), HighFive::File::ReadWrite);
        f.getAttribute(attr).write(val);
        return mod_path;
    };

    // 1. Mismatched max_process_memory_gb
    {
        const auto bad = make_modified_cache("cache_bad_mem.h5",
                                             "max_process_memory_gb", 8.0);
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::invalid_argument);
        std::filesystem::remove(bad, ec);
    }

    // 2. Mismatched frequency range
    {
        const auto bad =
            make_modified_cache("cache_bad_fmin.h5", "f_min", 130.0);
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::invalid_argument);
        std::filesystem::remove(bad, ec);
    }

    // 3. Mismatched min_pd
    {
        const auto bad =
            make_modified_cache("cache_bad_minpd.h5", "min_pd", 0.5F);
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::invalid_argument);
        std::filesystem::remove(bad, ec);
    }

    // 4. Mismatched poly_basis
    {
        const auto bad = make_modified_cache("cache_bad_poly.h5", "poly_basis",
                                             std::string("chebyshev"));
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::invalid_argument);
        std::filesystem::remove(bad, ec);
    }

    // 5. Non-existent file
    CHECK_THROWS_AS(reloaded.load_cache(outdir / "does_not_exist_xyz.h5"),
                    std::runtime_error);

    std::filesystem::remove(cache_file, ec);
}

TEST_CASE("EPFreqSweep executes sweep and writes unified results file",
          "[ep_freq_sweep]") {
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_sweep_test";
    const auto* const file_prefix = "test_sweep";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);

    // 140-142 Hz is one FFA region, 70-145 Hz is two (32 and 64 bins)
    const auto [f_min, f_max, nregions] = GENERATE(
        table<double, double, SizeType>({{140.0, 142.0, 1}, {70.0, 145.0, 2}}));
    CAPTURE(f_min, f_max);
    const auto cfg          = make_test_cfg(4.0, f_min, f_max);
    const auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps());

    // Run only 1 reference segment for fast test execution
    std::vector<SizeType> test_ref_segs{4U};

    EPFreqSweep sweep(cfg, /*show_progress=*/false, /*min_pd=*/0.1F, "taylor",
                      /*ref_ducy=*/0.1F, /*rfi_config=*/{},
                      /*plan_cache_file=*/std::nullopt,
                      /*n_runs=*/1U, test_ref_segs);

    sweep.execute(ts_e, ts_v, outdir, file_prefix);

    const auto result_file =
        outdir / std::format("{}_ep_results.h5", file_prefix);
    REQUIRE(std::filesystem::exists(result_file));

    // Inspect unified HDF5 file
    const HighFive::File h5(result_file.string(), HighFive::File::ReadOnly);
    REQUIRE(h5.hasAttribute("ep_sweep_version"));
    REQUIRE(h5.hasAttribute("nchunks"));
    REQUIRE(h5.hasAttribute("total_runtime"));
    REQUIRE(h5.hasAttribute("total_pruning_gflops"));

    SizeType nchunks{};
    h5.getAttribute("nchunks").read(nchunks);
    REQUIRE(nchunks >= 1);

    REQUIRE(h5.exist("chunks"));
    const auto chunks_grp = h5.getGroup("chunks");

    std::set<SizeType> region_nbins;
    for (SizeType i = 0; i < nchunks; ++i) {
        const auto chunk_name = std::format("chunk_{:04d}", i);
        REQUIRE(chunks_grp.exist(chunk_name));
        const auto chunk_grp = chunks_grp.getGroup(chunk_name);

        CHECK(chunk_grp.hasAttribute("nominal_f_start"));
        CHECK(chunk_grp.hasAttribute("nominal_f_end"));
        CHECK(chunk_grp.hasAttribute("max_sugg"));
        CHECK(chunk_grp.exist("threshold_scheme"));
        CHECK(chunk_grp.exist("branching_pattern"));
        CHECK(chunk_grp.exist("runs"));

        SizeType nbins{};
        chunk_grp.getAttribute("nbins").read(nbins);
        region_nbins.insert(nbins);
    }
    CHECK(region_nbins.size() == nregions);

    // Verify temporary per-chunk directory was cleaned up
    const auto tmp_dir = outdir / std::format(".tmp_{}_ep_chunks", file_prefix);
    CHECK(!std::filesystem::exists(tmp_dir));

    std::filesystem::remove_all(outdir, ec);
}
