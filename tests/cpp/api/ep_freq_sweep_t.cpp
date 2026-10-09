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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using loki::Backend;
using loki::Exec;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::EPRegionPlanner;
using loki::pipelines::EPFreqSweep;
using loki::search::PulsarSearchConfig;

namespace {

PulsarSearchConfig make_test_cfg(double max_memory_gb = 4.0,
                                 double f_min         = 140.0,
                                 double f_max         = 145.0,
                                 int n_threads        = 0,
                                 bool use_fourier     = false) {
    constexpr SizeType kNsamps = 1U << 16U;
    constexpr double kTsamp    = 64e-6;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = f_min, .max = f_max},
    };
    const int nthreads =
        n_threads > 0
            ? n_threads
            : std::clamp(static_cast<int>(std::thread::hardware_concurrency()),
                         1, 8);
    return {kNsamps,
            kTsamp,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/use_fourier,
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

void check_memory_bounded_chunks(const PulsarSearchConfig& cfg,
                                 const EPRegionPlanner<float>& planner) {
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

} // namespace

TEST_CASE("EPRegionPlanner plans are reproducible", "[ep_freq_sweep]") {
    // The planner seeds its threshold scheme, so one configuration always
    // gives the same chunking, thresholds and max_sugg. One planner is also
    // checked against the memory bound so this does not need its own plan.
    const auto cfg = make_test_cfg(4.0, 140.0, 145.0);
    const EPRegionPlanner<float> a(cfg, /*min_pd=*/0.1F, "taylor",
                                   /*ref_ducy=*/0.1F);
    const EPRegionPlanner<float> b(cfg, /*min_pd=*/0.1F, "taylor",
                                   /*ref_ducy=*/0.1F);
    check_memory_bounded_chunks(cfg, a);
    REQUIRE(a.get_nchunks() == b.get_nchunks());
    const auto& ca = a.get_chunk_cfgs();
    const auto& cb = b.get_chunk_cfgs();
    for (SizeType i = 0; i < ca.size(); ++i) {
        CHECK(ca[i].threshold_scheme == cb[i].threshold_scheme);
        CHECK(ca[i].max_sugg == cb[i].max_sugg);
        CHECK(ca[i].nominal_f_start == cb[i].nominal_f_start);
        CHECK(ca[i].nominal_f_end == cb[i].nominal_f_end);
    }
}

TEST_CASE("EPRegionPlanner wide band covers two regions and reports maxima",
          "[ep_freq_sweep]") {
    // 70-145 Hz spans two period octaves: 32 bins above 72.5 Hz, 64 below.
    // One planner serves the region, stats, and branch_max checks.
    const auto cfg = make_test_cfg(4.0, 70.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, /*min_pd=*/0.1F, "taylor",
                                         /*ref_ducy=*/0.1F);
    const auto& chunks = planner.get_chunk_cfgs();
    const auto& stats  = planner.get_stats();

    std::set<SizeType> region_nbins;
    for (const auto& chunk : chunks) {
        region_nbins.insert(chunk.cfg.get_nbins());
        CHECK(!chunk.threshold_scheme.empty());
    }
    CHECK(region_nbins == std::set<SizeType>{32U, 64U});

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
    // The sweep peak (per-thread workspaces of the largest nbins run plus the
    // shared FFA buffers) is at least any single chunk and within the limit.
    CHECK(stats.get_max_memory_gb() >= static_cast<float>(memory_gb));
    CHECK(stats.get_max_memory_gb() <= cfg.get_max_process_memory_gb());

    const auto& chunk_stats = stats.get_chunk_stats();
    REQUIRE(chunk_stats.size() == chunks.size());
    for (SizeType i = 0; i < chunks.size(); ++i) {
        const loki::plans::FFAPlan<float> plan(chunks[i].cfg);
        const auto bp     = plan.get_branching_pattern("taylor");
        const auto needed = std::max(static_cast<SizeType>(std::ceil(
                                         2.0 * *std::ranges::max_element(bp))),
                                     SizeType{32});
        CHECK(chunks[i].branch_max >= needed);
        CHECK(chunk_stats[i].branch_max == chunks[i].branch_max);
    }
}

TEST_CASE("EPRegionPlanner rejects a stale plan cache version",
          "[ep_freq_sweep]") {
    const auto cache_file =
        std::filesystem::temp_directory_path() / "loki_test_ep_plan_stale.h5";
    std::error_code ec;
    std::filesystem::remove(cache_file, ec);

    const auto cfg = make_test_cfg(4.0, 140.0, 145.0);
    {
        const EPRegionPlanner<float> planner(cfg, 0.1F, "taylor", 0.1F,
                                             cache_file);
    }
    {
        HighFive::File file(cache_file.string(), HighFive::File::ReadWrite);
        file.getAttribute("ep_plan_cache_version").write(std::string("1.0.0"));
    }
    CHECK_THROWS_AS(
        EPRegionPlanner<float>(cfg, 0.1F, "taylor", 0.1F, cache_file),
        std::invalid_argument);
    std::filesystem::remove(cache_file, ec);
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
    CHECK(reloaded.get_stats().get_max_memory_gb() ==
          planner.get_stats().get_max_memory_gb());
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

    // 6. Memory model inputs: model version, workers, harvest store
    {
        const auto bad = make_modified_cache("cache_bad_model.h5",
                                             "ep_memory_model_version", 1U);
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::invalid_argument);
        std::filesystem::remove(bad, ec);
    }
    {
        // The memory-model kind is informational: a plan from the other
        // backend loads, and its peak is rechecked with this backend's policy.
        const auto other = make_modified_cache(
            "cache_other_kind.h5", "ep_memory_model_kind", std::string("cuda"));
        CHECK_NOTHROW(reloaded.load_cache(other));
        std::filesystem::remove(other, ec);
    }
    {
        const auto bad = make_modified_cache(
            "cache_bad_workers.h5", "n_workers", cfg.get_nthreads() + 1);
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::invalid_argument);
        std::filesystem::remove(bad, ec);
    }
    {
        loki::algorithms::PruneRFIConfig harvest_rfi;
        harvest_rfi.harvest_scheme = {5.0F};
        CHECK_THROWS_AS(EPRegionPlanner<float>(cfg, 0.1F, "taylor", 0.1F,
                                               cache_file, harvest_rfi),
                        std::invalid_argument);
        // A disabled harvest is normalised: its cap does not matter.
        loki::algorithms::PruneRFIConfig no_harvest_rfi;
        no_harvest_rfi.max_harvests = 7;
        CHECK_NOTHROW(EPRegionPlanner<float>(cfg, 0.1F, "taylor", 0.1F,
                                             cache_file, no_harvest_rfi));
    }
    if (cfg.get_nthreads() > 1) {
        CHECK_THROWS_AS(EPRegionPlanner<float>(cfg, 0.1F, "taylor", 0.1F,
                                               cache_file, {}, SizeType{1}),
                        std::invalid_argument);
    }

    // 7. A plan that no longer fits the limit is rejected on load
    {
        const auto bad = outdir / "cache_bad_peak.h5";
        std::filesystem::copy_file(
            cache_file, bad, std::filesystem::copy_options::overwrite_existing);
        {
            const HighFive::File f(bad.string(), HighFive::File::ReadWrite);
            f.getGroup("chunks")
                .getGroup("chunk_0000")
                .getAttribute("max_sugg")
                .write(SizeType{1} << 30U);
        }
        CHECK_THROWS_AS(reloaded.load_cache(bad), std::runtime_error);
        std::filesystem::remove(bad, ec);
    }

    std::filesystem::remove(cache_file, ec);
}

TEST_CASE("EPFreqSweep executes sweep and writes unified results file",
          "[ep_freq_sweep]") {
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_sweep_test";
    const auto* const file_prefix = "test_sweep";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);

    // One FFA region. The two-region split is checked on the planner above.
    constexpr double kFMin  = 140.0;
    constexpr double kFMax  = 142.0;
    const auto cfg          = make_test_cfg(4.0, kFMin, kFMax);
    const auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps());

    // Run only 1 reference segment for fast test execution
    std::vector<SizeType> test_ref_segs{4U};

    EPFreqSweep sweep(cfg, /*show_progress=*/false, /*min_pd=*/0.1F, "taylor",
                      /*ref_ducy=*/0.1F, /*rfi_config=*/{},
                      /*plan_cache_file=*/std::nullopt,
                      /*n_runs=*/std::nullopt, test_ref_segs);

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
    CHECK(region_nbins.size() == 1);

    // Verify temporary per-chunk directory was cleaned up
    const auto tmp_dir = outdir / std::format(".tmp_{}_ep_chunks", file_prefix);
    CHECK(!std::filesystem::exists(tmp_dir));

    std::filesystem::remove_all(outdir, ec);
}

// ---------------------------------------------------------------------------
// CUDA backend
// ---------------------------------------------------------------------------

namespace {

bool cuda_available() { return loki::is_available(Backend::kCUDA); }

/// Names of the groups and datasets below @p grp, recursively, as paths.
void collect_names(const HighFive::Group& grp,
                   const std::string& prefix,
                   std::set<std::string>& out) {
    for (const auto& name : grp.listObjectNames()) {
        const auto path = prefix + "/" + name;
        out.insert(path);
        if (grp.getObjectType(name) == HighFive::ObjectType::Group) {
            collect_names(grp.getGroup(name), path, out);
        }
    }
}

} // namespace

TEST_CASE("EPFreqSweep CUDA executes a sweep and writes the unified file",
          "[ep_freq_sweep][cuda]") {
    if (!cuda_available()) {
        SKIP("needs a CUDA build");
    }
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_sweep_cuda_test";
    const auto cache_file    = outdir / "plan_cuda.h5";
    const auto* const prefix = "cuda_sweep";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);
    std::filesystem::create_directories(outdir);

    const auto cfg          = make_test_cfg(4.0, 140.0, 142.0);
    const auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps());
    const std::vector<SizeType> ref_segs{4U};

    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, cache_file,
                      std::nullopt, ref_segs, Exec::cuda(0));
    sweep.execute(ts_e, ts_v, outdir, prefix);

    const auto result_file = outdir / std::format("{}_ep_results.h5", prefix);
    REQUIRE(std::filesystem::exists(result_file));
    {
        const HighFive::File h5(result_file.string(), HighFive::File::ReadOnly);
        REQUIRE(h5.hasAttribute("ep_sweep_version"));
        REQUIRE(h5.hasAttribute("total_pruning_gflops"));
        SizeType nchunks{};
        h5.getAttribute("nchunks").read(nchunks);
        REQUIRE(nchunks >= 1);
        const auto chunks_grp = h5.getGroup("chunks");
        for (SizeType i = 0; i < nchunks; ++i) {
            const auto grp =
                chunks_grp.getGroup(std::format("chunk_{:04d}", i));
            CHECK(grp.exist("runs"));
            CHECK(grp.exist("threshold_scheme"));
            CHECK(grp.exist("branching_pattern"));
        }
    }
    CHECK(!std::filesystem::exists(outdir /
                                   std::format(".tmp_{}_ep_chunks", prefix)));

    SECTION("the plan cache loads on CUDA and on the CPU") {
        REQUIRE(std::filesystem::exists(cache_file));
        // Same backend: the plan is reused, so the sweep runs without a
        // threshold simulation.
        EPFreqSweep again(cfg, false, 0.1F, "taylor", 0.1F, {}, cache_file,
                          std::nullopt, ref_segs, Exec::cuda(0));
        again.execute(ts_e, ts_v, outdir, "cuda_sweep_again");
        CHECK(
            std::filesystem::exists(outdir / "cuda_sweep_again_ep_results.h5"));

        // The other backend: the CUDA plan has one worker, which this CPU
        // sweep also uses, and its peak is rechecked on the CPU.
        CHECK_NOTHROW(EPFreqSweep(cfg, false, 0.1F, "taylor", 0.1F, {},
                                  cache_file, std::nullopt, ref_segs));
    }

    std::filesystem::remove_all(outdir, ec);
}

TEST_CASE("EPFreqSweep CUDA rejects an active RFI config",
          "[ep_freq_sweep][cuda]") {
    if (!cuda_available()) {
        SKIP("needs a CUDA build");
    }
    const auto cfg = make_test_cfg(4.0, 140.0, 142.0);
    loki::algorithms::PruneRFIConfig rfi;
    rfi.harvest_scheme = {5.0F};
    CHECK_THROWS_AS(EPFreqSweep(cfg, false, 0.1F, "taylor", 0.1F, rfi,
                                std::nullopt, std::nullopt, std::nullopt,
                                Exec::cuda(0)),
                    std::invalid_argument);
}

TEST_CASE("EPFreqSweep CUDA runs both bin groups within the device budget",
          "[ep_freq_sweep][cuda]") {
    if (!cuda_available()) {
        SKIP("needs a CUDA build");
    }
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_sweep_cuda_groups";
    const auto cache_file = outdir / "plan.h5";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);
    std::filesystem::create_directories(outdir);

    // 70-145 Hz: 32 and 64 bins, so two chunk groups. The plan is cached so
    // the sweep below does not simulate the thresholds again.
    const auto cfg = make_test_cfg(4.0, 70.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, 0.1F, "taylor", 0.1F, cache_file,
                                         {}, std::nullopt, Exec::cuda(0));
    std::set<SizeType> region_nbins;
    for (const auto& chunk : planner.get_chunk_cfgs()) {
        region_nbins.insert(chunk.cfg.get_nbins());
        CHECK(chunk.chunk_memory_gb <= cfg.get_max_process_memory_gb());
        CHECK(chunk.ffa_transient_bytes > 0);
        CHECK(!chunk.threshold_scheme.empty());
    }
    CHECK(region_nbins == std::set<SizeType>{32U, 64U});
    const auto& stats = planner.get_stats();
    // The modelled device peak, which the sweep's tripwires enforce again
    // as it allocates, stays inside the device budget.
    CHECK(stats.get_max_memory_gb() <= stats.get_memory_limit_gb());
    CHECK(stats.get_memory_limit_gb() <= cfg.get_max_process_memory_gb());
    {
        const HighFive::File cache(cache_file.string(),
                                   HighFive::File::ReadOnly);
        std::string arch;
        std::string kind;
        cache.getAttribute("device_arch").read(arch);
        cache.getAttribute("ep_memory_model_kind").read(kind);
        CHECK(arch.starts_with("sm_"));
        CHECK(kind == "cuda");
    }

    const auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps());
    const std::vector<SizeType> ref_segs{4U};
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, cache_file,
                      std::nullopt, ref_segs, Exec::cuda(0));
    sweep.execute(ts_e, ts_v, outdir, "groups");

    const HighFive::File h5((outdir / "groups_ep_results.h5").string(),
                            HighFive::File::ReadOnly);
    SizeType nchunks{};
    h5.getAttribute("nchunks").read(nchunks);
    std::set<SizeType> written;
    const auto chunks_grp = h5.getGroup("chunks");
    for (SizeType i = 0; i < nchunks; ++i) {
        const auto grp = chunks_grp.getGroup(std::format("chunk_{:04d}", i));
        SizeType nbins{};
        grp.getAttribute("nbins").read(nbins);
        written.insert(nbins);
        CHECK(grp.exist("runs"));
    }
    CHECK(written == std::set<SizeType>{32U, 64U});
    std::filesystem::remove_all(outdir, ec);
}

TEST_CASE("EPFreqSweep CUDA matches the CPU sweep on the same plan",
          "[ep_freq_sweep][cuda]") {
    if (!cuda_available()) {
        SKIP("needs a CUDA build");
    }
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_sweep_parity_test";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);
    std::filesystem::create_directories(outdir);

    const auto cfg    = make_test_cfg(4.0, 140.0, 142.0, /*n_threads=*/1);
    auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps());
    // A strong pulse train, so that the pruning has survivors to compare.
    constexpr double kFSignal = 141.0;
    for (SizeType i = 0; i < ts_e.size(); ++i) {
        const double phase =
            std::fmod(static_cast<double>(i) * cfg.get_tsamp() * kFSignal, 1.0);
        if (phase < 0.05) {
            ts_e[i] += 4.0F;
        }
    }
    const std::vector<SizeType> ref_segs{4U};

    // The CPU plan, with one worker, is loaded as it is by the GPU sweep: a
    // plan from the other backend, rechecked with the GPU policy. Both
    // backends then prune the same chunks and thresholds.
    const auto cache_file = outdir / "plan.h5";
    {
        EPFreqSweep cpu(cfg, false, 0.1F, "taylor", 0.1F, {}, cache_file,
                        std::nullopt, ref_segs);
        cpu.execute(ts_e, ts_v, outdir, "cpu");
    }
    {
        EPFreqSweep gpu(cfg, false, 0.1F, "taylor", 0.1F, {}, cache_file,
                        std::nullopt, ref_segs, Exec::cuda(0));
        gpu.execute(ts_e, ts_v, outdir, "gpu");
    }

    const HighFive::File cpu_h5((outdir / "cpu_ep_results.h5").string(),
                                HighFive::File::ReadOnly);
    const HighFive::File gpu_h5((outdir / "gpu_ep_results.h5").string(),
                                HighFive::File::ReadOnly);
    std::set<std::string> cpu_names;
    std::set<std::string> gpu_names;
    collect_names(cpu_h5.getGroup("/"), "", cpu_names);
    collect_names(gpu_h5.getGroup("/"), "", gpu_names);
    // Same layout: every group and dataset, in every chunk and run. The CUDA
    // pruning kernels do not time their stages, so a run has no timer_stats
    // dataset there (a diagnostic only).
    std::erase_if(cpu_names, [](const std::string& n) {
        return n.ends_with("/timer_stats");
    });
    CHECK(cpu_names == gpu_names);

    SizeType cpu_nchunks{};
    SizeType gpu_nchunks{};
    cpu_h5.getAttribute("nchunks").read(cpu_nchunks);
    gpu_h5.getAttribute("nchunks").read(gpu_nchunks);
    CHECK(cpu_nchunks == gpu_nchunks);

    // The survivors agree: the GPU brute fold accumulates with atomics, so
    // the scores are compared to a tolerance, not bit for bit.
    const auto run_path = std::string("/chunks/chunk_0000/runs/004_00");
    const auto cpu_scores =
        cpu_h5.getDataSet(run_path + "/scores").read<std::vector<float>>();
    const auto gpu_scores =
        gpu_h5.getDataSet(run_path + "/scores").read<std::vector<float>>();
    REQUIRE(!cpu_scores.empty());
    REQUIRE(!gpu_scores.empty());
    const auto cpu_max = *std::ranges::max_element(cpu_scores);
    const auto gpu_max = *std::ranges::max_element(gpu_scores);
    CHECK(gpu_max == Catch::Approx(cpu_max).epsilon(0.05));
    CHECK(cpu_max > 8.0F);

    std::filesystem::remove_all(outdir, ec);
}

TEST_CASE("EPFreqSweep CUDA sweeps with Fourier folds",
          "[ep_freq_sweep][cuda]") {
    if (!cuda_available()) {
        SKIP("needs a CUDA build");
    }
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_sweep_cuda_fourier";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);

    const auto cfg          = make_test_cfg(4.0, 140.0, 142.0, /*n_threads=*/0,
                                            /*use_fourier=*/true);
    const auto [ts_e, ts_v] = make_noise_series(cfg.get_nsamps());
    // The tripwires of the engine (workspace and shared buffers against the
    // model) throw on a drift, so a clean sweep checks the Fourier model.
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, {}, std::nullopt,
                      std::nullopt, std::vector<SizeType>{4U}, Exec::cuda(0));
    CHECK_NOTHROW(sweep.execute(ts_e, ts_v, outdir, "fourier"));
    CHECK(std::filesystem::exists(outdir / "fourier_ep_results.h5"));
    std::filesystem::remove_all(outdir, ec);
}

TEST_CASE("EPRegionPlanner accepts a plan cache written on the other backend",
          "[ep_freq_sweep]") {
    const auto outdir =
        std::filesystem::temp_directory_path() / "loki_ep_cross_backend";
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);
    std::filesystem::create_directories(outdir);
    const auto cache_file = outdir / "plan_cpu.h5";

    const auto cfg = make_test_cfg(4.0, 140.0, 145.0);
    const EPRegionPlanner<float> planner(cfg, 0.1F, "taylor", 0.1F, cache_file);
    {
        // Relabel the plan as a CUDA plan. The chunks stay as planned, and the
        // loader rechecks the peak with the CPU policy.
        const HighFive::File f(cache_file.string(), HighFive::File::ReadWrite);
        f.getAttribute("ep_backend").write(std::string("cuda"));
        f.getAttribute("ep_memory_model_kind").write(std::string("cuda"));
    }
    const EPRegionPlanner<float> reloaded(cfg, 0.1F, "taylor", 0.1F,
                                          cache_file);
    REQUIRE(reloaded.get_nchunks() == planner.get_nchunks());
    CHECK(reloaded.get_stats().get_max_memory_gb() ==
          planner.get_stats().get_max_memory_gb());

    std::filesystem::remove_all(outdir, ec);
}
