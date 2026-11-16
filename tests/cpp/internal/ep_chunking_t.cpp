#include "lib/algorithms/ep_chunking.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"
#include "lib/common/plans_detail.hpp"
#include "lib/detail/utils.hpp"
#include "lib/search/prune_mask.hpp"

#include "ep_seed_slices.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::generate_ffa_regions;
using loki::algorithms::detail::ep_cuda_effective_max_sugg;
using loki::algorithms::detail::ep_sweep_peak_gb;
using loki::algorithms::detail::EPHarvestBound;
using loki::algorithms::detail::EPMemoryContext;
using loki::algorithms::detail::EPMemoryKind;
using loki::algorithms::detail::plan_chunks;
using loki::algorithms::detail::RegionDesign;
using loki::search::PulsarSearchConfig;

// Chunking is tested on synthetic region designs: the planner's only costly
// input (the threshold scheme) is replaced by a chosen safe_complexity, so
// the memory limit can be made to bind on a tiny search.

namespace {

constexpr double kTsamp = 64e-6;

PulsarSearchConfig make_cfg(double f_min, double f_max, int nthreads) {
    constexpr SizeType kNsamps = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = f_min, .max = f_max},
    };
    return {kNsamps,
            kTsamp,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            nthreads,
            /*max_process_memory_gb=*/8.0,
            /*region_growth_factor=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*nsegments=*/8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

/// One design per FFA region. The first region needs @p first_max_sugg
/// suggestions for a chunk spanning the whole region; the others the minimum.
std::vector<RegionDesign> make_designs(const PulsarSearchConfig& cfg,
                                       double first_max_sugg) {
    const auto regions = generate_ffa_regions(
        1.0 / cfg.get_f_max(), 1.0 / cfg.get_f_min(), cfg.get_tsamp(),
        cfg.get_nbins(), cfg.get_eta(), cfg.get_region_growth_factor(),
        cfg.get_nbins_max());
    std::vector<RegionDesign> designs;
    for (const auto& r : regions) {
        const auto rcfg =
            cfg.get_updated_ep_config(r.nbins, r.eta, r.f_start, r.f_end);
        const loki::plans::FFAPlan<float> plan(rcfg);
        const auto ncoords = static_cast<double>(plan.get_ncoords().back());
        designs.push_back(RegionDesign{
            .f_start   = r.f_start,
            .f_end     = r.f_end,
            .nbins     = r.nbins,
            .eta       = r.eta,
            .nsegments = plan.get_nsegments().back(),
            .safe_complexity =
                designs.empty() ? static_cast<float>(first_max_sugg / ncoords)
                                : 1.0F,
        });
    }
    return designs;
}

EPMemoryContext make_ctx(const PulsarSearchConfig& cfg,
                         int n_workers          = 0,
                         EPHarvestBound harvest = {}) {
    return {.nparams   = cfg.get_nparams(),
            .nsamps    = cfg.get_nsamps(),
            .n_workers = n_workers > 0 ? n_workers : cfg.get_nthreads(),
            .harvest   = harvest};
}

/// A stand-in for the CUB sizing query, so the CUDA policy runs without a
/// device: a fixed fraction of the leaves.
SizeType fake_cub_scratch_bytes(SizeType max_n_leaves, int /*device*/) {
    return 16 * max_n_leaves;
}

/// The CUDA policy: one worker, as on the device.
EPMemoryContext make_cuda_ctx(const PulsarSearchConfig& cfg) {
    return {.nparams           = cfg.get_nparams(),
            .nsamps            = cfg.get_nsamps(),
            .n_workers         = 1,
            .harvest           = {},
            .kind              = EPMemoryKind::kCuda,
            .cub_scratch_bytes = &fake_cub_scratch_bytes};
}

/// One synthetic design over [f_start, f_end] with a chosen complexity.
RegionDesign make_design(const PulsarSearchConfig& cfg,
                         double f_start,
                         double f_end,
                         SizeType nbins,
                         float safe_complexity) {
    const auto rcfg = cfg.get_updated_ep_config(nbins, 1.0, f_start, f_end);
    const loki::plans::FFAPlan<float> plan(rcfg);
    return RegionDesign{
        .f_start         = f_start,
        .f_end           = f_end,
        .nbins           = nbins,
        .eta             = 1.0,
        .nsegments       = plan.get_nsegments().back(),
        .safe_complexity = safe_complexity,
    };
}

/// A heavy run (large max_sugg, nbins 32) followed by a light run (nbins
/// 128) whose wide band sets a large shared FFA size only at the end.
std::vector<RegionDesign>
make_late_shared_designs(const PulsarSearchConfig& cfg) {
    return {make_design(cfg, 100.0, 110.0, 32, 6.0e4F),
            make_design(cfg, 40.0, 120.0, 128, 1.0e-3F)};
}

void check_plan_fits(const loki::algorithms::detail::ChunkPlan& plan,
                     const EPMemoryContext& ctx,
                     double limit_gb) {
    CHECK(plan.peak_memory_gb <= limit_gb);
    CHECK(plan.peak_memory_gb == ep_sweep_peak_gb<float>(plan.chunk_cfgs, ctx));
    for (const auto& c : plan.chunk_cfgs) {
        CHECK(c.buffer_size <= plan.buffer_size);
        CHECK(c.coord_size <= plan.coord_size);
        CHECK(c.fold_size <= plan.fold_size);
    }
}

SizeType count_nbins(const loki::algorithms::detail::ChunkPlan& plan,
                     SizeType nbins) {
    return static_cast<SizeType>(
        std::ranges::count_if(plan.chunk_cfgs, [&](const auto& c) {
            return c.cfg.get_nbins() == nbins;
        }));
}

} // namespace

TEST_CASE("EP chunking charges per-thread memory per nbins run, not per sweep",
          "[ep_chunking]") {
    // The first region's chunks are limited by their own huge max_sugg. The
    // next region (more bins, light) must not be charged that max_sugg: the
    // sweep frees the first run's workspaces before allocating the next.
    constexpr double kLimitGb = 0.2;
    const auto cfg            = make_cfg(70.0, 145.0, /*nthreads=*/2);
    const auto designs        = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    REQUIRE(designs.size() >= 2);
    REQUIRE(designs[0].nbins != designs[1].nbins);

    const auto ctx  = make_ctx(cfg);
    const auto plan = plan_chunks<float>(cfg, "taylor", designs,
                                         /*max_drift=*/0.0, kLimitGb, ctx);

    // The limit binds in the first region, not in the second.
    CHECK(count_nbins(plan, designs[0].nbins) >= 2);
    CHECK(count_nbins(plan, designs[1].nbins) == 1);

    check_plan_fits(plan, ctx, kLimitGb);
}

TEST_CASE("EP chunking covers every region without gaps", "[ep_chunking]") {
    const auto cfg     = make_cfg(70.0, 145.0, /*nthreads=*/2);
    const auto designs = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    const auto plan = plan_chunks<float>(cfg, "taylor", designs,
                                         /*max_drift=*/0.0, 0.2, make_ctx(cfg));
    double covered  = 0.0;
    for (const auto& c : plan.chunk_cfgs) {
        covered += c.nominal_f_end - c.nominal_f_start;
    }
    double span = 0.0;
    for (const auto& d : designs) {
        span += d.f_end - d.f_start;
    }
    CHECK(std::abs(covered - span) < 1e-6);
}

TEST_CASE("EP chunking reports the memory that was rejected", "[ep_chunking]") {
    const auto cfg     = make_cfg(70.0, 145.0, /*nthreads=*/2);
    const auto designs = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    CHECK_THROWS_WITH(
        plan_chunks<float>(cfg, "taylor", designs, /*max_drift=*/0.0, 1.0e-4,
                           make_ctx(cfg)),
        Catch::Matchers::ContainsSubstring("Cannot fit minimum viable chunk") &&
            Catch::Matchers::ContainsSubstring("Required memory"));
}

TEST_CASE("EP chunking charges the input series", "[ep_chunking]") {
    constexpr double kLimitGb = 0.2;
    const auto cfg            = make_cfg(70.0, 145.0, /*nthreads=*/2);
    const auto designs        = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    auto no_inputs            = make_ctx(cfg);
    no_inputs.nsamps          = 0;
    auto big_inputs           = make_ctx(cfg);
    // 0.05 GiB of ts_e + ts_v.
    big_inputs.nsamps = (SizeType{1} << 30U) / 20 / (2 * sizeof(float));

    const auto plan_without =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, no_inputs);
    const auto plan_with =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, big_inputs);
    check_plan_fits(plan_with, big_inputs, kLimitGb);
    CHECK(count_nbins(plan_with, designs[0].nbins) >
          count_nbins(plan_without, designs[0].nbins));
}

TEST_CASE("EP chunking charges only the workers that run", "[ep_chunking]") {
    // n_runs < nthreads: fewer workspaces are allocated, so chunks widen.
    constexpr double kLimitGb = 0.3;
    const auto cfg            = make_cfg(70.0, 145.0, /*nthreads=*/4);
    const auto designs        = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    const auto ctx_all        = make_ctx(cfg);
    const auto ctx_one        = make_ctx(cfg, /*n_workers=*/1);

    const auto plan_all =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx_all);
    const auto plan_one =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx_one);
    check_plan_fits(plan_all, ctx_all, kLimitGb);
    check_plan_fits(plan_one, ctx_one, kLimitGb);
    CHECK(plan_one.chunk_cfgs.size() < plan_all.chunk_cfgs.size());
}

TEST_CASE("EP chunking charges the harvest store only when enabled",
          "[ep_chunking]") {
    constexpr double kLimitGb = 0.2;
    const auto cfg            = make_cfg(70.0, 145.0, /*nthreads=*/2);
    const auto designs        = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    const auto ctx_off        = make_ctx(cfg);
    // ~22 MB per worker at nbins = 32.
    const auto ctx_on = make_ctx(cfg, 0,
                                 EPHarvestBound{.enabled      = true,
                                                .max_harvests = 1U << 16U,
                                                .store_folds  = true});

    // Disabled harvesting is normalised whatever the cap.
    loki::algorithms::PruneRFIConfig rfi;
    rfi.max_harvests = 1U << 16U;
    CHECK_FALSE(EPHarvestBound::from(rfi).enabled);
    CHECK(EPHarvestBound::from(rfi).max_harvests == 0);

    const auto plan_off =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx_off);
    const auto plan_on =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx_on);
    check_plan_fits(plan_on, ctx_on, kLimitGb);
    CHECK(count_nbins(plan_on, designs[0].nbins) >
          count_nbins(plan_off, designs[0].nbins));
}

TEST_CASE("EP chunking replans when the shared FFA size grows late",
          "[ep_chunking]") {
    // Pass 1 fills the limit with the heavy run next to a small shared size;
    // the light run then enlarges it, so the heavy run must be replanned.
    constexpr double kLimitGb = 0.12;
    const auto cfg            = make_cfg(30.0, 150.0, /*nthreads=*/2);
    const auto designs        = make_late_shared_designs(cfg);
    auto ctx                  = make_ctx(cfg);
    ctx.batch_size =
        128; // Keep scratch below the synthetic shared-buffer budgets.

    const auto plan =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx);
    CHECK(plan.replanned);
    CHECK(plan.shared_cap_scale == 1.0);
    check_plan_fits(plan, ctx, kLimitGb);
}

TEST_CASE("EP chunking searches a smaller shared size when the replan is "
          "infeasible",
          "[ep_chunking]") {
    // The heavy run's minimum-width chunk does not fit next to the light
    // run's full shared size: the light run must be split further so the
    // shared size shrinks.
    constexpr double kLimitGb = 0.065;
    const auto cfg            = make_cfg(30.0, 150.0, /*nthreads=*/2);
    const auto designs        = make_late_shared_designs(cfg);
    auto ctx                  = make_ctx(cfg);
    ctx.batch_size =
        128; // Keep scratch below the synthetic shared-buffer budgets.

    const auto plan =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx);
    CHECK(plan.replanned);
    CHECK(plan.shared_cap_scale < 1.0);
    CHECK(count_nbins(plan, designs[1].nbins) >= 2);
    check_plan_fits(plan, ctx, kLimitGb);
}

TEST_CASE("EP chunking CUDA policy fits the limit and counts the transient",
          "[ep_chunking]") {
    constexpr double kLimitGb = 2.0;
    const auto cfg            = make_cfg(70.0, 145.0, /*nthreads=*/4);
    const auto designs        = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    const auto ctx            = make_cuda_ctx(cfg);
    const auto plan =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx);

    check_plan_fits(plan, ctx, kLimitGb);
    REQUIRE(!plan.chunk_cfgs.empty());
    for (const auto& chunk : plan.chunk_cfgs) {
        CHECK(chunk.ffa_transient_bytes > 0);
        // The device world tree needs a capacity above its largest batch.
        CHECK(chunk.max_sugg >= (ctx.batch_size * chunk.branch_max) + 1);
    }
}

TEST_CASE("EP chunking CUDA policy replans when the shared size grows late",
          "[ep_chunking]") {
    // Same designs as the CPU replan test, with one worker. The late shared
    // size no longer fits, so the heavy region is planned again next to it.
    constexpr double kLimitGb = 0.12;
    const auto cfg            = make_cfg(30.0, 150.0, /*nthreads=*/4);
    const auto designs        = make_late_shared_designs(cfg);
    const auto ctx            = make_cuda_ctx(cfg);
    const auto plan =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx);
    CHECK(plan.replanned);
    CHECK(plan.shared_cap_scale == 1.0);
    check_plan_fits(plan, ctx, kLimitGb);
}

TEST_CASE("EP chunking CUDA policy requires the CUB sizer", "[ep_chunking]") {
    const auto cfg        = make_cfg(140.0, 145.0, /*nthreads=*/1);
    const auto designs    = make_designs(cfg, /*first_max_sugg=*/1.0e3);
    auto ctx              = make_cuda_ctx(cfg);
    ctx.cub_scratch_bytes = nullptr;
    CHECK_THROWS_AS(plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, ctx),
                    std::logic_error);
}

TEST_CASE("EP chunking with the CUDA policy charges one worker, whatever "
          "nthreads is",
          "[ep_chunking]") {
    constexpr double kLimitGb = 0.2;
    const auto designs_cfg    = make_cfg(70.0, 145.0, /*nthreads=*/1);
    const auto designs        = make_designs(designs_cfg, 2.0e6);

    // The thread count of the config must not change a CUDA plan.
    const auto cfg_few  = make_cfg(70.0, 145.0, /*nthreads=*/1);
    const auto cfg_many = make_cfg(70.0, 145.0, /*nthreads=*/16);
    const auto few      = plan_chunks<float>(cfg_few, "taylor", designs, 0.0,
                                             kLimitGb, make_cuda_ctx(cfg_few));
    const auto many     = plan_chunks<float>(cfg_many, "taylor", designs, 0.0,
                                             kLimitGb, make_cuda_ctx(cfg_many));
    REQUIRE(few.chunk_cfgs.size() == many.chunk_cfgs.size());
    CHECK(few.peak_memory_gb == many.peak_memory_gb);
    // The same chunks would need more under 16 CPU workers.
    CHECK(ep_sweep_peak_gb<float>(few.chunk_cfgs, make_ctx(cfg_many)) >
          few.peak_memory_gb);

    check_plan_fits(few, make_cuda_ctx(cfg_few), kLimitGb);
}

TEST_CASE("EP chunking with the CUDA policy floors max_sugg above a batch",
          "[ep_chunking]") {
    const auto cfg     = make_cfg(70.0, 145.0, /*nthreads=*/1);
    const auto designs = make_designs(cfg, /*first_max_sugg=*/1.0);
    const auto ctx     = make_cuda_ctx(cfg);
    const auto plan = plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, ctx);
    REQUIRE(!plan.chunk_cfgs.empty());
    for (const auto& c : plan.chunk_cfgs) {
        CHECK(c.max_sugg >=
              ep_cuda_effective_max_sugg(ctx.batch_size, c.branch_max, 0));
    }
}

TEST_CASE("EP chunking with the CUDA policy charges the FFA transient scratch",
          "[ep_chunking]") {
    const auto cfg     = make_cfg(100.0, 110.0, /*nthreads=*/1);
    const auto designs = make_designs(cfg, 1.0e4);
    const auto ctx     = make_cuda_ctx(cfg);
    const auto plan = plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, ctx);
    REQUIRE(!plan.chunk_cfgs.empty());
    for (const auto& c : plan.chunk_cfgs) {
        CHECK(c.ffa_transient_bytes > 0);
    }
    // The CPU policy has none.
    const auto cpu_ctx = make_ctx(cfg);
    const auto cpu =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, cpu_ctx);
    for (const auto& c : cpu.chunk_cfgs) {
        CHECK(c.ffa_transient_bytes == 0);
    }
    CHECK(plan.peak_memory_gb == ep_sweep_peak_gb<float>(plan.chunk_cfgs, ctx));
}

TEST_CASE("EP capacity rounds live ring population and a branched batch",
          "[ep_chunking]") {
    using loki::algorithms::detail::ep_tree_capacity;
    using loki::algorithms::detail::kEPRingGranularity;
    // live + batch * branch_max + 1, up to a multiple of the granularity.
    CHECK(ep_tree_capacity(4096, 64, 16, EPMemoryKind::kCuda) == 6144);
    CHECK(ep_tree_capacity(1U << 22U, 64, 16, EPMemoryKind::kCuda) ==
          (1U << 22U) + 2048);
    CHECK(ep_tree_capacity(0, 1, 1, EPMemoryKind::kCpu) == kEPRingGranularity);
    CHECK(ep_tree_capacity(1023, 1, 1, EPMemoryKind::kCpu) ==
          2 * kEPRingGranularity);
    CHECK_THROWS_AS(ep_tree_capacity(1U << 30U, 64, 16, EPMemoryKind::kCuda),
                    std::overflow_error);
}

TEST_CASE("EP ring margin covers the Monte Carlo error of the forecast",
          "[ep_chunking]") {
    using loki::algorithms::detail::ep_ring_forecast;
    using loki::algorithms::detail::ep_ring_margin;
    using loki::algorithms::detail::EPPopulationForecast;
    using loki::algorithms::detail::kEPRingMarginZ;
    // Exact (synthetic) forecasts have no margin.
    const EPPopulationForecast exact{.complexity = {2.0, 3.0}};
    CHECK(ep_ring_margin(exact) == std::vector<double>{1.0, 1.0});

    // eps_s = sqrt(sum_{j<=s} (1 - s_j) / (N s_j)); a pass rate of 1 (or 0)
    // adds nothing.
    const EPPopulationForecast mc{.complexity = {1.0, 0.5, 0.25, 0.25},
                                  .pass_rate  = {0.5, 0.5, 1.0, 0.25},
                                  .ntrials    = 1024};
    const auto margin = ep_ring_margin(mc);
    REQUIRE(margin.size() == 4);
    const double r_half    = 0.5 / (1024.0 * 0.5);
    const double r_quarter = 0.75 / (1024.0 * 0.25);
    const auto near        = [](double a, double b) {
        return std::abs(a - b) < 1e-12;
    };
    CHECK(near(margin[0], 1.0 + (kEPRingMarginZ * std::sqrt(r_half))));
    CHECK(near(margin[1], 1.0 + (kEPRingMarginZ * std::sqrt(2 * r_half))));
    CHECK(margin[2] == margin[1]);
    CHECK(near(margin[3],
               1.0 + (kEPRingMarginZ * std::sqrt((2 * r_half) + r_quarter))));

    // More trials, smaller margin.
    auto mc_large    = mc;
    mc_large.ntrials = 1U << 15U;
    CHECK(ep_ring_margin(mc_large)[3] < margin[3]);
    CHECK(ep_ring_margin(mc_large)[3] > 1.0);

    // L_0 = ncoords exactly; L_s = ceil(ncoords c_s m_s); live is the largest
    // adjacent sum.
    const auto ring = ep_ring_forecast(1000, mc.complexity, margin);
    const double l1 = std::ceil(1000.0 * 1.0 * margin[0]);
    const double l2 = std::ceil(1000.0 * 0.5 * margin[1]);
    CHECK(ring.peak == l1);
    CHECK(ring.live == 1000.0 + l1);
    CHECK(ring.live > l1 + l2);
    CHECK(ring.peak_complexity == 1.0);
}

TEST_CASE("EP chunking preserves forecast capacity above four million",
          "[ep_chunking]") {
    const auto cfg     = make_cfg(140.0, 145.0, 1);
    const auto designs = make_designs(cfg, 5.0e6);
    const auto ctx     = make_ctx(cfg);
    const auto plan = plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, ctx);
    REQUIRE(plan.chunk_cfgs.size() == 1);
    const auto& chunk = plan.chunk_cfgs.front();
    CHECK(chunk.predicted_peak >= 5'000'000);
    CHECK(chunk.predicted_live >= chunk.predicted_peak + chunk.ncoords);
    CHECK(chunk.max_sugg > (1U << 22U));
    CHECK(chunk.max_sugg % loki::algorithms::detail::kEPRingGranularity == 0);
    check_plan_fits(plan, ctx, 8.0);
}

TEST_CASE("EP forecasts account for adjacent populations, once per chunk",
          "[ep_chunking]") {
    const auto cfg       = make_cfg(140.0, 145.0, 1);
    auto design          = make_design(cfg, 140.0, 145.0, 32, 1.0F);
    SizeType evaluations = 0;
    design.design = [&evaluations](std::span<const float> bp, double, double) {
        ++evaluations;
        REQUIRE(bp.size() == 7);
        return loki::algorithms::detail::ChunkDesign{
            .thresholds = std::vector<float>(bp.size(), 0.0F),
            .forecast   = {.complexity = {3.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0}},
        };
    };
    const auto plan = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                         0.0, 8.0, make_ctx(cfg));
    REQUIRE(plan.chunk_cfgs.size() == 1);
    const auto& chunk = plan.chunk_cfgs.front();
    // The branching forecast does not depend on the run, so a chunk is
    // designed once whatever the number of runs.
    CHECK(evaluations == 1);
    CHECK(chunk.predicted_peak == 4 * chunk.ncoords);
    CHECK(chunk.predicted_live == 7 * chunk.ncoords);
    CHECK(chunk.max_sugg >
          chunk.predicted_live + chunk.batch_size * chunk.branch_max);
}

TEST_CASE("CUDA batch keeps the work per batch at 2^22 parent-bins",
          "[ep_chunking]") {
    using loki::algorithms::detail::ep_cuda_batch_for_nbins;
    constexpr SizeType kMax = SizeType{1} << 16U;
    CHECK(ep_cuda_batch_for_nbins(16, kMax) == kMax);
    CHECK(ep_cuda_batch_for_nbins(64, kMax) == kMax);
    CHECK(ep_cuda_batch_for_nbins(128, kMax) == kMax / 2);
    CHECK(ep_cuda_batch_for_nbins(1024, kMax) == 4096);
    // Not a power of two: rounded down.
    CHECK(ep_cuda_batch_for_nbins(1000, kMax) == 4096);
    // The planner's halved batch caps it.
    CHECK(ep_cuda_batch_for_nbins(1024, 1024) == 1024);
}

TEST_CASE("CUDA EP lowers the batch when a minimum chunk cannot fit",
          "[ep_chunking]") {
    const auto cfg     = make_cfg(140.0, 145.0, 1);
    const auto designs = make_designs(cfg, 1000.0);
    auto ctx           = make_cuda_ctx(cfg);
    ctx.batch_size     = 1U << 16U;
    // The planner says once when the batch was lowered, and only then.
    std::ostringstream log;
    const auto logger = spdlog::default_logger();
    auto sink         = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
    logger->sinks().push_back(sink);
    const auto ample =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, ctx);
    const auto ample_log = log.str();
    const auto tight =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, 0.03, ctx);
    std::erase(logger->sinks(), sink);
    CHECK(ample_log.find("CUDA batch is at most") == std::string::npos);
    CHECK(log.str().find("CUDA batch is at most") != std::string::npos);
    REQUIRE(!ample.chunk_cfgs.empty());
    CHECK(ample.chunk_cfgs.front().batch_size == ctx.batch_size);
    REQUIRE(!tight.chunk_cfgs.empty());
    CHECK(tight.chunk_cfgs.front().batch_size < ctx.batch_size);
    CHECK(std::has_single_bit(tight.chunk_cfgs.front().batch_size));
    check_plan_fits(tight, ctx, 0.03);
}

namespace {

/// The per-stage complexity of @p bp: @p scale times its running product,
/// as a schedule with every pass rate equal gives.
std::vector<double> scaled_complexity(std::span<const float> bp, double scale) {
    std::vector<double> c(bp.size());
    double prod = 1.0;
    for (SizeType s = 0; s < bp.size(); ++s) {
        prod *= static_cast<double>(bp[s]);
        c[s] = scale * prod;
    }
    return c;
}

/// A design whose per-seed complexity is @p scale times the running product
/// of the branching pattern it is designed for.
RegionDesign make_scaled_design(const PulsarSearchConfig& cfg,
                                double f_start,
                                double f_end,
                                double scale,
                                SizeType* ndesigns) {
    auto design   = make_design(cfg, f_start, f_end, 32, 1.0F);
    design.design = [=](std::span<const float> bp, double, double) {
        ++*ndesigns;
        return loki::algorithms::detail::ChunkDesign{
            .thresholds = std::vector<float>(bp.size(), 1.0F),
            .pd         = 0.5,
            .pd_se      = 0.01,
            .forecast   = {.complexity = scaled_complexity(bp, scale),
                           .pass_rate  = std::vector<double>(bp.size(), 0.5),
                           .ntrials    = 1U << 15U},
        };
    };
    return design;
}

} // namespace

TEST_CASE("EP chunking designs each region once", "[ep_chunking]") {
    constexpr double kLimitGb = 0.2;
    const auto cfg            = make_cfg(140.0, 145.0, /*nthreads=*/2);
    SizeType ndesigns         = 0;
    const auto design = make_scaled_design(cfg, 100.0, 145.0, 2.0e3, &ndesigns);
    const auto ctx    = make_ctx(cfg);
    const auto plan   = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                           0.0, kLimitGb, ctx);
    REQUIRE(plan.chunk_cfgs.size() >= 2);
    check_plan_fits(plan, ctx, kLimitGb);
    CHECK(ndesigns == 1);
    for (const auto& c : plan.chunk_cfgs) {
        CHECK(c.pd == 0.5);
        CHECK(c.threshold_scheme.size() == c.branching_pattern.size());
        CHECK(c.max_sugg % loki::algorithms::detail::kEPRingGranularity == 0);
    }
}

TEST_CASE("EP chunking carries the region's pass rates to each chunk",
          "[ep_chunking]") {
    // The scaled design's pass rates do not depend on the branching, so a
    // chunk's population from the region design must be the one a design of
    // the chunk's own branching pattern gives.
    constexpr double kScale   = 2.0e3;
    constexpr double kLimitGb = 0.2;
    const auto cfg            = make_cfg(140.0, 145.0, /*nthreads=*/2);
    SizeType ndesigns         = 0;
    const auto design =
        make_scaled_design(cfg, 100.0, 145.0, kScale, &ndesigns);
    const auto ctx  = make_ctx(cfg);
    const auto plan = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                         0.0, kLimitGb, ctx);
    REQUIRE(plan.chunk_cfgs.size() >= 2);
    for (const auto& c : plan.chunk_cfgs) {
        const auto own    = scaled_complexity(c.branching_pattern, kScale);
        const double peak = std::ranges::max(own);
        CHECK(std::abs(c.peak_complexity - peak) <= 1.0e-6 * peak);
    }
}

namespace {

/// Frequency cells of a chunk's seeds.
SizeType n_freq_cells(const loki::algorithms::EPChunkConfig& c) {
    const loki::plans::FFAPlan<float> plan(c.cfg);
    return plan.get_param_counts().back().back();
}

/// A designed region whose population grows as its branching with pass
/// rate @p pass per stage, times @p scale, and whose last stage keeps few
/// leaves, as real schedules do.
RegionDesign make_peaked_design(const PulsarSearchConfig& cfg,
                                double f_start,
                                double f_end,
                                double scale) {
    auto design   = make_design(cfg, f_start, f_end, 32, 1.0F);
    design.design = [=](std::span<const float> bp, double, double) {
        auto complexity   = scaled_complexity(bp, scale);
        complexity.back() = 1.0e-2;
        return loki::algorithms::detail::ChunkDesign{
            .thresholds = std::vector<float>(bp.size(), 1.0F),
            .pd         = 0.5,
            .pd_se      = 0.01,
            .forecast   = {.complexity = std::move(complexity),
                           .pass_rate  = std::vector<double>(bp.size(), 0.5),
                           .ntrials    = 1U << 15U},
        };
    };
    return design;
}

} // namespace

TEST_CASE("EP chunking prunes seeds in slices before narrowing a chunk",
          "[ep_chunking]") {
    const auto cfg    = make_cfg(140.0, 145.0, /*nthreads=*/1);
    const auto design = make_peaked_design(cfg, 140.0, 145.0, 2.0e2);
    const auto ctx    = make_ctx(cfg);
    const auto ample =
        plan_chunks<float>(cfg, "taylor", std::span(&design, 1), 0.0, 8.0, ctx);
    REQUIRE(ample.chunk_cfgs.size() == 1);
    const auto& whole = ample.chunk_cfgs.front();
    CHECK(whole.seed_slice_edges.empty());
    CHECK(whole.held_final_leaves == 0);
    CHECK(ample.chunk_stats.front().n_seed_slices == 1);

    const double limit = 0.5 * ample.peak_memory_gb;
    const auto tight = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                          0.0, limit, ctx);
    check_plan_fits(tight, ctx, limit);
    // The chunk keeps its width: its seeds are pruned in slices instead.
    REQUIRE(tight.chunk_cfgs.size() == 1);
    const auto& c     = tight.chunk_cfgs.front();
    const auto& edges = c.seed_slice_edges;
    REQUIRE(edges.size() >= 3);
    const auto k = edges.size() - 1;
    CHECK_NOTHROW(loki::search::check_seed_slice_edges(edges, n_freq_cells(c)));
    CHECK(tight.chunk_stats.front().n_seed_slices == k);
    CHECK(c.batch_size == whole.batch_size);
    CHECK(c.held_final_leaves > 0);
    CHECK(c.ncoords == whole.ncoords);
    // The largest slice holds at least its share of the whole live ring,
    // and less than all of it.
    CHECK(c.predicted_live < whole.predicted_live);
    CHECK(static_cast<double>(c.predicted_live) >=
          static_cast<double>(whole.predicted_live) / static_cast<double>(k));
    CHECK(c.max_sugg < whole.max_sugg);
    CHECK(k <= 64);
}

TEST_CASE("CUDA EP slices seeds where a smaller batch would cost more",
          "[ep_chunking]") {
    // A small batch: the tree, not the room for one branched batch, binds.
    const auto cfg    = make_cfg(140.0, 145.0, 1);
    const auto design = make_peaked_design(cfg, 140.0, 145.0, 2.0e2);
    auto ctx          = make_cuda_ctx(cfg);
    ctx.batch_size    = 1U << 6U;
    const auto ample =
        plan_chunks<float>(cfg, "taylor", std::span(&design, 1), 0.0, 8.0, ctx);
    REQUIRE(ample.chunk_cfgs.size() == 1);
    REQUIRE(ample.chunk_cfgs.front().seed_slice_edges.empty());
    const double limit = 0.6 * ample.peak_memory_gb;
    const auto tight = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                          0.0, limit, ctx);
    check_plan_fits(tight, ctx, limit);
    REQUIRE(tight.chunk_cfgs.size() == 1);
    const auto& c = tight.chunk_cfgs.front();
    CHECK(c.seed_slice_edges.size() >= 3);
    CHECK(c.batch_size == ample.chunk_cfgs.front().batch_size);
}

TEST_CASE("Branching stages per frequency cell give the branching mean",
          "[ep_chunking]") {
    // The planner forecasts a slice of seeds by summing its cells' leaves
    // per seed; their ratio over the whole grid must be the mean the
    // thresholds are designed for.
    const auto cfg =
        make_cfg(140.0, 145.0, 1).get_updated_ep_config(32, 1.0, 140.0, 145.0);
    const loki::plans::FFAPlan<float> plan(cfg);
    const auto mean = plan.forecast_branching("taylor").mean;
    std::vector<double> sums;
    const auto forecast = loki::plans::detail::simulate_branching_stages(
        plan, "taylor", [&](SizeType s, std::span<const double> w) {
            CHECK(s == sums.size());
            CHECK(w.size() == plan.get_param_counts().back().back());
            sums.push_back(std::accumulate(w.begin(), w.end(), 0.0));
        });
    CHECK(forecast.mean == mean);
    REQUIRE(sums.size() == mean.size());
    double prev = static_cast<double>(plan.get_param_counts().back().back());
    for (SizeType s = 0; s < sums.size(); ++s) {
        CHECK(sums[s] / prev == Catch::Approx(mean[s]).epsilon(1e-12));
        prev = sums[s];
    }
}

namespace {

/// FFA buffers of chunk @p c alone, in GB (the planner's ffa_gb).
double chunk_ffa_gb(const loki::algorithms::EPChunkConfig& c,
                    const EPMemoryContext& ctx) {
    using loki::algorithms::detail::ep_fixed_bytes;
    using loki::algorithms::detail::ep_input_bytes;
    const auto fixed = ep_fixed_bytes<float>(ctx, c.buffer_size, c.coord_size,
                                             c.ffa_transient_bytes);
    return static_cast<double>(fixed - ep_input_bytes(ctx.nsamps)) /
           loki::utils::kBytesPerGB;
}

SizeType n_slices(const loki::algorithms::EPChunkConfig& c) {
    return c.seed_slice_edges.empty() ? 1 : c.seed_slice_edges.size() - 1;
}

} // namespace

TEST_CASE("EP chunking widens a sliced chunk only within the FFA share",
          "[ep_chunking]") {
    // A wide region whose trees need slices: no chunk wider than the
    // minimum may hold slices with FFA buffers above a quarter of the limit.
    const auto cfg    = make_cfg(100.0, 145.0, /*nthreads=*/1);
    const auto design = make_peaked_design(cfg, 100.0, 145.0, 2.0e2);
    const auto ctx    = make_ctx(cfg);
    const auto ample =
        plan_chunks<float>(cfg, "taylor", std::span(&design, 1), 0.0, 8.0, ctx);
    bool sliced = false;
    for (const double share : {0.6, 0.4, 0.3, 0.2}) {
        const double limit = share * ample.peak_memory_gb;
        CAPTURE(limit);
        const auto plan = plan_chunks<float>(
            cfg, "taylor", std::span(&design, 1), 0.0, limit, ctx);
        check_plan_fits(plan, ctx, limit);
        for (const auto& c : plan.chunk_cfgs) {
            if (n_slices(c) == 1 ||
                c.nominal_f_end - c.nominal_f_start <= 2.0e-2) {
                continue;
            }
            sliced = true;
            CHECK(chunk_ffa_gb(c, ctx) <= 0.25 * limit);
            CHECK(n_slices(c) <= 64);
        }
    }
    CHECK(sliced);
}

TEST_CASE("EP chunking slices a minimum-width chunk as much as it needs",
          "[ep_chunking]") {
    // Doppler padding gives a minimum-width chunk many frequency cells; a
    // limit that only fits the tree of a few of them needs more than the 64
    // slices a widening chunk may take.
    const auto cfg          = make_cfg(140.0, 145.0, /*nthreads=*/1);
    const auto design       = make_peaked_design(cfg, 140.0, 145.0, 2.0e2);
    const auto ctx          = make_ctx(cfg);
    constexpr double kDrift = 0.05;
    const auto ample = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                          kDrift, 8.0, ctx);
    REQUIRE(!ample.chunk_cfgs.empty());
    REQUIRE(n_freq_cells(ample.chunk_cfgs.front()) > 64);
    SizeType most = 0;
    for (double limit = ample.peak_memory_gb; limit > 0.0; limit *= 0.97) {
        CAPTURE(limit);
        loki::algorithms::detail::ChunkPlan plan;
        try {
            plan = plan_chunks<float>(cfg, "taylor", std::span(&design, 1),
                                      kDrift, limit, ctx);
        } catch (const std::runtime_error&) {
            break;
        }
        check_plan_fits(plan, ctx, limit);
        for (const auto& c : plan.chunk_cfgs) {
            CHECK_NOTHROW(loki::search::check_seed_slice_edges(
                c.seed_slice_edges, n_freq_cells(c)));
            most = std::max(most, n_slices(c));
        }
        if (most > 64) {
            break;
        }
    }
    CHECK(most > 64);
}

TEST_CASE("CUDA EP halves the batch only when that forecasts a faster search",
          "[ep_chunking]") {
    const auto cfg    = make_cfg(140.0, 145.0, 1);
    const auto design = make_peaked_design(cfg, 140.0, 145.0, 2.0e2);
    auto ctx          = make_cuda_ctx(cfg);
    ctx.batch_size    = 1U << 12U;
    std::ostringstream log;
    const auto logger = spdlog::default_logger();
    auto sink         = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
    logger->sinks().push_back(sink);
    const auto ample =
        plan_chunks<float>(cfg, "taylor", std::span(&design, 1), 0.0, 8.0, ctx);
    const auto ample_log = log.str();
    bool chosen          = false;
    for (const double share : {0.8, 0.6, 0.5, 0.4, 0.3}) {
        log.str("");
        const double limit = share * ample.peak_memory_gb;
        CAPTURE(limit);
        const auto plan = plan_chunks<float>(
            cfg, "taylor", std::span(&design, 1), 0.0, limit, ctx);
        check_plan_fits(plan, ctx, limit);
        const auto text = log.str();
        const auto at   = text.find("saves seed slices (forecast search ");
        if (at == std::string::npos) {
            continue;
        }
        // "... search <new> s against <old> s per run": the new is faster.
        double faster = 0.0;
        double slower = 0.0;
        std::istringstream in(text.substr(at + 35));
        std::string word;
        in >> faster >> word >> word >> slower;
        CHECK(faster < slower);
        CHECK(plan.chunk_cfgs.front().batch_size < ctx.batch_size);
        chosen = true;
    }
    std::erase(logger->sinks(), sink);
    // Without slices the batch stays.
    CHECK(ample_log.find("saves seed slices") == std::string::npos);
    CHECK(ample.chunk_cfgs.front().batch_size == ctx.batch_size);
    CHECK(chosen);
}

TEST_CASE("EP chunking starts a group at the branch_max a later chunk needs",
          "[ep_chunking]") {
    // Two regions of one nbins: the first (branch_max 3 near 100 Hz) fills
    // the limit with its tree; a chunk of the second (branch_max 9 near
    // 400 Hz) fits alone but not next to that tree with its larger branch
    // workspace. The planner replans the group from that branch_max instead
    // of lowering the CUDA batch for every chunk.
    const auto cfg = loki::test::seed_slice_config(/*nparams=*/5,
                                                   /*fourier=*/false,
                                                   /*nthreads=*/1);
    const std::vector<RegionDesign> designs{
        make_peaked_design(cfg, 98.0, 100.0, 2.0e2),
        make_peaked_design(cfg, 400.0, 402.0, 2.0e2)};
    for (const bool cuda : {false, true}) {
        CAPTURE(cuda);
        auto ctx = cuda ? make_cuda_ctx(cfg) : make_ctx(cfg);
        if (cuda) {
            ctx.batch_size = 1U << 10U;
        }
        const auto ample =
            plan_chunks<float>(cfg, "taylor", designs, 0.0, 8.0, ctx);
        REQUIRE(ample.chunk_cfgs.size() == 2);
        REQUIRE(ample.chunk_cfgs.front().branch_max <
                ample.chunk_cfgs.back().branch_max);
        std::ostringstream log;
        const auto logger = spdlog::default_logger();
        auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log);
        logger->sinks().push_back(sink);
        bool floored = false;
        for (double limit = ample.peak_memory_gb; limit > 0.0 && !floored;
             limit *= 0.97) {
            CAPTURE(limit);
            log.str("");
            loki::algorithms::detail::ChunkPlan plan;
            try {
                plan =
                    plan_chunks<float>(cfg, "taylor", designs, 0.0, limit, ctx);
            } catch (const std::runtime_error&) {
                break;
            }
            if (log.str().find("raises the branch_max") == std::string::npos) {
                continue;
            }
            floored = true;
            check_plan_fits(plan, ctx, limit);
            for (const auto& c : plan.chunk_cfgs) {
                CHECK(c.batch_size == ample.chunk_cfgs.front().batch_size);
            }
        }
        std::erase(logger->sinks(), sink);
        CHECK(floored);
    }
}
