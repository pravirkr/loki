#include "lib/algorithms/ep_chunking.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::generate_ffa_regions;
using loki::algorithms::detail::ep_sweep_peak_gb;
using loki::algorithms::detail::EPHarvestBound;
using loki::algorithms::detail::EPMemoryContext;
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
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*bseg_ffa=*/kNsamps / 8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

/// One design per FFA region. The first region needs @p first_max_sugg
/// suggestions for a chunk spanning the whole region; the others the minimum.
std::vector<RegionDesign> make_designs(const PulsarSearchConfig& cfg,
                                       double first_max_sugg) {
    const auto regions =
        generate_ffa_regions(1.0 / cfg.get_f_max(), 1.0 / cfg.get_f_min(),
                             cfg.get_tsamp(), cfg.get_nbins(), cfg.get_eta(),
                             cfg.get_octave_scale(), cfg.get_nbins_max());
    std::vector<RegionDesign> designs;
    for (const auto& r : regions) {
        const auto rcfg =
            cfg.get_updated_ep_config(r.nbins, r.eta, r.f_start, r.f_end);
        const loki::plans::FFAPlan<float> plan(rcfg);
        const auto ncoords = static_cast<double>(plan.get_ncoords().back());
        designs.push_back(RegionDesign{
            .f_start          = r.f_start,
            .f_end            = r.f_end,
            .nbins            = r.nbins,
            .eta              = r.eta,
            .threshold_scheme = {1.0F},
            .bp_float         = {1.0F},
            .peak_complexity  = 1.0F,
            .safe_complexity =
                designs.empty() ? static_cast<float>(first_max_sugg / ncoords)
                                : 1.0F,
            .nsegments = plan.get_nsegments().back(),
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

/// One synthetic design over [f_start, f_end] with a chosen complexity.
RegionDesign make_design(const PulsarSearchConfig& cfg,
                         double f_start,
                         double f_end,
                         SizeType nbins,
                         float safe_complexity) {
    const auto rcfg = cfg.get_updated_ep_config(nbins, 1.0, f_start, f_end);
    const loki::plans::FFAPlan<float> plan(rcfg);
    return RegionDesign{
        .f_start          = f_start,
        .f_end            = f_end,
        .nbins            = nbins,
        .eta              = 1.0,
        .threshold_scheme = {1.0F},
        .bp_float         = {1.0F},
        .peak_complexity  = 1.0F,
        .safe_complexity  = safe_complexity,
        .nsegments        = plan.get_nsegments().back(),
    };
}

/// A heavy run (large max_sugg, nbins 32) followed by a light run (nbins
/// 128) whose wide band sets a large shared FFA size only at the end.
std::vector<RegionDesign>
make_late_shared_designs(const PulsarSearchConfig& cfg) {
    return {make_design(cfg, 100.0, 110.0, 32, 1.0e5F),
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
    constexpr double kLimitGb = 0.2;
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
    constexpr double kLimitGb = 0.16;
    const auto cfg            = make_cfg(30.0, 150.0, /*nthreads=*/2);
    const auto designs        = make_late_shared_designs(cfg);
    const auto ctx            = make_ctx(cfg);

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
    constexpr double kLimitGb = 0.12;
    const auto cfg            = make_cfg(30.0, 150.0, /*nthreads=*/2);
    const auto designs        = make_late_shared_designs(cfg);
    const auto ctx            = make_ctx(cfg);

    const auto plan =
        plan_chunks<float>(cfg, "taylor", designs, 0.0, kLimitGb, ctx);
    CHECK(plan.replanned);
    CHECK(plan.shared_cap_scale < 1.0);
    CHECK(count_nbins(plan, designs[1].nbins) >= 2);
    check_plan_fits(plan, ctx, kLimitGb);
}
