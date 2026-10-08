#include "lib/algorithms/ep_chunking.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/algorithms/regions.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/ep_memory.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::generate_ffa_regions;
using loki::algorithms::detail::ep_sweep_peak_gb;
using loki::algorithms::detail::kEPBatchSize;
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

    const auto plan = plan_chunks<float>(cfg, "taylor", designs,
                                         /*max_drift=*/0.0, kLimitGb);

    // The limit binds in the first region, not in the second.
    CHECK(count_nbins(plan, designs[0].nbins) >= 2);
    CHECK(count_nbins(plan, designs[1].nbins) == 1);

    CHECK(plan.peak_memory_gb <= kLimitGb);
    CHECK(plan.peak_memory_gb ==
          ep_sweep_peak_gb<float>(plan.chunk_cfgs, cfg.get_nthreads(),
                                  cfg.get_nparams(), kEPBatchSize));
    // The shared maxima cover every chunk.
    for (const auto& c : plan.chunk_cfgs) {
        CHECK(c.buffer_size <= plan.buffer_size);
        CHECK(c.coord_size <= plan.coord_size);
        CHECK(c.fold_size <= plan.fold_size);
    }
}

TEST_CASE("EP chunking covers every region without gaps", "[ep_chunking]") {
    const auto cfg     = make_cfg(70.0, 145.0, /*nthreads=*/2);
    const auto designs = make_designs(cfg, /*first_max_sugg=*/2.0e6);
    const auto plan    = plan_chunks<float>(cfg, "taylor", designs,
                                            /*max_drift=*/0.0, 0.2);
    double covered     = 0.0;
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
        plan_chunks<float>(cfg, "taylor", designs, /*max_drift=*/0.0, 1.0e-4),
        Catch::Matchers::ContainsSubstring("Cannot fit minimum viable chunk") &&
            Catch::Matchers::ContainsSubstring("Required memory"));
}
