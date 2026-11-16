#include "ep_seed_slices.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <hdf5.h>
#include <highfive/highfive.hpp>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/prune_engine.hpp"
#include "lib/search/candidate_processing.hpp"
#include "lib/search/cands.hpp"
#include "lib/search/prune_mask.hpp"

#include "temp_dir.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::ParamWindow;
using loki::algorithms::PruneRFIConfig;
using loki::search::GridMask;
using loki::search::PruneStats;
using loki::search::PruneStatsCollection;
using loki::search::SeedSlice;
using loki::search::detail::LeafReductionSpec;
using loki::test::read_run_results;
using loki::test::RunResults;
using loki::test::TempDir;

namespace {

SizeType n_freq_cells(const loki::search::PulsarSearchConfig& cfg) {
    const loki::plans::FFAPlan<float> plan(cfg);
    return plan.get_param_counts().back().back();
}

/// Prunes @p cfg on the CPU with the seed slices between @p edges (and the
/// per-run leaf reduction @p reduction) and returns every run of the result
/// file.
std::vector<RunResults>
run_cpu_edges(const loki::search::PulsarSearchConfig& cfg,
              const std::vector<float>& ts_e,
              const std::vector<float>& ts_v,
              const std::vector<SizeType>& edges,
              const PruneRFIConfig& rfi                         = {},
              const std::optional<LeafReductionSpec>& reduction = {}) {
    const TempDir outdir(fmt::format("loki-seed-slices-{}", edges.size()));
    const auto thresholds = loki::test::seed_slice_thresholds(cfg);
    const auto run        = [&]<typename FoldType>() {
        auto engine = loki::algorithms::detail::make_ep_cpu<FoldType>(
            cfg, thresholds, /*n_runs=*/2, /*ref_segs=*/std::nullopt,
            /*ascend_levels=*/{}, /*max_sugg=*/1U << 16U, /*batch_size=*/64,
            "taylor", /*show_progress=*/false, rfi);
        engine->set_seed_slices(edges);
        engine->set_leaf_reduction(reduction);
        engine->execute(ts_e, ts_v, outdir.path(), "slices");
    };
    if (cfg.get_use_fourier()) {
        run.template operator()<loki::ComplexType>();
    } else {
        run.template operator()<float>();
    }
    return read_run_results(outdir.path());
}

/// As run_cpu_edges, with @p n_slices even slices (clamped to the cells).
std::vector<RunResults> run_cpu(const loki::search::PulsarSearchConfig& cfg,
                                const std::vector<float>& ts_e,
                                const std::vector<float>& ts_v,
                                SizeType n_slices,
                                const PruneRFIConfig& rfi = {}) {
    return run_cpu_edges(
        cfg, ts_e, ts_v,
        loki::search::even_seed_slice_edges(n_slices, n_freq_cells(cfg)), rfi);
}

} // namespace

TEST_CASE("Seed slice edges split the frequency cells evenly and in order",
          "[ep][seed_slices]") {
    using loki::search::check_seed_slice_edges;
    using loki::search::even_seed_slice_edges;
    using Edges = std::vector<SizeType>;
    CHECK(even_seed_slice_edges(0, 10) == Edges{0, 10});
    CHECK(even_seed_slice_edges(4, 10) == Edges{0, 2, 5, 7, 10});
    CHECK(even_seed_slice_edges(40, 3) == Edges{0, 1, 2, 3});
    CHECK(even_seed_slice_edges(3, 0) == Edges{0, 0});
    for (const SizeType n_freq : {SizeType{1}, SizeType{7}, SizeType{64}}) {
        for (SizeType k = 1; k <= n_freq; ++k) {
            const auto edges = even_seed_slice_edges(k, n_freq);
            REQUIRE(edges.size() == k + 1);
            CHECK_NOTHROW(check_seed_slice_edges(edges, n_freq));
            for (SizeType s = 0; s < k; ++s) {
                CHECK(edges[s + 1] - edges[s] <= (n_freq / k) + 1);
            }
        }
    }
    CHECK_NOTHROW(check_seed_slice_edges({}, 10));
    CHECK_NOTHROW(check_seed_slice_edges(Edges{0, 1, 9, 10}, 10));
    for (const auto& bad : {Edges{0}, Edges{1, 10}, Edges{0, 9}, Edges{0, 11},
                            Edges{0, 5, 5, 10}, Edges{0, 6, 4, 10}}) {
        CHECK_THROWS_AS(check_seed_slice_edges(bad, 10), std::invalid_argument);
    }
}

TEST_CASE("Slice seeds cover every unmasked seed exactly once",
          "[ep][seed_slices]") {
    constexpr SizeType kNAccel = 5;
    constexpr SizeType kNFreq  = 11;
    GridMask mask(ParamLimit{.min = -1.0, .max = 1.0}, kNAccel,
                  ParamLimit{.min = 0.0, .max = 11.0}, kNFreq);
    // No geometry: an empty default mask masks nothing.
    const GridMask none;
    std::vector<SizeType> keep(kNAccel * kNFreq);
    const auto n_all = loki::search::select_slice_seeds(
        keep, kNAccel, kNFreq, {.f_begin = 0, .f_end = kNFreq}, none);
    CHECK(n_all == kNAccel * kNFreq);
    for (SizeType i = 0; i < n_all; ++i) {
        CHECK(keep[i] == i);
    }

    mask.add_window(ParamWindow{.f_lo = 2.2, .f_hi = 4.8});
    std::vector<SizeType> mask_keep(kNAccel * kNFreq);
    const auto n_unmasked = mask.select_seeds(mask_keep, kNAccel * kNFreq);
    REQUIRE(n_unmasked < kNAccel * kNFreq);
    for (const SizeType k : {SizeType{1}, SizeType{3}, kNFreq}) {
        std::set<SizeType> seen;
        const auto edges = loki::search::even_seed_slice_edges(k, kNFreq);
        for (SizeType s = 0; s < k; ++s) {
            const SeedSlice slice{.f_begin = edges[s], .f_end = edges[s + 1]};
            const auto n = loki::search::select_slice_seeds(
                keep, kNAccel, kNFreq, slice, mask);
            for (SizeType i = 0; i < n; ++i) {
                const auto f = keep[i] % kNFreq;
                CHECK(f >= slice.f_begin);
                CHECK(f < slice.f_end);
                CHECK_FALSE(mask.is_masked(keep[i]));
                CHECK(seen.insert(keep[i]).second);
                if (i > 0) {
                    CHECK(keep[i] > keep[i - 1]);
                }
            }
        }
        CHECK(seen ==
              std::set<SizeType>(mask_keep.begin(),
                                 mask_keep.begin() +
                                     static_cast<std::ptrdiff_t>(n_unmasked)));
    }
}

TEST_CASE("Slice stats merge by level into the stats of one prune",
          "[ep][seed_slices]") {
    // Slice A lives to level 2, slice B dies at level 1, scoring nothing.
    PruneStatsCollection a;
    a.update_stats(PruneStats{.level         = 0,
                              .score_min     = 1.0F,
                              .score_max     = 5.0F,
                              .n_branches    = 6,
                              .n_leaves      = 6,
                              .n_leaves_phy  = 6,
                              .n_leaves_surv = 6});
    a.update_stats(PruneStats{.level             = 1,
                              .threshold         = 2.0F,
                              .score_min         = 0.5F,
                              .score_max         = 6.0F,
                              .n_branches        = 6,
                              .n_leaves          = 12,
                              .n_leaves_resolved = 11,
                              .n_leaves_phy      = 10,
                              .n_leaves_surv     = 4,
                              .n_overloads       = 1});
    a.update_stats(PruneStats{.level             = 2,
                              .threshold         = 3.0F,
                              .score_min         = 2.5F,
                              .score_max         = 7.0F,
                              .n_branches        = 4,
                              .n_leaves          = 8,
                              .n_leaves_resolved = 8,
                              .n_leaves_phy      = 8,
                              .n_leaves_surv     = 3});
    PruneStatsCollection b;
    b.update_stats(PruneStats{.level           = 0,
                              .score_min       = 0.25F,
                              .score_max       = 3.0F,
                              .n_branches      = 4,
                              .n_leaves        = 4,
                              .n_leaves_phy    = 2,
                              .n_leaves_surv   = 2,
                              .n_leaves_masked = 2});
    b.update_stats(PruneStats{.level             = 1,
                              .threshold         = 2.0F,
                              .score_min         = 0.0F,
                              .score_max         = 0.0F,
                              .n_branches        = 2,
                              .n_leaves          = 4,
                              .n_leaves_resolved = 4,
                              .n_leaves_phy      = 0,
                              .n_leaves_surv     = 0,
                              .n_leaves_masked   = 4});

    for (const bool a_first : {true, false}) {
        PruneStatsCollection merged;
        merged.merge_slice(a_first ? a : b);
        merged.merge_slice(a_first ? b : a);
        REQUIRE(merged.get_nstages() == 3);
        const auto s0 = *merged.get_stats(0);
        CHECK(s0.score_min == 0.25F);
        CHECK(s0.score_max == 5.0F);
        CHECK(s0.n_branches == 10);
        CHECK(s0.n_leaves == 10);
        CHECK(s0.n_leaves_resolved == 1); // seeding resolves nothing
        CHECK(s0.n_leaves_phy == 8);
        CHECK(s0.n_leaves_surv == 8);
        CHECK(s0.n_leaves_masked == 2);
        const auto s1 = *merged.get_stats(1);
        CHECK(s1.threshold == 2.0F);
        CHECK(s1.score_min == 0.5F); // B scored nothing at level 1
        CHECK(s1.score_max == 6.0F);
        CHECK(s1.n_branches == 8);
        CHECK(s1.n_leaves == 16);
        CHECK(s1.n_leaves_resolved == 15);
        CHECK(s1.n_leaves_phy == 10);
        CHECK(s1.n_leaves_surv == 4);
        CHECK(s1.n_leaves_masked == 4);
        CHECK(s1.n_overloads == 1);
        const auto s2 = *merged.get_stats(2);
        CHECK(s2.score_min == 2.5F);
        CHECK(s2.n_leaves_surv == 3);
        const auto rows = merged.get_packed_data().first;
        CHECK(std::ranges::is_sorted(rows, {}, &PruneStats::level));
    }
}

TEST_CASE("CPU EP: seed slices give the leaves and stats of one prune",
          "[ep][seed_slices]") {
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/false,
                                                            /*nthreads=*/2);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    const auto n_freq       = n_freq_cells(cfg);
    REQUIRE(n_freq > 8);
    const auto whole = run_cpu(cfg, ts_e, ts_v, 1);
    REQUIRE(whole.size() == 2);
    for (const auto& run : whole) {
        REQUIRE(run.n_leaves() > 0);
        CHECK(run.total_overloads() == 0);
    }
    // Slices that do not divide the cells, and more slices than cells.
    for (const SizeType k : {SizeType{3}, SizeType{7}, n_freq + 5}) {
        CAPTURE(k);
        const auto sliced = run_cpu(cfg, ts_e, ts_v, k);
        loki::test::check_same_runs(whole, sliced);
    }
    // Uneven slices, as a planner balancing the tree of each slice sets.
    loki::test::check_same_runs(
        whole, run_cpu_edges(cfg, ts_e, ts_v,
                             {0, 1, 2, n_freq / 2, n_freq - 1, n_freq}));
    const auto engine = loki::algorithms::detail::make_ep_cpu<float>(
        cfg, loki::test::seed_slice_thresholds(cfg), 2, std::nullopt, {},
        1U << 16U, 256, "taylor", false, {});
    CHECK_THROWS_AS(engine->set_seed_slices({0, n_freq + 1}),
                    std::invalid_argument);
    CHECK_THROWS_AS(engine->set_seed_slices({0, 4, 4, n_freq}),
                    std::invalid_argument);
}

TEST_CASE("CPU EP: seed slices keep the per-run leaf reduction",
          "[ep][seed_slices]") {
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/false,
                                                            /*nthreads=*/1);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    auto options            = loki::search::CandidateOptions::ep_defaults();
    options.max_per_cell    = 2; // small enough that cells drop leaves
    const LeafReductionSpec spec{
        .options = options,
        .snr_min = 4.0,
        .tobs    = cfg.get_tobs(),
        .nbins   = static_cast<std::uint32_t>(cfg.get_nbins())};
    const auto whole = run_cpu_edges(cfg, ts_e, ts_v, {}, {}, spec);
    REQUIRE(whole.size() == 2);
    REQUIRE(whole.front().reduction.size() == 5);
    REQUIRE(whole.front().reduction[2] > 0); // cells dropped leaves
    // The same retained leaves and counts (leaf order differs, so an exact
    // tie could pick another representative; this input has none).
    loki::test::check_same_runs(
        whole,
        run_cpu_edges(cfg, ts_e, ts_v,
                      loki::search::even_seed_slice_edges(7, n_freq_cells(cfg)),
                      {}, spec));
}

TEST_CASE("CPU EP: seed slices are exact in the Fourier domain",
          "[ep][seed_slices]") {
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/true,
                                                            /*nthreads=*/2);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    const auto whole        = run_cpu(cfg, ts_e, ts_v, 1);
    REQUIRE(whole.size() == 2);
    REQUIRE(whole.front().n_leaves() > 0);
    CHECK(whole.front().total_overloads() == 0);
    loki::test::check_same_runs(whole, run_cpu(cfg, ts_e, ts_v, 3));
}

TEST_CASE("CPU EP: seed slices are exact under a static pulsar mask",
          "[ep][seed_slices]") {
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/false,
                                                            /*nthreads=*/1);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    // One slice per cell: the cells under the window are fully masked
    // slices.
    PruneRFIConfig rfi;
    rfi.pulsar_mask  = {ParamWindow{.f_lo = 140.4, .f_hi = 140.7}};
    const auto whole = run_cpu(cfg, ts_e, ts_v, 1, rfi);
    REQUIRE(whole.size() == 2);
    REQUIRE(whole.front().total_masked() > 0);
    for (const SizeType k : {SizeType{4}, n_freq_cells(cfg)}) {
        CAPTURE(k);
        loki::test::check_same_runs(whole, run_cpu(cfg, ts_e, ts_v, k, rfi));
    }
}

TEST_CASE("CPU EP: seed slices are exact in jerk and circular searches",
          "[ep][seed_slices]") {
    for (const SizeType nparams : {SizeType{3}, SizeType{5}}) {
        CAPTURE(nparams);
        const auto cfg = loki::test::seed_slice_config(nparams,
                                                       /*fourier=*/false,
                                                       /*nthreads=*/2);
        const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
        const auto whole        = run_cpu(cfg, ts_e, ts_v, 1);
        REQUIRE(whole.front().n_leaves() > 0);
        loki::test::check_same_runs(whole, run_cpu(cfg, ts_e, ts_v, 5));
    }
}

TEST_CASE("CPU EP: an early harvest carries across the slices of a run",
          "[ep][seed_slices]") {
    const auto cfg          = loki::test::seed_slice_config(/*nparams=*/2,
                                                            /*fourier=*/false,
                                                            /*nthreads=*/1);
    const auto [ts_e, ts_v] = loki::test::seed_slice_series(cfg);
    const auto thresholds   = loki::test::seed_slice_thresholds(cfg);
    // Harvest the injected pulsar from level 4.
    PruneRFIConfig rfi;
    rfi.harvest_scheme.assign(thresholds.size(), 1.0e6F);
    for (SizeType level = 4; level <= thresholds.size(); ++level) {
        rfi.harvest_scheme[level - 1] = 14.0F;
    }
    rfi.harvest_mask_ntiles = 2.0;
    const auto whole        = run_cpu(cfg, ts_e, ts_v, 1, rfi);
    REQUIRE(whole.front().n_harvested_total > 0);
    // One slice per cell: the pulsar's seeds sit in several slices, but the
    // mask its first harvest widened keeps the later slices from harvesting
    // it again.
    const auto sliced = run_cpu(cfg, ts_e, ts_v, n_freq_cells(cfg), rfi);
    REQUIRE(sliced.size() == whole.size());
    for (SizeType r = 0; r < whole.size(); ++r) {
        CHECK(sliced[r].n_harvested_total == whole[r].n_harvested_total);
    }
}
