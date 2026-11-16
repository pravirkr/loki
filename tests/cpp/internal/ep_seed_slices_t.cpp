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
#include "loki/search/configs.hpp"

#include "lib/algorithms/prune_engine.hpp"
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
using loki::test::read_run_results;
using loki::test::RunResults;
using loki::test::TempDir;

namespace {

/// Prunes @p cfg on the CPU with @p n_slices seed slices and returns every
/// run of the result file.
std::vector<RunResults> run_cpu(const loki::search::PulsarSearchConfig& cfg,
                                const std::vector<float>& ts_e,
                                const std::vector<float>& ts_v,
                                SizeType n_slices,
                                const PruneRFIConfig& rfi = {}) {
    const TempDir outdir(fmt::format("loki-seed-slices-{}", n_slices));
    const auto thresholds = loki::test::seed_slice_thresholds(cfg);
    const auto run        = [&]<typename FoldType>() {
        auto engine = loki::algorithms::detail::make_ep_cpu<FoldType>(
            cfg, thresholds, /*n_runs=*/2, /*ref_segs=*/std::nullopt,
            /*ascend_levels=*/{}, /*max_sugg=*/1U << 16U, /*batch_size=*/64,
            "taylor", /*show_progress=*/false, rfi);
        engine->set_seed_slices(n_slices);
        engine->execute(ts_e, ts_v, outdir.path(), "slices");
    };
    if (cfg.get_use_fourier()) {
        run.template operator()<loki::ComplexType>();
    } else {
        run.template operator()<float>();
    }
    return read_run_results(outdir.path());
}

SizeType n_freq_cells(const loki::search::PulsarSearchConfig& cfg) {
    const loki::plans::FFAPlan<float> plan(cfg);
    return plan.get_param_counts().back().back();
}

} // namespace

TEST_CASE("Seed slices split the frequency cells evenly and in order",
          "[ep][seed_slices]") {
    CHECK(loki::search::clamp_seed_slices(0, 10) == 1);
    CHECK(loki::search::clamp_seed_slices(4, 10) == 4);
    CHECK(loki::search::clamp_seed_slices(40, 10) == 10);
    CHECK(loki::search::clamp_seed_slices(3, 0) == 1);
    for (const SizeType n_freq : {SizeType{1}, SizeType{7}, SizeType{64}}) {
        for (SizeType k = 1; k <= n_freq; ++k) {
            SizeType next = 0;
            for (SizeType s = 0; s < k; ++s) {
                const auto slice = loki::search::seed_slice(s, k, n_freq);
                CHECK(slice.f_begin == next);
                CHECK(slice.f_end > slice.f_begin);
                CHECK(slice.f_end - slice.f_begin <= (n_freq / k) + 1);
                next = slice.f_end;
            }
            CHECK(next == n_freq);
        }
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
        for (SizeType s = 0; s < k; ++s) {
            const auto slice = loki::search::seed_slice(s, k, kNFreq);
            const auto n     = loki::search::select_slice_seeds(
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
    CHECK_THROWS(loki::algorithms::detail::make_ep_cpu<float>(
                     cfg, loki::test::seed_slice_thresholds(cfg), 2,
                     std::nullopt, {}, 1U << 16U, 256, "taylor", false, {})
                     ->set_seed_slices(0));
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
