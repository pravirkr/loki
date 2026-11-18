#include "ep_seed_slices.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

#include "lib/pipelines/ep_sweep_common.hpp"
#include "lib/search/cands.hpp"
#include "lib/search/slice_scratch.hpp"

namespace loki::test {

namespace {

/// Member @p name of every row of a compound dataset.
template <typename T>
std::vector<T>
read_member(const HighFive::DataSet& ds, const char* name, hid_t type) {
    std::vector<T> out(ds.getElementCount());
    const hid_t member = H5Tcreate(H5T_COMPOUND, sizeof(T));
    H5Tinsert(member, name, 0, type);
    const herr_t status =
        H5Dread(ds.getId(), member, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Tclose(member);
    REQUIRE(status >= 0);
    return out;
}

std::vector<search::PruneStats> read_level_stats(const HighFive::DataSet& ds) {
    const auto u64 = [&](const char* name) {
        return read_member<std::uint64_t>(ds, name, H5T_NATIVE_UINT64);
    };
    const auto f32 = [&](const char* name) {
        return read_member<float>(ds, name, H5T_NATIVE_FLOAT);
    };
    const auto level             = u64("level");
    const auto seg_idx           = u64("seg_idx");
    const auto threshold         = f32("threshold");
    const auto score_min         = f32("score_min");
    const auto score_max         = f32("score_max");
    const auto n_branches        = u64("n_branches");
    const auto n_leaves          = u64("n_leaves");
    const auto n_leaves_resolved = u64("n_leaves_resolved");
    const auto n_leaves_phy      = u64("n_leaves_phy");
    const auto n_leaves_surv     = u64("n_leaves_surv");
    const auto n_leaves_masked   = u64("n_leaves_masked");
    const auto n_leaves_vetoed   = u64("n_leaves_vetoed");
    const auto n_harvested       = u64("n_harvested");
    const auto n_overloads       = u64("n_overloads");
    std::vector<search::PruneStats> rows(level.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i] = {.level             = level[i],
                   .seg_idx           = seg_idx[i],
                   .threshold         = threshold[i],
                   .score_min         = score_min[i],
                   .score_max         = score_max[i],
                   .n_branches        = n_branches[i],
                   .n_leaves          = n_leaves[i],
                   .n_leaves_resolved = n_leaves_resolved[i],
                   .n_leaves_phy      = n_leaves_phy[i],
                   .n_leaves_surv     = n_leaves_surv[i],
                   .n_leaves_masked   = n_leaves_masked[i],
                   .n_leaves_vetoed   = n_leaves_vetoed[i],
                   .n_harvested       = n_harvested[i],
                   .n_overloads       = n_overloads[i]};
    }
    return rows;
}

/// Sorted leaves from flat rows of @p stride parameters each.
std::vector<SliceLeaf> make_leaves(const std::vector<double>& params,
                                   const std::vector<float>& scores,
                                   const std::vector<float>& scores_ep,
                                   const std::vector<std::uint32_t>& widths) {
    const auto n = scores.size();
    REQUIRE(scores_ep.size() == n);
    REQUIRE((widths.empty() || widths.size() == n));
    REQUIRE((n == 0 || params.size() % n == 0));
    const auto stride = n == 0 ? 0 : params.size() / n;
    std::vector<SliceLeaf> leaves(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto first =
            params.begin() + static_cast<std::ptrdiff_t>(i * stride);
        leaves[i] = {
            .params   = {first, first + static_cast<std::ptrdiff_t>(stride)},
            .score    = scores[i],
            .score_ep = scores_ep[i],
            .width    = widths.empty() ? 0U : widths[i]};
    }
    std::ranges::sort(leaves, [](const SliceLeaf& a, const SliceLeaf& b) {
        return std::tie(a.params, a.score, a.score_ep, a.width) <
               std::tie(b.params, b.score, b.score_ep, b.width);
    });
    return leaves;
}

void check_same_stats(const search::PruneStats& a,
                      const search::PruneStats& b) {
    CAPTURE(a.level);
    CHECK(a.level == b.level);
    CHECK(a.seg_idx == b.seg_idx);
    CHECK(a.threshold == b.threshold);
    CHECK(a.score_min == b.score_min);
    CHECK(a.score_max == b.score_max);
    CHECK(a.n_branches == b.n_branches);
    CHECK(a.n_leaves == b.n_leaves);
    CHECK(a.n_leaves_resolved == b.n_leaves_resolved);
    CHECK(a.n_leaves_phy == b.n_leaves_phy);
    CHECK(a.n_leaves_surv == b.n_leaves_surv);
    CHECK(a.n_leaves_masked == b.n_leaves_masked);
    CHECK(a.n_leaves_vetoed == b.n_leaves_vetoed);
    CHECK(a.n_harvested == b.n_harvested);
    CHECK(a.n_overloads == b.n_overloads);
}

} // namespace

bool SliceLeaf::operator==(const SliceLeaf& other) const = default;

SizeType RunResults::total_overloads() const noexcept {
    SizeType n = 0;
    for (const auto& s : stats) {
        n += s.n_overloads;
    }
    return n;
}

SizeType RunResults::total_masked() const noexcept {
    SizeType n = 0;
    for (const auto& s : stats) {
        n += s.n_leaves_masked;
    }
    return n;
}

search::PulsarSearchConfig seed_slice_config(SizeType nparams,
                                             bool fourier,
                                             int nthreads,
                                             double accel_max,
                                             SizeType nsegments) {
    std::vector<ParamLimit> limits{{.min = -accel_max, .max = accel_max},
                                   {.min = 140.0, .max = 160.0}};
    if (nparams >= 3) {
        limits.insert(limits.begin(),
                      ParamLimit{.min = -5000.0, .max = 5000.0});
    }
    if (nparams == 5) {
        // Circular: snap and crackle.
        limits.insert(limits.begin(),
                      {ParamLimit{.min = -3000.0, .max = 3000.0},
                       ParamLimit{.min = -3.0e4, .max = 3.0e4}});
    }
    REQUIRE(limits.size() == nparams);
    return {/*nsamps=*/1U << 18U,
            /*tsamp=*/64e-6,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/fourier,
            /*nthreads=*/nthreads,
            /*max_process_memory_gb=*/4.0,
            /*region_growth_factor=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            nsegments,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/nparams,
            /*p_orb_min=*/1e-5,
            /*m_c_max=*/10.0};
}

std::pair<std::vector<float>, std::vector<float>>
seed_slice_series(const search::PulsarSearchConfig& cfg) {
    const auto nsamps = cfg.get_nsamps();
    const auto tsamp  = cfg.get_tsamp();
    std::vector<float> ts_e(nsamps);
    std::vector<float> ts_v(nsamps, 1.0F);
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    constexpr double kFreq = 143.1;
    for (SizeType i = 0; i < nsamps; ++i) {
        const double t     = static_cast<double>(i) * tsamp;
        const double phase = (t * kFreq) - std::floor(t * kFreq);
        ts_e[i]            = noise(rng) + (phase < 0.05 ? 0.6F : 0.0F);
    }
    return {std::move(ts_e), std::move(ts_v)};
}

std::vector<float>
seed_slice_thresholds(const search::PulsarSearchConfig& cfg) {
    const auto n_levels = cfg.get_nsegments() - 1;
    std::vector<float> thresholds(n_levels);
    for (SizeType i = 0; i < n_levels; ++i) {
        thresholds[i] = 2.0F + (0.1F * static_cast<float>(i));
    }
    return thresholds;
}

std::vector<RunResults> read_run_results(const std::filesystem::path& dir) {
    std::optional<std::filesystem::path> file;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.ends_with("_results.h5")) {
            REQUIRE_FALSE(file.has_value());
            file = entry.path();
        }
    }
    REQUIRE(file.has_value());
    const HighFive::File h5(file->string(), HighFive::File::ReadOnly);
    const auto runs_group = h5.getGroup("runs");
    auto names            = runs_group.listObjectNames();
    std::ranges::sort(names);
    std::vector<RunResults> runs;
    for (const auto& name : names) {
        const auto run = runs_group.getGroup(name);
        std::vector<float> scores;
        std::vector<float> scores_ep;
        std::vector<std::uint32_t> widths;
        run.getDataSet("scores").read(scores);
        run.getDataSet("scores_ep").read(scores_ep);
        if (run.exist("width")) {
            run.getDataSet("width").read(widths);
        }
        const auto param_ds = run.getDataSet("param_sets");
        std::vector<double> params(param_ds.getElementCount());
        if (!params.empty()) {
            param_ds.read_raw(params.data());
        }
        RunResults out{.name   = name,
                       .leaves = make_leaves(params, scores, scores_ep, widths),
                       .stats = read_level_stats(run.getDataSet("level_stats")),
                       .circular          = {},
                       .n_harvested_total = 0,
                       .reduction         = {}};
        if (run.exist("level_stats_circular")) {
            const auto ds = run.getDataSet("level_stats_circular");
            const auto level =
                read_member<std::uint64_t>(ds, "level", H5T_NATIVE_UINT64);
            const auto snap    = read_member<std::uint64_t>(ds, "n_leaves_snap",
                                                            H5T_NATIVE_UINT64);
            const auto crackle = read_member<std::uint64_t>(
                ds, "n_leaves_crackle", H5T_NATIVE_UINT64);
            for (std::size_t i = 0; i < level.size(); ++i) {
                out.circular.push_back({level[i], snap[i], crackle[i]});
            }
        }
        for (const char* attr :
             {"n_final_leaves", "n_leaves_above_snr_min", "n_cell_dropped",
              "n_dominated", "n_leaves_retained"}) {
            if (run.hasAttribute(attr)) {
                out.reduction.push_back(
                    run.getAttribute(attr).read<std::uint64_t>());
            }
        }
        if (run.exist("harvest")) {
            run.getGroup("harvest")
                .getAttribute("n_harvested_total")
                .read(out.n_harvested_total);
        }
        runs.push_back(std::move(out));
    }
    return runs;
}

namespace {
std::vector<std::array<std::uint64_t, 3>>
circular_rows(const search::PruneStatsCollection& stats) {
    std::vector<std::array<std::uint64_t, 3>> rows;
    for (const auto& c : stats.get_circular_counts()) {
        rows.push_back({c.level, c.n_leaves_snap, c.n_leaves_crackle});
    }
    return rows;
}
} // namespace

RunResults to_run_results(const pipelines::detail::EPResultPayload& payload) {
    std::vector<double> leaves       = payload.leaves;
    std::vector<float> scores        = payload.scores;
    std::vector<float> scores_ep     = payload.scores_ep;
    std::vector<std::uint32_t> width = payload.widths;
    if (payload.scratch) {
        // A run pruned in seed slices: its rows are on disk.
        auto& scratch = *payload.scratch;
        scratch.seal();
        const auto n = scratch.rows();
        leaves.resize(n * scratch.leaves_stride());
        scores.resize(n);
        scores_ep.resize(n);
        width.resize(scratch.has_widths() ? n : 0);
        scratch.read(0, n, leaves, scores, scores_ep, width);
    }
    return {.name              = payload.run_name,
            .leaves            = make_leaves(leaves, scores, scores_ep, width),
            .stats             = payload.stats.get_packed_data().first,
            .circular          = circular_rows(payload.stats),
            .n_harvested_total = payload.n_harvested_total,
            .reduction         = {}};
}

void check_same_runs(const std::vector<RunResults>& a,
                     const std::vector<RunResults>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t r = 0; r < a.size(); ++r) {
        CAPTURE(a[r].name);
        CHECK(a[r].name == b[r].name);
        CHECK(a[r].n_harvested_total == b[r].n_harvested_total);
        CHECK(a[r].reduction == b[r].reduction);
        REQUIRE(a[r].n_leaves() == b[r].n_leaves());
        CHECK(a[r].leaves == b[r].leaves);
        CHECK(a[r].circular == b[r].circular);
        REQUIRE(a[r].stats.size() == b[r].stats.size());
        for (std::size_t i = 0; i < a[r].stats.size(); ++i) {
            check_same_stats(a[r].stats[i], b[r].stats[i]);
        }
    }
}

} // namespace loki::test
