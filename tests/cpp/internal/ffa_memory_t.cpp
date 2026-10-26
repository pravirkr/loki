#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/fold.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/coord.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/algorithms/planner_memory.hpp"
#include "lib/detail/utils.hpp"
#include "lib/utils/workspace_impl.hpp"

using loki::ComplexType;
using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::FFA;
using loki::algorithms::FFARegionPlanner;
using loki::detection::SnrHit;
using loki::plans::FFAPlan;
using loki::search::PulsarSearchConfig;
namespace detail = loki::algorithms::detail;

namespace {

constexpr SizeType kNsamps = SizeType{1} << 15U;
constexpr double kTsamp    = 1.0e-3;

/// Sets an environment variable for one scope (test hooks of the engine).
class ScopedEnv {
public:
    ScopedEnv(const char* name, const std::string& value) : m_name(name) {
        // NOLINTNEXTLINE(misc-include-cleaner): POSIX setenv
        setenv(m_name, value.c_str(), 1);
    }
    ~ScopedEnv() { unsetenv(m_name); } // NOLINT(misc-include-cleaner): POSIX
    ScopedEnv(const ScopedEnv&)            = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
    ScopedEnv(ScopedEnv&&)                 = delete;
    ScopedEnv& operator=(ScopedEnv&&)      = delete;

private:
    const char* m_name;
};

std::pair<std::vector<float>, std::vector<float>> make_series(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.1F * std::abs(dist(rng)));
    }
    return {std::move(ts_e), std::move(ts_v)};
}

PulsarSearchConfig make_cfg(SizeType nbins,
                            bool accel,
                            bool fourier,
                            int nthreads,
                            SizeType bseg    = 128,
                            ParamLimit freqs = {.min = 5.0, .max = 14.0}) {
    std::vector<ParamLimit> limits;
    if (accel) {
        limits.push_back({.min = -200.0, .max = 200.0});
    }
    limits.push_back(freqs);
    return {kNsamps,
            kTsamp,
            nbins,
            /*eta=*/0.5,
            limits,
            /*ducy_max=*/0.2,
            /*wtsp=*/1.5,
            fourier,
            nthreads,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/32,
            bseg};
}

template <typename FoldType>
std::vector<SnrHit> scored_hits(const PulsarSearchConfig& cfg,
                                const std::vector<float>& ts_e,
                                const std::vector<float>& ts_v,
                                float threshold) {
    FFA<FoldType> ffa(cfg, /*show_progress=*/false);
    std::vector<SnrHit> hits;
    ffa.execute_scored(ts_e, ts_v, threshold, cfg.get_scoring_widths(), hits);
    return hits;
}

bool same_hits(const std::vector<SnrHit>& a, const std::vector<SnrHit>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (SizeType i = 0; i < a.size(); ++i) {
        if (a[i].score_index != b[i].score_index ||
            std::memcmp(&a[i].snr, &b[i].snr, sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("Phase-step table batches are bit-exact", "[ffa][memory]") {
    const auto [ts_e, ts_v] = make_series(5);
    const SizeType nbins    = GENERATE(SizeType{32}, SizeType{50});
    const bool accel        = GENERATE(false, true);
    const int nthreads      = GENERATE(1, 3);
    const auto cfg = make_cfg(nbins, accel, /*fourier=*/true, nthreads);
    CAPTURE(nbins, accel, nthreads);

    // execute(): every level merged and stored (segment-major levels use
    // the table); execute_scored(): cone bands (frequency-only) or the
    // fused final level.
    const auto run_fold = [&] {
        FFA<ComplexType> ffa(cfg, /*show_progress=*/false);
        std::vector<ComplexType> fold(ffa.get_plan().get_buffer_size());
        ffa.execute(ts_e, ts_v, fold);
        fold.resize(ffa.get_plan().get_fold_size());
        return fold;
    };
    const auto fold_ref = run_fold();
    const auto hits_ref = scored_hits<ComplexType>(cfg, ts_e, ts_v, 1.0F);
    REQUIRE(!hits_ref.empty());
    for (const char* kb : {"24", "96"}) {
        CAPTURE(kb);
        const ScopedEnv table("LOKI_PHASE_TABLE_KB", kb);
        const auto fold = run_fold();
        REQUIRE(fold.size() == fold_ref.size());
        CHECK(std::memcmp(fold.data(), fold_ref.data(),
                          fold.size() * sizeof(ComplexType)) == 0);
        CHECK(same_hits(scored_hits<ComplexType>(cfg, ts_e, ts_v, 1.0F),
                        hits_ref));
    }
}

TEST_CASE("Hit batches are bit-exact and in score-index order",
          "[ffa][memory]") {
    const auto [ts_e, ts_v] = make_series(9);
    const bool fourier      = GENERATE(false, true);
    const bool accel        = GENERATE(false, true);
    const int nthreads      = GENERATE(1, 3);
    const auto cfg          = make_cfg(/*nbins=*/32, accel, fourier, nthreads);
    CAPTURE(fourier, accel, nthreads);
    const auto run = [&] {
        return fourier ? scored_hits<ComplexType>(cfg, ts_e, ts_v, 1.0F)
                       : scored_hits<float>(cfg, ts_e, ts_v, 1.0F);
    };
    const auto ref = run();
    REQUIRE(!ref.empty());
    for (SizeType i = 1; i < ref.size(); ++i) {
        REQUIRE(ref[i].score_index > ref[i - 1].score_index);
    }
    // One tile or row group per batch, and ragged batches.
    for (const char* coords : {"1", "700"}) {
        CAPTURE(coords);
        const ScopedEnv batch("LOKI_HIT_BATCH_COORDS", coords);
        CHECK(same_hits(run(), ref));
    }
    if (!fourier && !accel) {
        // Cone bands declined: the stored final fold is scored tile by tile.
        const ScopedEnv budget("LOKI_CONE_SCRATCH_KB", "1");
        const auto stored = run();
        const ScopedEnv batch("LOKI_HIT_BATCH_COORDS", "100");
        CHECK(same_hits(run(), stored));
    }
}

TEST_CASE("Final-level row groups fit the planned bound", "[ffa][memory]") {
    for (const SizeType nbins : {SizeType{32}, SizeType{64}}) {
        for (const SizeType bseg : {SizeType{64}, SizeType{256}}) {
            const auto cfg =
                make_cfg(nbins, /*accel=*/true, /*fourier=*/false, 1, bseg);
            FFAPlan<float> plan(cfg);
            const auto levels       = plan.resolve_coordinates();
            const auto& coords      = levels.back();
            const auto& counts      = plan.get_param_counts();
            const SizeType row_len  = counts.back().back();
            const SizeType prev_len = counts[counts.size() - 2].back();
            const SizeType nrows    = coords.size() / row_len;
            const auto key          = [&](SizeType row) {
                const auto& c = coords[row * row_len];
                return std::pair<SizeType, SizeType>{c.i_tail / prev_len,
                                                     c.i_head / prev_len};
            };
            SizeType largest = 0;
            for (SizeType row = 0; row < nrows;) {
                SizeType end = row + 1;
                while (end < nrows && key(end) == key(row)) {
                    ++end;
                }
                largest = std::max(largest, end - row);
                row     = end;
            }
            CAPTURE(nbins, bseg, nrows, largest);
            CHECK(largest <= detail::cpu_final_row_group_rows(plan));
        }
    }
}

TEST_CASE("CPU sweep memory model covers the workspace and the scratch",
          "[ffa][memory]") {
    for (const bool fourier : {false, true}) {
        for (const bool accel : {false, true}) {
            for (const SizeType nbins :
                 {SizeType{32}, SizeType{50}, SizeType{128}}) {
                for (const int nthreads : {1, 3}) {
                    CAPTURE(fourier, accel, nbins, nthreads);
                    // 128 bins need periods of 128 samples or more.
                    const auto cfg   = make_cfg(nbins, accel, fourier, nthreads,
                                                128, {.min = 3.0, .max = 7.0});
                    const auto check = [&]<typename FoldType>() {
                        const FFARegionPlanner<FoldType> planner(cfg);
                        const auto& stats = planner.get_stats();
                        // The sweep's allocations, from the planner's sizes.
                        loki::memory::FFAWorkspaceCPU<FoldType> ws(
                            stats.get_max_buffer_size(),
                            stats.get_max_coord_size(), cfg.get_nparams());
                        SizeType scratch_max = 0;
                        SizeType step_max    = 0;
                        for (const auto& chunk : planner.get_cfgs()) {
                            const FFAPlan<FoldType> plan(chunk);
                            REQUIRE(stats.get_max_buffer_size() >=
                                    detail::cpu_scored_buffer_size(plan));
                            scratch_max = std::max(
                                scratch_max,
                                (static_cast<SizeType>(nthreads) *
                                 detail::cpu_ffa_thread_scratch_bytes(plan)) +
                                    detail::cpu_hit_buffer_bytes(plan));
                            step_max = std::max(
                                step_max,
                                detail::cpu_phase_step_table_bytes(plan));
                        }
                        ws.ensure_phase_steps(step_max);
                        const SizeType fold_time_bytes =
                            stats.get_max_buffer_size_time() * sizeof(float);
                        const SizeType ws_bytes =
                            (ws.fold_internal.size() * sizeof(FoldType)) +
                            (ws.coords.size() * sizeof(loki::coord::FFACoord)) +
                            (ws.coords_freq.size() *
                             sizeof(loki::coord::FFACoordFreq));
                        const double modelled_gb =
                            static_cast<double>(
                                stats.get_buffer_memory_usage()) +
                            stats.get_coord_memory_usage() +
                            stats.get_scratch_memory_usage();
                        const double real_gb =
                            static_cast<double>(ws_bytes + fold_time_bytes +
                                                ws.phase_steps.size() +
                                                scratch_max) /
                            loki::utils::kBytesPerGB;
                        CHECK(modelled_gb >= real_gb * (1.0 - 1.0e-6));
                        CHECK(stats.get_freq_sweep_memory_usage() <=
                              cfg.get_max_process_memory_gb());
                    };
                    if (fourier) {
                        check.template operator()<ComplexType>();
                    } else {
                        check.template operator()<float>();
                    }
                }
            }
        }
    }
}

TEST_CASE("The phase-run estimate covers the brute-fold tables",
          "[ffa][memory]") {
    // Fast pulsars (runs of about one sample) and slow ones (long segments,
    // few runs per segment).
    for (const ParamLimit freqs : {ParamLimit{.min = 5.0, .max = 14.0},
                                   ParamLimit{.min = 0.2, .max = 0.6}}) {
        const auto cfg = make_cfg(/*nbins=*/32, /*accel=*/false,
                                  /*fourier=*/false, 1, 128, freqs);
        const FFARegionPlanner<float> planner(cfg);
        const double table_bytes =
            static_cast<double>(
                planner.get_stats().get_brutefold_memory_usage()) *
            loki::utils::kBytesPerGB;
        for (const auto& chunk : planner.get_cfgs()) {
            const FFAPlan<float> plan(chunk);
            const auto grid = plan.compute_param_grid(0).back();
            const loki::algorithms::BruteFold<float> bf(
                grid, plan.get_segment_lens().front(), chunk.get_nbins(),
                chunk.get_nsamps(), chunk.get_tsamp(), /*t_ref=*/0.0,
                loki::Exec::cpu(1));
            // Built from per-frequency lists of up to twice their runs.
            const double built = 3.0 * static_cast<double>(bf.runs().size()) *
                                 sizeof(loki::coord::PhaseRun);
            CAPTURE(freqs.min, chunk.get_nbins(), bf.runs().size());
            CHECK(built <= table_bytes * (1.0 + 1.0e-6));
        }
    }
}
