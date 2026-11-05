#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/fold.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/detail/psr_utils.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::compute_brute_fold;
using loki::algorithms::FFATime;
using loki::detection::snr_boxcar_3d;
using loki::psr_utils::get_phase_idx_uint;
using loki::search::PulsarSearchConfig;

namespace {

[[nodiscard]] std::vector<float> run_ffa(const PulsarSearchConfig& cfg,
                                         const std::vector<float>& ts_e,
                                         const std::vector<float>& ts_v,
                                         std::optional<SizeType> fuse_levels) {
    FFATime ffa(cfg, /*show_progress=*/false);
    ffa.set_fuse_levels(fuse_levels);
    std::vector<float> fold(ffa.get_plan().get_buffer_size(), 0.0F);
    ffa.execute(ts_e, ts_v, fold);
    fold.resize(ffa.get_plan().get_fold_size());
    return fold;
}

[[nodiscard]] PulsarSearchConfig make_cfg(SizeType nsamps,
                                          double tsamp,
                                          SizeType nbins,
                                          SizeType bseg,
                                          int nthreads,
                                          double f_min,
                                          double f_max) {
    const std::vector<ParamLimit> limits = {{.min = f_min, .max = f_max}};
    return {nsamps,
            tsamp,
            nbins,
            /*eta=*/0.5,
            limits,
            /*ducy_max=*/0.2,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            nthreads,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/32,
            bseg};
}

} // namespace

TEST_CASE("Cone bands are bit-exact with the plain BFS path", "[ffa][cone]") {
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.05F * (dist(rng) * dist(rng)));
    }

    // One thorough corner (every fuse level) and the opposite corner
    // (largest bins and segment, several threads, one level plus automatic).
    struct Corner {
        SizeType nbins;
        SizeType bseg;
        int nthreads;
        bool all_levels;
    };
    const std::array<Corner, 2> corners = {{
        {32, 64, 1, true},
        {64, 256, 4, false},
    }};
    for (const auto& corner : corners) {
        const auto cfg = make_cfg(kNsamps, kTsamp, corner.nbins, corner.bseg,
                                  corner.nthreads, 5.0, 14.0);
        const auto reference = run_ffa(cfg, ts_e, ts_v, SizeType{0});
        const std::array<SizeType, 4> all_k = {1, 2, 3, 4};
        const std::span<const SizeType> levels =
            corner.all_levels ? std::span<const SizeType>{all_k}
                              : std::span<const SizeType>{all_k.data(), 1};
        for (const SizeType k : levels) {
            const auto cone = run_ffa(cfg, ts_e, ts_v, k);
            INFO("nbins=" << corner.nbins << " bseg=" << corner.bseg
                          << " nthreads=" << corner.nthreads << " k=" << k);
            REQUIRE(cone.size() == reference.size());
            CHECK(std::memcmp(cone.data(), reference.data(),
                              reference.size() * sizeof(float)) == 0);
        }
        const auto automatic = run_ffa(cfg, ts_e, ts_v, std::nullopt);
        CHECK(std::memcmp(automatic.data(), reference.data(),
                          reference.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("Ragged cone tiles stay bit-exact", "[ffa][cone]") {
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    const std::vector<float> ts_v(kNsamps, 1.0F);
    for (float& sample : ts_e) {
        sample = dist(rng);
    }
    const auto cfg       = make_cfg(kNsamps, kTsamp, /*nbins=*/32, /*bseg=*/128,
                                    /*nthreads=*/2, 4.0, 12.0);
    const auto reference = run_ffa(cfg, ts_e, ts_v, SizeType{0});
    // NOLINTNEXTLINE(misc-include-cleaner): POSIX setenv
    REQUIRE(setenv("LOKI_CONE_TILE", "3", 1) == 0);
    const auto ragged = run_ffa(cfg, ts_e, ts_v, std::nullopt);
    unsetenv("LOKI_CONE_TILE"); // NOLINT(misc-include-cleaner): POSIX
    REQUIRE(ragged.size() == reference.size());
    CHECK(std::memcmp(ragged.data(), reference.data(),
                      reference.size() * sizeof(float)) == 0);
}

TEST_CASE("Run-length brute fold matches a direct bin sum within tolerance",
          "[ffa][runs]") {
    constexpr SizeType kNsamps = 4096;
    constexpr SizeType kSeg    = 256;
    constexpr SizeType kBins   = 32;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F;
    }
    const std::vector<double> freqs = {8.0, 11.5, 14.0};
    const auto got =
        compute_brute_fold<float>(ts_e, ts_v, freqs, kSeg, kBins, kTsamp,
                                  /*t_ref=*/0.0, loki::Exec::cpu(1));
    const SizeType nseg = kNsamps / kSeg;
    std::vector<float> ref(nseg * freqs.size() * 2 * kBins, 0.0F);
    for (SizeType iseg = 0; iseg < nseg; ++iseg) {
        for (SizeType ifreq = 0; ifreq < freqs.size(); ++ifreq) {
            float* fold_e =
                ref.data() + (((iseg * freqs.size()) + ifreq) * 2 * kBins);
            float* fold_v = fold_e + kBins;
            for (SizeType isamp = 0; isamp < kSeg; ++isamp) {
                const double proper_time = static_cast<double>(isamp) * kTsamp;
                const auto bin =
                    get_phase_idx_uint(proper_time, freqs[ifreq], kBins, 0.0);
                const SizeType idx = (iseg * kSeg) + isamp;
                fold_e[bin] += ts_e[idx];
                fold_v[bin] += ts_v[idx];
            }
        }
    }
    REQUIRE(got.size() == ref.size());
    for (SizeType i = 0; i < ref.size(); ++i) {
        const float scale = std::max(1.0F, std::abs(ref[i]));
        CHECK(std::abs(got[i] - ref[i]) <= 1.0e-4F * scale);
    }
}

TEST_CASE("execute_scored matches a threshold scan of snr_boxcar_3d",
          "[ffa][cone][score]") {
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(19);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (float& sample : ts_e) {
        sample = dist(rng);
    }
    const auto cfg = make_cfg(kNsamps, kTsamp, /*nbins=*/32, /*bseg=*/128,
                              /*nthreads=*/2, 6.0, 12.0);
    FFATime plain(cfg, /*show_progress=*/false);
    plain.set_fuse_levels(SizeType{0});
    std::vector<float> fold(plain.get_plan().get_buffer_size(), 0.0F);
    plain.execute(ts_e, ts_v, fold);
    const auto ncoords = plain.get_plan().get_ncoords().back();
    const auto nbins   = cfg.get_nbins();
    const auto widths  = cfg.get_scoring_widths();
    std::vector<float> scores(ncoords * widths.size());
    snr_boxcar_3d(std::span<const float>(fold).first(ncoords * 2 * nbins),
                  widths, scores, ncoords, nbins, loki::Exec::cpu(1));
    constexpr float kThreshold = 1.5F;
    std::vector<loki::detection::SnrHit> expected;
    for (SizeType i = 0; i < scores.size(); ++i) {
        if (scores[i] >= kThreshold) {
            expected.push_back({static_cast<uint32_t>(i), scores[i]});
        }
    }

    FFATime scored(cfg, /*show_progress=*/false);
    std::vector<loki::detection::SnrHit> hits;
    scored.execute_scored(ts_e, ts_v, kThreshold, widths, hits);
    REQUIRE(hits.size() == expected.size());
    for (SizeType i = 0; i < hits.size(); ++i) {
        CHECK(hits[i].score_index == expected[i].score_index);
        CHECK(hits[i].snr == expected[i].snr);
    }
}

TEST_CASE("execute_scored without cone bands matches a threshold scan",
          "[ffa][cone][score]") {
    // Base segments too long for the cone scratch (here forced with a 1 KiB
    // budget) make the cone path decline; the time-domain frequency-only
    // plan must then score the stored fold (it used to read accel
    // coordinates that frequency-only workspaces never allocate).
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(23);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (float& sample : ts_e) {
        sample = dist(rng);
    }
    const auto cfg = make_cfg(kNsamps, kTsamp, /*nbins=*/32, /*bseg=*/128,
                              /*nthreads=*/2, 6.0, 12.0);
    FFATime plain(cfg, /*show_progress=*/false);
    plain.set_fuse_levels(SizeType{0});
    std::vector<float> fold(plain.get_plan().get_buffer_size(), 0.0F);
    plain.execute(ts_e, ts_v, fold);
    const auto ncoords = plain.get_plan().get_ncoords().back();
    const auto nbins   = cfg.get_nbins();
    const auto widths  = cfg.get_scoring_widths();
    std::vector<float> scores(ncoords * widths.size());
    snr_boxcar_3d(std::span<const float>(fold).first(ncoords * 2 * nbins),
                  widths, scores, ncoords, nbins, loki::Exec::cpu(1));
    constexpr float kThreshold = 1.5F;
    std::vector<loki::detection::SnrHit> expected;
    for (SizeType i = 0; i < scores.size(); ++i) {
        if (scores[i] >= kThreshold) {
            expected.push_back({static_cast<uint32_t>(i), scores[i]});
        }
    }

    FFATime scored(cfg, /*show_progress=*/false);
    std::vector<loki::detection::SnrHit> hits;
    // NOLINTNEXTLINE(misc-include-cleaner): POSIX setenv
    REQUIRE(setenv("LOKI_CONE_SCRATCH_KB", "1", 1) == 0);
    scored.execute_scored(ts_e, ts_v, kThreshold, widths, hits);
    unsetenv("LOKI_CONE_SCRATCH_KB"); // NOLINT(misc-include-cleaner): POSIX
    REQUIRE(hits.size() == expected.size());
    for (SizeType i = 0; i < hits.size(); ++i) {
        CHECK(hits[i].score_index == expected[i].score_index);
        CHECK(hits[i].snr == expected[i].snr);
    }
}

TEST_CASE("Lossy-Fourier cone bands are bit-exact with unfused lossy init",
          "[ffa][cone][fourier]") {
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(29);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.05F * (dist(rng) * dist(rng)));
    }
    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 14.0}};
    for (const SizeType nbins : {SizeType{50}, SizeType{64}}) {
        for (const int nthreads : {1, 3}) {
            const PulsarSearchConfig cfg(kNsamps, kTsamp, nbins,
                                         /*eta=*/0.5, limits,
                                         /*ducy_max=*/0.2,
                                         /*wtsp=*/1.5,
                                         /*use_fourier=*/true, nthreads,
                                         /*max_process_memory_gb=*/4.0,
                                         /*octave_scale=*/2.0,
                                         /*nbins_max=*/1024,
                                         /*nbins_min_lossy_bf=*/32,
                                         /*bseg_brute=*/128);
            const auto widths          = cfg.get_scoring_widths();
            constexpr float kThreshold = 1.5F;

            loki::algorithms::FFAFourier unfused(cfg, /*show_progress=*/false);
            unfused.set_fuse_levels(SizeType{0});
            std::vector<loki::detection::SnrHit> expected;
            unfused.execute_scored(ts_e, ts_v, kThreshold, widths, expected);

            loki::algorithms::FFAFourier fused(cfg, /*show_progress=*/false);
            std::vector<loki::detection::SnrHit> got;
            fused.execute_scored(ts_e, ts_v, kThreshold, widths, got);

            INFO("nbins=" << nbins << " nthreads=" << nthreads);
            CHECK(fused.get_last_fuse_levels() >= 1);
            REQUIRE(got.size() == expected.size());
            for (SizeType i = 0; i < got.size(); ++i) {
                CHECK(got[i].score_index == expected[i].score_index);
                CHECK(std::memcmp(&got[i].snr, &expected[i].snr,
                                  sizeof(float)) == 0);
            }
        }
    }
}

TEST_CASE("BruteFold phase runs match the sample-by-sample reference",
          "[ffa][runs]") {
    constexpr double kTsamp = 64.0e-6;
    std::vector<double> freqs;
    for (int i = 0; i < 64; ++i) {
        freqs.push_back(0.05 + (0.0173 * static_cast<double>(i)));
        freqs.push_back(10.0 + (2.3456789 * static_cast<double>(i)));
    }
    for (const SizeType nbins : {SizeType{32}, SizeType{50}, SizeType{512}}) {
        for (const SizeType seg_len : {SizeType{256}, SizeType{2048}}) {
            for (const double t_ref :
                 {0.0, 0.5 * static_cast<double>(seg_len) * kTsamp}) {
                const loki::algorithms::BruteFold<float> bf(
                    freqs, seg_len, nbins, seg_len * 4, kTsamp, t_ref,
                    loki::Exec::cpu(3));
                const auto got_runs    = bf.runs();
                const auto got_offsets = bf.run_offsets();
                REQUIRE(got_offsets.size() == freqs.size() + 1);

                std::vector<loki::coord::PhaseRun> exp_runs;
                std::vector<SizeType> exp_offsets(freqs.size() + 1, 0);
                for (SizeType ifreq = 0; ifreq < freqs.size(); ++ifreq) {
                    uint32_t prev_bin = 0;
                    for (SizeType isamp = 0; isamp < seg_len; ++isamp) {
                        const double proper_time =
                            (static_cast<double>(isamp) * kTsamp) - t_ref;
                        const uint32_t iphase = get_phase_idx_uint(
                            proper_time, freqs[ifreq], nbins, 0.0);
                        if (isamp > 0 && iphase != prev_bin) {
                            exp_runs.push_back(
                                {.end = static_cast<uint32_t>(isamp),
                                 .bin = prev_bin});
                        }
                        prev_bin = iphase;
                    }
                    exp_runs.push_back({.end = static_cast<uint32_t>(seg_len),
                                        .bin = prev_bin});
                    exp_offsets[ifreq + 1] = exp_runs.size();
                }

                INFO("nbins=" << nbins << " seg_len=" << seg_len
                              << " t_ref=" << t_ref);
                CHECK(std::ranges::equal(got_offsets, exp_offsets));
                REQUIRE(got_runs.size() == exp_runs.size());
                CHECK(std::memcmp(got_runs.data(), exp_runs.data(),
                                  exp_runs.size() *
                                      sizeof(loki::coord::PhaseRun)) == 0);
            }
        }
    }
}
