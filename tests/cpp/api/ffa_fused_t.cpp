#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/algorithms/ffa.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::FFAFourier;
using loki::algorithms::FFATime;
using loki::detection::snr_boxcar_3d;
using loki::detection::SnrHit;
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
    // Only the final-level fold is meaningful.
    fold.resize(ffa.get_plan().get_fold_size());
    return fold;
}

} // namespace

TEST_CASE("Fused brute fold + FFA levels is bit-exact with the plain path",
          "[ffa][fused]") {
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.1F * (dist(rng) * dist(rng)));
    }

    // One thorough corner (every fuse level) and the opposite corner
    // (larger segment, several threads, one level plus automatic).
    const auto check_corner = [&](SizeType bseg_brute, int nthreads,
                                  std::initializer_list<SizeType> levels) {
        const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 14.0}};
        const PulsarSearchConfig cfg(
            kNsamps, kTsamp, /*nbins=*/32, /*eta=*/0.5, limits,
            /*ducy_max=*/0.2, /*wtsp=*/1.5, /*use_fourier=*/false, nthreads,
            /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
            /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, bseg_brute);
        const auto reference = run_ffa(cfg, ts_e, ts_v, SizeType{0});
        for (const SizeType k : levels) {
            const auto fused = run_ffa(cfg, ts_e, ts_v, k);
            REQUIRE(fused.size() == reference.size());
            const bool equal =
                std::memcmp(fused.data(), reference.data(),
                            reference.size() * sizeof(float)) == 0;
            INFO("bseg_brute=" << bseg_brute << " nthreads=" << nthreads
                               << " fuse_levels=" << k);
            CHECK(equal);
        }
        const auto automatic = run_ffa(cfg, ts_e, ts_v, std::nullopt);
        CHECK(std::memcmp(automatic.data(), reference.data(),
                          reference.size() * sizeof(float)) == 0);
    };
    check_corner(64, 1, {1, 2, 3, 4, 5});
    check_corner(256, 4, {1});
}

TEST_CASE("Accel execute_scored is bit-exact with a scan of the stored fold",
          "[ffa][fused][score]") {
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.1F * std::abs(dist(rng)));
    }
    const SizeType nbins                 = GENERATE(SizeType{32}, SizeType{50});
    const int nthreads                   = GENERATE(1, 3);
    const std::vector<ParamLimit> limits = {{.min = -200.0, .max = 200.0},
                                            {.min = 5.0, .max = 14.0}};
    const PulsarSearchConfig cfg(
        kNsamps, kTsamp, nbins, /*eta=*/0.5, limits, /*ducy_max=*/0.2,
        /*wtsp=*/1.5, /*use_fourier=*/false, nthreads,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, /*bseg_brute=*/128);
    const auto widths = cfg.get_scoring_widths();

    FFATime plain(cfg, /*show_progress=*/false);
    std::vector<float> fold(plain.get_plan().get_buffer_size(), 0.0F);
    plain.execute(ts_e, ts_v, fold);
    const auto ncoords = plain.get_plan().get_ncoords().back();
    REQUIRE(plain.get_plan().get_n_params() == 2);
    std::vector<float> scores(ncoords * widths.size());
    snr_boxcar_3d(std::span<const float>(fold).first(ncoords * 2 * nbins),
                  widths, scores, ncoords, nbins, loki::Exec::cpu(1));
    constexpr float kThreshold = 1.0F;
    std::vector<SnrHit> expected;
    for (SizeType i = 0; i < scores.size(); ++i) {
        if (scores[i] >= kThreshold) {
            expected.push_back({static_cast<uint32_t>(i), scores[i]});
        }
    }
    REQUIRE(!expected.empty());

    // Fused final level (default) and the stored-fold fallback (fusion off).
    for (const auto fuse :
         {std::optional<SizeType>{}, std::optional<SizeType>{SizeType{0}}}) {
        FFATime scored(cfg, /*show_progress=*/false);
        scored.set_fuse_levels(fuse);
        std::vector<SnrHit> hits;
        scored.execute_scored(ts_e, ts_v, kThreshold, widths, hits);
        INFO("nbins=" << nbins << " nthreads=" << nthreads
                      << " fused=" << !fuse.has_value());
        REQUIRE(hits.size() == expected.size());
        bool equal = true;
        for (SizeType i = 0; i < hits.size(); ++i) {
            equal =
                equal && hits[i].score_index == expected[i].score_index &&
                std::memcmp(&hits[i].snr, &expected[i].snr, sizeof(float)) == 0;
        }
        CHECK(equal);
    }
}

TEST_CASE("Fourier execute_scored matches a scan of the stored fold",
          "[ffa][fused][score]") {
    constexpr SizeType kNsamps = 1U << 15U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(91);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps);
    for (SizeType i = 0; i < kNsamps; ++i) {
        ts_e[i] = dist(rng);
        ts_v[i] = 1.0F + (0.1F * std::abs(dist(rng)));
    }
    // 32 bins: direct Fourier brute fold; 128 and 50: lossy init.
    const SizeType nbins = GENERATE(SizeType{32}, SizeType{50}, SizeType{128});
    const bool accel     = GENERATE(true, false);
    const int nthreads   = GENERATE(1, 3);
    std::vector<ParamLimit> limits;
    if (accel) {
        limits.push_back({.min = -200.0, .max = 200.0});
    }
    limits.push_back({.min = 5.0, .max = 14.0});
    const PulsarSearchConfig cfg(
        kNsamps, kTsamp, nbins, /*eta=*/0.5, limits, /*ducy_max=*/0.2,
        /*wtsp=*/1.5, /*use_fourier=*/true, nthreads,
        /*max_process_memory_gb=*/4.0, /*octave_scale=*/2.0,
        /*nbins_max=*/1024, /*nbins_min_lossy_bf=*/32, /*bseg_brute=*/128);
    const auto widths = cfg.get_scoring_widths();

    FFAFourier plain(cfg, /*show_progress=*/false);
    std::vector<float> fold(2 * plain.get_plan().get_buffer_size(), 0.0F);
    plain.execute(ts_e, ts_v, std::span<float>(fold));
    const auto ncoords = plain.get_plan().get_ncoords().back();
    std::vector<float> scores(ncoords * widths.size());
    snr_boxcar_3d(std::span<const float>(fold).first(ncoords * 2 * nbins),
                  widths, scores, ncoords, nbins, loki::Exec::cpu(1));
    constexpr float kThreshold = 1.0F;

    for (const auto fuse :
         {std::optional<SizeType>{}, std::optional<SizeType>{SizeType{0}}}) {
        FFAFourier scored(cfg, /*show_progress=*/false);
        scored.set_fuse_levels(fuse);
        std::vector<SnrHit> hits;
        scored.execute_scored(ts_e, ts_v, kThreshold, widths, hits);
        INFO("nbins=" << nbins << " accel=" << accel << " nthreads=" << nthreads
                      << " fused=" << !fuse.has_value());
        // Every hit is a stored-fold score above the threshold, in order.
        SizeType n_expected = 0;
        for (const float score : scores) {
            n_expected += score >= kThreshold ? 1 : 0;
        }
        REQUIRE(hits.size() == n_expected);
        SizeType n_exact = 0;
        float max_diff   = 0.0F;
        for (SizeType i = 0; i < hits.size(); ++i) {
            REQUIRE(hits[i].score_index < scores.size());
            if (i > 0) {
                REQUIRE(hits[i].score_index > hits[i - 1].score_index);
            }
            const float ref = scores[hits[i].score_index];
            n_exact += std::memcmp(&hits[i].snr, &ref, sizeof(float)) == 0;
            max_diff = std::max(max_diff, std::abs(hits[i].snr - ref));
        }
        INFO("exact " << n_exact << "/" << hits.size() << " max |diff| "
                      << max_diff);
        // Both paths invert with the lane FFT, whose per-profile result does
        // not depend on the batch or the thread count.
        CHECK(n_exact == hits.size());
        if (!accel && !fuse.has_value()) {
            // Frequency-only: scored by Fourier cone bands.
            CHECK(scored.get_last_fuse_levels() > 0);
        }
    }
}
