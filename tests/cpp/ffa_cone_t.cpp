#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "detail/psr_utils.hpp"
#include "loki/algorithms/ffa.hpp"
#include "loki/algorithms/fold.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

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
    return PulsarSearchConfig(nsamps, tsamp, nbins, /*eta=*/0.5, limits,
                              /*ducy_max=*/0.2, /*wtsp=*/1.5,
                              /*use_fourier=*/false, nthreads,
                              /*max_process_memory_gb=*/4.0,
                              /*octave_scale=*/2.0, /*nbins_max=*/1024,
                              /*nbins_min_lossy_bf=*/32, bseg);
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

    for (const SizeType nbins : {SizeType{16}, SizeType{32}, SizeType{64}}) {
        for (const SizeType bseg : {SizeType{64}, SizeType{256}}) {
            for (const int nthreads : {1, 4}) {
                const auto cfg =
                    make_cfg(kNsamps, kTsamp, nbins, bseg, nthreads, 5.0, 14.0);
                const auto reference = run_ffa(cfg, ts_e, ts_v, SizeType{0});
                for (const SizeType k :
                     {SizeType{1}, SizeType{2}, SizeType{3}, SizeType{4}}) {
                    const auto cone = run_ffa(cfg, ts_e, ts_v, k);
                    INFO("nbins=" << nbins << " bseg=" << bseg
                                  << " nthreads=" << nthreads << " k=" << k);
                    REQUIRE(cone.size() == reference.size());
                    CHECK(std::memcmp(cone.data(), reference.data(),
                                      reference.size() * sizeof(float)) == 0);
                }
                const auto automatic = run_ffa(cfg, ts_e, ts_v, std::nullopt);
                CHECK(std::memcmp(automatic.data(), reference.data(),
                                  reference.size() * sizeof(float)) == 0);
            }
        }
    }
}

TEST_CASE("Ragged cone tiles stay bit-exact", "[ffa][cone]") {
    constexpr SizeType kNsamps = 1U << 14U;
    constexpr double kTsamp    = 1.0e-3;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> ts_e(kNsamps);
    std::vector<float> ts_v(kNsamps, 1.0F);
    for (float& sample : ts_e) {
        sample = dist(rng);
    }
    const auto cfg       = make_cfg(kNsamps, kTsamp, /*nbins=*/32, /*bseg=*/128,
                                    /*nthreads=*/2, 4.0, 12.0);
    const auto reference = run_ffa(cfg, ts_e, ts_v, SizeType{0});
    REQUIRE(setenv("LOKI_CONE_TILE", "3", 1) == 0);
    const auto ragged = run_ffa(cfg, ts_e, ts_v, std::nullopt);
    unsetenv("LOKI_CONE_TILE");
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
