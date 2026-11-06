#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"

#include "lib/core/kernels.hpp"
#include "lib/core/taylor_ffa.hpp"
#include "lib/cpu/boxcar_kernels.hpp"
#include "lib/cpu/lane_fft.hpp"
#include "lib/detail/utils.hpp"

using loki::ComplexType;
using loki::SizeType;
using loki::coord::FFACoord;

namespace {

// Synthetic last three levels of an accel FFA. Level L-2 has 4 segments of
// `n2` rows; level L-1 rows read any level-(L-2) rows; final rows form a grid
// of `nrows` x `row_len` whose parents follow the accel layout (row r reads
// parent row r / 4, column ~ c / 2, the head with a per-row drift). A few
// rows drift far, so tiles must split to fit the parent scratch.
struct Levels {
    SizeType nbins;
    SizeType n2;
    SizeType nrows_prev;
    SizeType row_len_prev;
    SizeType nrows;
    SizeType row_len;
    std::vector<FFACoord> coords_prev;
    std::vector<FFACoord> coords;

    SizeType ncoords_prev() const { return nrows_prev * row_len_prev; }
    SizeType ncoords() const { return nrows * row_len; }
};

Levels make_levels(SizeType nbins, std::mt19937& rng) {
    Levels lv{.nbins        = nbins,
              .n2           = 300,
              .nrows_prev   = 5,
              .row_len_prev = 260,
              .nrows        = 19,
              .row_len      = 517,
              .coords_prev  = {},
              .coords       = {}};
    std::uniform_int_distribution<uint32_t> row2(
        0, static_cast<uint32_t>(lv.n2 - 1));
    std::uniform_real_distribution<float> shift(
        0.0F, static_cast<float>(nbins) - 0.01F);
    for (SizeType i = 0; i < lv.ncoords_prev(); ++i) {
        lv.coords_prev.push_back(
            {row2(rng), shift(rng), row2(rng), shift(rng)});
    }
    for (SizeType r = 0; r < lv.nrows; ++r) {
        const SizeType prow = std::min(r / 4, lv.nrows_prev - 1);
        // Rows 7 and 13 drift across most of the parent row.
        const auto drift =
            static_cast<std::int64_t>((r == 7 || r == 13) ? 200 : (r % 5) * 3);
        for (SizeType c = 0; c < lv.row_len; ++c) {
            const auto col      = static_cast<std::int64_t>(c / 2);
            const auto head_col = std::clamp<std::int64_t>(
                col + drift - ((r == 13) ? static_cast<std::int64_t>(c) : 0), 0,
                static_cast<std::int64_t>(lv.row_len_prev - 1));
            const auto tail_col = std::min<std::int64_t>(
                col, static_cast<std::int64_t>(lv.row_len_prev - 1));
            lv.coords.push_back(
                {static_cast<uint32_t>((prow * lv.row_len_prev) + tail_col),
                 shift(rng),
                 static_cast<uint32_t>((prow * lv.row_len_prev) + head_col),
                 shift(rng)});
        }
    }
    return lv;
}

struct Collected {
    SizeType stride;
    std::vector<float> profiles;
};

void collect_real(const float* profiles,
                  SizeType coord_begin,
                  SizeType nprofiles,
                  SizeType nbins,
                  void* ctx) {
    auto& out = *static_cast<Collected*>(ctx);
    std::memcpy(out.profiles.data() + (coord_begin * 2 * nbins), profiles,
                nprofiles * 2 * nbins * sizeof(float));
}

struct CollectedComplex {
    SizeType stride;
    std::vector<ComplexType> profiles;
};

void collect_complex(ComplexType* tile,
                     SizeType coord_begin,
                     SizeType count,
                     void* ctx) {
    auto& out = *static_cast<CollectedComplex*>(ctx);
    std::memcpy(
        static_cast<void*>(out.profiles.data() + (coord_begin * out.stride)),
        tile, count * out.stride * sizeof(ComplexType));
}

} // namespace

TEST_CASE("Two fused final levels are bit-exact with the stored level L-1",
          "[ffa][fused][final2]") {
    const SizeType nbins = GENERATE(SizeType{32}, SizeType{50});
    const SizeType tile  = GENERATE(SizeType{1}, SizeType{16}, SizeType{128});
    const int nthreads   = GENERATE(1, 3);
    std::mt19937 rng(static_cast<unsigned>(nbins * 7 + tile));
    const Levels lv = make_levels(nbins, rng);

    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> level2(4 * lv.n2 * 2 * nbins);
    for (auto& x : level2) {
        x = dist(rng);
    }
    std::vector<float> level1(2 * lv.ncoords_prev() * 2 * nbins);
    loki::core::ffa_iter(level2.data(), level1.data(), lv.coords_prev.data(),
                         lv.ncoords_prev(), lv.n2, /*nsegments=*/2, nbins, 1);

    Collected expected{.stride = 2 * nbins,
                       .profiles =
                           std::vector<float>(lv.ncoords() * 2 * nbins)};
    loki::core::ffa_final_scored(
        level1.data(), lv.coords.data(), lv.ncoords(), lv.ncoords_prev(), nbins,
        lv.row_len, lv.row_len_prev, tile, &collect_real, &expected, nthreads);
    Collected fused{.stride   = 2 * nbins,
                    .profiles = std::vector<float>(lv.ncoords() * 2 * nbins)};
    loki::core::ffa_final2_scored(level2.data(), lv.coords_prev.data(),
                                  lv.coords.data(), lv.ncoords(), lv.n2, nbins,
                                  lv.row_len, lv.row_len_prev, tile,
                                  &collect_real, &fused, nthreads);
    INFO("nbins=" << nbins << " tile=" << tile << " nthreads=" << nthreads);
    CHECK(std::memcmp(fused.profiles.data(), expected.profiles.data(),
                      expected.profiles.size() * sizeof(float)) == 0);
}

TEST_CASE("Two fused final Fourier levels are bit-exact with the stored "
          "level L-1",
          "[ffa][fused][final2]") {
    const SizeType nbins   = GENERATE(SizeType{32}, SizeType{50});
    const SizeType tile    = GENERATE(SizeType{32}, SizeType{128});
    const int nthreads     = GENERATE(1, 3);
    const SizeType nbins_f = (nbins / 2) + 1;
    std::mt19937 rng(static_cast<unsigned>(nbins * 11 + tile));
    const Levels lv = make_levels(nbins, rng);

    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<ComplexType> level2(4 * lv.n2 * 2 * nbins_f);
    for (auto& x : level2) {
        x = ComplexType{dist(rng), dist(rng)};
    }
    std::vector<ComplexType> level1(2 * lv.ncoords_prev() * 2 * nbins_f);
    loki::utils::UninitVector<std::byte> step_table(SizeType{1} << 20U);
    loki::core::ffa_complex_iter(
        level2.data(), level1.data(), lv.coords_prev.data(), lv.ncoords_prev(),
        lv.n2, /*nsegments=*/2, nbins_f, nbins, step_table, 1);

    const SizeType stride = 2 * nbins_f;
    CollectedComplex expected{
        .stride   = stride,
        .profiles = std::vector<ComplexType>(lv.ncoords() * stride)};
    loki::core::ffa_complex_final_tiles(
        level1.data(), lv.coords.data(), lv.ncoords(), lv.ncoords_prev(),
        nbins_f, nbins, lv.row_len, lv.row_len_prev, tile, &collect_complex,
        &expected, nthreads);
    CollectedComplex fused{.stride = stride,
                           .profiles =
                               std::vector<ComplexType>(lv.ncoords() * stride)};
    loki::core::ffa_complex_final2_tiles(
        level2.data(), lv.coords_prev.data(), lv.coords.data(), lv.ncoords(),
        lv.ncoords_prev(), lv.n2, nbins_f, nbins, lv.row_len, lv.row_len_prev,
        tile, &collect_complex, &fused, nthreads);
    INFO("nbins=" << nbins << " tile=" << tile << " nthreads=" << nthreads);
    CHECK(std::memcmp(static_cast<const void*>(fused.profiles.data()),
                      static_cast<const void*>(expected.profiles.data()),
                      expected.profiles.size() * sizeof(ComplexType)) == 0);
}

TEST_CASE("Single-pass Taylor FFA resolve is bit-exact with per-half resolve",
          "[ffa][resolve]") {
    const int nthreads     = GENERATE(1, 4);
    const SizeType n_accel = GENERATE(SizeType{2}, SizeType{11});
    const SizeType n_freq  = 137;
    const SizeType nbins   = 64;
    const double tseg      = 0.065536;
    const SizeType level   = 5;

    SECTION("n_params = 2 (accel)") {
        const std::vector<SizeType> cur  = {n_accel, n_freq};
        const std::vector<SizeType> prev = {std::max<SizeType>(1, n_accel / 2),
                                            n_freq / 2};
        const std::vector<loki::ParamLimit> limits = {
            {.min = -50.0, .max = 50.0},
            {.min = 100.0, .max = 100.5},
        };
        const SizeType ncoords = cur[0] * cur[1];
        std::vector<FFACoord> expected(ncoords);
        std::vector<FFACoord> single_pass(ncoords);
        std::vector<FFACoord> row_pass(ncoords);

        loki::core::ffa_taylor_resolve_poly_batch(
            cur, prev, limits, expected, level, /*latter=*/0, tseg, nbins,
            /*n_params=*/2, 1);
        loki::core::ffa_taylor_resolve_poly_batch(
            cur, prev, limits, expected, level, /*latter=*/1, tseg, nbins,
            /*n_params=*/2, 1);

        loki::core::ffa_taylor_resolve_poly_batch(
            cur, prev, limits, single_pass, level, tseg, nbins, /*n_params=*/2,
            nthreads);
        loki::core::ffa_taylor_resolve_accel_rows(cur, prev, limits, row_pass,
                                                  level, tseg, nbins, 0, cur[0],
                                                  nthreads);

        CHECK(std::memcmp(single_pass.data(), expected.data(),
                          ncoords * sizeof(FFACoord)) == 0);
        CHECK(std::memcmp(row_pass.data(), expected.data(),
                          ncoords * sizeof(FFACoord)) == 0);
    }

    SECTION("n_params = 3 (jerk)") {
        const SizeType n_jerk            = 3;
        const std::vector<SizeType> cur  = {n_jerk, n_accel, n_freq};
        const std::vector<SizeType> prev = {
            2, std::max<SizeType>(1, n_accel / 2), n_freq / 2};
        const std::vector<loki::ParamLimit> limits = {
            {.min = -2.0, .max = 2.0},
            {.min = -50.0, .max = 50.0},
            {.min = 100.0, .max = 100.5},
        };
        const SizeType ncoords = cur[0] * cur[1] * cur[2];
        std::vector<FFACoord> expected(ncoords);
        std::vector<FFACoord> single_pass(ncoords);

        loki::core::ffa_taylor_resolve_poly_batch(
            cur, prev, limits, expected, level, /*latter=*/0, tseg, nbins,
            /*n_params=*/3, 1);
        loki::core::ffa_taylor_resolve_poly_batch(
            cur, prev, limits, expected, level, /*latter=*/1, tseg, nbins,
            /*n_params=*/3, 1);

        loki::core::ffa_taylor_resolve_poly_batch(
            cur, prev, limits, single_pass, level, tseg, nbins, /*n_params=*/3,
            nthreads);

        CHECK(std::memcmp(single_pass.data(), expected.data(),
                          ncoords * sizeof(FFACoord)) == 0);
    }
}

namespace {
struct PrunedScoreCtx {
    std::span<const SizeType> widths;
    float threshold{};
    SizeType scored_profiles{0};
    std::vector<loki::detection::SnrHit> hits;
};

void score_pruned_tile(const float* profiles,
                       SizeType coord_begin,
                       SizeType nprofiles,
                       SizeType nbins,
                       void* ctx) {
    auto& state = *static_cast<PrunedScoreCtx*>(ctx);
    state.scored_profiles += nprofiles;
    loki::detection::detail::BoxcarScorer3D scorer(nbins, state.widths.data(),
                                                   state.widths.size());
    loki::detection::detail::append_boxcar_hits(
        scorer, profiles, coord_begin, nprofiles, state.threshold, state.hits);
}

struct PrunedFourierScoreCtx {
    SizeType nbins{};
    std::span<const SizeType> widths;
    float threshold{};
    SizeType scored_profiles{0};
    std::vector<loki::detection::SnrHit> hits;
};

void score_pruned_fourier_tile(ComplexType* tile,
                               SizeType coord_begin,
                               SizeType count,
                               void* ctx) {
    auto& state = *static_cast<PrunedFourierScoreCtx*>(ctx);
    state.scored_profiles += count;
    std::vector<float> profiles(count * 2 * state.nbins);
    loki::lane_fft::inverse_batch(tile, profiles.data(), count * 2, state.nbins,
                                  /*nthreads=*/1);
    loki::detection::detail::BoxcarScorer3D scorer(
        state.nbins, state.widths.data(), state.widths.size());
    loki::detection::detail::append_boxcar_hits(scorer, profiles.data(),
                                                coord_begin, count,
                                                state.threshold, state.hits);
}
} // namespace

TEST_CASE("Two fused final levels with parent-bound pruning preserve all hits",
          "[ffa][fused][final2][prune]") {
    const SizeType nbins               = 64;
    const SizeType tile                = 128;
    const std::vector<SizeType> widths = {1, 2, 4, 8};
    constexpr float kThreshold         = 5.0F;
    std::mt19937 rng(42);
    const Levels lv = make_levels(nbins, rng);

    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> level2(4 * lv.n2 * 2 * nbins);
    for (SizeType s = 0; s < 4 * lv.n2; ++s) {
        float* e = level2.data() + (s * 2 * nbins);
        float* v = e + nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            e[j] = noise(rng);
            v[j] = 16.0F + (0.5F * std::abs(noise(rng)));
        }
    }
    // Inject strong pulses into a couple of level-(L-2) rows so several final
    // coordinates exceed S/N = 5 while most tiles remain pure noise.
    for (SizeType s : {SizeType{5}, SizeType{42}, SizeType{305}}) {
        float* e = level2.data() + (s * 2 * nbins);
        for (SizeType j = 10; j < 14; ++j) {
            e[j] += 40.0F;
        }
    }

    PrunedScoreCtx ref_ctx{.widths          = widths,
                           .threshold       = kThreshold,
                           .scored_profiles = 0,
                           .hits            = {}};
    loki::core::ffa_final2_scored(level2.data(), lv.coords_prev.data(),
                                  lv.coords.data(), lv.ncoords(), lv.n2, nbins,
                                  lv.row_len, lv.row_len_prev, tile,
                                  &score_pruned_tile, &ref_ctx, /*nthreads=*/1,
                                  /*threshold=*/0.0F, widths);

    PrunedScoreCtx pruned_ctx{.widths          = widths,
                              .threshold       = kThreshold,
                              .scored_profiles = 0,
                              .hits            = {}};
    loki::core::ffa_final2_scored(level2.data(), lv.coords_prev.data(),
                                  lv.coords.data(), lv.ncoords(), lv.n2, nbins,
                                  lv.row_len, lv.row_len_prev, tile,
                                  &score_pruned_tile, &pruned_ctx,
                                  /*nthreads=*/1, kThreshold, widths);

    REQUIRE(!ref_ctx.hits.empty());
    CHECK(pruned_ctx.scored_profiles < ref_ctx.scored_profiles);
    REQUIRE(pruned_ctx.hits.size() == ref_ctx.hits.size());
    for (SizeType i = 0; i < ref_ctx.hits.size(); ++i) {
        CHECK(pruned_ctx.hits[i].score_index == ref_ctx.hits[i].score_index);
        CHECK(pruned_ctx.hits[i].snr == ref_ctx.hits[i].snr);
    }
}

TEST_CASE("Two fused final Fourier levels with parent-bound pruning preserve "
          "all hits",
          "[ffa][fused][final2][prune][fourier]") {
    const SizeType nbins               = GENERATE(SizeType{64}, SizeType{50});
    const SizeType nbins_f             = (nbins / 2) + 1;
    const SizeType tile                = 128;
    const std::vector<SizeType> widths = {1, 2, 4, 8};
    constexpr float kThreshold         = 5.0F;
    std::mt19937 rng(42 + static_cast<unsigned>(nbins));
    const Levels lv = make_levels(nbins, rng);

    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> level2_real(4 * lv.n2 * 2 * nbins);
    for (SizeType s = 0; s < 4 * lv.n2; ++s) {
        float* e = level2_real.data() + (s * 2 * nbins);
        float* v = e + nbins;
        for (SizeType j = 0; j < nbins; ++j) {
            e[j] = noise(rng);
            v[j] = 16.0F + (0.5F * std::abs(noise(rng)));
        }
    }
    for (SizeType s : {SizeType{5}, SizeType{42}, SizeType{305}}) {
        float* e = level2_real.data() + (s * 2 * nbins);
        for (SizeType j = 10; j < 14; ++j) {
            e[j] += 40.0F;
        }
    }
    std::vector<ComplexType> level2(4 * lv.n2 * 2 * nbins_f);
    loki::lane_fft::forward_batch(level2_real.data(), level2.data(),
                                  4 * lv.n2 * 2, nbins, /*nthreads=*/1);

    PrunedFourierScoreCtx ref_ctx{.nbins           = nbins,
                                  .widths          = widths,
                                  .threshold       = kThreshold,
                                  .scored_profiles = 0,
                                  .hits            = {}};
    loki::core::ffa_complex_final2_tiles(
        level2.data(), lv.coords_prev.data(), lv.coords.data(), lv.ncoords(),
        lv.ncoords_prev(), lv.n2, nbins_f, nbins, lv.row_len, lv.row_len_prev,
        tile, &score_pruned_fourier_tile, &ref_ctx, /*nthreads=*/1,
        /*threshold=*/0.0F, widths);

    PrunedFourierScoreCtx pruned_ctx{.nbins           = nbins,
                                     .widths          = widths,
                                     .threshold       = kThreshold,
                                     .scored_profiles = 0,
                                     .hits            = {}};
    loki::core::ffa_complex_final2_tiles(
        level2.data(), lv.coords_prev.data(), lv.coords.data(), lv.ncoords(),
        lv.ncoords_prev(), lv.n2, nbins_f, nbins, lv.row_len, lv.row_len_prev,
        tile, &score_pruned_fourier_tile, &pruned_ctx, /*nthreads=*/1,
        kThreshold, widths);

    REQUIRE(!ref_ctx.hits.empty());
    CHECK(pruned_ctx.scored_profiles < ref_ctx.scored_profiles);
    REQUIRE(pruned_ctx.hits.size() == ref_ctx.hits.size());
    for (SizeType i = 0; i < ref_ctx.hits.size(); ++i) {
        CHECK(pruned_ctx.hits[i].score_index == ref_ctx.hits[i].score_index);
        CHECK(pruned_ctx.hits[i].snr == ref_ctx.hits[i].snr);
    }
}
