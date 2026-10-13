#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/coord.hpp"
#include "loki/common/types.hpp"

#include "lib/core/kernels.hpp"

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
    loki::core::ffa_complex_iter(level2.data(), level1.data(),
                                 lv.coords_prev.data(), lv.ncoords_prev(),
                                 lv.n2, /*nsegments=*/2, nbins_f, nbins, 1);

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
