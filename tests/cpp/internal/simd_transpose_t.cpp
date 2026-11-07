#include <array>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <xsimd/xsimd.hpp>

#include "loki/common/types.hpp"

#include "lib/cpu/simd_utils.hpp"

// The block transposes of simd_utils against a scalar reference. The AVX2
// and AVX-512 primitives compile only on x86 builds with those ISAs.

namespace {

using loki::SizeType;

// Distinct value per (row, column), so a misplaced element shows.
float value_at(SizeType r, SizeType c) {
    return static_cast<float>((r * 1000) + c) + 0.5F;
}

} // namespace

TEST_CASE("simd_utils::transpose matches a scalar transpose", "[simd_utils]") {
    using Batch                = xsimd::batch<float>;
    constexpr SizeType kLanes  = Batch::size;
    constexpr SizeType kStride = (5 * kLanes) + 3; // rows are not packed
    for (const SizeType nbins : {kLanes, 3 * kLanes, 5 * kLanes}) {
        std::vector<float> src(kLanes * kStride);
        for (SizeType r = 0; r < kLanes; ++r) {
            for (SizeType c = 0; c < kStride; ++c) {
                src[(r * kStride) + c] = value_at(r, c);
            }
        }
        std::vector<float> dst(nbins * kLanes, -1.0F);
        loki::simd_utils::transpose<Batch>(src.data(), dst.data(), nbins,
                                           kStride);
        for (SizeType c = 0; c < nbins; ++c) {
            for (SizeType r = 0; r < kLanes; ++r) {
                REQUIRE(dst[(c * kLanes) + r] == value_at(r, c));
            }
        }
    }
}

#if defined(__AVX2__)
TEST_CASE("simd_utils::transpose8x8 writes dst[c][r] = src[r][c]",
          "[simd_utils]") {
    constexpr SizeType kN  = 8;
    constexpr SizeType kC0 = 3; // unaligned column offset
    std::array<std::vector<float>, kN> rows;
    std::array<const float*, kN> src{};
    for (SizeType r = 0; r < kN; ++r) {
        rows[r].resize(kC0 + kN + r); // uneven rows
        for (SizeType c = 0; c < rows[r].size(); ++c) {
            rows[r][c] = value_at(r, c);
        }
        src[r] = rows[r].data() + kC0;
    }
    std::vector<float> out(kN * 11, -1.0F);
    std::array<float*, kN> dst{};
    for (SizeType c = 0; c < kN; ++c) {
        dst[c] = out.data() + (c * 11);
    }
    loki::simd_utils::transpose8x8(src.data(), dst.data());
    for (SizeType c = 0; c < kN; ++c) {
        for (SizeType r = 0; r < kN; ++r) {
            REQUIRE(dst[c][r] == value_at(r, kC0 + c));
        }
        for (SizeType k = kN; k < 11; ++k) {
            REQUIRE(out[(c * 11) + k] == -1.0F);
        }
    }
}
#endif

#if defined(__AVX512F__)
TEST_CASE("simd_utils::transpose16x16 writes dst[c * stride + r] = src[r][c]",
          "[simd_utils]") {
    constexpr SizeType kN      = 16;
    constexpr SizeType kC0     = 5;
    constexpr SizeType kStride = 19;
    std::array<std::vector<float>, kN> rows;
    std::array<const float*, kN> src{};
    for (SizeType r = 0; r < kN; ++r) {
        rows[r].resize(kC0 + kN + r);
        for (SizeType c = 0; c < rows[r].size(); ++c) {
            rows[r][c] = value_at(r, c);
        }
        src[r] = rows[r].data() + kC0;
    }
    std::vector<float> out(kN * kStride, -1.0F);
    loki::simd_utils::transpose16x16(src.data(), out.data(), kStride);
    for (SizeType c = 0; c < kN; ++c) {
        for (SizeType r = 0; r < kN; ++r) {
            REQUIRE(out[(c * kStride) + r] == value_at(r, kC0 + c));
        }
        for (SizeType k = kN; k < kStride; ++k) {
            REQUIRE(out[(c * kStride) + k] == -1.0F);
        }
    }
}
#endif
