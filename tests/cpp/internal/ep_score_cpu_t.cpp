#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/plans.hpp"
#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/search/configs.hpp"

#include "lib/core/dynamic.hpp"
#include "lib/core/kernels.hpp"
#include "lib/cpu/boxcar_kernels.hpp"

// CPU pruning scorers: the exact SIMD max scorer, and the fused
// shift-add + score tiles of the prune functors.

namespace {

using loki::ComplexType;
using loki::ParamLimit;
using loki::SizeType;
using loki::search::PulsarSearchConfig;

PulsarSearchConfig
make_cfg(SizeType nbins, bool use_fourier, bool use_kadane = false) {
    constexpr SizeType kNsamps = 1U << 16U;
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = 140.0, .max = 145.0},
    };
    return {kNsamps,
            64e-6,
            nbins,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.2,
            use_fourier,
            /*nthreads=*/1,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*bseg_ffa=*/kNsamps / 8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2,
            /*p_orb_min=*/1e-5,
            /*m_c_max=*/10.0,
            /*m_p_min=*/1.4,
            /*propagator_significance=*/2.0,
            /*validation_significance=*/5.0,
            /*use_conservative_tile=*/false,
            use_kadane};
}

template <typename FoldType>
std::unique_ptr<loki::core::PruneDPFuncts<FoldType>>
make_functs(const PulsarSearchConfig& cfg) {
    const loki::plans::FFAPlan<FoldType> plan(cfg);
    return loki::core::create_prune_dp_functs<FoldType>(
        "taylor", plan.get_param_counts().back(),
        plan.get_dparams_actual().back(), plan.get_nsegments().back(),
        plan.get_tsegments().back(), cfg, /*batch_size=*/64,
        /*branch_max=*/16);
}

/// Random (e, v) profiles [n][2][nbins]; about 1 in 20 variances <= 0.
std::vector<float>
random_profiles(SizeType n, SizeType nbins, std::mt19937& rng) {
    std::normal_distribution<float> e_dist(0.0F, 1.0F);
    std::uniform_real_distribution<float> v_dist(-0.05F, 1.0F);
    std::vector<float> folds(n * 2 * nbins);
    for (SizeType p = 0; p < n; ++p) {
        for (SizeType j = 0; j < nbins; ++j) {
            folds[(p * 2 * nbins) + j]         = e_dist(rng);
            folds[(p * 2 * nbins) + nbins + j] = v_dist(rng);
        }
    }
    return folds;
}

template <typename FoldType>
std::vector<FoldType>
random_folds(SizeType n, SizeType stride, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<FoldType> folds(n * stride);
    for (auto& x : folds) {
        if constexpr (std::is_same_v<FoldType, ComplexType>) {
            x = ComplexType{dist(rng), dist(rng)};
        } else {
            x = dist(rng) + 0.5F; // mostly positive variances
        }
    }
    return folds;
}

/// The fused shift_add_score against shift_add then score_and_filter, on
/// a tree whose active region wraps around the circular buffer.
template <typename FoldType>
void check_fused_matches_unfused(const PulsarSearchConfig& cfg,
                                 SizeType n_leaves) {
    constexpr bool kComplex = std::is_same_v<FoldType, ComplexType>;
    const SizeType nbins    = cfg.get_nbins();
    const SizeType stride =
        2 * (kComplex ? cfg.get_nbins_f() : cfg.get_nbins());
    constexpr SizeType kCapacity = 97;
    constexpr SizeType kNffa     = 41;
    const SizeType physical_start{80}; // logical 17.. wraps
    std::mt19937 rng(1234U + static_cast<std::uint32_t>(nbins + n_leaves));

    const auto folds_tree = random_folds<FoldType>(kCapacity, stride, rng);
    const auto folds_ffa  = random_folds<FoldType>(kNffa, stride, rng);
    std::uniform_int_distribution<SizeType> tree_idx(0, kCapacity - 1);
    std::uniform_int_distribution<SizeType> ffa_idx(0, kNffa - 1);
    std::uniform_real_distribution<float> shift_dist(0.0F,
                                                     static_cast<float>(nbins));
    std::vector<SizeType> origins(n_leaves);
    std::vector<SizeType> indices_ffa(n_leaves);
    std::vector<float> phase_shift(n_leaves);
    for (SizeType i = 0; i < n_leaves; ++i) {
        origins[i]     = tree_idx(rng);
        indices_ffa[i] = ffa_idx(rng);
        phase_shift[i] = shift_dist(rng);
    }

    auto functs_a = make_functs<FoldType>(cfg);
    auto functs_b = make_functs<FoldType>(cfg);

    // Unfused reference.
    std::vector<FoldType> folds_a(n_leaves * stride);
    std::vector<float> scores_a(n_leaves);
    std::vector<SizeType> indices_a = origins;
    functs_a->shift_add(folds_tree, indices_a, folds_ffa, indices_ffa,
                        phase_shift, folds_a, n_leaves, physical_start,
                        kCapacity);
    // Threshold at the median score, so about half the leaves pass.
    functs_a->score_and_filter(folds_a, scores_a, indices_a,
                               std::numeric_limits<float>::lowest(), n_leaves);
    auto sorted = scores_a;
    std::ranges::sort(sorted);
    const float threshold = sorted[n_leaves / 2];
    indices_a             = origins;
    const auto n_pass_a   = functs_a->score_and_filter(
        folds_a, scores_a, indices_a, threshold, n_leaves);

    // Fused.
    std::vector<FoldType> folds_b(n_leaves * stride);
    std::vector<float> scores_b(n_leaves);
    std::vector<SizeType> indices_b = origins;
    const auto n_pass_b             = functs_b->shift_add_score(
        folds_tree, indices_b, folds_ffa, indices_ffa, phase_shift,
        loki::core::PassingFolds<FoldType>{.data = folds_b.data()}, scores_b,
        threshold, n_leaves, physical_start, kCapacity);

    // Fused, passing folds appended to a ring (the world-tree direct path).
    constexpr SizeType kRing = 131;
    const SizeType ring_start{kRing - 7};
    std::vector<FoldType> ring(kRing * stride);
    std::vector<float> scores_c(n_leaves);
    std::vector<SizeType> indices_c = origins;
    const auto n_pass_c             = functs_b->shift_add_score(
        folds_tree, indices_c, folds_ffa, indices_ffa, phase_shift,
        loki::core::PassingFolds<FoldType>{.data          = ring.data(),
                                           .ring_start    = ring_start,
                                           .ring_capacity = kRing},
        scores_c, threshold, n_leaves, physical_start, kCapacity);
    REQUIRE(n_pass_c == n_pass_a);
    REQUIRE(indices_c == indices_b);

    REQUIRE(n_pass_b == n_pass_a);
    REQUIRE(n_pass_a > 0);
    for (SizeType i = 0; i < n_leaves; ++i) {
        REQUIRE(std::bit_cast<std::uint32_t>(scores_b[i]) ==
                std::bit_cast<std::uint32_t>(scores_a[i]));
    }
    const auto same_bits = [](FoldType x, FoldType y) {
        if constexpr (kComplex) {
            return std::bit_cast<std::uint64_t>(x) ==
                   std::bit_cast<std::uint64_t>(y);
        } else {
            return std::bit_cast<std::uint32_t>(x) ==
                   std::bit_cast<std::uint32_t>(y);
        }
    };
    for (SizeType k = 0; k < n_pass_a; ++k) {
        REQUIRE(indices_b[k] == indices_a[k]);
        const auto at  = indices_a[k] * stride;
        const auto ref = std::span(folds_a).subspan(at, stride);
        REQUIRE(std::ranges::equal(std::span(folds_b).subspan(at, stride), ref,
                                   same_bits));
        const auto slot = (ring_start + k) % kRing;
        REQUIRE(std::ranges::equal(
            std::span(ring).subspan(slot * stride, stride), ref, same_bits));
    }
}

} // namespace

TEST_CASE("BoxcarScorer3D max is bit-exact with the scalar max scorer",
          "[ep][score]") {
    for (const SizeType nbins :
         {7U, 16U, 32U, 50U, 64U, 100U, 128U, 256U, 509U, 1024U}) {
        for (const auto [ducy, wtsp] : {std::pair{0.3, 1.2}, {0.5, 1.1}}) {
            const auto widths =
                loki::detection::generate_box_width_trials(nbins, ducy, wtsp);
            constexpr SizeType kN = 45; // two full groups and a ragged one
            std::mt19937 rng(static_cast<std::uint32_t>(nbins));
            const auto folds = random_profiles(kN, nbins, rng);

            loki::detection::BoxcarWidthsCache cache(widths, nbins);
            std::vector<float> ref(kN);
            loki::detection::snr_boxcar_3d_max_with_cache(folds, ref, kN, nbins,
                                                          cache);

            loki::detection::detail::BoxcarScorer3D scorer(nbins, widths.data(),
                                                           widths.size());
            std::vector<float> rows(kN);
            scorer.score_max(folds.data(), kN, rows.data());

            // Lane layout [nbins][16] of each group of 16 profiles.
            constexpr SizeType kL =
                loki::detection::detail::BoxcarScorer3D::kLaneStride;
            std::vector<float> lanes(kN);
            std::vector<float> e(nbins * kL);
            std::vector<float> v(nbins * kL);
            for (SizeType p0 = 0; p0 < kN; p0 += kL) {
                const SizeType m = std::min(kL, kN - p0);
                std::ranges::fill(e, 0.0F);
                std::ranges::fill(v, 1.0F);
                for (SizeType l = 0; l < m; ++l) {
                    for (SizeType j = 0; j < nbins; ++j) {
                        e[(j * kL) + l] = folds[((p0 + l) * 2 * nbins) + j];
                        v[(j * kL) + l] =
                            folds[((p0 + l) * 2 * nbins) + nbins + j];
                    }
                }
                scorer.score_lanes_max(e.data(), v.data(), m,
                                       lanes.data() + p0);
            }
            for (SizeType p = 0; p < kN; ++p) {
                INFO("nbins " << nbins << " profile " << p);
                REQUIRE(std::bit_cast<std::uint32_t>(rows[p]) ==
                        std::bit_cast<std::uint32_t>(ref[p]));
                REQUIRE(std::bit_cast<std::uint32_t>(lanes[p]) ==
                        std::bit_cast<std::uint32_t>(ref[p]));
            }
        }
    }
}

TEST_CASE("Fused EP shift-add and score matches the unfused chain",
          "[ep][score]") {
    for (const SizeType nbins : {32U, 50U, 64U, 100U}) {
        for (const SizeType n_leaves : {1U, 16U, 37U, 100U}) {
            for (const bool kadane : {false, true}) {
                INFO("nbins " << nbins << " n_leaves " << n_leaves << " kadane "
                              << kadane);
                check_fused_matches_unfused<ComplexType>(
                    make_cfg(nbins, true, kadane), n_leaves);
                check_fused_matches_unfused<float>(
                    make_cfg(nbins, false, kadane), n_leaves);
            }
        }
    }
}

TEST_CASE("EP complex shift-add is independent of the leaf blocking",
          "[ep][score]") {
    constexpr SizeType kNbins  = 64;
    constexpr SizeType kNbinsF = (kNbins / 2) + 1;
    constexpr SizeType kStride = 2 * kNbinsF;
    constexpr SizeType kLeaves = 53;
    constexpr SizeType kOffset = 5; // blocks start at different leaves
    constexpr SizeType kCap    = kLeaves;
    std::mt19937 rng(7U);
    const auto tree = random_folds<ComplexType>(kCap, kStride, rng);
    const auto ffa  = random_folds<ComplexType>(kLeaves, kStride, rng);
    std::vector<SizeType> idx(kLeaves);
    std::vector<float> shifts(kLeaves);
    std::uniform_real_distribution<float> shift_dist(0.0F, kNbins);
    for (SizeType i = 0; i < kLeaves; ++i) {
        idx[i]    = i;
        shifts[i] = shift_dist(rng);
    }
    std::vector<ComplexType> all(kLeaves * kStride);
    loki::core::shift_add_linear_complex_batch(
        tree.data(), idx.data(), ffa.data(), idx.data(), shifts.data(),
        all.data(), kNbinsF, kNbins, kLeaves, 0, kCap);
    std::vector<ComplexType> part((kLeaves - kOffset) * kStride);
    loki::core::shift_add_linear_complex_batch(
        tree.data(), idx.data() + kOffset, ffa.data(), idx.data() + kOffset,
        shifts.data() + kOffset, part.data(), kNbinsF, kNbins,
        kLeaves - kOffset, 0, kCap);
    for (SizeType i = 0; i < part.size(); ++i) {
        REQUIRE(std::bit_cast<std::uint64_t>(part[i]) ==
                std::bit_cast<std::uint64_t>(all[(kOffset * kStride) + i]));
    }

    // Accuracy against a double-precision phase rotation.
    double max_err = 0.0;
    for (SizeType i = 0; i < kLeaves; ++i) {
        for (SizeType k = 0; k < kStride; ++k) {
            const SizeType kh = k % kNbinsF;
            const double angle =
                -2.0 * std::numbers::pi * static_cast<double>(kh) *
                static_cast<double>(shifts[i]) / static_cast<double>(kNbins);
            const std::complex<double> head(ffa[(i * kStride) + k].real(),
                                            ffa[(i * kStride) + k].imag());
            const std::complex<double> ref =
                std::complex<double>(tree[(i * kStride) + k].real(),
                                     tree[(i * kStride) + k].imag()) +
                (head * std::polar(1.0, angle));
            const auto got = all[(i * kStride) + k];
            max_err        = std::max(
                max_err,
                std::abs(std::complex<double>(got.real(), got.imag()) - ref));
        }
    }
    CHECK(max_err < 1e-4);
}
