#include <complex>
#include <numeric>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"
#include "loki/utils/world_tree.hpp"

using loki::ComplexType;
using loki::SizeType;
using loki::memory::WorldTree;

namespace {

template <typename T> T make_value(SizeType i) {
    if constexpr (std::is_same_v<T, ComplexType>) {
        return ComplexType(static_cast<float>(i), -static_cast<float>(i));
    } else {
        return static_cast<T>(i);
    }
}

} // namespace

TEMPLATE_TEST_CASE("WorldTree::add_initial_scattered matches add_initial",
                   "[world_tree]",
                   float,
                   ComplexType) {
    constexpr SizeType kCapacity = 64;
    constexpr SizeType kNParams  = 2;
    constexpr SizeType kNBins    = 8;
    constexpr SizeType kBatch    = 16;
    constexpr SizeType kNSeeds   = 40;

    WorldTree<TestType> tree_ref(kCapacity, kNParams, kNBins, kBatch);
    WorldTree<TestType> tree_sc(kCapacity, kNParams, kNBins, kBatch);
    const auto leaves_stride = tree_ref.get_leaves_stride();
    const auto folds_stride  = tree_ref.get_folds_stride();

    std::vector<double> leaves(kNSeeds * leaves_stride);
    std::vector<TestType> folds(kNSeeds * folds_stride);
    std::vector<float> scores(kNSeeds);
    for (SizeType i = 0; i < kNSeeds; ++i) {
        for (SizeType j = 0; j < leaves_stride; ++j) {
            leaves[(i * leaves_stride) + j] =
                static_cast<double>((i * 1000) + j);
        }
        for (SizeType j = 0; j < folds_stride; ++j) {
            folds[(i * folds_stride) + j] = make_value<TestType>((i * 100) + j);
        }
        scores[i] = static_cast<float>(i) * 0.25F;
    }

    SECTION("identity index set") {
        std::vector<SizeType> identity(kNSeeds);
        std::iota(identity.begin(), identity.end(), SizeType{0});

        tree_ref.add_initial(leaves, folds, scores, kNSeeds);
        tree_sc.add_initial_scattered(leaves, folds, scores, identity, kNSeeds);

        REQUIRE(tree_sc.get_size() == tree_ref.get_size());
        REQUIRE(tree_sc.get_size() == kNSeeds);
        REQUIRE(tree_sc.get_score_max() == tree_ref.get_score_max());
        REQUIRE(tree_sc.get_score_min() == tree_ref.get_score_min());
        for (SizeType i = 0; i < kNSeeds; ++i) {
            REQUIRE(tree_sc.get_scores()[i] == tree_ref.get_scores()[i]);
            for (SizeType j = 0; j < leaves_stride; ++j) {
                REQUIRE(tree_sc.get_leaves()[(i * leaves_stride) + j] ==
                        tree_ref.get_leaves()[(i * leaves_stride) + j]);
            }
            for (SizeType j = 0; j < folds_stride; ++j) {
                REQUIRE(tree_sc.get_folds()[(i * folds_stride) + j] ==
                        tree_ref.get_folds()[(i * folds_stride) + j]);
            }
        }
    }

    SECTION("subset keeps the selected seeds in order") {
        // Keep every third seed
        std::vector<SizeType> keep;
        for (SizeType i = 0; i < kNSeeds; i += 3) {
            keep.push_back(i);
        }
        // Buffer is oversized on purpose: only the first n are used.
        keep.resize(kNSeeds, 0);
        const auto n_keep = (kNSeeds + 2) / 3;

        // Pre-fill with something to verify reset() is applied
        tree_sc.add_initial(leaves, folds, scores, kNSeeds);
        tree_sc.add_initial_scattered(leaves, folds, scores, keep, n_keep);

        REQUIRE(tree_sc.get_size() == n_keep);
        for (SizeType k = 0; k < n_keep; ++k) {
            const auto src = keep[k];
            REQUIRE(tree_sc.get_scores()[k] == scores[src]);
            for (SizeType j = 0; j < leaves_stride; ++j) {
                REQUIRE(tree_sc.get_leaves()[(k * leaves_stride) + j] ==
                        leaves[(src * leaves_stride) + j]);
            }
            for (SizeType j = 0; j < folds_stride; ++j) {
                REQUIRE(tree_sc.get_folds()[(k * folds_stride) + j] ==
                        folds[(src * folds_stride) + j]);
            }
        }
        REQUIRE(tree_sc.get_score_max() == scores[keep[n_keep - 1]]);
        REQUIRE(tree_sc.get_score_min() == scores[keep[0]]);
    }

    SECTION("drop_best removes the highest-score seed") {
        tree_sc.add_initial(leaves, folds, scores, kNSeeds);
        REQUIRE(tree_sc.get_score_max() == scores[kNSeeds - 1]);
        REQUIRE(tree_sc.drop_best());
        REQUIRE(tree_sc.get_size() == kNSeeds - 1);
        REQUIRE(tree_sc.get_score_max() == scores[kNSeeds - 2]);
        while (tree_sc.get_size() > 0) {
            REQUIRE(tree_sc.drop_best());
        }
        REQUIRE_FALSE(tree_sc.drop_best());
        REQUIRE(tree_sc.get_size() == 0);
    }

    SECTION("empty selection yields an empty tree") {
        std::vector<SizeType> keep(kNSeeds, 0);
        tree_sc.add_initial(leaves, folds, scores, kNSeeds);
        tree_sc.add_initial_scattered(leaves, folds, scores, keep, 0);
        REQUIRE(tree_sc.get_size() == 0);
    }

    SECTION("out-of-range index is rejected") {
        std::vector<SizeType> keep{0, 1, kNSeeds + 5};
        REQUIRE_THROWS(
            tree_sc.add_initial_scattered(leaves, folds, scores, keep, 3));
        std::vector<SizeType> too_short{0, 1};
        REQUIRE_THROWS(
            tree_sc.add_initial_scattered(leaves, folds, scores, too_short, 3));
    }
}
