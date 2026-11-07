#include "lib/utils/world_tree.hpp"

#include <numeric>
#include <span>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"

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

    SECTION("add_initial takes the leading seeds of a larger batch") {
        // A sweep sizes its seed buffers for its largest chunk
        constexpr SizeType kNUsed = 25;
        tree_sc.add_initial(leaves, folds, scores, kNUsed);
        REQUIRE(tree_sc.get_size() == kNUsed);
        for (SizeType i = 0; i < kNUsed; ++i) {
            REQUIRE(tree_sc.get_scores()[i] == scores[i]);
            for (SizeType j = 0; j < leaves_stride; ++j) {
                REQUIRE(tree_sc.get_leaves()[(i * leaves_stride) + j] ==
                        leaves[(i * leaves_stride) + j]);
            }
            for (SizeType j = 0; j < folds_stride; ++j) {
                REQUIRE(tree_sc.get_folds()[(i * folds_stride) + j] ==
                        folds[(i * folds_stride) + j]);
            }
        }
        REQUIRE(tree_sc.get_score_max() == scores[kNUsed - 1]);
        REQUIRE_THROWS(tree_sc.add_initial(
            leaves, folds, std::span<const float>(scores).first(kNUsed - 1),
            kNUsed));
    }
}

TEST_CASE("WorldTree overflow threshold uses descending median and keeps "
          "score == threshold",
          "[world_tree]") {
    constexpr SizeType kCapacity = 8;
    constexpr SizeType kNParams  = 1;
    constexpr SizeType kNBins    = 4;
    constexpr SizeType kBatch    = 8;

    WorldTree<float> tree(kCapacity, kNParams, kNBins, kBatch);
    const auto leaves_stride = tree.get_leaves_stride();
    const auto folds_stride  = tree.get_folds_stride();

    const float exact_thresh =
        std::nextafter(35.0F, std::numeric_limits<float>::max());
    // The tree holds [10, 20, 30, exact_thresh, 50, 60]; a batch of
    // [25, 35, 45, 55] makes 10 candidates for 8 slots. Sorted descending:
    // [60, 55, 50, 45, exact_thresh, 35, 30, 25, 20, 10]; the median (rank 5,
    // 35) bumped by one ULP is exact_thresh, so the leaves scoring exactly
    // the threshold survive in the tree (>=) as they do in the batch:
    // 3 tree leaves (exact_thresh, 50, 60) and 2 batch leaves (45, 55).
    std::vector<double> batch1_leaves(6 * leaves_stride, 1.0);
    std::vector<float> batch1_folds(6 * folds_stride, 1.0F);
    std::vector<float> batch1_scores{10.0F,        20.0F, 30.0F,
                                     exact_thresh, 50.0F, 60.0F};
    std::vector<SizeType> batch1_indices{0, 1, 2, 3, 4, 5};

    tree.prepare_in_place_update();
    REQUIRE(tree.add_batch_scattered(batch1_leaves, batch1_folds, batch1_scores,
                                     batch1_indices, 5.0F, 6) == 5.0F);

    std::vector<double> batch2_leaves(4 * leaves_stride, 2.0);
    std::vector<float> batch2_folds(4 * folds_stride, 2.0F);
    std::vector<float> batch2_scores{25.0F, 35.0F, 45.0F, 55.0F};
    std::vector<SizeType> batch2_indices{0, 1, 2, 3};

    const float new_thresh = tree.add_batch_scattered(
        batch2_leaves, batch2_folds, batch2_scores, batch2_indices, 5.0F, 4);
    REQUIRE(new_thresh == exact_thresh);
    REQUIRE(tree.get_size() == 5);
}
