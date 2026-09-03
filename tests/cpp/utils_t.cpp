#include <cstddef>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/utils.hpp"

using Catch::Matchers::WithinAbs;

namespace {

constexpr double kNear = 1e-9;

} // namespace

TEST_CASE("diff_max", "[utils]") {
    SECTION("Uniform offset") {
        std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> y = {0.5F, 1.5F, 2.5F, 3.5F, 4.5F};
        REQUIRE(loki::utils::diff_max(x.data(), y.data(), x.size()) == 0.5F);
    }

    SECTION("Positive max difference") {
        std::vector<float> x = {1.0F, 4.0F, 3.0F, 2.0F, 10.0F};
        std::vector<float> y = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
        REQUIRE(loki::utils::diff_max(x.data(), y.data(), x.size()) == 9.5F);
    }

    SECTION("Negative max difference") {
        std::vector<float> x = {1.0F, 4.0F, 3.0F, 2.0F, 10.0F};
        std::vector<float> y = {20.0F, 20.0F, 20.0F, 20.0F, 20.0F};
        REQUIRE(loki::utils::diff_max(x.data(), y.data(), x.size()) == -10.0F);
    }

    SECTION("Empty input returns lowest float") {
        std::vector<float> x = {1.0F};
        std::vector<float> y = {2.0F};
        REQUIRE(loki::utils::diff_max(x.data(), y.data(), 0) ==
                std::numeric_limits<float>::lowest());
    }
}

TEST_CASE("circular_prefix_sum", "[utils]") {
    const std::vector<float> x             = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> k_full_prefix = {1.0F, 3.0F, 6.0F, 10.0F, 15.0F};

    SECTION("Exact cycle length") {
        std::vector<float> out(x.size());
        loki::utils::circular_prefix_sum(x.data(), out.data(), x.size(),
                                         x.size());
        REQUIRE(out == k_full_prefix);
    }

    SECTION("Partial output shorter than input cycle") {
        std::vector<float> out(x.size() - 2);
        loki::utils::circular_prefix_sum(x.data(), out.data(), x.size(),
                                         out.size());
        REQUIRE(out == std::vector<float>{1.0F, 3.0F, 6.0F});
    }

    SECTION("Single wrap around") {
        std::vector<float> out(x.size() + x.size());
        loki::utils::circular_prefix_sum(x.data(), out.data(), x.size(),
                                         out.size());
        const std::vector<float> expected = {
            1.0F, 3.0F, 6.0F, 10.0F, 15.0F, 16.0F, 18.0F, 21.0F, 25.0F, 30.0F,
        };
        REQUIRE(out == expected);
    }

    SECTION("Multiple wraps use general path") {
        constexpr loki::SizeType kNbins = 3;
        const std::vector<float> bins   = {1.0F, 2.0F, 3.0F};
        constexpr loki::SizeType kNsum  = 8;
        std::vector<float> out(kNsum);
        loki::utils::circular_prefix_sum(bins.data(), out.data(), kNbins,
                                         kNsum);
        // 1, 3, 6 | 7, 9, 12 | 13, 15
        const std::vector<float> expected = {
            1.0F, 3.0F, 6.0F, 7.0F, 9.0F, 12.0F, 13.0F, 15.0F,
        };
        REQUIRE(out == expected);
    }

    SECTION("Zero nbins or nsum is a no-op") {
        std::vector<float> out(4, -1.0F);
        loki::utils::circular_prefix_sum(x.data(), out.data(), 0, 4);
        loki::utils::circular_prefix_sum(x.data(), out.data(), x.size(), 0);
        REQUIRE(out == std::vector<float>(4, -1.0F));
    }
}

TEST_CASE("find_nearest_sorted_idx", "[utils]") {
    const std::vector<double> values = {0.0, 1.0, 2.0, 4.0, 8.0};

    SECTION("Exact match") {
        REQUIRE(loki::utils::find_nearest_sorted_idx(values, 4.0) == 3);
    }

    SECTION("Between values picks closer neighbour") {
        REQUIRE(loki::utils::find_nearest_sorted_idx(values, 3.0) == 2);
        REQUIRE(loki::utils::find_nearest_sorted_idx(values, 3.9) == 3);
    }

    SECTION("Tie breaks toward lower index") {
        REQUIRE(loki::utils::find_nearest_sorted_idx(values, 1.5) == 1);
    }

    SECTION("Below range clamps to first element") {
        REQUIRE(loki::utils::find_nearest_sorted_idx(values, -100.0) == 0);
    }

    SECTION("Above range clamps to last element") {
        REQUIRE(loki::utils::find_nearest_sorted_idx(values, 100.0) ==
                values.size() - 1);
    }

    SECTION("Empty array throws") {
        REQUIRE_THROWS_AS(loki::utils::find_nearest_sorted_idx({}, 1.0),
                          std::invalid_argument);
    }
}

TEST_CASE("find_nearest_sorted_idx_scan", "[utils]") {
    const std::vector<double> values = {0.0, 1.0, 2.0, 4.0, 8.0};

    SECTION("Matches binary search on scattered queries") {
        loki::SizeType hint = 0;
        REQUIRE(loki::utils::find_nearest_sorted_idx_scan(values, 0.0, hint) ==
                0);
        REQUIRE(hint == 0);

        REQUIRE(loki::utils::find_nearest_sorted_idx_scan(values, 4.0, hint) ==
                3);
        REQUIRE(hint == 3);

        REQUIRE(loki::utils::find_nearest_sorted_idx_scan(values, 7.9, hint) ==
                4);
        REQUIRE(hint == 4);
    }

    SECTION("Hint is clamped when out of range") {
        loki::SizeType hint = values.size() + 10;
        REQUIRE(loki::utils::find_nearest_sorted_idx_scan(values, 1.0, hint) ==
                1);
        REQUIRE(hint == 1);
    }
}

TEST_CASE("find_neighbouring_indices", "[utils][find_neighbouring_indices]") {
    const std::vector<loki::SizeType> indices = {0, 2, 4, 6, 8, 10};

    SECTION("Target in the middle") {
        REQUIRE(loki::utils::find_neighbouring_indices(indices, 4, 3) ==
                std::vector<loki::SizeType>{2, 4, 6});
    }

    SECTION("Target at the beginning") {
        REQUIRE(loki::utils::find_neighbouring_indices(indices, 0, 3) ==
                std::vector<loki::SizeType>{0, 2, 4});
    }

    SECTION("Target at the end") {
        REQUIRE(loki::utils::find_neighbouring_indices(indices, 10, 3) ==
                std::vector<loki::SizeType>{6, 8, 10});
    }

    SECTION("Target not in indices") {
        REQUIRE(loki::utils::find_neighbouring_indices(indices, 5, 3) ==
                std::vector<loki::SizeType>{4, 6, 8});
    }

    SECTION("Num larger than indices size returns all") {
        REQUIRE(loki::utils::find_neighbouring_indices(indices, 4, 10) ==
                indices);
    }

    SECTION("Empty indices throws") {
        REQUIRE_THROWS_AS(loki::utils::find_neighbouring_indices({}, 0, 1),
                          std::invalid_argument);
    }

    SECTION("Num is zero throws") {
        REQUIRE_THROWS_AS(loki::utils::find_neighbouring_indices(indices, 4, 0),
                          std::invalid_argument);
    }

    SECTION("Single element indices") {
        const std::vector<loki::SizeType> single = {5};
        REQUIRE(loki::utils::find_neighbouring_indices(single, 5, 3) == single);
    }
}

TEST_CASE("find_nearest_index", "[utils]") {
    const std::vector<float> values = {1.0F, 4.0F, 7.0F, 10.0F};

    REQUIRE(loki::utils::find_nearest_index(values, 4.1F) == 1);
    REQUIRE(loki::utils::find_nearest_index(values, 5.5F) == 1);
    REQUIRE(loki::utils::find_nearest_index(values, 1.0F) == 0);
}

TEST_CASE("find_lower_bin_index", "[utils]") {
    const std::vector<float> edges = {0.0F, 2.0F, 4.0F, 8.0F};

    REQUIRE(loki::utils::find_lower_bin_index(edges, -1.0F) == -1);
    REQUIRE(loki::utils::find_lower_bin_index(edges, 0.0F) == 0);
    REQUIRE(loki::utils::find_lower_bin_index(edges, 3.0F) == 1);
    REQUIRE(loki::utils::find_lower_bin_index(edges, 8.0F) ==
            static_cast<loki::IndexType>(edges.size() - 1));
}

TEST_CASE("linspace", "[utils]") {
    SECTION("Endpoint included") {
        const auto values = loki::utils::linspace(0.0, 1.0, 5, true);
        REQUIRE(values.size() == 5);
        REQUIRE_THAT(values.front(), WithinAbs(0.0, kNear));
        REQUIRE_THAT(values.back(), WithinAbs(1.0, kNear));
        REQUIRE_THAT(values[2], WithinAbs(0.5, kNear));
    }

    SECTION("Endpoint excluded") {
        const auto values = loki::utils::linspace(0.0, 1.0, 5, false);
        REQUIRE(values.size() == 5);
        REQUIRE_THAT(values.front(), WithinAbs(0.0, kNear));
        REQUIRE_THAT(values.back(), WithinAbs(0.8, kNear));
    }

    SECTION("Single sample returns start") {
        const auto values = loki::utils::linspace(3.0, 9.0, 1, true);
        REQUIRE(values == std::vector<double>{3.0});
    }

    SECTION("Zero samples returns empty vector") {
        REQUIRE(loki::utils::linspace(0.0, 1.0, 0, true).empty());
    }
}

TEST_CASE("determine_ref_segs", "[utils]") {
    SECTION("Evenly spaced anchors from n_runs") {
        const auto anchors =
            loki::utils::determine_ref_segs(10, 5, std::nullopt);
        REQUIRE(anchors == std::vector<loki::SizeType>{0, 2, 4, 6, 9});
    }

    SECTION("Single run anchors at zero") {
        REQUIRE(loki::utils::determine_ref_segs(10, 1, std::nullopt) ==
                std::vector<loki::SizeType>{0});
    }

    SECTION("Explicit ref_segs passthrough") {
        const std::vector<loki::SizeType> custom = {1, 3, 7};
        REQUIRE(loki::utils::determine_ref_segs(10, std::nullopt, custom) ==
                custom);
    }

    SECTION("Invalid n_runs throws") {
        REQUIRE_THROWS_AS(loki::utils::determine_ref_segs(5, 0, std::nullopt),
                          std::runtime_error);
        REQUIRE_THROWS_AS(loki::utils::determine_ref_segs(5, 6, std::nullopt),
                          std::runtime_error);
    }

    SECTION("Missing both arguments throws") {
        REQUIRE_THROWS_AS(
            loki::utils::determine_ref_segs(5, std::nullopt, std::nullopt),
            std::runtime_error);
    }
}

TEST_CASE("determine_ref_segs_pareto", "[utils]") {
    SECTION("Single run anchors at centre") {
        REQUIRE(loki::utils::determine_ref_segs_pareto(10, 1, std::nullopt) ==
                std::vector<loki::SizeType>{5});
    }

    SECTION("Multiple runs stay inside margins") {
        const auto anchors =
            loki::utils::determine_ref_segs_pareto(20, 4, std::nullopt);
        REQUIRE(anchors.size() == 4);
        REQUIRE(anchors.front() >= 2);
        REQUIRE(anchors.back() <= 17);
        for (loki::SizeType i = 1; i < anchors.size(); ++i) {
            REQUIRE(anchors[i] >= anchors[i - 1]);
        }
    }

    SECTION("Explicit ref_segs passthrough") {
        const std::vector<loki::SizeType> custom = {2, 8, 14};
        REQUIRE(loki::utils::determine_ref_segs_pareto(20, std::nullopt,
                                                       custom) == custom);
    }
}
