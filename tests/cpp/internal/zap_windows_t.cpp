#include "lib/search/zap_windows.hpp"

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/search/candidates.hpp"

using loki::search::FrequencyWindow;
using loki::search::detail::ZapWindowIndex;

TEST_CASE("Zap windows merge and test closed intervals", "[candidates]") {
    const std::vector<FrequencyWindow> windows{{.lo = 50.0, .hi = 50.5},
                                               {.lo = 10.0, .hi = 11.0},
                                               {.lo = 10.5, .hi = 12.0},
                                               {.lo = 30.0, .hi = 30.0}};
    const ZapWindowIndex index(windows);
    REQUIRE(index.windows().size() == 3); // [10, 12], [30, 30], [50, 50.5]
    CHECK(index.contains(10.0));
    CHECK(index.contains(11.5));
    CHECK(index.contains(12.0));
    CHECK_FALSE(index.contains(12.0001));
    CHECK(index.contains(30.0));
    CHECK_FALSE(index.contains(29.999));
    CHECK_FALSE(index.contains(9.0));
    CHECK_FALSE(index.contains(51.0));
    CHECK(ZapWindowIndex({}).empty());
    CHECK_FALSE(ZapWindowIndex({}).flags(10.0));
}

TEST_CASE("Zap window flags cover every ratio p/q with p, q <= 4",
          "[candidates]") {
    const ZapWindowIndex index(
        std::vector<FrequencyWindow>{{.lo = 419.8, .hi = 420.2}});
    // f0 * p / q = 420 for p / q in 1, 2, 3, 4, 1/2, 1/3, 1/4, 3/2, 2/3,
    // 4/3, 3/4.
    for (const double f0 : {420.0, 210.0, 140.0, 105.0, 840.0, 1260.0, 1680.0,
                            280.0, 630.0, 315.0, 560.0}) {
        INFO("f0 = " << f0);
        CHECK(index.flags(f0));
    }
    // Ratios 5, 1/5 and 5/4 are outside the set.
    for (const double f0 : {84.0, 2100.0, 336.0, 421.0}) {
        INFO("f0 = " << f0);
        CHECK_FALSE(index.flags(f0));
    }
}
