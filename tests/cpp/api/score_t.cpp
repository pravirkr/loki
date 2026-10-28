#include "loki/detection/score.hpp"

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/types.hpp"

using loki::SizeType;
using loki::detection::generate_box_width_trials;

TEST_CASE("generate_box_width_trials ends at ducy_max", "[score]") {
    // Defaults: the step after 9 overshoots floor(0.2 * 64) = 12
    REQUIRE(generate_box_width_trials(64) ==
            std::vector<SizeType>{1, 2, 3, 4, 6, 9, 12});
    // A step that lands on the widest box is not repeated
    REQUIRE(generate_box_width_trials(64, 0.5, 2.0) ==
            std::vector<SizeType>{1, 2, 4, 8, 16, 32});
    // A box as wide as the profile leaves no off-pulse bins
    REQUIRE(generate_box_width_trials(64, 1.0, 2.0) ==
            std::vector<SizeType>{1, 2, 4, 8, 16, 32, 63});
}

TEST_CASE("generate_box_width_trials steps by at most wtsp up to the widest "
          "box",
          "[score]") {
    const double ducy_max = GENERATE(0.1, 0.2, 0.3, 0.5, 1.0);
    const double wtsp     = GENERATE(1.2, 1.5, 2.0, 2.45);
    for (SizeType nbins = 2; nbins <= 1024; ++nbins) {
        const auto widths = generate_box_width_trials(nbins, ducy_max, wtsp);
        const auto wmax   = std::clamp(
            static_cast<SizeType>(ducy_max * static_cast<double>(nbins)),
            SizeType{1}, nbins - 1);
        INFO("nbins=" << nbins << " ducy_max=" << ducy_max << " wtsp=" << wtsp);
        REQUIRE(widths.front() == 1);
        REQUIRE(widths.back() == wmax);
        for (SizeType i = 1; i < widths.size(); ++i) {
            const auto step =
                std::max(widths[i - 1] + 1,
                         static_cast<SizeType>(
                             wtsp * static_cast<double>(widths[i - 1])));
            REQUIRE(widths[i] > widths[i - 1]);
            REQUIRE(widths[i] <= step);
        }
    }
}
