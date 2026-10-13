#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"

using loki::SizeType;
using loki::detection::MatchedFilter;

TEST_CASE("MatchedFilter is the maximum of each profile-template circular "
          "convolution",
          "[score][matched_filter]") {
    // More profiles than templates, and an nbins that pads to a power of two.
    const SizeType nbins = GENERATE(SizeType{50}, SizeType{64});
    const auto shape     = GENERATE("boxcar", "gaussian");
    const std::vector<SizeType> widths{1, 3, 7};
    constexpr SizeType kNprofiles = 5;
    MatchedFilter filter(widths, kNprofiles, nbins, shape);
    const SizeType npad       = filter.get_nbins();
    const SizeType ntemplates = filter.get_ntemplates();
    REQUIRE(npad == 64);
    REQUIRE(ntemplates == widths.size());
    const auto templates = filter.get_templates();

    std::mt19937 rng(23);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> profiles(kNprofiles * nbins);
    for (auto& x : profiles) {
        x = dist(rng);
    }
    std::vector<float> out(kNprofiles * ntemplates);
    filter.compute(profiles, out);

    INFO("nbins=" << nbins << " shape=" << shape);
    for (SizeType p = 0; p < kNprofiles; ++p) {
        for (SizeType t = 0; t < ntemplates; ++t) {
            // Profile zero-padded to npad bins, convolved with the template.
            double best = std::numeric_limits<double>::lowest();
            for (SizeType k = 0; k < npad; ++k) {
                double acc = 0.0;
                for (SizeType m = 0; m < nbins; ++m) {
                    acc += static_cast<double>(profiles[(p * nbins) + m]) *
                           templates[(t * npad) + ((k + npad - m) % npad)];
                }
                best = std::max(best, acc);
            }
            CHECK(std::abs(out[(p * ntemplates) + t] - best) <= 1e-4);
        }
    }
}
