#include <catch2/catch_test_macros.hpp>

#include "loki/thresholds.hpp"

TEST_CASE("DynamicThresholdScheme", "[thresholds]") {
    std::vector<float> branching_pattern = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F};

    float ref_ducy       = 0.5F;
    SizeType nbins       = 64;
    SizeType ntrials     = 1024;
    SizeType nprobs      = 10;
    float prob_min       = 0.05F;
    float snr_final      = 8.0F;
    SizeType nthresholds = 100;
    float ducy_max       = 0.3F;
    float wtsp           = 1.0F;
    float beam_width     = 0.7F;
    int nthreads         = 1;
    loki::thresholds::DynamicThresholdScheme dyn_scheme(
        branching_pattern, ref_ducy, nbins, ntrials, nprobs, prob_min,
        snr_final, nthresholds, ducy_max, wtsp, beam_width, nthreads);
    SECTION("get_branching_pattern") {
        std::vector<float> bp = dyn_scheme.get_branching_pattern();
        REQUIRE(bp == branching_pattern);
    }
    SECTION("get_profile") {
        std::vector<float> profile = dyn_scheme.get_profile();
        REQUIRE(profile.size() == nbins);
    }
    SECTION("get_thresholds") {
        std::vector<float> thresholds = dyn_scheme.get_thresholds();
        REQUIRE(thresholds.size() == nthresholds);
    }
}
