#include "loki/search/orbit.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using Catch::Matchers::WithinRel;
using loki::ParamLimit;
using loki::SizeType;
using loki::search::circular_param_limits;
using loki::search::CircularOrbitLimits;
using loki::search::FFATomlConfig;
using loki::search::implied_circular_orbit;

namespace {

constexpr double kTobs = 33554432.0 * 6.4e-5;

/// A frequency-only FFA document with [search.orbit] @p orbit and the
/// [search] keys @p search_extra.
std::string orbit_doc(const std::string& orbit,
                      const std::string& search_extra = "") {
    return "[input]\n"
           "nsamps = 65536\n"
           "tsamp = 0.000064\n"
           "[search]\n"
           "f_min = 140.0\n"
           "f_max = 142.0\n"
           "nbins = 32\n" +
           search_extra + "[search.orbit]\n" + orbit;
}

} // namespace

TEST_CASE("circular_param_limits matches pyloki's from_circular",
          "[config][orbit]") {
    // pyloki.config.ParamLimits.from_circular((25, 29), 5 T, 10, 1.2,
    // poly_order=4), T = 2^25 * 64 us
    const auto limits = circular_param_limits(
        25.0, 29.0, 4,
        {.p_orb_min = 5.0 * kTobs, .m_c_max = 10.0, .m_p_min = 1.2});
    REQUIRE(limits.size() == 4);
    CHECK_THAT(limits[0].max, WithinRel(1.707753527281e-04, 1e-6));
    CHECK_THAT(limits[1].max, WithinRel(2.918402526231e-01, 1e-6));
    CHECK_THAT(limits[2].max, WithinRel(4.987296567715e+02, 1e-6));
    CHECK_THAT(limits[3].min, WithinRel(2.492892702389e+01, 1e-9));
    CHECK_THAT(limits[3].max, WithinRel(2.908244465228e+01, 1e-9));
    for (SizeType i = 0; i < 3; ++i) {
        CHECK(limits[i].min == -limits[i].max);
    }
}

TEST_CASE("circular_param_limits has one row per searched derivative",
          "[config][orbit]") {
    const CircularOrbitLimits orbit{.p_orb_min = 5.0 * kTobs};
    const auto accel = circular_param_limits(25.0, 29.0, 2, orbit);
    const auto jerk  = circular_param_limits(25.0, 29.0, 3, orbit);
    REQUIRE(accel.size() == 2);
    REQUIRE(jerk.size() == 3);
    // The rows are the same derivatives whatever the order.
    CHECK(accel[0].max == jerk[1].max);
    CHECK(accel[1].min == jerk[2].min);
    // jerk = accel * omega on a circular orbit
    const double omega = 2.0 * std::numbers::pi / orbit.p_orb_min;
    CHECK_THAT(jerk[0].max, WithinRel(jerk[1].max * omega, 1e-12));
}

TEST_CASE("circular_param_limits rejects unphysical inputs",
          "[config][orbit]") {
    const CircularOrbitLimits orbit{.p_orb_min = 1000.0};
    CHECK_THROWS_AS(circular_param_limits(25.0, 29.0, 1, orbit),
                    std::invalid_argument);
    CHECK_THROWS_AS(circular_param_limits(25.0, 29.0, 5, orbit),
                    std::invalid_argument);
    CHECK_THROWS_AS(circular_param_limits(29.0, 25.0, 2, orbit),
                    std::invalid_argument);
    CHECK_THROWS_AS(circular_param_limits(0.0, 25.0, 2, orbit),
                    std::invalid_argument);
    CHECK_THROWS_AS(circular_param_limits(25.0, 29.0, 2, {.p_orb_min = 0.0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(circular_param_limits(
                        25.0, 29.0, 2, {.p_orb_min = 1000.0, .m_c_max = -1.0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        circular_param_limits(25.0, 29.0, 2,
                              {.p_orb_min = std::nan(""), .m_c_max = 10.0}),
        std::invalid_argument);
    // A microsecond orbit needs a velocity above c.
    CHECK_THROWS_AS(circular_param_limits(25.0, 29.0, 2, {.p_orb_min = 1e-6}),
                    std::invalid_argument);
}

TEST_CASE("implied_circular_orbit inverts circular_param_limits",
          "[config][orbit]") {
    const CircularOrbitLimits orbit{
        .p_orb_min = 5.0 * kTobs, .m_c_max = 10.0, .m_p_min = 1.2};
    const auto limits  = circular_param_limits(25.0, 29.0, 3, orbit);
    const auto implied = implied_circular_orbit(limits, orbit.m_p_min);
    REQUIRE(implied.has_value());
    CHECK_THAT(implied->p_orb, WithinRel(orbit.p_orb_min, 1e-9));
    CHECK_THAT(implied->m_c_min, WithinRel(orbit.m_c_max, 1e-6));

    // The production box of the past: accel 1000, jerk 4 over 2147 s is an
    // orbit of 0.73 T_obs.
    const std::vector<ParamLimit> box{
        {-4.0, 4.0}, {-1000.0, 1000.0}, {25.0, 29.0}};
    const auto fast = implied_circular_orbit(box);
    REQUIRE(fast.has_value());
    CHECK_THAT(fast->p_orb, WithinRel(2.0 * std::numbers::pi * 250.0, 1e-12));
    CHECK(fast->p_orb < kTobs);

    // Without both an acceleration and a jerk limit there is no orbit.
    const std::vector<ParamLimit> accel{{-1000.0, 1000.0}, {25.0, 29.0}};
    CHECK_FALSE(implied_circular_orbit(accel).has_value());
    const std::vector<ParamLimit> no_jerk{
        {0.0, 0.0}, {-1000.0, 1000.0}, {25.0, 29.0}};
    CHECK_FALSE(implied_circular_orbit(no_jerk).has_value());
}

TEST_CASE("[search.orbit] sets the parameter limits", "[config][orbit]") {
    const auto cfg = FFATomlConfig::from_string(
        orbit_doc("order = \"jerk\"\np_orb_min_tobs = 5.0\nm_c_max = 8.0\n"));
    REQUIRE(cfg.orbit.has_value());
    const auto search   = cfg.to_search_config();
    const double tobs   = 65536.0 * 0.000064;
    const auto expected = circular_param_limits(
        140.0, 142.0, 3,
        {.p_orb_min = 5.0 * tobs, .m_c_max = 8.0, .m_p_min = 1.2});
    const auto limits = search.get_param_limits();
    REQUIRE(limits.size() == expected.size());
    for (SizeType i = 0; i < limits.size(); ++i) {
        CHECK(limits[i].min == expected[i].min);
        CHECK(limits[i].max == expected[i].max);
    }
    CHECK(search.get_nparams() == 3);

    // Seconds instead of units of T_obs; zero acc/jerk keys mean "not set".
    const auto seconds = FFATomlConfig::from_string(
                             orbit_doc("order = \"accel\"\np_orb_min = 100.0\n",
                                       "acc_min = 0.0\nacc_max = 0.0\n"))
                             .to_search_config();
    CHECK(seconds.get_nparams() == 2);
}

TEST_CASE("[search.orbit] rejects incomplete or conflicting input",
          "[config][orbit]") {
    const auto convert = [](const std::string& doc) {
        return FFATomlConfig::from_string(doc).to_search_config();
    };
    // Explicit limits next to the orbit
    CHECK_THROWS_AS(convert(orbit_doc("order = \"jerk\"\np_orb_min = 100.0\n",
                                      "acc_min = -1.0\nacc_max = 1.0\n")),
                    std::invalid_argument);
    // Neither or both periods
    CHECK_THROWS_AS(convert(orbit_doc("order = \"jerk\"\n")),
                    std::invalid_argument);
    CHECK_THROWS_AS(convert(orbit_doc("order = \"jerk\"\np_orb_min = 100.0\n"
                                      "p_orb_min_tobs = 5.0\n")),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        convert(orbit_doc("order = \"jerk\"\np_orb_min_tobs = 0\n")),
        std::invalid_argument);
    // Order missing or unknown, a non-numeric value, an unknown key
    CHECK_THROWS(convert(orbit_doc("p_orb_min = 100.0\n")));
    CHECK_THROWS_AS(
        convert(orbit_doc("order = \"crackle\"\np_orb_min = 1e3\n")),
        std::invalid_argument);
    CHECK_THROWS(
        convert(orbit_doc("order = \"jerk\"\np_orb_min = \"long\"\n")));
    CHECK_THROWS(convert(orbit_doc("order = \"jerk\"\np_orb_min = 100.0\n"
                                   "ecc_max = 0.1\n")));
    // An unphysical orbit
    CHECK_THROWS_AS(convert(orbit_doc("order = \"jerk\"\np_orb_min = 1e-6\n")),
                    std::invalid_argument);
}
