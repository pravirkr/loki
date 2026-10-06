#include "loki/simulation/modulate.hpp"

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lib/detail/utils.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("derivative delay matches the Taylor polynomial", "[simulation]") {
    loki::simulation::DerivativeTerms terms;
    terms.shift = 3.0;
    terms.vel   = -1.5;
    terms.acc   = 0.25;
    terms.jerk  = -0.05;
    terms.snap  = 0.01;
    loki::simulation::DerivativeModulator modulator(terms);
    const std::vector<double> time{0.0, 1.25, 4.0, 10.0};
    const double t_ref = 2.0;
    std::vector<double> proper(time.size());
    modulator.generate(time, t_ref, proper);
    for (std::size_t i = 0; i < time.size(); ++i) {
        const double dt    = time[i] - t_ref;
        const double delay = terms.shift + (terms.vel * dt) +
                             (terms.acc * dt * dt / 2.0) +
                             (terms.jerk * dt * dt * dt / 6.0) +
                             (terms.snap * dt * dt * dt * dt / 24.0);
        REQUIRE_THAT(proper[i],
                     WithinAbs(time[i] - (delay / loki::utils::kCval), 1e-9));
    }
}

TEST_CASE("circular mass law and derivative round-trip", "[simulation]") {
    constexpr double kPorb = 86400.0;
    constexpr double kMc   = 0.2;
    constexpr double kMp   = 1.4;
    constexpr double kPsi  = 0.4;
    const double semi =
        0.005 * std::pow((kMp + kMc) * kPorb * kPorb, 1.0 / 3.0);
    const double expected = semi * (kMc / (kMp + kMc));
    loki::simulation::CircularModulator from_mass(kPorb, kPsi, std::nullopt,
                                                  kMc, kMp, 1.0);
    REQUIRE_THAT(from_mass.x_orb(), WithinRel(expected, 1e-12));

    loki::simulation::CircularModulator orbit(5000.0, kPsi, 1.5, std::nullopt);
    const auto deriv = orbit.to_derivatives();
    loki::simulation::DerivativeModulator taylor(
        {deriv.shift, deriv.vel, deriv.acc, deriv.jerk, deriv.snap});
    const auto recovered = taylor.to_circular();
    REQUIRE_THAT(recovered.p_orb, WithinRel(orbit.p_orb(), 1e-8));
    REQUIRE_THAT(recovered.x_orb, WithinRel(orbit.x_orb(), 1e-8));
    REQUIRE_THAT(recovered.psi, WithinAbs(kPsi, 1e-8));
}

TEST_CASE("circular_t0 divides by c and kepler is rejected", "[simulation]") {
    constexpr double kAmplitude = loki::utils::kCval;
    loki::simulation::CircularT0Modulator modulator(kAmplitude, 100.0, 0.0);
    const std::vector<double> time{25.0};
    std::vector<double> proper(1);
    modulator.generate(time, 0.0, proper);
    const double omega = 2.0 * std::acos(-1.0) / 100.0;
    const double delay = std::sin(omega * 25.0);
    REQUIRE_THAT(proper[0], WithinAbs(25.0 - delay, 1e-8));

    loki::simulation::ModulatorParams params;
    REQUIRE_THROWS(loki::simulation::make_modulator("keplerian", params));
    REQUIRE_THROWS(loki::simulation::make_modulator("kepler", params));
}
