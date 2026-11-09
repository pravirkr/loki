#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"
#include "loki/utils/psr_utils.hpp"

using loki::SizeType;
using loki::psr_utils::leaf_phase;
using loki::psr_utils::LeafModel;

namespace {

constexpr double kC = 299792458.0;

// A leaf as an EP result stores it: derivatives highest first down to accel,
// then (freq, err), (delay, err), (f0, basis). Errors are relative to each
// value, so they set its significance.
std::vector<double> make_leaf(const std::vector<double>& derivs_high_first,
                              double freq,
                              double rel_err) {
    std::vector<double> leaf;
    for (const double d : derivs_high_first) {
        leaf.push_back(d);
        leaf.push_back(std::abs(d) * rel_err);
    }
    leaf.insert(leaf.end(), {freq, 1e-6, 0.0, 0.0, freq, 0.0});
    return leaf;
}

// Line-of-sight distance of a circular orbit, R sin(w t + psi), and its
// derivatives at t_ref; the delay relative to t_ref with the velocity gauged
// away is d(t) - d(t_ref) - d'(t_ref) (t - t_ref).
struct Orbit {
    double radius;
    double omega;
    double psi;
    [[nodiscard]] double deriv(int k, double t) const {
        // d^k/dt^k of R sin(w t + psi) = R w^k sin(w t + psi + k pi / 2)
        return radius * std::pow(omega, k) *
               std::sin((omega * t) + psi + (k * std::numbers::pi / 2.0));
    }
    [[nodiscard]] double delay(double t, double t_ref) const {
        return deriv(0, t) - deriv(0, t_ref) - (deriv(1, t_ref) * (t - t_ref));
    }
};

} // namespace

TEST_CASE("leaf_phase follows a Taylor leaf's polynomial", "[psr_utils]") {
    const double freq  = 142.3;
    const double accel = 50.0;
    const double jerk  = -0.02;
    const auto leaf    = make_leaf({jerk, accel}, freq, 1.0);
    const std::vector<double> times{0.0, 100.0, 512.0, 1024.0};
    const double t_ref = 512.0;
    std::vector<double> phase(times.size());
    REQUIRE(leaf_phase(leaf, 3, times, t_ref, phase) == LeafModel::kTaylor);
    for (SizeType i = 0; i < times.size(); ++i) {
        const double dt = times[i] - t_ref;
        const double delay =
            (accel * dt * dt / 2.0) + (jerk * dt * dt * dt / 6.0);
        REQUIRE(phase[i] ==
                Catch::Approx(freq * (dt - (delay / kC))).epsilon(1e-12));
    }
}

TEST_CASE("leaf_phase follows a circular orbit in a five-parameter search",
          "[psr_utils]") {
    // A 2-hour orbit of 0.5 light-seconds, observed for an hour.
    const double freq  = 300.0;
    const double t_ref = 1800.0;
    const std::vector<double> times{0.0, 600.0, 1800.0, 3000.0, 3600.0};

    SECTION("snap and accel significant: omega from them") {
        const Orbit orbit{0.5 * kC, 2.0 * std::numbers::pi / 7200.0, 0.3};
        const auto leaf =
            make_leaf({orbit.deriv(5, t_ref), orbit.deriv(4, t_ref),
                       orbit.deriv(3, t_ref), orbit.deriv(2, t_ref)},
                      freq, 1e-3);
        std::vector<double> phase(times.size());
        REQUIRE(leaf_phase(leaf, 5, times, t_ref, phase) ==
                LeafModel::kCircularSnap);
        for (SizeType i = 0; i < times.size(); ++i) {
            const double dt = times[i] - t_ref;
            const double expected =
                freq * (dt - (orbit.delay(times[i], t_ref) / kC));
            REQUIRE(phase[i] == Catch::Approx(expected).margin(1e-6));
        }
    }
    SECTION("accel and snap through zero: omega from crackle and jerk") {
        // sin(w t_ref + psi) = 0: the even derivatives vanish.
        const double omega = 2.0 * std::numbers::pi / 7200.0;
        const Orbit orbit{0.5 * kC, omega, -omega * t_ref};
        auto leaf = make_leaf({orbit.deriv(5, t_ref), orbit.deriv(4, t_ref),
                               orbit.deriv(3, t_ref), orbit.deriv(2, t_ref)},
                              freq, 1e-3);
        // Snap and accel measured as zero within an error: not significant.
        // (Rows: crackle 0-1, snap 2-3, jerk 4-5, accel 6-7.)
        leaf[2] = 0.0;
        leaf[3] = 1e-6;
        leaf[6] = 0.0;
        leaf[7] = 1e-6;
        std::vector<double> phase(times.size());
        REQUIRE(leaf_phase(leaf, 5, times, t_ref, phase) ==
                LeafModel::kCircularCrackle);
        for (SizeType i = 0; i < times.size(); ++i) {
            const double dt = times[i] - t_ref;
            const double expected =
                freq * (dt - (orbit.delay(times[i], t_ref) / kC));
            REQUIRE(phase[i] == Catch::Approx(expected).margin(1e-6));
        }
    }
    SECTION("nothing significant: the Taylor polynomial") {
        const auto leaf = make_leaf({1e-12, -2e-9, 3e-6, 0.4}, freq, 10.0);
        std::vector<double> phase(times.size());
        REQUIRE(leaf_phase(leaf, 5, times, t_ref, phase) == LeafModel::kTaylor);
        for (SizeType i = 0; i < times.size(); ++i) {
            const double dt    = times[i] - t_ref;
            const double delay = (0.4 * std::pow(dt, 2) / 2.0) +
                                 (3e-6 * std::pow(dt, 3) / 6.0) +
                                 (-2e-9 * std::pow(dt, 4) / 24.0) +
                                 (1e-12 * std::pow(dt, 5) / 120.0);
            REQUIRE(phase[i] ==
                    Catch::Approx(freq * (dt - (delay / kC))).epsilon(1e-12));
        }
    }
}

TEST_CASE("leaf_phase rejects a malformed leaf", "[psr_utils]") {
    const std::vector<double> times{0.0, 1.0};
    std::vector<double> phase(times.size());
    const std::vector<double> short_leaf(5, 0.0);
    REQUIRE_THROWS_AS(leaf_phase(short_leaf, 1, times, 0.0, phase),
                      std::runtime_error);
    const auto leaf = make_leaf({1.0}, 10.0, 1.0);
    REQUIRE_THROWS_AS(leaf_phase(leaf, 6, times, 0.0, phase),
                      std::runtime_error);
    std::vector<double> too_few(1);
    REQUIRE_THROWS_AS(leaf_phase(leaf, 2, times, 0.0, too_few),
                      std::runtime_error);
}
