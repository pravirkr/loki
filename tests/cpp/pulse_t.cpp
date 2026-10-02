#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/detection/score.hpp"
#include "loki/psr_utils.hpp"
#include "loki/pulse_detail.hpp"
#include "loki/simulation/pulse.hpp"

using Catch::Matchers::WithinAbs;
namespace detail = loki::simulation::detail;

namespace {

double circular_distance(double lhs, double rhs) {
    double delta = std::fmod(lhs - rhs, 1.0);
    if (delta < -0.5) {
        delta += 1.0;
    }
    if (delta > 0.5) {
        delta -= 1.0;
    }
    return std::abs(delta);
}

void check_lut(detail::PulseShapeKind shape, double ducy, double phi0) {
    constexpr loki::SizeType kGrid = 4096;
    const auto lut = detail::build_cdf_lut(shape, ducy, phi0, kGrid);
    REQUIRE(lut.size() == kGrid + 1);
    REQUIRE(lut.front() == 0.0F);
    REQUIRE(lut.back() == 1.0F);
    for (loki::SizeType i = 1; i < lut.size(); ++i) {
        REQUIRE(lut[i] >= lut[i - 1]);
    }

    double mass   = 0.0;
    double moment = 0.0;
    float peak    = 0.0F;
    int above     = 0;
    for (loki::SizeType i = 0; i < kGrid; ++i) {
        const float density = lut[i + 1] - lut[i];
        REQUIRE(density >= 0.0F);
        peak = std::max(peak, density);
        mass += static_cast<double>(density);
        moment += static_cast<double>(density) *
                  ((static_cast<double>(i) + 0.5) / static_cast<double>(kGrid));
    }
    REQUIRE_THAT(mass, WithinAbs(1.0, 1e-5));
    const double centroid = moment / mass;
    REQUIRE(circular_distance(centroid, phi0) < 0.02);
    for (loki::SizeType i = 0; i < kGrid; ++i) {
        if ((lut[i + 1] - lut[i]) > (0.1F * peak)) {
            ++above;
        }
    }
    const double width =
        static_cast<double>(above) / static_cast<double>(kGrid);
    REQUIRE_THAT(width, WithinAbs(ducy, 0.02));
}

float folded_snr(const loki::io::TimeSeries& series, double period) {
    const auto nbins = static_cast<loki::SizeType>(period / series.get_dt());
    const double freq = 1.0 / period;
    std::vector<float> vals(nbins, 0.0F);
    std::vector<float> vars(nbins, 0.0F);
    const auto intensity = series.get_ts_e();
    const auto variance  = series.get_ts_v();
    for (loki::SizeType i = 0; i < intensity.size(); ++i) {
        const double time = static_cast<double>(i) * series.get_dt();
        const auto bin =
            loki::psr_utils::get_phase_idx_uint(time, freq, nbins, 0.0);
        vals[bin] += intensity[i];
        vars[bin] += variance[i];
    }
    std::vector<float> profile(nbins);
    for (loki::SizeType i = 0; i < nbins; ++i) {
        profile[i] = vals[i] / std::sqrt(vars[i]);
    }
    std::vector<loki::SizeType> widths;
    for (loki::SizeType width = 1; width < nbins / 2; ++width) {
        widths.push_back(width);
    }
    std::vector<float> snr(widths.size());
    loki::detection::snr_boxcar_1d(profile, widths, snr, 1.0F);
    return std::ranges::max(snr);
}

} // namespace

TEST_CASE("pulse lookup tables conserve mass and width", "[simulation]") {
    check_lut(detail::PulseShapeKind::kBoxcar, 0.1, 0.5);
    check_lut(detail::PulseShapeKind::kGaussian, 0.1, 0.35);
    check_lut(detail::PulseShapeKind::kVonMises, 0.2, 0.6);
    REQUIRE_THROWS(detail::parse_pulse_shape("lorentz"));
}

TEST_CASE("pulse template integrates to one and wraps", "[simulation]") {
    constexpr double kPeriod        = 1.0;
    constexpr double kDt            = 0.001;
    constexpr loki::SizeType kSamps = 1000;
    for (const char* shape : {"boxcar", "gaussian", "von_mises"}) {
        for (double phi0 : {0.5, 0.02}) {
            const auto kind = detail::parse_pulse_shape(shape);
            const auto lut  = detail::build_cdf_lut(kind, 0.1, phi0, 4096);
            std::vector<double> time(kSamps);
            for (loki::SizeType i = 0; i < kSamps; ++i) {
                time[i] = static_cast<double>(i) * kDt;
            }
            const auto signal =
                detail::generate_pulse_template(time, kDt, kPeriod, lut);
            double sum = 0.0;
            for (float sample : signal) {
                REQUIRE(sample >= 0.0F);
                sum += static_cast<double>(sample);
            }
            REQUIRE_THAT(sum, WithinAbs(1.0, 2e-3));
        }
    }
}

TEST_CASE("generated timeseries hits the target snr", "[simulation]") {
    constexpr double kPeriod        = 0.05;
    constexpr double kDt            = 1.0e-4;
    constexpr loki::SizeType kSamps = 20000;
    constexpr double kSnr           = 30.0;
    for (const char* shape : {"boxcar", "gaussian", "von_mises"}) {
        loki::simulation::PulseSignalConfig first(kPeriod, kDt, kSamps, kSnr,
                                                  0.1, "derivative", {},
                                                  std::nullopt, 42);
        const auto series = first.generate(shape, 0.5);
        REQUIRE(series.get_nsamps() == kSamps);
        REQUIRE_THAT(series.get_dt(), WithinAbs(kDt, 0.0));
        REQUIRE_THAT(static_cast<double>(folded_snr(series, kPeriod)),
                     WithinAbs(kSnr, 0.05));

        loki::simulation::PulseSignalConfig repeat(kPeriod, kDt, kSamps, kSnr,
                                                   0.1, "derivative", {},
                                                   std::nullopt, 42);
        const auto again = repeat.generate(shape, 0.5);
        for (loki::SizeType i = 0; i < kSamps; ++i) {
            REQUIRE(again.get_ts_e()[i] == series.get_ts_e()[i]);
        }
        const auto next = repeat.generate(shape, 0.5);
        bool differs    = false;
        for (loki::SizeType i = 0; i < kSamps; ++i) {
            if (next.get_ts_e()[i] != again.get_ts_e()[i]) {
                differs = true;
                break;
            }
        }
        REQUIRE(differs);
    }
}
