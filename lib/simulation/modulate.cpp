#include "loki/simulation/modulate.hpp"

#include <cmath>
#include <numbers>
#include <utility>

#include "detail/error_check.hpp"
#include "detail/utils.hpp"

namespace loki::simulation {
namespace {

constexpr double kPi = std::numbers::pi;

void check_time_span(std::span<const double> time,
                     std::span<double> proper_time) {
    error_check::check_equal(time.size(), proper_time.size(),
                             "proper_time length must match time");
}

[[nodiscard]] double positive_mod(double value, double period) {
    double wrapped = std::fmod(value, period);
    if (wrapped < 0.0) {
        wrapped += period;
    }
    return wrapped;
}

[[nodiscard]] double int_pow(double base, int exponent) {
    double result = 1.0;
    double factor = base;
    int remaining = exponent;
    while (remaining > 0) {
        if ((remaining & 1) != 0) {
            result *= factor;
        }
        factor *= factor;
        remaining >>= 1;
    }
    return result;
}

[[nodiscard]] CircularOrbit
circular_from_derivatives(double acc, double jerk, double snap) {
    if (acc == 0.0 && snap == 0.0) {
        throw error_check::DetailedException(
            "degenerate phase: cannot recover omega from acceleration and "
            "snap");
    }
    if (acc * snap >= 0.0) {
        throw error_check::DetailedException(
            "incompatible with a circular orbit: acceleration * snap must be "
            "negative");
    }
    const double omega = std::sqrt(-snap / acc);
    CircularOrbit orbit;
    orbit.p_orb         = (2.0 * kPi) / omega;
    const double omega2 = omega * omega;
    const double omega3 = omega2 * omega;
    const double x_sin  = -acc / (utils::kCval * omega2);
    const double x_cos  = -jerk / (utils::kCval * omega3);
    orbit.x_orb         = std::hypot(x_sin, x_cos);
    orbit.psi           = std::atan2(x_sin, x_cos);
    return orbit;
}

} // namespace

DerivativeModulator::DerivativeModulator(DerivativeTerms terms)
    : m_terms(terms) {}

void DerivativeModulator::generate(std::span<const double> time,
                                   double t_ref,
                                   std::span<double> proper_time) const {
    check_time_span(time, proper_time);
    const double shift = m_terms.shift;
    const double vel   = m_terms.vel;
    const double acc   = m_terms.acc;
    const double jerk  = m_terms.jerk;
    const double snap  = m_terms.snap;
    const double c     = utils::kCval;
    const auto n       = static_cast<IndexType>(time.size());
#pragma omp parallel for schedule(static) default(none)                        \
    shared(time, proper_time)                                                  \
    firstprivate(n, t_ref, shift, vel, acc, jerk, snap, c)
    for (IndexType i = 0; i < n; ++i) {
        const double dt = time[static_cast<SizeType>(i)] - t_ref;
        const double delay =
            shift +
            (dt * (vel + (dt * ((acc * 0.5) +
                                (dt * ((jerk / 6.0) + (dt * snap / 24.0)))))));
        proper_time[static_cast<SizeType>(i)] =
            time[static_cast<SizeType>(i)] - (delay / c);
    }
}

std::unique_ptr<Modulator> DerivativeModulator::clone() const {
    return std::make_unique<DerivativeModulator>(m_terms);
}

CircularOrbit DerivativeModulator::to_circular() const {
    return circular_from_derivatives(m_terms.acc, m_terms.jerk, m_terms.snap);
}

DerivativeSeriesModulator::DerivativeSeriesModulator(std::vector<double> coeffs)
    : m_coeffs(std::move(coeffs)) {
    error_check::check(!m_coeffs.empty(),
                       "derivative series needs at least one coefficient");
}

void DerivativeSeriesModulator::generate(std::span<const double> time,
                                         double t_ref,
                                         std::span<double> proper_time) const {
    check_time_span(time, proper_time);
    std::vector<double> scaled(m_coeffs.size());
    double factorial = 1.0;
    scaled[0]        = m_coeffs[0];
    for (SizeType k = 1; k < m_coeffs.size(); ++k) {
        factorial *= static_cast<double>(k);
        scaled[k] = m_coeffs[k] / factorial;
    }
    const double c = utils::kCval;
    const auto n   = static_cast<IndexType>(time.size());
#pragma omp parallel for schedule(static) default(none)                        \
    shared(time, proper_time, scaled) firstprivate(n, t_ref, c)
    for (IndexType i = 0; i < n; ++i) {
        const double dt = time[static_cast<SizeType>(i)] - t_ref;
        double delay    = 0.0;
        for (SizeType k = scaled.size(); k-- > 0;) {
            delay = (dt * delay) + scaled[k];
        }
        proper_time[static_cast<SizeType>(i)] =
            time[static_cast<SizeType>(i)] - (delay / c);
    }
}

std::unique_ptr<Modulator> DerivativeSeriesModulator::clone() const {
    return std::make_unique<DerivativeSeriesModulator>(m_coeffs);
}

CircularOrbit DerivativeSeriesModulator::to_circular() const {
    error_check::check(m_coeffs.size() >= 5,
                       "need at least snap to recover a circular orbit");
    return circular_from_derivatives(m_coeffs[2], m_coeffs[3], m_coeffs[4]);
}

CircularModulator::CircularModulator(double p_orb,
                                     double psi,
                                     std::optional<double> x_orb,
                                     std::optional<double> m_c,
                                     double m_p,
                                     double sin_i)
    : m_p_orb(p_orb),
      m_psi(psi),
      m_x_orb(0.0) {
    error_check::check(p_orb > 0.0, "orbital period must be positive");
    if (x_orb.has_value()) {
        m_x_orb = *x_orb;
        return;
    }
    error_check::check(m_c.has_value(),
                       "circular modulation needs x_orb or companion mass");
    const double total = m_p + *m_c;
    error_check::check(total != 0.0,
                       "pulsar plus companion mass must be non-zero");
    const double semi_major =
        0.005 * std::pow(total * p_orb * p_orb, 1.0 / 3.0);
    m_x_orb = semi_major * (*m_c / total) * sin_i;
}

void CircularModulator::generate(std::span<const double> time,
                                 double t_ref,
                                 std::span<double> proper_time) const {
    check_time_span(time, proper_time);
    const double omega = (2.0 * kPi) / m_p_orb;
    const double x_orb = m_x_orb;
    const double psi   = m_psi;
    const auto n       = static_cast<IndexType>(time.size());
#pragma omp parallel for schedule(static) default(none)                        \
    shared(time, proper_time) firstprivate(n, t_ref, omega, x_orb, psi)
    for (IndexType i = 0; i < n; ++i) {
        const double delay =
            x_orb *
            std::sin((omega * (time[static_cast<SizeType>(i)] - t_ref)) + psi);
        proper_time[static_cast<SizeType>(i)] =
            time[static_cast<SizeType>(i)] - delay;
    }
}

std::unique_ptr<Modulator> CircularModulator::clone() const {
    return std::make_unique<CircularModulator>(m_p_orb, m_psi, m_x_orb,
                                               std::nullopt);
}

DerivativeSeries CircularModulator::to_derivatives() const {
    const double omega  = (2.0 * kPi) / m_p_orb;
    const double omega2 = omega * omega;
    const double x_sin  = m_x_orb * std::sin(m_psi);
    const double x_cos  = m_x_orb * std::cos(m_psi);
    DerivativeSeries deriv;
    deriv.shift   = utils::kCval * x_sin;
    deriv.vel     = utils::kCval * x_cos * omega;
    deriv.acc     = -utils::kCval * x_sin * omega2;
    deriv.jerk    = -utils::kCval * x_cos * omega2 * omega;
    deriv.snap    = -deriv.acc * omega2;
    deriv.crackle = -deriv.jerk * omega2;
    return deriv;
}

GaugeDerivatives CircularModulator::to_derivatives_gauge(double f_ref) const {
    const DerivativeSeries deriv = to_derivatives();
    const double scale           = 1.0 - (deriv.vel / utils::kCval);
    error_check::check(scale != 0.0, "gauge scale vanished");
    GaugeDerivatives gauge;
    gauge.freq    = scale * f_ref;
    gauge.vel     = 0.0;
    gauge.acc     = deriv.acc / scale;
    gauge.jerk    = deriv.jerk / scale;
    gauge.snap    = deriv.snap / scale;
    gauge.crackle = deriv.crackle / scale;
    return gauge;
}

std::vector<double> CircularModulator::to_derivatives_series(int n) const {
    error_check::check(n >= 0, "derivative order must be non-negative");
    const double omega  = (2.0 * kPi) / m_p_orb;
    const double omega2 = omega * omega;
    const double x_sin  = m_x_orb * std::sin(m_psi);
    const double x_cos  = m_x_orb * std::cos(m_psi);
    std::vector<double> coeffs(static_cast<SizeType>(n) + 1, 0.0);
    coeffs[0] = utils::kCval * x_sin;
    if (n >= 1) {
        coeffs[1] = utils::kCval * x_cos * omega;
    }
    if (n >= 2) {
        coeffs[2] = -utils::kCval * x_sin * omega2;
    }
    if (n >= 3) {
        coeffs[3] = -utils::kCval * x_cos * omega2 * omega;
    }
    if (n >= 4) {
        coeffs[4]          = utils::kCval * x_sin * omega2 * omega2;
        const double d2    = coeffs[2];
        const double d3    = coeffs[3];
        const double ratio = d2 == 0.0 ? 0.0 : coeffs[4] / d2;
        for (int k = 5; k <= n; ++k) {
            if ((k % 2) == 0) {
                coeffs[static_cast<SizeType>(k)] =
                    int_pow(ratio, (k - 2) / 2) * d2;
            } else {
                coeffs[static_cast<SizeType>(k)] =
                    int_pow(ratio, (k - 3) / 2) * d3;
            }
        }
    }
    return coeffs;
}

CircularT0Modulator::CircularT0Modulator(double a, double p_orb, double t0)
    : m_a(a),
      m_p_orb(p_orb),
      m_t0(t0) {
    error_check::check(p_orb > 0.0, "orbital period must be positive");
}

void CircularT0Modulator::generate(std::span<const double> time,
                                   double t_ref,
                                   std::span<double> proper_time) const {
    check_time_span(time, proper_time);
    const double omega = (2.0 * kPi) / m_p_orb;
    const double phi =
        (2.0 * kPi) * positive_mod(t_ref - m_t0, m_p_orb) / m_p_orb;
    const double amplitude = m_a;
    const double c         = utils::kCval;
    const auto n           = static_cast<IndexType>(time.size());
#pragma omp parallel for schedule(static) default(none)                        \
    shared(time, proper_time) firstprivate(n, t_ref, omega, phi, amplitude, c)
    for (IndexType i = 0; i < n; ++i) {
        const double delay =
            amplitude *
            std::sin((omega * (time[static_cast<SizeType>(i)] - t_ref)) + phi);
        proper_time[static_cast<SizeType>(i)] =
            time[static_cast<SizeType>(i)] - (delay / c);
    }
}

std::unique_ptr<Modulator> CircularT0Modulator::clone() const {
    return std::make_unique<CircularT0Modulator>(m_a, m_p_orb, m_t0);
}

std::unique_ptr<Modulator> make_modulator(std::string_view type,
                                          const ModulatorParams& params) {
    if (type == "derivative") {
        DerivativeTerms terms;
        terms.shift = params.shift;
        terms.vel   = params.vel;
        terms.acc   = params.acc;
        terms.jerk  = params.jerk;
        terms.snap  = params.snap;
        return std::make_unique<DerivativeModulator>(terms);
    }
    if (type == "derivative_series") {
        return std::make_unique<DerivativeSeriesModulator>(params.coeffs);
    }
    if (type == "circular") {
        return std::make_unique<CircularModulator>(params.p_orb, params.psi,
                                                   params.x_orb, params.m_c,
                                                   params.m_p, params.sin_i);
    }
    if (type == "circular_t0") {
        return std::make_unique<CircularT0Modulator>(params.a, params.p_orb,
                                                     params.t0);
    }
    if (type == "keplerian" || type == "kepler") {
        throw error_check::DetailedException(
            "Keplerian modulation is not supported");
    }
    throw error_check::DetailedException("unknown modulator type: " +
                                         std::string(type));
}

} // namespace loki::simulation
