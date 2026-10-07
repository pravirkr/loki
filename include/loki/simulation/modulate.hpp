#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace loki::simulation {

/// Displacement derivatives at the reference epoch, in meters.
struct DerivativeTerms {
    double shift{0.0};
    double vel{0.0};
    double acc{0.0};
    double jerk{0.0};
    double snap{0.0};
};

/// Displacement derivatives through crackle, in meters.
struct DerivativeSeries {
    double shift{0.0};
    double vel{0.0};
    double acc{0.0};
    double jerk{0.0};
    double snap{0.0};
    double crackle{0.0};
};

/// Frequency-gauge derivatives with the line-of-sight velocity removed.
struct GaugeDerivatives {
    double freq{0.0};
    double vel{0.0};
    double acc{0.0};
    double jerk{0.0};
    double snap{0.0};
    double crackle{0.0};
};

/// Non-relativistic circular orbit.
struct CircularOrbit {
    double p_orb{0.0};
    double psi{0.0};
    double x_orb{0.0};
};

/**
 * @brief Parameters for make_modulator.
 *
 * Unused fields are ignored. `circular` needs `p_orb`, `psi`, and either
 * `x_orb` (light-seconds) or `m_c`. `circular_t0` needs `a`, `p_orb`, and
 * `t0`. `derivative_series` needs `coeffs`.
 */
struct ModulatorParams {
    double shift{0.0};
    double vel{0.0};
    double acc{0.0};
    double jerk{0.0};
    double snap{0.0};
    std::vector<double> coeffs;
    double p_orb{0.0};
    double psi{0.0};
    std::optional<double> x_orb;
    std::optional<double> m_c;
    double m_p{1.4};
    double sin_i{1.0};
    double a{0.0};
    double t0{0.0};
};

/**
 * @brief Maps barycentric time to proper time.
 *
 * `proper_time` has the same length as `time`. Both are in seconds.
 */
class Modulator {
public:
    virtual ~Modulator()                   = default;
    Modulator(const Modulator&)            = delete;
    Modulator& operator=(const Modulator&) = delete;
    Modulator(Modulator&&)                 = delete;
    Modulator& operator=(Modulator&&)      = delete;

    virtual void generate(std::span<const double> time,
                          double t_ref,
                          std::span<double> proper_time) const = 0;

    [[nodiscard]] virtual std::unique_ptr<Modulator> clone() const = 0;

protected:
    Modulator() = default;
};

/// Taylor delay up to snap. Coefficients are meters; the delay is divided by c.
class DerivativeModulator final : public Modulator {
public:
    explicit DerivativeModulator(DerivativeTerms terms = {});

    void generate(std::span<const double> time,
                  double t_ref,
                  std::span<double> proper_time) const override;

    [[nodiscard]] std::unique_ptr<Modulator> clone() const override;
    [[nodiscard]] CircularOrbit to_circular() const;
    [[nodiscard]] const DerivativeTerms& terms() const noexcept {
        return m_terms;
    }

private:
    DerivativeTerms m_terms;
};

/// Taylor delay of arbitrary order. Coefficients are meters.
class DerivativeSeriesModulator final : public Modulator {
public:
    explicit DerivativeSeriesModulator(std::vector<double> coeffs);

    void generate(std::span<const double> time,
                  double t_ref,
                  std::span<double> proper_time) const override;

    [[nodiscard]] std::unique_ptr<Modulator> clone() const override;
    [[nodiscard]] CircularOrbit to_circular() const;
    [[nodiscard]] const std::vector<double>& coeffs() const noexcept {
        return m_coeffs;
    }

private:
    std::vector<double> m_coeffs;
};

/**
 * @brief Non-relativistic circular orbit.
 *
 * `x_orb` is the projected semi-major axis in light-seconds. Unlike the
 * derivative models, generate() does not divide the delay by c. If `x_orb`
 * is omitted, it is computed from `m_c` with the pyloki mass law
 * `a = 0.005 * ((m_p + m_c) * p_orb^2)^(1/3)`.
 */
class CircularModulator final : public Modulator {
public:
    CircularModulator(double p_orb,
                      double psi,
                      std::optional<double> x_orb,
                      std::optional<double> m_c,
                      double m_p   = 1.4,
                      double sin_i = 1.0);

    void generate(std::span<const double> time,
                  double t_ref,
                  std::span<double> proper_time) const override;

    [[nodiscard]] std::unique_ptr<Modulator> clone() const override;
    [[nodiscard]] DerivativeSeries to_derivatives() const;
    [[nodiscard]] GaugeDerivatives to_derivatives_gauge(double f_ref) const;
    [[nodiscard]] std::vector<double> to_derivatives_series(int n) const;

    [[nodiscard]] double p_orb() const noexcept { return m_p_orb; }
    [[nodiscard]] double psi() const noexcept { return m_psi; }
    [[nodiscard]] double x_orb() const noexcept { return m_x_orb; }

private:
    double m_p_orb;
    double m_psi;
    double m_x_orb{0.0};
};

/**
 * @brief Circular delay referenced to an epoch `t0`.
 *
 * `a` is in seconds. generate() divides the geometric delay by c.
 */
class CircularT0Modulator final : public Modulator {
public:
    CircularT0Modulator(double a, double p_orb, double t0);

    void generate(std::span<const double> time,
                  double t_ref,
                  std::span<double> proper_time) const override;

    [[nodiscard]] std::unique_ptr<Modulator> clone() const override;

    [[nodiscard]] double a() const noexcept { return m_a; }
    [[nodiscard]] double p_orb() const noexcept { return m_p_orb; }
    [[nodiscard]] double t0() const noexcept { return m_t0; }

private:
    double m_a;
    double m_p_orb;
    double m_t0;
};

[[nodiscard]] std::unique_ptr<Modulator>
make_modulator(std::string_view type, const ModulatorParams& params);

} // namespace loki::simulation
