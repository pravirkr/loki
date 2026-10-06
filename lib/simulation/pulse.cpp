#include "loki/simulation/pulse.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "loki/common/types.hpp"
#include "loki/detection/score.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/simulation/modulate.hpp"

#include "lib/detail/error_check.hpp"
#include "lib/detail/math.hpp"
#include "lib/detail/psr_utils.hpp"
#include "lib/simulation/pulse_detail.hpp"

namespace loki::simulation {
namespace {

constexpr double kPi        = std::numbers::pi;
constexpr double kDamping   = 0.8;
constexpr SizeType kMinBins = 4;

[[nodiscard]] double positive_mod(double value, double period) {
    double wrapped = std::fmod(value, period);
    if (wrapped < 0.0) {
        wrapped += period;
    }
    return wrapped;
}

[[nodiscard]] double uniform_cdf(double x, double loc, double scale) {
    if (!(x > loc)) {
        return 0.0;
    }
    if (x >= loc + scale) {
        return 1.0;
    }
    return (x - loc) / scale;
}

[[nodiscard]] double normal_cdf(double x, double mean, double sigma) {
    const double z = (x - mean) / (sigma * std::numbers::sqrt2);
    return 0.5 * std::erfc(-z);
}

[[nodiscard]] std::vector<float> finish_lut(std::vector<double> lut) {
    double previous = lut.front();
    for (double& value : lut) {
        value    = std::max(value, previous);
        previous = value;
    }
    error_check::check(lut.back() > 0.0, "pulse CDF has no mass");
    const double norm = lut.back();
    std::vector<float> out(lut.size());
    for (SizeType i = 0; i < lut.size(); ++i) {
        out[i] = static_cast<float>(lut[i] / norm);
    }
    out.front() = 0.0F;
    for (SizeType i = 1; i < out.size(); ++i) {
        out[i] = std::min(std::max(out[i], out[i - 1]), 1.0F);
    }
    out.back() = 1.0F;
    return out;
}

struct Folded {
    std::vector<double> vals;
    std::vector<double> vars;
};

[[nodiscard]] Folded fold_proper_time(std::span<const float> ts_e,
                                      std::span<const float> ts_v,
                                      std::span<const double> proper_time,
                                      double freq,
                                      SizeType nbins) {
    Folded folded;
    folded.vals.assign(nbins, 0.0);
    folded.vars.assign(nbins, 0.0);
    for (SizeType i = 0; i < ts_e.size(); ++i) {
        const auto bin =
            psr_utils::get_phase_idx_uint(proper_time[i], freq, nbins, 0.0);
        folded.vals[bin] += static_cast<double>(ts_e[i]);
        folded.vars[bin] += static_cast<double>(ts_v[i]);
    }
    return folded;
}

[[nodiscard]] double population_std(std::span<const double> values,
                                    std::span<const std::uint8_t> keep) {
    double sum     = 0.0;
    SizeType count = 0;
    for (SizeType i = 0; i < values.size(); ++i) {
        if (keep[i] != 0) {
            sum += values[i];
            ++count;
        }
    }
    error_check::check(count > 1, "off-pulse window is empty");
    const double mean = sum / static_cast<double>(count);
    double accum      = 0.0;
    for (SizeType i = 0; i < values.size(); ++i) {
        if (keep[i] != 0) {
            const double delta = values[i] - mean;
            accum += delta * delta;
        }
    }
    return std::sqrt(accum / static_cast<double>(count));
}

[[nodiscard]] float max_boxcar_snr(std::span<const float> profile) {
    error_check::check(profile.size() >= kMinBins,
                       "folded profile needs at least 4 phase bins");
    std::vector<SizeType> widths;
    widths.reserve(profile.size() / 2);
    for (SizeType width = 1; width < profile.size() / 2; ++width) {
        widths.push_back(width);
    }
    std::vector<float> snr(widths.size());
    detection::snr_boxcar_1d(profile, widths, snr, 1.0F);
    return std::ranges::max(snr);
}

[[nodiscard]] double calibrate_scale(const Folded& noise,
                                     const Folded& tmpl,
                                     double snr_target,
                                     double noise_scale,
                                     int max_iter,
                                     double tol) {
    const SizeType nbins = noise.vals.size();
    std::vector<float> tmpl_norm(nbins);
    std::vector<float> noise_norm(nbins);
    for (SizeType i = 0; i < nbins; ++i) {
        const double variance = noise.vars[i] * noise_scale * noise_scale;
        error_check::check(variance > 0.0,
                           "folded profile has an empty phase bin");
        const double denom = std::sqrt(variance);
        tmpl_norm[i]       = static_cast<float>(tmpl.vals[i] / denom);
        noise_norm[i] =
            static_cast<float>((noise.vals[i] * noise_scale) / denom);
    }
    const auto snr_template = static_cast<double>(max_boxcar_snr(tmpl_norm));
    error_check::check(std::abs(snr_template) > 0.0,
                       "pulse template has zero folded SNR");
    double current = (snr_target / snr_template) * noise_scale;
    std::vector<float> combined(nbins);
    for (int iter = 0; iter < max_iter; ++iter) {
        for (SizeType i = 0; i < nbins; ++i) {
            combined[i] =
                noise_norm[i] + (static_cast<float>(current) * tmpl_norm[i]);
        }
        const auto measured = static_cast<double>(max_boxcar_snr(combined));
        const double diff   = measured - snr_target;
        if ((diff * diff) < (tol * tol)) {
            break;
        }
        current -= kDamping * diff / snr_template;
    }
    return current;
}

} // namespace

namespace detail {

PulseShapeKind parse_pulse_shape(std::string_view shape) {
    if (shape == "boxcar") {
        return PulseShapeKind::kBoxcar;
    }
    if (shape == "gaussian") {
        return PulseShapeKind::kGaussian;
    }
    if (shape == "von_mises") {
        return PulseShapeKind::kVonMises;
    }
    throw error_check::DetailedException("unknown pulse shape: " +
                                         std::string(shape));
}

std::vector<float>
build_cdf_lut(PulseShapeKind shape, double width, double pos, SizeType ngrid) {
    error_check::check(width > 0.0 && width < 1.0,
                       "pulse width must be in (0, 1)");
    error_check::check(ngrid >= 2, "CDF grid must contain at least 2 cells");
    const double phase0 = positive_mod(pos, 1.0);
    std::vector<double> lut(ngrid + 1, 0.0);
    if (shape == PulseShapeKind::kVonMises) {
        const double kappa = std::numbers::ln10 /
                             (2.0 * std::pow(std::sin(kPi * width / 2.0), 2.0));
        std::vector<double> pdf(ngrid + 1);
        for (SizeType j = 0; j <= ngrid; ++j) {
            const double phase =
                static_cast<double>(j) / static_cast<double>(ngrid);
            const double theta = (2.0 * kPi) * (phase - phase0);
            pdf[j]             = std::exp(kappa * (std::cos(theta) - 1.0));
        }
        for (SizeType j = 0; j < ngrid; ++j) {
            lut[j + 1] = lut[j] + (0.5 * (pdf[j] + pdf[j + 1]) /
                                   static_cast<double>(ngrid));
        }
        return finish_lut(std::move(lut));
    }

    const int nalias     = shape == PulseShapeKind::kBoxcar ? 1 : 4;
    const double sigma   = width / (2.0 * std::sqrt(2.0 * std::numbers::ln10));
    const double box_loc = phase0 - (width / 2.0);
    const auto cdf_at    = [&](double phase) {
        double sum = 0.0;
        for (int alias = -nalias; alias <= nalias; ++alias) {
            const double x = phase + static_cast<double>(alias);
            if (shape == PulseShapeKind::kBoxcar) {
                sum += uniform_cdf(x, box_loc, width);
            } else {
                sum += normal_cdf(x, phase0, sigma);
            }
        }
        return sum;
    };
    const double origin = cdf_at(0.0);
    for (SizeType j = 0; j <= ngrid; ++j) {
        const double phase =
            static_cast<double>(j) / static_cast<double>(ngrid);
        lut[j] = cdf_at(phase) - origin;
    }
    return finish_lut(std::move(lut));
}

std::vector<float> generate_pulse_template(std::span<const double> proper_time,
                                           double dt,
                                           double period,
                                           std::span<const float> cdf_lut) {
    error_check::check(period > 0.0, "period must be positive");
    error_check::check(dt > 0.0, "dt must be positive");
    error_check::check(cdf_lut.size() >= 2, "CDF lookup table is empty");
    const auto ngrid        = static_cast<SizeType>(cdf_lut.size() - 1);
    const double inv_period = 1.0 / period;
    const auto n            = static_cast<IndexType>(proper_time.size());
    std::vector<float> out(proper_time.size());
#pragma omp parallel for schedule(static) default(none)                        \
    shared(proper_time, cdf_lut, out)                                          \
    firstprivate(n, ngrid, dt, period, inv_period)
    for (IndexType i = 0; i < n; ++i) {
        const double t0 = proper_time[static_cast<SizeType>(i)];
        const double t1 = t0 + dt;
        const double p0 = positive_mod(t0, period) * inv_period;
        const double p1 = positive_mod(t1, period) * inv_period;
        const double x0 = p0 * static_cast<double>(ngrid);
        const double x1 = p1 * static_cast<double>(ngrid);
        const auto i0   = std::min(static_cast<SizeType>(x0), ngrid - 1);
        const auto i1   = std::min(static_cast<SizeType>(x1), ngrid - 1);
        const double f0 = x0 - static_cast<double>(i0);
        const double f1 = x1 - static_cast<double>(i1);
        const double c0 = ((1.0 - f0) * cdf_lut[i0]) + (f0 * cdf_lut[i0 + 1]);
        const double c1 = ((1.0 - f1) * cdf_lut[i1]) + (f1 * cdf_lut[i1 + 1]);
        double mass     = 0.0;
        if (p1 > p0) {
            mass = c1 - c0;
        } else {
            mass = (static_cast<double>(cdf_lut[ngrid]) - c0) +
                   (c1 - static_cast<double>(cdf_lut[0]));
        }
        if (mass < 0.0 && mass > -1e-6) {
            mass = 0.0;
        }
        out[static_cast<SizeType>(i)] = static_cast<float>(mass);
    }
    return out;
}

} // namespace detail

class PulseSignalConfig::Impl {
public:
    Impl(double period_in,
         double dt_in,
         SizeType nsamps_in,
         double snr_in,
         double ducy_in,
         const std::string& mod_type_in,
         const ModulatorParams& mod_in,
         std::optional<double> mod_tref_in,
         std::optional<std::uint64_t> seed_in)
        : period(period_in),
          dt(dt_in),
          nsamps(nsamps_in),
          snr(snr_in),
          ducy(ducy_in),
          modulator(make_modulator(mod_type_in, mod_in)),
          rng{math::PCG32(seed_in.value_or(std::random_device{}()))} {
        error_check::check(period > 0.0, "period must be positive");
        error_check::check(dt > 0.0, "dt must be positive");
        error_check::check(nsamps > 0, "nsamps must be positive");
        error_check::check(snr > 0.0, "snr must be positive");
        error_check::check(ducy > 0.0 && ducy < 1.0, "ducy must be in (0, 1)");
        const double span = static_cast<double>(nsamps) * dt;
        mod_tref          = mod_tref_in.value_or(span / 2.0);
    }

    Impl(const Impl& other)
        : period(other.period),
          dt(other.dt),
          nsamps(other.nsamps),
          snr(other.snr),
          ducy(other.ducy),
          mod_tref(other.mod_tref),
          modulator(other.modulator->clone()),
          rng(other.rng),
          normal(other.normal) {}

    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;
    ~Impl()                      = default;

    [[nodiscard]] std::vector<double> proper_time() const {
        std::vector<double> time(nsamps);
        std::vector<double> proper(nsamps);
        for (SizeType i = 0; i < nsamps; ++i) {
            time[i] = static_cast<double>(i) * dt;
        }
        modulator->generate(time, mod_tref, proper);
        return proper;
    }

    [[nodiscard]] float unit_normal() { return normal(rng); }

    double period;
    double dt;
    SizeType nsamps;
    double snr;
    double ducy;
    double mod_tref{0.0};
    struct UnitNormalEngine {
        // NOLINTNEXTLINE(readability-identifier-naming) -- URBG requires
        // result_type.
        using result_type = std::uint32_t;
        math::PCG32 rng;

        static constexpr result_type min() noexcept {
            return std::numeric_limits<result_type>::min();
        }
        static constexpr result_type max() noexcept {
            return std::numeric_limits<result_type>::max();
        }
        result_type operator()() { return rng(); }
    };

    std::unique_ptr<Modulator> modulator;
    UnitNormalEngine rng;
    std::normal_distribution<float> normal{0.0F, 1.0F};
};

PulseSignalConfig::PulseSignalConfig(double period,
                                     double dt,
                                     SizeType nsamps,
                                     double snr,
                                     double ducy,
                                     const std::string& mod_type,
                                     const ModulatorParams& mod,
                                     std::optional<double> mod_tref,
                                     std::optional<std::uint64_t> seed)
    : m_impl(std::make_unique<Impl>(
          period, dt, nsamps, snr, ducy, mod_type, mod, mod_tref, seed)) {}

PulseSignalConfig::~PulseSignalConfig() = default;

PulseSignalConfig::PulseSignalConfig(const PulseSignalConfig& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

PulseSignalConfig&
PulseSignalConfig::operator=(const PulseSignalConfig& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}

PulseSignalConfig::PulseSignalConfig(PulseSignalConfig&&) noexcept = default;

PulseSignalConfig&
PulseSignalConfig::operator=(PulseSignalConfig&&) noexcept = default;

io::TimeSeries PulseSignalConfig::generate(std::string_view shape,
                                           double phi0,
                                           int max_iter,
                                           double tol) {
    error_check::check(max_iter >= 0, "max_iter must be non-negative");
    error_check::check(tol >= 0.0, "tol must be non-negative");
    const double phase = positive_mod(phi0, 1.0);
    const auto nbins   = static_cast<SizeType>(m_impl->period / m_impl->dt);
    error_check::check(nbins >= kMinBins,
                       "period/dt must be at least 4 phase bins");
    const SizeType ngrid =
        std::max<SizeType>(4096, static_cast<SizeType>(100.0 / m_impl->ducy));
    const auto kind = detail::parse_pulse_shape(shape);
    const std::vector<float> lut =
        detail::build_cdf_lut(kind, m_impl->ducy, phase, ngrid);
    const std::vector<double> proper = m_impl->proper_time();
    const std::vector<float> tmpl    = detail::generate_pulse_template(
        proper, m_impl->dt, m_impl->period, lut);

    std::vector<float> noise(m_impl->nsamps);
    std::vector<float> unit_var(m_impl->nsamps, 1.0F);
    for (float& sample : noise) {
        sample = m_impl->unit_normal();
    }
    const double freq = 1.0 / m_impl->period;
    const Folded folded_noise =
        fold_proper_time(noise, unit_var, proper, freq, nbins);
    const Folded folded_tmpl =
        fold_proper_time(tmpl, unit_var, proper, freq, nbins);

    std::vector<std::uint8_t> off_pulse(nbins, 1);
    const auto center = static_cast<IndexType>(
                            std::llround(phase * static_cast<double>(nbins))) %
                        static_cast<IndexType>(nbins);
    const auto width_bins =
        std::max<SizeType>(1, static_cast<SizeType>(std::llround(
                                  m_impl->ducy * static_cast<double>(nbins))));
    const auto half = static_cast<IndexType>(width_bins / 2);
    for (SizeType k = 0; k < width_bins; ++k) {
        IndexType idx = center - half + static_cast<IndexType>(k);
        idx %= static_cast<IndexType>(nbins);
        if (idx < 0) {
            idx += static_cast<IndexType>(nbins);
        }
        off_pulse[static_cast<SizeType>(idx)] = 0;
    }
    const double noise_scale =
        1.0 / population_std(folded_noise.vals, off_pulse);
    const double signal_scale = calibrate_scale(
        folded_noise, folded_tmpl, m_impl->snr, noise_scale, max_iter, tol);

    std::vector<float> ts_e(m_impl->nsamps);
    std::vector<float> ts_v(m_impl->nsamps);
    const auto variance = static_cast<float>(noise_scale * noise_scale);
    for (SizeType i = 0; i < m_impl->nsamps; ++i) {
        ts_e[i] =
            static_cast<float>((noise_scale * static_cast<double>(noise[i])) +
                               (signal_scale * static_cast<double>(tmpl[i])));
        ts_v[i] = variance;
    }
    return {std::move(ts_e), std::move(ts_v), m_impl->dt};
}

io::TimeSeries PulseSignalConfig::generate_noise() {
    const double tol_bins = m_impl->ducy * m_impl->period / m_impl->dt;
    const double stdnoise =
        std::sqrt(static_cast<double>(m_impl->nsamps) * m_impl->ducy) /
        m_impl->snr / tol_bins;
    std::vector<float> ts_e(m_impl->nsamps);
    std::vector<float> ts_v(m_impl->nsamps,
                            static_cast<float>(stdnoise * stdnoise));
    for (float& sample : ts_e) {
        sample = m_impl->unit_normal() * static_cast<float>(stdnoise);
    }
    return {std::move(ts_e), std::move(ts_v), m_impl->dt};
}

double PulseSignalConfig::period() const noexcept { return m_impl->period; }
double PulseSignalConfig::dt() const noexcept { return m_impl->dt; }
SizeType PulseSignalConfig::nsamps() const noexcept { return m_impl->nsamps; }
double PulseSignalConfig::snr() const noexcept { return m_impl->snr; }
double PulseSignalConfig::ducy() const noexcept { return m_impl->ducy; }
double PulseSignalConfig::tobs() const noexcept {
    return static_cast<double>(m_impl->nsamps) * m_impl->dt;
}
double PulseSignalConfig::freq() const noexcept { return 1.0 / m_impl->period; }

} // namespace loki::simulation
