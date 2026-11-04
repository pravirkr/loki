#include "loki/simulation/configs.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <toml++/toml.hpp> // NOLINT(misc-include-cleaner) -- toml::table

#include "lib/detail/toml_schema.hpp"
#include "lib/detail/toml_utils.hpp"
#include "lib/simulation/config_keys.hpp"

namespace loki::simulation {

namespace {

using Sim = SimulateTomlConfig;

const toml_utils::Schema<Sim>& sim_schema() {
    using toml_utils::field;
    static const toml_utils::Schema<Sim> schema{
        // [signal]
        field<Sim>("signal", "period",
                   [](auto& c) -> auto& { return c.period; }),
        field<Sim>("signal", "dt", [](auto& c) -> auto& { return c.dt; }),
        field<Sim>("signal", "nsamps",
                   [](auto& c) -> auto& { return c.nsamps; }),
        field<Sim>("signal", "snr", [](auto& c) -> auto& { return c.snr; }),
        field<Sim>("signal", "ducy", [](auto& c) -> auto& { return c.ducy; }),
        field<Sim>("signal", "shape", [](auto& c) -> auto& { return c.shape; }),
        field<Sim>("signal", "phi0", [](auto& c) -> auto& { return c.phi0; }),
        field<Sim>("signal", "seed", [](auto& c) -> auto& { return c.seed; }),
        // [modulation]
        field<Sim>("modulation", "type",
                   [](auto& c) -> auto& { return c.mod_type; }),
        field<Sim>("modulation", "mod_tref",
                   [](auto& c) -> auto& { return c.mod_tref; }),
        field<Sim>("modulation", "shift",
                   [](auto& c) -> auto& { return c.mod.shift; }),
        field<Sim>("modulation", "vel",
                   [](auto& c) -> auto& { return c.mod.vel; }),
        field<Sim>("modulation", "acc",
                   [](auto& c) -> auto& { return c.mod.acc; }),
        field<Sim>("modulation", "jerk",
                   [](auto& c) -> auto& { return c.mod.jerk; }),
        field<Sim>("modulation", "snap",
                   [](auto& c) -> auto& { return c.mod.snap; }),
        field<Sim>("modulation", "coeffs",
                   [](auto& c) -> auto& { return c.mod.coeffs; }),
        field<Sim>("modulation", "p_orb",
                   [](auto& c) -> auto& { return c.mod.p_orb; }),
        field<Sim>("modulation", "psi",
                   [](auto& c) -> auto& { return c.mod.psi; }),
        field<Sim>("modulation", "x_orb",
                   [](auto& c) -> auto& { return c.mod.x_orb; }),
        field<Sim>("modulation", "m_c",
                   [](auto& c) -> auto& { return c.mod.m_c; }),
        field<Sim>("modulation", "m_p",
                   [](auto& c) -> auto& { return c.mod.m_p; }),
        field<Sim>("modulation", "sin_i",
                   [](auto& c) -> auto& { return c.mod.sin_i; }),
        field<Sim>("modulation", "a", [](auto& c) -> auto& { return c.mod.a; }),
        field<Sim>("modulation", "t0",
                   [](auto& c) -> auto& { return c.mod.t0; }),
        // [output]
        field<Sim>("output", "path", [](auto& c) -> auto& { return c.output; }),
        field<Sim>("output", "format",
                   [](auto& c) -> auto& { return c.format; }),
    };
    return schema;
}

constexpr std::array kTableOrder{
    std::string_view{"signal"},
    std::string_view{"modulation"},
    std::string_view{"output"},
};

} // namespace

std::string_view SimulateTomlConfig::default_toml_string() {
    return R"(# Loki Pulse Simulation Configuration

[signal]
period = 0.5          # Pulse period in seconds
dt = 0.0001           # Sample interval in seconds (tsamp)
nsamps = 2097152      # Number of samples (e.g. 2^21)
snr = 100.0           # Target folded boxcar SNR
ducy = 0.1            # Pulse duty cycle (FWTM in phase)
shape = "gaussian"    # Pulse shape: "gaussian", "boxcar", or "von_mises"
phi0 = 0.5            # Pulse centre in phase [0, 1)
# seed = 42           # Optional RNG seed

[modulation]
type = "derivative"   # "derivative", "derivative_series", "circular", or "circular_t0"
# mod_tref = 0.0      # Reference time for modulation in seconds
shift = 0.0           # Line-of-sight shift in meters
vel = 0.0             # Line-of-sight velocity in m/s
acc = 0.0             # Line-of-sight acceleration in m/s^2
jerk = 0.0            # Line-of-sight jerk in m/s^3
snap = 0.0            # Line-of-sight snap in m/s^4
# coeffs = [0.0, 0.0] # Derivative series coefficients in meters
# Orbital parameters for circular / circular_t0:
# p_orb = 7200.0      # Orbital period in seconds
# psi = 0.0           # Orbital phase at reference epoch
# x_orb = 1.0         # Projected semi-major axis in light-seconds
# m_c = 0.2           # Companion mass in solar masses
# m_p = 1.4           # Pulsar mass in solar masses
# sin_i = 1.0         # Sine of inclination
# a = 1.0             # circular_t0 delay amplitude in seconds
# t0 = 0.0            # circular_t0 reference epoch in seconds

[output]
path = "simulated_pulse.tim"  # Output .tim or .dat path
format = "tim"                 # "tim" or "dat"
)";
}

void SimulateTomlConfig::write_default(const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(
            fmt::format("Could not open file for writing default config: '{}'",
                        path.string()));
    }
    file << default_toml_string();
}

SimulateTomlConfig
SimulateTomlConfig::from_string(std::string_view toml_content) {
    const toml::table tbl = toml_utils::parse(toml_content);
    toml_utils::reject_nonfinite_floats(tbl, toml_content, "");
    toml_utils::check_document(sim_schema(), tbl,
                               "Invalid simulation TOML config");
    SimulateTomlConfig cfg;
    toml_utils::read_fields(sim_schema(), tbl, cfg);
    return cfg;
}

std::vector<std::string> simulate_config_keys() {
    return toml_utils::schema_keys(sim_schema());
}

std::string SimulateTomlConfig::to_toml_string() const {
    return toml_utils::to_toml_text(
        toml_utils::write_fields(sim_schema(), *this), kTableOrder);
}

void SimulateTomlConfig::validate() const {
    std::vector<std::string> errors;
    const auto check = [&](bool ok, std::string_view what) {
        if (!ok) {
            errors.emplace_back(what);
        }
    };
    const auto one_of = [&](std::string_view value,
                            std::span<const std::string_view> allowed,
                            std::string_view what) {
        if (!toml_utils::is_allowed_key(value, allowed)) {
            errors.push_back(
                fmt::format("{} is '{}'{}", what, value,
                            toml_utils::did_you_mean(value, allowed)));
        }
    };
    static constexpr std::array kShapes{
        std::string_view{"gaussian"},
        std::string_view{"boxcar"},
        std::string_view{"von_mises"},
    };
    static constexpr std::array kModTypes{
        std::string_view{"derivative"},
        std::string_view{"derivative_series"},
        std::string_view{"circular"},
        std::string_view{"circular_t0"},
    };
    static constexpr std::array kFormats{
        std::string_view{"tim"},
        std::string_view{"dat"},
    };
    check(period > 0.0, "signal.period (--period) must be positive");
    check(dt > 0.0, "signal.dt (--dt) must be positive");
    check(nsamps > 0, "signal.nsamps (--nsamps) must be positive");
    check(snr > 0.0, "signal.snr (--snr) must be positive");
    check(ducy > 0.0 && ducy < 1.0, "signal.ducy (--ducy) must be in (0, 1)");
    one_of(shape, kShapes,
           "signal.shape (--shape) must be gaussian, boxcar or von_mises; it");
    one_of(mod_type, kModTypes,
           "modulation.type (--mod) must be derivative, derivative_series, "
           "circular or circular_t0; it");
    one_of(format, kFormats, "output.format (--format) must be tim or dat; it");
    check(!output.empty(), "output.path (-o/--output) must be set");
    toml_utils::throw_if_errors(errors, "Invalid simulation settings");
}

SimulateTomlConfig SimulateTomlConfig::load(const std::filesystem::path& path) {
    return toml_utils::load_file(path, &SimulateTomlConfig::from_string);
}

} // namespace loki::simulation
