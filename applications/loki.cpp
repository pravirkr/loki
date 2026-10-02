#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "loki/io/timeseries.hpp"
#include "loki/simulation/pulse.hpp"

namespace {

std::string lower_copy(std::string text) {
    for (char& ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return text;
}

std::filesystem::path resolve_output(std::filesystem::path path,
                                     const std::string& format) {
    const std::string requested = lower_copy(format);
    if (!requested.empty() && requested != "tim" && requested != "dat") {
        throw std::invalid_argument("format must be tim or dat");
    }
    if (!requested.empty()) {
        path.replace_extension("." + requested);
        return path;
    }
    const std::string ext = lower_copy(path.extension().string());
    if (ext != ".tim" && ext != ".dat") {
        throw std::invalid_argument(
            "output must end in .tim or .dat, or pass --format");
    }
    return path;
}

int run_simulate(double period,
                 double dt,
                 loki::SizeType nsamps,
                 double snr,
                 double ducy,
                 const std::string& shape,
                 double phi0,
                 const std::optional<std::uint64_t>& seed,
                 const std::filesystem::path& output,
                 const std::string& mod_type,
                 const loki::simulation::ModulatorParams& mod,
                 const std::optional<double>& mod_tref) {
    loki::simulation::PulseSignalConfig config(period, dt, nsamps, snr, ducy,
                                               mod_type, mod, mod_tref, seed);
    auto series = config.generate(shape, phi0);
    series.write(output);
    SPDLOG_INFO("Wrote {} samples to {}", series.get_nsamps(), output.string());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    CLI::App app{"Loki"};
    app.require_subcommand(1);

    auto* simulate =
        app.add_subcommand("simulate", "Write a simulated pulse timeseries");
    auto* search =
        app.add_subcommand("search", "Search a timeseries for pulsars");

    double period = 0.0;
    double dt     = 0.0;
    auto nsamps   = loki::SizeType{1} << 21U;
    double snr    = 100.0;
    double ducy   = 0.1;
    std::string shape{"gaussian"};
    double phi0 = 0.5;
    std::optional<std::uint64_t> seed;
    std::string output;
    std::string format;
    std::string mod_type{"derivative"};
    loki::simulation::ModulatorParams mod;
    std::optional<double> mod_tref;

    simulate->add_option("--period", period, "Pulse period in seconds")
        ->required();
    simulate->add_option("--dt", dt, "Sample interval in seconds")->required();
    simulate->add_option("--nsamps", nsamps, "Number of samples");
    simulate->add_option("--snr", snr, "Target folded boxcar SNR");
    simulate->add_option("--ducy", ducy, "Pulse FWTM in phase");
    simulate->add_option("--shape", shape, "boxcar, gaussian, or von_mises");
    simulate->add_option("--phi0", phi0, "Pulse centre in phase");
    simulate->add_option("--seed", seed, "RNG seed");
    simulate->add_option("-o,--output", output, "Output .tim or .dat path")
        ->required();
    simulate->add_option("--format", format, "tim or dat");
    simulate->add_option(
        "--mod", mod_type,
        "derivative, derivative_series, circular, or circular_t0");
    simulate->add_option("--shift", mod.shift, "Line-of-sight shift in meters");
    simulate->add_option("--vel", mod.vel, "Line-of-sight velocity in m/s");
    simulate->add_option("--acc", mod.acc,
                         "Line-of-sight acceleration in m/s^2");
    simulate->add_option("--jerk", mod.jerk, "Line-of-sight jerk in m/s^3");
    simulate->add_option("--snap", mod.snap, "Line-of-sight snap in m/s^4");
    simulate
        ->add_option("--coeffs", mod.coeffs,
                     "Comma-separated derivative-series coefficients in meters")
        ->delimiter(',');
    simulate->add_option("--p-orb", mod.p_orb, "Orbital period in seconds");
    simulate->add_option("--psi", mod.psi,
                         "Orbital phase at the reference epoch");
    simulate->add_option("--x-orb", mod.x_orb,
                         "Projected semi-major axis in light-seconds");
    simulate->add_option("--m-c", mod.m_c, "Companion mass in solar masses");
    simulate->add_option("--m-p", mod.m_p, "Pulsar mass in solar masses");
    simulate->add_option("--sin-i", mod.sin_i, "Sine of the inclination");
    simulate->add_option("--a", mod.a,
                         "circular_t0 delay amplitude in seconds");
    simulate->add_option("--t0", mod.t0,
                         "circular_t0 reference epoch in seconds");
    simulate->add_option("--mod-tref", mod_tref,
                         "Modulation reference time in seconds");

    std::string search_input;
    search->add_option("--input", search_input, "Input timeseries");

    CLI11_PARSE(app, argc, argv);

    try {
        if (search->parsed()) {
            SPDLOG_ERROR("search mode is not implemented");
            return 1;
        }
        if (simulate->parsed()) {
            const auto path = resolve_output(output, format);
            return run_simulate(period, dt, nsamps, snr, ducy, shape, phi0,
                                seed, path, mod_type, mod, mod_tref);
        }
    } catch (const std::exception& ex) {
        SPDLOG_ERROR("{}", ex.what());
        return 1;
    }
    return 0;
}
