#include <algorithm>
#include <bit>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>
#include <toml++/toml.hpp>
#include <omp.h>

#include "loki/common/types.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"
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

std::optional<std::string> find_config_arg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            return std::string(argv[i + 1]);
        }
        if (arg.starts_with("--config=")) {
            return std::string(arg.substr(9));
        }
    }
    return std::nullopt;
}

bool has_subcommand(int argc, char** argv, std::string_view name) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == name) {
            return true;
        }
    }
    return false;
}

struct SimulateTomlConfig {
    double period{0.0};
    double dt{0.0};
    loki::SizeType nsamps{1U << 21U};
    double snr{100.0};
    double ducy{0.1};
    std::string shape{"gaussian"};
    double phi0{0.5};
    std::optional<std::uint64_t> seed;
    std::filesystem::path output{"simulated_pulse.tim"};
    std::string format{"tim"};
    std::string mod_type{"derivative"};
    loki::simulation::ModulatorParams mod;
    std::optional<double> mod_tref;

    static std::string default_toml_string() {
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

    static void write_default(const std::filesystem::path& path) {
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " +
                                     path.string());
        }
        file << default_toml_string();
    }

    static SimulateTomlConfig load(const std::filesystem::path& path) {
        auto tbl = toml::parse_file(path.string());
        SimulateTomlConfig cfg;

        if (auto* sig = tbl["signal"].as_table()) {
            cfg.period = sig->get("period")->value_or(cfg.period);
            cfg.dt     = sig->get("dt")->value_or(cfg.dt);
            if (auto* n = sig->get("nsamps")) {
                if (auto val = n->value<int64_t>()) {
                    cfg.nsamps = static_cast<loki::SizeType>(*val);
                }
            }
            cfg.snr   = sig->get("snr")->value_or(cfg.snr);
            cfg.ducy  = sig->get("ducy")->value_or(cfg.ducy);
            cfg.shape = sig->get("shape")->value_or(cfg.shape);
            cfg.phi0  = sig->get("phi0")->value_or(cfg.phi0);
            if (auto* s = sig->get("seed")) {
                if (auto val = s->value<int64_t>()) {
                    cfg.seed = static_cast<std::uint64_t>(*val);
                }
            }
        }

        if (auto* mod_tbl = tbl["modulation"].as_table()) {
            cfg.mod_type = mod_tbl->get("type")->value_or(cfg.mod_type);
            if (auto* tr = mod_tbl->get("mod_tref")) {
                cfg.mod_tref = tr->value<double>();
            }
            cfg.mod.shift = mod_tbl->get("shift")->value_or(cfg.mod.shift);
            cfg.mod.vel   = mod_tbl->get("vel")->value_or(cfg.mod.vel);
            cfg.mod.acc   = mod_tbl->get("acc")->value_or(cfg.mod.acc);
            cfg.mod.jerk  = mod_tbl->get("jerk")->value_or(cfg.mod.jerk);
            cfg.mod.snap  = mod_tbl->get("snap")->value_or(cfg.mod.snap);
            if (auto* coeffs_arr = mod_tbl->get_as<toml::array>("coeffs")) {
                cfg.mod.coeffs.clear();
                for (const auto& elem : *coeffs_arr) {
                    if (auto val = elem.value<double>()) {
                        cfg.mod.coeffs.push_back(*val);
                    }
                }
            }
            cfg.mod.p_orb = mod_tbl->get("p_orb")->value_or(cfg.mod.p_orb);
            cfg.mod.psi   = mod_tbl->get("psi")->value_or(cfg.mod.psi);
            if (auto* xo = mod_tbl->get("x_orb")) {
                cfg.mod.x_orb = xo->value<double>();
            }
            if (auto* mc = mod_tbl->get("m_c")) {
                cfg.mod.m_c = mc->value<double>();
            }
            cfg.mod.m_p   = mod_tbl->get("m_p")->value_or(cfg.mod.m_p);
            cfg.mod.sin_i = mod_tbl->get("sin_i")->value_or(cfg.mod.sin_i);
            cfg.mod.a     = mod_tbl->get("a")->value_or(cfg.mod.a);
            cfg.mod.t0    = mod_tbl->get("t0")->value_or(cfg.mod.t0);
        }

        if (auto* out_tbl = tbl["output"].as_table()) {
            cfg.output = out_tbl->get("path")->value_or(cfg.output.string());
            cfg.format = out_tbl->get("format")->value_or(cfg.format);
        }

        return cfg;
    }
};

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

enum class NsampsPolicy { kFail, kTruncate };

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open config file: " +
                                 path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

int run_search_ffa(const loki::search::FFATomlConfig& toml_cfg,
                   std::string_view config_toml,
                   bool dry_run,
                   NsampsPolicy nsamps_policy) {
    if (toml_cfg.f_min <= 0.0 || toml_cfg.f_max <= toml_cfg.f_min) {
        throw std::invalid_argument(
            std::format("Invalid frequency range: [{:.3f}, {:.3f}] Hz",
                        toml_cfg.f_min, toml_cfg.f_max));
    }

    if (toml_cfg.use_cuda && !loki::is_available(loki::Backend::kCUDA)) {
        throw std::runtime_error(
            "--cuda was requested but this build has LOKI_CUDA=OFF");
    }

    const auto preview_cfg = toml_cfg.to_search_config(
        toml_cfg.nsamps.value_or(1U << 21U), toml_cfg.tsamp.value_or(6.4e-5));
    if (dry_run) {
        const auto exec = toml_cfg.use_cuda
                              ? loki::Exec::cuda(toml_cfg.device_id)
                              : loki::Exec::cpu();
        loki::pipelines::FFAFreqSweep dry(preview_cfg, /*show_progress=*/false,
                                           exec);
        SPDLOG_INFO("Dry run complete: planner constructed successfully.");
        return 0;
    }

    if (toml_cfg.timeseries_path.empty()) {
        throw std::invalid_argument(
            "Input timeseries path must be specified via -i/--input or "
            "[input].timeseries in config");
    }
    const std::filesystem::path ts_path = toml_cfg.timeseries_path;
    if (!std::filesystem::exists(ts_path)) {
        throw std::runtime_error(std::format(
            "Input timeseries does not exist: {}", ts_path.string()));
    }

    loki::io::ReadOptions read_opts;
    read_opts.preprocess             = toml_cfg.preprocess;
    read_opts.filter_window          = toml_cfg.filter_window;
    read_opts.fast_median            = toml_cfg.fast_median;
    read_opts.fast_median_min_points = toml_cfg.fast_median_min_points;
    // Config value 0 means "use all hardware threads".
    read_opts.nthreads = toml_cfg.nthreads <= 0 ? omp_get_max_threads()
                                                : toml_cfg.nthreads;
    SPDLOG_INFO("Loading timeseries from: {}", ts_path.string());
    // FFA assumes finite ts_e and positive ts_v (enforced in TimeSeries).
    auto ts = loki::io::TimeSeries::read(ts_path, read_opts);
    SPDLOG_INFO(
        "Loaded timeseries: nsamps = {}, dt = {:.6e} s, tobs = {:.2f} s",
        ts.get_nsamps(), ts.get_dt(), ts.get_tobs());

    loki::SizeType actual_nsamps = ts.get_nsamps();
    if (!std::has_single_bit(actual_nsamps)) {
        const auto pow2_nsamps = std::bit_floor(actual_nsamps);
        if (nsamps_policy == NsampsPolicy::kFail) {
            throw std::invalid_argument(std::format(
                "Timeseries length {} is not a power of 2; use --nsamps-policy "
                "truncate",
                actual_nsamps));
        }
        if (nsamps_policy == NsampsPolicy::kTruncate) {
            SPDLOG_WARN(
                "Timeseries length {} is not a power of 2; truncating search "
                "to {}",
                actual_nsamps, pow2_nsamps);
            actual_nsamps = pow2_nsamps;
        }
    }

    auto ffa_cfg = toml_cfg.to_search_config(actual_nsamps, ts.get_dt());

    const std::filesystem::path outdir_path = toml_cfg.outdir;
    std::filesystem::create_directories(outdir_path);

    SPDLOG_INFO("Starting FFA search: f0=[{:.3f}, {:.3f}] Hz, nbins={}, "
                "eta={:.2f}, snr_min={:.2f}",
                toml_cfg.f_min, toml_cfg.f_max, ffa_cfg.get_nbins(),
                ffa_cfg.get_eta(), ffa_cfg.get_snr_min());

    // The CPU thread count travels in ffa_cfg; Exec only picks the backend.
    const auto exec = toml_cfg.use_cuda ? loki::Exec::cuda(toml_cfg.device_id)
                                        : loki::Exec::cpu();
    if (exec.backend == loki::Backend::kCUDA) {
        SPDLOG_INFO("Using CUDA backend on device {}", toml_cfg.device_id);
    }
    loki::pipelines::FFAFreqSweep sweep(ffa_cfg, /*show_progress=*/true, exec);
    sweep.execute(ts.get_ts_e().first(actual_nsamps),
                  ts.get_ts_v().first(actual_nsamps), outdir_path,
                  toml_cfg.prefix, config_toml);

    const auto result_file =
        outdir_path / std::format("{}_ffa_results.h5", toml_cfg.prefix);
    SPDLOG_INFO("FFA search complete. Inspecting results in {}",
                result_file.string());

    if (std::filesystem::exists(result_file)) {
        try {
            HighFive::File h5(result_file.string(), HighFive::File::ReadOnly);
            if (h5.exist("snr")) {
                auto snr_dset      = h5.getDataSet("snr");
                const auto dims    = snr_dset.getDimensions();
                const auto n_cands = dims.empty() ? 0UL : dims[0];
                if (n_cands > 0) {
                    float top_snr           = 0.0F;
                    size_t top_idx          = 0;
                    constexpr size_t kChunk = 65536;
                    for (size_t offset = 0; offset < n_cands;
                         offset += kChunk) {
                        const size_t count = std::min(kChunk, n_cands - offset);
                        std::vector<float> snrs(count);
                        snr_dset.select({offset}, {count}).read(snrs);
                        const auto max_it =
                            std::max_element(snrs.begin(), snrs.end());
                        if (max_it != snrs.end() && *max_it >= top_snr) {
                            top_snr = *max_it;
                            top_idx = offset +
                                      static_cast<size_t>(
                                          std::distance(snrs.begin(), max_it));
                        }
                    }

                    SPDLOG_INFO("=== Search Summary ===");
                    SPDLOG_INFO("Total candidates detected : {}", n_cands);
                    SPDLOG_INFO(
                        "Top candidate SNR          : {:.2f} (candidate "
                        "#{})",
                        top_snr, top_idx);

                    if (h5.exist("param_sets")) {
                        auto p_dset = h5.getDataSet("param_sets");
                        auto p_dims = p_dset.getDimensions();
                        if (p_dims.size() == 2 && p_dims[0] == n_cands &&
                            p_dims[1] > 0) {
                            std::vector<double> top_params(p_dims[1]);
                            p_dset.select({top_idx, 0}, {1, p_dims[1]})
                                .read_raw(top_params.data());
                            if (h5.hasAttribute("param_names")) {
                                std::vector<std::string> p_names;
                                h5.getAttribute("param_names").read(p_names);
                                std::string p_str;
                                for (size_t pi = 0;
                                     pi < std::min(top_params.size(),
                                                   p_names.size());
                                     ++pi) {
                                    if (pi > 0) {
                                        p_str += ", ";
                                    }
                                    p_str +=
                                        std::format("{}={:.6f}", p_names[pi],
                                                    top_params[pi]);
                                }
                                SPDLOG_INFO("Top candidate parameters   : [{}]",
                                            p_str);
                            }
                        }
                    }
                } else {
                    SPDLOG_INFO("=== Search Summary ===");
                    SPDLOG_INFO("Total candidates detected : 0 above S/N "
                                "threshold {:.2f}",
                                toml_cfg.snr_min);
                }
            } else {
                SPDLOG_INFO("=== Search Summary ===");
                SPDLOG_INFO(
                    "Total candidates detected : 0 above S/N threshold {:.2f}",
                    toml_cfg.snr_min);
            }
        } catch (const std::exception& e) {
            SPDLOG_WARN("HDF5 result summary inspection failed: {}", e.what());
        }
    }

    SPDLOG_INFO("Saved HDF5 results to: {}", result_file.string());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    CLI::App app{"Loki - Fast Pulsar Search Pipeline"};
    app.require_subcommand(1);

    // Pre-inspect configuration argument if passed to populate base config
    // before CLI option registration
    const auto cfg_arg = find_config_arg(argc, argv);

    // =========================================================================
    // Subcommand: simulate
    // =========================================================================
    auto* simulate =
        app.add_subcommand("simulate", "Write a simulated pulse timeseries");

    SimulateTomlConfig sim_cfg;
    if (cfg_arg.has_value() && has_subcommand(argc, argv, "simulate")) {
        sim_cfg = SimulateTomlConfig::load(*cfg_arg);
    }

    std::string sim_config_path;
    std::string gen_sim_config_path;
    simulate->add_option("-c,--config", sim_config_path,
                         "Load simulation configuration from TOML file");
    auto* opt_gen_sim =
        simulate
            ->add_option("-g,--generate-config", gen_sim_config_path,
                         "Generate default TOML simulation configuration file "
                         "[optional output path]")
            ->expected(0, 1);

    auto* grp_signal = simulate->add_option_group("Signal Parameters");
    grp_signal->add_option("--period", sim_cfg.period,
                           "Pulse period in seconds");
    grp_signal->add_option("--dt", sim_cfg.dt, "Sample interval in seconds");
    grp_signal->add_option("--nsamps", sim_cfg.nsamps, "Number of samples");
    grp_signal->add_option("--snr", sim_cfg.snr, "Target folded boxcar SNR");
    grp_signal->add_option("--ducy", sim_cfg.ducy, "Pulse FWTM in phase");
    grp_signal->add_option("--shape", sim_cfg.shape,
                           "Pulse shape: boxcar, gaussian, or von_mises");
    grp_signal->add_option("--phi0", sim_cfg.phi0, "Pulse centre in phase");
    grp_signal->add_option("--seed", sim_cfg.seed, "RNG seed");

    auto* grp_output = simulate->add_option_group("Output Options");
    grp_output->add_option("-o,--output", sim_cfg.output,
                           "Output .tim or .dat path");
    grp_output->add_option("--format", sim_cfg.format, "Format: tim or dat");

    auto* grp_mod = simulate->add_option_group("Modulation Options");
    grp_mod->add_option("--mod", sim_cfg.mod_type,
                        "Modulation type: derivative, derivative_series, "
                        "circular, circular_t0");
    grp_mod->add_option("--shift", sim_cfg.mod.shift,
                        "Line-of-sight shift in meters");
    grp_mod->add_option("--vel", sim_cfg.mod.vel,
                        "Line-of-sight velocity in m/s");
    grp_mod->add_option("--acc", sim_cfg.mod.acc,
                        "Line-of-sight acceleration in m/s^2");
    grp_mod->add_option("--jerk", sim_cfg.mod.jerk,
                        "Line-of-sight jerk in m/s^3");
    grp_mod->add_option("--snap", sim_cfg.mod.snap,
                        "Line-of-sight snap in m/s^4");
    grp_mod
        ->add_option("--coeffs", sim_cfg.mod.coeffs,
                     "Comma-separated derivative-series coefficients in meters")
        ->delimiter(',');
    grp_mod->add_option("--p-orb", sim_cfg.mod.p_orb,
                        "Orbital period in seconds");
    grp_mod->add_option("--psi", sim_cfg.mod.psi,
                        "Orbital phase at the reference epoch");
    grp_mod->add_option("--x-orb", sim_cfg.mod.x_orb,
                        "Projected semi-major axis in light-seconds");
    grp_mod->add_option("--m-c", sim_cfg.mod.m_c,
                        "Companion mass in solar masses");
    grp_mod->add_option("--m-p", sim_cfg.mod.m_p,
                        "Pulsar mass in solar masses");
    grp_mod->add_option("--sin-i", sim_cfg.mod.sin_i,
                        "Sine of the inclination");
    grp_mod->add_option("--a", sim_cfg.mod.a,
                        "circular_t0 delay amplitude in seconds");
    grp_mod->add_option("--t0", sim_cfg.mod.t0,
                        "circular_t0 reference epoch in seconds");
    grp_mod->add_option("--mod-tref", sim_cfg.mod_tref,
                        "Modulation reference time in seconds");

    // =========================================================================
    // Subcommand: search ffa
    // =========================================================================
    auto* search =
        app.add_subcommand("search", "Search a timeseries for pulsars");
    search->require_subcommand(1);

    auto* ffa = search->add_subcommand(
        "ffa", "End-to-end Fast Folding Algorithm search");

    loki::search::FFATomlConfig ffa_cfg;
    if (cfg_arg.has_value() && has_subcommand(argc, argv, "search")) {
        try {
            ffa_cfg = loki::search::FFATomlConfig::load(*cfg_arg);
        } catch (const std::exception& ex) {
            SPDLOG_ERROR("{}", ex.what());
            return 1;
        }
    }

    bool ffa_dry_run              = false;
    std::string nsamps_policy_str = "fail";

    std::string ffa_config_path;
    std::string gen_ffa_config_path;
    ffa->add_option("-c,--config", ffa_config_path,
                    "Path to TOML configuration file");
    auto* opt_gen_ffa =
        ffa->add_option("-g,--generate-config", gen_ffa_config_path,
                        "Generate default TOML configuration file [optional "
                        "output path]")
            ->expected(0, 1);

    auto* grp_io = ffa->add_option_group("Input/Output Options");
    grp_io->add_option("-i,--input", ffa_cfg.timeseries_path,
                       "Input timeseries (.tim or .dat)");
    grp_io->add_option("-o,--outdir", ffa_cfg.outdir,
                       "Output directory for candidate files");
    grp_io->add_option("-p,--prefix", ffa_cfg.prefix,
                       "Prefix for output candidate files");
    grp_io->add_flag(
        "--preprocess,!--no-preprocess", ffa_cfg.preprocess,
        "Enable/disable timeseries baseline detrending and normalisation");
    grp_io->add_option(
        "--filter-window", ffa_cfg.filter_window,
        "Running median filter window in seconds for baseline detrending");
    grp_io->add_flag("--fast-median,!--no-fast-median", ffa_cfg.fast_median,
                     "Approximate long running-median windows by block "
                     "averaging (default: on)");
    grp_io->add_option("--fast-median-min-points",
                       ffa_cfg.fast_median_min_points,
                       "Width of the short series used by --fast-median");

    auto* grp_range = ffa->add_option_group("Search Parameter Range");
    grp_range->add_option("--fmin", ffa_cfg.f_min,
                          "Minimum search frequency in Hz");
    grp_range->add_option("--fmax", ffa_cfg.f_max,
                          "Maximum search frequency in Hz");
    grp_range->add_option("--acc-min", ffa_cfg.acc_min,
                          "Minimum acceleration in m/s^2");
    grp_range->add_option("--acc-max", ffa_cfg.acc_max,
                          "Maximum acceleration in m/s^2");
    grp_range->add_option("--jerk-min", ffa_cfg.jerk_min,
                          "Minimum jerk in m/s^3");
    grp_range->add_option("--jerk-max", ffa_cfg.jerk_max,
                          "Maximum jerk in m/s^3");

    auto* grp_search = ffa->add_option_group("Detection & Folding Parameters");
    grp_search->add_option("--nbins", ffa_cfg.nbins,
                           "Phase bin count (default: 64)");
    grp_search->add_option("--eta", ffa_cfg.eta,
                           "Tolerance parameter (default: 1.0)");
    grp_search->add_option("--snr-min", ffa_cfg.snr_min,
                           "Candidate SNR detection threshold (default: 5.0)");
    grp_search->add_option("--ducy-max", ffa_cfg.ducy_max,
                           "Maximum duty cycle to evaluate (default: 0.2)");
    grp_search->add_option("--wtsp", ffa_cfg.wtsp,
                           "Width stepping factor (default: 1.5)");
    grp_search->add_flag(
        "--fourier,!--time", ffa_cfg.use_fourier,
        "Use Fourier-domain FFA folding (default) or Time-domain folding");
    ffa->add_flag("--dry-run", ffa_dry_run,
                  "Validate config and memory planner without loading data");
    ffa->add_option(
        "--nsamps-policy", nsamps_policy_str,
        "When nsamps is not a power of 2: fail (default) or truncate");

    auto* grp_perf = ffa->add_option_group("Performance & Limits");
    grp_perf->add_option(
        "--threads", ffa_cfg.nthreads,
        "OpenMP threads (0 = hardware concurrency, default: 0)");
    grp_perf->add_option(
        "--memory-gb", ffa_cfg.max_process_memory_gb,
        "Memory cap in GB: total process RSS on CPU, device memory on CUDA");
    grp_perf->add_option("--octave-scale", ffa_cfg.octave_scale,
                         "Octave scaling factor (default: 2.0)");
    grp_perf->add_option("--nbins-max", ffa_cfg.nbins_max,
                         "Maximum allowed folding bins (default: 1024)");
    grp_perf->add_option("--nbins-min-lossy-bf", ffa_cfg.nbins_min_lossy_bf,
                         "Minimum bins before lossy brute fold (default: 64)");
    grp_perf->add_option(
        "--max-passing-candidates", ffa_cfg.max_passing_candidates,
        "Maximum candidate buffer capacity (default: 4194304)");

    // Always listed; a CPU-only build rejects --cuda at run time.
    auto* grp_cuda = ffa->add_option_group("CUDA Options");
    grp_cuda->add_flag("--cuda", ffa_cfg.use_cuda,
                       "Execute FFA sweep on NVIDIA GPU using CUDA");
    grp_cuda->add_option("--device", ffa_cfg.device_id,
                         "CUDA GPU device index (default: 0)");

    CLI11_PARSE(app, argc, argv);

    try {
        if (simulate->parsed()) {
            if (*opt_gen_sim) {
                if (gen_sim_config_path.empty()) {
                    gen_sim_config_path = "loki_simulate.toml";
                }
                SimulateTomlConfig::write_default(gen_sim_config_path);
                SPDLOG_INFO("Wrote default simulation configuration to {}",
                            gen_sim_config_path);
                return 0;
            }

            if (sim_cfg.period <= 0.0) {
                throw std::invalid_argument(
                    "--period must be positive (provide via CLI or --config)");
            }
            if (sim_cfg.dt <= 0.0) {
                throw std::invalid_argument(
                    "--dt must be positive (provide via CLI or --config)");
            }
            if (sim_cfg.output.empty()) {
                throw std::invalid_argument("-o/--output must be specified "
                                            "(provide via CLI or --config)");
            }

            const auto path = resolve_output(sim_cfg.output, sim_cfg.format);
            return run_simulate(
                sim_cfg.period, sim_cfg.dt, sim_cfg.nsamps, sim_cfg.snr,
                sim_cfg.ducy, sim_cfg.shape, sim_cfg.phi0, sim_cfg.seed, path,
                sim_cfg.mod_type, sim_cfg.mod, sim_cfg.mod_tref);
        }

        if (search->parsed() && ffa->parsed()) {
            if (*opt_gen_ffa) {
                if (gen_ffa_config_path.empty()) {
                    gen_ffa_config_path = "loki_ffa.toml";
                }
                loki::search::FFATomlConfig::write_default(gen_ffa_config_path);
                SPDLOG_INFO("Wrote default FFA configuration to {}",
                            gen_ffa_config_path);
                return 0;
            }

            std::string config_toml;
            const auto config_path =
                !ffa_config_path.empty()
                    ? std::filesystem::path(ffa_config_path)
                    : (cfg_arg.has_value() ? std::filesystem::path(*cfg_arg)
                                           : std::filesystem::path{});
            if (!config_path.empty() && std::filesystem::exists(config_path)) {
                config_toml = read_text_file(config_path);
            }

            NsampsPolicy policy = NsampsPolicy::kFail;
            if (nsamps_policy_str == "truncate") {
                policy = NsampsPolicy::kTruncate;
            } else if (nsamps_policy_str != "fail") {
                throw std::invalid_argument(
                    "--nsamps-policy must be fail or truncate");
            }

            return run_search_ffa(ffa_cfg, config_toml, ffa_dry_run, policy);
        }
    } catch (const std::exception& ex) {
        SPDLOG_ERROR("{}", ex.what());
        return 1;
    }

    return 0;
}
