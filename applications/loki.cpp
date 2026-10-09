#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>
#include <highfive/highfive.hpp>
#include <omp.h>
#include <spdlog/spdlog.h>
#include <toml++/toml.hpp>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"
#include "loki/simulation/modulate.hpp"
#include "loki/simulation/pulse.hpp"

namespace {

using loki::SizeType;

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

std::optional<std::string> find_config_arg(int argc, char* const* argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            return std::string(argv[i + 1]);
        }
        if (arg.starts_with("--config=")) {
            return std::string(arg.substr(9));
        }
    }
    return std::nullopt;
}

bool has_subcommand(int argc, char* const* argv, std::string_view name) {
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

    static std::string_view default_toml_string() {
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
            if (auto const* n = sig->get("nsamps")) {
                if (auto val = n->value<int64_t>()) {
                    cfg.nsamps = static_cast<loki::SizeType>(*val);
                }
            }
            cfg.snr   = sig->get("snr")->value_or(cfg.snr);
            cfg.ducy  = sig->get("ducy")->value_or(cfg.ducy);
            cfg.shape = sig->get("shape")->value_or(cfg.shape);
            cfg.phi0  = sig->get("phi0")->value_or(cfg.phi0);
            if (auto const* s = sig->get("seed")) {
                if (auto val = s->value<int64_t>()) {
                    cfg.seed = static_cast<std::uint64_t>(*val);
                }
            }
        }

        if (auto* mod_tbl = tbl["modulation"].as_table()) {
            cfg.mod_type = mod_tbl->get("type")->value_or(cfg.mod_type);
            if (auto const* tr = mod_tbl->get("mod_tref")) {
                cfg.mod_tref = tr->value<double>();
            }
            cfg.mod.shift = mod_tbl->get("shift")->value_or(cfg.mod.shift);
            cfg.mod.vel   = mod_tbl->get("vel")->value_or(cfg.mod.vel);
            cfg.mod.acc   = mod_tbl->get("acc")->value_or(cfg.mod.acc);
            cfg.mod.jerk  = mod_tbl->get("jerk")->value_or(cfg.mod.jerk);
            cfg.mod.snap  = mod_tbl->get("snap")->value_or(cfg.mod.snap);
            if (auto const* coeffs_arr =
                    mod_tbl->get_as<toml::array>("coeffs")) {
                cfg.mod.coeffs.clear();
                for (const auto& elem : *coeffs_arr) {
                    if (auto val = elem.value<double>()) {
                        cfg.mod.coeffs.push_back(*val);
                    }
                }
            }
            cfg.mod.p_orb = mod_tbl->get("p_orb")->value_or(cfg.mod.p_orb);
            cfg.mod.psi   = mod_tbl->get("psi")->value_or(cfg.mod.psi);
            if (auto const* xo = mod_tbl->get("x_orb")) {
                cfg.mod.x_orb = xo->value<double>();
            }
            if (auto const* mc = mod_tbl->get("m_c")) {
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
    const auto series = config.generate(shape, phi0);
    series.write(output);
    SPDLOG_INFO("Wrote {} samples to {}", series.get_nsamps(), output.string());
    return 0;
}

enum class NsampsPolicy : std::uint8_t { kFail, kTruncate };

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open config file: " +
                                 path.string());
    }
    return {std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>()};
}

/// Shared by the FFA and EP searches: the frequency range must be ordered and
/// start above zero.
void check_frequency_range(const loki::search::FFATomlConfig& cfg) {
    if (cfg.f_min <= 0.0 || cfg.f_max <= cfg.f_min) {
        throw std::invalid_argument(
            std::format("Invalid frequency range: [{:.3f}, {:.3f}] Hz",
                        cfg.f_min, cfg.f_max));
    }
}

/// The CPU thread count travels in the search config; Exec only picks the
/// backend and device. Construction rejects a backend this build lacks.
loki::Exec search_exec(const loki::search::FFATomlConfig& cfg) {
    const loki::Exec exec{
        .backend  = cfg.backend,
        .nthreads = 1,
        .device   = cfg.device,
    };
    if (!loki::is_available(exec.backend)) {
        throw std::runtime_error(std::format(
            "backend '{}' was requested but this build does not contain it",
            loki::to_string(exec.backend)));
    }
    return exec;
}

/// A search timeseries and the summary of its preprocessing.
struct SearchTimeSeries {
    loki::io::TimeSeries ts;
    std::optional<loki::io::PreprocessReport> report;
};

/// Reads the raw timeseries of a search from [input] (or -i) and builds the
/// folding inputs as [preprocessing] asks.
SearchTimeSeries
load_search_timeseries(const loki::search::FFATomlConfig& cfg) {
    if (cfg.timeseries_path.empty()) {
        throw std::invalid_argument(
            "Input timeseries path must be specified via -i/--input or "
            "[input].timeseries in config");
    }
    const std::filesystem::path ts_path = cfg.timeseries_path;
    if (!std::filesystem::exists(ts_path)) {
        throw std::runtime_error(std::format(
            "Input timeseries does not exist: {}", ts_path.string()));
    }

    // Config value 0 means "use all hardware threads".
    const int nthreads =
        cfg.nthreads <= 0 ? omp_get_max_threads() : cfg.nthreads;
    loki::io::ReadOptions read_opts;
    read_opts.preprocess = false;
    read_opts.nthreads   = nthreads;
    SPDLOG_INFO("Loading timeseries from: {}", ts_path.string());
    SearchTimeSeries out{.ts = loki::io::TimeSeries::read(ts_path, read_opts),
                         .report = std::nullopt};
    auto& ts = out.ts;
    SPDLOG_INFO(
        "Loaded timeseries: nsamps = {}, dt = {:.6e} s, tobs = {:.2f} s",
        ts.get_nsamps(), ts.get_dt(), ts.get_tobs());
    if (!cfg.preprocess) {
        return out;
    }
    const auto options = cfg.to_preprocess_options();
    const bool robust  = options.method == loki::io::PreprocessMethod::kRobust;
    const auto rep     = ts.preprocess(options, loki::Exec::cpu(nthreads));
    const double dt    = ts.get_dt();
    if (robust) {
        SPDLOG_INFO(
            "Preprocessing (robust): baseline window = {:.3f} s, variance "
            "window = {:.3f} s, block = {} samples, masked = {:.3f}%, "
            "clipped = {}, zapped bins = {}, longest masked run = {:.3f} s",
            static_cast<double>(rep.baseline_window) * dt,
            static_cast<double>(rep.variance_window) * dt, rep.block_size,
            100.0 * static_cast<double>(rep.n_masked) /
                static_cast<double>(rep.nsamps),
            rep.n_clipped, rep.n_zapped,
            static_cast<double>(rep.longest_masked_run) * dt);
    } else {
        SPDLOG_INFO("Preprocessing (zscore): baseline window = {:.3f} s",
                    static_cast<double>(rep.baseline_window) * dt);
    }
    out.report = rep;
    return out;
}

/// Masked runs as long as a brute-fold segment can leave a fold bin with no
/// weight in the per-segment scores of a pruning search.
void warn_long_masked_run(const std::optional<loki::io::PreprocessReport>& rep,
                          loki::SizeType bseg_brute) {
    if (rep && 2 * rep->longest_masked_run >= bseg_brute) {
        SPDLOG_WARN("Preprocessing masked a contiguous run of {} samples, "
                    "comparable to the brute-fold segment ({} samples); "
                    "scores of that segment are unreliable",
                    rep->longest_masked_run, bseg_brute);
    }
}

/// The sample count a search runs on, as --nsamps-policy asks for.
loki::SizeType power_of_two_nsamps(loki::SizeType actual_nsamps,
                                   NsampsPolicy policy) {
    if (std::has_single_bit(actual_nsamps)) {
        return actual_nsamps;
    }
    if (policy == NsampsPolicy::kFail) {
        throw std::invalid_argument(std::format(
            "Timeseries length {} is not a power of 2; use --nsamps-policy "
            "truncate",
            actual_nsamps));
    }
    const auto pow2_nsamps = std::bit_floor(actual_nsamps);
    SPDLOG_WARN("Timeseries length {} is not a power of 2; truncating search "
                "to {}",
                actual_nsamps, pow2_nsamps);
    return pow2_nsamps;
}

int run_search_ffa(const loki::search::FFATomlConfig& toml_cfg,
                   std::string_view config_toml,
                   bool dry_run,
                   NsampsPolicy nsamps_policy) {
    check_frequency_range(toml_cfg);
    const loki::Exec exec = search_exec(toml_cfg);

    const auto preview_cfg = toml_cfg.to_search_config(
        toml_cfg.nsamps.value_or(1U << 21U), toml_cfg.tsamp.value_or(6.4e-5));
    if (dry_run) {
        const loki::pipelines::FFAFreqSweep dry(preview_cfg,
                                                /*show_progress=*/false, exec);
        SPDLOG_INFO("Dry run complete: planner constructed successfully.");
        return 0;
    }

    auto loaded = load_search_timeseries(toml_cfg);
    auto& ts    = loaded.ts;
    const loki::SizeType actual_nsamps =
        power_of_two_nsamps(ts.get_nsamps(), nsamps_policy);

    const auto ffa_cfg = toml_cfg.to_search_config(actual_nsamps, ts.get_dt());
    warn_long_masked_run(loaded.report, ffa_cfg.get_bseg_brute());

    const std::filesystem::path outdir_path = toml_cfg.outdir;
    std::filesystem::create_directories(outdir_path);

    SPDLOG_INFO("Starting FFA search: f0=[{:.3f}, {:.3f}] Hz, nbins={}, "
                "eta={:.2f}, snr_min={:.2f}",
                toml_cfg.f_min, toml_cfg.f_max, ffa_cfg.get_nbins(),
                ffa_cfg.get_eta(), ffa_cfg.get_snr_min());

    if (exec.backend != loki::Backend::kCPU) {
        SPDLOG_INFO("Using {} backend on device {}",
                    loki::to_string(exec.backend), exec.device);
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
            const HighFive::File h5(result_file.string(),
                                    HighFive::File::ReadOnly);
            if (h5.exist("snr")) {
                const auto snr_dset = h5.getDataSet("snr");
                const auto dims     = snr_dset.getDimensions();
                const auto n_cands  = dims.empty() ? 0UL : dims[0];
                if (n_cands > 0) {
                    float top_snr           = 0.0F;
                    size_t top_idx          = 0;
                    constexpr size_t kChunk = 65536;
                    for (size_t offset = 0; offset < n_cands;
                         offset += kChunk) {
                        const size_t count = std::min(kChunk, n_cands - offset);
                        std::vector<float> snrs(count);
                        snr_dset.select({offset}, {count}).read(snrs);
                        const auto max_it = std::ranges::max_element(snrs);
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
                        const auto p_dset = h5.getDataSet("param_sets");
                        auto p_dims       = p_dset.getDimensions();
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

// =============================================================================
// EP search (`search ep`)
// =============================================================================

/// Writes the chunk table, the groups of equal nbins and the memory peak of an
/// EP plan to stdout.
template <typename FoldType>
void print_ep_plan(const loki::algorithms::EPRegionPlanner<FoldType>& planner,
                   SizeType workers,
                   const std::filesystem::path& cache_path) {
    const auto& chunks = planner.get_chunk_cfgs();
    const auto& stats  = planner.get_stats();
    if (chunks.empty()) {
        std::cout << "EP plan: no chunks (the frequency range is empty)\n";
        return;
    }

    SizeType nbins_lo = chunks.front().cfg.get_nbins();
    SizeType nbins_hi = nbins_lo;
    for (const auto& chunk : chunks) {
        nbins_lo = std::min(nbins_lo, chunk.cfg.get_nbins());
        nbins_hi = std::max(nbins_hi, chunk.cfg.get_nbins());
    }
    std::cout << std::format(
        "EP plan: {} chunks, nbins {}..{}, peak {:.3f} GB against a limit of "
        "{:.3f} GB\n",
        chunks.size(), nbins_lo, nbins_hi, stats.get_max_memory_gb(),
        stats.get_memory_limit_gb());

    std::cout << std::format(
        "{:>6} {:>6} {:>23} {:>23} {:>9} {:>9} {:>10} {:>6} {:>9}\n", "chunk",
        "nbins", "nominal f [Hz]", "actual f [Hz]", "ncoords", "max_sugg",
        "branch_max", "nseg", "mem [GB]");
    for (SizeType i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        std::cout << std::format(
            "{:>6} {:>6} {:>10.3f} - {:<10.3f} {:>10.3f} - {:<10.3f} {:>9} "
            "{:>9} {:>10} {:>6} {:>9.3f}\n",
            i, chunk.cfg.get_nbins(), chunk.nominal_f_start,
            chunk.nominal_f_end, chunk.actual_f_start, chunk.actual_f_end,
            chunk.ncoords, chunk.max_sugg, chunk.branch_max, chunk.nsegments,
            chunk.chunk_memory_gb);
    }

    std::cout << "Groups of equal nbins, each with its own workspaces:\n";
    for (SizeType begin = 0; begin < chunks.size();) {
        const auto nbins    = chunks[begin].cfg.get_nbins();
        SizeType end        = begin;
        SizeType max_sugg   = 0;
        SizeType branch_max = 0;
        while (end < chunks.size() && chunks[end].cfg.get_nbins() == nbins) {
            max_sugg   = std::max(max_sugg, chunks[end].max_sugg);
            branch_max = std::max(branch_max, chunks[end].branch_max);
            ++end;
        }
        std::cout << std::format(
            "  chunks {}..{}: nbins {}, max_sugg {}, branch_max {}, {} "
            "worker(s)\n",
            begin, end - 1, nbins, max_sugg, branch_max, workers);
        begin = end;
    }
    std::cout << std::format("Plan cache: {}\n", cache_path.string());
}

/// `search ep --plan-only`: plans the chunks, writes the plan cache and prints
/// the plan. No timeseries is read, so the sample count comes from the config.
int run_plan_ep(const loki::search::EPTomlConfig& toml_cfg) {
    check_frequency_range(toml_cfg);
    const loki::Exec exec = search_exec(toml_cfg);
    if (!toml_cfg.nsamps.has_value() || !toml_cfg.tsamp.has_value()) {
        throw std::invalid_argument(
            "--plan-only needs the sample count and interval: set [input] "
            "nsamps and tsamp, or pass --nsamps and --tsamp (the plan depends "
            "on nsamps)");
    }
    const auto cfg =
        toml_cfg.to_ep_search_config(*toml_cfg.nsamps, *toml_cfg.tsamp);
    const auto cache_path = toml_cfg.plan_cache.value_or(
        std::filesystem::path(toml_cfg.outdir) /
        std::format("{}_ep_plan.h5", toml_cfg.prefix));
    // The same worker count the sweep uses, so the cache matches it. The GPU
    // sweep prunes its runs on one worker.
    const SizeType n_workers = loki::algorithms::ep_sweep_n_workers(
        cfg.get_nthreads(), toml_cfg.n_runs, toml_cfg.ref_segs);
    const SizeType workers =
        exec.backend == loki::Backend::kCPU ? n_workers : SizeType{1};

    SPDLOG_INFO("Planning EP search (no data read): f=[{:.3f}, {:.3f}] Hz, "
                "nsamps={}, nbins={}, backend={}",
                cfg.get_f_min(), cfg.get_f_max(), cfg.get_nsamps(),
                cfg.get_nbins(), loki::to_string(exec.backend));

    if (cfg.get_use_fourier()) {
        const loki::algorithms::EPRegionPlanner<loki::ComplexType> planner(
            cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
            cache_path, {}, n_workers, exec);
        print_ep_plan(planner, workers, cache_path);
    } else {
        const loki::algorithms::EPRegionPlanner<float> planner(
            cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
            cache_path, {}, n_workers, exec);
        print_ep_plan(planner, workers, cache_path);
    }
    return 0;
}

/// Prints the best score over every run of an EP result file, and the
/// parameters of that leaf. A run has `scores` (one float per leaf) and
/// `param_sets` (n_params + 2 rows of a (value, error) pair per leaf). Rows
/// 0..n_params-1 are the parameters, in the order of `param_names`.
void print_ep_summary(const std::filesystem::path& result_file) {
    if (!std::filesystem::exists(result_file)) {
        return;
    }
    try {
        const HighFive::File h5(result_file.string(), HighFive::File::ReadOnly);
        std::vector<std::string> param_names;
        if (h5.hasAttribute("param_names")) {
            h5.getAttribute("param_names").read(param_names);
        }
        const SizeType n_params = param_names.size();

        SizeType n_runs   = 0;
        SizeType n_leaves = 0;
        float top_score   = std::numeric_limits<float>::lowest();
        bool found        = false;
        std::string top_chunk;
        std::string top_run;
        SizeType top_leaf = 0;
        if (h5.exist("chunks")) {
            const auto chunks = h5.getGroup("chunks");
            for (const auto& chunk_name : chunks.listObjectNames()) {
                const auto chunk = chunks.getGroup(chunk_name);
                if (!chunk.exist("runs")) {
                    continue;
                }
                const auto runs = chunk.getGroup("runs");
                for (const auto& run_name : runs.listObjectNames()) {
                    const auto run = runs.getGroup(run_name);
                    if (!run.exist("scores")) {
                        continue;
                    }
                    ++n_runs;
                    std::vector<float> scores;
                    run.getDataSet("scores").read(scores);
                    n_leaves += scores.size();
                    for (SizeType leaf = 0; leaf < scores.size(); ++leaf) {
                        if (scores[leaf] > top_score) {
                            top_score = scores[leaf];
                            top_chunk = chunk_name;
                            top_run   = run_name;
                            top_leaf  = leaf;
                            found     = true;
                        }
                    }
                }
            }
        }

        SPDLOG_INFO("=== Search Summary ===");
        SPDLOG_INFO("Runs pruned                   : {} ({} leaves)", n_runs,
                    n_leaves);
        if (found) {
            const auto run    = h5.getGroup("chunks")
                                    .getGroup(top_chunk)
                                    .getGroup("runs")
                                    .getGroup(top_run);
            const auto p_dset = run.getDataSet("param_sets");
            std::vector<double> leaf_values((n_params + 2) * 2);
            p_dset.select({top_leaf, 0, 0}, {1, n_params + 2, 2})
                .read_raw(leaf_values.data());
            std::string p_str;
            for (SizeType pi = 0; pi < n_params; ++pi) {
                if (pi > 0) {
                    p_str += ", ";
                }
                p_str += std::format("{}={:.6f}", param_names[pi],
                                     leaf_values[pi * 2]);
            }
            SPDLOG_INFO("Top score                      : {:.2f} (chunk {}, "
                        "run {}, leaf {})",
                        top_score, top_chunk, top_run, top_leaf);
            SPDLOG_INFO("Top candidate parameters      : [{}]", p_str);
        } else {
            SPDLOG_INFO("No leaves were written by the pruning");
        }
        SPDLOG_INFO("Saved HDF5 results to: {}", result_file.string());
    } catch (const std::exception& e) {
        SPDLOG_WARN("HDF5 result summary inspection failed: {}", e.what());
    }
}

/// `search ep`: runs the EP sweep over the timeseries and prints a summary.
int run_search_ep(const loki::search::EPTomlConfig& toml_cfg,
                  bool dry_run,
                  NsampsPolicy nsamps_policy) {
    check_frequency_range(toml_cfg);
    const loki::Exec exec  = search_exec(toml_cfg);
    const auto preview_cfg = toml_cfg.to_ep_search_config(
        toml_cfg.nsamps.value_or(1U << 21U), toml_cfg.tsamp.value_or(6.4e-5));
    if (dry_run) {
        const loki::pipelines::EPFreqSweep dry(
            preview_cfg, /*show_progress=*/false, toml_cfg.min_pd,
            toml_cfg.poly_basis, toml_cfg.ref_ducy, {}, toml_cfg.plan_cache,
            toml_cfg.n_runs, toml_cfg.ref_segs, exec);
        SPDLOG_INFO("Dry run complete: planner constructed successfully.");
        return 0;
    }

    auto loaded = load_search_timeseries(toml_cfg);
    auto& ts    = loaded.ts;
    const loki::SizeType actual_nsamps =
        power_of_two_nsamps(ts.get_nsamps(), nsamps_policy);
    const auto ep_cfg =
        toml_cfg.to_ep_search_config(actual_nsamps, ts.get_dt());
    warn_long_masked_run(loaded.report, ep_cfg.get_bseg_brute());

    const std::filesystem::path outdir_path = toml_cfg.outdir;
    std::filesystem::create_directories(outdir_path);

    SPDLOG_INFO("Starting EP search: f0=[{:.3f}, {:.3f}] Hz, nbins={}, "
                "eta={:.2f}, poly_basis={}, min_pd={:.3f}",
                toml_cfg.f_min, toml_cfg.f_max, ep_cfg.get_nbins(),
                ep_cfg.get_eta(), toml_cfg.poly_basis, toml_cfg.min_pd);
    if (exec.backend != loki::Backend::kCPU) {
        SPDLOG_INFO("Using {} backend on device {}",
                    loki::to_string(exec.backend), exec.device);
    }

    loki::pipelines::EPFreqSweep sweep(
        ep_cfg, /*show_progress=*/true, toml_cfg.min_pd, toml_cfg.poly_basis,
        toml_cfg.ref_ducy, {}, toml_cfg.plan_cache, toml_cfg.n_runs,
        toml_cfg.ref_segs, exec);
    sweep.execute(ts.get_ts_e().first(actual_nsamps),
                  ts.get_ts_v().first(actual_nsamps), outdir_path,
                  toml_cfg.prefix);

    print_ep_summary(outdir_path /
                     std::format("{}_ep_results.h5", toml_cfg.prefix));
    return 0;
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape,misc-const-correctness): setup errors abort; main signature is fixed
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
    auto const* opt_gen_sim =
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

    // Each search reads only its own file type: an [ep] file is not an FFA
    // file.
    loki::search::FFATomlConfig ffa_cfg;
    if (cfg_arg.has_value() && has_subcommand(argc, argv, "ffa")) {
        try {
            ffa_cfg = loki::search::FFATomlConfig::load(*cfg_arg);
        } catch (const std::exception& ex) {
            SPDLOG_ERROR("{}", ex.what());
            return 1;
        }
    }

    loki::search::EPTomlConfig ep_cfg;
    if (cfg_arg.has_value() && has_subcommand(argc, argv, "ep")) {
        try {
            ep_cfg = loki::search::EPTomlConfig::load(*cfg_arg);
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
    auto const* opt_gen_ffa =
        ffa->add_option("-g,--generate-config", gen_ffa_config_path,
                        "Generate default TOML configuration file [optional "
                        "output path]")
            ->expected(0, 1);

    std::string preproc_method;
    auto* grp_io = ffa->add_option_group("Input/Output Options");
    grp_io->add_option("-i,--input", ffa_cfg.timeseries_path,
                       "Input timeseries (.tim or .dat)");
    grp_io->add_option("-o,--outdir", ffa_cfg.outdir,
                       "Output directory for candidate files");
    grp_io->add_option("-p,--prefix", ffa_cfg.prefix,
                       "Prefix for output candidate files");
    grp_io->add_flag("--preprocess,!--no-preprocess", ffa_cfg.preprocess,
                     "Enable/disable preprocessing of the raw timeseries");
    grp_io
        ->add_option("--preproc-method", preproc_method,
                     "Preprocessing method: robust (default) or zscore "
                     "(legacy detrend + z-score)")
        ->transform(CLI::IsMember({"robust", "zscore"}, CLI::ignore_case));
    grp_io->add_option("--filter-window", ffa_cfg.preprocessing.filter_window,
                       "Baseline (running median) window in seconds");
    grp_io->add_flag("--fast-median,!--no-fast-median",
                     ffa_cfg.preprocessing.fast_median,
                     "zscore: approximate long running-median windows by "
                     "block averaging (default: on)");
    grp_io->add_option("--fast-median-min-points",
                       ffa_cfg.preprocessing.fast_median_min_points,
                       "zscore: width of the short series used by "
                       "--fast-median");

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

    // Always listed; a build without the chosen backend rejects it at run
    // time with the list of available backends.
    std::string backend_name;
    grp_perf
        ->add_option("--backend", backend_name,
                     "Execution backend: cpu or cuda (default: cpu)")
        ->transform(CLI::IsMember({"cpu", "cuda"}, CLI::ignore_case));
    grp_perf->add_option("--device", ffa_cfg.device,
                         "GPU device ordinal for --backend cuda (default: 0)");

    // =========================================================================
    // Subcommand: search ep
    // =========================================================================
    auto* ep =
        search->add_subcommand("ep", "End-to-end Extreme Pruning (EP) search");

    std::string ep_config_path;
    std::string gen_ep_config_path;
    ep->add_option("-c,--config", ep_config_path,
                   "Path to TOML configuration file");
    auto const* opt_gen_ep =
        ep->add_option("-g,--generate-config", gen_ep_config_path,
                       "Generate default TOML configuration file [optional "
                       "output path]")
            ->expected(0, 1);

    bool ep_dry_run                  = false;
    bool ep_plan_only                = false;
    std::string ep_nsamps_policy_str = "fail";
    std::string ep_plan_cache_path;
    std::string ep_backend_name;
    std::vector<SizeType> ep_ref_segs;

    std::string ep_preproc_method;
    auto* ep_grp_io = ep->add_option_group("Input/Output Options");
    ep_grp_io->add_option("-i,--input", ep_cfg.timeseries_path,
                          "Input timeseries (.tim or .dat)");
    ep_grp_io->add_option("-o,--outdir", ep_cfg.outdir,
                          "Output directory for the results file");
    ep_grp_io->add_option("-p,--prefix", ep_cfg.prefix,
                          "Prefix for the results file");
    ep_grp_io->add_flag("--preprocess,!--no-preprocess", ep_cfg.preprocess,
                        "Enable/disable preprocessing of the raw timeseries");
    ep_grp_io
        ->add_option("--preproc-method", ep_preproc_method,
                     "Preprocessing method: robust (default) or zscore "
                     "(legacy detrend + z-score)")
        ->transform(CLI::IsMember({"robust", "zscore"}, CLI::ignore_case));
    ep_grp_io->add_option("--filter-window", ep_cfg.preprocessing.filter_window,
                          "Baseline (running median) window in seconds");
    ep_grp_io->add_flag("--fast-median,!--no-fast-median",
                        ep_cfg.preprocessing.fast_median,
                        "zscore: approximate long running-median windows by "
                        "block averaging (default: on)");
    ep_grp_io->add_option("--fast-median-min-points",
                          ep_cfg.preprocessing.fast_median_min_points,
                          "zscore: width of the short series used by "
                          "--fast-median");
    ep_grp_io->add_option("--nsamps", ep_cfg.nsamps,
                          "Sample count, for --plan-only (a search reads it "
                          "from the timeseries)");
    ep_grp_io->add_option("--tsamp", ep_cfg.tsamp,
                          "Sample interval in seconds, for --plan-only");
    ep_grp_io->add_option("--plan-cache", ep_plan_cache_path,
                          "Plan cache file (HDF5) to write or reuse");

    auto* ep_grp_range = ep->add_option_group("Search Parameter Range");
    ep_grp_range->add_option("--fmin", ep_cfg.f_min,
                             "Minimum search frequency in Hz");
    ep_grp_range->add_option("--fmax", ep_cfg.f_max,
                             "Maximum search frequency in Hz");
    ep_grp_range->add_option("--acc-min", ep_cfg.acc_min,
                             "Minimum acceleration in m/s^2");
    ep_grp_range->add_option("--acc-max", ep_cfg.acc_max,
                             "Maximum acceleration in m/s^2");
    ep_grp_range->add_option("--jerk-min", ep_cfg.jerk_min,
                             "Minimum jerk in m/s^3");
    ep_grp_range->add_option("--jerk-max", ep_cfg.jerk_max,
                             "Maximum jerk in m/s^3");

    auto* ep_grp_search =
        ep->add_option_group("Detection & Folding Parameters");
    ep_grp_search->add_option("--nbins", ep_cfg.nbins,
                              "Phase bin count (default: 64)");
    ep_grp_search->add_option("--eta", ep_cfg.eta,
                              "Tolerance parameter (default: 1.0)");
    ep_grp_search->add_option(
        "--snr-min", ep_cfg.snr_min,
        "Candidate SNR detection threshold (default: 5.0)");
    ep_grp_search->add_option("--ducy-max", ep_cfg.ducy_max,
                              "Maximum duty cycle to evaluate (default: 0.2)");
    ep_grp_search->add_option("--wtsp", ep_cfg.wtsp,
                              "Width stepping factor (default: 1.5)");
    ep_grp_search->add_flag(
        "--fourier,!--time", ep_cfg.use_fourier,
        "Use Fourier-domain folding (default) or time-domain folding");

    auto* ep_grp_pruning = ep->add_option_group("EP Pruning Parameters");
    ep_grp_pruning
        ->add_option("--poly-basis", ep_cfg.poly_basis,
                     "Polynomial basis: taylor (default) or chebyshev")
        ->check(CLI::IsMember({"taylor", "chebyshev"}));
    ep_grp_pruning->add_option(
        "--min-pd", ep_cfg.min_pd,
        "Minimum detection probability of the threshold scheme, in (0, 1] "
        "(default: 0.1)");
    ep_grp_pruning->add_option(
        "--ref-ducy", ep_cfg.ref_ducy,
        "Reference duty cycle of the pruning, in (0, 1] (default: 0.1)");
    ep_grp_pruning->add_option("--prune-poly-order", ep_cfg.prune_poly_order,
                               "Polynomial order of the pruning (default: 3)");
    ep_grp_pruning->add_option("--n-runs", ep_cfg.n_runs,
                               "Number of runs to prune (default: all)");
    ep_grp_pruning
        ->add_option("--ref-segs", ep_ref_segs,
                     "Explicit reference segments, comma separated")
        ->delimiter(',');
    ep_grp_pruning->add_option("--p-orb-min", ep_cfg.p_orb_min,
                               "Minimum orbital period prior in seconds");
    ep_grp_pruning->add_option("--m-c-max", ep_cfg.m_c_max,
                               "Maximum companion mass prior in solar masses");
    ep_grp_pruning->add_option("--m-p-min", ep_cfg.m_p_min,
                               "Minimum pulsar mass prior in solar masses");
    ep_grp_pruning->add_option("--propagator-significance",
                               ep_cfg.propagator_significance,
                               "Propagator significance level (default: 2.0)");
    ep_grp_pruning->add_option("--validation-significance",
                               ep_cfg.validation_significance,
                               "Validation significance level (default: 5.0)");
    ep_grp_pruning->add_flag(
        "--conservative-tile,!--no-conservative-tile",
        ep_cfg.use_conservative_tile,
        "Use the conservative tile of the Fourier search (default: off)");

    auto* ep_grp_perf = ep->add_option_group("Performance & Limits");
    ep_grp_perf->add_option(
        "--threads", ep_cfg.nthreads,
        "OpenMP threads (0 = hardware concurrency, default: 0)");
    ep_grp_perf->add_option(
        "--memory-gb", ep_cfg.max_process_memory_gb,
        "Memory cap in GB: total process RSS on CPU, device memory on CUDA");
    ep_grp_perf->add_option("--octave-scale", ep_cfg.octave_scale,
                            "Octave scaling factor (default: 2.0)");
    ep_grp_perf->add_option("--nbins-max", ep_cfg.nbins_max,
                            "Maximum allowed folding bins (default: 1024)");
    ep_grp_perf->add_option(
        "--nbins-min-lossy-bf", ep_cfg.nbins_min_lossy_bf,
        "Minimum bins before lossy brute fold (default: 64)");
    ep_grp_perf->add_option(
        "--max-passing-candidates", ep_cfg.max_passing_candidates,
        "Maximum candidate buffer capacity (default: 4194304)");
    ep_grp_perf
        ->add_option("--backend", ep_backend_name,
                     "Execution backend: cpu or cuda (default: cpu)")
        ->transform(CLI::IsMember({"cpu", "cuda"}, CLI::ignore_case));
    ep_grp_perf->add_option(
        "--device", ep_cfg.device,
        "GPU device ordinal for --backend cuda (default: 0)");

    ep->add_flag("--dry-run", ep_dry_run,
                 "Validate config and build the sweep planner without loading "
                 "data");
    ep->add_flag("--plan-only", ep_plan_only,
                 "Plan the chunks, write the plan cache and print the plan; "
                 "reads no data");
    ep->add_option("--nsamps-policy", ep_nsamps_policy_str,
                   "When nsamps is not a power of 2: fail (default) or "
                   "truncate");

    CLI11_PARSE(app, argc, argv);
    if (!backend_name.empty()) {
        ffa_cfg.backend = loki::parse_backend(backend_name);
    }
    if (!ep_backend_name.empty()) {
        ep_cfg.backend = loki::parse_backend(ep_backend_name);
    }
    const auto parse_method = [](const std::string& name) {
        return name == "zscore" ? loki::io::PreprocessMethod::kZScore
                                : loki::io::PreprocessMethod::kRobust;
    };
    if (!preproc_method.empty()) {
        ffa_cfg.preprocessing.method = parse_method(preproc_method);
    }
    if (!ep_preproc_method.empty()) {
        ep_cfg.preprocessing.method = parse_method(ep_preproc_method);
    }
    if (!ep_plan_cache_path.empty()) {
        ep_cfg.plan_cache = std::filesystem::path(ep_plan_cache_path);
    }
    if (!ep_ref_segs.empty()) {
        ep_cfg.ref_segs = ep_ref_segs;
    }

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
            std::filesystem::path config_path;
            if (!ffa_config_path.empty()) {
                config_path = ffa_config_path;
            } else if (cfg_arg.has_value()) {
                config_path = *cfg_arg;
            }
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

        if (search->parsed() && ep->parsed()) {
            if (*opt_gen_ep) {
                if (gen_ep_config_path.empty()) {
                    gen_ep_config_path = "loki_ep.toml";
                }
                loki::search::EPTomlConfig::write_default(gen_ep_config_path);
                SPDLOG_INFO("Wrote default EP configuration to {}",
                            gen_ep_config_path);
                return 0;
            }
            if (ep_dry_run && ep_plan_only) {
                throw std::invalid_argument(
                    "--dry-run and --plan-only are separate modes; pass one");
            }

            NsampsPolicy policy = NsampsPolicy::kFail;
            if (ep_nsamps_policy_str == "truncate") {
                policy = NsampsPolicy::kTruncate;
            } else if (ep_nsamps_policy_str != "fail") {
                throw std::invalid_argument(
                    "--nsamps-policy must be fail or truncate");
            }

            if (ep_plan_only) {
                return run_plan_ep(ep_cfg);
            }
            return run_search_ep(ep_cfg, ep_dry_run, policy);
        }
    } catch (const std::exception& ex) {
        SPDLOG_ERROR("{}", ex.what());
        return 1;
    }

    return 0;
}
