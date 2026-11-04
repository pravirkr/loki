#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <highfive/highfive.hpp>
#include <omp.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/interrupt.hpp"
#include "loki/common/progress.hpp"
#include "loki/common/types.hpp"
#include "loki/io/candidate_file.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"
#include "loki/search/provenance.hpp"
#include "loki/simulation/configs.hpp"
#include "loki/simulation/modulate.hpp"
#include "loki/simulation/pulse.hpp"

#include "console.hpp"
#include "help_format.hpp"

namespace {

using loki::SizeType;

#ifndef LOKI_VERSION
#define LOKI_VERSION "unknown"
#endif

/// Console and log-file settings shared by every subcommand.
struct OutputOptions {
    int verbose{0}; // -v: debug, -vv: trace
    int quiet{0};   // -q: warnings, no progress bars; -qq: errors only
    std::string log_file;
};

std::string version_text() {
    std::string backends;
    for (const auto b : loki::available_backends()) {
        backends += backends.empty() ? "" : ", ";
        backends += loki::to_string(b);
    }
    return fmt::format("loki {} (backends: {})", LOKI_VERSION, backends);
}

/// The command line of this run, set by configure_output().
std::string& run_command_line() {
    static std::string command;
    return command;
}

/// Applies @p opts to the default logger and the progress mode. The console
/// shows its level; a log file gets at least the info messages, whatever
/// the console shows (more with -v), and is appended to.
void configure_output(const OutputOptions& opts,
                      int argc,
                      const char* const* argv) {
    auto& console_out = loki::cli::Console::get();
    console_out.configure(/*verbose=*/opts.verbose > 0);
    auto console = spdlog::level::info;
    if (opts.quiet >= 2) {
        console = spdlog::level::err;
    } else if (opts.quiet == 1) {
        console = spdlog::level::warn;
    } else if (opts.verbose == 1) {
        console = spdlog::level::debug;
    } else if (opts.verbose >= 2) {
        console = spdlog::level::trace;
    }
    const auto sink_colors = console_out.color() ? spdlog::color_mode::always
                                                 : spdlog::color_mode::never;
    const auto logger      = spdlog::default_logger();
    for (const auto& sink : logger->sinks()) {
        sink->set_level(console);
        if (const auto color_sink = std::dynamic_pointer_cast<
                spdlog::sinks::ansicolor_sink<spdlog::details::console_mutex>>(
                sink)) {
            color_sink->set_color_mode(sink_colors);
        }
    }
    logger->set_level(console);
    if (!opts.log_file.empty()) {
        const auto file_level = std::min(console, spdlog::level::info);
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            opts.log_file, /*truncate=*/false);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
        file_sink->set_level(file_level);
        loki::cli::Console::get().set_log_file(file_sink);
        logger->sinks().push_back(std::move(file_sink));
        logger->set_level(std::min(console, file_level));
        // Every line reaches the file at once, even if the job is killed.
        logger->flush_on(spdlog::level::trace);
    }
    if (opts.quiet > 0) {
        loki::set_progress_mode(loki::ProgressMode::kOff);
    }
    std::string& command = run_command_line();
    command.clear();
    for (int i = 0; i < argc; ++i) {
        command += i == 0 ? "" : " ";
        command +=
            argv[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
    spdlog::debug("{}: {}", version_text(), command);
}

/// Records how @p result_file was produced: the effective config (after
/// defaults and command-line overrides, with the sample count and interval
/// the search read from its input), the command line and the version.
template <typename TomlConfig>
void record_provenance(const std::filesystem::path& result_file,
                       TomlConfig cfg,
                       loki::SizeType nsamps,
                       double tsamp) {
    if (!std::filesystem::exists(result_file)) {
        return;
    }
    cfg.nsamps = nsamps;
    cfg.tsamp  = tsamp;
    loki::search::write_run_provenance(
        result_file, {
                         .config_effective = cfg.to_toml_string(),
                         .command_line     = run_command_line(),
                         .loki_version     = LOKI_VERSION,
                     });
}

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
    // Writes a unit-variance file. For --no-preprocess.
    const auto ts_v = series.get_ts_v();
    if (ts_v.empty() || !(ts_v.front() > 0.0F)) {
        throw std::runtime_error("simulated timeseries has no valid variance");
    }
    const auto inv_sigma =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(ts_v.front())));
    for (auto& sample : series.get_ts_e()) {
        sample *= inv_sigma;
    }
    std::ranges::fill(series.get_ts_v(), 1.0F);
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
            fmt::format("Invalid frequency range: [{:.3f}, {:.3f}] Hz",
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
        throw std::runtime_error(fmt::format(
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
/// folding inputs as [preprocessing] asks, on the search's backend.
SearchTimeSeries load_search_timeseries(const loki::search::FFATomlConfig& cfg,
                                        loki::Exec exec) {
    if (cfg.timeseries_path.empty()) {
        throw std::invalid_argument(
            "Input timeseries path must be specified via -i/--input or "
            "[input].timeseries in config");
    }
    const std::filesystem::path ts_path = cfg.timeseries_path;
    if (!std::filesystem::exists(ts_path)) {
        throw std::runtime_error(fmt::format(
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
    // GPU searches preprocess on their device; the CPU thread count serves
    // the options the GPU runs on the host (z-score, spectral zap).
    exec.nthreads   = nthreads;
    const auto rep  = ts.preprocess(options, exec);
    const double dt = ts.get_dt();
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
        throw std::invalid_argument(fmt::format(
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

struct SearchDims {
    loki::SizeType nsamps;
    double tsamp;
};

/// The sample count and interval a dry run plans with: those of the input
/// header when the config names an input (only the header is read), else
/// [input] nsamps / tsamp, else 2^21 samples of 64 us.
SearchDims dry_run_dims(const loki::search::FFATomlConfig& cfg,
                        NsampsPolicy nsamps_policy) {
    if (!cfg.timeseries_path.empty()) {
        const std::filesystem::path ts_path = cfg.timeseries_path;
        if (!std::filesystem::exists(ts_path)) {
            throw std::runtime_error(fmt::format(
                "Input timeseries does not exist: {}", ts_path.string()));
        }
        const auto info = loki::io::TimeSeries::read_info(ts_path);
        SPDLOG_INFO("Input header: nsamps = {}, dt = {:.6e} s, tobs = {:.2f} s",
                    info.nsamps, info.dt,
                    static_cast<double>(info.nsamps) * info.dt);
        return {.nsamps = power_of_two_nsamps(info.nsamps, nsamps_policy),
                .tsamp  = info.dt};
    }
    const SearchDims dims{.nsamps = cfg.nsamps.value_or(1U << 21U),
                          .tsamp  = cfg.tsamp.value_or(6.4e-5)};
    if (!cfg.nsamps.has_value() || !cfg.tsamp.has_value()) {
        SPDLOG_WARN("No input timeseries: planning with nsamps = {}, tsamp = "
                    "{:.6e} s (set -i/--input, or [input] nsamps and tsamp)",
                    dims.nsamps, dims.tsamp);
    }
    return dims;
}

/// Plans the FFA chunks of @p cfg (no sweep buffers) and prints them.
template <loki::SupportedFoldType FoldType>
void print_ffa_plan(const loki::search::FFASearchConfig& cfg, loki::Exec exec) {
    // On a GPU, the planner caps the limit by the device's free memory, as
    // the sweep does, so the plan printed is the one the sweep would run.
    const loki::algorithms::FFARegionPlanner<FoldType> planner(cfg, exec);
    const auto& cfgs   = planner.get_cfgs();
    const auto& chunks = planner.get_chunk_stats();
    const auto& con    = loki::cli::Console::get();
    std::cout
        << con.title("FFA plan:")
        << fmt::format(
               " {} chunks, nsamps {}, tobs {:.2f} s, sweep memory {:.3f} "
               "GB against a limit of {:.3f} GB\n",
               cfgs.size(), cfg.get_nsamps(), cfg.get_tobs(),
               planner.get_stats().get_freq_sweep_memory_usage(),
               planner.get_cfgs().empty()
                   ? cfg.get_max_process_memory_gb()
                   : planner.get_cfgs().front().get_max_process_memory_gb());
    std::cout << con.header(fmt::format("{:>6} {:>6} {:>23} {:>23} {:>9}",
                                        "chunk", "nbins", "nominal f [Hz]",
                                        "actual f [Hz]", "mem [GB]"))
              << "\n";
    for (SizeType i = 0; i < std::min(cfgs.size(), chunks.size()); ++i) {
        const auto& chunk = chunks[i];
        std::cout << fmt::format(
            "{:>6} {:>6} {:>10.3f} - {:<10.3f} {:>10.3f} - {:<10.3f} "
            "{:>9.3f}\n",
            i, cfgs[i].get_nbins(), chunk.nominal_f_start, chunk.nominal_f_end,
            chunk.actual_f_start, chunk.actual_f_end, chunk.total_memory_gb);
    }
}

int run_search_ffa(const loki::search::FFATomlConfig& toml_cfg,
                   std::string_view config_toml,
                   bool dry_run,
                   NsampsPolicy nsamps_policy) {
    check_frequency_range(toml_cfg);
    toml_cfg.candidates.validate();
    toml_cfg.observation.validate();
    const loki::Exec exec = search_exec(toml_cfg);

    if (dry_run) {
        const auto dims  = dry_run_dims(toml_cfg, nsamps_policy);
        auto preview_cfg = toml_cfg.to_search_config(dims.nsamps, dims.tsamp);
        if (toml_cfg.candidates.enabled &&
            exec.backend == loki::Backend::kCPU) {
            const auto remaining = preview_cfg.get_max_process_memory_gb() -
                                   toml_cfg.candidates.memory_gb;
            if (remaining <= 0) {
                throw std::invalid_argument(
                    "candidate host memory budget must be smaller than total "
                    "CPU search memory budget");
            }
            preview_cfg.set_max_process_memory_gb(remaining);
        }
        if (preview_cfg.get_use_fourier()) {
            print_ffa_plan<loki::ComplexType>(preview_cfg, exec);
        } else {
            print_ffa_plan<float>(preview_cfg, exec);
        }
        SPDLOG_INFO("Dry run complete: planner constructed successfully.");
        return 0;
    }

    // Create the GPU context while the timeseries loads; the sweep's first
    // device call would otherwise pay for it. A failure surfaces there.
    std::thread device_init;
    if (exec.backend != loki::Backend::kCPU) {
        device_init = std::thread([exec] {
            try {
                loki::initialize_device(exec);
            } catch (...) { // NOLINT(bugprone-empty-catch): reported later
            }
        });
    }
    auto loaded = load_search_timeseries(toml_cfg, exec);
    auto& ts    = loaded.ts;
    if (device_init.joinable()) {
        device_init.join();
    }
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
    auto observation = ts.get_observation();
    if (toml_cfg.observation.dm) {
        observation.dm = toml_cfg.observation.dm;
    }
    if (toml_cfg.observation.start_mjd) {
        observation.start_mjd = toml_cfg.observation.start_mjd;
    }
    if (toml_cfg.observation.barycentric) {
        observation.barycentric = toml_cfg.observation.barycentric;
    }
    loki::pipelines::FFAFreqSweep sweep(ffa_cfg, toml_cfg.candidates,
                                        /*show_progress=*/true, exec);
    sweep.execute(ts.get_ts_e().first(actual_nsamps),
                  ts.get_ts_v().first(actual_nsamps), outdir_path,
                  toml_cfg.prefix, config_toml, observation);

    const auto result_file =
        outdir_path / fmt::format("{}_ffa_results.h5", toml_cfg.prefix);
    record_provenance(result_file, toml_cfg, actual_nsamps, ts.get_dt());
    SPDLOG_INFO("FFA search complete. Inspecting results in {}",
                result_file.string());

    if (std::filesystem::exists(result_file)) {
        try {
            const HighFive::File h5(result_file.string(),
                                    HighFive::File::ReadOnly);
            std::vector<std::pair<std::string, std::string>> rows;
            const auto n_cands = h5.exist("snr") ? [&] {
                const auto dims = h5.getDataSet("snr").getDimensions();
                return dims.empty() ? 0UL : dims[0];
            }()
                                                 : 0UL;
            if (n_cands > 0) {
                const auto snr_dset     = h5.getDataSet("snr");
                float top_snr           = 0.0F;
                size_t top_idx          = 0;
                constexpr size_t kChunk = 65536;
                for (size_t offset = 0; offset < n_cands; offset += kChunk) {
                    const size_t count = std::min(kChunk, n_cands - offset);
                    std::vector<float> snrs(count);
                    snr_dset.select({offset}, {count}).read(snrs);
                    const auto max_it = std::ranges::max_element(snrs);
                    if (max_it != snrs.end() && *max_it >= top_snr) {
                        top_snr = *max_it;
                        top_idx = offset + static_cast<size_t>(std::distance(
                                               snrs.begin(), max_it));
                    }
                }

                // Root datasets hold the selected candidates; every passing
                // trial is in /detections when retained. With candidate
                // processing disabled they hold every trial.
                if (h5.exist("detections/snr")) {
                    const auto n_detections =
                        h5.getDataSet("detections/snr").getDimensions();
                    rows.emplace_back("Detections above snr_min",
                                      std::to_string(n_detections.empty()
                                                         ? 0UL
                                                         : n_detections[0]));
                }
                rows.emplace_back(h5.exist("candidates")
                                      ? "Selected candidates"
                                      : "Detections above snr_min",
                                  std::to_string(n_cands));
                rows.emplace_back(
                    "Top candidate SNR",
                    fmt::format("{:.2f} (candidate #{})", top_snr, top_idx));

                if (h5.exist("param_sets")) {
                    const auto p_dset = h5.getDataSet("param_sets");
                    auto p_dims       = p_dset.getDimensions();
                    if (p_dims.size() == 2 && p_dims[0] == n_cands &&
                        p_dims[1] > 0 && h5.hasAttribute("param_names")) {
                        std::vector<double> top_params(p_dims[1]);
                        p_dset.select({top_idx, 0}, {1, p_dims[1]})
                            .read_raw(top_params.data());
                        std::vector<std::string> p_names;
                        h5.getAttribute("param_names").read(p_names);
                        std::string p_str;
                        for (size_t pi = 0;
                             pi < std::min(top_params.size(), p_names.size());
                             ++pi) {
                            if (pi > 0) {
                                p_str += ", ";
                            }
                            p_str += fmt::format("{}={:.6f}", p_names[pi],
                                                 top_params[pi]);
                        }
                        rows.emplace_back("Top candidate parameters",
                                          fmt::format("[{}]", p_str));
                    }
                }
            } else {
                rows.emplace_back("Total candidates detected",
                                  fmt::format("0 above S/N threshold {:.2f}",
                                              toml_cfg.snr_min));
            }
            loki::cli::Console::get().summary("Search Summary", rows);
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

/// A count as a power of two with one decimal ("2^20.0"), as the library logs
/// it.
std::string format_pow2(loki::SizeType n) {
    if (n == 0) {
        return "0";
    }
    return fmt::format("2^{:.1f}", std::log2(static_cast<double>(n)));
}

/// Writes the chunk table, the groups of equal nbins and the memory peak of an
/// EP plan to stdout.
template <typename FoldType>
void print_ep_plan(const loki::algorithms::EPRegionPlanner<FoldType>& planner,
                   SizeType workers,
                   const std::optional<std::filesystem::path>& cache_path) {
    const auto& chunks = planner.get_chunk_cfgs();
    const auto& stats  = planner.get_stats();
    const auto& con    = loki::cli::Console::get();
    if (chunks.empty()) {
        std::cout << con.title("EP plan:")
                  << " no chunks (the frequency range is empty)\n";
        return;
    }

    SizeType nbins_lo = chunks.front().cfg.get_nbins();
    SizeType nbins_hi = nbins_lo;
    for (const auto& chunk : chunks) {
        nbins_lo = std::min(nbins_lo, chunk.cfg.get_nbins());
        nbins_hi = std::max(nbins_hi, chunk.cfg.get_nbins());
    }
    std::cout
        << con.title("EP plan:")
        << fmt::format(
               " {} chunks, nbins {}..{}, peak {:.3f} GB against a limit of "
               "{:.3f} GB\n",
               chunks.size(), nbins_lo, nbins_hi, stats.get_max_memory_gb(),
               stats.get_memory_limit_gb());

    std::cout << con.header(fmt::format(
                     "{:>6} {:>6} {:>23} {:>23} {:>9} {:>9} {:>10} {:>6} "
                     "{:>9} {:>7}",
                     "chunk", "nbins", "nominal f [Hz]", "actual f [Hz]",
                     "ncoords", "max_sugg", "branch_max", "nseg", "mem [GB]",
                     "Pd"))
              << "\n";
    for (SizeType i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        std::cout << fmt::format(
            "{:>6} {:>6} {:>10.3f} - {:<10.3f} {:>10.3f} - {:<10.3f} {:>9} "
            "{:>9} {:>10} {:>6} {:>9.3f} {:>7.4f}\n",
            i, chunk.cfg.get_nbins(), chunk.nominal_f_start,
            chunk.nominal_f_end, chunk.actual_f_start, chunk.actual_f_end,
            chunk.ncoords, format_pow2(chunk.max_sugg), chunk.branch_max,
            chunk.nsegments, chunk.chunk_memory_gb, chunk.pd);
        std::cout << con.muted(fmt::format(
                         "       predicted peak {}, live ring {}, batch {}",
                         format_pow2(chunk.predicted_peak),
                         format_pow2(chunk.predicted_live), chunk.batch_size))
                  << "\n";
    }

    std::cout << con.title("Groups of equal nbins")
              << ", each with its own workspaces:\n";
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
        std::cout << fmt::format(
            "  chunks {}..{}: nbins {}, max_sugg {}, branch_max {}, {} "
            "worker(s)\n",
            begin, end - 1, nbins, format_pow2(max_sugg), branch_max, workers);
        begin = end;
    }
    std::cout << con.title("Plan cache:")
              << fmt::format(" {}\n",
                             cache_path ? cache_path->string() : "none");
}

/// Candidate processing runs after merge, while the shared engine buffers
/// remain resident. Use exactly the facade's reservation for cache/dry runs.
loki::search::PulsarSearchConfig
ep_candidate_planning_config(const loki::search::EPTomlConfig& config,
                             const loki::search::PulsarSearchConfig& input,
                             loki::Exec exec) {
    auto cfg = input;
    if (config.candidates.enabled && config.poly_basis == "taylor" &&
        cfg.get_nparams() <= 4) {
        config.candidates.validate();
        if (exec.backend == loki::Backend::kCPU) {
            const auto remaining =
                cfg.get_max_process_memory_gb() - config.candidates.memory_gb;
            if (remaining <= 0) {
                throw std::invalid_argument(
                    "candidate host memory budget must be smaller than total "
                    "CPU search memory budget");
            }
            cfg.set_max_process_memory_gb(remaining);
        }
    }
    return cfg;
}

/// Plans the EP chunks of @p cfg (or loads them from @p cache_path, writing
/// the cache when it does not exist) and prints them. No sweep is built.
void plan_and_print_ep(const loki::search::EPTomlConfig& toml_cfg,
                       const loki::search::PulsarSearchConfig& input_cfg,
                       const std::optional<std::filesystem::path>& cache_path,
                       loki::Exec exec) {
    const auto cfg = ep_candidate_planning_config(toml_cfg, input_cfg, exec);
    // The same worker count the sweep uses, so the cache matches it. The GPU
    // sweep prunes its runs on one worker.
    const SizeType n_workers = loki::algorithms::ep_sweep_n_workers(
        cfg.get_nthreads(), toml_cfg.n_runs, toml_cfg.ref_segs);
    const SizeType workers =
        exec.backend == loki::Backend::kCPU ? n_workers : SizeType{1};
    if (cfg.get_use_fourier()) {
        const loki::algorithms::EPRegionPlanner<loki::ComplexType> planner(
            cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
            cache_path, {}, n_workers, toml_cfg.n_runs, toml_cfg.ref_segs,
            exec);
        print_ep_plan(planner, workers, cache_path);
    } else {
        const loki::algorithms::EPRegionPlanner<float> planner(
            cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
            cache_path, {}, n_workers, toml_cfg.n_runs, toml_cfg.ref_segs,
            exec);
        print_ep_plan(planner, workers, cache_path);
    }
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
        fmt::format("{}_ep_plan.h5", toml_cfg.prefix));

    SPDLOG_INFO("Planning EP search (no data read): f=[{:.3f}, {:.3f}] Hz, "
                "nsamps={}, nbins={}, backend={}",
                cfg.get_f_min(), cfg.get_f_max(), cfg.get_nsamps(),
                cfg.get_nbins(), loki::to_string(exec.backend));
    plan_and_print_ep(toml_cfg, cfg, cache_path, exec);
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
        if (h5.hasAttribute("n_candidates")) {
            std::uint64_t detections = 0;
            std::uint64_t selected   = 0;
            h5.getAttribute("n_detections").read(detections);
            h5.getAttribute("n_candidates").read(selected);
            SPDLOG_INFO("Final EP detections above candidate S/N : {}",
                        detections);
            SPDLOG_INFO("Selected EP candidates                  : {}",
                        selected);
        }
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

        std::vector<std::pair<std::string, std::string>> rows;
        rows.emplace_back("Runs pruned",
                          fmt::format("{} ({} leaves)", n_runs, n_leaves));
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
                p_str += fmt::format("{}={:.6f}", param_names[pi],
                                     leaf_values[pi * 2]);
            }
            rows.emplace_back("Top score",
                              fmt::format("{:.2f} (chunk {}, run {}, leaf {})",
                                          top_score, top_chunk, top_run,
                                          top_leaf));
            rows.emplace_back("Top candidate parameters",
                              fmt::format("[{}]", p_str));
        } else {
            rows.emplace_back("Top candidate",
                              "none: the pruning wrote no leaves");
        }
        loki::cli::Console::get().summary("Search Summary", rows);
        SPDLOG_INFO("Saved HDF5 results to: {}", result_file.string());
    } catch (const std::exception& e) {
        SPDLOG_WARN("HDF5 result summary inspection failed: {}", e.what());
    }
}

template <loki::SupportedFoldType FoldType>
std::optional<std::filesystem::path>
prepare_ep_plan_cache(const loki::search::EPTomlConfig& toml_cfg,
                      const loki::search::PulsarSearchConfig& cfg,
                      loki::Exec exec) {
    if (toml_cfg.plan_cache || !toml_cfg.use_plan_cache) {
        return toml_cfg.plan_cache;
    }
    const auto path    = std::filesystem::path(toml_cfg.outdir) /
                         fmt::format("{}_ep_plan.h5", toml_cfg.prefix);
    const auto workers = loki::algorithms::ep_sweep_n_workers(
        cfg.get_nthreads(), toml_cfg.n_runs, toml_cfg.ref_segs);
    // An implicit cache that no longer matches the configuration is replanned;
    // planning errors themselves propagate.
    if (std::filesystem::exists(path)) {
        try {
            const loki::algorithms::EPRegionPlanner<FoldType> cached(
                cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
                path, {}, workers, toml_cfg.n_runs, toml_cfg.ref_segs, exec);
            return path;
        } catch (const std::exception& error) {
            SPDLOG_WARN("Automatic EP plan cache '{}' is stale, replanning: {}",
                        path.string(), error.what());
        }
    }
    const loki::algorithms::EPRegionPlanner<FoldType> fresh(
        cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
        std::nullopt, {}, workers, toml_cfg.n_runs, toml_cfg.ref_segs, exec);
    fresh.save_cache(path);
    return path;
}

std::optional<std::filesystem::path>
prepare_ep_plan_cache(const loki::search::EPTomlConfig& toml_cfg,
                      const loki::search::PulsarSearchConfig& input_cfg,
                      loki::Exec exec) {
    const auto cfg = ep_candidate_planning_config(toml_cfg, input_cfg, exec);
    return cfg.get_use_fourier()
               ? prepare_ep_plan_cache<loki::ComplexType>(toml_cfg, cfg, exec)
               : prepare_ep_plan_cache<float>(toml_cfg, cfg, exec);
}

/// `search ep`: runs the EP sweep over the timeseries and prints a summary.
int run_search_ep(const loki::search::EPTomlConfig& toml_cfg,
                  bool dry_run,
                  NsampsPolicy nsamps_policy) {
    check_frequency_range(toml_cfg);
    const loki::Exec exec = search_exec(toml_cfg);
    if (dry_run) {
        const auto dims = dry_run_dims(toml_cfg, nsamps_policy);
        const auto preview_cfg =
            toml_cfg.to_ep_search_config(dims.nsamps, dims.tsamp);
        // The plan cache is prepared as the search would prepare it, so a
        // search after the dry run reuses it.
        plan_and_print_ep(toml_cfg, preview_cfg,
                          prepare_ep_plan_cache(toml_cfg, preview_cfg, exec),
                          exec);
        SPDLOG_INFO("Dry run complete: planner constructed successfully.");
        return 0;
    }

    auto loaded = load_search_timeseries(toml_cfg, exec);
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

    const auto cache = prepare_ep_plan_cache(toml_cfg, ep_cfg, exec);
    const bool common_candidates = toml_cfg.candidates.enabled &&
                                   toml_cfg.poly_basis == "taylor" &&
                                   ep_cfg.get_nparams() <= 4;
    auto sweep                   = common_candidates
                                       ? loki::pipelines::EPFreqSweep(
                                             ep_cfg, toml_cfg.candidates, true, toml_cfg.min_pd,
                                             toml_cfg.poly_basis, toml_cfg.ref_ducy, {}, cache,
                                             toml_cfg.n_runs, toml_cfg.ref_segs, exec)
                                       : loki::pipelines::EPFreqSweep(
                                             ep_cfg, true, toml_cfg.min_pd, toml_cfg.poly_basis,
                                             toml_cfg.ref_ducy, {}, cache, toml_cfg.n_runs,
                                             toml_cfg.ref_segs, exec);
    if (toml_cfg.candidates.enabled && !common_candidates) {
        SPDLOG_WARN("Common EP candidates currently support polynomial Taylor "
                    "only; writing run diagnostics");
    }
    auto observation = ts.get_observation();
    if (toml_cfg.observation.dm) {
        observation.dm = toml_cfg.observation.dm;
    }
    if (toml_cfg.observation.start_mjd) {
        observation.start_mjd = toml_cfg.observation.start_mjd;
    }
    if (toml_cfg.observation.barycentric) {
        observation.barycentric = toml_cfg.observation.barycentric;
    }
    sweep.execute(ts.get_ts_e().first(actual_nsamps),
                  ts.get_ts_v().first(actual_nsamps), outdir_path,
                  toml_cfg.prefix, observation);

    const auto result_file =
        outdir_path / fmt::format("{}_ep_results.h5", toml_cfg.prefix);
    record_provenance(result_file, toml_cfg, actual_nsamps, ts.get_dt());
    print_ep_summary(result_file);
    return 0;
}

// ---------------------------------------------------------------------------
// loki inspect
// ---------------------------------------------------------------------------

/// A parameter column label with its unit.
std::string param_label(const std::string& name) {
    if (name == "freq") {
        return "freq (Hz)";
    }
    if (name == "accel") {
        return "accel (m/s^2)";
    }
    if (name == "jerk") {
        return "jerk (m/s^3)";
    }
    return name;
}

std::string_view param_unit(const std::string& name) {
    if (name == "freq") {
        return " Hz";
    }
    if (name == "accel") {
        return " m/s^2";
    }
    if (name == "jerk") {
        return " m/s^3";
    }
    return "";
}

std::string layout_text(const loki::io::CandidateFileSummary& s) {
    if (s.layout == "common") {
        return fmt::format("candidate schema {}", s.schema_version);
    }
    if (s.layout == "legacy_ffa") {
        return "legacy flat FFA results";
    }
    return "raw EP runs";
}

std::string format_param(const std::string& name, double value) {
    return name == "freq" ? fmt::format("{:.6f}", value)
                          : fmt::format("{:.6g}", value);
}

/// A finite double, checked on its bits (-ffast-math, docs/architecture.md).
bool is_finite_bits(double value) {
    constexpr std::uint64_t kExponent = 0x7FF0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value) & kExponent) != kExponent;
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char ch : text) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out += fmt::format("\\u{:04x}", static_cast<int>(ch));
            } else {
                out += ch;
            }
        }
    }
    return out + "\"";
}

std::string json_number(double value) {
    return is_finite_bits(value) ? fmt::format("{}", value) : "null";
}

// A float in its shortest form (29.77668, not 29.776679992675781).
std::string json_number(float value) {
    return is_finite_bits(static_cast<double>(value)) ? fmt::format("{}", value)
                                                      : "null";
}

template <typename T> std::string json_optional(const std::optional<T>& value) {
    if (!value) {
        return "null";
    }
    if constexpr (std::is_same_v<T, std::string>) {
        return json_string(*value);
    } else {
        return json_number(*value);
    }
}

std::string inspect_json(const loki::io::CandidateFileSummary& s,
                         bool with_profile) {
    std::string out  = "{\n";
    const auto field = [&out](std::string_view key, const std::string& value,
                              bool last = false) {
        out += fmt::format("  {}: {}{}\n", json_string(key), value,
                           last ? "" : ",");
    };
    field("file", json_string(s.path.string()));
    field("layout", json_string(s.layout));
    field("algorithm", json_string(s.algorithm));
    field("schema_version", json_string(s.schema_version));
    field("complete", s.complete ? "true" : "false");
    field("producer_version", json_string(s.producer_version));
    field("writer_version", json_string(s.writer_version));
    field("backend", json_optional(s.backend));
    field("source", json_string(s.source_name));
    field("input_path", json_string(s.input_path));
    field("nsamps", fmt::format("{}", s.nsamps));
    field("tsamp", json_number(s.tsamp));
    field("tobs", json_number(s.tobs));
    field("dm", json_optional(s.dm));
    field("snr_min", json_optional(s.snr_min));
    std::string names;
    for (const auto& name : s.param_names) {
        names += (names.empty() ? "" : ", ") + json_string(name);
    }
    field("param_names", "[" + names + "]");
    std::string ranges;
    for (const auto& range : s.ranges) {
        ranges +=
            fmt::format("{}\n    {{\"name\": {}, \"min\": {}, \"max\": {}}}",
                        ranges.empty() ? "" : ",", json_string(range.name),
                        json_number(range.min), json_number(range.max));
    }
    field("ranges", ranges.empty() ? "[]" : "[" + ranges + "\n  ]");
    field("loki_version", json_optional(s.loki_version));
    field("command_line", json_optional(s.command_line));
    field("config_effective", json_optional(s.config_effective));
    field("config_toml", json_optional(s.config_toml));
    field("table", json_string(s.table));
    field("n_rows", fmt::format("{}", s.n_rows));
    field("n_detections", s.n_detections ? fmt::format("{}", *s.n_detections)
                                         : std::string("null"));
    std::string rows;
    for (std::size_t i = 0; i < s.top.size(); ++i) {
        const auto& row = s.top[i];
        std::string params;
        for (std::size_t p = 0; p < row.params.size(); ++p) {
            params += fmt::format("{}{}: {}", p == 0 ? "" : ", ",
                                  json_string(s.param_names[p]),
                                  json_number(row.params[p]));
        }
        rows += fmt::format(
            "{}\n    {{\"rank\": {}, \"row\": {}, \"snr\": {}, \"params\": "
            "{{{}}}, \"width\": {}, \"nbins\": {}, \"source\": {}}}",
            i == 0 ? "" : ",", i + 1, row.row, json_number(row.snr), params,
            row.width, row.nbins, json_string(row.source));
    }
    field("top_candidates", rows.empty() ? "[]" : "[" + rows + "\n  ]",
          !with_profile);
    if (with_profile) {
        std::string profile = "null";
        if (s.top_profile) {
            profile.clear();
            for (const float value : *s.top_profile) {
                profile += (profile.empty() ? "" : ", ") + json_number(value);
            }
            profile = "[" + profile + "]";
        }
        field("profile", profile, true);
    }
    return out + "}\n";
}

/// The profile as a column plot of @p height rows of Unicode eighth blocks,
/// at most @p max_width columns (bins averaged to fit).
std::vector<std::string> render_profile(std::span<const float> profile,
                                        std::size_t height,
                                        std::size_t max_width) {
    if (profile.empty() || height == 0) {
        return {};
    }
    const auto ncols = std::min(profile.size(), max_width);
    std::vector<double> cols(ncols, 0.0);
    for (std::size_t c = 0; c < ncols; ++c) {
        const auto lo = c * profile.size() / ncols;
        const auto hi = std::max(lo + 1, (c + 1) * profile.size() / ncols);
        for (auto i = lo; i < hi; ++i) {
            cols[c] += profile[i];
        }
        cols[c] /= static_cast<double>(hi - lo);
    }
    const auto [lo_it, hi_it] = std::ranges::minmax_element(cols);
    const double lo           = *lo_it;
    const double span         = *hi_it - lo;
    static constexpr std::array<std::string_view, 9> kBlocks = {
        " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█",
    };
    constexpr std::size_t kEighths = 8;
    std::vector<std::string> lines(height);
    for (std::size_t c = 0; c < ncols; ++c) {
        // Height in eighths of a row; a flat profile draws one low line.
        const auto level = span > 0.0
                               ? static_cast<std::size_t>(std::lround(
                                     (cols[c] - lo) / span *
                                     static_cast<double>(height * kEighths)))
                               : std::size_t{1};
        for (std::size_t r = 0; r < height; ++r) {
            const auto base   = (height - 1 - r) * kEighths;
            const auto filled = level > base ? level - base : 0;
            lines[r] += kBlocks.at(std::min(filled, kEighths));
        }
    }
    for (auto& line : lines) {
        line.erase(line.find_last_not_of(' ') + 1);
    }
    return lines;
}

std::string inspect_text(const loki::io::CandidateFileSummary& s,
                         bool with_profile) {
    const auto& con = loki::cli::Console::get();
    std::vector<std::pair<std::string, std::string>> meta;
    meta.emplace_back("File", s.path.string());
    const auto layout = layout_text(s);
    meta.emplace_back("Search", fmt::format("{} ({}{})", s.algorithm, layout,
                                            s.complete ? "" : ", INCOMPLETE"));
    // The provenance version, else the candidate writer's producer version.
    const auto& loki_version =
        s.loki_version ? *s.loki_version : s.producer_version;
    std::string version =
        loki_version.empty() ? std::string() : "loki " + loki_version;
    if (!s.writer_version.empty()) {
        version += fmt::format("{}writer {}{}", version.empty() ? "" : " (",
                               s.writer_version, version.empty() ? "" : ")");
    }
    if (version.empty()) {
        version = "not recorded";
    }
    meta.emplace_back("Version", version);
    meta.emplace_back("Backend", s.backend.value_or("not recorded"));
    if (s.command_line && !s.command_line->empty()) {
        meta.emplace_back("Command", *s.command_line);
    }
    if (!s.input_path.empty()) {
        meta.emplace_back("Input", s.input_path);
    }
    if (!s.source_name.empty()) {
        meta.emplace_back("Source", s.source_name);
    }
    meta.emplace_back("nsamps", fmt::format("{}", s.nsamps));
    meta.emplace_back("tsamp", fmt::format("{:.6g} s", s.tsamp));
    meta.emplace_back("tobs", fmt::format("{:.6g} s", s.tobs));
    if (s.dm) {
        meta.emplace_back("DM", fmt::format("{:.4f}", *s.dm));
    }
    for (const auto& range : s.ranges) {
        meta.emplace_back(fmt::format("Range {}", range.name),
                          fmt::format("[{}, {}]{}",
                                      format_param(range.name, range.min),
                                      format_param(range.name, range.max),
                                      param_unit(range.name)));
    }
    if (s.snr_min) {
        meta.emplace_back("S/N min", fmt::format("{:.2f}", *s.snr_min));
    }
    std::string rows_text = fmt::format("{}", s.n_rows);
    if (s.n_detections) {
        rows_text += fmt::format(" (from {} detections)", *s.n_detections);
    }
    meta.emplace_back(s.layout == "ep_runs" ? "Leaves" : "Candidates",
                      rows_text);

    std::size_t key_width = 0;
    for (const auto& [key, value] : meta) {
        key_width = std::max(key_width, key.size());
    }
    std::string out = con.heading("Run metadata") + "\n";
    for (const auto& [key, value] : meta) {
        out += fmt::format("  {}  {}\n",
                           con.muted(fmt::format("{:<{}}", key, key_width)),
                           con.strong(value));
    }

    // Top candidates: padded as plain text first, so styles keep alignment.
    out += "\n" +
           con.heading(fmt::format("Top candidates ({} of {})", s.top.size(),
                                   s.n_rows)) +
           "\n";
    if (s.top.empty()) {
        out += "  none\n";
    } else {
        std::vector<std::string> header{"#"};
        for (const auto& name : s.param_names) {
            header.push_back(param_label(name));
        }
        header.insert(header.end(), {"S/N", "width", "nbins", "source"});
        std::vector<std::vector<std::string>> cells;
        for (std::size_t i = 0; i < s.top.size(); ++i) {
            const auto& row = s.top[i];
            std::vector<std::string> line{fmt::format("{}", i + 1)};
            for (std::size_t p = 0; p < row.params.size(); ++p) {
                line.push_back(format_param(s.param_names[p], row.params[p]));
            }
            line.push_back(fmt::format("{:.2f}", row.snr));
            line.push_back(row.width == 0 ? "-" : fmt::format("{}", row.width));
            line.push_back(row.nbins == 0 ? "-" : fmt::format("{}", row.nbins));
            line.push_back(row.source.empty() ? "-" : row.source);
            cells.push_back(std::move(line));
        }
        std::vector<std::size_t> widths(header.size());
        for (std::size_t c = 0; c < header.size(); ++c) {
            widths[c] = header[c].size();
            for (const auto& line : cells) {
                widths[c] = std::max(widths[c], line[c].size());
            }
        }
        const auto snr_col  = header.size() - 4;
        const auto last_col = header.size() - 1;
        const auto pad      = [&](std::size_t c, const std::string& text) {
            // Numbers align right; the source, last, is not padded.
            return c == last_col ? text
                                 : fmt::format("{:>{}}", text, widths[c]);
        };
        out += " ";
        for (std::size_t c = 0; c < header.size(); ++c) {
            out += " " + con.header(pad(c, header[c]));
        }
        out += "\n";
        for (const auto& line : cells) {
            out += " ";
            for (std::size_t c = 0; c < line.size(); ++c) {
                const auto text = pad(c, line[c]);
                if (c == snr_col) {
                    out += " " + con.score(text);
                } else if (c == 0 || c == last_col) {
                    out += " " + con.muted(text);
                } else {
                    out += " " + text;
                }
            }
            out += "\n";
        }
    }

    if (with_profile) {
        out += "\n" + con.heading("Folded profile (top candidate)") + "\n";
        if (!s.top_profile) {
            out += "  This file stores no folded profiles.\n";
        } else {
            constexpr std::size_t kHeight = 8;
            constexpr std::size_t kWidth  = 64;
            for (const auto& line :
                 render_profile(*s.top_profile, kHeight, kWidth)) {
                out += "  " + con.accent(line) + "\n";
            }
            out += "  " +
                   con.muted(fmt::format("{} bins, phase 0 to 1",
                                         s.top_profile->size())) +
                   "\n";
        }
    }

    // The effective configuration (after defaults and overrides) when the
    // file records it, else the configuration file as written.
    const bool effective = s.config_effective && !s.config_effective->empty();
    const auto& config   = effective ? s.config_effective : s.config_toml;
    if (config && !config->empty()) {
        out += "\n" +
               con.heading(effective ? "Configuration (config_effective)"
                                     : "Configuration (config_toml)") +
               "\n";
        std::string_view text = *config;
        while (!text.empty()) {
            const auto nl   = text.find('\n');
            const auto line = text.substr(0, nl);
            out += line.empty() ? "\n" : "  " + con.muted(line) + "\n";
            text = nl == std::string_view::npos ? std::string_view{}
                                                : text.substr(nl + 1);
        }
    }
    return out;
}

int run_inspect(const std::filesystem::path& path,
                std::size_t top_n,
                bool json,
                bool with_profile) {
    const auto summary = loki::io::read_candidate_file_summary(path, top_n);
    loki::cli::Console::print(json ? inspect_json(summary, with_profile)
                                   : inspect_text(summary, with_profile));
    return 0;
}

} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape,misc-const-correctness): setup errors abort; main signature is fixed
int main(int argc, char** argv) {
    // The help text and errors before the flags are parsed use it too.
    loki::cli::Console::get().configure(/*verbose=*/false);
    CLI::App app{"Loki - Fast Pulsar Search Pipeline"};
    app.formatter(std::make_shared<loki::cli::HelpFormatter>());
    app.require_subcommand(1);
    app.set_version_flag("--version", [] { return version_text(); });
    app.footer("Environment: NO_COLOR (no colour), CLICOLOR_FORCE (colour "
               "when piped), LOKI_PROGRESS (rich, plain or off).");

    // Global output options: accepted before or after the subcommand.
    OutputOptions output;
    auto* grp_logging = app.add_option_group("Output & Logging");
    auto* opt_verbose = grp_logging->add_flag(
        "-v,--verbose", output.verbose, "More detail: -v debug, -vv trace");
    grp_logging
        ->add_flag("-q,--quiet", output.quiet,
                   "Less output: -q warnings (no progress bars), -qq errors "
                   "only")
        ->excludes(opt_verbose);
    grp_logging->add_option("--log-file", output.log_file,
                            "Also write the log to this file (plain text; "
                            "info and up, more with -v)");

    // Pre-inspect configuration argument if passed to populate base config
    // before CLI option registration
    const auto cfg_arg = find_config_arg(argc, argv);

    // =========================================================================
    // Subcommand: simulate
    // =========================================================================
    auto* simulate =
        app.add_subcommand("simulate", "Write a simulated pulse timeseries");

    loki::simulation::SimulateTomlConfig sim_cfg;
    if (cfg_arg.has_value() && has_subcommand(argc, argv, "simulate")) {
        try {
            sim_cfg = loki::simulation::SimulateTomlConfig::load(*cfg_arg);
        } catch (const std::exception& ex) {
            loki::cli::Console::get().error(ex.what());
            return 1;
        }
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
            loki::cli::Console::get().error(ex.what());
            return 1;
        }
    }

    loki::search::EPTomlConfig ep_cfg;
    if (cfg_arg.has_value() && has_subcommand(argc, argv, "ep")) {
        try {
            ep_cfg = loki::search::EPTomlConfig::load(*cfg_arg);
        } catch (const std::exception& ex) {
            loki::cli::Console::get().error(ex.what());
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
    grp_io->add_option("--dm", ffa_cfg.observation.dm,
                       "Override single-trial dispersion measure (pc cm^-3)");
    grp_io->add_option("--start-mjd", ffa_cfg.observation.start_mjd,
                       "Override MJD of the first input sample");
    grp_io->add_option(
        "--barycentric", ffa_cfg.observation.barycentric,
        "Override timing frame: true (barycentric) or false (topocentric)");
    auto* grp_candidates = ffa->add_option_group("Candidate processing");
    grp_candidates->add_flag(
        "--candidates,!--no-candidates", ffa_cfg.candidates.enabled,
        "Enable/disable candidate processing; disabled writes the flat "
        "detection file of earlier versions (default: enabled)");
    grp_candidates->add_flag(
        "--cluster,!--no-cluster", ffa_cfg.candidates.cluster,
        "Enable/disable phase-track clustering (default: enabled)");
    grp_candidates
        ->add_option("--cluster-grouping", ffa_cfg.candidates.grouping,
                     "Grouping: transitive (default) or anchored")
        ->check(CLI::IsMember({"transitive", "anchored"}));
    grp_candidates->add_option(
        "--cluster-radius", ffa_cfg.candidates.radius,
        "Clustering radius in narrower pulse widths (default: 1)");
    grp_candidates->add_option("--cluster-phase-radius",
                               ffa_cfg.candidates.phase_radius,
                               "Minimum suppression radius in phase cycles "
                               "(default: 0.05; 0 disables floor)");
    grp_candidates->add_flag(
        "--retain-detections,!--no-retain-detections",
        ffa_cfg.candidates.retain_detections,
        "Keep raw detections and membership (default: enabled)");
    grp_candidates->add_flag("--flag-harmonics,!--no-flag-harmonics",
                             ffa_cfg.candidates.flag_harmonics,
                             "Flag harmonic relationships between clusters");
    grp_candidates->add_flag(
        "--remove-harmonics,!--no-remove-harmonics",
        ffa_cfg.candidates.remove_harmonics,
        "Exclude flagged harmonics; requires --flag-harmonics");
    grp_candidates->add_option(
        "--harmonic-denominator-max",
        ffa_cfg.candidates.harmonic_denominator_max,
        "Maximum rational harmonic denominator (default: 100)");
    grp_candidates->add_option("--harmonic-phase-distance",
                               ffa_cfg.candidates.harmonic_phase_distance,
                               "Harmonic track tolerance in pulse widths");
    grp_candidates->add_option("--harmonic-snr-distance",
                               ffa_cfg.candidates.harmonic_snr_distance,
                               "Harmonic S/N agreement tolerance");
    grp_candidates->add_option("--max-candidates",
                               ffa_cfg.candidates.max_candidates,
                               "Maximum folding candidates; zero is unlimited");
    grp_candidates->add_option(
        "--candidate-memory-gb", ffa_cfg.candidates.memory_gb,
        "Additional host budget reserved for candidate processing");
    grp_candidates->add_option("--candidate-tempdir",
                               ffa_cfg.candidates.temporary_directory,
                               "Directory for bounded external sorting");

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
    grp_search->add_option(
        "--snr-min", ffa_cfg.snr_min,
        "Minimum candidate SNR for final HDF5 (default: 7.0)");
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
        "Memory cap in GB of 2^30 bytes (as Slurm's --mem=16G): total "
        "process RSS on CPU, device memory on CUDA");
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
    bool ep_no_plan_cache            = false;
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
    ep_grp_io->add_flag("--no-plan-cache", ep_no_plan_cache,
                        "Disable automatic EP plan caching");
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

    ep->add_option("--dm", ep_cfg.observation.dm,
                   "Override single-trial dispersion measure");
    ep->add_option("--start-mjd", ep_cfg.observation.start_mjd,
                   "Override MJD of the first sample");
    ep->add_option("--barycentric", ep_cfg.observation.barycentric,
                   "Override barycentric/topocentric timing frame");
    auto* ep_grp_candidates = ep->add_option_group("Candidate processing");
    ep_grp_candidates->add_flag(
        "--candidates,!--no-candidates", ep_cfg.candidates.enabled,
        "Enable/disable candidate processing; disabled writes only the run "
        "diagnostics of earlier versions (default: enabled)");
    ep_grp_candidates->add_flag(
        "--cluster,!--no-cluster", ep_cfg.candidates.cluster,
        "Enable/disable phase-track clustering (default: enabled)");
    ep_grp_candidates
        ->add_option("--cluster-grouping", ep_cfg.candidates.grouping,
                     "Grouping: transitive (default) or anchored")
        ->check(CLI::IsMember({"transitive", "anchored"}));
    ep_grp_candidates->add_option(
        "--cluster-radius", ep_cfg.candidates.radius,
        "Clustering radius in narrower pulse widths (default: 1)");
    ep_grp_candidates->add_option("--cluster-phase-radius",
                                  ep_cfg.candidates.phase_radius,
                                  "Minimum suppression radius in phase cycles "
                                  "(default: 0.05; 0 disables floor)");
    ep_grp_candidates->add_flag(
        "--retain-detections,!--no-retain-detections",
        ep_cfg.candidates.retain_detections,
        "Keep raw detections and membership (default: enabled)");
    ep_grp_candidates->add_flag("--flag-harmonics,!--no-flag-harmonics",
                                ep_cfg.candidates.flag_harmonics,
                                "Flag harmonic relationships between clusters");
    ep_grp_candidates->add_flag(
        "--remove-harmonics,!--no-remove-harmonics",
        ep_cfg.candidates.remove_harmonics,
        "Exclude flagged harmonics; requires --flag-harmonics");
    ep_grp_candidates->add_option(
        "--harmonic-denominator-max",
        ep_cfg.candidates.harmonic_denominator_max,
        "Maximum rational harmonic denominator (default: 100)");
    ep_grp_candidates->add_option("--harmonic-phase-distance",
                                  ep_cfg.candidates.harmonic_phase_distance,
                                  "Harmonic track tolerance in pulse widths");
    ep_grp_candidates->add_option("--harmonic-snr-distance",
                                  ep_cfg.candidates.harmonic_snr_distance,
                                  "Harmonic S/N agreement tolerance");
    ep_grp_candidates->add_option(
        "--max-candidates", ep_cfg.candidates.max_candidates,
        "Maximum folding candidates; zero is unlimited");
    ep_grp_candidates->add_option(
        "--candidate-memory-gb", ep_cfg.candidates.memory_gb,
        "Additional host budget reserved for candidate processing");
    ep_grp_candidates->add_option("--candidate-tempdir",
                                  ep_cfg.candidates.temporary_directory,
                                  "Directory for bounded external sorting");

    auto* ep_grp_search =
        ep->add_option_group("Detection & Folding Parameters");
    ep_grp_search->add_option("--nbins", ep_cfg.nbins,
                              "Phase bin count (default: 64)");
    ep_grp_search->add_option("--eta", ep_cfg.eta,
                              "Tolerance parameter (default: 1.0)");
    ep_grp_search->add_option("--snr-min", ep_cfg.snr_min,
                              "Minimum final candidate S/N, independent of the "
                              "pruning target (default: 7.0)");
    ep_grp_search->add_option("--ducy-max", ep_cfg.ducy_max,
                              "Maximum duty cycle to evaluate (default: 0.2)");
    ep_grp_search->add_option("--wtsp", ep_cfg.wtsp,
                              "Width stepping factor (default: 1.5)");
    ep_grp_search->add_flag(
        "--fourier,!--time", ep_cfg.use_fourier,
        "Use Fourier-domain folding (default) or time-domain folding");

    auto* ep_grp_pruning = ep->add_option_group("EP Pruning Parameters");
    ep_grp_pruning->add_option(
        "--target-snr", ep_cfg.target_snr,
        "EP entropic information limit (default by search: frequency/accel "
        "8.5, jerk 9.2, snap 9.8, circular 10.0)");
    ep_grp_pruning
        ->add_option("--poly-basis", ep_cfg.poly_basis,
                     "Polynomial basis: taylor (default) or chebyshev")
        ->check(CLI::IsMember({"taylor", "chebyshev"}));
    ep_grp_pruning->add_option(
        "--min-pd", ep_cfg.min_pd,
        "Requested detection probability cutoff, in (0, 1] "
        "(default: 0.1; shortfalls warn and accept the best available scheme)");
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
        "Memory cap in GB of 2^30 bytes (as Slurm's --mem=16G): total "
        "process RSS on CPU, device memory on CUDA");
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

    // =========================================================================
    // Subcommand: inspect
    // =========================================================================
    auto* inspect = app.add_subcommand(
        "inspect", "Summarise an FFA or EP result file: run metadata, top "
                   "candidates and the stored configuration");
    std::string inspect_path;
    std::size_t inspect_top = 10;
    bool inspect_json       = false;
    bool inspect_profile    = false;
    inspect->add_option("file", inspect_path, "Result HDF5 file")
        ->required()
        ->check(CLI::ExistingFile);
    inspect->add_option("-n,--top", inspect_top,
                        "Number of candidates to list (default: 10)");
    inspect->add_flag("--profile", inspect_profile,
                      "Draw the folded profile of the top candidate, when the "
                      "file stores one");
    inspect->add_flag("--json", inspect_json,
                      "Write the summary as JSON to stdout");

    simulate->fallthrough();
    search->fallthrough();
    inspect->fallthrough();
    ffa->fallthrough();
    ep->fallthrough();

    CLI11_PARSE(app, argc, argv);
    try {
        configure_output(output, argc, argv);
    } catch (const std::exception& ex) { // e.g. an unwritable --log-file
        SPDLOG_ERROR("{}", ex.what());
        return 1;
    }
    if (!backend_name.empty()) {
        ffa_cfg.backend = loki::parse_backend(backend_name);
    }
    if (!ep_backend_name.empty()) {
        ep_cfg.backend = loki::parse_backend(ep_backend_name);
    }
    if (ep_no_plan_cache) {
        ep_cfg.use_plan_cache = false;
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

    // Ctrl-C, or SIGTERM (e.g. Slurm's scancel), stops the run at its next
    // checkpoint and unwinds cleanly; a second Ctrl-C kills it at once.
    const loki::InterruptGuard interrupt_guard(/*handle_terminate=*/true);
    try {
        if (simulate->parsed()) {
            if (*opt_gen_sim) {
                if (gen_sim_config_path.empty()) {
                    gen_sim_config_path = "loki_simulate.toml";
                }
                loki::simulation::SimulateTomlConfig::write_default(
                    gen_sim_config_path);
                SPDLOG_INFO("Wrote default simulation configuration to {}",
                            gen_sim_config_path);
                return 0;
            }

            sim_cfg.validate();

            const auto path = resolve_output(sim_cfg.output, sim_cfg.format);
            return run_simulate(
                sim_cfg.period, sim_cfg.dt, sim_cfg.nsamps, sim_cfg.snr,
                sim_cfg.ducy, sim_cfg.shape, sim_cfg.phi0, sim_cfg.seed, path,
                sim_cfg.mod_type, sim_cfg.mod, sim_cfg.mod_tref);
        }

        if (inspect->parsed()) {
            return run_inspect(inspect_path, inspect_top, inspect_json,
                               inspect_profile);
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
    } catch (const loki::Interrupted&) {
        const int sig = loki::interrupt_signal();
        SPDLOG_WARN("Interrupted{}: stopped before finishing; outputs may be "
                    "incomplete",
                    sig == 0 ? "" : fmt::format(" by signal {}", sig));
        // The shell's convention for a run ended by a signal.
        return 128 + (sig == 0 ? SIGINT : sig);
    } catch (const std::exception& ex) {
        loki::cli::Console::get().error(ex.what());
        return 1;
    }

    return 0;
}
