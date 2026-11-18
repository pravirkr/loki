#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
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
#if __has_include(<sys/resource.h>)
#include <sys/resource.h>
#define LOKI_CLI_HAS_RUSAGE 1
#endif

#include "loki/algorithms/ep_regions.hpp"
#include "loki/algorithms/regions.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/interrupt.hpp"
#include "loki/common/plans.hpp"
#include "loki/common/progress.hpp"
#include "loki/common/types.hpp"
#include "loki/io/candidate_file.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/io/search_series.hpp"
#include "loki/io/sift.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/pipelines/ffa_freq_sweep.hpp"
#include "loki/search/configs.hpp"
#include "loki/search/provenance.hpp"
#include "loki/search/refold.hpp"
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
/// the search read from its input), the command line, the version, and the
/// raw TOML text where not yet stored.
template <typename TomlConfig>
void record_provenance(const std::filesystem::path& result_file,
                       TomlConfig cfg,
                       loki::SizeType nsamps,
                       double tsamp,
                       std::string_view config_toml = {}) {
    if (!std::filesystem::exists(result_file)) {
        return;
    }
    cfg.nsamps = nsamps;
    cfg.tsamp  = tsamp;
    loki::search::write_run_provenance(
        result_file, {.config_effective     = cfg.to_toml_string(),
                      .command_line         = run_command_line(),
                      .loki_version         = LOKI_VERSION,
                      .config_toml          = std::string(config_toml),
                      .publish_command_line = {}});
}

/// Runs @p execute, a search that keeps `<spool>` when its post-processing
/// fails. If it fails after keeping a new spool, records the provenance of
/// the search there (as record_provenance) so `loki publish` can recover the
/// configuration, then rethrows the original error.
template <typename TomlConfig, typename Execute>
void execute_recording_spool(Execute&& execute,
                             const std::filesystem::path& spool,
                             const TomlConfig& cfg,
                             loki::SizeType nsamps,
                             double tsamp,
                             std::string_view config_toml = {}) {
    std::error_code error;
    const auto before =
        std::filesystem::exists(spool, error)
            ? std::optional(std::filesystem::last_write_time(spool, error))
            : std::nullopt;
    try {
        std::forward<Execute>(execute)();
    } catch (...) {
        try {
            if (std::filesystem::exists(spool, error) &&
                (!before ||
                 std::filesystem::last_write_time(spool, error) != *before)) {
                record_provenance(spool, cfg, nsamps, tsamp, config_toml);
            }
        } catch (const std::exception& e) {
            spdlog::warn("could not record the configuration in '{}': {}",
                         spool.string(), e.what());
        }
        throw;
    }
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

/// The folding inputs of a search, on its backend, and the summary of their
/// preprocessing. The raw series is not kept.
struct SearchTimeSeries {
    loki::io::SearchSeries series;
    loki::io::ObservationMetadata observation;
    loki::SizeType nsamps{};
    double dt{};
    std::optional<loki::io::PreprocessReport> report;
    double load_s{};       ///< Wall time to read the series.
    double preprocess_s{}; ///< Wall time to preprocess it.
};

using SteadyClock = std::chrono::steady_clock;

double seconds_since(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

/// Peak resident set size of this process in GB (2^30 bytes), the quantity a
/// batch scheduler's memory limit applies to.
std::optional<double> peak_rss_gb() {
#ifdef LOKI_CLI_HAS_RUSAGE
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        // ru_maxrss is in bytes on macOS / BSD, KiB on Linux.
        return static_cast<double>(usage.ru_maxrss) /
               (1024.0 * 1024.0 * 1024.0);
#else
        return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#endif
    }
#endif
    return std::nullopt;
}

/// Writes /run_stats (timings, peak RSS unless @p measure_rss is false,
/// overloads). Never fails the run.
void record_run_stats(const std::filesystem::path& result_file,
                      loki::search::RunStats stats,
                      bool measure_rss = true) {
    if (!std::filesystem::exists(result_file)) {
        return;
    }
    stats.peak_rss_gb = measure_rss ? peak_rss_gb() : std::nullopt;
    try {
        loki::search::write_run_stats(result_file, stats);
    } catch (const std::exception& ex) {
        SPDLOG_WARN("Could not record run statistics in {}: {}",
                    result_file.string(), ex.what());
        return;
    }
    std::string text;
    for (const auto& [stage, seconds] : stats.timing_s) {
        text += fmt::format("{}{} {:.1f} s", text.empty() ? "" : ", ", stage,
                            seconds);
    }
    SPDLOG_INFO("Run statistics: {}{}", text,
                stats.peak_rss_gb
                    ? fmt::format(", peak RSS {:.2f} GB", *stats.peak_rss_gb)
                    : std::string());
}

/// The zap self-check per band; a noise zap that left a coherent residual
/// (or overfilled) warns.
void log_zap_bands(const loki::io::PreprocessReport& rep,
                   loki::io::ZapMode mode) {
    // 0.1, or three standard errors of a mean of n Exp(1) powers.
    constexpr loki::SizeType kMinBins = 64;
    for (const auto& b : rep.zap_bands) {
        const double tolerance =
            std::max(0.1, 3.0 / std::sqrt(static_cast<double>(b.n_zapped)));
        const bool off = mode == loki::io::ZapMode::kNoise &&
                         b.n_zapped >= kMinBins &&
                         std::abs(b.power_ratio - 1.0) > tolerance;
        if (off) {
            SPDLOG_WARN("Zap self-check: {:.0f}-{:.0f} Hz, {} bins, zapped / "
                        "other power = {:.3f} (expected 1)",
                        b.f_lo, b.f_hi, b.n_zapped, b.power_ratio);
        } else {
            SPDLOG_INFO("Zap self-check: {:.0f}-{:.0f} Hz, {} bins, zapped / "
                        "other power = {:.3f}",
                        b.f_lo, b.f_hi, b.n_zapped, b.power_ratio);
        }
    }
}

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
    const auto load_start = SteadyClock::now();
    const auto ts         = loki::io::TimeSeries::read(ts_path, read_opts);
    SearchTimeSeries out;
    out.load_s      = seconds_since(load_start);
    out.observation = ts.get_observation();
    out.nsamps      = ts.get_nsamps();
    out.dt          = ts.get_dt();
    SPDLOG_INFO(
        "Loaded timeseries: nsamps = {}, dt = {:.6e} s, tobs = {:.2f} s",
        ts.get_nsamps(), ts.get_dt(), ts.get_tobs());
    if (!cfg.preprocess) {
        out.series = loki::io::SearchSeries::from_host(
            ts.get_ts_e(), ts.get_ts_v(), out.dt, exec);
        return out;
    }
    const auto options = cfg.to_preprocess_options();
    const bool robust  = options.method == loki::io::PreprocessMethod::kRobust;
    // A GPU search preprocesses on its device and keeps the result there;
    // the CPU thread count serves the z-score method, which the GPU runs on
    // the host.
    exec.nthreads               = nthreads;
    const auto preprocess_start = SteadyClock::now();
    auto [series, rep]          = loki::io::SearchSeries::preprocess(
        ts.get_ts_e(), out.dt, options, exec);
    out.series       = std::move(series);
    out.preprocess_s = seconds_since(preprocess_start);
    const double dt  = out.dt;
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
        log_zap_bands(rep, options.zap_mode);
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

struct SearchDims {
    loki::SizeType nsamps;
    double tsamp;
};

/// The sample count and interval a dry run plans with: those of the input
/// header when the config names an input (only the header is read), else
/// [input] nsamps / tsamp, else 2^21 samples of 64 us.
SearchDims dry_run_dims(const loki::search::FFATomlConfig& cfg) {
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
        return {.nsamps = info.nsamps, .tsamp = info.dt};
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
                   bool dry_run) {
    check_frequency_range(toml_cfg);
    toml_cfg.candidates.validate();
    toml_cfg.observation.validate();
    if (toml_cfg.max_host_memory_gb.has_value()) {
        SPDLOG_WARN("performance.max_host_memory_gb is ignored by FFA "
                    "searches (search ep on CUDA only)");
    }
    const loki::Exec exec = search_exec(toml_cfg);

    if (dry_run) {
        const auto dims  = dry_run_dims(toml_cfg);
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
    const auto run_start = SteadyClock::now();
    auto loaded          = load_search_timeseries(toml_cfg, exec);
    if (device_init.joinable()) {
        device_init.join();
    }
    const loki::SizeType actual_nsamps = loaded.nsamps;
    const double dt                    = loaded.dt;

    const auto ffa_cfg = toml_cfg.to_search_config(actual_nsamps, dt);
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
    auto observation = loaded.observation;
    if (toml_cfg.observation.dm) {
        observation.dm = toml_cfg.observation.dm;
    }
    if (toml_cfg.observation.start_mjd) {
        observation.start_mjd = toml_cfg.observation.start_mjd;
    }
    if (toml_cfg.observation.barycentric) {
        observation.barycentric = toml_cfg.observation.barycentric;
    }
    const auto plan_start = SteadyClock::now();
    loki::pipelines::FFAFreqSweep sweep(ffa_cfg, toml_cfg.candidate_options(),
                                        /*show_progress=*/true, exec);
    const double plan_s     = seconds_since(plan_start);
    const auto search_start = SteadyClock::now();
    execute_recording_spool(
        [&] {
            sweep.execute(loaded.series, outdir_path, toml_cfg.prefix,
                          config_toml, observation);
        },
        outdir_path / fmt::format("{}_ffa_spool.h5", toml_cfg.prefix), toml_cfg,
        actual_nsamps, dt);
    const double search_s = seconds_since(search_start);

    const auto result_file =
        outdir_path / fmt::format("{}_ffa_results.h5", toml_cfg.prefix);
    record_provenance(result_file, toml_cfg, actual_nsamps, dt);
    record_run_stats(result_file,
                     {.timing_s    = {{"load", loaded.load_s},
                                      {"preprocess", loaded.preprocess_s},
                                      {"plan", plan_s},
                                      {"search_and_postprocess", search_s},
                                      {"total", seconds_since(run_start)}},
                      .peak_rss_gb = std::nullopt});
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
        const auto n_slices = chunk.seed_slice_edges.empty()
                                  ? SizeType{1}
                                  : chunk.seed_slice_edges.size() - 1;
        std::cout << con.muted(fmt::format(
                         "       predicted peak {}, live ring {}, batch {}{}",
                         format_pow2(chunk.predicted_peak),
                         format_pow2(chunk.predicted_live), chunk.batch_size,
                         n_slices > 1
                             ? fmt::format(", {} seed slices (forecast per "
                                           "slice)",
                                           n_slices)
                             : std::string{}))
                  << "\n";
    }

    // The sweep's groups: contiguous chunks of equal nbins and batch.
    std::cout << con.title("Groups of equal nbins and batch")
              << ", each with its own workspaces:\n";
    for (SizeType begin = 0; begin < chunks.size();) {
        const auto nbins    = chunks[begin].cfg.get_nbins();
        const auto batch    = chunks[begin].batch_size;
        SizeType end        = begin;
        SizeType max_sugg   = 0;
        SizeType branch_max = 0;
        while (end < chunks.size() && chunks[end].cfg.get_nbins() == nbins &&
               chunks[end].batch_size == batch) {
            max_sugg   = std::max(max_sugg, chunks[end].max_sugg);
            branch_max = std::max(branch_max, chunks[end].branch_max);
            ++end;
        }
        std::cout << fmt::format(
            "  chunks {}..{}: nbins {}, batch {}, max_sugg {}, branch_max {}, "
            "{} worker(s)\n",
            begin, end - 1, nbins, batch, format_pow2(max_sugg), branch_max,
            workers);
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
    if (config.candidates.enabled && config.poly_basis == "taylor") {
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

/// Host memory of a CUDA EP search, GB (2^30). The device holds the search;
/// the host holds the CUDA context throughout, the series and its variance
/// while it loads and preprocesses, then the final leaves of the runs in
/// flight while it searches, and the candidate budget after.
struct EPHostForecast {
    double load_gb{0.0};
    double search_gb{0.0};
    [[nodiscard]] double peak_gb() const noexcept {
        return std::max(load_gb, search_gb);
    }
};

/// The process with its CUDA context and no data (0.15 GB measured, L40,
/// CUDA 12).
constexpr double kCudaHostBaseGb = 0.15;
constexpr double kBytesPerGb     = 1024.0 * 1024.0 * 1024.0;

EPHostForecast ep_cuda_host_forecast(const loki::search::EPTomlConfig& cfg,
                                     SizeType nsamps,
                                     SizeType nparams) {
    // The series and its variance, 4 bytes a sample each. The z-score
    // preprocessing runs on the host into a second pair.
    const bool host_preprocess =
        cfg.preprocess && cfg.to_preprocess_options().method !=
                              loki::io::PreprocessMethod::kRobust;
    const double bytes_per_sample = host_preprocess ? 16.0 : 8.0;
    const auto host = loki::algorithms::ep_cuda_host_memory(nparams);
    const bool candidates =
        cfg.candidates.enabled && cfg.poly_basis == "taylor";
    // Candidate processing starts once the sweep has written every run.
    const double search_data_gb =
        std::max(static_cast<double>(host.sweep_bytes()) / kBytesPerGb,
                 candidates ? cfg.candidates.memory_gb : 0.0);
    return {.load_gb =
                kCudaHostBaseGb +
                (bytes_per_sample * static_cast<double>(nsamps) / kBytesPerGb),
            .search_gb = kCudaHostBaseGb + search_data_gb};
}

/// Logs the host-memory forecast of a CUDA EP search and stops one whose
/// forecast is above performance.max_host_memory_gb. On the CPU the cap is
/// ignored: max_process_memory_gb caps the host memory there.
void check_ep_host_memory(const loki::search::EPTomlConfig& cfg,
                          loki::Exec exec,
                          SizeType nsamps,
                          SizeType nparams) {
    if (exec.backend == loki::Backend::kCPU) {
        if (cfg.max_host_memory_gb.has_value()) {
            SPDLOG_WARN("performance.max_host_memory_gb is ignored on the "
                        "CPU: max_process_memory_gb caps the host memory "
                        "there");
        }
        return;
    }
    const auto forecast = ep_cuda_host_forecast(cfg, nsamps, nparams);
    const auto limit    = cfg.max_host_memory_gb;
    SPDLOG_INFO("Host memory forecast: {:.2f} GB (loading {:.2f} GB, "
                "searching {:.2f} GB){}",
                forecast.peak_gb(), forecast.load_gb, forecast.search_gb,
                limit ? fmt::format(", limit {:.2f} GB", *limit)
                      : std::string{});
    if (limit && forecast.peak_gb() > *limit) {
        throw std::runtime_error(fmt::format(
            "host memory forecast {:.2f} GB (loading {:.2f} GB for {} "
            "samples, searching {:.2f} GB) is above "
            "performance.max_host_memory_gb = {:.2f} GB",
            forecast.peak_gb(), forecast.load_gb, nsamps, forecast.search_gb,
            *limit));
    }
}

/// Prints the host memory of a CUDA plan: the forecast, and what a run of
/// one slice holds if a source fills its tree (not bounded by the forecast).
template <typename FoldType>
void print_ep_host_memory(
    const loki::algorithms::EPRegionPlanner<FoldType>& planner,
    const loki::search::EPTomlConfig& toml_cfg,
    const loki::search::PulsarSearchConfig& cfg) {
    const auto forecast =
        ep_cuda_host_forecast(toml_cfg, cfg.get_nsamps(), cfg.get_nparams());
    const auto host = loki::algorithms::ep_cuda_host_memory(cfg.get_nparams());
    SizeType unsliced_max_sugg = 0;
    for (const auto& chunk : planner.get_chunk_cfgs()) {
        if (chunk.seed_slice_edges.size() <= 2) {
            unsliced_max_sugg = std::max(unsliced_max_sugg, chunk.max_sugg);
        }
    }
    const auto& con = loki::cli::Console::get();
    std::cout << con.title("Host memory:")
              << fmt::format(" {:.2f} GB forecast (loading {:.2f} GB, "
                             "searching {:.2f} GB){}\n",
                             forecast.peak_gb(), forecast.load_gb,
                             forecast.search_gb,
                             toml_cfg.max_host_memory_gb
                                 ? fmt::format(" against a limit of {:.2f} GB",
                                               *toml_cfg.max_host_memory_gb)
                                 : std::string{});
    if (unsliced_max_sugg > 0) {
        std::cout << con.muted(fmt::format(
                         "       a run of one seed slice whose tree a source "
                         "fills holds up to {:.2f} GB more (not in the "
                         "forecast)",
                         static_cast<double>(unsliced_max_sugg *
                                             host.leaf_bytes) /
                             kBytesPerGb))
                  << "\n";
    }
}

/// `--pulsar-mask F_LO:F_HI` values as [f_lo, f_hi] windows.
std::vector<std::array<double, 2>>
parse_pulsar_mask(const std::vector<std::string>& values) {
    std::vector<std::array<double, 2>> windows;
    for (const auto& value : values) {
        const auto colon = value.find(':');
        try {
            if (colon == std::string::npos) {
                throw std::invalid_argument("missing ':'");
            }
            std::size_t used_lo = 0;
            std::size_t used_hi = 0;
            const auto lo_text  = value.substr(0, colon);
            const auto hi_text  = value.substr(colon + 1);
            const double f_lo   = std::stod(lo_text, &used_lo);
            const double f_hi   = std::stod(hi_text, &used_hi);
            if (used_lo != lo_text.size() || used_hi != hi_text.size()) {
                throw std::invalid_argument("trailing characters");
            }
            windows.push_back({f_lo, f_hi});
        } catch (const std::exception&) {
            throw std::invalid_argument(fmt::format(
                "--pulsar-mask expects F_LO:F_HI in Hz (got '{}')", value));
        }
    }
    return windows;
}

/// Plans the EP chunks of @p cfg (or loads them from @p cache_path, writing
/// the cache when it does not exist) and prints them. No sweep is built.
void plan_and_print_ep(const loki::search::EPTomlConfig& toml_cfg,
                       const loki::search::PulsarSearchConfig& input_cfg,
                       const std::optional<std::filesystem::path>& cache_path,
                       loki::Exec exec,
                       bool force_replan = false) {
    const auto cfg = ep_candidate_planning_config(toml_cfg, input_cfg, exec);
    // A dry run stops where the search would.
    check_ep_host_memory(toml_cfg, exec, cfg.get_nsamps(), cfg.get_nparams());
    // The same worker count the sweep uses, so the cache matches it. The GPU
    // sweep prunes its runs on one worker.
    const SizeType n_workers = loki::algorithms::ep_sweep_n_workers(
        cfg.get_nthreads(), toml_cfg.n_runs, toml_cfg.ref_segs);
    const SizeType workers =
        exec.backend == loki::Backend::kCPU ? n_workers : SizeType{1};
    const auto load_path = force_replan ? std::nullopt : cache_path;
    if (cfg.get_use_fourier()) {
        const loki::algorithms::EPRegionPlanner<loki::ComplexType> planner(
            cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
            load_path, toml_cfg.prune_rfi_config(), n_workers, toml_cfg.n_runs,
            toml_cfg.ref_segs, exec);
        if (force_replan && cache_path.has_value()) {
            planner.save_cache(*cache_path);
        }
        print_ep_plan(planner, workers, cache_path);
        if (exec.backend != loki::Backend::kCPU) {
            print_ep_host_memory(planner, toml_cfg, cfg);
        }
    } else {
        const loki::algorithms::EPRegionPlanner<float> planner(
            cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
            load_path, toml_cfg.prune_rfi_config(), n_workers, toml_cfg.n_runs,
            toml_cfg.ref_segs, exec);
        if (force_replan && cache_path.has_value()) {
            planner.save_cache(*cache_path);
        }
        print_ep_plan(planner, workers, cache_path);
        if (exec.backend != loki::Backend::kCPU) {
            print_ep_host_memory(planner, toml_cfg, cfg);
        }
    }
}

/// `search ep --plan-only`: plans the chunks, writes the plan cache and prints
/// the plan. No samples are read: the sample count and interval come from
/// [input] nsamps and tsamp (--nsamps, --tsamp) when both are set, else from
/// the header of the input timeseries.
int run_plan_ep(loki::search::EPTomlConfig toml_cfg,
                bool clamp_default_runs = false) {
    check_frequency_range(toml_cfg);
    toml_cfg.check_rfi_backend();
    const loki::Exec exec = search_exec(toml_cfg);
    if ((!toml_cfg.nsamps.has_value() || !toml_cfg.tsamp.has_value()) &&
        !toml_cfg.timeseries_path.empty() &&
        std::filesystem::exists(toml_cfg.timeseries_path)) {
        const auto dims = dry_run_dims(toml_cfg);
        toml_cfg.nsamps = dims.nsamps;
        toml_cfg.tsamp  = dims.tsamp;
    }
    if (!toml_cfg.nsamps.has_value() || !toml_cfg.tsamp.has_value()) {
        throw std::invalid_argument(
            "--plan-only needs the sample count and interval: pass an input "
            "timeseries (-i, its header is read), set [input] nsamps and "
            "tsamp, or pass --nsamps and --tsamp (the plan depends on "
            "nsamps)");
    }
    const auto cfg =
        toml_cfg.to_ep_search_config(*toml_cfg.nsamps, *toml_cfg.tsamp);
    if (clamp_default_runs && toml_cfg.n_runs.has_value()) {
        const loki::plans::FFAPlanBase plan_base(cfg);
        toml_cfg.n_runs =
            std::min(*toml_cfg.n_runs, plan_base.get_nsegments_ref_pool());
    }
    const auto cache_path = toml_cfg.plan_cache.value_or(
        std::filesystem::path(toml_cfg.outdir) /
        fmt::format("{}_ep_plan.h5", toml_cfg.prefix));

    SPDLOG_INFO("Planning EP search (no data read): f=[{:.3f}, {:.3f}] Hz, "
                "nsamps={}, nbins={}, backend={}",
                cfg.get_f_min(), cfg.get_f_max(), cfg.get_nsamps(),
                cfg.get_nbins(), loki::to_string(exec.backend));
    plan_and_print_ep(toml_cfg, cfg, cache_path, exec, /*force_replan=*/true);
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
        std::vector<std::pair<std::string, std::string>> rows;
        if (h5.hasAttribute("n_candidates")) {
            std::uint64_t detections = 0;
            std::uint64_t selected   = 0;
            h5.getAttribute("n_detections").read(detections);
            h5.getAttribute("n_candidates").read(selected);
            SPDLOG_INFO("Final EP detections above candidate S/N : {}",
                        detections);
            SPDLOG_INFO("Selected EP candidates                  : {}",
                        selected);
            rows.emplace_back("Detections above snr_min",
                              std::to_string(detections));
            rows.emplace_back("Selected candidates", std::to_string(selected));
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
            rows.emplace_back("Runs pruned",
                              fmt::format("{} ({} leaves)", n_runs, n_leaves));
        }

        if (h5.exist("candidates/snr")) {
            const auto snr_dset  = h5.getDataSet("candidates/snr");
            const auto dims      = snr_dset.getDimensions();
            const size_t n_cands = dims.empty() ? 0UL : dims[0];
            if (n_cands > 0) {
                float top_snr           = std::numeric_limits<float>::lowest();
                size_t top_idx          = 0;
                constexpr size_t kChunk = 65536;
                for (size_t offset = 0; offset < n_cands; offset += kChunk) {
                    const size_t count = std::min(kChunk, n_cands - offset);
                    std::vector<float> snrs(count);
                    snr_dset.select({offset}, {count}).read(snrs);
                    const auto max_it = std::ranges::max_element(snrs);
                    if (max_it != snrs.end() && *max_it > top_snr) {
                        top_snr = *max_it;
                        top_idx = offset + static_cast<size_t>(std::distance(
                                               snrs.begin(), max_it));
                    }
                }
                rows.emplace_back(
                    "Top candidate SNR",
                    fmt::format("{:.2f} (candidate #{})", top_snr, top_idx));
                if (h5.exist("candidates/param_sets") && n_params > 0) {
                    const auto p_dset = h5.getDataSet("candidates/param_sets");
                    const auto p_dims = p_dset.getDimensions();
                    if (p_dims.size() == 2 && p_dims[0] == n_cands &&
                        p_dims[1] >= n_params) {
                        std::vector<double> top_params(p_dims[1]);
                        p_dset.select({top_idx, 0}, {1, p_dims[1]})
                            .read_raw(top_params.data());
                        std::string p_str;
                        for (SizeType pi = 0; pi < n_params; ++pi) {
                            if (pi > 0) {
                                p_str += ", ";
                            }
                            p_str += fmt::format("{}={:.6f}", param_names[pi],
                                                 top_params[pi]);
                        }
                        rows.emplace_back("Top candidate parameters",
                                          fmt::format("[{}]", p_str));
                    }
                }
            } else {
                rows.emplace_back("Top candidate", "none above snr_min");
            }
        } else if (found) {
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
        // Early-harvested sources are not in the candidate tables.
        if (h5.exist("harvested/snr")) {
            std::vector<float> h_snr;
            h5.getDataSet("harvested/snr").read(h_snr);
            if (!h_snr.empty()) {
                std::vector<double> h_par(n_params);
                h5.getDataSet("harvested/param_sets")
                    .select({0, 0}, {1, n_params})
                    .read_raw(h_par.data());
                rows.emplace_back(
                    "Harvested sources",
                    fmt::format("{} (strongest: f = {:.6f} Hz, partial S/N "
                                "{:.1f}); see /harvested",
                                h_snr.size(), h_par.back(), h_snr.front()));
            }
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
    if (!toml_cfg.plan_cache.has_value() && !toml_cfg.use_plan_cache) {
        return std::nullopt;
    }
    const auto path = toml_cfg.plan_cache.value_or(
        std::filesystem::path(toml_cfg.outdir) /
        fmt::format("{}_ep_plan.h5", toml_cfg.prefix));
    const auto workers = loki::algorithms::ep_sweep_n_workers(
        cfg.get_nthreads(), toml_cfg.n_runs, toml_cfg.ref_segs);
    // A cache that no longer matches the configuration is replanned and
    // overwritten; planning errors themselves propagate.
    if (std::filesystem::exists(path)) {
        try {
            const loki::algorithms::EPRegionPlanner<FoldType> cached(
                cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
                path, toml_cfg.prune_rfi_config(), workers, toml_cfg.n_runs,
                toml_cfg.ref_segs, exec);
            return path;
        } catch (const std::exception& error) {
            SPDLOG_WARN("EP plan cache '{}' is stale, replanning: {}",
                        path.string(), error.what());
        }
    }
    const loki::algorithms::EPRegionPlanner<FoldType> fresh(
        cfg, toml_cfg.min_pd, toml_cfg.poly_basis, toml_cfg.ref_ducy,
        std::nullopt, toml_cfg.prune_rfi_config(), workers, toml_cfg.n_runs,
        toml_cfg.ref_segs, exec);
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
/// Stage timings of an EP run. The sweep records its own runtime
/// (`total_runtime`); the rest of execute() is candidate postprocessing.
loki::search::RunStats ep_run_stats(const std::filesystem::path& result_file,
                                    const SearchTimeSeries& loaded,
                                    double plan_s,
                                    double execute_s,
                                    double total_s) {
    loki::search::RunStats stats;
    stats.timing_s  = {{"load", loaded.load_s},
                       {"preprocess", loaded.preprocess_s},
                       {"plan", plan_s}};
    double search_s = -1.0;
    try {
        if (std::filesystem::exists(result_file)) {
            const HighFive::File h5(result_file.string(),
                                    HighFive::File::ReadOnly);
            if (h5.hasAttribute("total_runtime")) {
                float runtime = 0.0F;
                h5.getAttribute("total_runtime").read(runtime);
                search_s = runtime;
            }
        }
        // NOLINTNEXTLINE(bugprone-empty-catch): /run_stats is optional; fall back to combined timing
    } catch (const std::exception&) {
    }
    if (search_s >= 0.0 && search_s <= execute_s) {
        stats.timing_s.emplace_back("search", search_s);
        stats.timing_s.emplace_back("postprocess", execute_s - search_s);
    } else {
        stats.timing_s.emplace_back("search_and_postprocess", execute_s);
    }
    stats.timing_s.emplace_back("total", total_s);
    return stats;
}

int run_search_ep(loki::search::EPTomlConfig toml_cfg,
                  std::string_view config_toml,
                  bool dry_run,
                  bool clamp_default_runs = false) {
    check_frequency_range(toml_cfg);
    // Before any data is read, in a dry run too.
    toml_cfg.check_rfi_backend();
    const loki::Exec exec = search_exec(toml_cfg);
    if (dry_run) {
        const auto dims = dry_run_dims(toml_cfg);
        const auto preview_cfg =
            toml_cfg.to_ep_search_config(dims.nsamps, dims.tsamp);
        if (clamp_default_runs && toml_cfg.n_runs.has_value()) {
            const loki::plans::FFAPlanBase plan_base(preview_cfg);
            toml_cfg.n_runs =
                std::min(*toml_cfg.n_runs, plan_base.get_nsegments_ref_pool());
        }
        // The plan cache is prepared as the search would prepare it, so a
        // search after the dry run reuses it.
        plan_and_print_ep(toml_cfg, preview_cfg,
                          prepare_ep_plan_cache(toml_cfg, preview_cfg, exec),
                          exec);
        SPDLOG_INFO("Dry run complete: planner constructed successfully.");
        return 0;
    }

    // Before the samples are read, so a series too large for the host is
    // refused before it is allocated.
    if (!toml_cfg.timeseries_path.empty() &&
        std::filesystem::exists(toml_cfg.timeseries_path)) {
        const auto info =
            loki::io::TimeSeries::read_info(toml_cfg.timeseries_path);
        check_ep_host_memory(
            toml_cfg, exec, info.nsamps,
            toml_cfg.to_ep_search_config(info.nsamps, info.dt).get_nparams());
    }
    const auto run_start               = SteadyClock::now();
    auto loaded                        = load_search_timeseries(toml_cfg, exec);
    const loki::SizeType actual_nsamps = loaded.nsamps;
    const double dt                    = loaded.dt;
    const auto ep_cfg = toml_cfg.to_ep_search_config(actual_nsamps, dt);
    if (clamp_default_runs && toml_cfg.n_runs.has_value()) {
        const loki::plans::FFAPlanBase plan_base(ep_cfg);
        toml_cfg.n_runs =
            std::min(*toml_cfg.n_runs, plan_base.get_nsegments_ref_pool());
    }
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

    const auto plan_start = SteadyClock::now();
    const auto cache      = prepare_ep_plan_cache(toml_cfg, ep_cfg, exec);
    const bool common_candidates =
        toml_cfg.candidates.enabled && toml_cfg.poly_basis == "taylor";
    const auto rfi_config = toml_cfg.prune_rfi_config();
    auto sweep =
        common_candidates
            ? loki::pipelines::EPFreqSweep(
                  ep_cfg, toml_cfg.candidate_options(), true, toml_cfg.min_pd,
                  toml_cfg.poly_basis, toml_cfg.ref_ducy, rfi_config, cache,
                  toml_cfg.n_runs, toml_cfg.ref_segs, exec)
            : loki::pipelines::EPFreqSweep(
                  ep_cfg, true, toml_cfg.min_pd, toml_cfg.poly_basis,
                  toml_cfg.ref_ducy, rfi_config, cache, toml_cfg.n_runs,
                  toml_cfg.ref_segs, exec);
    if (toml_cfg.candidates.enabled && !common_candidates) {
        SPDLOG_WARN("Common EP candidates currently support polynomial Taylor "
                    "only; writing run diagnostics");
    }
    auto observation = loaded.observation;
    if (toml_cfg.observation.dm) {
        observation.dm = toml_cfg.observation.dm;
    }
    if (toml_cfg.observation.start_mjd) {
        observation.start_mjd = toml_cfg.observation.start_mjd;
    }
    if (toml_cfg.observation.barycentric) {
        observation.barycentric = toml_cfg.observation.barycentric;
    }
    const double plan_s     = seconds_since(plan_start);
    const auto search_start = SteadyClock::now();
    execute_recording_spool(
        [&] {
            sweep.execute(loaded.series, outdir_path, toml_cfg.prefix,
                          observation);
        },
        outdir_path / fmt::format("{}_ep_spool.h5", toml_cfg.prefix), toml_cfg,
        actual_nsamps, dt, config_toml);
    const double execute_s = seconds_since(search_start);

    const auto result_file =
        outdir_path / fmt::format("{}_ep_results.h5", toml_cfg.prefix);
    record_provenance(result_file, toml_cfg, actual_nsamps, dt, config_toml);
    record_run_stats(result_file,
                     ep_run_stats(result_file, loaded, plan_s, execute_s,
                                  seconds_since(run_start)));
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
            "{}\n    {{\"rank\": {}, \"row\": {}, \"id\": {}, \"snr\": {}, "
            "\"params\": {{{}}}, \"width\": {}, \"nbins\": {}, "
            "\"source\": {}}}",
            i == 0 ? "" : ",", i + 1, row.row, row.id, json_number(row.snr),
            params, row.width, row.nbins, json_string(row.source));
    }
    if (s.n_harvested_sources > 0) {
        std::string harvested;
        for (std::size_t i = 0; i < s.harvested.size(); ++i) {
            const auto& row = s.harvested[i];
            std::string params;
            for (std::size_t p = 0; p < row.params.size(); ++p) {
                params += fmt::format("{}{}: {}", p == 0 ? "" : ", ",
                                      json_string(s.param_names[p]),
                                      json_number(row.params[p]));
            }
            harvested += fmt::format(
                "{}\n    {{\"rank\": {}, \"partial_snr\": {}, \"params\": "
                "{{{}}}, \"source\": {}}}",
                i == 0 ? "" : ",", i + 1, json_number(row.snr), params,
                json_string(row.source));
        }
        field("n_harvested_sources", fmt::format("{}", s.n_harvested_sources));
        field("harvested",
              harvested.empty() ? "[]" : "[" + harvested + "\n  ]");
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

    if (s.n_harvested_sources > 0) {
        out += "\n" +
               con.heading(fmt::format(
                   "Harvested sources ({} of {}; partial S/N at the harvest "
                   "level)",
                   s.harvested.size(), s.n_harvested_sources)) +
               "\n";
        for (std::size_t i = 0; i < s.harvested.size(); ++i) {
            const auto& row = s.harvested[i];
            std::string params;
            for (std::size_t p = 0; p < row.params.size(); ++p) {
                params +=
                    fmt::format("{}{} {}", p == 0 ? "" : ", ",
                                param_label(s.param_names[p]),
                                format_param(s.param_names[p], row.params[p]));
            }
            out += fmt::format("  {:>3}  {}  {}  {}\n",
                               con.muted(fmt::format("{}", i + 1)),
                               con.score(fmt::format("{:6.2f}", row.snr)),
                               params, con.muted(row.source));
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

/// `loki refold` inputs.
struct RefoldCli {
    std::filesystem::path series;
    std::filesystem::path result;
    std::vector<std::uint64_t> ids;
    std::size_t top{0};
    std::filesystem::path config;
    std::vector<double> freqs;
    std::optional<double> fd;
    std::optional<double> acc;
    std::optional<double> fdd;
    std::optional<double> jerk;
    double snap{0.0};
    double crackle{0.0};
    bool circular{false};
    bool no_preprocess{false};
    bool json{false};
    bool profile{false};
    int threads{0};
    loki::search::RefoldOptions options;
};

/// One thing to refold and where it came from.
struct RefoldItem {
    std::string label;
    loki::search::RefoldTarget target;
    std::optional<loki::io::CandidateRow> candidate;
};

/// Preprocessing of a refold: the search configuration's (a result file's
/// config_effective, or -c), else the defaults.
struct RefoldSetup {
    bool preprocess{true};
    loki::io::PreprocessOptions options;
    double propagator_significance{2.0};
    std::string source{"defaults"};
};

/// @p algorithm "ep" or "ffa" reads the configuration as that search's;
/// empty tries EP, then FFA.
RefoldSetup refold_setup(const std::optional<std::string>& text,
                         std::string source,
                         std::string_view algorithm) {
    RefoldSetup setup;
    if (!text || text->empty()) {
        return setup;
    }
    std::string error;
    if (algorithm != "ffa") {
        try {
            const auto cfg   = loki::search::EPTomlConfig::from_string(*text);
            setup.preprocess = cfg.preprocess;
            setup.options    = cfg.to_preprocess_options();
            setup.propagator_significance = cfg.propagator_significance;
            setup.source                  = std::move(source);
            return setup;
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    if (algorithm != "ep") {
        try {
            const auto cfg   = loki::search::FFATomlConfig::from_string(*text);
            setup.preprocess = cfg.preprocess;
            setup.options    = cfg.to_preprocess_options();
            setup.source     = std::move(source);
            return setup;
        } catch (const std::exception& e) {
            if (error.empty()) {
                error = e.what();
            }
        }
    }
    throw std::invalid_argument(fmt::format(
        "cannot read the search configuration ({}): {}\nPass the search's "
        "configuration, updated to this version, with -c",
        source, error));
}

/// The target of a candidate row: its leaf with the search's model (EP), or
/// the Taylor polynomial of its native parameters (FFA).
loki::search::RefoldTarget candidate_target(const loki::io::CandidateRow& row,
                                            std::size_t nparams,
                                            double propagator_significance) {
    if (!row.leaf.empty()) {
        return {.leaf                    = row.leaf,
                .n_params                = nparams,
                .model                   = loki::search::RefoldModel::kLeaf,
                .propagator_significance = propagator_significance};
    }
    // Below five parameters the leaf model is the Taylor polynomial; a
    // five-parameter leaf may be an orbit, which its values alone do not say.
    if (nparams >= 5) {
        throw std::invalid_argument(fmt::format(
            "the result file does not keep the leaf of candidate {}, needed "
            "for the five-parameter phase model; refold its parameters with "
            "--f ... --circular",
            row.id));
    }
    // Native order [..., jerk, accel, freq]; derivatives from accel up.
    std::vector<double> derivs(row.params.rbegin() + 1, row.params.rend());
    return loki::search::refold_taylor_target(row.params.back(), derivs);
}

std::string
refold_json(const RefoldCli& cli,
            loki::SizeType nsamps,
            double tsamp,
            double t_ref,
            const RefoldSetup& setup,
            const std::vector<RefoldItem>& items,
            const std::vector<loki::search::RefoldResult>& results) {
    std::string out = "{\n";
    out += fmt::format("  \"series\": {},\n", json_string(cli.series.string()));
    out += fmt::format("  \"result_file\": {},\n",
                       cli.result.empty() ? "null"
                                          : json_string(cli.result.string()));
    out +=
        fmt::format("  \"nsamps\": {},\n  \"tsamp\": {},\n  \"t_ref\": {},\n",
                    nsamps, json_number(tsamp), json_number(t_ref));
    out += fmt::format("  \"preprocessing\": {},\n",
                       json_string(setup.preprocess ? setup.source : "off"));
    out += "  \"targets\": [\n";
    const auto list = [](const auto& values) {
        std::string text = "[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            text += fmt::format("{}{}", i == 0 ? "" : ", ",
                                json_number(static_cast<double>(values[i])));
        }
        return text + "]";
    };
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        const auto& r    = results[i];
        const double f   = r.freq + r.df;
        const double fd  = r.fdot + r.dfd;
        out += "    {";
        out += fmt::format("\"label\": {}, ", json_string(item.label));
        if (item.candidate) {
            out += fmt::format(
                "\"id\": {}, \"search_snr\": {}, \"search_nbins\": {}, "
                "\"search_width\": {}, \"source\": {}, ",
                item.candidate->id, json_number(item.candidate->snr),
                item.candidate->nbins, item.candidate->width,
                json_string(item.candidate->source));
        }
        const double accel = f != 0.0 ? -fd / f * 299792458.0 : 0.0;
        out += fmt::format(
            "\"model\": {}, \"nbins\": {}, \"freq\": {}, \"fdot\": {}, "
            "\"snr_in\": {}, \"width_in\": {}, \"snr\": {}, \"width\": {}, "
            "\"start_bin\": {}, \"f_best\": {}, \"fdot_best\": {}, "
            "\"accel_best\": {}, \"split_snr\": {}, \"split_start_bin\": {}",
            json_string(r.model), r.nbins, json_number(r.freq),
            json_number(r.fdot), json_number(r.snr_in), r.width_in,
            json_number(r.snr), r.width, r.start_bin, json_number(f),
            json_number(fd), json_number(accel), list(r.split_snr),
            list(r.split_start_bin));
        if (cli.profile) {
            out += fmt::format(", \"profile\": {}", list(r.profile));
        }
        out += i + 1 == items.size() ? "}\n" : "},\n";
    }
    return out + "  ]\n}\n";
}

std::string
refold_text(const RefoldCli& cli,
            loki::SizeType nsamps,
            loki::SizeType nsamps_file,
            double tsamp,
            double t_ref,
            const RefoldSetup& setup,
            const std::vector<RefoldItem>& items,
            const std::vector<loki::search::RefoldResult>& results) {
    const auto& con = loki::cli::Console::get();
    std::string out = con.heading(fmt::format("Refold of {}",
                                              cli.series.filename().string())) +
                      "\n";
    out += fmt::format(
        "  {} of {} samples, tsamp {:.6e} s, T {:.2f} s, t_ref {:.3f} s, "
        "preprocessing: {}, nsub {}\n",
        nsamps, nsamps_file, tsamp, static_cast<double>(nsamps) * tsamp, t_ref,
        setup.preprocess ? setup.source : "off", cli.options.nsub);
    out += con.muted("  S/N: best boxcar (widths 1..nbins/2) of the refold; a "
                     "rejection test only (noise also refolds near its "
                     "search S/N)") +
           "\n\n";
    out += con.header(fmt::format(
               "  {:<24} {:>8} {:>13} {:>7} {:>14} {:>11} {:>9} {:>7} {:>7}  "
               "{}",
               "target", "model", "f_in", "S/N_in", "f_best", "fd_best",
               "acc_best", "S/N", "w/nb", "search S/N w/nb")) +
           "\n";
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item        = items[i];
        const auto& r           = results[i];
        const double f          = r.freq + r.df;
        const double fd         = r.fdot + r.dfd;
        std::string search_cols = "-";
        if (item.candidate) {
            search_cols =
                fmt::format("{:.2f} {}/{}", item.candidate->snr,
                            item.candidate->width, item.candidate->nbins);
        }
        const auto model   = r.model == "taylor"          ? "taylor"
                             : r.model == "circular_snap" ? "circ_s"
                                                          : "circ_c";
        const double accel = f != 0.0 ? -fd / f * 299792458.0 : 0.0;
        out += fmt::format(
            "  {:<24} {:>8} {:13.6f} {:7.2f} {:14.7f} {:11.3e} {:9.3f} {} "
            "{:>7}  {}\n",
            item.label, model, r.freq, r.snr_in, f, fd, accel,
            con.score(fmt::format("{:7.2f}", r.snr)),
            fmt::format("{}/{}", r.width, r.nbins), search_cols);
        if (!r.split_snr.empty()) {
            std::string parts;
            for (std::size_t j = 0; j < r.split_snr.size(); ++j) {
                parts += fmt::format(" {:.2f}@{}", r.split_snr[j],
                                     r.split_start_bin[j]);
            }
            out += con.muted(fmt::format(
                       "    split {} (S/N@start bin; full start bin {}, "
                       "width {}):{}",
                       r.split_snr.size(), r.start_bin, r.width, parts)) +
                   "\n";
        }
        if (cli.profile) {
            std::string prof;
            for (const float x : r.profile) {
                prof += fmt::format(" {:5.1f}", x);
            }
            out += con.muted("    profile:" + prof) + "\n";
        }
    }
    return out;
}

int run_refold(RefoldCli cli) {
    const int nthreads = cli.threads > 0 ? cli.threads : omp_get_max_threads();
    cli.options.nthreads = nthreads;

    // Targets, and the configuration of the search that found them.
    std::optional<std::string> config_text;
    std::string config_source;
    if (!cli.config.empty()) {
        std::ifstream in(cli.config);
        if (!in) {
            throw std::invalid_argument(
                fmt::format("cannot read {}", cli.config.string()));
        }
        config_text   = std::string(std::istreambuf_iterator<char>(in), {});
        config_source = cli.config.filename().string();
    }
    loki::io::CandidateFileSummary summary;
    std::vector<loki::io::CandidateRow> rows;
    if (!cli.result.empty()) {
        for (const auto id : cli.ids) {
            rows.push_back(
                loki::io::read_candidate_by_id(cli.result, id, &summary));
        }
        if (cli.top > 0) {
            summary =
                loki::io::read_candidate_file_summary(cli.result, cli.top);
            rows.insert(rows.end(), summary.top.begin(), summary.top.end());
        }
        if (cli.ids.empty() && cli.top == 0) {
            throw std::invalid_argument("--result needs --id or --top");
        }
        if (!config_text) {
            config_text   = summary.config_effective ? summary.config_effective
                                                     : summary.config_toml;
            config_source = "the result file's configuration";
        }
    } else if (!cli.ids.empty() || cli.top > 0) {
        throw std::invalid_argument("--id and --top need --result");
    }
    auto setup = refold_setup(config_text, config_source,
                              cli.config.empty() ? summary.algorithm : "");
    if (cli.no_preprocess) {
        setup.preprocess = false;
    }

    std::vector<RefoldItem> items;
    for (const auto& row : rows) {
        items.push_back(
            {.label     = fmt::format("id {}", row.id),
             .target    = candidate_target(row, summary.param_names.size(),
                                           setup.propagator_significance),
             .candidate = row});
    }
    for (const double f : cli.freqs) {
        const double accel =
            cli.acc ? *cli.acc : -cli.fd.value_or(0.0) * 299792458.0 / f;
        const double jerk =
            cli.jerk ? *cli.jerk : -cli.fdd.value_or(0.0) * 299792458.0 / f;
        const std::array<double, 4> derivs{accel, jerk, cli.snap, cli.crackle};
        items.push_back(
            {.label     = fmt::format("f {:.6f}", f),
             .target    = cli.circular
                              ? loki::search::refold_circular_target(
                                    f, derivs, setup.propagator_significance)
                              : loki::search::refold_taylor_target(f, derivs),
             .candidate = std::nullopt});
    }
    if (items.empty()) {
        throw std::invalid_argument(
            "nothing to refold: give --f, or --result with --id or --top");
    }

    // The series, preprocessed as the search did, over the searched span.
    loki::io::ReadOptions read_opts;
    read_opts.preprocess = false;
    read_opts.nthreads   = nthreads;
    auto ts              = loki::io::TimeSeries::read(cli.series, read_opts);
    if (setup.preprocess) {
        static_cast<void>(
            ts.preprocess(setup.options, loki::Exec::cpu(nthreads)));
    }
    const double tsamp = ts.get_dt();
    auto nsamps        = ts.get_nsamps();
    double t_ref       = static_cast<double>(nsamps) * tsamp / 2.0;
    if (!cli.result.empty()) {
        if (summary.tsamp > 0.0 &&
            std::abs(summary.tsamp - tsamp) > 1e-9 * tsamp) {
            throw std::invalid_argument(fmt::format(
                "the series has tsamp {} s but the result file {} s", tsamp,
                summary.tsamp));
        }
        if (summary.nsamps > nsamps) {
            throw std::invalid_argument(fmt::format(
                "the result file searched {} samples but the series has {}",
                summary.nsamps, nsamps));
        }
        if (summary.nsamps > 0) {
            nsamps = summary.nsamps;
        }
        t_ref = summary.reference_time;
    }

    std::vector<loki::search::RefoldResult> results;
    results.reserve(items.size());
    const auto e = std::span<const float>(ts.get_ts_e()).first(nsamps);
    const auto v = std::span<const float>(ts.get_ts_v()).first(nsamps);
    for (const auto& item : items) {
        results.push_back(
            loki::search::refold(e, v, tsamp, t_ref, item.target, cli.options));
    }
    loki::cli::Console::print(
        cli.json ? refold_json(cli, nsamps, tsamp, t_ref, setup, items, results)
                 : refold_text(cli, nsamps, ts.get_nsamps(), tsamp, t_ref,
                               setup, items, results));
    return 0;
}

// ---------------------------------------------------------------------------
// loki publish
// ---------------------------------------------------------------------------

struct PublishCli {
    std::filesystem::path file;
    std::filesystem::path output;
    std::filesystem::path config;
    std::filesystem::path input;
    bool keep_spool{false};
    bool force{false}; ///< Replace an existing output.
    // Candidate options; unset keeps the search's.
    std::optional<double> memory_gb;
    std::optional<bool> cluster;
    std::optional<std::uint64_t> max_candidates;
    std::optional<std::uint64_t> max_per_cell;
    std::optional<bool> retain_detections;
    std::optional<std::string> tempdir;
    std::optional<double> snr_min;
    // Observation metadata overrides.
    std::optional<double> dm;
    std::optional<double> start_mjd;
    std::optional<bool> barycentric;
};

/// What a search left to publish: a spool kept after a failed
/// post-processing, or the raw file of a `--no-candidates` search.
struct PublishSource {
    std::string algorithm; ///< "ffa" or "ep".
    bool spool{false};     ///< A `*_spool.h5`: removed once published.
    std::uint64_t nsamps{};
    double tsamp{};
    std::optional<std::string> config_effective;
    std::optional<std::string> config_toml;
    std::optional<std::string> command_line;
    /// The options a search that reduced its EP runs used (its first
    /// reduced run's `reduced_*` attributes; publish must match them).
    std::optional<loki::search::CandidateOptions> reduced;
    std::optional<double> reduced_snr_min;
};

PublishSource read_publish_source(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    if (file.hasAttribute("schema_version")) {
        throw std::invalid_argument(fmt::format(
            "'{}' is already a published result file", path.string()));
    }
    PublishSource src;
    if (file.hasAttribute("ffa_version")) {
        src.algorithm = "ffa";
    } else if (file.hasAttribute("ep_sweep_version")) {
        src.algorithm = "ep";
    } else {
        throw std::invalid_argument(fmt::format(
            "'{}' is not a loki FFA or EP search spool", path.string()));
    }
    const auto name = path.filename().string();
    src.spool       = name.ends_with("_spool.h5");
    const auto text = [&](const char* key) -> std::optional<std::string> {
        if (!file.hasAttribute(key)) {
            return std::nullopt;
        }
        auto value = file.getAttribute(key).read<std::string>();
        return value.empty() ? std::nullopt : std::optional(std::move(value));
    };
    src.config_effective = text("config_effective");
    src.config_toml      = text("config_toml");
    src.command_line     = text("command_line");
    file.getAttribute("nsamps").read(src.nsamps);
    file.getAttribute("tsamp").read(src.tsamp);
    if (src.algorithm == "ep") {
        if (const auto basis = text("poly_basis");
            basis && *basis != "taylor") {
            throw std::invalid_argument(fmt::format(
                "'{}' is a {} search; candidates support the Taylor basis "
                "only",
                path.string(), *basis));
        }
        if (!file.exist("chunks")) {
            return src;
        }
        const auto chunks = file.getGroup("chunks");
        for (const auto& chunk : chunks.listObjectNames()) {
            const auto group = chunks.getGroup(chunk);
            if (!group.exist("runs")) {
                continue;
            }
            const auto runs = group.getGroup("runs");
            for (const auto& run_name : runs.listObjectNames()) {
                const auto run = runs.getGroup(run_name);
                if (!run.hasAttribute("reduced_max_per_cell")) {
                    continue;
                }
                const auto f64 = [&](const char* key) {
                    return run.getAttribute(key).read<double>();
                };
                loki::search::CandidateOptions options;
                options.max_per_cell = run.getAttribute("reduced_max_per_cell")
                                           .read<std::uint64_t>();
                options.cell_bins    = f64("reduced_cell_bins");
                options.cell_log_width  = f64("reduced_cell_log_width");
                options.dominance_kappa = f64("reduced_dominance_kappa");
                options.radius          = f64("reduced_radius");
                options.phase_radius    = f64("reduced_phase_radius");
                src.reduced             = options;
                src.reduced_snr_min     = f64("reduced_snr_min");
                return src;
            }
        }
    }
    return src;
}

/// The stage timings a search recorded in its own `/run_stats` (the raw
/// file of a `--no-candidates` search has them; a spool has none).
std::vector<std::pair<std::string, double>>
recorded_timings(const std::filesystem::path& file) {
    std::vector<std::pair<std::string, double>> timings;
    constexpr std::string_view kPrefix = "timing_";
    constexpr std::string_view kSuffix = "_s";
    try {
        const HighFive::File h5(file.string(), HighFive::File::ReadOnly);
        if (!h5.exist("run_stats")) {
            return timings;
        }
        const auto group = h5.getGroup("run_stats");
        for (const auto& name : group.listAttributeNames()) {
            if (name.size() > kPrefix.size() + kSuffix.size() &&
                name.starts_with(kPrefix) && name.ends_with(kSuffix)) {
                timings.emplace_back(
                    name.substr(kPrefix.size(),
                                name.size() - kPrefix.size() - kSuffix.size()),
                    group.getAttribute(name).read<double>());
            }
        }
    } catch (const std::exception&) {
        // The search's timings are optional: publish records its own.
        timings.clear();
    }
    return timings;
}

/// The product of @p cli: `-o`, else the spool's name with `_results.h5`.
std::filesystem::path publish_output(const PublishCli& cli,
                                     const PublishSource& src) {
    auto output = cli.output;
    if (output.empty()) {
        const auto name          = cli.file.filename().string();
        const std::string suffix = fmt::format("_{}_spool.h5", src.algorithm);
        if (!name.ends_with(suffix)) {
            throw std::invalid_argument(fmt::format(
                "pass the result file to write with -o ('{}' is not a "
                "'*{}')",
                cli.file.string(), suffix));
        }
        output = cli.file.parent_path() /
                 (name.substr(0, name.size() - suffix.size()) +
                  fmt::format("_{}_results.h5", src.algorithm));
    }
    if (std::filesystem::exists(output)) {
        std::error_code error;
        if (std::filesystem::equivalent(output, cli.file, error)) {
            throw std::invalid_argument("the output is the file to publish");
        }
        if (!cli.force) {
            throw std::invalid_argument(fmt::format(
                "'{}' already exists; replace it with --force or choose "
                "another path with -o",
                output.string()));
        }
    }
    return output;
}

/// The observation metadata of the searched series, as the search wrote it:
/// the series header, then the configuration's [input] overrides, then
/// @p cli's. @p found tells whether the series was found.
template <typename TomlConfig>
loki::io::ObservationMetadata publish_observation(const PublishCli& cli,
                                                  const PublishSource& src,
                                                  const TomlConfig& cfg,
                                                  bool& found) {
    loki::io::ObservationMetadata observation;
    const std::filesystem::path series =
        cli.input.empty() ? std::filesystem::path(cfg.timeseries_path)
                          : cli.input;
    if (!series.empty() && std::filesystem::exists(series)) {
        const auto info = loki::io::TimeSeries::read_info(series);
        if (std::abs(info.dt - src.tsamp) > 1e-9 * src.tsamp ||
            info.nsamps < src.nsamps) {
            throw std::invalid_argument(fmt::format(
                "'{}' ({} samples of {} s) is not the searched series ({} "
                "samples of {} s); pass that with -i",
                series.string(), info.nsamps, info.dt, src.nsamps, src.tsamp));
        }
        observation = info.observation;
        found       = true;
    } else {
        found = false;
        SPDLOG_WARN("The searched series{} was not found: the result will "
                    "lack its observation metadata (source, DM, start MJD); "
                    "pass it with -i",
                    series.empty() ? ""
                                   : fmt::format(" '{}'", series.string()));
    }
    const auto set = [](auto& field, const auto& value) {
        if (value) {
            field = value;
        }
    };
    set(observation.dm, cfg.observation.dm);
    set(observation.start_mjd, cfg.observation.start_mjd);
    set(observation.barycentric, cfg.observation.barycentric);
    set(observation.dm, cli.dm);
    set(observation.start_mjd, cli.start_mjd);
    set(observation.barycentric, cli.barycentric);
    return observation;
}

template <typename TomlConfig>
int publish_as(const PublishCli& cli, const PublishSource& src) {
    constexpr bool kEp = std::is_same_v<TomlConfig, loki::search::EPTomlConfig>;
    // The search's configuration: -c, else the one the file stores.
    std::optional<std::string> text;
    std::string source;
    if (!cli.config.empty()) {
        text   = read_text_file(cli.config);
        source = cli.config.string();
    } else if (src.config_effective) {
        text   = src.config_effective;
        source = "the stored effective configuration";
    } else if (src.config_toml) {
        text   = src.config_toml;
        source = "the stored configuration";
    }
    TomlConfig cfg;
    if (text) {
        try {
            cfg = TomlConfig::from_string(*text);
        } catch (const std::exception& e) {
            throw std::invalid_argument(fmt::format(
                "cannot read the search configuration ({}): {}\nPass the "
                "search's configuration, updated to this version, with -c",
                source, e.what()));
        }
        SPDLOG_INFO("Candidate options from {}", source);
    } else {
        SPDLOG_WARN("'{}' stores no configuration: publishing with the "
                    "default candidate options (pass the search's with -c)",
                    cli.file.string());
    }

    auto options                  = cfg.candidate_options();
    options.enabled               = true;
    std::optional<double> snr_min = cli.snr_min;
    if (src.reduced && !cli.max_per_cell) {
        // The search reduced its runs: publish exactly as it did.
        options.max_per_cell    = src.reduced->max_per_cell;
        options.cell_bins       = src.reduced->cell_bins;
        options.cell_log_width  = src.reduced->cell_log_width;
        options.dominance_kappa = src.reduced->dominance_kappa;
        options.radius          = src.reduced->radius;
        options.phase_radius    = src.reduced->phase_radius;
        if (!snr_min) {
            snr_min = src.reduced_snr_min;
        }
    }
    if (cli.memory_gb) {
        options.memory_gb = *cli.memory_gb;
    }
    if (cli.cluster) {
        options.cluster = *cli.cluster;
    }
    if (cli.max_candidates) {
        options.max_candidates = *cli.max_candidates;
    }
    if (cli.max_per_cell) {
        options.max_per_cell = *cli.max_per_cell;
    }
    if (cli.retain_detections) {
        options.retain_detections = *cli.retain_detections;
    }
    if (cli.tempdir) {
        options.temporary_directory = *cli.tempdir;
    }
    if (!kEp && cli.snr_min) {
        throw std::invalid_argument(
            "--snr-min applies to EP files; an FFA spool keeps the "
            "search's detections");
    }
    const auto output      = publish_output(cli, src);
    bool series_found      = false;
    const auto observation = publish_observation(cli, src, cfg, series_found);

    const auto start = SteadyClock::now();
    if constexpr (kEp) {
        loki::pipelines::EPFreqSweep::publish_candidates(
            cli.file, output, options, observation, snr_min);
    } else {
        loki::pipelines::FFAFreqSweep::publish_candidates(cli.file, output,
                                                          options, observation);
    }
    const double publish_s = seconds_since(start);

    // Provenance: the search's command line and configuration (with the
    // options this publish used), and the command that published it.
    std::string effective;
    if (text) {
        cfg.candidates = options;
        if (snr_min) {
            cfg.snr_min = *snr_min;
        }
        cfg.nsamps = src.nsamps;
        cfg.tsamp  = src.tsamp;
        effective  = cfg.to_toml_string();
    }
    loki::search::write_run_provenance(
        output, {.config_effective     = effective,
                 .command_line         = src.command_line.value_or(""),
                 .loki_version         = LOKI_VERSION,
                 .config_toml          = src.config_toml.value_or(""),
                 .publish_command_line = run_command_line()});

    // Run statistics: the search's timings, if it recorded any, and this
    // publish; the counts (overloads, harvests) come from the copied runs.
    // Not the peak RSS: this process only published.
    auto timings = recorded_timings(cli.file);
    timings.emplace_back("publish", publish_s);
    record_run_stats(output,
                     {.timing_s = std::move(timings), .peak_rss_gb = {}},
                     /*measure_rss=*/false);

    // Read the product back before the spool goes.
    const auto summary = loki::io::read_candidate_file_summary(output, 0);
    if (!summary.complete || summary.schema_version.empty()) {
        throw std::runtime_error(fmt::format(
            "the published '{}' did not read back complete; '{}' is kept",
            output.string(), cli.file.string()));
    }
    SPDLOG_INFO("Published {} candidates from '{}' to '{}' in {:.1f} s",
                summary.n_rows, cli.file.string(), output.string(), publish_s);
    if (kEp) {
        print_ep_summary(output);
    }
    // A product published without the search's configuration or series
    // cannot be completed later: keep the spool to publish again.
    if (src.spool && !cli.keep_spool && (!text || !series_found)) {
        SPDLOG_WARN("Kept the spool '{}': the result was published without "
                    "the search's {}; publish again with {} --force for a "
                    "complete result",
                    cli.file.string(), text ? "series" : "configuration",
                    text ? "-i" : "-c");
    } else if (src.spool && !cli.keep_spool) {
        std::filesystem::remove(cli.file);
        SPDLOG_INFO("Removed the spool '{}' (--keep-spool keeps it)",
                    cli.file.string());
    }
    return 0;
}

int run_publish(const PublishCli& cli) {
    const auto src = read_publish_source(cli.file);
    return src.algorithm == "ep"
               ? publish_as<loki::search::EPTomlConfig>(cli, src)
               : publish_as<loki::search::FFATomlConfig>(cli, src);
}

// ---------------------------------------------------------------------------
// loki sift
// ---------------------------------------------------------------------------

struct SiftCli {
    std::vector<std::string> patterns;
    loki::io::SiftOptions options;
    std::size_t top{50};
    std::filesystem::path csv;
    bool json{false};
    bool members{false};
};

/// Whether @p name matches @p pattern ('*' any run, '?' one character).
bool wildcard_match(std::string_view pattern, std::string_view name) {
    std::size_t p     = 0;
    std::size_t n     = 0;
    std::size_t star  = std::string_view::npos;
    std::size_t match = 0;
    while (n < name.size()) {
        if (p < pattern.size() &&
            (pattern[p] == '?' || pattern[p] == name[n])) {
            ++p;
            ++n;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star  = p++;
            match = n;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            n = ++match;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

/// The files of @p patterns: a path as given, or a quoted pattern with '*'
/// or '?' in its file name (not in its directories), sorted.
std::vector<std::filesystem::path>
expand_patterns(const std::vector<std::string>& patterns) {
    std::vector<std::filesystem::path> files;
    for (const auto& pattern : patterns) {
        const std::filesystem::path path(pattern);
        const auto name = path.filename().string();
        if (name.find_first_of("*?") == std::string::npos) {
            files.push_back(path);
            continue;
        }
        const auto dir = path.parent_path().empty() ? std::filesystem::path(".")
                                                    : path.parent_path();
        std::vector<std::filesystem::path> matched;
        if (std::filesystem::is_directory(dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.is_regular_file() &&
                    wildcard_match(name, entry.path().filename().string())) {
                    matched.push_back(path.parent_path() /
                                      entry.path().filename());
                }
            }
        }
        if (matched.empty()) {
            throw std::invalid_argument(
                fmt::format("no file matches '{}'", pattern));
        }
        std::ranges::sort(matched);
        files.insert(files.end(), matched.begin(), matched.end());
    }
    return files;
}

std::string sift_harmonic(const loki::io::SiftGroup& g) {
    return g.harmonic_of < 0
               ? std::string()
               : fmt::format("{}/{} of #{}", g.harmonic_numerator,
                             g.harmonic_denominator, g.harmonic_of);
}

std::string sift_segments(const loki::io::SiftGroup& g) {
    std::string text;
    for (const auto segment : g.segments) {
        if (!text.empty()) {
            text += ' ';
        }
        text += std::to_string(segment);
    }
    return text;
}

std::string sift_json(const SiftCli& cli, const loki::io::SiftResult& r) {
    std::string out = "{\n  \"files\": [";
    for (std::size_t i = 0; i < r.files.size(); ++i) {
        const auto& f = r.files[i];
        out += fmt::format(
            "{}\n    {{\"path\": {}, \"dm\": {}, \"start_mjd\": {}, "
            "\"segment\": {}}}",
            i == 0 ? "" : ",", json_string(f.path.string()),
            json_optional(f.dm), json_optional(f.start_mjd), f.segment);
    }
    out += fmt::format("\n  ],\n  \"n_candidates\": {},\n  \"n_segments\": "
                       "{},\n  \"groups\": [",
                       r.n_candidates, r.n_segments);
    for (std::size_t i = 0; i < r.groups.size(); ++i) {
        const auto& g = r.groups[i];
        const auto& b = g.members.front();
        std::string curve;
        for (const auto& [dm, snr] : g.dm_curve) {
            curve += fmt::format("{}[{}, {}]", curve.empty() ? "" : ", ",
                                 json_number(dm), json_number(snr));
        }
        std::string segments;
        for (const auto s : g.segments) {
            if (!segments.empty()) {
                segments += ", ";
            }
            segments += std::to_string(s);
        }
        out += fmt::format(
            "{}\n    {{\"index\": {}, \"snr\": {}, \"freq\": {}, \"fdot\": {}, "
            "\"duty_cycle\": {}, \"file\": {}, \"id\": {}, \"dm\": {}, "
            "\"n_members\": {}, \"segments\": [{}], \"dm_curve\": [{}], "
            "\"edge_ratio\": {}, \"extrapolation_bins\": {}, "
            "\"harmonic_of\": {}, \"harmonic_numerator\": {}, "
            "\"harmonic_denominator\": {}, \"harmonic_offset_bins\": {}",
            i == 0 ? "" : ",", i, json_number(b.snr), json_number(b.freq),
            json_number(b.fdot), json_number(b.duty_cycle), b.file, b.id,
            json_optional(r.files[b.file].dm), g.members.size(), segments,
            curve, json_optional(g.edge_ratio),
            json_optional(g.extrapolation_bins), g.harmonic_of,
            g.harmonic_numerator, g.harmonic_denominator,
            json_optional(g.harmonic_offset_bins));
        if (cli.members) {
            out += ", \"members\": [";
            for (std::size_t m = 0; m < g.members.size(); ++m) {
                const auto& x = g.members[m];
                out += fmt::format(
                    "{}{{\"file\": {}, \"id\": {}, \"snr\": {}, \"freq\": {}, "
                    "\"fdot\": {}}}",
                    m == 0 ? "" : ", ", x.file, x.id, json_number(x.snr),
                    json_number(x.freq), json_number(x.fdot));
            }
            out += "]";
        }
        out += "}";
    }
    out += "\n  ]\n}\n";
    return out;
}

void write_sift_csv(const std::filesystem::path& path,
                    const loki::io::SiftResult& r) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            fmt::format("cannot write '{}'", path.string()));
    }
    out << "index,snr,freq,fdot,duty_cycle,dm,segment,file,id,n_members,"
           "n_segments,segments,dm_lo,dm_hi,edge_ratio,extrapolation_bins,"
           "harmonic_of,harmonic_numerator,harmonic_denominator,"
           "harmonic_offset_bins,dm_curve\n";
    const auto opt = [](const std::optional<double>& v) {
        return v ? fmt::format("{:.17g}", *v) : std::string();
    };
    for (std::size_t i = 0; i < r.groups.size(); ++i) {
        const auto& g = r.groups[i];
        const auto& b = g.members.front();
        std::string curve;
        for (const auto& [dm, snr] : g.dm_curve) {
            curve +=
                fmt::format("{}{:g}:{:.2f}", curve.empty() ? "" : " ", dm, snr);
        }
        out << fmt::format(
            "{},{:.9g},{:.17g},{:.17g},{:.6g},{},{},\"{}\",{},{},{},{},{},{},"
            "{},{},{},{},{},{},{}\n",
            i, b.snr, b.freq, b.fdot, b.duty_cycle, opt(r.files[b.file].dm),
            r.files[b.file].segment, r.files[b.file].path.string(), b.id,
            g.members.size(), g.segments.size(), sift_segments(g),
            g.dm_curve.empty() ? ""
                               : fmt::format("{:g}", g.dm_curve.front().first),
            g.dm_curve.empty() ? ""
                               : fmt::format("{:g}", g.dm_curve.back().first),
            opt(g.edge_ratio), opt(g.extrapolation_bins), g.harmonic_of,
            g.harmonic_numerator, g.harmonic_denominator,
            opt(g.harmonic_offset_bins), curve);
    }
}

std::string sift_text(const SiftCli& cli, const loki::io::SiftResult& r) {
    std::string out = fmt::format(
        "{} candidates (S/N >= {:g}) from {} files in {} segments: {} groups\n",
        r.n_candidates, cli.options.snr_min, r.files.size(), r.n_segments,
        r.groups.size());
    out +=
        fmt::format("{:>4} {:>7} {:>14} {:>10} {:>12} {:>8} {:>15} {:>5} "
                    "{:>5} {:>6} {:>7}  {}\n",
                    "#", "S/N", "f (Hz)", "P (ms)", "fdot (Hz/s)", "DM",
                    "DM range", "segs", "duty", "edge", "extrap", "harmonic");
    for (std::size_t i = 0; i < std::min(cli.top, r.groups.size()); ++i) {
        const auto& g  = r.groups[i];
        const auto& b  = g.members.front();
        const auto& dm = r.files[b.file].dm;
        out += fmt::format(
            "{:>4} {:>7.2f} {:>14.8f} {:>10.5f} {:>12.4e} {:>8} {:>15} {:>5} "
            "{:>5.3f} {:>6} {:>7}  {}\n",
            i, b.snr, b.freq, 1e3 / b.freq, b.fdot,
            dm ? fmt::format("{:.2f}", *dm) : "-",
            g.dm_curve.empty()
                ? "-"
                : fmt::format("{:.2f}-{:.2f}", g.dm_curve.front().first,
                              g.dm_curve.back().first),
            g.segments.size(), b.duty_cycle,
            g.edge_ratio ? fmt::format("{:.2f}", *g.edge_ratio) : "-",
            g.extrapolation_bins ? fmt::format("{:.1f}", *g.extrapolation_bins)
                                 : "-",
            sift_harmonic(g));
    }
    return out;
}

int run_sift(const SiftCli& cli) {
    const auto files  = expand_patterns(cli.patterns);
    const auto start  = SteadyClock::now();
    const auto result = loki::io::sift(files, cli.options);
    if (!cli.csv.empty()) {
        write_sift_csv(cli.csv, result);
    }
    if (cli.json) {
        std::cout << sift_json(cli, result);
    } else {
        std::cout << sift_text(cli, result);
    }
    SPDLOG_INFO("Sifted {} files in {:.1f} s", files.size(),
                seconds_since(start));
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

    bool ffa_dry_run = false;

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
        "Largest numerator and denominator of a harmonic ratio (default: 16)");
    grp_candidates->add_option("--harmonic-phase-distance",
                               ffa_cfg.candidates.harmonic_phase_distance,
                               "Harmonic track tolerance in pulse widths");
    grp_candidates->add_option(
        "--harmonic-snr-distance", ffa_cfg.candidates.harmonic_snr_distance,
        "How far a harmonic's S/N may exceed its parent's");
    grp_candidates->add_option(
        "--harmonic-doppler-tolerance",
        ffa_cfg.candidates.harmonic_doppler_tolerance,
        "Fractional frequency within which a bright parent takes a harmonic "
        "without a track test (bright binaries, e.g. 3e-4); 0 disables "
        "(default: 0)");
    grp_candidates->add_option("--harmonic-doppler-min-snr",
                               ffa_cfg.candidates.harmonic_doppler_min_snr,
                               "Minimum parent S/N for that (default: 20)");
    grp_candidates->add_option(
        "--max-per-cell", ffa_cfg.candidates.max_per_cell,
        "Strongest trials kept per frequency cell of each run or region "
        "before grouping; zero "
        "keeps all");
    grp_candidates->add_option("--cell-bins", ffa_cfg.candidates.cell_bins,
                               "Frequency cell width in Fourier bins (1/tobs)");
    grp_candidates->add_option("--cell-log-width",
                               ffa_cfg.candidates.cell_log_width,
                               "Frequency cell width in ln(frequency)");
    grp_candidates->add_option(
        "--dominance-kappa", ffa_cfg.candidates.dominance_kappa,
        "Absorb trials a stronger trial of the region explains; zero "
        "disables");
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

    auto* grp_perf = ffa->add_option_group("Performance & Limits");
    grp_perf->add_option(
        "--threads", ffa_cfg.nthreads,
        "OpenMP threads (0 = hardware concurrency, default: 0)");
    grp_perf->add_option(
        "--memory-gb", ffa_cfg.max_process_memory_gb,
        "Memory cap in GB of 2^30 bytes (as Slurm's --mem=16G): total "
        "process RSS on CPU, device memory on CUDA");
    grp_perf->add_option("--region-growth-factor", ffa_cfg.region_growth_factor,
                         "Period growth factor between search regions "
                         "(default: 1.5)");
    grp_perf->add_option("--nbins-max", ffa_cfg.nbins_max,
                         "Maximum allowed folding bins (default: 1024)");
    grp_perf->add_option("--nbins-min-lossy-bf", ffa_cfg.nbins_min_lossy_bf,
                         "Minimum bins before lossy brute fold (default: 64)");
    grp_perf->add_option(
        "--max-passing-candidates", ffa_cfg.max_passing_candidates,
        "Maximum candidate buffer capacity (default: 4194304)");
    grp_perf->add_option("--bseg-brute", ffa_cfg.bseg_brute,
                         "Minimum brute-force fold segment length in samples");

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

    bool ep_dry_run       = false;
    bool ep_plan_only     = false;
    bool ep_no_plan_cache = false;
    std::string ep_plan_cache_path;
    std::string ep_backend_name;
    std::vector<SizeType> ep_ref_segs;
    std::optional<bool> ep_harvest;
    std::optional<bool> ep_impulsive_veto;
    std::optional<float> ep_harvest_min_snr;
    std::optional<float> ep_harvest_offset;
    std::vector<std::string> ep_pulsar_mask;

    std::string ep_preproc_method;
    auto* ep_grp_io = ep->add_option_group("Input/Output Options");
    ep_grp_io->add_option("-i,--input", ep_cfg.timeseries_path,
                          "Input timeseries (.tim or .dat)");
    ep_grp_io->add_option("-o,--outdir", ep_cfg.outdir,
                          "Output directory for the results file");
    ep_grp_io->add_option("-p,--prefix", ep_cfg.prefix,
                          "Prefix for the results file");
    ep_grp_io->add_option("--dm", ep_cfg.observation.dm,
                          "Override single-trial dispersion measure");
    ep_grp_io->add_option("--start-mjd", ep_cfg.observation.start_mjd,
                          "Override MJD of the first sample");
    ep_grp_io->add_option("--barycentric", ep_cfg.observation.barycentric,
                          "Override barycentric/topocentric timing frame");
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
                          "Sample count, for --plan-only without an input "
                          "(otherwise read from the input's header)");
    ep_grp_io->add_option("--tsamp", ep_cfg.tsamp,
                          "Sample interval in seconds, for --plan-only "
                          "without an input");
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
        "Largest numerator and denominator of a harmonic ratio (default: 16)");
    ep_grp_candidates->add_option("--harmonic-phase-distance",
                                  ep_cfg.candidates.harmonic_phase_distance,
                                  "Harmonic track tolerance in pulse widths");
    ep_grp_candidates->add_option(
        "--harmonic-snr-distance", ep_cfg.candidates.harmonic_snr_distance,
        "How far a harmonic's S/N may exceed its parent's");
    ep_grp_candidates->add_option(
        "--harmonic-doppler-tolerance",
        ep_cfg.candidates.harmonic_doppler_tolerance,
        "Fractional frequency within which a bright parent takes a harmonic "
        "without a track test (bright binaries, e.g. 3e-4); 0 disables "
        "(default: 0)");
    ep_grp_candidates->add_option("--harmonic-doppler-min-snr",
                                  ep_cfg.candidates.harmonic_doppler_min_snr,
                                  "Minimum parent S/N for that (default: 20)");
    ep_grp_candidates->add_option(
        "--max-per-cell", ep_cfg.candidates.max_per_cell,
        "Strongest trials kept per frequency cell of each run or region "
        "before grouping; zero "
        "keeps all");
    ep_grp_candidates->add_option(
        "--cell-bins", ep_cfg.candidates.cell_bins,
        "Frequency cell width in Fourier bins (1/tobs)");
    ep_grp_candidates->add_option("--cell-log-width",
                                  ep_cfg.candidates.cell_log_width,
                                  "Frequency cell width in ln(frequency)");
    ep_grp_candidates->add_option(
        "--dominance-kappa", ep_cfg.candidates.dominance_kappa,
        "Absorb trials a stronger trial of the run explains; zero disables");
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
    ep_grp_pruning->add_option(
        "--overflow-keep-fraction", ep_cfg.overflow_keep_fraction,
        "Fraction of the stage's survivors and batch a full world tree keeps "
        "when a batch overflows it, in (0, 1] (default: 0.5)");

    auto* ep_grp_rfi = ep->add_option_group(
        "RFI control (sets [ep.rfi], see the config template; the "
        "impulsive veto is CPU only)");
    ep_grp_rfi
        ->add_option("--pulsar-mask", ep_pulsar_mask,
                     "Frequency window F_LO:F_HI in Hz masked at every "
                     "acceleration; repeat for more (replaces "
                     "ep.rfi.pulsar_mask)")
        ->take_all();
    ep_grp_rfi->add_flag("--harvest,!--no-harvest", ep_harvest,
                         "Early harvesting of leaves far above the pruning "
                         "threshold (ep.rfi.harvest)");
    ep_grp_rfi->add_option("--harvest-min-snr", ep_harvest_min_snr,
                           "Lowest harvest threshold (ep.rfi.harvest_min_snr, "
                           "default: 40)");
    ep_grp_rfi->add_option("--harvest-offset", ep_harvest_offset,
                           "Harvest threshold above the pruning threshold "
                           "(ep.rfi.harvest_offset, default: 10)");
    ep_grp_rfi->add_flag("--impulsive-veto,!--no-impulsive-veto",
                         ep_impulsive_veto,
                         "Impulsive-segment veto (ep.rfi.impulsive_veto)");

    auto* ep_grp_perf = ep->add_option_group("Performance & Limits");
    ep_grp_perf->add_option(
        "--threads", ep_cfg.nthreads,
        "OpenMP threads (0 = hardware concurrency, default: 0)");
    ep_grp_perf->add_option(
        "--memory-gb", ep_cfg.max_process_memory_gb,
        "Memory cap in GB of 2^30 bytes (as Slurm's --mem=16G): total "
        "process RSS on CPU, device memory on CUDA");
    ep_grp_perf
        ->add_option("--max-host-memory-gb", ep_cfg.max_host_memory_gb,
                     "Host memory cap in GB of a CUDA search "
                     "(performance.max_host_memory_gb): it stops before "
                     "loading when its forecast is above it. Ignored on CPU")
        ->check(CLI::PositiveNumber);
    ep_grp_perf->add_option("--region-growth-factor",
                            ep_cfg.region_growth_factor,
                            "Period growth factor between search regions "
                            "(default: 1.5)");
    ep_grp_perf->add_option("--nbins-max", ep_cfg.nbins_max,
                            "Maximum allowed folding bins (default: 1024)");
    ep_grp_perf->add_option(
        "--nbins-min-lossy-bf", ep_cfg.nbins_min_lossy_bf,
        "Minimum bins before lossy brute fold (default: 64)");
    ep_grp_perf->add_option(
        "--max-passing-candidates", ep_cfg.max_passing_candidates,
        "Maximum candidate buffer capacity (default: 4194304)");
    ep_grp_perf->add_option(
        "--nsegments", ep_cfg.nsegments,
        "Segments the pruning starts from, a power of 2 (default: 128)");
    ep_grp_perf->add_option(
        "--bseg-brute", ep_cfg.bseg_brute,
        "Minimum brute-force fold segment length in samples");
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
                 "reads no samples");

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

    // =========================================================================
    // Subcommand: refold
    // =========================================================================
    auto* refold = app.add_subcommand(
        "refold",
        "Refold a series at a candidate's parameters and scan a local f "
        "(and fdot) grid: S/N with the search's preprocessing and phase "
        "model. A rejection test; noise also refolds near its search S/N");
    RefoldCli refold_cli;
    refold
        ->add_option("series", refold_cli.series,
                     "The searched timeseries (.tim or .dat)")
        ->required()
        ->check(CLI::ExistingFile);
    auto* grp_refold_target = refold->add_option_group("Target");
    grp_refold_target
        ->add_option("--result", refold_cli.result,
                     "Result file of the search: candidates by --id or "
                     "--top, its configuration's preprocessing, searched "
                     "span and reference time")
        ->check(CLI::ExistingFile);
    grp_refold_target->add_option(
        "--id", refold_cli.ids,
        "Candidate id(s) in --result (EP: run << 32 | leaf)");
    grp_refold_target->add_option("--top", refold_cli.top,
                                  "The strongest N candidates of --result");
    grp_refold_target->add_option("--f", refold_cli.freqs,
                                  "Frequency (Hz) at the series midpoint; "
                                  "several share the derivatives below");
    auto* opt_refold_fd = grp_refold_target->add_option(
        "--fd", refold_cli.fd, "Frequency derivative (Hz/s)");
    grp_refold_target
        ->add_option("--acc", refold_cli.acc, "Acceleration (m/s^2)")
        ->excludes(opt_refold_fd);
    auto* opt_refold_fdd = grp_refold_target->add_option(
        "--fdd", refold_cli.fdd, "Second frequency derivative (Hz/s^2)");
    grp_refold_target->add_option("--jerk", refold_cli.jerk, "Jerk (m/s^3)")
        ->excludes(opt_refold_fdd);
    grp_refold_target->add_option("--snap", refold_cli.snap, "Snap (m/s^4)");
    grp_refold_target->add_option("--crackle", refold_cli.crackle,
                                  "Crackle (m/s^5)");
    grp_refold_target->add_flag(
        "--circular", refold_cli.circular,
        "Fold --f targets as the five-parameter search does: a circular "
        "orbit when snap and accel (else crackle and jerk) imply one");
    auto* grp_refold_grid = refold->add_option_group("Grid and scoring");
    grp_refold_grid->add_option(
        "--nbins", refold_cli.options.nbins,
        "Phase bins (default: min(64, max(8, samples per period)))");
    grp_refold_grid->add_option("--nsub", refold_cli.options.nsub,
                                "Subintegrations (default: 128)");
    grp_refold_grid->add_option(
        "--df-steps", refold_cli.options.df_steps,
        "+- frequency steps (default: cover +- --df-span-t / T)");
    grp_refold_grid->add_option("--df-span-t", refold_cli.options.df_span_t,
                                "+- frequency span in units of 1/T "
                                "(default: 3)");
    grp_refold_grid->add_option(
        "--df-step-bins", refold_cli.options.df_step_bins,
        "Frequency step in units of 1/(nbins T) (default: 0.25)");
    grp_refold_grid->add_option("--fd-steps", refold_cli.options.fd_steps,
                                "+- fdot steps (default: 0, no fdot scan)");
    grp_refold_grid->add_option("--fd-span", refold_cli.options.fd_span,
                                "+- fdot span (Hz/s)");
    grp_refold_grid->add_option(
        "--split", refold_cli.options.split,
        "Also score the best point in N contiguous time parts");
    auto* grp_refold_io = refold->add_option_group("Input and output");
    grp_refold_io
        ->add_option("-c,--config", refold_cli.config,
                     "Search configuration whose [preprocessing] to apply "
                     "(default: the result file's, else the defaults)")
        ->check(CLI::ExistingFile);
    grp_refold_io->add_flag("--no-preprocess", refold_cli.no_preprocess,
                            "Fold the series as read");
    grp_refold_io->add_option("-t,--threads", refold_cli.threads,
                              "CPU threads (default: all)");
    grp_refold_io->add_flag("--profile", refold_cli.profile,
                            "Print the best profile (signal / sqrt(var))");
    grp_refold_io->add_flag("--json", refold_cli.json,
                            "Write the results as JSON to stdout");

    // =========================================================================
    // Subcommand: publish
    // =========================================================================
    auto* publish = app.add_subcommand(
        "publish", "Publish the candidates of a kept search spool (or of a "
                   "--no-candidates search) without searching again");
    PublishCli publish_cli;
    publish
        ->add_option("file", publish_cli.file,
                     "The *_ffa_spool.h5 or *_ep_spool.h5 a failed "
                     "post-processing kept, or a --no-candidates result")
        ->required()
        ->check(CLI::ExistingFile);
    auto* grp_publish_io = publish->add_option_group("Input and output");
    grp_publish_io->add_option(
        "-o,--output", publish_cli.output,
        "Result file to write (default: the spool's name with _results.h5)");
    grp_publish_io
        ->add_option("-c,--config", publish_cli.config,
                     "The search's configuration (default: the one the file "
                     "stores, else the defaults)")
        ->check(CLI::ExistingFile);
    grp_publish_io->add_option(
        "-i,--input", publish_cli.input,
        "The searched series, for its observation metadata (default: the "
        "configuration's)");
    grp_publish_io->add_flag("--keep-spool", publish_cli.keep_spool,
                             "Keep the spool after publishing it");
    grp_publish_io->add_flag("--force", publish_cli.force,
                             "Replace an existing result file (atomically, "
                             "once the new one is written)");
    grp_publish_io->add_option("--dm", publish_cli.dm,
                               "Override dispersion measure (pc cm^-3)");
    grp_publish_io->add_option("--start-mjd", publish_cli.start_mjd,
                               "Override MJD of the first input sample");
    grp_publish_io->add_option(
        "--barycentric", publish_cli.barycentric,
        "Override timing frame: true (barycentric) or false (topocentric)");
    auto* grp_publish_cand = publish->add_option_group(
        "Candidate processing (default: the search's)");
    grp_publish_cand->add_option("--memory-gb,--candidate-memory-gb",
                                 publish_cli.memory_gb,
                                 "Host memory budget for candidate processing");
    grp_publish_cand->add_option("--candidate-tempdir", publish_cli.tempdir,
                                 "Directory for bounded external sorting");
    grp_publish_cand->add_flag_function(
        "--cluster,!--no-cluster",
        [&publish_cli](std::int64_t count) { publish_cli.cluster = count > 0; },
        "Enable/disable phase-track clustering");
    grp_publish_cand->add_flag_function(
        "--retain-detections,!--no-retain-detections",
        [&publish_cli](std::int64_t count) {
            publish_cli.retain_detections = count > 0;
        },
        "Keep raw detections and membership");
    grp_publish_cand->add_option(
        "--max-candidates", publish_cli.max_candidates,
        "Maximum folding candidates; zero is unlimited");
    grp_publish_cand->add_option(
        "--max-per-cell", publish_cli.max_per_cell,
        "Strongest trials kept per frequency cell (default: as the search; "
        "an EP search that reduced its runs needs the same)");
    grp_publish_cand->add_option(
        "--snr-min", publish_cli.snr_min,
        "EP: candidate S/N minimum (default: the search's)");

    // =========================================================================
    // Subcommand: sift
    // =========================================================================
    auto* sift = app.add_subcommand(
        "sift",
        "Group the candidates of many result files (DM trials x segments) by "
        "frequency: S/N against DM, segments, harmonic relations");
    SiftCli sift_cli;
    sift->add_option("files", sift_cli.patterns,
                     "Published result files; a quoted pattern such as "
                     "'results/*.h5' is expanded")
        ->required();
    sift->add_option("--snr-min", sift_cli.options.snr_min,
                     "Ignore candidates below this S/N (default: 7)");
    sift->add_option("--tol", sift_cli.options.tolerance,
                     "Group frequencies within this fraction (default: 2e-4)");
    sift->add_option("--min-segs", sift_cli.options.min_segments,
                     "Drop groups seen in fewer segments (default: 1)");
    sift->add_option("--harmonic-order", sift_cli.options.harmonic_order_max,
                     "Largest p, q of harmonic relations between groups; 0 "
                     "disables (default: 16)");
    sift->add_option("--harmonic-bins", sift_cli.options.harmonic_bins,
                     "Harmonic relations must agree to within this many "
                     "frequency bins in a file with both (default: 2)");
    sift->add_option("-n,--top", sift_cli.top, "Groups to list (default: 50)");
    sift->add_option("--csv", sift_cli.csv, "Also write every group as CSV");
    sift->add_flag("--json", sift_cli.json,
                   "Write every group as JSON to stdout");
    sift->add_flag("--members", sift_cli.members,
                   "With --json: list each group's members (file, id) for "
                   "loki refold");

    simulate->fallthrough();
    search->fallthrough();
    inspect->fallthrough();
    refold->fallthrough();
    publish->fallthrough();
    sift->fallthrough();
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
    if (!ep_pulsar_mask.empty() || ep_harvest || ep_impulsive_veto ||
        ep_harvest_min_snr || ep_harvest_offset) {
        auto& rfi = ep_cfg.rfi ? *ep_cfg.rfi : ep_cfg.rfi.emplace();
        if (!ep_pulsar_mask.empty()) {
            try {
                rfi.pulsar_mask = parse_pulsar_mask(ep_pulsar_mask);
            } catch (const std::exception& ex) {
                loki::cli::Console::get().error(ex.what());
                return 1;
            }
        }
        if (ep_harvest) {
            rfi.harvest = *ep_harvest;
        }
        if (ep_harvest_min_snr) {
            rfi.harvest_min_snr = *ep_harvest_min_snr;
        }
        if (ep_harvest_offset) {
            rfi.harvest_offset = *ep_harvest_offset;
        }
        if (ep_impulsive_veto) {
            rfi.impulsive_veto = *ep_impulsive_veto;
        }
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

        if (refold->parsed()) {
            return run_refold(std::move(refold_cli));
        }

        if (publish->parsed()) {
            return run_publish(publish_cli);
        }

        if (sift->parsed()) {
            return run_sift(sift_cli);
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

            return run_search_ffa(ffa_cfg, config_toml, ffa_dry_run);
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

            std::string config_toml;
            std::filesystem::path config_path;
            if (!ep_config_path.empty()) {
                config_path = ep_config_path;
            } else if (cfg_arg.has_value()) {
                config_path = *cfg_arg;
            }
            if (!config_path.empty() && std::filesystem::exists(config_path)) {
                config_toml = read_text_file(config_path);
            }
            bool clamp_default_runs = false;
            // Neither the config nor the command line chose the runs.
            if (!ep_cfg.n_runs.has_value() && !ep_cfg.ref_segs.has_value()) {
                ep_cfg.n_runs      = 16;
                clamp_default_runs = true;
            }

            if (ep_plan_only) {
                return run_plan_ep(ep_cfg, clamp_default_runs);
            }
            return run_search_ep(ep_cfg, config_toml, ep_dry_run,
                                 clamp_default_runs);
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
