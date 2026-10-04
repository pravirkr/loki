#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>

#include <benchmark/benchmark.h>

#ifndef LOKI_APP_PATH
#define LOKI_APP_PATH ""
#endif

namespace {

// Frequency-only FFA baseline: dump one timeseries, then time `loki search ffa`
// once per folding mode.
// Run: loki_benchmarks --benchmark_filter=BM_FFA_Freq

constexpr int kNsamps = 1 << 23;

struct TempRoot {
    std::filesystem::path root;
    std::filesystem::path timeseries;

    TempRoot()
        : root(std::filesystem::temp_directory_path() / "loki_ffa_bench"),
          timeseries(root / "pulse.tim") {
        std::filesystem::create_directories(root);
    }

    TempRoot(const TempRoot&)            = delete;
    TempRoot& operator=(const TempRoot&) = delete;
    TempRoot(TempRoot&&)                 = delete;
    TempRoot& operator=(TempRoot&&)      = delete;

    ~TempRoot() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

[[nodiscard]] TempRoot& temp_root() {
    static TempRoot root;
    return root;
}

[[nodiscard]] std::string log_excerpt(const std::filesystem::path& log_path) {
    std::ifstream input(log_path);
    if (!input.is_open()) {
        return "log unavailable";
    }
    std::string content(std::istreambuf_iterator<char>{input},
                        std::istreambuf_iterator<char>{});
    constexpr std::size_t kMaxChars = 400;
    if (content.size() > kMaxChars) {
        content.erase(0, content.size() - kMaxChars);
    }
    for (char& ch : content) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return content;
}

void dump_timeseries_once() {
    static bool dumped = false;
    if (dumped) {
        return;
    }
    const std::string app_path = LOKI_APP_PATH;
    if (app_path.empty()) {
        throw std::runtime_error(
            "loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const auto log_path       = temp_root().root / "simulate.log";
    const std::string command = std::format(
        "\"{}\" simulate --period 0.007 --dt 6.4e-5 --nsamps {} --snr 10 "
        "--ducy 0.1 --shape gaussian --phi0 0.5 --seed 1 -o \"{}\" > \"{}\" "
        "2>&1",
        app_path, kNsamps, temp_root().timeseries.string(), log_path.string());
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error(
            std::format("loki simulate failed: {}", log_excerpt(log_path)));
    }
    dumped = true;
}

// bseg_brute == 0 keeps the library default (cost-model selector); any other
// value is forwarded as performance.bseg_brute for every chunk.
void write_search_config(const std::filesystem::path& path,
                         const std::filesystem::path& timeseries,
                         const std::filesystem::path& outdir,
                         std::string_view prefix,
                         bool use_fourier,
                         long bseg_brute = 0) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("failed to write FFA bench TOML");
    }
    // The offline script uses f_max = 500 Hz (1 / 0.002). That period is
    // shorter than nbins * tsamp (32 * 6.4e-5 s), which the region planner
    // rejects. 488 Hz is the closest round cutoff that still satisfies it.
    output << std::format("[input]\n"
                          "timeseries = \"{}\"\n"
                          "preprocess = false\n"
                          "\n"
                          "[search]\n"
                          "f_min = 1.0\n"
                          "f_max = 488.0\n"
                          "acc_min = 0.0\n"
                          "acc_max = 0.0\n"
                          "nbins = 32\n"
                          "eta = 1.0\n"
                          "ducy_max = 0.5\n"
                          "wtsp = 1.2\n"
                          "snr_min = 8.0\n"
                          "use_fourier = {}\n"
                          "\n"
                          "[performance]\n"
                          "nthreads = 8\n"
                          "max_process_memory_gb = 16.0\n"
                          "octave_scale = 1.5\n"
                          "nbins_max = 1024\n"
                          "nbins_min_lossy_bf = 32\n"
                          "{}"
                          "\n"
                          "[output]\n"
                          "outdir = \"{}\"\n"
                          "prefix = \"{}\"\n",
                          timeseries.string(), use_fourier ? "true" : "false",
                          bseg_brute > 0
                              ? std::format("bseg_brute = {}\n", bseg_brute)
                              : std::string{},
                          outdir.string(), prefix);
}

struct SweepTimers {
    double brutefold_s{0.0}; // includes table build
    double brute_table_s{0.0};
    double merge_s{0.0};
};

// Sums the per-chunk "FFA Chunk detail" lines emitted by the frequency sweep.
[[nodiscard]] SweepTimers parse_sweep_timers(const std::filesystem::path& log) {
    SweepTimers total;
    std::ifstream input(log);
    static const std::regex kChunk(
        R"(FFA Chunk detail: .*brutefold_s=([0-9.]+) brute_table_s=([0-9.]+) ffa_s=([0-9.]+))");
    std::string line;
    while (std::getline(input, line)) {
        std::smatch match;
        if (std::regex_search(line, match, kChunk)) {
            total.brutefold_s += std::stod(match[1]);
            total.brute_table_s += std::stod(match[2]);
            total.merge_s += std::stod(match[3]);
        }
    }
    return total;
}

void run_freq_search(benchmark::State& state,
                     bool use_fourier,
                     long bseg_brute = 0) {
    try {
        dump_timeseries_once();
    } catch (const std::exception& ex) {
        state.SkipWithError(ex.what());
        return;
    }

    const std::string tag =
        std::string(use_fourier ? "fourier" : "time") +
        (bseg_brute > 0 ? std::format("_bseg{}", bseg_brute) : std::string{});
    const auto case_dir    = temp_root().root / tag;
    const auto outdir      = case_dir / "out";
    const auto config_path = case_dir / "config.toml";
    const auto log_path    = case_dir / "search.log";
    std::filesystem::create_directories(outdir);
    try {
        write_search_config(config_path, temp_root().timeseries, outdir, tag,
                            use_fourier, bseg_brute);
    } catch (const std::exception& ex) {
        state.SkipWithError(ex.what());
        return;
    }

    const std::string app_path = LOKI_APP_PATH;
    const std::string command =
        std::format("\"{}\" search ffa --config \"{}\" > \"{}\" 2>&1", app_path,
                    config_path.string(), log_path.string());
    for (auto _ : state) {
        if (std::system(command.c_str()) != 0) {
            state.SkipWithError(
                std::format("loki search ffa failed: {}", log_excerpt(log_path))
                    .c_str());
            break;
        }
    }
    const auto timers = parse_sweep_timers(log_path);
    state.counters["brutefold_s"]   = timers.brutefold_s;
    state.counters["brute_table_s"] = timers.brute_table_s;
    state.counters["merge_s"]       = timers.merge_s;
    const auto denom                = timers.brutefold_s + timers.merge_s;
    state.counters["brute_share"] =
        denom > 0.0 ? timers.brutefold_s / denom : 0.0;
}

void ffa_freq_time(benchmark::State& state) { run_freq_search(state, false); }

void ffa_freq_fourier(benchmark::State& state) { run_freq_search(state, true); }

BENCHMARK(ffa_freq_time)
    ->Name("BM_FFA_Freq/time")
    ->Iterations(1)
    ->Repetitions(1)
    ->UseRealTime()
    ->Unit(benchmark::kSecond);

BENCHMARK(ffa_freq_fourier)
    ->Name("BM_FFA_Freq/fourier")
    ->Iterations(1)
    ->Repetitions(1)
    ->UseRealTime()
    ->Unit(benchmark::kSecond);

// bseg_brute sweep: Arg(0) is the library default, other values override
// performance.bseg_brute for every chunk. Per-chunk detail (brute / table /
// merge time) is reported as counters; see also bench/bseg_brute_sweep.py for
// the per-chunk breakdown.
void ffa_freq_time_bseg(benchmark::State& state) {
    run_freq_search(state, false, state.range(0));
}
void ffa_freq_fourier_bseg(benchmark::State& state) {
    run_freq_search(state, true, state.range(0));
}

BENCHMARK(ffa_freq_time_bseg)
    ->Name("BM_FFA_Freq_Bseg/time")
    ->Arg(0)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Iterations(1)
    ->Repetitions(1)
    ->UseRealTime()
    ->Unit(benchmark::kSecond);

BENCHMARK(ffa_freq_fourier_bseg)
    ->Name("BM_FFA_Freq_Bseg/fourier")
    ->Arg(0)
    ->Arg(256)
    ->Arg(1024)
    ->Iterations(1)
    ->Repetitions(1)
    ->UseRealTime()
    ->Unit(benchmark::kSecond);

} // namespace
