#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/io/timeseries.hpp"
#include "loki/simulation/modulate.hpp"
#include "loki/simulation/pulse.hpp"

#ifndef LOKI_APP_PATH
#define LOKI_APP_PATH ""
#endif

namespace {

constexpr loki::SizeType kNsamps = loki::SizeType{1} << 23U;
constexpr double kPeriod         = 0.01;
constexpr double kDt             = 64e-6;
constexpr double kTrueFreq       = 100.0;
constexpr double kTrueAccel      = 200.0;
constexpr double kSearchSnrMin   = 6.0;
// Injected folded S/N is 10. Recovery must stay at or above this floor.
constexpr double kRecoverSnrMin = 8.0;
// One frequency step at this length is ~3e-5 Hz. One acceleration step is ~2.6
// m/s^2.
constexpr double kFreqTol  = 1.0e-4;
constexpr double kAccelTol = 5.0;

[[nodiscard]] loki::io::TimeSeries
to_unit_variance(loki::io::TimeSeries series) {
    const float noise_std = std::sqrt(series.get_ts_v().front());
    std::vector<float> ts_e(series.get_ts_e().begin(), series.get_ts_e().end());
    for (float& sample : ts_e) {
        sample /= noise_std;
    }
    std::vector<float> ts_v(ts_e.size(), 1.0F);
    return {std::move(ts_e), std::move(ts_v), series.get_dt()};
}

[[nodiscard]] loki::io::TimeSeries make_injected_series(double accel_mps2) {
    loki::simulation::ModulatorParams mod;
    mod.acc = accel_mps2;
    loki::simulation::PulseSignalConfig cfg(
        kPeriod, kDt, kNsamps, 10.0, 0.1, "derivative", mod, std::nullopt, 1U);
    return to_unit_variance(cfg.generate("gaussian", 0.5));
}

void write_ffa_toml(const std::filesystem::path& path,
                    const std::filesystem::path& timeseries,
                    const std::filesystem::path& outdir,
                    const std::string& prefix,
                    bool use_fourier,
                    bool search_accel,
                    double f_min,
                    double f_max) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to write FFA recovery TOML");
    }
    out << "[input]\n";
    out << "timeseries = \"" << timeseries.string() << "\"\n";
    out << "preprocess = false\n\n";
    out << "[search]\n";
    out << "f_min = " << f_min << "\n";
    out << "f_max = " << f_max << "\n";
    if (search_accel) {
        out << "acc_min = 180.0\n";
        out << "acc_max = 220.0\n";
    } else {
        out << "acc_min = 0.0\n";
        out << "acc_max = 0.0\n";
    }
    out << "nbins = 64\n";
    out << "eta = 1.0\n";
    out << "ducy_max = 0.2\n";
    out << "wtsp = 1.5\n";
    out << "snr_min = " << kSearchSnrMin << "\n";
    out << "use_fourier = " << (use_fourier ? "true" : "false") << "\n\n";
    out << "[performance]\n";
    out << "nthreads = 4\n";
    out << "max_process_memory_gb = 4.0\n\n";
    out << "[output]\n";
    out << "outdir = \"" << outdir.string() << "\"\n";
    out << "prefix = \"" << prefix << "\"\n";
}

// Highest S/N among candidates inside the frequency (and acceleration) window.
// Returns 0 when the file has no such candidate.
[[nodiscard]] float best_recovered_snr(const std::filesystem::path& result_h5,
                                       bool expect_accel) {
    if (!std::filesystem::exists(result_h5)) {
        return 0.0F;
    }
    const HighFive::File file(result_h5.string(), HighFive::File::ReadOnly);
    if (!file.exist("snr") || !file.exist("param_sets")) {
        return 0.0F;
    }
    std::vector<std::string> param_names;
    if (file.hasAttribute("param_names")) {
        file.getAttribute("param_names").read(param_names);
    }
    std::vector<float> snrs;
    file.getDataSet("snr").read(snrs);
    std::vector<std::vector<double>> param_rows;
    file.getDataSet("param_sets").read(param_rows);
    if (snrs.size() != param_rows.size()) {
        return 0.0F;
    }

    std::size_t freq_col  = param_names.size();
    std::size_t accel_col = param_names.size();
    for (std::size_t i = 0; i < param_names.size(); ++i) {
        if (param_names[i] == "freq") {
            freq_col = i;
        }
        if (param_names[i] == "accel") {
            accel_col = i;
        }
    }
    if (freq_col >= param_names.size()) {
        freq_col = param_rows.empty() ? 0 : param_rows[0].size() - 1;
    }

    float best = 0.0F;
    for (std::size_t i = 0; i < snrs.size(); ++i) {
        if (param_rows[i].size() <= freq_col) {
            continue;
        }
        const double freq = param_rows[i][freq_col];
        if (std::abs(freq - kTrueFreq) > kFreqTol) {
            continue;
        }
        if (expect_accel) {
            if (accel_col >= param_names.size() ||
                param_rows[i].size() <= accel_col) {
                continue;
            }
            const double accel = param_rows[i][accel_col];
            if (std::abs(accel - kTrueAccel) > kAccelTol) {
                continue;
            }
        }
        best = std::max(best, snrs[i]);
    }
    return best;
}

[[nodiscard]] float max_snr(const std::filesystem::path& result_h5) {
    if (!std::filesystem::exists(result_h5)) {
        return 0.0F;
    }
    const HighFive::File file(result_h5.string(), HighFive::File::ReadOnly);
    if (!file.exist("snr")) {
        return 0.0F;
    }
    std::vector<float> snrs;
    file.getDataSet("snr").read(snrs);
    if (snrs.empty()) {
        return 0.0F;
    }
    return *std::ranges::max_element(snrs);
}

void run_cli_search(const std::filesystem::path& toml_path) {
    const std::string app_path = LOKI_APP_PATH;
    if (app_path.empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const std::string cmd =
        std::format(R"("{}" search ffa --config "{}" --no-preprocess)",
                    app_path, toml_path.string());
    // NOLINTNEXTLINE(bugprone-command-processor): runs the built CLI on purpose
    const int rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);
}

} // namespace

TEST_CASE("FFA CLI recovers injected pulsar in narrow search window",
          "[ffa][recovery][integration]") {
    const auto base =
        std::filesystem::temp_directory_path() / "loki_ffa_recovery_cli";
    std::filesystem::create_directories(base);

    const auto spin_ts  = base / "spin.tim";
    const auto accel_ts = base / "accel.tim";
    make_injected_series(0.0).write(spin_ts);
    make_injected_series(kTrueAccel).write(accel_ts);

    struct Case {
        std::filesystem::path ts_path;
        std::string tag;
        bool search_accel;
        bool use_fourier;
    };
    const std::vector<Case> cases = {
        {spin_ts, "spin_time", false, false},
        {spin_ts, "spin_fourier", false, true},
        {accel_ts, "accel_time", true, false},
        {accel_ts, "accel_fourier", true, true},
    };

    for (const auto& case_def : cases) {
        const auto outdir    = base / case_def.tag;
        const auto toml      = base / (case_def.tag + ".toml");
        const auto prefix    = case_def.tag;
        const auto result_h5 = outdir / (prefix + "_ffa_results.h5");

        std::filesystem::create_directories(outdir);
        std::filesystem::remove(result_h5);

        write_ffa_toml(toml, case_def.ts_path, outdir, prefix,
                       case_def.use_fourier, case_def.search_accel, 99.95,
                       100.05);
        run_cli_search(toml);

        const float best = best_recovered_snr(result_h5, case_def.search_accel);
        INFO("case: " << case_def.tag << " best recovered S/N: " << best);
        REQUIRE(best >= kRecoverSnrMin);
    }

    {
        const auto outdir    = base / "offband";
        const auto toml      = base / "offband.toml";
        const auto prefix    = std::string("offband");
        const auto result_h5 = outdir / "offband_ffa_results.h5";
        std::filesystem::create_directories(outdir);
        std::filesystem::remove(result_h5);
        write_ffa_toml(toml, spin_ts, outdir, prefix, /*use_fourier=*/true,
                       /*search_accel=*/false, 50.0, 50.1);
        run_cli_search(toml);
        const float peak = max_snr(result_h5);
        INFO("offband peak S/N: " << peak);
        REQUIRE(peak < kRecoverSnrMin);
    }

    std::filesystem::remove_all(base);
}
