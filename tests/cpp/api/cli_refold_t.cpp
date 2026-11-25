#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "loki/common/types.hpp"
#include "loki/simulation/modulate.hpp"
#include "loki/simulation/pulse.hpp"

#include "temp_dir.hpp"

#ifndef LOKI_APP_PATH
#define LOKI_APP_PATH ""
#endif

namespace {

/// Runs the CLI with @p args; returns its exit status and output.
std::pair<int, std::string> run_cli(const std::string& args,
                                    const std::filesystem::path& log) {
    const std::string cmd = fmt::format(R"("{}" -q {} > "{}" 2>&1)",
                                        LOKI_APP_PATH, args, log.string());
    // NOLINTNEXTLINE(bugprone-command-processor): runs the built CLI on purpose
    const int status = std::system(cmd.c_str());
    std::ifstream in(log);
    return {status, std::string((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>())};
}

/// The text after `"key": ` in the first target of refold's JSON.
std::string json_field(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\": ");
    REQUIRE(at != std::string::npos);
    const auto start = at + key.size() + 4;
    return text.substr(start, text.find_first_of(",}", start) - start);
}

} // namespace

TEST_CASE("refold CLI folds parameters given as accel or fdot", "[cli]") {
    if (std::string(LOKI_APP_PATH).empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir base("loki-cli-refold");
    // 2^14 samples of 1 ms, a 20 Hz pulse at 1e5 m/s^2.
    constexpr double kFreq  = 20.0;
    constexpr double kAccel = 1.0e5;
    loki::simulation::ModulatorParams mod;
    mod.acc           = kAccel;
    auto ts           = loki::simulation::PulseSignalConfig(
                            1.0 / kFreq, 1e-3, loki::SizeType{1} << 14U, 20.0, 0.1,
                            "derivative", mod, std::nullopt, 11)
                            .generate("gaussian");
    const auto series = base / "pulse.tim";
    ts.write(series);
    const auto log = base / "refold.log";

    const auto [acc_status, by_acc] =
        run_cli(fmt::format(R"(refold "{}" --f {} --acc {} -t 1 --json)",
                            series.string(), kFreq, kAccel),
                log);
    INFO(by_acc);
    REQUIRE(acc_status == 0);
    const double fd = -kFreq * kAccel / 299792458.0;
    const auto [fd_status, by_fd] =
        run_cli(fmt::format(R"(refold "{}" --f {} --fd {:.17g} -t 1 --json)",
                            series.string(), kFreq, fd),
                log);
    INFO(by_fd);
    REQUIRE(fd_status == 0);
    CHECK(json_field(by_acc, "model") == "\"taylor\"");
    CHECK(json_field(by_acc, "preprocessing").empty() == false);
    CHECK(json_field(by_fd, "snr") == json_field(by_acc, "snr"));
    CHECK(json_field(by_fd, "snr_in") == json_field(by_acc, "snr_in"));
    CHECK(std::stod(json_field(by_acc, "snr")) > 10.0);

    // Five parameters as an orbit: zero orbital terms stay a polynomial.
    const auto [circ_status, circ] = run_cli(
        fmt::format(R"(refold "{}" --f {} --acc {} --circular -t 1 --json)",
                    series.string(), kFreq, kAccel),
        log);
    INFO(circ);
    REQUIRE(circ_status == 0);
    CHECK(json_field(circ, "model") == "\"taylor\"");
    // ... and says so, which a Taylor request does not.
    CHECK(circ.find("--circular folded with the Taylor model") !=
          std::string::npos);
    CHECK(by_acc.find("--circular") == std::string::npos);

    // Candidates need a result file.
    const auto [bad_status, bad] =
        run_cli(fmt::format(R"(refold "{}" --id 3)", series.string()), log);
    CHECK(bad_status != 0);
    CHECK(bad.find("--id and --top need --result") != std::string::npos);
}

TEST_CASE("refold CLI keeps a slow pulse out of the baseline", "[cli]") {
    if (std::string(LOKI_APP_PATH).empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir base("loki-cli-refold-slow");
    // A wide 0.25 Hz pulse (duty cycle 0.3, 1.2 s): the default 1 s baseline
    // window would subtract most of it (S/N 12 of 35); without a search
    // configuration the window is raised to 10 periods, as in the searches.
    auto ts           = loki::simulation::PulseSignalConfig(
                            4.0, 4e-3, loki::SizeType{1} << 16U, 30.0, 0.3, "derivative",
                            loki::simulation::ModulatorParams{}, std::nullopt, 5)
                            .generate("gaussian");
    const auto series = base / "slow.tim";
    ts.write(series);
    const auto log            = base / "refold.log";
    const auto [status, text] = run_cli(
        fmt::format(R"(refold "{}" --f 0.25 -t 1 --json)", series.string()),
        log);
    INFO(text);
    REQUIRE(status == 0);
    const auto [raw_status, raw] = run_cli(
        fmt::format(R"(refold "{}" --f 0.25 -t 1 --json --no-preprocess)",
                    series.string()),
        log);
    INFO(raw);
    REQUIRE(raw_status == 0);
    CHECK(std::stod(json_field(text, "snr")) >
          0.6 * std::stod(json_field(raw, "snr")));
}
