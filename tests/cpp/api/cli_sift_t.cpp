#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <highfive/highfive.hpp>

#include "temp_dir.hpp"

#ifndef LOKI_APP_PATH
#define LOKI_APP_PATH ""
#endif

namespace {

/// Runs the CLI with @p args; returns its exit status and stdout (stderr
/// goes to @p log).
std::pair<int, std::string> run_cli(const std::string& args,
                                    const std::filesystem::path& out,
                                    const std::filesystem::path& log) {
    const std::string cmd =
        fmt::format(R"("{}" -q {} > "{}" 2> "{}")", LOKI_APP_PATH, args,
                    out.string(), log.string());
    // NOLINTNEXTLINE(bugprone-command-processor): runs the built CLI on purpose
    const int status = std::system(cmd.c_str());
    std::ifstream in(out);
    return {status, std::string((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>())};
}

void write_result(const std::filesystem::path& path, double dm, float snr) {
    HighFive::File file(path.string(), HighFive::File::Truncate);
    file.createAttribute("schema_version", std::string("1.0"));
    auto obs = file.createGroup("observation");
    obs.createAttribute("dm", dm);
    obs.createAttribute("start_mjd", 60000.0);
    obs.createAttribute("tobs", 1000.0);
    auto table = file.createGroup("candidates");
    table.createDataSet("f0", std::vector<double>{100.0, 200.0});
    table.createDataSet("f1", std::vector<double>{0.0, 0.0});
    table.createDataSet("snr", std::vector<float>{snr, snr - 2.0F});
    table.createDataSet("duty_cycle", std::vector<double>{0.1, 0.1});
    table.createDataSet("id", std::vector<std::uint64_t>{3, 4});
}

} // namespace

TEST_CASE("sift CLI groups files given by a pattern", "[cli]") {
    if (std::string(LOKI_APP_PATH).empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir dir("loki-cli-sift");
    write_result(dir / "r_dm10.h5", 10.0, 9.0F);
    write_result(dir / "r_dm20.h5", 20.0, 12.0F);
    write_result(dir / "r_dm30.h5", 30.0, 9.0F);
    const auto out = dir / "out.txt";
    const auto log = dir / "log.txt";
    const auto csv = dir / "groups.csv";

    const auto [status, json] =
        run_cli(fmt::format(R"(sift "{}" --json --members --csv "{}")",
                            (dir / "r_*.h5").string(), csv.string()),
                out, log);
    INFO(json);
    REQUIRE(status == 0);
    CHECK(json.find("\"n_candidates\": 6") != std::string::npos);
    CHECK(json.find("\"dm_curve\": [[10, 9], [20, 12], [30, 9]]") !=
          std::string::npos);
    CHECK(json.find("\"harmonic_of\": 0, \"harmonic_numerator\": 2") !=
          std::string::npos);
    CHECK(json.find("\"members\": [{\"file\": 1, \"id\": 3") !=
          std::string::npos);
    std::ifstream table(csv);
    std::string header;
    std::getline(table, header);
    CHECK(header.starts_with("index,snr,freq,"));
    std::size_t rows = 0;
    for (std::string line; std::getline(table, line);) {
        ++rows;
    }
    CHECK(rows == 2);

    const auto [text_status, text] = run_cli(
        fmt::format(R"(sift "{}" "{}" -n 1)", (dir / "r_dm10.h5").string(),
                    (dir / "r_dm20.h5").string()),
        out, log);
    CHECK(text_status == 0);
    CHECK(text.find("4 candidates (S/N >= 7) from 2 files in 1 segments: 2 "
                    "groups") != std::string::npos);

    const auto [none, none_out] = run_cli(
        fmt::format(R"(sift "{}")", (dir / "x_*.h5").string()), out, log);
    CHECK(none != 0);
}
