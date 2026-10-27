#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "loki/common/types.hpp"
#include "loki/io/timeseries.hpp"

#ifndef LOKI_APP_PATH
#define LOKI_APP_PATH ""
#endif

TEST_CASE("FFA CLI dry run plans with the input header", "[cli]") {
    const std::string app_path = LOKI_APP_PATH;
    if (app_path.empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const auto base =
        std::filesystem::temp_directory_path() / "loki_cli_dry_run_header";
    std::filesystem::create_directories(base);

    // 2^20 samples of 64 us: tobs 67.11 s. The config sets neither nsamps nor
    // tsamp, so only the header can give these.
    constexpr loki::SizeType kNsamps = loki::SizeType{1} << 20U;
    const auto ts_path               = base / "input.tim";
    loki::io::TimeSeries(std::vector<float>(kNsamps, 0.0F),
                         std::vector<float>(kNsamps, 1.0F), 6.4e-5)
        .write(ts_path);

    const auto toml_path = base / "ffa.toml";
    {
        std::ofstream out(toml_path);
        out << "[input]\n"
            << "timeseries = \"" << ts_path.string() << "\"\n"
            << "[search]\nf_min = 1.0\nf_max = 10.0\nnbins = 64\n"
            << "[performance]\nnthreads = 1\nmax_process_memory_gb = 4.0\n";
    }
    const auto log_path = base / "dry_run.log";
    const std::string cmd =
        std::format(R"("{}" search ffa --config "{}" --dry-run > "{}" 2>&1)",
                    app_path, toml_path.string(), log_path.string());
    // NOLINTNEXTLINE(bugprone-command-processor): runs the built CLI on purpose
    REQUIRE(std::system(cmd.c_str()) == 0);

    std::ifstream log(log_path);
    const std::string text((std::istreambuf_iterator<char>(log)),
                           std::istreambuf_iterator<char>());
    INFO(text);
    REQUIRE(text.find("nsamps 1048576, tobs 67.11 s") != std::string::npos);
    REQUIRE(text.find("No input timeseries") == std::string::npos);
    REQUIRE(text.find("limit of 3.000 GB") != std::string::npos);
}
