#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/timeseries.hpp"

#include "temp_dir.hpp"

#ifndef LOKI_APP_PATH
#define LOKI_APP_PATH ""
#endif

TEST_CASE("FFA CLI dry run plans with the input header", "[cli]") {
    const std::string app_path = LOKI_APP_PATH;
    if (app_path.empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir base("loki-cli-dry-run-header");

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
        fmt::format(R"("{}" search ffa --config "{}" --dry-run > "{}" 2>&1)",
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

    // Non-power-of-2 input is planned natively on all 100000 samples.
    constexpr loki::SizeType kNp2Nsamps = 100000;
    const auto np2_path                 = base / "input_np2.tim";
    loki::io::TimeSeries(std::vector<float>(kNp2Nsamps, 0.0F),
                         std::vector<float>(kNp2Nsamps, 1.0F), 6.4e-5)
        .write(np2_path);
    const auto np2_toml = base / "ffa_np2.toml";
    {
        std::ofstream out(np2_toml);
        out << "[input]\n"
            << "timeseries = \"" << np2_path.string() << "\"\n"
            << "[search]\nf_min = 1.0\nf_max = 10.0\nnbins = 64\n"
            << "[performance]\nnthreads = 1\nmax_process_memory_gb = 4.0\n";
    }
    const auto np2_log = base / "dry_run_np2.log";
    const std::string cmd_native =
        fmt::format(R"("{}" search ffa --config "{}" --dry-run > "{}" 2>&1)",
                    app_path, np2_toml.string(), np2_log.string());
    // NOLINTNEXTLINE(bugprone-command-processor): runs the built CLI on purpose
    REQUIRE(std::system(cmd_native.c_str()) == 0);
    {
        std::ifstream log_np2(np2_log);
        const std::string text_np2((std::istreambuf_iterator<char>(log_np2)),
                                   std::istreambuf_iterator<char>());
        INFO(text_np2);
        REQUIRE(text_np2.find("nsamps 100000") != std::string::npos);
    }
}

namespace {

/// Runs `search ep --dry-run` on @p toml with @p extra flags; returns the exit
/// status and the output.
std::pair<int, std::string> ep_dry_run(const loki::test::TempDir& base,
                                       const std::filesystem::path& toml,
                                       const std::string& extra,
                                       const std::string& tag) {
    const auto log = base / (tag + ".log");
    const std::string cmd =
        fmt::format(R"("{}" search ep --config "{}" --dry-run {} > "{}" 2>&1)",
                    LOKI_APP_PATH, toml.string(), extra, log.string());
    // NOLINTNEXTLINE(bugprone-command-processor): runs the built CLI on purpose
    const int status = std::system(cmd.c_str());
    std::ifstream in(log);
    return {status, std::string((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>())};
}

} // namespace

TEST_CASE("EP CLI dry run checks the host memory of a CUDA search", "[cli]") {
    const std::string app_path = LOKI_APP_PATH;
    if (app_path.empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir base("loki-cli-dry-run-host-memory");
    constexpr loki::SizeType kNsamps = loki::SizeType{1} << 15U;
    const auto ts_path               = base / "input.tim";
    loki::io::TimeSeries(std::vector<float>(kNsamps, 0.0F),
                         std::vector<float>(kNsamps, 1.0F), 6.4e-5)
        .write(ts_path);
    const auto write_toml = [&](const std::string& backend) {
        const auto path = base / (backend + ".toml");
        std::ofstream out(path);
        out << "[input]\ntimeseries = \"" << ts_path.string() << "\"\n"
            << "[preprocessing]\npreprocess = false\n"
            << "[search]\nf_min = 140.0\nf_max = 142.0\nacc_min = -10.0\n"
            << "acc_max = 10.0\nnbins = 32\n"
            << "[ep]\nn_runs = 1\nmin_pd = 0.1\nprune_poly_order = 2\n"
            << "nsegments = 8\n"
            << "[performance]\nnthreads = 1\nmax_process_memory_gb = 4.0\n"
            << "backend = \"" << backend << "\"\n"
            << "[output]\noutdir = \"" << base.path().string() << "\"\n";
        return path;
    };

    // The CPU caps host memory with max_process_memory_gb: the key is
    // ignored, with a warning.
    {
        const auto [status, text] = ep_dry_run(
            base, write_toml("cpu"), "--max-host-memory-gb 0.01", "cpu");
        INFO(text);
        CHECK(status == 0);
        CHECK(text.find("max_host_memory_gb is ignored on the CPU") !=
              std::string::npos);
    }
    if (!loki::is_available(loki::Backend::kCUDA)) {
        SKIP("CUDA backend not built");
    }
    const auto cuda = write_toml("cuda");
    {
        const auto [status, text] =
            ep_dry_run(base, cuda, "--max-host-memory-gb 64", "cuda_fits");
        INFO(text);
        CHECK(status == 0);
        CHECK(text.find("Host memory:") != std::string::npos);
        CHECK(text.find("against a limit of 64.00 GB") != std::string::npos);
    }
    {
        // Below the CUDA context alone: no search fits.
        const auto [status, text] =
            ep_dry_run(base, cuda, "--max-host-memory-gb 0.01", "cuda_over");
        INFO(text);
        CHECK(status != 0);
        CHECK(text.find("is above performance.max_host_memory_gb") !=
              std::string::npos);
    }
}
