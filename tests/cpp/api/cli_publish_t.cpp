#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <highfive/highfive.hpp>

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

/// Writes a simulated pulse series and a search configuration that reads it.
std::filesystem::path
write_case(const loki::test::TempDir& dir,
           const std::string& algorithm,
           const std::string& preprocessing = "preprocess = false\n") {
    const bool ep           = algorithm == "ep";
    const double freq       = ep ? 141.0 : 100.0;
    const double dt         = ep ? 6.4e-5 : 1e-4;
    const loki::SizeType ns = ep ? (1U << 15U) : (1U << 16U);
    auto ts                 = loki::simulation::PulseSignalConfig(
                                  1.0 / freq, dt, ns, 25.0, 0.1, "derivative",
                                  loki::simulation::ModulatorParams{}, std::nullopt, 5)
                                  .generate("gaussian");
    const auto series       = dir / (algorithm + ".tim");
    ts.write(series);
    const auto config = dir / (algorithm + ".toml");
    std::ofstream out(config);
    out << fmt::format("[input]\ntimeseries = \"{}\"\n", series.string());
    out << "[preprocessing]\n" << preprocessing;
    if (ep) {
        out << "[search]\nf_min = 140.0\nf_max = 142.0\nacc_min = -10.0\n"
               "acc_max = 10.0\nnbins = 32\n"
               "[ep]\nn_runs = 1\nmin_pd = 0.1\nprune_poly_order = 2\n"
               "nsegments = 8\n";
    } else {
        out << "[search]\nf_min = 99.0\nf_max = 101.0\nnbins = 64\n";
    }
    out << "[performance]\nnthreads = 2\nmax_process_memory_gb = 4.0\n"
           "backend = \"cpu\"\n";
    return config;
}

/// The candidate rows and counts of a published result file.
struct Product {
    std::vector<float> snr;
    std::vector<std::vector<double>> params;
    std::vector<double> f0;
    std::vector<std::uint8_t> in_zap_window;
    std::uint64_t n_candidates{};
    std::uint64_t n_detections{};
    std::string input_path;
    std::string command_line;
    std::string publish_command_line;
};

Product read_product(const std::filesystem::path& path) {
    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    Product p;
    file.getDataSet("snr").read(p.snr);
    file.getDataSet("param_sets").read(p.params);
    if (file.exist("candidates")) {
        file.getDataSet("candidates/f0").read(p.f0);
        file.getDataSet("candidates/in_zap_window").read(p.in_zap_window);
    }
    file.getAttribute("n_candidates").read(p.n_candidates);
    file.getAttribute("n_detections").read(p.n_detections);
    const auto observation = file.getGroup("observation");
    if (observation.hasAttribute("input_path")) {
        observation.getAttribute("input_path").read(p.input_path);
    }
    if (file.hasAttribute("command_line")) {
        file.getAttribute("command_line").read(p.command_line);
    }
    if (file.hasAttribute("publish_command_line")) {
        file.getAttribute("publish_command_line").read(p.publish_command_line);
    }
    return p;
}

/// The candidates of @p published equal those of @p searched.
void check_candidates(const Product& published, const Product& searched) {
    CHECK(published.n_candidates == searched.n_candidates);
    CHECK(published.n_detections == searched.n_detections);
    CHECK(published.snr == searched.snr);
    CHECK(published.params == searched.params);
    CHECK(published.in_zap_window == searched.in_zap_window);
}

void check_same(const Product& published, const Product& searched) {
    check_candidates(published, searched);
    CHECK(published.input_path == searched.input_path);
    CHECK(published.command_line.find(" search ") != std::string::npos);
    CHECK(published.publish_command_line.find(" publish ") !=
          std::string::npos);
}

} // namespace

TEST_CASE("publish CLI publishes a kept spool as the search would", "[cli]") {
    if (std::string(LOKI_APP_PATH).empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir dir("loki-cli-publish");
    const auto log = dir / "publish.log";

    // An unusable scratch directory fails the post-processing after the
    // search.
    std::ofstream(dir / "not_a_dir") << "x";
    const auto bad_tempdir = dir / "not_a_dir" / "x";

    SECTION("FFA: the stored options fail again until overridden") {
        const auto config = write_case(dir, "ffa");
        const auto [ok, ok_log] =
            run_cli(fmt::format(R"(search ffa -c "{}" -o "{}" -p run)",
                                config.string(), (dir / "ok").string()),
                    log);
        INFO(ok_log);
        REQUIRE(ok == 0);
        const auto [failed, failed_log] = run_cli(
            fmt::format(
                R"(search ffa -c "{}" -o "{}" -p run --candidate-tempdir "{}")",
                config.string(), (dir / "kept").string(), bad_tempdir.string()),
            log);
        INFO(failed_log);
        REQUIRE(failed != 0);
        const auto spool = dir / "kept" / "run_ffa_spool.h5";
        REQUIRE(std::filesystem::exists(spool));
        CHECK(failed_log.find("loki publish") != std::string::npos);
        {
            // The search's configuration is kept with it.
            const HighFive::File file(spool.string(), HighFive::File::ReadOnly);
            CHECK(file.hasAttribute("config_effective"));
            CHECK(file.hasAttribute("command_line"));
        }
        // The stored scratch directory fails again; the spool stays.
        const auto [again, again_log] =
            run_cli(fmt::format(R"(publish "{}")", spool.string()), log);
        CHECK(again != 0);
        CHECK(std::filesystem::exists(spool));
        const auto [status, text] =
            run_cli(fmt::format(R"(publish "{}" --candidate-tempdir "{}")",
                                spool.string(), dir.path().string()),
                    log);
        INFO(text);
        REQUIRE(status == 0);
        CHECK_FALSE(std::filesystem::exists(spool));
        const auto output = dir / "kept" / "run_ffa_results.h5";
        check_same(read_product(output),
                   read_product(dir / "ok" / "run_ffa_results.h5"));
        // A published file is refused.
        const auto [twice, twice_log] =
            run_cli(fmt::format(R"(publish "{}" -o "{}")", output.string(),
                                (dir / "again.h5").string()),
                    log);
        CHECK(twice != 0);
        CHECK(twice_log.find("already a published result file") !=
              std::string::npos);
    }

    SECTION("EP: publish keeps the spool with --keep-spool") {
        const auto config = write_case(dir, "ep");
        const auto [ok, ok_log] =
            run_cli(fmt::format(R"(search ep -c "{}" -o "{}" -p run)",
                                config.string(), (dir / "ok").string()),
                    log);
        INFO(ok_log);
        REQUIRE(ok == 0);
        const auto [failed, failed_log] = run_cli(
            fmt::format(
                R"(search ep -c "{}" -o "{}" -p run --candidate-tempdir "{}")",
                config.string(), (dir / "kept").string(), bad_tempdir.string()),
            log);
        INFO(failed_log);
        REQUIRE(failed != 0);
        const auto spool = dir / "kept" / "run_ep_spool.h5";
        REQUIRE(std::filesystem::exists(spool));
        const auto [status, text] = run_cli(
            fmt::format(R"(publish "{}" --candidate-tempdir "{}" --keep-spool)",
                        spool.string(), dir.path().string()),
            log);
        INFO(text);
        REQUIRE(status == 0);
        CHECK(std::filesystem::exists(spool));
        check_same(read_product(dir / "kept" / "run_ep_results.h5"),
                   read_product(dir / "ok" / "run_ep_results.h5"));
        {
            // Run statistics: the copied runs' counts and the publish time,
            // not the publishing process's RSS.
            const HighFive::File file(
                (dir / "kept" / "run_ep_results.h5").string(),
                HighFive::File::ReadOnly);
            REQUIRE(file.exist("run_stats"));
            const auto stats = file.getGroup("run_stats");
            CHECK(stats.hasAttribute("timing_publish_s"));
            CHECK_FALSE(stats.hasAttribute("peak_rss_gb"));
            REQUIRE(stats.hasAttribute("n_runs"));
            CHECK(stats.getAttribute("n_runs").read<std::uint64_t>() > 0);
            CHECK(stats.hasAttribute("n_overloads"));
        }
        // The product is not overwritten.
        const auto [exists, exists_log] =
            run_cli(fmt::format(R"(publish "{}")", spool.string()), log);
        CHECK(exists != 0);
        CHECK(exists_log.find("already exists") != std::string::npos);
        const auto [forced, forced_log] = run_cli(
            fmt::format(R"(publish "{}" --force --candidate-tempdir "{}")",
                        spool.string(), dir.path().string()),
            log);
        INFO(forced_log);
        CHECK(forced == 0);
        CHECK_FALSE(std::filesystem::exists(spool));
    }

    SECTION("EP: without its configuration or series the spool is kept") {
        // A reduction other than the defaults: only the spool's reduced_*
        // attributes can tell publish what it was.
        const auto config       = write_case(dir, "ep");
        const auto [ok, ok_log] = run_cli(
            fmt::format(R"(search ep -c "{}" -o "{}" -p run --max-per-cell 8)",
                        config.string(), (dir / "ok").string()),
            log);
        INFO(ok_log);
        REQUIRE(ok == 0);
        const auto [failed, failed_log] = run_cli(
            fmt::format(R"(search ep -c "{}" -o "{}" -p run --max-per-cell 8 )"
                        R"(--candidate-tempdir "{}")",
                        config.string(), (dir / "kept").string(),
                        bad_tempdir.string()),
            log);
        INFO(failed_log);
        REQUIRE(failed != 0);
        const auto spool = dir / "kept" / "run_ep_spool.h5";
        REQUIRE(std::filesystem::exists(spool));
        const auto expected = read_product(dir / "ok" / "run_ep_results.h5");

        // No stored configuration (as a spool of an earlier version).
        const auto bare = dir / "bare_ep_spool.h5";
        std::filesystem::copy_file(spool, bare);
        {
            HighFive::File file(bare.string(), HighFive::File::ReadWrite);
            for (const char* key : {"config_effective", "config_toml",
                                    "command_line", "loki_version"}) {
                if (file.hasAttribute(key)) {
                    file.deleteAttribute(key);
                }
            }
        }
        const auto [status, text] =
            run_cli(fmt::format(R"(publish "{}" --candidate-tempdir "{}")",
                                bare.string(), dir.path().string()),
                    log);
        INFO(text);
        REQUIRE(status == 0);
        CHECK(text.find("stores no configuration") != std::string::npos);
        CHECK(text.find("Kept the spool") != std::string::npos);
        CHECK(std::filesystem::exists(bare));
        check_candidates(read_product(dir / "bare_ep_results.h5"), expected);

        // The configuration but not its series.
        const auto series = dir / "ep.tim";
        std::filesystem::rename(series, dir / "moved.tim");
        const auto [moved, moved_log] = run_cli(
            fmt::format(R"(publish "{}" -o "{}" --candidate-tempdir "{}")",
                        spool.string(), (dir / "no_series.h5").string(),
                        dir.path().string()),
            log);
        INFO(moved_log);
        REQUIRE(moved == 0);
        CHECK(moved_log.find("was not found") != std::string::npos);
        CHECK(std::filesystem::exists(spool));
        // With the series given, the spool is published and removed.
        const auto [found, found_log] = run_cli(
            fmt::format(R"(publish "{}" -i "{}" --candidate-tempdir "{}")",
                        spool.string(), (dir / "moved.tim").string(),
                        dir.path().string()),
            log);
        INFO(found_log);
        REQUIRE(found == 0);
        CHECK_FALSE(std::filesystem::exists(spool));
        const auto published = read_product(dir / "kept" / "run_ep_results.h5");
        check_candidates(published, expected);
        CHECK(published.input_path == (dir / "moved.tim").string());
    }

    SECTION("a --no-candidates result needs -o and is kept") {
        const auto config = write_case(dir, "ffa");
        const auto [ok, ok_log] =
            run_cli(fmt::format(R"(search ffa -c "{}" -o "{}" -p run)",
                                config.string(), (dir / "ok").string()),
                    log);
        INFO(ok_log);
        REQUIRE(ok == 0);
        const auto [raw, raw_log] = run_cli(
            fmt::format(R"(search ffa -c "{}" -o "{}" -p run --no-candidates)",
                        config.string(), (dir / "raw").string()),
            log);
        INFO(raw_log);
        REQUIRE(raw == 0);
        const auto file = dir / "raw" / "run_ffa_results.h5";
        const auto [no_out, no_out_log] =
            run_cli(fmt::format(R"(publish "{}")", file.string()), log);
        CHECK(no_out != 0);
        CHECK(no_out_log.find("-o") != std::string::npos);
        const auto output = dir / "published.h5";
        const auto [status, text] =
            run_cli(fmt::format(R"(publish "{}" -o "{}")", file.string(),
                                output.string()),
                    log);
        INFO(text);
        REQUIRE(status == 0);
        CHECK(std::filesystem::exists(file));
        check_same(read_product(output),
                   read_product(dir / "ok" / "run_ffa_results.h5"));
        // The search's timings carry over, next to the publish time.
        const HighFive::File published(output.string(),
                                       HighFive::File::ReadOnly);
        REQUIRE(published.exist("run_stats"));
        const auto stats = published.getGroup("run_stats");
        CHECK(stats.hasAttribute("timing_total_s"));
        CHECK(stats.hasAttribute("timing_publish_s"));
    }
}

TEST_CASE("search and publish flag candidates at the birdies", "[cli]") {
    if (std::string(LOKI_APP_PATH).empty()) {
        SKIP("loki_app was not built (LOKI_APP_PATH undefined)");
    }
    const loki::test::TempDir dir("loki-cli-zap-window");
    const auto log = dir / "zap.log";
    // The pulsar is at 100 Hz; a birdie at its second harmonic flags it.
    const auto config = write_case(dir, "ffa", "birdies = [[200.0, 0.4]]\n");
    const auto [ok, ok_log] =
        run_cli(fmt::format(R"(search ffa -c "{}" -o "{}" -p run)",
                            config.string(), (dir / "ok").string()),
                log);
    INFO(ok_log);
    REQUIRE(ok == 0);
    const auto searched = read_product(dir / "ok" / "run_ffa_results.h5");
    REQUIRE(searched.f0.size() == searched.in_zap_window.size());
    std::size_t flagged = 0;
    for (std::size_t i = 0; i < searched.f0.size(); ++i) {
        INFO("f0 = " << searched.f0[i]);
        const bool in = std::abs((2 * searched.f0[i]) - 200.0) <= 0.2;
        CHECK(searched.in_zap_window[i] == (in ? 1 : 0));
        flagged += searched.in_zap_window[i];
    }
    CHECK(flagged > 0);

    // A publish of the kept spool flags the same rows.
    const auto bad_tempdir = dir / "not_a_dir" / "x";
    std::ofstream(dir / "not_a_dir") << "x";
    const auto [failed, failed_log] = run_cli(
        fmt::format(
            R"(search ffa -c "{}" -o "{}" -p run --candidate-tempdir "{}")",
            config.string(), (dir / "kept").string(), bad_tempdir.string()),
        log);
    REQUIRE(failed != 0);
    const auto spool = dir / "kept" / "run_ffa_spool.h5";
    const auto [status, text] =
        run_cli(fmt::format(R"(publish "{}" --candidate-tempdir "{}")",
                            spool.string(), dir.path().string()),
                log);
    INFO(text);
    REQUIRE(status == 0);
    check_candidates(read_product(dir / "kept" / "run_ffa_results.h5"),
                     searched);
}
