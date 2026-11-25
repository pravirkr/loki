#include "loki/io/sift.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <highfive/highfive.hpp>

#include "temp_dir.hpp"

namespace {

struct Row {
    double freq;
    float snr;
};

constexpr double kTobs = 1000.0;

/// Writes the parts of a published result file sift reads.
std::filesystem::path write_result(const loki::test::TempDir& dir,
                                   const std::string& name,
                                   std::optional<double> dm,
                                   double start_mjd,
                                   const std::vector<Row>& rows) {
    const auto path = dir / name;
    HighFive::File file(path.string(), HighFive::File::Truncate);
    file.createAttribute("schema_version", std::string("1.0"));
    file.createAttribute("tobs", kTobs);
    auto obs = file.createGroup("observation");
    if (dm) {
        obs.createAttribute("dm", *dm);
    }
    obs.createAttribute("start_mjd", start_mjd);
    obs.createAttribute("reference_mjd", start_mjd + (kTobs / 2 / 86400.0));
    obs.createAttribute("tobs", kTobs);
    auto table = file.createGroup("candidates");
    std::vector<double> f0;
    std::vector<double> f1;
    std::vector<float> snr;
    std::vector<double> duty;
    std::vector<std::uint64_t> id;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        f0.push_back(rows[i].freq);
        f1.push_back(0.0);
        snr.push_back(rows[i].snr);
        duty.push_back(0.1);
        id.push_back(i);
    }
    table.createDataSet("f0", f0);
    table.createDataSet("f1", f1);
    table.createDataSet("snr", snr);
    table.createDataSet("duty_cycle", duty);
    table.createDataSet("id", id);
    return path;
}

} // namespace

TEST_CASE("sift groups candidates over DM trials and segments", "[sift]") {
    const loki::test::TempDir dir("loki-sift");
    std::vector<std::filesystem::path> files;
    // A pulsar at 100 Hz peaking at the middle DM in both segments, its
    // second harmonic in the first, interference at 50.123 Hz flat in DM in
    // the first, and a candidate below the S/N minimum.
    const std::vector<double> dms{10.0, 20.0, 30.0};
    const std::vector<float> pulsar{10.0F, 15.0F, 10.0F};
    for (std::size_t s = 0; s < 2; ++s) {
        for (std::size_t d = 0; d < dms.size(); ++d) {
            std::vector<Row> rows{{100.0 + (1e-6 * static_cast<double>(d)),
                                   pulsar[d] - static_cast<float>(s)}};
            if (s == 0) {
                rows.push_back({50.123, 9.0F});
                rows.push_back({77.7, 6.0F});
                if (d == 1) {
                    rows.push_back({200.0, 8.0F});
                    // Within 2e-4 of 3x the pulsar, but 10 bins of 1/T off
                    // it in the same file: not a harmonic.
                    rows.push_back({300.01, 7.5F});
                }
            }
            files.push_back(
                write_result(dir, fmt::format("s{}_dm{}.h5", s, d), dms[d],
                             60000.0 + (0.1 * static_cast<double>(s)), rows));
        }
    }
    const auto result = loki::io::sift(files);
    CHECK(result.n_segments == 2);
    CHECK(result.n_candidates == 6 + 3 + 1 + 1);
    REQUIRE(result.groups.size() == 4);

    const auto& psr = result.groups[0];
    CHECK(psr.members.front().snr == 15.0F);
    CHECK(psr.members.size() == 6);
    CHECK(psr.segments == std::vector<loki::SizeType>{0, 1});
    REQUIRE(psr.dm_curve.size() == 3);
    CHECK(psr.dm_curve[1].first == 20.0);
    CHECK(psr.dm_curve[1].second == 15.0F);
    REQUIRE(psr.edge_ratio.has_value());
    CHECK(*psr.edge_ratio == Catch::Approx(10.0 / 15.0));
    REQUIRE(psr.extrapolation_bins.has_value());
    CHECK(*psr.extrapolation_bins < 0.01);
    CHECK(psr.harmonic_of == -1);

    const auto& rfi = result.groups[1];
    CHECK(rfi.members.front().freq == 50.123);
    CHECK(*rfi.edge_ratio == Catch::Approx(1.0));
    CHECK(rfi.segments.size() == 1);
    CHECK_FALSE(rfi.extrapolation_bins.has_value());

    const auto& harmonic = result.groups[2];
    CHECK(harmonic.members.front().freq == 200.0);
    CHECK(harmonic.harmonic_of == 0);
    CHECK(harmonic.harmonic_numerator == 2);
    CHECK(harmonic.harmonic_denominator == 1);
    REQUIRE(harmonic.harmonic_offset_bins.has_value());
    CHECK(*harmonic.harmonic_offset_bins < 0.01);
    CHECK(result.groups[3].members.front().freq == 300.01);
    CHECK(result.groups[3].harmonic_of == -1);

    SECTION("options") {
        loki::io::SiftOptions options;
        options.min_segments = 2;
        CHECK(loki::io::sift(files, options).groups.size() == 1);
        options                    = {};
        options.snr_min            = 5.0;
        options.harmonic_order_max = 0;
        const auto all             = loki::io::sift(files, options);
        CHECK(all.groups.size() == 5);
        CHECK(all.groups[2].harmonic_of == -1);
        options.tolerance = 0.0;
        CHECK_THROWS_AS(loki::io::sift(files, options), std::invalid_argument);
    }
    SECTION("files without /observation/dm skip DM curves cleanly") {
        const std::vector<std::filesystem::path> nodm{
            write_result(dir, "nodm_0.h5", std::nullopt, 60000.0,
                         {{100.0, 12.0F}, {200.0, 8.0F}}),
            write_result(dir, "nodm_1.h5", std::nullopt, 60000.1,
                         {{100.0 + 1e-6, 11.0F}}),
        };
        const auto res = loki::io::sift(nodm);
        REQUIRE(res.groups.size() == 2);
        CHECK(res.groups[0].dm_curve.empty());
        CHECK_FALSE(res.groups[0].edge_ratio.has_value());
        CHECK(res.groups[0].segments == std::vector<loki::SizeType>{0, 1});
        CHECK(res.groups[1].harmonic_of == 0);
        CHECK(res.groups[1].harmonic_numerator == 2);
        CHECK(res.groups[1].harmonic_denominator == 1);
    }

    SECTION("one DM trial has no edge ratio") {
        const std::vector<std::filesystem::path> one{
            write_result(dir, "dm1_0.h5", 10.0, 60000.0, {{100.0, 12.0F}}),
            write_result(dir, "dm1_1.h5", 10.0, 60000.1,
                         {{100.0 + 1e-6, 11.0F}}),
        };
        const auto res = loki::io::sift(one);
        REQUIRE(res.groups.size() == 1);
        CHECK(res.groups[0].dm_curve.size() == 1);
        CHECK_FALSE(res.groups[0].edge_ratio.has_value());
    }

    SECTION("a raw file is refused") {
        const auto raw = dir / "raw.h5";
        {
            HighFive::File file(raw.string(), HighFive::File::Truncate);
            file.createAttribute("ep_sweep_version", std::string("1.0.0"));
        }
        files.push_back(raw);
        CHECK_THROWS_AS(loki::io::sift(files), std::invalid_argument);
        CHECK_THROWS_AS(loki::io::sift(std::vector<std::filesystem::path>{
                            dir / "missing.h5"}),
                        std::runtime_error);
    }
}
