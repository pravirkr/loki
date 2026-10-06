#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "search/cands.hpp"

using loki::cands::FFAResultMetadata;
using loki::cands::FFAResultWriter;

TEST_CASE("FFAResultWriter creates datasets and completes atomically",
          "[ffa][hdf5]") {
    const auto dir  = std::filesystem::temp_directory_path();
    const auto path = dir / "loki_ffa_writer_zero.h5";
    const auto tmp  = std::filesystem::path(path.string() + ".tmp");

    std::filesystem::remove(path);
    std::filesystem::remove(tmp);

    FFAResultWriter writer(path, FFAResultWriter::Mode::kWrite);
    FFAResultMetadata meta;
    meta.param_names = {"freq"};
    meta.config_toml = "# test";
    meta.tsamp       = 1.0e-4;
    meta.nsamps      = 1024;
    meta.tobs        = 0.1024;
    meta.f_min       = 1.0;
    meta.f_max       = 100.0;
    meta.snr_min     = 5.0;
    meta.ducy_max    = 0.2;
    meta.wtsp        = 1.5;
    meta.nbins_min   = 64;
    meta.nbins_max   = 64;
    meta.octave_scale = 2.0;
    meta.eta         = 1.0;
    meta.use_fourier = true;
    writer.write_metadata(meta);
    writer.finalize();

    REQUIRE(std::filesystem::exists(path));
    REQUIRE_FALSE(std::filesystem::exists(tmp));

    const HighFive::File file(path.string(), HighFive::File::ReadOnly);
    REQUIRE(file.exist("snr"));
    REQUIRE(file.exist("param_sets"));
    REQUIRE(file.exist("width"));
    REQUIRE(file.exist("nbins"));
    REQUIRE(file.hasAttribute("ffa_version"));
    REQUIRE(file.hasAttribute("complete"));
    int complete = 0;
    file.getAttribute("complete").read(complete);
    REQUIRE(complete == 1);

    std::filesystem::remove(path);
}
