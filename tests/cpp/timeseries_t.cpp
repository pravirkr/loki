#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/io/timeseries.hpp"

using Catch::Matchers::WithinAbs;

namespace {

class TempDir {
public:
    TempDir() {
        std::mt19937_64 rng{std::random_device{}()};
        const auto root =
            std::filesystem::temp_directory_path() / "loki-io-tests";
        std::filesystem::create_directories(root);
        m_path = root / std::to_string(rng());
        std::filesystem::create_directories(m_path);
    }

    ~TempDir() { std::filesystem::remove_all(m_path); }

    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return m_path; }

private:
    std::filesystem::path m_path;
};

} // namespace

TEST_CASE("timeseries rejects non-finite or non-positive variance", "[io]") {
    const std::vector<float> intensity{1.0F, 2.0F};
    REQUIRE_THROWS_AS(
        (loki::io::TimeSeries(intensity, {1.0F, 0.0F}, 0.1)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        (loki::io::TimeSeries(intensity, {1.0F, std::numeric_limits<float>::quiet_NaN()}, 0.1)),
        std::invalid_argument);
}

TEST_CASE("timeseries stores intensity, variance, and sample interval",
          "[io]") {
    const std::vector<float> intensity{1.0F, 2.0F, 4.0F};
    const std::vector<float> variance{0.5F, 0.5F, 0.5F};
    const loki::io::TimeSeries series(intensity, variance, 0.25);

    REQUIRE(series.get_nsamps() == 3);
    REQUIRE_THAT(series.get_dt(), WithinAbs(0.25, 0.0));
    REQUIRE_THAT(series.get_tobs(), WithinAbs(0.75, 0.0));
    REQUIRE(series.get_ts_e()[2] == 4.0F);
    REQUIRE(series.get_ts_v()[0] == 0.5F);
}

TEST_CASE("timeseries leaves the payload unchanged when preprocessing is off",
          "[io]") {
    TempDir dir;
    std::vector<float> samples{1.5F, -2.0F, 3.25F, 0.0F};
    std::vector<float> variance(samples.size(), 4.0F);
    const loki::io::TimeSeries series(samples, variance, 6.4e-5);
    const auto path = dir.path() / "raw.tim";
    series.write(path);

    loki::io::ReadOptions options;
    options.preprocess = false;
    const auto loaded  = loki::io::TimeSeries::read(path, options);
    REQUIRE(loaded.get_nsamps() == samples.size());
    REQUIRE_THAT(loaded.get_dt(), WithinAbs(6.4e-5, 0.0));
    for (std::size_t i = 0; i < samples.size(); ++i) {
        REQUIRE(loaded.get_ts_e()[i] == samples[i]);
        REQUIRE(loaded.get_ts_v()[i] == 1.0F);
    }
}

TEST_CASE("timeseries preprocessing removes a constant offset", "[io]") {
    TempDir dir;
    const std::vector<float> samples(64, 7.0F);
    const std::vector<float> variance(samples.size(), 1.0F);
    const loki::io::TimeSeries series(samples, variance, 0.05);
    const auto path = dir.path() / "offset.tim";
    series.write(path);

    loki::io::ReadOptions options;
    options.preprocess    = true;
    options.filter_window = 1.0;
    const auto loaded     = loki::io::TimeSeries::read(path, options);
    double mean           = 0.0;
    for (loki::SizeType i = 0; i < loaded.get_nsamps(); ++i) {
        mean += static_cast<double>(loaded.get_ts_e()[i]);
        REQUIRE(loaded.get_ts_v()[i] == 1.0F);
    }
    mean /= static_cast<double>(loaded.get_nsamps());
    REQUIRE_THAT(mean, WithinAbs(0.0, 1e-5));
}

TEST_CASE("timeseries z-score scales a varying series", "[io]") {
    TempDir dir;
    std::vector<float> samples(32);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<float>(i);
    }
    const std::vector<float> variance(samples.size(), 1.0F);
    const loki::io::TimeSeries series(samples, variance, 1.0);
    const auto path = dir.path() / "ramp.tim";
    series.write(path);

    loki::io::ReadOptions options;
    options.preprocess    = true;
    options.filter_window = 100.0;
    options.loc           = loki::io::LocMethod::kMean;
    options.scale         = loki::io::ScaleMethod::kStd;
    const auto loaded     = loki::io::TimeSeries::read(path, options);

    double mean  = 0.0;
    double accum = 0.0;
    for (float sample : loaded.get_ts_e()) {
        mean += static_cast<double>(sample);
    }
    mean /= static_cast<double>(loaded.get_nsamps());
    for (float sample : loaded.get_ts_e()) {
        const double delta = static_cast<double>(sample) - mean;
        accum += delta * delta;
    }
    const double stddev =
        std::sqrt(accum / static_cast<double>(loaded.get_nsamps()));
    REQUIRE_THAT(mean, WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(stddev, WithinAbs(1.0, 1e-3));
    REQUIRE(loaded.get_ts_v()[0] == 1.0F);
}
