#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "loki/common/backend.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/search/configs.hpp"

TEST_CASE("FFATomlConfig rejects unknown keys", "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = 64
unknown_key = true
)";
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(toml),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig accepts an exact running median", "[config][ffa]") {
    const std::string toml = R"(
[preprocessing]
fast_median = false
fast_median_min_points = 51
[search]
f_min = 1.0
f_max = 10.0
nbins = 64
)";
    const auto cfg         = loki::search::FFATomlConfig::from_string(toml);
    REQUIRE(cfg.preprocessing.fast_median == false);
    REQUIRE(cfg.preprocessing.fast_median_min_points == 51);
}

TEST_CASE("FFATomlConfig rejects a zero fast-median width", "[config][ffa]") {
    const std::string toml = R"(
[preprocessing]
fast_median_min_points = 0
)";
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(toml),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig reads the [preprocessing] table", "[config][ffa]") {
    const std::string toml = R"(
[preprocessing]
preprocess = true
method = "zscore"
gain_model = "multiplicative"
filter_window = 2
min_window_periods = 0
variance_window = 4.5
window_blocks = 51
n_iter = 3
block_scales = [0.01, 1]
block_sigma = 5.0
min_good_fraction = 0.5
clip_sigma = 0
zap_periodic = true
zap_sigma = 7.0
zap_whiten_bins = 501
birdies = [[50.0, 0.1], [60, 0]]
)";
    const auto cfg         = loki::search::FFATomlConfig::from_string(toml);
    const auto& o          = cfg.preprocessing;
    REQUIRE(o.method == loki::io::PreprocessMethod::kZScore);
    REQUIRE(o.gain_model == loki::io::GainModel::kMultiplicative);
    REQUIRE(o.filter_window == 2.0);
    REQUIRE(cfg.min_window_periods == 0.0);
    REQUIRE(o.variance_window == 4.5);
    REQUIRE(o.window_blocks == 51);
    REQUIRE(o.n_iter == 3);
    REQUIRE(o.block_scales == std::vector<double>{0.01, 1.0});
    REQUIRE(o.block_sigma == 5.0);
    REQUIRE(o.min_good_fraction == 0.5);
    REQUIRE(o.clip_sigma == 0.0);
    REQUIRE(o.zap_periodic);
    REQUIRE(o.zap_sigma == 7.0);
    REQUIRE(o.zap_whiten_bins == 501);
    REQUIRE(o.birdies.size() == 2);
    REQUIRE(o.birdies[0].freq == 50.0);
    REQUIRE(o.birdies[0].width == 0.1);
    REQUIRE(o.birdies[1].freq == 60.0);
}

TEST_CASE("FFATomlConfig rejects bad [preprocessing] values", "[config][ffa]") {
    const auto bad = GENERATE(
        as<std::string>{}, //
        "method = \"median\"", "gain_model = \"none\"", "filter_window = -1.0",
        "min_window_periods = -1.0", "window_blocks = 0",
        "block_scales = [0.0]", "block_scales = 1.0", "block_sigma = 0.0",
        "min_good_fraction = 1.5", "clip_sigma = -1.0", "zap_sigma = 0.0",
        "zap_whiten_bins = 2", "birdies = [[50.0]]", "birdies = [[-1.0, 0.1]]",
        "zap_periodic = 1", "n_iter = 1.5", "unknown = 1");
    CAPTURE(bad);
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(
                          "[preprocessing]\n" + bad + "\n"),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig points moved [input] keys to [preprocessing]",
          "[config][ffa]") {
    for (const std::string key :
         {"preprocess = false", "filter_window = 1.0", "fast_median = true",
          "fast_median_min_points = 11"}) {
        CAPTURE(key);
        try {
            (void)loki::search::FFATomlConfig::from_string("[input]\n" + key +
                                                           "\n");
            FAIL("expected a rejection");
        } catch (const std::invalid_argument& err) {
            REQUIRE(std::string(err.what()).find("[preprocessing]") !=
                    std::string::npos);
        }
    }
}

TEST_CASE("FFATomlConfig resolves the baseline window from f_min",
          "[config][ffa]") {
    auto cfg                        = loki::search::FFATomlConfig{};
    cfg.f_min                       = 0.5;
    cfg.preprocessing.filter_window = 1.0;
    REQUIRE(cfg.to_preprocess_options().filter_window == 20.0);
    cfg.min_window_periods = 0.0;
    REQUIRE(cfg.to_preprocess_options().filter_window == 1.0);
    cfg.min_window_periods          = 10.0;
    cfg.preprocessing.filter_window = 30.0;
    REQUIRE(cfg.to_preprocess_options().filter_window == 30.0);
    cfg.preprocessing.filter_window = 0.0; // whole series: kept
    REQUIRE(cfg.to_preprocess_options().filter_window == 0.0);
}

TEST_CASE("FFATomlConfig rejects negative nbins", "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = -4
)";
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(toml),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig rejects use_boxcar_kadane at preflight",
          "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = 64
use_boxcar_kadane = true
)";
    const auto cfg         = loki::search::FFATomlConfig::from_string(toml);
    REQUIRE_THROWS_AS(cfg.to_search_config(), std::invalid_argument);
}

TEST_CASE("FFATomlConfig reads the execution backend", "[config][ffa]") {
    const std::string toml = R"(
[search]
f_min = 1.0
f_max = 10.0
nbins = 64
[performance]
backend = "cuda"
device = 1
)";
    const auto cfg         = loki::search::FFATomlConfig::from_string(toml);
    REQUIRE(cfg.backend == loki::Backend::kCUDA);
    REQUIRE(cfg.device == 1);

    const auto defaults = loki::search::FFATomlConfig::from_string("");
    REQUIRE(defaults.backend == loki::Backend::kCPU);
    REQUIRE(defaults.device == 0);
}

TEST_CASE("FFATomlConfig rejects an unknown backend name", "[config][ffa]") {
    const std::string toml = R"(
[performance]
backend = "gpu"
)";
    REQUIRE_THROWS_AS(loki::search::FFATomlConfig::from_string(toml),
                      std::invalid_argument);
}

TEST_CASE("FFATomlConfig rejects the removed CUDA keys with a hint",
          "[config][ffa]") {
    const auto message_for = [](const std::string& toml) -> std::string {
        try {
            (void)loki::search::FFATomlConfig::from_string(toml);
        } catch (const std::invalid_argument& err) {
            return err.what();
        }
        return {};
    };
    REQUIRE(message_for("[cuda]\nenable = true\n").find("backend") !=
            std::string::npos);
    REQUIRE(message_for("[performance]\nuse_cuda = true\n")
                .find("performance.backend") != std::string::npos);
    REQUIRE(message_for("[performance]\ndevice_id = 0\n")
                .find("performance.device") != std::string::npos);
}

TEST_CASE("FFA candidate TOML options are strict",
          "[ffa][config][candidates]") {
    const auto cfg = loki::search::FFATomlConfig::from_string(
        "[candidates]\ncluster = false\nradius = 2.0\nretain_detections = "
        "false\n"
        "[input]\ndm = 0.0\nstart_mjd = 60000.0\nbarycentric = false\n");
    REQUIRE_FALSE(cfg.candidates.cluster);
    REQUIRE_FALSE(cfg.candidates.retain_detections);
    REQUIRE(cfg.observation.dm.has_value());
    REQUIRE(*cfg.observation.dm == 0);
    REQUIRE(cfg.candidates.enabled);
    REQUIRE_FALSE(loki::search::FFATomlConfig::from_string(
                      "[candidates]\nenabled = false\n")
                      .candidates.enabled);
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\nenabled = 0\n"));
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\nraduis = 2.0\n"));
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\ncluster = 1\n"));
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\nradius = -1.0\n"));
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\nremove_harmonics = true\n"));
}

TEST_CASE("FFA production threshold and phase suppression are configurable",
          "[ffa][config][candidates]") {
    const auto defaults = loki::search::FFATomlConfig::from_string("");
    REQUIRE(defaults.snr_min == 7);
    REQUIRE(defaults.candidates.phase_radius == 0.05);
    REQUIRE(defaults.candidates.grouping == "transitive");
    const auto legacy = loki::search::FFATomlConfig::from_string(
        "[search]\nsnr_min = 6.0\n[candidates]\nphase_radius = 0.0\n");
    REQUIRE(legacy.snr_min == 6);
    REQUIRE(legacy.candidates.phase_radius == 0);
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\nphase_radius = -0.1\n"));
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\nphase_radius = \"wide\"\n"));
}

TEST_CASE("Candidate grouping policy is strict and configurable",
          "[config][candidates]") {
    const auto cfg = loki::search::FFATomlConfig::from_string(
        "[candidates]\ngrouping = \"anchored\"\n");
    REQUIRE(cfg.candidates.grouping == "anchored");
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\ngrouping = \"invalid\"\n"));
    REQUIRE_THROWS(loki::search::FFATomlConfig::from_string(
        "[candidates]\ngrouping = true\n"));
}
