#include <filesystem>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "loki/common/types.hpp"
#include "loki/search/configs.hpp"

using Catch::Matchers::WithinAbs;

namespace {

class TempDir {
public:
    TempDir() {
        std::mt19937_64 rng{std::random_device{}()};
        const auto root =
            std::filesystem::temp_directory_path() / "loki-config-tests";
        std::filesystem::create_directories(root);
        m_path = root / std::to_string(rng());
        std::filesystem::create_directories(m_path);
    }

    ~TempDir() { std::filesystem::remove_all(m_path); }

    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&)                 = delete;
    TempDir& operator=(TempDir&&)      = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return m_path; }

private:
    std::filesystem::path m_path;
};

} // namespace

TEST_CASE("FFASearchConfig standard construction and getters",
          "[config][ffa]") {
    const std::vector<loki::ParamLimit> limits = {{1.0, 10.0}};
    const loki::search::FFASearchConfig cfg(
        /*nsamps=*/1048576,
        /*tsamp=*/0.0001,
        /*nbins=*/64,
        /*eta=*/1.0, limits,
        /*ducy_max=*/0.2,
        /*wtsp=*/1.5,
        /*use_fourier=*/true,
        /*nthreads=*/4,
        /*max_process_memory_gb=*/4.0,
        /*octave_scale=*/2.0,
        /*nbins_max=*/1024,
        /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/32,
        /*bseg_ffa=*/64,
        /*snr_min=*/6.0,
        /*max_passing_candidates=*/1000000,
        /*use_boxcar_kadane=*/false);

    REQUIRE(cfg.get_nsamps() == 1048576);
    REQUIRE_THAT(cfg.get_tsamp(), WithinAbs(0.0001, 1e-12));
    REQUIRE_THAT(cfg.get_tobs(), WithinAbs(1048576 * 0.0001, 1e-6));
    REQUIRE(cfg.get_nbins() == 64);
    REQUIRE_THAT(cfg.get_eta(), WithinAbs(1.0, 1e-9));
    REQUIRE(cfg.get_param_limits().size() == 1);
    REQUIRE_THAT(cfg.get_param_limits()[0].min, WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(cfg.get_param_limits()[0].max, WithinAbs(10.0, 1e-9));
    REQUIRE_THAT(cfg.get_ducy_max(), WithinAbs(0.2, 1e-9));
    REQUIRE_THAT(cfg.get_wtsp(), WithinAbs(1.5, 1e-9));
    REQUIRE(cfg.get_use_fourier() == true);
    REQUIRE(cfg.get_nthreads() >= 1);
    REQUIRE(cfg.get_nthreads() <= 4);
    REQUIRE_THAT(cfg.get_max_process_memory_gb(), WithinAbs(4.0, 1e-9));
    REQUIRE_THAT(cfg.get_octave_scale(), WithinAbs(2.0, 1e-9));
    REQUIRE(cfg.get_nbins_max() == 1024);
    REQUIRE(cfg.get_nbins_min_lossy_bf() == 64);
    REQUIRE(cfg.get_bseg_brute() == 32);
    REQUIRE(cfg.get_bseg_ffa() == 64);
    REQUIRE_THAT(cfg.get_snr_min(), WithinAbs(6.0, 1e-9));
    REQUIRE(cfg.get_max_passing_candidates() == 1000000);
    REQUIRE(cfg.get_use_boxcar_kadane() == false);
    REQUIRE(cfg.get_use_conservative_tile() == false);
}

TEST_CASE("FFASearchConfig copy, move and get_updated_config",
          "[config][ffa]") {
    const std::vector<loki::ParamLimit> limits = {{2.0, 20.0}};
    const loki::search::FFASearchConfig orig(262144, 0.0002, 64, 1.0, limits,
                                             0.15, 1.4, false, 2, 2.0, 2.0, 512,
                                             32, 16, 32, 5.0, 500000, true);

    // Copy construction
    loki::search::FFASearchConfig copied = orig;
    REQUIRE(copied.get_nsamps() == orig.get_nsamps());
    REQUIRE(copied.get_nbins() == orig.get_nbins());
    REQUIRE(copied.get_use_boxcar_kadane() == true);
    REQUIRE(copied.get_use_fourier() == false);

    // Move construction
    const loki::search::FFASearchConfig moved = std::move(copied);
    REQUIRE(moved.get_nsamps() == 262144);
    REQUIRE(moved.get_nbins() == 64);

    // get_updated_config
    const std::vector<loki::ParamLimit> new_limits = {{10.0, 50.0}};
    const auto updated = orig.get_updated_config(128, 0.5, new_limits);
    REQUIRE(updated.get_nbins() == 128);
    REQUIRE_THAT(updated.get_eta(), WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(updated.get_param_limits()[0].min, WithinAbs(10.0, 1e-9));
    REQUIRE_THAT(updated.get_param_limits()[0].max, WithinAbs(50.0, 1e-9));
    REQUIRE(updated.get_nsamps() == orig.get_nsamps());
    REQUIRE_THAT(updated.get_tsamp(), WithinAbs(orig.get_tsamp(), 1e-12));

    // get_updated_config with f_min, f_max
    const auto updated2 = orig.get_updated_config(32, 1.5, 5.0, 15.0);
    REQUIRE(updated2.get_nbins() == 32);
    REQUIRE_THAT(updated2.get_eta(), WithinAbs(1.5, 1e-9));
    REQUIRE_THAT(updated2.get_param_limits()[0].min, WithinAbs(5.0, 1e-9));
    REQUIRE_THAT(updated2.get_param_limits()[0].max, WithinAbs(15.0, 1e-9));
}

TEST_CASE("FFASearchConfig validation throws on invalid arguments",
          "[config][ffa]") {
    const std::vector<loki::ParamLimit> valid_limits = {{1.0, 10.0}};

    // Zero nsamps
    REQUIRE_THROWS_AS(loki::search::FFASearchConfig(
                          0, 0.0001, 64, 1.0, valid_limits, 0.2, 1.5, true, 1,
                          8.0, 2.0, 1024, 64, std::nullopt, std::nullopt, 5.0,
                          1000, false),
                      std::runtime_error);

    // Negative tsamp
    REQUIRE_THROWS_AS(loki::search::FFASearchConfig(
                          1000, -0.0001, 64, 1.0, valid_limits, 0.2, 1.5, true,
                          1, 8.0, 2.0, 1024, 64, std::nullopt, std::nullopt,
                          5.0, 1000, false),
                      std::runtime_error);

    // Non-power of 2 nbins
    REQUIRE_THROWS_AS(loki::search::FFASearchConfig(
                          1000, 0.0001, 63, 1.0, valid_limits, 0.2, 1.5, true,
                          1, 8.0, 2.0, 1024, 64, std::nullopt, std::nullopt,
                          5.0, 1000, false),
                      std::runtime_error);

    // Invalid ducy_max
    REQUIRE_THROWS_AS(loki::search::FFASearchConfig(
                          1000, 0.0001, 64, 1.0, valid_limits, 0.0, 1.5, true,
                          1, 8.0, 2.0, 1024, 64, std::nullopt, std::nullopt,
                          5.0, 1000, false),
                      std::runtime_error);
    REQUIRE_THROWS_AS(loki::search::FFASearchConfig(
                          1000, 0.0001, 64, 1.0, valid_limits, 1.5, 1.5, true,
                          1, 8.0, 2.0, 1024, 64, std::nullopt, std::nullopt,
                          5.0, 1000, false),
                      std::runtime_error);

    // Empty param_limits
    REQUIRE_THROWS_AS(
        loki::search::FFASearchConfig(1000, 0.0001, 64, 1.0, {}, 0.2, 1.5, true,
                                      1, 8.0, 2.0, 1024, 64, std::nullopt,
                                      std::nullopt, 5.0, 1000, false),
        std::runtime_error);

    // Inverted frequency limits
    const std::vector<loki::ParamLimit> inv_limits = {{10.0, 1.0}};
    REQUIRE_THROWS_AS(loki::search::FFASearchConfig(
                          1000, 0.0001, 64, 1.0, inv_limits, 0.2, 1.5, true, 1,
                          8.0, 2.0, 1024, 64, std::nullopt, std::nullopt, 5.0,
                          1000, false),
                      std::runtime_error);
}

TEST_CASE("EPSearchConfig inherits and adds EP parameters", "[config][ep]") {
    const std::vector<loki::ParamLimit> limits = {{0.5, 5.0}};
    const loki::search::EPSearchConfig ep_cfg(
        /*nsamps=*/524288,
        /*tsamp=*/0.0001,
        /*nbins=*/64,
        /*eta=*/1.0, limits,
        /*ducy_max=*/0.2,
        /*wtsp=*/1.5,
        /*use_fourier=*/true,
        /*nthreads=*/2,
        /*max_process_memory_gb=*/4.0,
        /*octave_scale=*/2.0,
        /*nbins_max=*/1024,
        /*nbins_min_lossy_bf=*/64,
        /*bseg_brute=*/32,
        /*bseg_ffa=*/64,
        /*snr_min=*/5.0,
        /*max_passing_candidates=*/100000,
        /*prune_poly_order=*/2,
        /*p_orb_min=*/7200.0,
        /*m_c_max=*/0.5,
        /*m_p_min=*/1.2,
        /*propagator_significance=*/3.0,
        /*validation_significance=*/4.0,
        /*use_conservative_tile=*/true,
        /*use_boxcar_kadane=*/false);

    // Verify it is a valid FFASearchConfig
    const loki::search::FFASearchConfig& base_ref = ep_cfg;
    REQUIRE(base_ref.get_nsamps() == 524288);
    REQUIRE(base_ref.get_nbins() == 64);

    // EP specific parameters
    REQUIRE(ep_cfg.get_prune_poly_order() == 2);
    REQUIRE_THAT(ep_cfg.get_p_orb_min(), WithinAbs(7200.0, 1e-9));
    REQUIRE_THAT(ep_cfg.get_m_c_max(), WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(ep_cfg.get_m_p_min(), WithinAbs(1.2, 1e-9));
    REQUIRE_THAT(ep_cfg.get_propagator_significance(), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(ep_cfg.get_validation_significance(), WithinAbs(4.0, 1e-9));
    REQUIRE(ep_cfg.get_use_conservative_tile() == true);
    REQUIRE(ep_cfg.get_x_mass_const() > 0.0);

    // PulsarSearchConfig alias compatibility
    static_assert(std::is_same_v<loki::search::PulsarSearchConfig,
                                 loki::search::EPSearchConfig>);
}

TEST_CASE("FFATomlConfig default generation and parsing", "[config][toml]") {
    const std::string_view default_toml =
        loki::search::FFATomlConfig::default_toml_string();
    REQUIRE_FALSE(default_toml.empty());
    REQUIRE(default_toml.find("[input]") != std::string_view::npos);
    REQUIRE(default_toml.find("[search]") != std::string_view::npos);
    REQUIRE(default_toml.find("[performance]") != std::string_view::npos);
    REQUIRE(default_toml.find("[output]") != std::string_view::npos);

    // Parse default TOML
    auto cfg = loki::search::FFATomlConfig::from_string(default_toml);
    REQUIRE(cfg.timeseries_path == "input.tim");
    REQUIRE(cfg.preprocess == true);
    REQUIRE(cfg.fast_median == true);
    REQUIRE(cfg.fast_median_min_points == 101);
    REQUIRE_THAT(cfg.f_min, WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(cfg.f_max, WithinAbs(100.0, 1e-9));
    REQUIRE(cfg.nbins == 64);
    REQUIRE_THAT(cfg.eta, WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(cfg.snr_min, WithinAbs(5.0, 1e-9));
    REQUIRE(cfg.use_fourier == true);
    REQUIRE(cfg.prefix == "loki");

    // Convert to FFASearchConfig with supplied nsamps and tsamp
    const auto search_cfg = cfg.to_search_config(1048576, 0.0001);
    REQUIRE(search_cfg.get_nsamps() == 1048576);
    REQUIRE_THAT(search_cfg.get_tsamp(), WithinAbs(0.0001, 1e-12));
    REQUIRE(search_cfg.get_nbins() == 64);
    REQUIRE_THAT(search_cfg.get_param_limits()[0].min, WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(search_cfg.get_param_limits()[0].max, WithinAbs(100.0, 1e-9));
}

TEST_CASE("FFATomlConfig file roundtrip in temporary directory",
          "[config][toml]") {
    const TempDir tmp;
    const auto config_path = tmp.path() / "test_ffa_config.toml";

    // Write default TOML to file
    loki::search::FFATomlConfig::write_default(config_path);
    REQUIRE(std::filesystem::exists(config_path));

    // Load back from file
    auto loaded = loki::search::FFATomlConfig::load(config_path);
    REQUIRE(loaded.timeseries_path == "input.tim");
    REQUIRE(loaded.nbins == 64);

    // Direct FFASearchConfig::from_toml
    const auto search_cfg =
        loki::search::FFASearchConfig::from_toml(config_path, 2097152, 0.00005);
    REQUIRE(search_cfg.get_nsamps() == 2097152);
    REQUIRE_THAT(search_cfg.get_tsamp(), WithinAbs(0.00005, 1e-12));
    REQUIRE(search_cfg.get_nbins() == 64);
}

TEST_CASE("FFATomlConfig multi-dimensional parameter grids", "[config][toml]") {
    // 2D grid: frequency + acceleration
    const std::string toml_2d = R"(
[input]
timeseries = "test.tim"

[search]
f_min = 2.0
f_max = 20.0
acc_min = -1.5
acc_max = 1.5
nbins = 32
eta = 0.8
snr_min = 7.0

[performance]
nthreads = 2
)";
    auto cfg_2d = loki::search::FFATomlConfig::from_string(toml_2d);
    REQUIRE(cfg_2d.acc_min.has_value());
    REQUIRE(cfg_2d.acc_max.has_value());
    REQUIRE_THAT(*cfg_2d.acc_min, WithinAbs(-1.5, 1e-9));
    REQUIRE_THAT(*cfg_2d.acc_max, WithinAbs(1.5, 1e-9));

    const auto sc_2d = cfg_2d.to_search_config(131072, 0.001);
    REQUIRE(sc_2d.get_nparams() == 2);
    REQUIRE_THAT(sc_2d.get_param_limits()[0].min, WithinAbs(-1.5, 1e-9));
    REQUIRE_THAT(sc_2d.get_param_limits()[0].max, WithinAbs(1.5, 1e-9));
    REQUIRE_THAT(sc_2d.get_param_limits()[1].min, WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(sc_2d.get_param_limits()[1].max, WithinAbs(20.0, 1e-9));

    // 3D grid: frequency + acceleration + jerk
    const std::string toml_3d = R"(
[search]
f_min = 5.0
f_max = 15.0
acc_min = -2.0
acc_max = 2.0
jerk_min = -0.05
jerk_max = 0.05
)";
    const auto cfg_3d = loki::search::FFATomlConfig::from_string(toml_3d);
    const auto sc_3d  = cfg_3d.to_search_config(131072, 0.001);
    REQUIRE(sc_3d.get_nparams() == 3);
    REQUIRE_THAT(sc_3d.get_param_limits()[0].min, WithinAbs(-0.05, 1e-9));
    REQUIRE_THAT(sc_3d.get_param_limits()[0].max, WithinAbs(0.05, 1e-9));
    REQUIRE_THAT(sc_3d.get_param_limits()[1].min, WithinAbs(-2.0, 1e-9));
    REQUIRE_THAT(sc_3d.get_param_limits()[1].max, WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(sc_3d.get_param_limits()[2].min, WithinAbs(5.0, 1e-9));
    REQUIRE_THAT(sc_3d.get_param_limits()[2].max, WithinAbs(15.0, 1e-9));
}
