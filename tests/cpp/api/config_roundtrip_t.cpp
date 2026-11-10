#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <highfive/H5File.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/io/preprocess.hpp"
#include "loki/search/configs.hpp"
#include "loki/search/provenance.hpp"

#include "temp_dir.hpp"

using loki::search::EPTomlConfig;
using loki::search::FFATomlConfig;

namespace {

/// Every FFA setting away from its default, so a key the writer drops shows
/// up as a field that comes back at its default.
void set_non_defaults(FFATomlConfig& cfg) {
    cfg.timeseries_path         = R"(data/run "1"\obs.tim)";
    cfg.nsamps                  = 1U << 20U;
    cfg.tsamp                   = 6.4e-5;
    cfg.observation.dm          = 42.5;
    cfg.observation.start_mjd   = 60000.123456789;
    cfg.observation.barycentric = true;
    cfg.preprocess              = false;
    auto& o                     = cfg.preprocessing;
    o.method                    = loki::io::PreprocessMethod::kRobust;
    o.gain_model                = loki::io::GainModel::kMultiplicative;
    o.filter_window             = 2.5;
    cfg.min_window_periods      = 7.0;
    o.variance_window           = 3.0;
    o.window_blocks             = 77;
    o.fast_median               = false;
    o.fast_median_min_points    = 55;
    o.n_iter                    = 3;
    o.block_scales              = {0.02, 0.1, 0.5};
    o.block_sigma               = 5.5;
    o.min_good_fraction         = 0.4;
    o.clip_sigma                = 5.0;
    o.zap_periodic              = true;
    o.zap_sigma                 = 7.5;
    o.zap_whiten_bins           = 501;
    o.birdies   = {{.freq = 50.0, .width = 0.1}, {.freq = 60.0, .width = 0.25}};
    cfg.f_min   = 0.1;
    cfg.f_max   = 123.456;
    cfg.acc_min = -25.0;
    cfg.acc_max = 30.0;
    cfg.jerk_min               = -0.5;
    cfg.jerk_max               = 0.75;
    cfg.nbins                  = 128;
    cfg.eta                    = 0.5;
    cfg.ducy_max               = 0.3;
    cfg.wtsp                   = 1.25;
    cfg.snr_min                = 6.5;
    cfg.use_fourier            = false;
    cfg.use_boxcar_kadane      = true;
    cfg.nthreads               = 3;
    cfg.device                 = 1;
    cfg.max_process_memory_gb  = 12.5;
    cfg.octave_scale           = 1.5;
    cfg.nbins_max              = 512;
    cfg.nbins_min_lossy_bf     = 32;
    cfg.bseg_brute             = 256;
    cfg.max_passing_candidates = 1000;
    cfg.outdir                 = "out dir/";
    cfg.prefix                 = "run_7";
    auto& c                    = cfg.candidates;
    c.enabled                  = false;
    c.cluster                  = false;
    c.grouping                 = "anchored";
    c.radius                   = 2.0;
    c.phase_radius             = 0.1;
    c.retain_detections        = false;
    c.flag_harmonics           = true;
    c.remove_harmonics         = true;
    c.harmonic_denominator_max = 16;
    c.harmonic_phase_distance  = 0.5;
    c.harmonic_snr_distance    = 2.0;
    c.max_candidates           = 50;
    c.memory_gb                = 2.5;
    c.temporary_directory      = "/scratch/tmp";
}

void check_same(const FFATomlConfig& a, const FFATomlConfig& b) {
    CHECK(a.timeseries_path == b.timeseries_path);
    CHECK(a.nsamps == b.nsamps);
    CHECK(a.tsamp == b.tsamp);
    CHECK(a.observation.dm == b.observation.dm);
    CHECK(a.observation.start_mjd == b.observation.start_mjd);
    CHECK(a.observation.barycentric == b.observation.barycentric);
    CHECK(a.preprocess == b.preprocess);
    const auto& o = a.preprocessing;
    const auto& p = b.preprocessing;
    CHECK(o.method == p.method);
    CHECK(o.gain_model == p.gain_model);
    CHECK(o.filter_window == p.filter_window);
    CHECK(a.min_window_periods == b.min_window_periods);
    CHECK(o.variance_window == p.variance_window);
    CHECK(o.window_blocks == p.window_blocks);
    CHECK(o.fast_median == p.fast_median);
    CHECK(o.fast_median_min_points == p.fast_median_min_points);
    CHECK(o.n_iter == p.n_iter);
    CHECK(o.block_scales == p.block_scales);
    CHECK(o.block_sigma == p.block_sigma);
    CHECK(o.min_good_fraction == p.min_good_fraction);
    CHECK(o.clip_sigma == p.clip_sigma);
    CHECK(o.zap_periodic == p.zap_periodic);
    CHECK(o.zap_sigma == p.zap_sigma);
    CHECK(o.zap_whiten_bins == p.zap_whiten_bins);
    REQUIRE(o.birdies.size() == p.birdies.size());
    for (std::size_t i = 0; i < o.birdies.size(); ++i) {
        CHECK(o.birdies[i].freq == p.birdies[i].freq);
        CHECK(o.birdies[i].width == p.birdies[i].width);
    }
    CHECK(a.f_min == b.f_min);
    CHECK(a.f_max == b.f_max);
    CHECK(a.acc_min == b.acc_min);
    CHECK(a.acc_max == b.acc_max);
    CHECK(a.jerk_min == b.jerk_min);
    CHECK(a.jerk_max == b.jerk_max);
    REQUIRE(a.orbit.has_value() == b.orbit.has_value());
    if (a.orbit) {
        CHECK(a.orbit->order == b.orbit->order);
        CHECK(a.orbit->p_orb_min == b.orbit->p_orb_min);
        CHECK(a.orbit->p_orb_min_tobs == b.orbit->p_orb_min_tobs);
        CHECK(a.orbit->m_c_max == b.orbit->m_c_max);
        CHECK(a.orbit->m_p_min == b.orbit->m_p_min);
    }
    CHECK(a.nbins == b.nbins);
    CHECK(a.eta == b.eta);
    CHECK(a.ducy_max == b.ducy_max);
    CHECK(a.wtsp == b.wtsp);
    CHECK(a.snr_min == b.snr_min);
    CHECK(a.use_fourier == b.use_fourier);
    CHECK(a.use_boxcar_kadane == b.use_boxcar_kadane);
    CHECK(a.nthreads == b.nthreads);
    CHECK(a.backend == b.backend);
    CHECK(a.device == b.device);
    CHECK(a.max_process_memory_gb == b.max_process_memory_gb);
    CHECK(a.octave_scale == b.octave_scale);
    CHECK(a.nbins_max == b.nbins_max);
    CHECK(a.nbins_min_lossy_bf == b.nbins_min_lossy_bf);
    CHECK(a.bseg_brute == b.bseg_brute);
    CHECK(a.max_passing_candidates == b.max_passing_candidates);
    CHECK(a.outdir == b.outdir);
    CHECK(a.prefix == b.prefix);
    const auto& c = a.candidates;
    const auto& d = b.candidates;
    CHECK(c.enabled == d.enabled);
    CHECK(c.cluster == d.cluster);
    CHECK(c.grouping == d.grouping);
    CHECK(c.radius == d.radius);
    CHECK(c.phase_radius == d.phase_radius);
    CHECK(c.retain_detections == d.retain_detections);
    CHECK(c.flag_harmonics == d.flag_harmonics);
    CHECK(c.remove_harmonics == d.remove_harmonics);
    CHECK(c.harmonic_denominator_max == d.harmonic_denominator_max);
    CHECK(c.harmonic_phase_distance == d.harmonic_phase_distance);
    CHECK(c.harmonic_snr_distance == d.harmonic_snr_distance);
    CHECK(c.max_candidates == d.max_candidates);
    CHECK(c.memory_gb == d.memory_gb);
    CHECK(c.temporary_directory == d.temporary_directory);
}

void check_same_ep(const EPTomlConfig& a, const EPTomlConfig& b) {
    check_same(a, b);
    CHECK(a.target_snr == b.target_snr);
    CHECK(a.poly_basis == b.poly_basis);
    CHECK(a.min_pd == b.min_pd);
    CHECK(a.ref_ducy == b.ref_ducy);
    CHECK(a.prune_poly_order == b.prune_poly_order);
    CHECK(a.n_runs == b.n_runs);
    CHECK(a.ref_segs == b.ref_segs);
    CHECK(a.nsegments == b.nsegments);
    CHECK(a.p_orb_min == b.p_orb_min);
    CHECK(a.m_c_max == b.m_c_max);
    CHECK(a.m_p_min == b.m_p_min);
    CHECK(a.propagator_significance == b.propagator_significance);
    CHECK(a.validation_significance == b.validation_significance);
    CHECK(a.use_conservative_tile == b.use_conservative_tile);
    CHECK(a.plan_cache == b.plan_cache);
    CHECK(a.use_plan_cache == b.use_plan_cache);
}

} // namespace

TEST_CASE("FFATomlConfig to_toml_string gives back every setting",
          "[config][ffa]") {
    FFATomlConfig cfg;
    set_non_defaults(cfg);
    const auto text = cfg.to_toml_string();
    CAPTURE(text);
    check_same(cfg, FFATomlConfig::from_string(text));
    // Unset optional settings are left out, as in a file that omits them.
    FFATomlConfig plain;
    const auto plain_text = plain.to_toml_string();
    CHECK_THAT(plain_text, !Catch::Matchers::ContainsSubstring("acc_min"));
    CHECK_THAT(plain_text, !Catch::Matchers::ContainsSubstring("bseg_brute"));
    check_same(plain, FFATomlConfig::from_string(plain_text));
}

TEST_CASE("FFATomlConfig to_toml_string keeps an orbit table",
          "[config][ffa]") {
    auto cfg        = FFATomlConfig::from_string(R"(
[search]
f_min = 100.0
f_max = 200.0
[search.orbit]
order = "jerk"
p_orb_min_tobs = 5.0
m_c_max = 8.0
)");
    const auto text = cfg.to_toml_string();
    CAPTURE(text);
    check_same(cfg, FFATomlConfig::from_string(text));
}

TEST_CASE("EPTomlConfig to_toml_string gives back every setting",
          "[config][ep]") {
    EPTomlConfig cfg;
    set_non_defaults(cfg);
    cfg.target_snr              = 9.5;
    cfg.poly_basis              = "chebyshev";
    cfg.min_pd                  = 0.25F;
    cfg.ref_ducy                = 0.15F;
    cfg.prune_poly_order        = 4;
    cfg.ref_segs                = std::vector<loki::SizeType>{0, 3, 7};
    cfg.nsegments               = 64;
    cfg.p_orb_min               = 3600.0;
    cfg.m_c_max                 = 2.0;
    cfg.m_p_min                 = 1.3;
    cfg.propagator_significance = 2.5;
    cfg.validation_significance = 4.5;
    cfg.use_conservative_tile   = true;
    cfg.plan_cache              = std::filesystem::path("plans/ep.h5");
    cfg.use_plan_cache          = false;
    const auto text             = cfg.to_toml_string();
    CAPTURE(text);
    check_same_ep(cfg, EPTomlConfig::from_string(text));

    cfg.ref_segs.reset();
    cfg.n_runs = 16;
    check_same_ep(cfg, EPTomlConfig::from_string(cfg.to_toml_string()));
}

TEST_CASE("The default templates survive to_toml_string", "[config]") {
    const auto ffa =
        FFATomlConfig::from_string(FFATomlConfig::default_toml_string());
    check_same(ffa, FFATomlConfig::from_string(ffa.to_toml_string()));
    const auto ep =
        EPTomlConfig::from_string(EPTomlConfig::default_toml_string());
    check_same_ep(ep, EPTomlConfig::from_string(ep.to_toml_string()));
}

TEST_CASE("write_run_provenance writes and replaces root attributes",
          "[config][provenance]") {
    const loki::test::TempDir dir("loki-provenance");
    const auto path = dir / "provenance.h5";
    {
        HighFive::File file(path.string(), HighFive::File::Overwrite);
        file.createAttribute("config_toml", std::string("raw = 1"));
    }
    loki::search::write_run_provenance(path, {.config_effective = "a = 1\n",
                                              .command_line = "loki search ffa",
                                              .loki_version = "1.2.3"});
    loki::search::write_run_provenance(path, {.config_effective = "a = 2\n",
                                              .command_line     = "",
                                              .loki_version     = ""});
    {
        const HighFive::File file(path.string(), HighFive::File::ReadOnly);
        CHECK(file.getAttribute("config_effective").read<std::string>() ==
              "a = 2\n");
        CHECK(file.getAttribute("command_line").read<std::string>() ==
              "loki search ffa");
        CHECK(file.getAttribute("loki_version").read<std::string>() == "1.2.3");
        CHECK(file.getAttribute("config_toml").read<std::string>() ==
              "raw = 1");
    }
    std::filesystem::remove(path);
    REQUIRE_THROWS_AS(loki::search::write_run_provenance(path, {}),
                      std::runtime_error);
}
