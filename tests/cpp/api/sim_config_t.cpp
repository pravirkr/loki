#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "loki/simulation/configs.hpp"

namespace {

using loki::simulation::SimulateTomlConfig;

std::string message_for(std::string_view toml) {
    try {
        (void)SimulateTomlConfig::from_string(toml);
    } catch (const std::exception& ex) {
        return ex.what();
    }
    return {};
}

} // namespace

TEST_CASE("SimulateTomlConfig keeps the defaults of missing keys",
          "[config][simulate]") {
    // A table with only some of its keys used to dereference a null node.
    const auto cfg = SimulateTomlConfig::from_string("[signal]\nperiod = 0.25\n"
                                                     "[modulation]\nacc = 5\n"
                                                     "[output]\nformat = "
                                                     "\"dat\"\n");
    const SimulateTomlConfig defaults;
    CHECK(cfg.period == 0.25);
    CHECK(cfg.dt == defaults.dt);
    CHECK(cfg.nsamps == defaults.nsamps);
    CHECK(cfg.snr == defaults.snr);
    CHECK(cfg.shape == defaults.shape);
    CHECK_FALSE(cfg.seed.has_value());
    CHECK(cfg.mod_type == defaults.mod_type);
    CHECK(cfg.mod.acc == 5.0);
    CHECK(cfg.mod.m_p == defaults.mod.m_p);
    CHECK_FALSE(cfg.mod.x_orb.has_value());
    CHECK(cfg.output == defaults.output);
    CHECK(cfg.format == "dat");
    CHECK(SimulateTomlConfig::from_string("").period == defaults.period);
}

TEST_CASE("SimulateTomlConfig reads every key", "[config][simulate]") {
    const auto cfg = SimulateTomlConfig::from_string(R"(
[signal]
period = 1
dt = 6.4e-5
nsamps = 4096
snr = 12.0
ducy = 0.05
shape = "boxcar"
phi0 = 0.25
seed = 42

[modulation]
type = "circular"
mod_tref = 10.0
coeffs = [1, 2.5]
p_orb = 7200.0
psi = 0.5
x_orb = 1.5
m_c = 0.2
m_p = 1.3
sin_i = 0.9
a = 2.0
t0 = 3.0

[output]
path = "out.tim"
)");
    CHECK(cfg.period == 1.0);
    CHECK(cfg.dt == 6.4e-5);
    CHECK(cfg.nsamps == 4096);
    CHECK(cfg.shape == "boxcar");
    CHECK(cfg.seed == 42U);
    CHECK(cfg.mod_type == "circular");
    CHECK(cfg.mod_tref == 10.0);
    CHECK(cfg.mod.coeffs == std::vector<double>{1.0, 2.5});
    CHECK(cfg.mod.x_orb == 1.5);
    CHECK(cfg.mod.m_c == 0.2);
    CHECK(cfg.mod.sin_i == 0.9);
    CHECK(cfg.output == std::filesystem::path("out.tim"));
}

TEST_CASE("SimulateTomlConfig rejects unknown tables and keys",
          "[config][simulate]") {
    CHECK_THAT(
        message_for("[signal]\nperoid = 0.5\n"),
        Catch::Matchers::ContainsSubstring("signal.peroid: unknown key"));
    CHECK_THAT(message_for("[noise]\nsigma = 1.0\n"),
               Catch::Matchers::ContainsSubstring("'noise'"));
    CHECK_THAT(message_for("signal = 1\n"),
               Catch::Matchers::ContainsSubstring("signal must be a table"));
}

TEST_CASE("SimulateTomlConfig rejects values of the wrong type",
          "[config][simulate]") {
    CHECK_THAT(message_for("[signal]\nperiod = \"0.5\"\n"),
               Catch::Matchers::ContainsSubstring("signal.period must be a "
                                                  "number"));
    CHECK_THAT(
        message_for("[signal]\nsnr = true\n"),
        Catch::Matchers::ContainsSubstring("signal.snr must be a number"));
    CHECK_THAT(message_for("[signal]\nnsamps = 4096.0\n"),
               Catch::Matchers::ContainsSubstring("signal.nsamps must be an "
                                                  "integer"));
    CHECK_THAT(message_for("[signal]\nnsamps = -1\n"),
               Catch::Matchers::ContainsSubstring("non-negative"));
    CHECK_THAT(message_for("[signal]\nseed = -1\n"),
               Catch::Matchers::ContainsSubstring("non-negative"));
    CHECK_THAT(message_for("[signal]\nshape = 1\n"),
               Catch::Matchers::ContainsSubstring("signal.shape must be a "
                                                  "string"));
    CHECK_THAT(message_for("[modulation]\ncoeffs = [1.0, \"x\"]\n"),
               Catch::Matchers::ContainsSubstring("modulation.coeffs"));
    CHECK_THAT(message_for("[modulation]\ncoeffs = 1.0\n"),
               Catch::Matchers::ContainsSubstring("modulation.coeffs"));
}

TEST_CASE("SimulateTomlConfig rejects nan and inf by their literal",
          "[config][simulate]") {
    CHECK_THAT(message_for("[signal]\nsnr = nan\n"),
               Catch::Matchers::ContainsSubstring("signal.snr must be finite"));
    CHECK_THAT(message_for("[modulation]\nacc = -inf\n"),
               Catch::Matchers::ContainsSubstring("modulation.acc"));
    CHECK_THAT(message_for("[signal\n"),
               Catch::Matchers::ContainsSubstring("TOML parse error at line"));
}

TEST_CASE("SimulateTomlConfig default document loads", "[config][simulate]") {
    const auto cfg = SimulateTomlConfig::from_string(
        SimulateTomlConfig::default_toml_string());
    CHECK(cfg.period == 0.5);
    CHECK(cfg.dt == 0.0001);
    CHECK(cfg.nsamps == 2097152);
    CHECK(cfg.format == "tim");

    const auto path =
        std::filesystem::temp_directory_path() / "loki_sim_config_t.toml";
    SimulateTomlConfig::write_default(path);
    const auto loaded = SimulateTomlConfig::load(path);
    CHECK(loaded.period == cfg.period);
    CHECK(loaded.output == cfg.output);
    std::filesystem::remove(path);
    REQUIRE_THROWS_AS(SimulateTomlConfig::load(path), std::runtime_error);
}

TEST_CASE("SimulateTomlConfig suggests the key a typo meant",
          "[config][simulate]") {
    CHECK_THAT(message_for("[signal]\nperoid = 0.5\n"),
               Catch::Matchers::ContainsSubstring("(did you mean 'period'?)"));
    CHECK_THAT(message_for("[sginal]\nperiod = 0.5\n"),
               Catch::Matchers::ContainsSubstring("(did you mean 'signal'?)"));
    // A different word, not a typo: no suggestion.
    CHECK_THAT(message_for("[signal]\ncolour = 1\n"),
               !Catch::Matchers::ContainsSubstring("did you mean"));
}

TEST_CASE("SimulateTomlConfig load names the file of an error",
          "[config][simulate]") {
    const auto path =
        std::filesystem::temp_directory_path() / "loki_sim_config_bad_t.toml";
    {
        std::ofstream file(path);
        file << "[signal]\nsnr = \"high\"\n";
    }
    try {
        (void)SimulateTomlConfig::load(path);
        FAIL("load accepted a string snr");
    } catch (const std::invalid_argument& ex) {
        CHECK_THAT(ex.what(),
                   Catch::Matchers::StartsWith(path.string() + ": ") &&
                       Catch::Matchers::ContainsSubstring(
                           "signal.snr must be a number"));
    }
    std::filesystem::remove(path);
}

TEST_CASE("SimulateTomlConfig validate names the key and flag",
          "[config][simulate]") {
    const auto validate_message = [](const SimulateTomlConfig& cfg) {
        try {
            cfg.validate();
        } catch (const std::invalid_argument& ex) {
            return std::string(ex.what());
        }
        return std::string{};
    };
    const auto defaults = SimulateTomlConfig::from_string(
        SimulateTomlConfig::default_toml_string());
    CHECK(validate_message(defaults).empty());

    auto cfg  = defaults;
    cfg.ducy  = 1.5;
    cfg.shape = "gausian";
    CHECK_THAT(
        validate_message(cfg),
        Catch::Matchers::ContainsSubstring(
            "signal.ducy (--ducy) must be in (0, 1)") &&
            Catch::Matchers::ContainsSubstring("(did you mean 'gaussian'?)"));

    cfg                = defaults;
    cfg.period         = 0.0;
    cfg.mod_type       = "spiral";
    cfg.format         = "fits";
    const auto message = validate_message(cfg);
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("signal.period"));
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("modulation.type"));
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("output.format"));
}

TEST_CASE("SimulateTomlConfig to_toml_string gives back every setting",
          "[config][simulate]") {
    SimulateTomlConfig cfg;
    cfg.period      = 0.0123;
    cfg.dt          = 6.4e-5;
    cfg.nsamps      = 1U << 18U;
    cfg.snr         = 25.5;
    cfg.ducy        = 0.05;
    cfg.shape       = "von_mises";
    cfg.phi0        = 0.25;
    cfg.seed        = 1234;
    cfg.mod_type    = "circular";
    cfg.mod_tref    = 10.0;
    cfg.mod.shift   = 1.0;
    cfg.mod.vel     = 2.0;
    cfg.mod.acc     = 3.0;
    cfg.mod.jerk    = 4.0;
    cfg.mod.snap    = 5.0;
    cfg.mod.coeffs  = {0.5, -0.25};
    cfg.mod.p_orb   = 7200.0;
    cfg.mod.psi     = 0.3;
    cfg.mod.x_orb   = 1.5;
    cfg.mod.m_c     = 0.2;
    cfg.mod.m_p     = 1.35;
    cfg.mod.sin_i   = 0.9;
    cfg.mod.a       = 0.01;
    cfg.mod.t0      = 12.0;
    cfg.output      = "out dir/sim \"1\".dat";
    cfg.format      = "dat";
    const auto text = cfg.to_toml_string();
    CAPTURE(text);
    const auto back = SimulateTomlConfig::from_string(text);
    CHECK(back.period == cfg.period);
    CHECK(back.dt == cfg.dt);
    CHECK(back.nsamps == cfg.nsamps);
    CHECK(back.snr == cfg.snr);
    CHECK(back.ducy == cfg.ducy);
    CHECK(back.shape == cfg.shape);
    CHECK(back.phi0 == cfg.phi0);
    CHECK(back.seed == cfg.seed);
    CHECK(back.mod_type == cfg.mod_type);
    CHECK(back.mod_tref == cfg.mod_tref);
    CHECK(back.mod.shift == cfg.mod.shift);
    CHECK(back.mod.vel == cfg.mod.vel);
    CHECK(back.mod.acc == cfg.mod.acc);
    CHECK(back.mod.jerk == cfg.mod.jerk);
    CHECK(back.mod.snap == cfg.mod.snap);
    CHECK(back.mod.coeffs == cfg.mod.coeffs);
    CHECK(back.mod.p_orb == cfg.mod.p_orb);
    CHECK(back.mod.psi == cfg.mod.psi);
    CHECK(back.mod.x_orb == cfg.mod.x_orb);
    CHECK(back.mod.m_c == cfg.mod.m_c);
    CHECK(back.mod.m_p == cfg.mod.m_p);
    CHECK(back.mod.sin_i == cfg.mod.sin_i);
    CHECK(back.mod.a == cfg.mod.a);
    CHECK(back.mod.t0 == cfg.mod.t0);
    CHECK(back.output == cfg.output);
    CHECK(back.format == cfg.format);
}
