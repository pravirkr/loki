#include "loki/search/refold.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <highfive/highfive.hpp>

#include "loki/common/types.hpp"
#include "loki/io/candidate_file.hpp"
#include "loki/simulation/modulate.hpp"
#include "loki/simulation/pulse.hpp"

#include "temp_dir.hpp"

using loki::SizeType;
using loki::search::refold;
using loki::search::refold_circular_target;
using loki::search::refold_taylor_target;
using loki::search::RefoldModel;
using loki::search::RefoldOptions;

// Small simulated series: 2^16 samples of 1 ms (65.5 s), a 10 Hz pulse.
namespace {

constexpr SizeType kNsamps = 1U << 16U;
constexpr double kDt       = 1e-3;
constexpr double kFreq     = 10.0;
constexpr double kSnr      = 30.0;
constexpr double kC        = 299792458.0;
constexpr double kTref     = static_cast<double>(kNsamps) * kDt / 2.0;

struct Series {
    std::vector<float> e;
    std::vector<float> v;
};

Series simulate(const char* mod_type,
                const loki::simulation::ModulatorParams& mod) {
    loki::simulation::PulseSignalConfig cfg(1.0 / kFreq, kDt, kNsamps, kSnr,
                                            /*ducy=*/0.1, mod_type, mod,
                                            /*mod_tref=*/std::nullopt,
                                            /*seed=*/7);
    auto ts      = cfg.generate("gaussian");
    const auto e = ts.get_ts_e();
    const auto v = ts.get_ts_v();
    return {.e = {e.begin(), e.end()}, .v = {v.begin(), v.end()}};
}

} // namespace

TEST_CASE("refold recovers an accelerated pulse at its parameters",
          "[refold]") {
    // f * a * T^2 / (8 c) is about 3.5 cycles: a constant-frequency fold
    // smears the pulse.
    constexpr double kAccel = 2.0e5;
    loki::simulation::ModulatorParams mod;
    mod.acc       = kAccel;
    const auto s  = simulate("derivative", mod);
    const auto ac = std::array{kAccel};
    const auto at =
        refold(s.e, s.v, kDt, kTref, refold_taylor_target(kFreq, ac));
    CHECK(at.model == "taylor");
    CHECK(at.nbins == 64);
    CHECK(at.snr_in > 0.7 * kSnr);
    CHECK(at.snr >= at.snr_in);
    CHECK(at.fdot == Catch::Approx(-kFreq * kAccel / kC));
    const auto flat = refold(s.e, s.v, kDt, kTref,
                             refold_taylor_target(kFreq, std::array{0.0}));
    CHECK(flat.snr < 0.5 * at.snr);

    SECTION("the frequency scan finds an offset target") {
        RefoldOptions options;
        const double step = options.df_step_bins / (64.0 * kNsamps * kDt);
        const auto off =
            refold(s.e, s.v, kDt, kTref,
                   refold_taylor_target(kFreq + (64 * step), ac), options);
        // 64 steps drift the pulse by 16 bins over the observation.
        CHECK(off.snr > 1.5F * off.snr_in);
        CHECK(std::abs(off.df + (64 * step)) <= 8 * step);
        CHECK(off.snr == Catch::Approx(at.snr).margin(0.1 * at.snr));
    }
    SECTION("an fdot scan finds an offset acceleration") {
        RefoldOptions options;
        options.fd_steps = 4;
        options.fd_span  = 4.0 * kFreq * 1.0e4 / kC;
        const auto off   = refold(
            s.e, s.v, kDt, kTref,
            refold_taylor_target(kFreq, std::array{kAccel + 2.0e4}), options);
        CHECK(off.snr > off.snr_in);
        CHECK(std::abs((off.fdot + off.dfd) - at.fdot) < options.fd_span / 2.0);
    }
    SECTION("the time parts carry the signal") {
        RefoldOptions options;
        options.split    = 4;
        const auto parts = refold(s.e, s.v, kDt, kTref,
                                  refold_taylor_target(kFreq, ac), options);
        REQUIRE(parts.split_snr.size() == 4);
        double sum_sq = 0.0;
        for (const float x : parts.split_snr) {
            CHECK(x > 0.25 * parts.snr);
            sum_sq += static_cast<double>(x) * x;
        }
        CHECK(std::sqrt(sum_sq) ==
              Catch::Approx(parts.snr).margin(0.25 * parts.snr));
    }
}

TEST_CASE("refold's own Taylor phase equals the leaf model below five "
          "parameters",
          "[refold]") {
    loki::simulation::ModulatorParams mod;
    mod.acc      = 1.0e5;
    mod.jerk     = 2.0e3;
    const auto s = simulate("derivative", mod);
    auto target  = refold_taylor_target(kFreq, std::array{1.0e5, 2.0e3});
    REQUIRE(target.n_params == 3);
    const auto taylor = refold(s.e, s.v, kDt, kTref, target);
    target.model      = RefoldModel::kLeaf;
    const auto leaf   = refold(s.e, s.v, kDt, kTref, target);
    CHECK(leaf.model == "taylor");
    CHECK(leaf.snr_in == taylor.snr_in);
    CHECK(leaf.snr == taylor.snr);
    CHECK(leaf.df == taylor.df);
    CHECK(leaf.profile == taylor.profile);
}

TEST_CASE("refold folds a circular orbit with the search's model", "[refold]") {
    // One orbit per observation: Taylor terms through jerk cannot follow it.
    constexpr double kPorb = static_cast<double>(kNsamps) * kDt;
    constexpr double kX    = 0.2; // light-seconds: f x = 2 cycles
    constexpr double kPsi  = 1.0;
    loki::simulation::ModulatorParams mod;
    mod.p_orb    = kPorb;
    mod.psi      = kPsi;
    mod.x_orb    = kX;
    const auto s = simulate("circular", mod);
    // Delay c x sin(w dt + psi): derivatives at t_ref, and the observed
    // frequency there.
    const double w  = 2.0 * std::numbers::pi / kPorb;
    const double cx = kC * kX;
    const std::array<double, 4> derivs{-cx * w * w * std::sin(kPsi),
                                       -cx * w * w * w * std::cos(kPsi),
                                       cx * std::pow(w, 4) * std::sin(kPsi),
                                       cx * std::pow(w, 5) * std::cos(kPsi)};
    const double f_obs = kFreq * (1.0 - (kX * w * std::cos(kPsi)));
    const auto orbit =
        refold(s.e, s.v, kDt, kTref, refold_circular_target(f_obs, derivs));
    CHECK(orbit.model == "circular_snap");
    CHECK(orbit.snr_in > 0.7 * kSnr);
    const auto jerk_only =
        refold(s.e, s.v, kDt, kTref,
               refold_taylor_target(f_obs, std::span(derivs).first(2)));
    CHECK(jerk_only.model == "taylor");
    CHECK(jerk_only.snr < 0.6 * orbit.snr);
}

TEST_CASE("refold checks its inputs", "[refold]") {
    const std::vector<float> e(1000, 0.0F);
    const std::vector<float> v(1000, 1.0F);
    const auto target = refold_taylor_target(kFreq, std::array<double, 0>{});
    CHECK(target.n_params == 1);
    CHECK_THROWS_AS(refold(e, std::span(v).first(999), kDt, 0.5, target),
                    std::runtime_error);
    RefoldOptions options;
    options.nsub = 2000;
    CHECK_THROWS_AS(refold(e, v, kDt, 0.5, target, options),
                    std::runtime_error);
    options.nsub  = 8;
    options.split = 9;
    CHECK_THROWS_AS(refold(e, v, kDt, 0.5, target, options),
                    std::runtime_error);
    CHECK_THROWS_AS(refold_taylor_target(-1.0, std::array{0.0}),
                    std::runtime_error);
    CHECK_THROWS_AS(
        refold_taylor_target(kFreq, std::array{1.0, 1.0, 1.0, 1.0, 1.0}),
        std::runtime_error);
    // Trailing zero derivatives are dropped.
    CHECK(refold_taylor_target(kFreq, std::array{3.0, 0.0, 0.0}).n_params == 2);

    const double nan =
        std::bit_cast<double>(std::uint64_t{0x7FF8000000000000ULL});
    RefoldOptions bad_step;
    bad_step.df_step_bins = 0.0;
    CHECK_THROWS_AS(refold(e, v, kDt, 0.5, target, bad_step),
                    std::runtime_error);
    bad_step.df_step_bins = -0.1;
    CHECK_THROWS_AS(refold(e, v, kDt, 0.5, target, bad_step),
                    std::runtime_error);
    RefoldOptions bad_span_t;
    bad_span_t.df_span_t = -1.0;
    CHECK_THROWS_AS(refold(e, v, kDt, 0.5, target, bad_span_t),
                    std::runtime_error);
    RefoldOptions bad_fd_span;
    bad_fd_span.fd_span = -1.0;
    CHECK_THROWS_AS(refold(e, v, kDt, 0.5, target, bad_fd_span),
                    std::runtime_error);
    CHECK_THROWS_AS(refold(e, v, nan, 0.5, target), std::runtime_error);
    CHECK_THROWS_AS(refold(e, v, kDt, nan, target), std::runtime_error);
    CHECK_THROWS_AS(refold_taylor_target(nan, std::array{0.0}),
                    std::runtime_error);
    CHECK_THROWS_AS(refold_taylor_target(kFreq, std::array{nan}),
                    std::runtime_error);
}

TEST_CASE("refold handles partially and fully masked series without Gibbs "
          "blowup",
          "[refold]") {
    std::mt19937_64 rng(42);
    std::normal_distribution<float> norm(0.0F, 1.0F);
    std::vector<float> e(kNsamps);
    std::vector<float> v(kNsamps, 1.0F);
    for (SizeType i = 0; i < kNsamps; ++i) {
        e[i] = norm(rng);
    }
    // Completely mask the first split and sprinkle masked blocks in the second.
    for (SizeType i = 0; i < kNsamps / 2; ++i) {
        e[i] = 0.0F;
        v[i] = 0.0F;
    }
    for (SizeType blk = (kNsamps / 2) + 1000; blk + 250 < kNsamps;
         blk += 3000) {
        for (SizeType i = blk; i < blk + 250; ++i) {
            e[i] = 0.0F;
            v[i] = 0.0F;
        }
    }
    RefoldOptions opts;
    opts.nsub         = 16;
    opts.split        = 2;
    const auto target = refold_taylor_target(kFreq, std::array<double, 0>{});
    const auto res    = refold(e, v, kDt, kTref, target, opts);
    CHECK(res.snr < 10.0F);
    REQUIRE(res.split_snr.size() == 2);
    CHECK(res.split_snr[0] == 0.0F);
    REQUIRE_FALSE(res.profile.empty());
    for (const float p : res.profile) {
        const volatile auto bits = std::bit_cast<std::uint32_t>(p);
        CHECK((bits & 0x7F800000U) != 0x7F800000U);
    }

    const std::vector<float> zero_e(4096, 0.0F);
    const std::vector<float> zero_v(4096, 0.0F);
    const auto empty = refold(zero_e, zero_v, kDt, 2048.0 * kDt, target);
    CHECK(empty.snr == 0.0F);
    CHECK(empty.snr_in == 0.0F);
    CHECK(empty.width == 0U);
    REQUIRE_FALSE(empty.profile.empty());
    for (const float p : empty.profile) {
        CHECK(p == 0.0F);
    }
}

TEST_CASE("legacy_ffa candidates carry reference_time and resolve by id for "
          "refold",
          "[refold][candidates]") {
    constexpr double kAccel = 2.0e5;
    loki::simulation::ModulatorParams mod;
    mod.acc      = kAccel;
    const auto s = simulate("derivative", mod);

    const loki::test::TempDir dir("loki-legacy-ffa-refold");
    const auto path         = dir / "legacy.h5";
    const std::uint64_t cid = (std::uint64_t{2} << 32U) | 5ULL;
    {
        HighFive::File h5(path.string(), HighFive::File::Truncate);
        h5.createAttribute("ffa_version", std::string("0.1.0-cpp"));
        h5.createAttribute("nsamps", static_cast<std::uint64_t>(kNsamps));
        h5.createAttribute("tsamp", kDt);
        h5.createAttribute("tobs", static_cast<double>(kNsamps) * kDt);
        h5.createAttribute("param_names",
                           std::vector<std::string>{"accel", "freq"});
        h5.createDataSet("snr", std::vector<float>{static_cast<float>(kSnr)});
        h5.createDataSet("param_sets",
                         std::vector<std::vector<double>>{{kAccel, kFreq}});
        h5.createDataSet("width", std::vector<std::uint16_t>{4U});
        h5.createDataSet("nbins", std::vector<std::uint16_t>{64U});
        h5.createDataSet("detection_id", std::vector<std::uint64_t>{cid});
    }

    const auto summary = loki::io::read_candidate_file_summary(path);
    CHECK(summary.layout == "legacy_ffa");
    CHECK(summary.reference_time == Catch::Approx(kTref));
    REQUIRE(summary.top.size() == 1);

    loki::io::CandidateFileSummary meta;
    const auto row = loki::io::read_candidate_by_id(path, cid, &meta);
    CHECK(meta.reference_time == Catch::Approx(kTref));
    CHECK(row.id == cid);
    CHECK(row.row == 0);
    CHECK(row.width == 4U);
    CHECK(row.nbins == 64U);
    CHECK(row.source == "region 2");
    REQUIRE(row.params.size() == 2);
    CHECK(row.params[0] == Catch::Approx(kAccel));
    CHECK(row.params[1] == Catch::Approx(kFreq));
    CHECK_THROWS_AS(loki::io::read_candidate_by_id(path, ~0ULL),
                    std::invalid_argument);

    const auto target =
        refold_taylor_target(row.params.back(), std::span(row.params).first(1));
    const auto res = refold(s.e, s.v, kDt, meta.reference_time, target);
    CHECK(res.snr_in > 0.7 * kSnr);
    CHECK(res.snr >= res.snr_in);
}
