#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <highfive/highfive.hpp>

#include "loki/algorithms/prune_rfi.hpp"
#include "loki/common/backend.hpp"
#include "loki/common/types.hpp"
#include "loki/io/candidate_file.hpp"
#include "loki/pipelines/ep_freq_sweep.hpp"
#include "loki/search/candidates.hpp"
#include "loki/search/configs.hpp"
#include "loki/search/provenance.hpp"

#include "temp_dir.hpp"

using loki::ParamLimit;
using loki::SizeType;
using loki::algorithms::HarvestDefaults;
using loki::algorithms::make_default_harvest_scheme;
using loki::algorithms::PruneRFIConfig;
using loki::pipelines::EPFreqSweep;
using loki::search::CandidateOptions;
using loki::search::EPRFIOptions;
using loki::search::EPTomlConfig;
using loki::search::PulsarSearchConfig;
using loki::test::TempDir;

namespace {

/// An EP document with @p rfi spliced in as its [ep.rfi] table (none if
/// empty) and @p perf in [performance].
std::string make_doc(const std::string& rfi,
                     const std::string& perf = "nthreads = 2\n") {
    std::string doc = "[input]\n"
                      "nsamps = 65536\n"
                      "tsamp = 0.000064\n"
                      "[search]\n"
                      "f_min = 140.0\n"
                      "f_max = 142.0\n"
                      "acc_min = -10.0\n"
                      "acc_max = 10.0\n"
                      "nbins = 32\n"
                      "[ep]\n"
                      "n_runs = 2\n"
                      "nsegments = 8\n";
    if (!rfi.empty()) {
        doc += "[ep.rfi]\n" + rfi;
    }
    return doc + "[performance]\n" + perf;
}

std::string message_for(const std::string& toml) {
    try {
        (void)EPTomlConfig::from_string(toml);
    } catch (const std::invalid_argument& err) {
        return err.what();
    }
    return {};
}

constexpr auto kFullRfi = "pulsar_mask = [[49.9, 50.1], [59.5, 60.5]]\n"
                          "n_harmonics = 3\n"
                          "fractional_harmonics = true\n"
                          "harvest = true\n"
                          "harvest_min_level = 4\n"
                          "harvest_offset = 6.5\n"
                          "harvest_min_snr = 12.0\n"
                          "harvest_mask_ntiles = 2.0\n"
                          "max_harvests = 64\n"
                          "harvest_store_folds = false\n"
                          "impulsive_veto = true\n"
                          "impulsive_kappa = 5.0\n"
                          "impulsive_min_level = 3\n"
                          "impulsive_min_snr = 9.0\n";

PulsarSearchConfig make_sweep_cfg(double f_min, double f_max) {
    const std::vector<ParamLimit> limits{
        ParamLimit{.min = -10.0, .max = 10.0},
        ParamLimit{.min = f_min, .max = f_max},
    };
    return {/*nsamps=*/1U << 16U,
            /*tsamp=*/64e-6,
            /*nbins=*/32,
            /*eta=*/1.0,
            limits,
            /*ducy_max=*/0.3,
            /*wtsp=*/1.5,
            /*use_fourier=*/false,
            /*nthreads=*/2,
            /*max_process_memory_gb=*/4.0,
            /*octave_scale=*/2.0,
            /*nbins_max=*/1024,
            /*nbins_min_lossy_bf=*/64,
            /*bseg_brute=*/1024,
            /*nsegments=*/8,
            /*snr_min=*/5.0,
            /*max_passing_candidates=*/1U << 22U,
            /*prune_poly_order=*/2};
}

/// Unit-variance noise plus, for each (frequency, amplitude), a train of
/// top-hat pulses 5 % of the period wide.
std::pair<std::vector<float>, std::vector<float>>
make_series(SizeType nsamps,
            double tsamp,
            const std::vector<std::pair<double, float>>& pulsars) {
    std::vector<float> ts_e(nsamps);
    std::vector<float> ts_v(nsamps, 1.0F);
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    for (SizeType i = 0; i < nsamps; ++i) {
        ts_e[i]        = noise(rng);
        const double t = static_cast<double>(i) * tsamp;
        for (const auto& [freq, amp] : pulsars) {
            const double phase = (t * freq) - std::floor(t * freq);
            if (phase < 0.05) {
                ts_e[i] += amp;
            }
        }
    }
    return {std::move(ts_e), std::move(ts_v)};
}

/// Frequencies (last parameter) and scores of the final leaves of a run.
std::vector<std::pair<double, float>> final_leaves(const HighFive::Group& run) {
    const auto dims = run.getDataSet("param_sets").getDimensions();
    std::vector<std::pair<double, float>> out;
    if (dims.empty() || dims[0] == 0) {
        return out;
    }
    std::vector<double> params(dims[0] * dims[1] * dims[2]);
    run.getDataSet("param_sets").read_raw(params.data());
    std::vector<float> scores;
    run.getDataSet("scores").read(scores);
    const std::size_t stride = dims[1] * dims[2];
    // [..., accel, freq, ...]: the frequency is parameter nparams - 1 = 1.
    for (std::size_t i = 0; i < dims[0]; ++i) {
        out.emplace_back(params[(i * stride) + 2], scores[i]);
    }
    return out;
}

} // namespace

TEST_CASE("[ep.rfi] is absent by default and writes nothing",
          "[config][ep][rfi]") {
    const auto cfg = EPTomlConfig::from_string(make_doc(""));
    CHECK_FALSE(cfg.rfi.has_value());
    CHECK_FALSE(cfg.prune_rfi_config().is_active());
    CHECK(cfg.to_toml_string().find("rfi") == std::string::npos);
    CHECK_NOTHROW(cfg.check_rfi_backend());
}

TEST_CASE("[ep.rfi] round-trips every key", "[config][ep][rfi]") {
    const auto cfg = EPTomlConfig::from_string(make_doc(kFullRfi));
    REQUIRE(cfg.rfi.has_value());
    const auto& rfi = *cfg.rfi;
    REQUIRE(rfi.pulsar_mask.size() == 2);
    CHECK(rfi.pulsar_mask[1][0] == 59.5);
    CHECK(rfi.pulsar_mask[1][1] == 60.5);
    CHECK(rfi.n_harmonics == 3);
    CHECK(rfi.fractional_harmonics);
    CHECK(rfi.harvest);
    CHECK(rfi.harvest_min_level == 4);
    CHECK(rfi.harvest_offset == 6.5F);
    CHECK(rfi.harvest_min_snr == 12.0F);
    CHECK(rfi.harvest_mask_ntiles == 2.0);
    CHECK(rfi.max_harvests == 64);
    CHECK_FALSE(rfi.harvest_store_folds);
    CHECK(rfi.impulsive_veto);
    CHECK(rfi.impulsive_kappa == 5.0);
    CHECK(rfi.impulsive_min_level == 3);
    CHECK(rfi.impulsive_min_snr == 9.0F);

    const auto text  = cfg.to_toml_string();
    const auto again = EPTomlConfig::from_string(text);
    REQUIRE(again.rfi.has_value());
    CHECK(again.rfi->pulsar_mask == rfi.pulsar_mask);
    CHECK(again.rfi->harvest_offset == rfi.harvest_offset);
    CHECK(again.rfi->impulsive_min_snr == rfi.impulsive_min_snr);
    CHECK(again.to_toml_string() == text);

    // A table with every mechanism off is kept, and inactive.
    const auto off = EPTomlConfig::from_string(
        make_doc("harvest = false\nharvest_min_snr = 20.0\n"));
    REQUIRE(off.rfi.has_value());
    CHECK_FALSE(off.rfi->is_active());
    CHECK_FALSE(off.prune_rfi_config().is_active());
    CHECK(off.rfi->harvest_min_snr == 20.0F);
}

TEST_CASE("[ep.rfi] maps onto PruneRFIConfig", "[config][ep][rfi]") {
    const auto lib =
        EPTomlConfig::from_string(make_doc(kFullRfi)).prune_rfi_config();
    REQUIRE(lib.pulsar_mask.size() == 2);
    CHECK(lib.pulsar_mask[0].f_lo == 49.9);
    CHECK(lib.pulsar_mask[0].f_hi == 50.1);
    // Frequency windows mask every acceleration.
    CHECK(lib.pulsar_mask[0].a_lo == std::numeric_limits<double>::lowest());
    CHECK(lib.pulsar_mask[0].a_hi == std::numeric_limits<double>::max());
    CHECK(lib.n_harmonics == 3);
    CHECK(lib.fractional_harmonics);
    CHECK(lib.harvest_scheme.empty());
    REQUIRE(lib.harvest_defaults.has_value());
    CHECK(lib.harvest_defaults->min_level == 4);
    CHECK(lib.harvest_defaults->offset == 6.5F);
    CHECK(lib.harvest_defaults->min_snr == 12.0F);
    CHECK(lib.has_harvest());
    CHECK(lib.harvest_mask_ntiles == 2.0);
    CHECK(lib.max_harvests == 64);
    CHECK_FALSE(lib.harvest_store_folds);
    CHECK(lib.impulsive_veto);
    CHECK(lib.impulsive_kappa == 5.0);
    CHECK(lib.impulsive_min_level == 3);
    CHECK(lib.impulsive_min_snr == 9.0F);

    const auto no_harvest =
        EPTomlConfig::from_string(make_doc("impulsive_veto = true\n"))
            .prune_rfi_config();
    CHECK_FALSE(no_harvest.has_harvest());
    CHECK(no_harvest.is_active());
}

TEST_CASE("[ep.rfi] rejects unknown keys, wrong types and bad values",
          "[config][ep][rfi]") {
    // A near miss names the key and suggests the right one.
    const auto typo = message_for(make_doc("harvest_min_snrr = 3.0\n"));
    CHECK(typo.find("ep.rfi.harvest_min_snrr") != std::string::npos);
    CHECK(typo.find("harvest_min_snr") != std::string::npos);
    CHECK_FALSE(message_for(make_doc("harvest_scheme = [5.0]\n")).empty());
    CHECK_FALSE(message_for(make_doc("harvest = 1\n")).empty());
    CHECK_FALSE(message_for(make_doc("n_harmonics = -1\n")).empty());
    CHECK(
        message_for(make_doc("n_harmonics = 33\nfractional_harmonics = true\n"))
            .find("ep.rfi.fractional_harmonics") != std::string::npos);
    CHECK_FALSE(message_for(make_doc("pulsar_mask = [50.0, 51.0]\n")).empty());
    CHECK_FALSE(
        message_for(make_doc("pulsar_mask = [[50.0, 51.0, 52.0]]\n")).empty());
    CHECK(message_for(make_doc("pulsar_mask = [[51.0, 50.0]]\n"))
              .find("ep.rfi.pulsar_mask") != std::string::npos);
    CHECK_FALSE(message_for(make_doc("pulsar_mask = [[0.0, 1.0]]\n")).empty());
    CHECK_FALSE(message_for(make_doc("pulsar_mask = [[nan, 1.0]]\n")).empty());
    CHECK(message_for(make_doc("harvest_min_level = 0\n"))
              .find("ep.rfi.harvest_min_level") != std::string::npos);
    CHECK_FALSE(message_for(make_doc("harvest_mask_ntiles = 0.0\n")).empty());
    CHECK_FALSE(message_for(make_doc("max_harvests = 0\n")).empty());
    CHECK_FALSE(message_for(make_doc("impulsive_kappa = -1.0\n")).empty());
    // The RFI keys belong in [ep.rfi], not in [ep] or a top-level [rfi].
    auto in_ep        = make_doc("");
    const auto anchor = std::string("nsegments = 8\n");
    in_ep.insert(in_ep.find(anchor) + anchor.size(), "harvest = true\n");
    CHECK_FALSE(message_for(in_ep).empty());
    CHECK_FALSE(message_for(make_doc("") + "[rfi]\nharvest = true\n").empty());
}

TEST_CASE("[ep.rfi] is checked again after command-line overrides",
          "[config][ep][rfi]") {
    auto cfg = EPTomlConfig::from_string(make_doc("harvest = true\n"));
    CHECK_NOTHROW(cfg.to_ep_search_config());
    cfg.rfi->pulsar_mask.push_back({60.0, 50.0});
    CHECK_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);
}

TEST_CASE("An active [ep.rfi] needs the CPU backend", "[config][ep][rfi]") {
    auto cfg = EPTomlConfig::from_string(
        make_doc("harvest = true\n", "nthreads = 2\nbackend = \"cuda\"\n"));
    try {
        cfg.check_rfi_backend();
        FAIL("an active [ep.rfi] with backend cuda must be rejected");
    } catch (const std::invalid_argument& err) {
        CHECK(std::string(err.what()).find("backend") != std::string::npos);
    }
    CHECK_THROWS_AS(cfg.to_ep_search_config(), std::invalid_argument);
    // The command line can switch the backend back; an inactive table is
    // fine on any backend.
    cfg.backend = loki::Backend::kCPU;
    CHECK_NOTHROW(cfg.check_rfi_backend());
    cfg.backend      = loki::Backend::kCUDA;
    cfg.rfi->harvest = false;
    CHECK_NOTHROW(cfg.check_rfi_backend());
}

TEST_CASE("PruneRFIConfig expands harvest defaults from a threshold scheme",
          "[rfi]") {
    PruneRFIConfig rfi;
    CHECK_FALSE(rfi.is_active());
    rfi.harvest_defaults =
        HarvestDefaults{.min_level = 3, .offset = 2.0F, .min_snr = 6.0F};
    CHECK(rfi.has_harvest());
    CHECK(rfi.is_active());
    CHECK_NOTHROW(rfi.validate(8));

    const std::vector<float> thresholds{1.0F, 2.0F, 3.0F, 4.5F,
                                        5.0F, 6.0F, 7.0F};
    const auto expanded = rfi.expand_harvest(thresholds);
    CHECK_FALSE(expanded.harvest_defaults.has_value());
    CHECK(expanded.harvest_scheme ==
          make_default_harvest_scheme(thresholds, 3, 2.0F, 6.0F));
    CHECK(expanded.harvest_scheme[1] == loki::algorithms::kHarvestDisabled);
    CHECK(expanded.harvest_scheme[2] == 6.0F);
    CHECK(expanded.harvest_scheme[3] == 6.5F);
    CHECK_NOTHROW(expanded.validate(8));
    // Without defaults, an unchanged copy.
    CHECK(expanded.expand_harvest(thresholds).harvest_scheme ==
          expanded.harvest_scheme);

    // An explicit scheme and the defaults are exclusive.
    auto both           = rfi;
    both.harvest_scheme = expanded.harvest_scheme;
    CHECK_THROWS(both.validate(8));
    // min_level must name a pruning level.
    auto late                        = rfi;
    late.harvest_defaults->min_level = 8;
    CHECK_THROWS(late.validate(8));
    CHECK_NOTHROW(late.validate(16));
    late.harvest_defaults->min_level = 0;
    CHECK_THROWS(late.validate(16));
}

TEST_CASE("EPFreqSweep expands the harvest scheme for each chunk",
          "[ep_freq_sweep][rfi]") {
    // 70-145 Hz spans two period octaves (64 and 32 bins), so the chunks
    // have different threshold schemes.
    const TempDir outdir("loki-ep-rfi-chunks");
    const auto cfg = make_sweep_cfg(70.0, 145.0);
    const auto [ts_e, ts_v] =
        make_series(cfg.get_nsamps(), cfg.get_tsamp(), {});
    PruneRFIConfig rfi;
    const HarvestDefaults defaults{
        .min_level = 3, .offset = 4.0F, .min_snr = 9.0F};
    rfi.harvest_defaults = defaults;
    EPFreqSweep sweep(cfg, /*show_progress=*/false, 0.1F, "taylor", 0.1F, rfi,
                      std::nullopt, std::nullopt, std::vector<SizeType>{4U});
    sweep.execute(ts_e, ts_v, outdir.path(), "rfi");

    const HighFive::File h5((outdir / "rfi_ep_results.h5").string(),
                            HighFive::File::ReadOnly);
    const auto chunks = h5.getGroup("chunks");
    std::set<std::vector<float>> schemes;
    for (const auto& name : chunks.listObjectNames()) {
        CAPTURE(name);
        const auto chunk = chunks.getGroup(name);
        REQUIRE(chunk.exist("rfi_config"));
        std::vector<float> thresholds;
        std::vector<float> harvest;
        chunk.getDataSet("threshold_scheme").read(thresholds);
        chunk.getDataSet("rfi_config/harvest_scheme").read(harvest);
        CHECK(harvest ==
              make_default_harvest_scheme(thresholds, defaults.min_level,
                                          defaults.offset, defaults.min_snr));
        schemes.insert(harvest);
        for (const auto& run : chunk.getGroup("runs").listObjectNames()) {
            CHECK(chunk.exist("runs/" + run + "/harvest"));
        }
    }
    CHECK(schemes.size() >= 2);
}

TEST_CASE("EPFreqSweep without RFI control records no rfi_config",
          "[ep_freq_sweep][rfi]") {
    const TempDir outdir("loki-ep-rfi-off");
    const auto cfg = make_sweep_cfg(140.0, 142.0);
    const auto [ts_e, ts_v] =
        make_series(cfg.get_nsamps(), cfg.get_tsamp(), {});
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, PruneRFIConfig{},
                      std::nullopt, std::nullopt, std::vector<SizeType>{4U});
    sweep.execute(ts_e, ts_v, outdir.path(), "off");
    loki::search::write_run_stats(outdir / "off_ep_results.h5",
                                  loki::search::RunStats{});
    const HighFive::File h5((outdir / "off_ep_results.h5").string(),
                            HighFive::File::ReadOnly);
    const auto chunk = h5.getGroup("chunks/chunk_0000");
    CHECK_FALSE(chunk.exist("rfi_config"));
    // The RFI counters are written only for an RFI-controlled sweep.
    const auto stats = h5.getGroup("run_stats");
    CHECK_FALSE(stats.hasAttribute("n_leaves_masked"));
    CHECK_FALSE(stats.hasAttribute("n_leaves_vetoed"));
    CHECK_FALSE(h5.exist("harvested"));
    CHECK_FALSE(stats.hasAttribute("n_harvested_sources"));
    const auto runs = chunk.getGroup("runs");
    for (const auto& run : runs.listObjectNames()) {
        CHECK_FALSE(runs.exist(run + "/harvest"));
    }
}

TEST_CASE("EPFreqSweep masks, harvests and vetoes on the CPU",
          "[ep_freq_sweep][rfi]") {
    // A bright source at 141 Hz is harvested; one at 143.5 Hz is masked.
    constexpr double kHarvested = 141.0;
    constexpr double kMasked    = 143.5;
    const auto cfg              = make_sweep_cfg(140.0, 145.0);
    const auto [ts_e, ts_v] =
        make_series(cfg.get_nsamps(), cfg.get_tsamp(),
                    {{kHarvested, 1.0F}, {kMasked, 1.0F}});

    const auto near = [](const std::vector<std::pair<double, float>>& leaves,
                         double f, double tol) {
        float best = 0.0F;
        for (const auto& [freq, score] : leaves) {
            if (std::abs(freq - f) <= tol) {
                best = std::max(best, score);
            }
        }
        return best;
    };

    // Reference: no RFI control. Both sources are strong final leaves.
    const TempDir ref_dir("loki-ep-rfi-ref");
    EPFreqSweep reference(cfg, false, 0.1F, "taylor", 0.1F, PruneRFIConfig{},
                          std::nullopt, std::nullopt,
                          std::vector<SizeType>{4U});
    reference.execute(ts_e, ts_v, ref_dir.path(), "ref");
    std::vector<std::pair<double, float>> ref_leaves;
    {
        const HighFive::File h5((ref_dir / "ref_ep_results.h5").string(),
                                HighFive::File::ReadOnly);
        const auto chunks = h5.getGroup("chunks");
        for (const auto& name : chunks.listObjectNames()) {
            const auto runs = chunks.getGroup(name + "/runs");
            for (const auto& run : runs.listObjectNames()) {
                const auto leaves = final_leaves(runs.getGroup(run));
                ref_leaves.insert(ref_leaves.end(), leaves.begin(),
                                  leaves.end());
            }
        }
    }
    REQUIRE(near(ref_leaves, kHarvested, 0.05) > 20.0F);
    REQUIRE(near(ref_leaves, kMasked, 0.05) > 20.0F);

    PruneRFIConfig rfi;
    rfi.pulsar_mask.push_back({.f_lo = kMasked - 0.05, .f_hi = kMasked + 0.05});
    rfi.harvest_defaults =
        HarvestDefaults{.min_level = 4, .offset = 6.0F, .min_snr = 25.0F};
    rfi.impulsive_veto = true;
    const TempDir outdir("loki-ep-rfi-on");
    EPFreqSweep sweep(cfg, false, 0.1F, "taylor", 0.1F, rfi, std::nullopt,
                      std::nullopt, std::vector<SizeType>{4U});
    sweep.execute(ts_e, ts_v, outdir.path(), "on");

    const HighFive::File h5((outdir / "on_ep_results.h5").string(),
                            HighFive::File::ReadOnly);
    const auto chunks = h5.getGroup("chunks");
    std::vector<std::pair<double, float>> leaves;
    std::vector<double> harvested_f;
    for (const auto& name : chunks.listObjectNames()) {
        const auto runs = chunks.getGroup(name + "/runs");
        for (const auto& run_name : runs.listObjectNames()) {
            const auto run   = runs.getGroup(run_name);
            const auto these = final_leaves(run);
            leaves.insert(leaves.end(), these.begin(), these.end());
            REQUIRE(run.exist("harvest"));
            const auto dims =
                run.getDataSet("harvest/param_sets").getDimensions();
            if (dims[0] == 0) {
                continue;
            }
            std::vector<double> params(dims[0] * dims[1] * dims[2]);
            run.getDataSet("harvest/param_sets").read_raw(params.data());
            for (std::size_t i = 0; i < dims[0]; ++i) {
                harvested_f.push_back(params[(i * dims[1] * dims[2]) + 2]);
            }
        }
    }
    // The bright source is harvested and absent from the final leaves; the
    // masked one never reaches them.
    REQUIRE_FALSE(harvested_f.empty());
    CHECK(std::ranges::any_of(harvested_f, [&](double f) {
        return std::abs(f - kHarvested) < 0.05;
    }));
    CHECK(near(leaves, kHarvested, 0.05) < 10.0F);
    CHECK(near(leaves, kMasked, 0.05) < 10.0F);

    // The result lists the harvested source once, merged over its runs.
    REQUIRE(h5.exist("harvested"));
    std::vector<double> h_params;
    std::vector<float> h_snr;
    std::vector<std::uint64_t> h_records;
    h5.getDataSet("harvested/snr").read(h_snr);
    h5.getDataSet("harvested/n_records").read(h_records);
    REQUIRE_FALSE(h_snr.empty());
    const auto h_dims = h5.getDataSet("harvested/param_sets").getDimensions();
    h_params.resize(h_dims[0] * h_dims[1]);
    h5.getDataSet("harvested/param_sets").read_raw(h_params.data());
    CHECK(std::abs(h_params[h_dims[1] - 1] - kHarvested) < 0.05);
    CHECK(std::ranges::is_sorted(h_snr, std::greater<>{}));
    std::uint64_t total = 0;
    for (const auto r : h_records) {
        total += r;
    }
    CHECK(total == harvested_f.size());
}

namespace {

/// The strongest detection of a published product within @p tol of @p f.
float best_detection(const HighFive::File& h5, double f, double tol) {
    const auto table = h5.getGroup("detections");
    const auto dims  = table.getDataSet("param_sets").getDimensions();
    std::vector<float> snr;
    table.getDataSet("snr").read(snr);
    std::vector<double> params(dims[0] * dims[1]);
    if (dims[0] != 0) {
        table.getDataSet("param_sets").read_raw(params.data());
    }
    float best = 0.0F;
    for (std::size_t i = 0; i < dims[0]; ++i) {
        const double freq = params[(i * dims[1]) + dims[1] - 1];
        if (std::abs(freq - f) <= tol) {
            best = std::max(best, snr[i]);
        }
    }
    return best;
}

} // namespace

TEST_CASE("EP candidates publish an RFI-controlled sweep",
          "[ep_freq_sweep][rfi][candidates]") {
    constexpr double kHarvested = 141.0;
    constexpr double kMasked    = 143.5;
    constexpr double kKept      = 142.3;
    const auto cfg              = make_sweep_cfg(140.0, 145.0);
    const auto [ts_e, ts_v] =
        make_series(cfg.get_nsamps(), cfg.get_tsamp(),
                    {{kHarvested, 1.0F}, {kMasked, 1.0F}, {kKept, 0.35F}});
    auto options      = CandidateOptions::ep_defaults();
    options.memory_gb = 0.25;

    const auto publish = [&](const PruneRFIConfig& rfi, const TempDir& dir) {
        EPFreqSweep sweep(cfg, options, false, 0.1F, "taylor", 0.1F, rfi,
                          std::nullopt, std::nullopt,
                          std::vector<SizeType>{2U, 4U});
        sweep.execute(ts_e, ts_v, dir.path(), "c");
        return dir / "c_ep_results.h5";
    };

    const TempDir ref_dir("loki-ep-rfi-cand-ref");
    const HighFive::File ref(publish(PruneRFIConfig{}, ref_dir).string(),
                             HighFive::File::ReadOnly);
    REQUIRE(best_detection(ref, kHarvested, 0.05) > 20.0F);
    REQUIRE(best_detection(ref, kMasked, 0.05) > 20.0F);
    const float kept_ref = best_detection(ref, kKept, 0.05);
    REQUIRE(kept_ref > 8.0F);

    PruneRFIConfig rfi;
    rfi.pulsar_mask.push_back({.f_lo = kMasked - 0.05, .f_hi = kMasked + 0.05});
    rfi.harvest_defaults =
        HarvestDefaults{.min_level = 4, .offset = 6.0F, .min_snr = 25.0F};
    rfi.impulsive_veto = true;
    const TempDir dir("loki-ep-rfi-cand");
    const auto product = publish(rfi, dir);
    loki::search::write_run_stats(product, loki::search::RunStats{});
    const HighFive::File h5(product.string(), HighFive::File::ReadOnly);
    REQUIRE(h5.exist("candidates"));
    // The bright sources are gone from the detections, the weak one stays.
    CHECK(best_detection(h5, kHarvested, 0.05) < 10.0F);
    CHECK(best_detection(h5, kMasked, 0.05) < 10.0F);
    CHECK(best_detection(h5, kKept, 0.05) > 0.8F * kept_ref);
    // Every run keeps its harvests, with at least one at the bright source.
    SizeType n_runs      = 0;
    SizeType n_harvested = 0;
    const auto chunks    = h5.getGroup("chunks");
    for (const auto& chunk : chunks.listObjectNames()) {
        CHECK(chunks.exist(chunk + "/rfi_config"));
        const auto runs = chunks.getGroup(chunk + "/runs");
        for (const auto& run : runs.listObjectNames()) {
            ++n_runs;
            REQUIRE(runs.exist(run + "/harvest"));
            n_harvested +=
                runs.getDataSet(run + "/harvest/scores").getDimensions().at(0);
        }
    }
    CHECK(n_runs == 2);
    CHECK(n_harvested >= 2);
    const auto stats = h5.getGroup("run_stats");
    CHECK(stats.getAttribute("n_harvests_recorded").read<std::uint64_t>() ==
          n_harvested);
    CHECK(stats.getAttribute("n_harvested").read<std::uint64_t>() >=
          n_harvested);
    CHECK(stats.getAttribute("max_harvested_per_run").read<std::uint64_t>() >=
          1);
    // The masked source's leaves are counted; the veto total is written even
    // when nothing was vetoed, so a no-op is distinguishable.
    CHECK(stats.getAttribute("n_leaves_masked").read<std::uint64_t>() >= 1);
    CHECK(stats.hasAttribute("n_leaves_vetoed"));
    // Harvested sources reach the published product and its summary.
    REQUIRE(h5.exist("harvested"));
    const auto n_sources = h5.getGroup("harvested")
                               .getAttribute("n_sources")
                               .read<std::uint64_t>();
    CHECK(n_sources >= 1);
    CHECK(stats.getAttribute("n_harvested_sources").read<std::uint64_t>() ==
          n_sources);
    const auto summary = loki::io::read_candidate_file_summary(product, 5);
    CHECK(summary.n_harvested_sources == n_sources);
    REQUIRE_FALSE(summary.harvested.empty());
    CHECK(std::abs(summary.harvested.front().params.back() - kHarvested) <
          0.05);
}
